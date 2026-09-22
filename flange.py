"""Paper flange detector. Inputs are local; no network or image export."""
from pathlib import Path
import cv2
import numpy as np
CANNY_LOW, CANNY_HIGH, CLOSE_KERNEL = 80, 220, 10
LENGTH_MIN, SOLIDITY_MIN, SOLIDITY_MAX, THICKNESS_MAX = 80., .60, .84, 5.
SUPPORT_MIN = 2
BIPOLAR_NEGATIVE_MAX, BIPOLAR_POSITIVE_MIN = -20., 10.
WIDE_DARK_THICKNESS_MIN, WIDE_DARK_CONTRAST_MIN = 3.5, 12.


def read_gray(path: Path) -> np.ndarray:
    image = cv2.imdecode(np.fromfile(path, dtype=np.uint8), cv2.IMREAD_GRAYSCALE)
    if image is None:
        raise RuntimeError(f'无法读取图像: {path}')
    return image

def load_group(folder: Path) -> tuple[list[Path], list[np.ndarray]]:
    paths = sorted((p for p in folder.iterdir() if p.suffix.lower() in {'.jpg', '.png', '.bmp'}))
    if len(paths) != 4:
        raise RuntimeError(f'{folder} 应有4张图，实际为{len(paths)}张')
    return (paths, [read_gray(path) for path in paths])

def build_roi(images: list[np.ndarray]) -> tuple[np.ndarray, np.ndarray]:
    fused = images[0].copy()
    for image in images[1:]:
        fused = cv2.max(fused, image)
    small = cv2.resize(fused, None, fx=0.25, fy=0.25)
    blurred = cv2.GaussianBlur(small, (3, 3), 0)
    _, binary = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
    binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (11, 11)))
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours = [contour for contour in contours if cv2.contourArea(contour) > 300]
    if not contours:
        raise RuntimeError('未找到法兰外轮廓')
    (cx, cy), radius = cv2.minEnclosingCircle(max(contours, key=cv2.contourArea))
    center = (cx / 0.25, cy / 0.25)
    radius /= 0.25
    mask = np.zeros_like(fused)
    cv2.circle(mask, (round(center[0]), round(center[1])), int(radius), 255, cv2.FILLED)
    x = max(0, int(center[0] - radius))
    y = max(0, int(center[1] - radius))
    width = min(fused.shape[1] - x, int(2 * radius))
    height = min(fused.shape[0] - y, int(2 * radius))
    small_roi = cv2.resize(fused[y:y + height, x:x + width], None, fx=0.5, fy=0.5)
    params = cv2.SimpleBlobDetector_Params()
    params.filterByColor = True
    params.blobColor = 0
    params.minThreshold = 10
    params.maxThreshold = 200
    params.thresholdStep = 10
    params.filterByArea = True
    params.minArea = 1000
    params.maxArea = 80000
    params.filterByInertia = True
    params.minInertiaRatio = 0.5
    params.filterByCircularity = True
    params.minCircularity = 0.4
    params.filterByConvexity = False
    for keypoint in cv2.SimpleBlobDetector_create(params).detect(small_roi):
        gx = keypoint.pt[0] / 0.5 + x
        gy = keypoint.pt[1] / 0.5 + y
        hole_radius = keypoint.size / 2.0 / 0.5
        if np.hypot(gx - center[0], gy - center[1]) <= radius * 0.95:
            cv2.circle(mask, (round(gx), round(gy)), int(hole_radius) + 4, 0, cv2.FILLED)
    mask = cv2.erode(mask, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (45, 45)))
    return (fused, mask)

def candidate_mask(images: list[np.ndarray], roi: np.ndarray) -> tuple[np.ndarray, list[np.ndarray]]:
    union = np.zeros_like(roi)
    per_light = []
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    for image in images:
        edges = cv2.Canny(image, CANNY_LOW, CANNY_HIGH, apertureSize=3, L2gradient=True)
        edges = cv2.dilate(edges, kernel)
        edges = cv2.bitwise_and(edges, roi)
        light_mask = np.zeros_like(roi)
        contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for contour in contours:
            if cv2.arcLength(contour, False) < 8:
                continue
            (_, _), (width, height), _ = cv2.minAreaRect(contour)
            if min(width, height) / max(width, height) > 0.75:
                continue
            cv2.drawContours(light_mask, [contour], -1, 255, cv2.FILLED)
        per_light.append(light_mask)
        union = cv2.bitwise_or(union, light_mask)
    merged = cv2.morphologyEx(union, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (CLOSE_KERNEL, CLOSE_KERNEL)))
    return (merged, per_light)

def features(ims, roi, low, high):
    global CANNY_LOW, CANNY_HIGH
    CANNY_LOW = low
    CANNY_HIGH = high
    merged, per = candidate_mask(ims, roi)
    cs, _ = cv2.findContours(merged, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    fs = []
    for c in cs:
        length = cv2.arcLength(c, False)
        area = cv2.contourArea(c)
        hull = cv2.contourArea(cv2.convexHull(c))
        (_, _), (w, h), _ = cv2.minAreaRect(c)
        if length < 30 or hull == 0 or max(w, h) == 0:
            continue
        x, y, bw, bh = cv2.boundingRect(c)
        x0 = max(x - 7, 0)
        y0 = max(y - 7, 0)
        x1 = min(x + bw + 7, roi.shape[1])
        y1 = min(y + bh + 7, roi.shape[0])
        cc = c - [x0, y0]
        mask = np.zeros((y1 - y0, x1 - x0), np.uint8)
        cv2.drawContours(mask, [cc], -1, 255, -1)
        ring = cv2.subtract(cv2.dilate(mask, np.ones((11, 11), np.uint8)), mask)
        contrast = [cv2.mean(im[y0:y1, x0:x1], ring)[0] - cv2.mean(im[y0:y1, x0:x1], mask)[0] for im in ims]
        support = sum((bool(cv2.countNonZero(cv2.bitwise_and(p[y0:y1, x0:x1], mask))) for p in per))
        m = cv2.moments(c)
        center = [int(m['m10'] / m['m00']), int(m['m01'] / m['m00'])] if m['m00'] else c[0, 0].tolist()
        f = dict(center_x=center[0], center_y=center[1], length=length, area=area, thickness=area / length, solidity=area / hull, box_long=max(w, h), aspect=min(w, h) / max(w, h), light_support=support, best_contrast=max(contrast, key=abs), bbox=[x, y, bw, bh], contour=c[:, 0, :].tolist())
        f.update({f'contrast_l{k + 1}': v for k, v in enumerate(contrast)})
        fs.append(f)
    return fs

def accepted(row: dict[str, object]) -> tuple[bool, str]:
    length = float(row['length'])
    thickness = float(row['thickness'])
    solidity = float(row['solidity'])
    contrast = float(row['best_contrast'])
    support = int(row['light_support'])
    if length < LENGTH_MIN:
        return (False, 'length')
    if not SOLIDITY_MIN <= solidity <= SOLIDITY_MAX:
        return (False, 'solidity')
    if thickness > THICKNESS_MAX:
        return (False, 'thickness')
    if support >= SUPPORT_MIN:
        return (True, 'multi_light')
    other_positive = max((float(row[f'contrast_l{index}']) for index in range(1, 5)))
    if contrast <= BIPOLAR_NEGATIVE_MAX and other_positive >= BIPOLAR_POSITIVE_MIN:
        return (True, 'bipolar_bright')
    if thickness >= WIDE_DARK_THICKNESS_MIN and contrast >= WIDE_DARK_CONTRAST_MIN:
        return (True, 'wide_dark')
    return (False, 'weak_single_light')

def long_branch(f):
    return f['length'] >= 300 and 0.2 <= f['solidity'] <= 0.84 and (f['thickness'] <= 4) and (f['aspect'] <= 0.2) and (f['light_support'] >= 2) and (abs(f['best_contrast']) >= 10)

def thin_polarity(f):
    vals = [f[f'contrast_l{k}'] for k in range(1, 5)]
    return 100 <= f['length'] and f['box_long'] >= 60 and (0.6 <= f['solidity'] <= 0.84) and (f['thickness'] <= 3.5) and (f['aspect'] <= 0.2) and (min(vals) <= -8) and (max(vals) >= 10)

def strict_old(f):
    keep, reason = accepted(f)
    return keep and (reason != 'wide_dark' or f['aspect'] <= 0.25)

def overlap(f, g):
    c = np.array(g['contour'], np.int32).reshape(-1, 1, 2)
    return abs(cv2.pointPolygonTest(c, (float(f['center_x']), float(f['center_y'])), True)) <= 15

def combined(base, extra, strict=False):
    keep = [dict(f, branch='original') for f in base if (strict_old(f) if strict else accepted(f)[0])]
    for f in base:
        if long_branch(f) and (not any((overlap(f, k) for k in keep))):
            keep.append(dict(f, branch='long_thin'))
    for f in extra:
        if thin_polarity(f) and (not any((overlap(f, k) for k in keep))):
            keep.append(dict(f, branch='thin_polarity'))
    return keep

def refine_candidates(features, roi, length=21, width=3, max_added_fraction=0.2):
    masks = []
    for f in features:
        m = np.zeros_like(roi)
        cv2.drawContours(m, [np.asarray(f['contour'], np.int32)], -1, 255, -1)
        masks.append(m)
    result = []
    for i, (f, m) in enumerate(zip(features, masks)):
        pts = np.asarray(f['contour'], np.float64).reshape(-1, 2)
        v = np.linalg.eigh(np.cov(pts.T))[1][:, -1]
        size = length + 4
        kernel = np.zeros((size, size), np.uint8)
        mid = np.array([size // 2] * 2)
        cv2.line(kernel, tuple(np.rint(mid - v * (length - 1) / 2).astype(int)), tuple(np.rint(mid + v * (length - 1) / 2).astype(int)), 1, width)
        closed = cv2.morphologyEx(m, cv2.MORPH_CLOSE, kernel)
        closed = cv2.bitwise_or(closed, m)
        added = cv2.subtract(closed, m)
        gain = cv2.countNonZero(added) / max(1, cv2.countNonZero(m))
        other = np.zeros_like(m)
        for j, mj in enumerate(masks):
            if i != j:
                other = cv2.bitwise_or(other, mj)
        near_other = cv2.dilate(other, np.ones((3, 3), np.uint8))
        reason = 'accepted'
        if gain > max_added_fraction:
            reason = 'area_guard'
        elif cv2.countNonZero(cv2.bitwise_and(added, cv2.bitwise_not(roi))):
            reason = 'roi_guard'
        elif cv2.countNonZero(cv2.bitwise_and(added, near_other)):
            reason = 'neighbor_guard'
        if reason != 'accepted':
            closed = m
        contours = cv2.findContours(closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[0]
        if len(contours) != 1:
            closed = m
            contours = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[0]
            reason = 'component_guard'
        out = dict(f)
        out['raw_contour'] = f['contour']
        out['contour'] = max(contours, key=cv2.contourArea).reshape(-1, 2).tolist()
        out['refinement'] = {'status': reason, 'length_px': length, 'width_px': width, 'proposed_area_fraction': gain, 'added_pixels': cv2.countNonZero(cv2.subtract(closed, m)), 'features_scope': 'original candidate before contour refinement'}
        result.append(out)
    return result
def detect(images, refine=True):
    if len(images) != 4 or any(im is None or im.ndim != 2 or im.dtype != np.uint8 or im.shape != images[0].shape for im in images):
        raise ValueError("Require four aligned uint8 grayscale images of equal dimensions")
    _, roi = build_roi(images)
    if not np.any(roi):
        raise ValueError("Empty ROI; do not classify as normal")
    selected = combined(features(images, roi, 80, 220), features(images, roi, 60, 180), True)
    return refine_candidates(selected, roi) if refine else selected

if __name__ == '__main__':
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument('--images', nargs=4, required=True, type=Path, help='L1 L2 L3 L4, in that order')
    args = p.parse_args()
    result = detect([read_gray(x) for x in args.images])
    print({'decision': 'NG' if result else 'OK', 'candidate_count': len(result)})
