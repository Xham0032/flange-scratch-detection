from dataclasses import dataclass
import cv2
import numpy as np
import flange

@dataclass(frozen=True)
class Profile:
    name: str = 'flange'
    low: int = 80
    high: int = 220
    dilate: int = 3
    close: int = 10
    min_arc: float = 8
    max_aspect: float = 0.75
    erosion: int = 45
    ring: int = 11
    length: float = 80
    solidity_low: float = 0.6
    solidity_high: float = 0.84
    thickness: float = 5
    support: int = 2
    contrast_floor: float = 0
    rescue_scale: float = 1

def washer_profile():
    return Profile(name='washer_calibrated', high=240, dilate=1, close=1, min_arc=20.476373047941085, max_aspect=0.5218395291685864, erosion=49, ring=21, length=400, solidity_low=0.17662509461149922, solidity_high=0.9495695842036624, thickness=10.690075128887745, support=4, contrast_floor=12, rescue_scale=0.25)

def validate(images):
    if len(images) != 4:
        raise ValueError('Exactly four directional images are required')
    if any((im is None or im.ndim != 2 or im.dtype != np.uint8 for im in images)):
        raise ValueError('Inputs must be readable uint8 grayscale images')
    if any((im.shape != images[0].shape for im in images)):
        raise ValueError('Directional image sizes differ')
    if min(images[0].shape) < 32:
        raise ValueError('Images too small for the configured ROI procedure')

def roi_for(images, p):
    validate(images)
    fused, roi = flange.build_roi(images)
    if p.erosion != 45:
        small = cv2.resize(fused, None, fx=0.25, fy=0.25)
        _, binary = cv2.threshold(cv2.GaussianBlur(small, (3, 3), 0), 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
        binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (11, 11)))
        cs, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cs = [c for c in cs if cv2.contourArea(c) > 300]
        if not cs:
            raise ValueError('ROI outer contour not found')
        (cx, cy), radius = cv2.minEnclosingCircle(max(cs, key=cv2.contourArea))
        cx, cy, radius = (cx * 4, cy * 4, radius * 4)
        roi = np.zeros_like(fused)
        cv2.circle(roi, (round(cx), round(cy)), int(radius), 255, -1)
        x, y = (max(0, int(cx - radius)), max(0, int(cy - radius)))
        w, h = (min(fused.shape[1] - x, int(2 * radius)), min(fused.shape[0] - y, int(2 * radius)))
        crop = cv2.resize(fused[y:y + h, x:x + w], None, fx=0.5, fy=0.5)
        b = cv2.SimpleBlobDetector_Params()
        b.filterByColor = True
        b.blobColor = 0
        b.minThreshold = 10
        b.maxThreshold = 200
        b.thresholdStep = 10
        b.filterByArea = True
        b.minArea = 1000
        b.maxArea = 80000
        b.filterByInertia = True
        b.minInertiaRatio = 0.5
        b.filterByCircularity = True
        b.minCircularity = 0.4
        b.filterByConvexity = False
        for k in cv2.SimpleBlobDetector_create(b).detect(crop):
            gx, gy = (k.pt[0] * 2 + x, k.pt[1] * 2 + y)
            if np.hypot(gx - cx, gy - cy) <= radius * 0.95:
                cv2.circle(roi, (round(gx), round(gy)), int(k.size) + 4, 0, -1)
        roi = cv2.erode(roi, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (p.erosion, p.erosion)))
    # Historical Washer evaluation retains empty masks in the metric denominator.
    # This is localization evaluation, not an OK workpiece classification.
    return (fused, roi)

def masks_for(images, roi, p, low=None, high=None, clip=False):
    per = []
    for im in images:
        edge = cv2.Canny(im, p.low if low is None else low, p.high if high is None else high, apertureSize=3, L2gradient=True)
        if p.dilate > 1:
            edge = cv2.dilate(edge, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (p.dilate, p.dilate)))
        edge = cv2.bitwise_and(edge, roi)
        cs, _ = cv2.findContours(edge, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        m = np.zeros_like(roi)
        for c in cs:
            if cv2.arcLength(c, False) < p.min_arc:
                continue
            _, (w, h), _ = cv2.minAreaRect(c)
            if min(w, h) <= 0 or min(w, h) / max(w, h) > p.max_aspect:
                continue
            cv2.drawContours(m, [c], -1, 255, -1)
        if clip:
            m = cv2.bitwise_and(m, roi)
        per.append(m)
    union = np.bitwise_or.reduce(per)
    if p.close > 1:
        union = cv2.morphologyEx(union, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (p.close, p.close)))
    if clip:
        union = cv2.bitwise_and(union, roi)
    return (union, per)

def support_evidence(mask, per, fraction=0.05, min_pixels=3):
    pixels = int(np.count_nonzero(mask))
    active = [(m > 0) & (mask > 0) for m in per]
    counts = [int(np.count_nonzero(m)) for m in active]
    ratios = [n / max(pixels, 1) for n in counts]
    valid = [n >= min_pixels and r >= fraction for n, r in zip(counts, ratios)]
    votes = np.sum([m for m, v in zip(active, valid) if v], axis=0) if any(valid) else np.zeros_like(mask)
    joint = float(np.count_nonzero(votes >= 2) / max(pixels, 1))
    return dict(light_support=sum((n > 0 for n in counts)), strong_support=sum(valid), support_pixels=counts, support_fractions=ratios, joint_fraction=joint)

def extract(images, roi, p, low=None, high=None, clip=False):
    union, per = masks_for(images, roi, p, low, high, clip)
    cs, _ = cv2.findContours(union, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    out = []
    pad = p.ring // 2 + 2
    for c in cs:
        length = cv2.arcLength(c, False)
        area = cv2.contourArea(c)
        hull = cv2.contourArea(cv2.convexHull(c))
        _, (w, h), _ = cv2.minAreaRect(c)
        if length <= 0 or hull <= 0 or min(w, h) <= 0:
            continue
        x, y, bw, bh = cv2.boundingRect(c)
        x0 = max(0, x - pad)
        y0 = max(0, y - pad)
        x1 = min(roi.shape[1], x + bw + pad)
        y1 = min(roi.shape[0], y + bh + pad)
        mask = np.zeros((y1 - y0, x1 - x0), np.uint8)
        cv2.drawContours(mask, [c - [x0, y0]], -1, 255, -1)
        ring = cv2.subtract(cv2.dilate(mask, np.ones((p.ring, p.ring), np.uint8)), mask)
        clean = cv2.bitwise_and(ring, roi[y0:y1, x0:x1])
        contrasts = [cv2.mean(im[y0:y1, x0:x1], ring)[0] - cv2.mean(im[y0:y1, x0:x1], mask)[0] for im in images]
        valid_contrasts = [cv2.mean(im[y0:y1, x0:x1], clean)[0] - cv2.mean(im[y0:y1, x0:x1], mask)[0] for im in images] if np.any(clean) else [0.0] * 4
        ev = support_evidence(mask, [m[y0:y1, x0:x1] for m in per])
        moment = cv2.moments(c)
        center = [int(moment['m10'] / moment['m00']), int(moment['m01'] / moment['m00'])] if moment['m00'] else c[0, 0].tolist()
        pts = c[:, 0, :].astype(float)
        _, vec = np.linalg.eigh(np.cov(pts.T))
        v = vec[:, -1]
        f = dict(length=length, area=area, thickness=area / length, solidity=area / hull, box_long=max(w, h), aspect=min(w, h) / max(w, h), center_x=center[0], center_y=center[1], bbox=[x, y, bw, bh], contour=c[:, 0, :].tolist(), angle=float(np.degrees(np.arctan2(v[1], v[0])) % 180), best_contrast=max(contrasts, key=abs), clean_contrasts=valid_contrasts, ring_pixels=int(np.count_nonzero(clean)), **ev)
        f.update({f'contrast_l{i + 1}': v for i, v in enumerate(contrasts)})
        out.append(f)
    return (out, union)

def paint(fs, shape):
    mask = np.zeros(shape, np.uint8)
    for f in fs:
        cv2.drawContours(mask, [np.asarray(f['contour'], np.int32).reshape(-1, 1, 2)], -1, 255, -1)
    return mask