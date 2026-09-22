# Candidate-level flange scratch detection

Code and fixed Washer sample assignments accompanying **A Candidate-Level Multi-Directional Illumination Fusion Method for Flange Surface Scratch Detection**.

This is a local release candidate. It has not yet been published to GitHub. A license has not yet been selected; no open-source license is implied by this draft.

## Contents and data boundary

- `flange.py`: the paper's three-path flange detector and constrained contour refinement.
- `washer_core.py` and `evaluate_washer.py`: the separately calibrated Washer localization pipeline and five-stage evaluation.
- `configs/`: parameter inventories and the executable Washer profile.
- `splits/washer_20_30.csv`: the exact fixed assignment of 50 physical Washer samples to 20 tuning and 30 held-out samples. IDs retain leading zeros and must be read as strings.
- `check_release.py`: an allowlist check and ZIP builder that refuses extra files.

**No images are included or authorized for redistribution in this repository.** This applies to flange images, Washer images, annotations/masks, crops, overlays, screenshots, encoded images, and image-derived contour caches. Manuscript Word/PDF files and their embedded images are also excluded. The self-collected flange images cannot be publicly distributed. Users must obtain Washer independently from its official distribution: <https://doi.org/10.5281/zenodo.5513768>. Dataset rights are separate from any future code license.

The commands below read user-supplied local files. They do not upload data, write image files, or open network connections. Evaluation output is local numerical metrics only. Do not add local datasets or generated output files to the repository.

## Installation

Python 3.12 was used for release validation. From this directory:

```sh
python -m venv .venv
```

Activate that environment using the command appropriate for your operating system, then:

```sh
python -m pip install -r requirements.txt
```

The requirements pin the versions used in local validation. Other OpenCV versions may change contour extraction or ROI behavior and require regression checks.

## Flange inference

Pass four aligned grayscale images in L1, L2, L3, L4 order (left, bottom, right, top illumination). Paper thresholds are calibrated for 2448 × 2048 images and the original acquisition setup.

```sh
python flange.py --images /local/L1.png /local/L2.png /local/L3.png /local/L4.png
```

The CLI prints only the workpiece decision and candidate count. `detect(images, refine=True)` returns candidate features and refined contours in memory for local use; these must not be added to the release. Missing or invalid inputs raise an error rather than returning an OK classification. Candidate pools are generated at 80/220 and 60/180; results are selected by main, long-thin, and thin-scratch paths in that order.

`configs/flange_parameters.json` is a complete inventory of the explicitly specified paper settings, not a configuration loader. The faithful executable constants and threshold rules reside in `flange.py`. Editing that JSON alone does not change inference. The module changes Canny globals while constructing pools, so concurrent threads must use separate processes.

## Reproduce Washer Table 3

Provide the official directory layout locally:

```text
Washer/
  Train/<sample_id>/washer_<sample_id>_101.png
  Train/<sample_id>/washer_<sample_id>_104.png
  Train/<sample_id>/washer_<sample_id>_107.png
  Train/<sample_id>/washer_<sample_id>_110.png
  Train/<sample_id>/washer_<sample_id>_mask.png
  Test/<sample_id>/...
```

Run from any working directory:

```sh
python evaluate_washer.py --data-root /local/Washer --output /local-results/washer_metrics.json
```

`--split` and `--config` optionally select alternative local files. Defaults resolve relative to the script. The evaluation reads `configs/washer_parameters.json`; it does not search or optimize thresholds. The supplied split file is authoritative. No random seed is needed to replay an explicit assignment, and no unverifiable historical seed is invented.

The split pools the official 32 Train and 18 Test samples before assigning 20/30. This is **not the official benchmark protocol** and does not demonstrate cross-domain transfer of the flange detector.

Connected components below eight pixels are removed. P measures predicted components intersecting the ground-truth mask dilated with a 5 × 5 ellipse; R measures ground-truth components intersecting the separately dilated prediction. Matching permits one-to-many and many-to-one overlap. Per-sample P/R/F1 are macro-averaged over all 30 held-out samples, including samples with empty predictions. Empty Washer ROI masks produce empty localization predictions under the historical evaluation; they are not classified as OK workpieces. The reported F1 is the mean of sample F1 values, not the harmonic mean of aggregate P/R.

## Validation and limits

The release was locally validated on 2026-09-22:

- Flange: all 78 groups matched the existing implementation exactly, including numerical features and refined contours; nine output regions in total. The images and detailed outputs remain private, so external users cannot reproduce the private-data scores from this package alone.
- Washer: all 50 assigned samples were processed. Held-out results reproduce all five Table 3 rows at the paper's three-decimal precision: union 0.146/0.704/0.231; geometry 0.649/0.592/0.596; directional support 0.674/0.591/0.611; contrast 0.675/0.579/0.601; joint 0.703/0.578/0.617 (P/R/F1).

The public package covers the proposed flange inference and Washer stage evaluation. It does not yet include portable runners for every historical flange baseline or parameter search, and does not claim full reproduction of every paper experiment. Same-batch tuning, four defective flanges, and the custom Washer split limit the findings. Counts of output regions are not instance accuracy, and directional overlap does not establish physical scratch correspondence.

## Build an upload package

```sh
python check_release.py
python check_release.py --zip /local-release/flange-scratch-code-only.zip
```

Only the allowlisted text files may be packaged. Python bytecode caches and a local `.git` directory are excluded; any other unexpected file causes failure. Select the code license and create a dedicated repository before publishing. Do not initialize or upload the parent research directory.
