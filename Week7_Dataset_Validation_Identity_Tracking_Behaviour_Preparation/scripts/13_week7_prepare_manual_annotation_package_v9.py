from pathlib import Path
from datetime import datetime
import csv
import json
import shutil
import zipfile

import cv2
import pandas as pd


ROOT = Path.home() / "PigBench"

W6 = ROOT / "Week6_Unibo_Dataset_Validation"
W7 = ROOT / "Week7_Dataset_Validation_Identity_Tracking_Behaviour_Preparation"

FRAME_INDEX_PATH = W6 / "outputs" / "unified_ground_truth" / "week6_scanpoint_frame_index.csv"
V7_POOL_PATH = W7 / "outputs" / "roi_and_crate_mapping" / "week7_gt_pen_candidate_pool_v7_corrected.csv"

OUT_ROOT = W7 / "outputs" / "manual_annotation_v9"
CVAT_DIR = OUT_ROOT / "cvat_coco_import"
IMG_DIR = CVAT_DIR / "images"
ANN_DIR = CVAT_DIR / "annotations"

OUT_JSON = ANN_DIR / "instances_default.json"
OUT_ZIP = OUT_ROOT / "week7_manual_annotation_v9_cvat_coco_import.zip"
OUT_METADATA = OUT_ROOT / "week7_manual_annotation_v9_candidate_metadata.csv"
OUT_FRAME_MAP = OUT_ROOT / "week7_manual_annotation_v9_frame_image_map.csv"
OUT_README = OUT_ROOT / "README_manual_annotation_v9.md"
OUT_SUMMARY = OUT_ROOT / "week7_manual_annotation_v9_package_summary.csv"

OUT_NOTES = W7 / "notes"
OUT_NOTES.mkdir(parents=True, exist_ok=True)
OUT_NOTE = OUT_NOTES / "week7_manual_annotation_v9_package_notes.md"


def safe_to_csv(df, path):
    df.to_csv(
        path,
        index=False,
        quoting=csv.QUOTE_ALL,
        escapechar="\\",
        lineterminator="\n",
    )


def resolve_path(value):
    if pd.isna(value):
        return None

    value = str(value).strip()
    if not value:
        return None

    candidates = [
        Path(value),
        W6 / value,
        W7 / value,
        ROOT / value,
        Path.home() / value,
    ]

    for c in candidates:
        if c.exists():
            return c

    return None


def first_col(df, candidates):
    for c in candidates:
        if c in df.columns:
            return c
    return None


def category_for_class(v7_class):
    if v7_class == "strict_auto_safe":
        return "candidate_safe"
    if v7_class == "existing_selected_review_required":
        return "candidate_review"
    if v7_class == "existing_ignored_high_risk_review":
        return "candidate_risk"
    if v7_class == "new_low_threshold_recall_candidate":
        return "candidate_new"
    return None


# Clean previous package folder.
if CVAT_DIR.exists():
    shutil.rmtree(CVAT_DIR)

IMG_DIR.mkdir(parents=True, exist_ok=True)
ANN_DIR.mkdir(parents=True, exist_ok=True)
OUT_ROOT.mkdir(parents=True, exist_ok=True)

frames = pd.read_csv(FRAME_INDEX_PATH)
pool = pd.read_csv(V7_POOL_PATH)

image_col = first_col(frames, ["frame_image_path", "image_path", "frame_path", "scanpoint_frame_path"])
if image_col is None:
    raise RuntimeError("No image path column found in frame index.")

for c in ["x1", "y1", "x2", "y2", "candidate_score", "detector_score"]:
    if c in pool.columns:
        pool[c] = pd.to_numeric(pool[c], errors="coerce")

pool = pool.dropna(subset=["x1", "y1", "x2", "y2"]).copy()

# COCO categories.
categories = [
    {"id": 1, "name": "gt_pen_pig", "supercategory": "pig"},
    {"id": 2, "name": "candidate_safe", "supercategory": "pig_candidate"},
    {"id": 3, "name": "candidate_review", "supercategory": "pig_candidate"},
    {"id": 4, "name": "candidate_risk", "supercategory": "pig_candidate"},
    {"id": 5, "name": "candidate_new", "supercategory": "pig_candidate"},
]

cat_name_to_id = {c["name"]: c["id"] for c in categories}

images = []
annotations = []
frame_map_rows = []
metadata_rows = []

image_id_by_scan_frame = {}

ann_id = 1
img_id = 1

frame_list = frames.drop_duplicates("scan_frame_id").sort_values("scan_frame_id").copy()

for _, fr in frame_list.iterrows():
    scan_frame_id = str(fr["scan_frame_id"])
    src_path = resolve_path(fr[image_col])

    if src_path is None:
        continue

    img = cv2.imread(str(src_path))
    if img is None:
        continue

    h, w = img.shape[:2]

    dst_name = f"{scan_frame_id}.jpg"
    dst_path = IMG_DIR / dst_name

    cv2.imwrite(str(dst_path), img)

    images.append({
        "id": img_id,
        "file_name": f"images/{dst_name}",
        "width": int(w),
        "height": int(h),
    })

    image_id_by_scan_frame[scan_frame_id] = img_id

    frame_map_rows.append({
        "scan_frame_id": scan_frame_id,
        "coco_image_id": img_id,
        "source_image_path": str(src_path),
        "cvat_image_file": f"images/{dst_name}",
        "width": int(w),
        "height": int(h),
        "expected_gt_pen_pigs": 6,
    })

    img_id += 1

# Add candidate annotations.
candidate_pool = pool.copy()
candidate_pool["cvat_category_name"] = candidate_pool["v7_candidate_class"].apply(category_for_class)
candidate_pool = candidate_pool[candidate_pool["cvat_category_name"].notna()].copy()

for _, r in candidate_pool.iterrows():
    scan_frame_id = str(r["scan_frame_id"])

    if scan_frame_id not in image_id_by_scan_frame:
        continue

    x1 = float(r["x1"])
    y1 = float(r["y1"])
    x2 = float(r["x2"])
    y2 = float(r["y2"])

    bw = max(1.0, x2 - x1)
    bh = max(1.0, y2 - y1)

    cat_name = str(r["cvat_category_name"])
    cat_id = cat_name_to_id[cat_name]

    annotations.append({
        "id": ann_id,
        "image_id": int(image_id_by_scan_frame[scan_frame_id]),
        "category_id": int(cat_id),
        "bbox": [x1, y1, bw, bh],
        "area": float(bw * bh),
        "iscrowd": 0,
        "attributes": {
            "scan_frame_id": scan_frame_id,
            "v7_candidate_id": str(r.get("v7_candidate_id", "")),
            "v7_candidate_class": str(r.get("v7_candidate_class", "")),
            "candidate_source": str(r.get("candidate_source", "")),
            "candidate_score": str(r.get("candidate_score", "")),
        },
    })

    metadata_rows.append({
        "coco_annotation_id": ann_id,
        "scan_frame_id": scan_frame_id,
        "v7_candidate_id": r.get("v7_candidate_id", ""),
        "v7_candidate_class": r.get("v7_candidate_class", ""),
        "cvat_category_name": cat_name,
        "candidate_source": r.get("candidate_source", ""),
        "candidate_score": r.get("candidate_score", ""),
        "detector_score": r.get("detector_score", ""),
        "x1": x1,
        "y1": y1,
        "x2": x2,
        "y2": y2,
        "manual_instruction": "In CVAT: keep/adjust if true GT-pen pig; delete if wrong pen, duplicate, or false positive.",
    })

    ann_id += 1

coco = {
    "info": {
        "description": "Week 7 manual annotation package for corrected GT-pen pig boxes",
        "version": "v9",
        "year": 2026,
        "date_created": datetime.now().isoformat(timespec="seconds"),
    },
    "licenses": [],
    "images": images,
    "annotations": annotations,
    "categories": categories,
}

OUT_JSON.write_text(json.dumps(coco, indent=2))

safe_to_csv(pd.DataFrame(metadata_rows), OUT_METADATA)
safe_to_csv(pd.DataFrame(frame_map_rows), OUT_FRAME_MAP)

readme = f"""# Week 7 Manual Annotation Package v9

## Goal

Correct the pig boxes for the annotated ground-truth pen.

This package is intended for CVAT or another COCO-compatible box annotation tool.

## What to do in the annotation tool

For every scanpoint frame:

1. Keep only pigs that belong to the annotated GT pen.
2. Delete boxes from adjacent/wrong pens.
3. Delete duplicate boxes.
4. Adjust inaccurate boxes.
5. Draw missing true GT-pen pig boxes.
6. Aim for six GT-pen pigs per scanpoint when visible.

## Labels

- `gt_pen_pig`: use this for newly drawn missing true pigs.
- `candidate_safe`: previous automatic safe candidate. Still delete it if it is wrong.
- `candidate_review`: previous review candidate.
- `candidate_risk`: previous risky ignored candidate.
- `candidate_new`: low-threshold recall candidate.

Final export can contain any of these labels. The next import script will treat all remaining pig boxes as corrected GT-pen boxes, unless you delete them.

## Important

Do not keep boxes from outside the annotated GT pen.

## Files

- COCO annotation JSON: `annotations/instances_default.json`
- Images: `images/*.jpg`
- Candidate metadata: `{OUT_METADATA}`
- Frame map: `{OUT_FRAME_MAP}`
"""

OUT_README.write_text(readme)

# Zip package.
if OUT_ZIP.exists():
    OUT_ZIP.unlink()

with zipfile.ZipFile(OUT_ZIP, "w", compression=zipfile.ZIP_DEFLATED) as z:
    for p in CVAT_DIR.rglob("*"):
        if p.is_file():
            z.write(p, p.relative_to(CVAT_DIR))
    z.write(OUT_README, "README_manual_annotation_v9.md")

summary = pd.DataFrame([
    {"item": "scanpoint_frames_packaged", "value": len(images)},
    {"item": "candidate_annotations_packaged", "value": len(annotations)},
    {"item": "categories", "value": ", ".join([c["name"] for c in categories])},
    {"item": "zip_path", "value": str(OUT_ZIP)},
    {"item": "zip_exists", "value": OUT_ZIP.exists()},
    {"item": "zip_size_mb", "value": round(OUT_ZIP.stat().st_size / (1024 * 1024), 2) if OUT_ZIP.exists() else ""},
])

safe_to_csv(summary, OUT_SUMMARY)

OUT_NOTE.write_text(
    "# Week 7 Manual Annotation Package v9\n\n"
    "## Purpose\n\n"
    "The automatic detector/ROI workflow still produced wrong boxes in many frames. "
    "Therefore, all 72 scanpoint frames are exported to a COCO-compatible manual annotation package.\n\n"
    "## Output\n\n"
    f"- CVAT/COCO zip: `{OUT_ZIP}`\n"
    f"- Candidate metadata: `{OUT_METADATA}`\n"
    f"- Frame map: `{OUT_FRAME_MAP}`\n"
    f"- COCO JSON: `{OUT_JSON}`\n\n"
    "## Manual rule\n\n"
    "Keep only true GT-pen pigs, delete wrong-pen boxes, adjust inaccurate boxes, and draw missing true GT-pen pigs.\n"
)

print("Saved:")
print(OUT_ZIP)
print(OUT_JSON)
print(OUT_METADATA)
print(OUT_FRAME_MAP)
print(OUT_README)
print(OUT_SUMMARY)
print(OUT_NOTE)
print()
print("=== package summary ===")
print(summary.to_string(index=False))
