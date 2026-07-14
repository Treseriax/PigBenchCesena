from pathlib import Path
from datetime import datetime
import csv
import re

import pandas as pd
from PIL import Image


ROOT = Path.home() / "PigBench"
W7 = ROOT / "Week7_Dataset_Validation_Identity_Tracking_Behaviour_Preparation"

V21_ROOT = W7 / "outputs" / "primary_split_v21"
PRIMARY_ALL = V21_ROOT / "week7_primary_split_v21_all.csv"

OUT_ROOT = W7 / "outputs" / "model_ready_crop_dataset_v22"
CROP_ROOT = OUT_ROOT / "crops_by_split_and_behaviour"
OUT_ROOT.mkdir(parents=True, exist_ok=True)
CROP_ROOT.mkdir(parents=True, exist_ok=True)

OUT_INDEX = OUT_ROOT / "week7_model_ready_crop_dataset_v22_index.csv"
OUT_SUMMARY = OUT_ROOT / "week7_model_ready_crop_dataset_v22_summary.csv"
OUT_CLASS_DIST = OUT_ROOT / "week7_model_ready_crop_dataset_v22_class_distribution.csv"
OUT_SPLIT_DIST = OUT_ROOT / "week7_model_ready_crop_dataset_v22_split_distribution.csv"
OUT_HARD_ISSUES = OUT_ROOT / "week7_model_ready_crop_dataset_v22_hard_issues.csv"
OUT_README = OUT_ROOT / "README_model_ready_crop_dataset_v22.md"
OUT_NOTE = W7 / "notes" / "week7_model_ready_crop_dataset_v22_notes.md"


def safe_to_csv(df, path):
    df.to_csv(
        path,
        index=False,
        quoting=csv.QUOTE_ALL,
        escapechar="\\",
        lineterminator="\n",
    )


def slug(s):
    s = str(s).strip()
    s = re.sub(r"[^A-Za-z0-9_.-]+", "_", s)
    s = re.sub(r"_+", "_", s).strip("_")
    return s or "unknown"


def find_image_path(row):
    candidates = [
        "frame_image_path",
        "frame_image_path_label",
        "frame_image_path_identity",
        "image_path",
        "img_path",
        "path",
    ]

    for c in candidates:
        if c in row.index:
            p = str(row.get(c, "")).strip()
            if p and p.lower() not in ["nan", "none", "null"]:
                return p

    return ""


def bbox_from_row(row):
    direct_sets = [
        ("x1", "y1", "x2", "y2"),
        ("final_x1", "final_y1", "final_x2", "final_y2"),
        ("bbox_x1", "bbox_y1", "bbox_x2", "bbox_y2"),
        ("xmin", "ymin", "xmax", "ymax"),
    ]

    for cols in direct_sets:
        if all(c in row.index for c in cols):
            try:
                return tuple(float(row[c]) for c in cols)
            except Exception:
                pass

    for c in ["bbox", "final_bbox", "box"]:
        if c in row.index:
            text = str(row[c])
            nums = re.findall(r"-?\d+(?:\.\d+)?", text)
            if len(nums) >= 4:
                return tuple(float(x) for x in nums[:4])

    return None


df_all = pd.read_csv(PRIMARY_ALL)

for c in [
    "split_v20",
    "scan_frame_id",
    "final_box_id",
    "behaviour_code",
    "behaviour_label",
    "visual_marker_colour_v18c",
    "behaviour_pig_id_v18c",
]:
    if c in df_all.columns:
        df_all[c] = df_all[c].fillna("").astype(str).str.strip()

if "split_v20" not in df_all.columns:
    raise ValueError("Missing split_v20 column in primary split dataset.")

if "behaviour_code" not in df_all.columns:
    raise ValueError("Missing behaviour_code column in primary split dataset.")

# Create folder structure.
for split in ["train", "val", "test"]:
    for code in sorted(df_all["behaviour_code"].dropna().astype(str).unique()):
        (CROP_ROOT / split / slug(code)).mkdir(parents=True, exist_ok=True)

index_rows = []
hard_rows = []

for i, row in df_all.iterrows():
    split = str(row.get("split_v20", "")).strip()
    behaviour_code = str(row.get("behaviour_code", "")).strip()
    behaviour_label = str(row.get("behaviour_label", "")).strip()
    scan_frame_id = str(row.get("scan_frame_id", "")).strip()
    final_box_id = str(row.get("final_box_id", "")).strip()
    visual_colour = str(row.get("visual_marker_colour_v18c", "")).strip()
    behaviour_pig_id = str(row.get("behaviour_pig_id_v18c", "")).strip()

    if split not in ["train", "val", "test"]:
        hard_rows.append({
            "row_index": i,
            "issue_type": "invalid_split",
            "issue_detail": split,
        })
        continue

    image_path = find_image_path(row)
    bbox = bbox_from_row(row)

    if not image_path:
        hard_rows.append({
            "row_index": i,
            "issue_type": "missing_image_path",
            "issue_detail": f"{scan_frame_id} {final_box_id}",
        })
        continue

    img_p = Path(image_path)
    if not img_p.exists():
        hard_rows.append({
            "row_index": i,
            "issue_type": "image_file_not_found",
            "issue_detail": image_path,
        })
        continue

    if bbox is None:
        hard_rows.append({
            "row_index": i,
            "issue_type": "missing_bbox",
            "issue_detail": f"{scan_frame_id} {final_box_id}",
        })
        continue

    try:
        x1, y1, x2, y2 = bbox

        with Image.open(img_p) as im:
            im = im.convert("RGB")
            w, h = im.size

            x1c = max(0, min(w - 1, int(round(x1))))
            y1c = max(0, min(h - 1, int(round(y1))))
            x2c = max(0, min(w, int(round(x2))))
            y2c = max(0, min(h, int(round(y2))))

            if x2c <= x1c or y2c <= y1c:
                hard_rows.append({
                    "row_index": i,
                    "issue_type": "invalid_bbox_after_clamp",
                    "issue_detail": f"{scan_frame_id} {final_box_id}: {bbox} -> {(x1c, y1c, x2c, y2c)}",
                })
                continue

            crop = im.crop((x1c, y1c, x2c, y2c))

            crop_name = (
                f"{slug(scan_frame_id)}__{slug(final_box_id)}__"
                f"{slug(visual_colour)}__{slug(behaviour_pig_id)}__{slug(behaviour_code)}.jpg"
            )

            crop_rel = Path(split) / slug(behaviour_code) / crop_name
            crop_out = CROP_ROOT / crop_rel
            crop.save(crop_out, quality=95)

            index_rows.append({
                "row_index_original": i,
                "split": split,
                "scan_frame_id": scan_frame_id,
                "final_box_id": final_box_id,
                "visual_marker_colour_v18c": visual_colour,
                "behaviour_pig_id_v18c": behaviour_pig_id,
                "behaviour_code": behaviour_code,
                "behaviour_label": behaviour_label,
                "source_image_path": image_path,
                "crop_path": str(crop_out),
                "crop_relative_path": str(crop_rel),
                "image_width": w,
                "image_height": h,
                "x1": x1,
                "y1": y1,
                "x2": x2,
                "y2": y2,
                "x1_clamped": x1c,
                "y1_clamped": y1c,
                "x2_clamped": x2c,
                "y2_clamped": y2c,
                "crop_width": x2c - x1c,
                "crop_height": y2c - y1c,
            })

    except Exception as e:
        hard_rows.append({
            "row_index": i,
            "issue_type": "crop_export_exception",
            "issue_detail": f"{scan_frame_id} {final_box_id}: {repr(e)}",
        })

index_df = pd.DataFrame(index_rows)
hard_issues = pd.DataFrame(hard_rows, columns=["row_index", "issue_type", "issue_detail"])

safe_to_csv(index_df, OUT_INDEX)
safe_to_csv(hard_issues, OUT_HARD_ISSUES)

if len(index_df):
    class_dist = (
        index_df.groupby(["split", "behaviour_code", "behaviour_label"], dropna=False)
        .size()
        .reset_index(name="crop_count")
        .sort_values(["split", "crop_count"], ascending=[True, False])
    )

    split_dist = (
        index_df.groupby("split", dropna=False)
        .agg(
            crop_count=("crop_path", "count"),
            frame_count=("scan_frame_id", "nunique"),
            behaviour_class_count=("behaviour_code", "nunique"),
        )
        .reset_index()
        .sort_values("split")
    )
else:
    class_dist = pd.DataFrame(columns=["split", "behaviour_code", "behaviour_label", "crop_count"])
    split_dist = pd.DataFrame(columns=["split", "crop_count", "frame_count", "behaviour_class_count"])

safe_to_csv(class_dist, OUT_CLASS_DIST)
safe_to_csv(split_dist, OUT_SPLIT_DIST)

summary = pd.DataFrame([{
    "source_primary_split": str(PRIMARY_ALL),
    "expected_rows": int(len(df_all)),
    "exported_crops": int(len(index_df)),
    "hard_issue_count": int(len(hard_issues)),
    "crop_root": str(CROP_ROOT),
    "train_crops": int((index_df["split"] == "train").sum()) if len(index_df) else 0,
    "val_crops": int((index_df["split"] == "val").sum()) if len(index_df) else 0,
    "test_crops": int((index_df["split"] == "test").sum()) if len(index_df) else 0,
    "behaviour_class_count": int(index_df["behaviour_code"].nunique()) if len(index_df) else 0,
    "ready_for_feature_extraction": bool(len(hard_issues) == 0 and len(index_df) == len(df_all)),
    "generated_at": datetime.now().isoformat(timespec="seconds"),
}])

safe_to_csv(summary, OUT_SUMMARY)

ready = bool(summary.iloc[0]["ready_for_feature_extraction"])

readme_text = f"""# Week 7 Model-ready Crop Dataset v22

## Purpose

This dataset exports one pig crop image for every training-ready fused behaviour row from the locked primary split v21.

## Source

Primary split: {PRIMARY_ALL}

## Folder structure

crops_by_split_and_behaviour/
  train/
    STI/
    LAI/
    ...
  val/
    ...
  test/
    ...

## Summary

- Expected rows: {int(summary.iloc[0]['expected_rows'])}
- Exported crops: {int(summary.iloc[0]['exported_crops'])}
- Hard issue count: {int(summary.iloc[0]['hard_issue_count'])}
- Ready for feature extraction: {ready}

## Files

- week7_model_ready_crop_dataset_v22_index.csv
- week7_model_ready_crop_dataset_v22_summary.csv
- week7_model_ready_crop_dataset_v22_class_distribution.csv
- week7_model_ready_crop_dataset_v22_split_distribution.csv
- week7_model_ready_crop_dataset_v22_hard_issues.csv
"""

OUT_README.write_text(readme_text)

OUT_NOTE.write_text(
    "# Week 7 Model-ready Crop Dataset v22\n\n"
    "## Purpose\n\n"
    "This step exports a model-ready crop dataset from the locked v21 primary split. "
    "Each crop corresponds to one training-ready pig box with a fused behaviour label.\n\n"
    "## Summary\n\n"
    f"- Expected rows: `{int(summary.iloc[0]['expected_rows'])}`\n"
    f"- Exported crops: `{int(summary.iloc[0]['exported_crops'])}`\n"
    f"- Hard issue count: `{int(summary.iloc[0]['hard_issue_count'])}`\n"
    f"- Train crops: `{int(summary.iloc[0]['train_crops'])}`\n"
    f"- Validation crops: `{int(summary.iloc[0]['val_crops'])}`\n"
    f"- Test crops: `{int(summary.iloc[0]['test_crops'])}`\n"
    f"- Ready for feature extraction: `{ready}`\n\n"
    "## Outputs\n\n"
    f"- Crop root: `{CROP_ROOT}`\n"
    f"- Index: `{OUT_INDEX}`\n"
    f"- Summary: `{OUT_SUMMARY}`\n"
    f"- Class distribution: `{OUT_CLASS_DIST}`\n"
    f"- Split distribution: `{OUT_SPLIT_DIST}`\n"
    f"- Hard issues: `{OUT_HARD_ISSUES}`\n"
    f"- README: `{OUT_README}`\n"
)

print("Saved:")
print(CROP_ROOT)
print(OUT_INDEX)
print(OUT_SUMMARY)
print(OUT_CLASS_DIST)
print(OUT_SPLIT_DIST)
print(OUT_HARD_ISSUES)
print(OUT_README)
print(OUT_NOTE)

print()
print("=== v22 model-ready crop dataset summary ===")
print(summary.to_string(index=False))

print()
print("=== v22 split distribution ===")
print(split_dist.to_string(index=False))

print()
print("=== v22 hard issues ===")
if len(hard_issues):
    print(hard_issues.head(50).to_string(index=False))
else:
    print("No hard issues found.")
