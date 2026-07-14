from pathlib import Path
import csv
import math
import json
from datetime import datetime

import cv2
import numpy as np
import pandas as pd


ROOT = Path.home() / "PigBench"
W6 = ROOT / "Week6_Unibo_Dataset_Validation"

FEAT = W6 / "outputs" / "feature_extractors"
STATS = W6 / "outputs" / "dataset_statistics"
VIS = W6 / "outputs" / "visual_label_check"
NOTES = W6 / "notes"

SAM_FEATURES_PATH = FEAT / "week6_sam_box_prompt_segmentation_features.csv"
SAM_FEATURES_JSON_PATH = FEAT / "week6_sam_box_prompt_segmentation_features.json"
SAM_FRAME_SUMMARY_PATH = FEAT / "week6_sam_box_prompt_segmentation_frame_summary.csv"
SAM_FAILED_PATH = STATS / "week6_sam_box_prompt_segmentation_failed_rows.csv"
SAM_EXCEPTION_PATH = STATS / "week6_sam_box_prompt_segmentation_exception.txt"

BASELINE_SEG_PATH = FEAT / "week6_preliminary_bbox_guided_segmentation_features.csv"

OVERLAY_DIR = VIS / "sam_box_prompt_full_segmentation" / "frame_overlays"
MASK_DIR = VIS / "sam_box_prompt_full_segmentation" / "masks"

QUALITY_SUMMARY_PATH = STATS / "week6_sam_box_prompt_segmentation_quality_summary.csv"
COMPARISON_PATH = STATS / "week6_sam_vs_grabcut_otsu_segmentation_comparison.csv"
COMPARISON_DETAILED_PATH = STATS / "week6_sam_vs_grabcut_otsu_segmentation_comparison_detailed.csv"
STATUS_PATH = STATS / "week6_sam_box_prompt_segmentation_status_summary.csv"
RECOVERY_PATH = STATS / "week6_sam_box_prompt_segmentation_recovery_summary.csv"
CONTACT_SHEET_PATH = VIS / "week6_sam_box_prompt_segmentation_contact_sheet.jpg"
NOTE_PATH = NOTES / "week6_sam_box_prompt_full_segmentation_notes.md"

STATS.mkdir(parents=True, exist_ok=True)
NOTES.mkdir(parents=True, exist_ok=True)
VIS.mkdir(parents=True, exist_ok=True)


def safe_to_csv(df, path):
    df.to_csv(
        path,
        index=False,
        quoting=csv.QUOTE_ALL,
        escapechar="\\",
        lineterminator="\n",
    )


def read_csv_or_empty(path):
    path = Path(path)
    if not path.exists():
        return pd.DataFrame()
    try:
        return pd.read_csv(path)
    except Exception:
        return pd.DataFrame()


def create_contact_sheet(image_paths, out_path, thumb_w=360, thumb_h=240, cols=4):
    valid = []

    for p in image_paths:
        p = Path(p)

        if not p.exists():
            continue

        img = cv2.imread(str(p))

        if img is None:
            continue

        valid.append((p, img))

    if not valid:
        return False

    rows = math.ceil(len(valid) / cols)
    sheet = np.full((rows * thumb_h, cols * thumb_w, 3), 255, dtype=np.uint8)

    for i, (p, img) in enumerate(valid):
        r = i // cols
        c = i % cols

        resized = cv2.resize(img, (thumb_w, thumb_h), interpolation=cv2.INTER_AREA)

        y0 = r * thumb_h
        x0 = c * thumb_w

        sheet[y0:y0 + thumb_h, x0:x0 + thumb_w] = resized

        text = p.stem[:44]
        cv2.putText(
            sheet,
            text,
            (x0 + 6, y0 + 20),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            (0, 0, 0),
            1,
            cv2.LINE_AA,
        )

    cv2.imwrite(str(out_path), sheet)
    return True


def choose_baseline_area_col(base):
    candidates = [
        "mask_area_pixels",
        "segmentation_area_pixels",
        "foreground_area_pixels",
        "mask_area",
        "successful_or_fallback_mask_area_pixels",
    ]

    for c in candidates:
        if c in base.columns:
            return c

    # fallback: any column with area and pixel in name
    for c in base.columns:
        lc = c.lower()
        if "area" in lc and ("pixel" in lc or "pixels" in lc):
            return c

    return None


def compare_with_baseline(sam_df, base_df):
    if len(sam_df) == 0:
        return pd.DataFrame([
            {
                "comparison": "sam_feature_rows",
                "status": "FAIL",
                "evidence": "0 rows",
            }
        ]), pd.DataFrame()

    if len(base_df) == 0:
        return pd.DataFrame([
            {
                "comparison": "baseline_file_readable",
                "status": "WARN",
                "evidence": f"missing_or_empty={BASELINE_SEG_PATH}",
            }
        ]), pd.DataFrame()

    key_cols = []

    for col in ["scan_frame_id", "det_id"]:
        if col in sam_df.columns and col in base_df.columns:
            key_cols.append(col)

    if len(key_cols) < 2:
        return pd.DataFrame([
            {
                "comparison": "join_keys_available",
                "status": "WARN",
                "evidence": f"sam_cols={list(sam_df.columns)}; base_cols={list(base_df.columns)}",
            }
        ]), pd.DataFrame()

    base_area_col = choose_baseline_area_col(base_df)

    if base_area_col is None:
        return pd.DataFrame([
            {
                "comparison": "baseline_area_column_available",
                "status": "WARN",
                "evidence": f"base_cols={list(base_df.columns)}",
            }
        ]), pd.DataFrame()

    sam_area_col = "mask_area_pixels"

    if sam_area_col not in sam_df.columns:
        return pd.DataFrame([
            {
                "comparison": "sam_area_column_available",
                "status": "FAIL",
                "evidence": f"sam_cols={list(sam_df.columns)}",
            }
        ]), pd.DataFrame()

    left_cols = key_cols + [
        "row_index",
        "mask_area_pixels",
        "bbox_area_pixels",
        "mask_area_fraction_of_bbox",
        "sam_predicted_iou_score",
    ]

    left_cols = [c for c in left_cols if c in sam_df.columns]

    right_cols = key_cols + [base_area_col]
    right_cols = list(dict.fromkeys(right_cols))

    merged = sam_df[left_cols].merge(
        base_df[right_cols],
        on=key_cols,
        how="left",
        suffixes=("_sam", "_baseline"),
    )

    # After merge, if both tables have mask_area_pixels, pandas renames them.
    if "mask_area_pixels_sam" in merged.columns:
        sam_area_after = "mask_area_pixels_sam"
    elif "mask_area_pixels" in merged.columns:
        sam_area_after = "mask_area_pixels"
    else:
        sam_area_after = None

    if f"{base_area_col}_baseline" in merged.columns:
        base_area_after = f"{base_area_col}_baseline"
    elif base_area_col in merged.columns:
        base_area_after = base_area_col
    else:
        base_area_after = None

    if sam_area_after is None or base_area_after is None:
        return pd.DataFrame([
            {
                "comparison": "post_merge_area_columns_available",
                "status": "WARN",
                "evidence": f"merged_cols={list(merged.columns)}",
            }
        ]), merged

    merged["sam_mask_area_pixels_for_comparison"] = pd.to_numeric(
        merged[sam_area_after],
        errors="coerce",
    )

    merged["baseline_mask_area_pixels_for_comparison"] = pd.to_numeric(
        merged[base_area_after],
        errors="coerce",
    )

    matched = int(merged["baseline_mask_area_pixels_for_comparison"].notna().sum())

    if matched == 0:
        comparison = pd.DataFrame([
            {
                "comparison": "matched_rows",
                "status": "WARN",
                "evidence": "0",
            }
        ])

        return comparison, merged

    merged["sam_minus_baseline_area_pixels"] = (
        merged["sam_mask_area_pixels_for_comparison"]
        - merged["baseline_mask_area_pixels_for_comparison"]
    )

    merged["sam_to_baseline_area_ratio"] = (
        merged["sam_mask_area_pixels_for_comparison"]
        / merged["baseline_mask_area_pixels_for_comparison"].replace(0, np.nan)
    )

    ratio_clean = merged["sam_to_baseline_area_ratio"].replace([np.inf, -np.inf], np.nan)

    comparison = pd.DataFrame([
        {
            "comparison": "matched_rows",
            "status": "PASS" if matched > 0 else "WARN",
            "evidence": matched,
        },
        {
            "comparison": "sam_rows",
            "status": "INFO",
            "evidence": len(sam_df),
        },
        {
            "comparison": "baseline_rows",
            "status": "INFO",
            "evidence": len(base_df),
        },
        {
            "comparison": "mean_sam_mask_area_pixels",
            "status": "INFO",
            "evidence": float(merged["sam_mask_area_pixels_for_comparison"].mean()),
        },
        {
            "comparison": "mean_baseline_mask_area_pixels",
            "status": "INFO",
            "evidence": float(merged["baseline_mask_area_pixels_for_comparison"].mean()),
        },
        {
            "comparison": "median_sam_to_baseline_area_ratio",
            "status": "INFO",
            "evidence": float(ratio_clean.median()),
        },
        {
            "comparison": "mean_sam_to_baseline_area_ratio",
            "status": "INFO",
            "evidence": float(ratio_clean.mean()),
        },
    ])

    return comparison, merged


sam = read_csv_or_empty(SAM_FEATURES_PATH)
frame_summary = read_csv_or_empty(SAM_FRAME_SUMMARY_PATH)
failed = read_csv_or_empty(SAM_FAILED_PATH)
baseline = read_csv_or_empty(BASELINE_SEG_PATH)

mask_files = sorted(MASK_DIR.glob("*.png")) if MASK_DIR.exists() else []
overlay_files = sorted(OVERLAY_DIR.glob("*_sam_box_prompt_overlay.jpg")) if OVERLAY_DIR.exists() else []

comparison, detailed = compare_with_baseline(sam, baseline)
safe_to_csv(comparison, COMPARISON_PATH)

if len(detailed):
    safe_to_csv(detailed, COMPARISON_DETAILED_PATH)

if len(sam):
    quality = pd.DataFrame([
        {
            "metric": "total_input_detections_expected",
            "value": 540,
        },
        {
            "metric": "successful_sam_masks",
            "value": len(sam),
        },
        {
            "metric": "failed_sam_masks",
            "value": len(failed),
        },
        {
            "metric": "success_rate",
            "value": len(sam) / 540,
        },
        {
            "metric": "frame_overlay_files",
            "value": len(overlay_files),
        },
        {
            "metric": "mask_png_files",
            "value": len(mask_files),
        },
        {
            "metric": "mean_sam_predicted_iou_score",
            "value": float(pd.to_numeric(sam.get("sam_predicted_iou_score"), errors="coerce").mean()),
        },
        {
            "metric": "median_sam_predicted_iou_score",
            "value": float(pd.to_numeric(sam.get("sam_predicted_iou_score"), errors="coerce").median()),
        },
        {
            "metric": "mean_mask_area_fraction_of_bbox",
            "value": float(pd.to_numeric(sam.get("mask_area_fraction_of_bbox"), errors="coerce").mean()),
        },
        {
            "metric": "median_mask_area_fraction_of_bbox",
            "value": float(pd.to_numeric(sam.get("mask_area_fraction_of_bbox"), errors="coerce").median()),
        },
        {
            "metric": "device",
            "value": str(sam.get("device", pd.Series(["unknown"])).iloc[0]),
        },
        {
            "metric": "sam_model_type",
            "value": str(sam.get("sam_model_type", pd.Series(["unknown"])).iloc[0]),
        },
    ])
else:
    quality = pd.DataFrame([
        {
            "metric": "successful_sam_masks",
            "value": 0,
        }
    ])

safe_to_csv(quality, QUALITY_SUMMARY_PATH)

# representative contact sheet
if len(overlay_files) > 16:
    idx = np.linspace(0, len(overlay_files) - 1, 16).round().astype(int)
    selected = [overlay_files[i] for i in idx]
else:
    selected = overlay_files

contact_sheet_ok = create_contact_sheet(selected, CONTACT_SHEET_PATH)

status = pd.DataFrame([
    {
        "item": "sam_feature_table_exists",
        "status": "PASS" if SAM_FEATURES_PATH.exists() else "FAIL",
        "evidence": str(SAM_FEATURES_PATH),
    },
    {
        "item": "sam_feature_rows",
        "status": "PASS" if len(sam) == 540 else "WARN",
        "evidence": len(sam),
    },
    {
        "item": "sam_mask_png_files",
        "status": "PASS" if len(mask_files) == 540 else "WARN",
        "evidence": len(mask_files),
    },
    {
        "item": "sam_frame_summary_rows",
        "status": "PASS" if len(frame_summary) == 72 else "WARN",
        "evidence": len(frame_summary),
    },
    {
        "item": "sam_frame_overlay_files",
        "status": "PASS" if len(overlay_files) == 72 else "WARN",
        "evidence": len(overlay_files),
    },
    {
        "item": "sam_failed_rows",
        "status": "PASS" if len(failed) == 0 else "WARN",
        "evidence": len(failed),
    },
    {
        "item": "sam_vs_baseline_comparison",
        "status": "PASS" if len(comparison) and not (comparison["status"] == "FAIL").any() else "WARN",
        "evidence": str(COMPARISON_PATH),
    },
    {
        "item": "sam_contact_sheet",
        "status": "PASS" if contact_sheet_ok else "WARN",
        "evidence": str(CONTACT_SHEET_PATH),
    },
    {
        "item": "previous_exception_recorded",
        "status": "INFO" if SAM_EXCEPTION_PATH.exists() else "INFO",
        "evidence": str(SAM_EXCEPTION_PATH) if SAM_EXCEPTION_PATH.exists() else "No previous exception file.",
    },
])

safe_to_csv(status, STATUS_PATH)

recovery = pd.DataFrame([
    {
        "metric": "recovery_generated_at",
        "value": datetime.now().isoformat(timespec="seconds"),
    },
    {
        "metric": "reason",
        "value": "Original full Segment Anything Model run completed masks but failed during post-hoc baseline comparison because of a pandas merge column-name collision.",
    },
    {
        "metric": "reran_sam_inference",
        "value": False,
    },
    {
        "metric": "used_existing_sam_feature_outputs",
        "value": True,
    },
    {
        "metric": "sam_rows",
        "value": len(sam),
    },
    {
        "metric": "frame_summary_rows",
        "value": len(frame_summary),
    },
    {
        "metric": "overlay_files",
        "value": len(overlay_files),
    },
    {
        "metric": "mask_png_files",
        "value": len(mask_files),
    },
])

safe_to_csv(recovery, RECOVERY_PATH)

with open(NOTE_PATH, "w") as f:
    f.write("# Week 6 Segment Anything Model Full Box-Prompt Segmentation\n\n")

    f.write("## Purpose\n\n")
    f.write(
        "This step applies Segment Anything Model to all pig detector bounding boxes. "
        "Each detector bounding box is used as a box prompt, producing one automatic segmentation mask per detection.\n\n"
    )

    f.write("## Recovery note\n\n")
    f.write(
        "The original full run completed Segment Anything Model mask generation for all detections, "
        "but the final post-processing script stopped during baseline comparison because of a pandas column-name collision. "
        "This recovery step did not rerun Segment Anything Model inference. It finalized the already generated masks, overlays, feature tables, comparison table, status table, and documentation.\n\n"
    )

    f.write("## Quality summary\n\n")
    f.write(quality.to_markdown(index=False))
    f.write("\n\n")

    f.write("## Status checks\n\n")
    f.write(status.to_markdown(index=False))
    f.write("\n\n")

    f.write("## Comparison with classical GrabCut/Otsu baseline\n\n")
    f.write(comparison.to_markdown(index=False))
    f.write("\n\n")

    f.write("## Outputs\n\n")
    f.write(f"- Segment Anything Model feature table: `{SAM_FEATURES_PATH}`\n")
    f.write(f"- Segment Anything Model JSON features: `{SAM_FEATURES_JSON_PATH}`\n")
    f.write(f"- Frame summary: `{SAM_FRAME_SUMMARY_PATH}`\n")
    f.write(f"- Quality summary: `{QUALITY_SUMMARY_PATH}`\n")
    f.write(f"- Status summary: `{STATUS_PATH}`\n")
    f.write(f"- Baseline comparison: `{COMPARISON_PATH}`\n")
    f.write(f"- Detailed baseline comparison: `{COMPARISON_DETAILED_PATH}`\n")
    f.write(f"- Mask directory: `{MASK_DIR}`\n")
    f.write(f"- Overlay directory: `{OVERLAY_DIR}`\n")
    f.write(f"- Contact sheet: `{CONTACT_SHEET_PATH}`\n\n")

    f.write("## Interpretation\n\n")

    if len(sam) == 540 and len(overlay_files) == 72 and len(failed) == 0:
        f.write(
            "The full Segment Anything Model box-prompt segmentation is complete. "
            "It produced one automatic Segment Anything Model mask per detector bounding box, for 540 masks across 72 scanpoint frames. "
            "This is a stronger foundation-model-based segmentation route in addition to the earlier classical GrabCut/Otsu baseline. "
            "The masks remain automatic segmentation outputs, not manual segmentation ground truth.\n"
        )
    else:
        f.write(
            "The Segment Anything Model output is partially complete. Inspect the status checks before using it as a final segmentation deliverable.\n"
        )

print("Saved:")
print(QUALITY_SUMMARY_PATH)
print(COMPARISON_PATH)
print(COMPARISON_DETAILED_PATH if len(detailed) else "No detailed comparison written.")
print(STATUS_PATH)
print(RECOVERY_PATH)
print(CONTACT_SHEET_PATH)
print(NOTE_PATH)

print()
print("=== Recovery summary ===")
print(recovery.to_string(index=False))

print()
print("=== Status ===")
print(status.to_string(index=False))

print()
print("=== Comparison ===")
print(comparison.to_string(index=False))
