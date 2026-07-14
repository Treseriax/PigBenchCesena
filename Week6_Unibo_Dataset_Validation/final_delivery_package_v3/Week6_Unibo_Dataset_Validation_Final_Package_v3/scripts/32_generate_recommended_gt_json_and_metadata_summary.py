from pathlib import Path
import csv
import json
import pandas as pd
import numpy as np


ROOT = Path.home() / "PigBench"
W6 = ROOT / "Week6_Unibo_Dataset_Validation"

OUT_GT = W6 / "outputs/unified_ground_truth"
OUT_STATS = W6 / "outputs/dataset_statistics"
NOTES = W6 / "notes"

GT_CSV = OUT_GT / "week6_unified_ground_truth_v2_with_recommended_split_v2.csv"
FRAME_INDEX = OUT_GT / "week6_scanpoint_frame_index.csv"
DETECTIONS_QC = W6 / "outputs/feature_extractors/week6_yolov8s_all_scanpoint_detections_with_qc_flags.csv"

OUT_GT.mkdir(parents=True, exist_ok=True)
OUT_STATS.mkdir(parents=True, exist_ok=True)
NOTES.mkdir(parents=True, exist_ok=True)


def safe_to_csv(df, path):
    df.to_csv(
        path,
        index=False,
        quoting=csv.QUOTE_ALL,
        escapechar="\\",
        lineterminator="\n"
    )


def json_clean_value(v):
    if pd.isna(v):
        return None
    if isinstance(v, (np.integer,)):
        return int(v)
    if isinstance(v, (np.floating,)):
        return float(v)
    if isinstance(v, (np.bool_,)):
        return bool(v)
    return v


def df_to_clean_records(df):
    records = []
    for row in df.to_dict(orient="records"):
        records.append({k: json_clean_value(v) for k, v in row.items()})
    return records


if not GT_CSV.exists():
    raise FileNotFoundError(GT_CSV)

gt = pd.read_csv(GT_CSV)
frames = pd.read_csv(FRAME_INDEX) if FRAME_INDEX.exists() else pd.DataFrame()
dets = pd.read_csv(DETECTIONS_QC) if DETECTIONS_QC.exists() else pd.DataFrame()

# ------------------------------------------------------------
# 1) Recommended GT JSON, records format
# ------------------------------------------------------------
gt_records = df_to_clean_records(gt)

gt_json_path = OUT_GT / "week6_unified_ground_truth_v2_with_recommended_split_v2.json"

with open(gt_json_path, "w") as f:
    json.dump(
        {
            "dataset": "Week6_Unibo_Dataset_Validation",
            "version": "recommended_split_v2",
            "record_count": len(gt_records),
            "notes": {
                "manual_labels": "Manual Excel scan-sampling labels.",
                "bbox_fields": "GT bbox fields are not manual bbox ground truth. Detector bboxes are stored separately in feature extractor outputs.",
                "video_linkage": "Direct TLC videos are high confidence. Recovered c-token videos remain medium confidence.",
                "split_policy": "Recommended split v2 uses video-hour units to avoid leakage."
            },
            "records": gt_records,
        },
        f,
        indent=2,
        ensure_ascii=False,
    )

# ------------------------------------------------------------
# 2) Schema JSON
# ------------------------------------------------------------
schema_fields = []

for col in gt.columns:
    dtype = str(gt[col].dtype)

    if col in ["record_id"]:
        role = "unique_record_identifier"
    elif col in ["video_id", "video_path", "video_match_status", "video_mapping_confidence"]:
        role = "video_linkage"
    elif col in ["crate_id", "pen_id", "camera_id", "date", "hour_start", "hour_end"]:
        role = "experiment_metadata"
    elif col in ["timestamp", "frame_index", "timestamp_sec_in_video"]:
        role = "temporal_alignment"
    elif col in ["pig_id", "colour_id", "colour_raw"]:
        role = "manual_pig_colour_identity"
    elif col.startswith("bbox") or col in ["bbox_available", "bbox_source"]:
        role = "bbox_placeholder_or_source_flag"
    elif col in ["behaviour_code", "behaviour_label", "label_source"]:
        role = "manual_behaviour_label"
    elif "split" in col:
        role = "split_protocol"
    else:
        role = "other_metadata"

    schema_fields.append({
        "field": col,
        "dtype": dtype,
        "role": role,
        "non_null_count": int(gt[col].notna().sum()),
        "example_value": json_clean_value(gt[col].dropna().iloc[0]) if gt[col].notna().any() else None,
    })

schema = {
    "dataset": "Week6_Unibo_Dataset_Validation",
    "table": "week6_unified_ground_truth_v2_with_recommended_split_v2",
    "record_count": len(gt),
    "field_count": len(gt.columns),
    "primary_key": "record_id",
    "manual_label_granularity": "10-minute scan sampling",
    "bbox_policy": "Detector bboxes are separate from manual GT; GT bbox fields are not manual bbox annotations.",
    "split_policy": "video-hour-level no leakage; recommended_split_v2",
    "fields": schema_fields,
}

schema_json_path = OUT_GT / "week6_unified_ground_truth_v2_with_recommended_split_v2_schema.json"

with open(schema_json_path, "w") as f:
    json.dump(schema, f, indent=2, ensure_ascii=False)

schema_csv_path = OUT_STATS / "week6_recommended_gt_schema_fields.csv"
safe_to_csv(pd.DataFrame(schema_fields), schema_csv_path)

# ------------------------------------------------------------
# 3) Nested frame annotation JSON for viewer/interface
# ------------------------------------------------------------
frame_records = []

if len(frames):
    frame_base = frames.copy()
else:
    # Build minimal frame base from GT if frame index table is unavailable.
    group_cols = [
        "video_id",
        "video_path",
        "video_match_status",
        "camera_id",
        "crate_id",
        "pen_id",
        "timestamp",
        "frame_index",
        "timestamp_sec_in_video",
        "recommended_split_v2",
    ]
    group_cols = [c for c in group_cols if c in gt.columns]
    frame_base = gt[group_cols].drop_duplicates().copy()
    frame_base["scan_frame_id"] = [f"scanframe_{i:04d}" for i in range(len(frame_base))]

for _, fr in frame_base.iterrows():
    scan_frame_id = fr.get("scan_frame_id", None)

    if scan_frame_id is not None and "scan_frame_id" in gt.columns:
        labels = gt[gt["scan_frame_id"] == scan_frame_id].copy()
    else:
        # Fallback: match timestamp/video.
        labels = gt[
            (gt["video_id"].astype(str) == str(fr.get("video_id", "")))
            & (gt["timestamp"].astype(str) == str(fr.get("timestamp", "")))
        ].copy()

    label_records = []

    for _, lab in labels.iterrows():
        label_records.append({
            "record_id": json_clean_value(lab.get("record_id")),
            "colour_id": json_clean_value(lab.get("colour_id")),
            "colour_raw": json_clean_value(lab.get("colour_raw")),
            "pig_id": json_clean_value(lab.get("pig_id")),
            "behaviour_code": json_clean_value(lab.get("behaviour_code")),
            "behaviour_label": json_clean_value(lab.get("behaviour_label")),
            "label_source": json_clean_value(lab.get("label_source")),
            "bbox_available_in_manual_gt": json_clean_value(lab.get("bbox_available")),
        })

    detection_records = []

    if len(dets) and scan_frame_id is not None and "scan_frame_id" in dets.columns:
        dfg = dets[dets["scan_frame_id"].astype(str) == str(scan_frame_id)].copy()

        for _, d in dfg.iterrows():
            detection_records.append({
                "det_id": json_clean_value(d.get("det_id")),
                "bbox_xyxy": [
                    json_clean_value(d.get("x1")),
                    json_clean_value(d.get("y1")),
                    json_clean_value(d.get("x2")),
                    json_clean_value(d.get("y2")),
                ],
                "score": json_clean_value(d.get("score")),
                "bbox_area": json_clean_value(d.get("bbox_area")),
                "bbox_count_issue_type": json_clean_value(d.get("bbox_count_issue_type")),
                "recommended_use": json_clean_value(d.get("recommended_use")),
            })

    frame_records.append({
        "scan_frame_id": json_clean_value(scan_frame_id),
        "video_id": json_clean_value(fr.get("video_id")),
        "video_path": json_clean_value(fr.get("video_path")),
        "frame_image_path": json_clean_value(fr.get("frame_image_path")),
        "video_match_status": json_clean_value(fr.get("video_match_status")),
        "camera_id": json_clean_value(fr.get("camera_id")),
        "crate_id": json_clean_value(fr.get("crate_id")),
        "pen_id": json_clean_value(fr.get("pen_id")),
        "timestamp": json_clean_value(fr.get("timestamp")),
        "frame_index": json_clean_value(fr.get("frame_index")),
        "timestamp_sec_in_video": json_clean_value(fr.get("timestamp_sec_in_video")),
        "manual_labels": label_records,
        "detector_bboxes": detection_records,
    })

nested_json_path = OUT_GT / "week6_scanpoint_annotations_nested_for_viewer.json"

with open(nested_json_path, "w") as f:
    json.dump(
        {
            "dataset": "Week6_Unibo_Dataset_Validation",
            "version": "viewer_nested_annotations_v1",
            "frame_count": len(frame_records),
            "notes": {
                "manual_labels": "Six manual colour-ID behaviour labels per scanpoint frame.",
                "detector_bboxes": "Automatic YOLOv8-s detections with QC flags, not manual GT bboxes.",
            },
            "frames": frame_records,
        },
        f,
        indent=2,
        ensure_ascii=False,
    )

# ------------------------------------------------------------
# 4) Crate/pen/camera/video summary
# ------------------------------------------------------------
summary_group_cols = [
    "date",
    "camera_id",
    "crate_id",
    "pen_id",
    "video_id",
    "video_match_status",
    "video_mapping_confidence",
    "hour_start",
    "hour_end",
    "recommended_split_v2",
]
summary_group_cols = [c for c in summary_group_cols if c in gt.columns]

agg_dict = {
    "record_id": "count",
    "colour_id": pd.Series.nunique,
    "behaviour_code": pd.Series.nunique,
}

summary = (
    gt.groupby(summary_group_cols, dropna=False)
    .agg(agg_dict)
    .reset_index()
    .rename(columns={
        "record_id": "label_count",
        "colour_id": "unique_colour_ids",
        "behaviour_code": "unique_behaviour_codes",
    })
    .sort_values([c for c in ["date", "camera_id", "crate_id", "pen_id", "hour_start"] if c in summary_group_cols])
)

if len(dets) and len(frames):
    det_frame_summary = (
        dets.groupby("scan_frame_id")
        .size()
        .reset_index(name="bbox_count")
    )

    frame_det = frames.merge(det_frame_summary, on="scan_frame_id", how="left")
    frame_det["bbox_count"] = frame_det["bbox_count"].fillna(0)

    det_group_cols = [c for c in ["video_id", "timestamp"] if c in frame_det.columns]
    if "video_id" in frame_det.columns:
        video_det = (
            frame_det.groupby("video_id")
            .agg(
                scanpoint_frame_count=("scan_frame_id", "count"),
                total_bboxes=("bbox_count", "sum"),
                mean_bboxes_per_frame=("bbox_count", "mean"),
            )
            .reset_index()
        )

        summary = summary.merge(video_det, on="video_id", how="left")

summary_path = OUT_STATS / "week6_camera_pen_crate_video_summary.csv"
safe_to_csv(summary, summary_path)

# ------------------------------------------------------------
# 5) Verification summary
# ------------------------------------------------------------
verification_rows = [
    {
        "check_name": "recommended_gt_json_exists",
        "status": "PASS" if gt_json_path.exists() else "FAIL",
        "observed": str(gt_json_path.exists()),
        "expected": "True",
    },
    {
        "check_name": "recommended_gt_json_record_count",
        "status": "PASS" if len(gt_records) == len(gt) == 432 else "FAIL",
        "observed": len(gt_records),
        "expected": 432,
    },
    {
        "check_name": "schema_json_exists",
        "status": "PASS" if schema_json_path.exists() else "FAIL",
        "observed": str(schema_json_path.exists()),
        "expected": "True",
    },
    {
        "check_name": "nested_viewer_json_frame_count",
        "status": "PASS" if len(frame_records) == 72 else "WARN",
        "observed": len(frame_records),
        "expected": 72,
    },
    {
        "check_name": "camera_pen_crate_summary_exists",
        "status": "PASS" if summary_path.exists() else "FAIL",
        "observed": str(summary_path.exists()),
        "expected": "True",
    },
]

verification = pd.DataFrame(verification_rows)
verification_path = OUT_STATS / "week6_recommended_gt_json_generation_verification.csv"
safe_to_csv(verification, verification_path)

# ------------------------------------------------------------
# 6) Notes
# ------------------------------------------------------------
note_path = NOTES / "week6_recommended_gt_json_and_metadata_summary_notes.md"

with open(note_path, "w") as f:
    f.write("# Week 6 Recommended GT JSON and Metadata Summary\n\n")

    f.write("## Purpose\n\n")
    f.write(
        "This step closes the Task 2 JSON gap and strengthens the Task 1 dataset metadata summary. "
        "It creates a recommended GT JSON, schema file, nested viewer annotation JSON, and an explicit camera/pen/crate/video summary table.\n\n"
    )

    f.write("## Outputs\n\n")
    f.write(f"- Recommended GT JSON: `{gt_json_path}`\n")
    f.write(f"- GT schema JSON: `{schema_json_path}`\n")
    f.write(f"- GT schema fields CSV: `{schema_csv_path}`\n")
    f.write(f"- Nested viewer annotation JSON: `{nested_json_path}`\n")
    f.write(f"- Camera/pen/crate/video summary: `{summary_path}`\n")
    f.write(f"- Verification: `{verification_path}`\n\n")

    f.write("## Verification\n\n")
    f.write(verification.to_markdown(index=False))
    f.write("\n\n")

    f.write("## Camera/pen/crate/video summary preview\n\n")
    f.write(summary.head(20).to_markdown(index=False) if len(summary) else "No summary generated.")
    f.write("\n\n")

    f.write("## Interpretation\n\n")
    f.write(
        "The recommended GT is now available in both CSV and JSON format. "
        "The nested viewer JSON is prepared for visualization interfaces because it groups each scanpoint frame with its six manual labels and detector bboxes. "
        "Detector bboxes remain separate automatic outputs and should not be confused with manual bbox ground truth.\n"
    )

print("Saved:")
print(gt_json_path)
print(schema_json_path)
print(schema_csv_path)
print(nested_json_path)
print(summary_path)
print(verification_path)
print(note_path)

print()
print("=== Verification ===")
print(verification.to_string(index=False))

print()
print("=== Camera/pen/crate/video summary preview ===")
print(summary.head(20).to_string(index=False))
