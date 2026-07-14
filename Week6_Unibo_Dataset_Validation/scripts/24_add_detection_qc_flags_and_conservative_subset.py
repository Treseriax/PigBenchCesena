from pathlib import Path
import csv
import pandas as pd


PROJECT_ROOT = Path.home() / "PigBench"
W6 = PROJECT_ROOT / "Week6_Unibo_Dataset_Validation"

OUT_STATS = W6 / "outputs/dataset_statistics"
OUT_FEAT = W6 / "outputs/feature_extractors"
NOTES = W6 / "notes"

DETECTIONS = OUT_FEAT / "week6_yolov8s_all_scanpoint_detections.csv"
FRAME_SUMMARY = OUT_FEAT / "week6_yolov8s_all_scanpoint_frame_summary.csv"
LOW = OUT_STATS / "week6_integrity_low_bbox_frames.csv"
HIGH = OUT_STATS / "week6_integrity_high_bbox_frames.csv"
THRESHOLD_SUMMARY = OUT_STATS / "week6_bbox_warning_threshold_sensitivity_summary.csv"
DUPLICATE_SUMMARY = OUT_STATS / "week6_bbox_warning_duplicate_overlap_summary.csv"

OUT_FEAT.mkdir(parents=True, exist_ok=True)
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


def read_optional(path):
    if path.exists():
        return pd.read_csv(path)
    return pd.DataFrame()


if not DETECTIONS.exists():
    raise FileNotFoundError(DETECTIONS)

dets = pd.read_csv(DETECTIONS)
frame_summary = read_optional(FRAME_SUMMARY)
low = read_optional(LOW)
high = read_optional(HIGH)
threshold_summary = read_optional(THRESHOLD_SUMMARY)
duplicate_summary = read_optional(DUPLICATE_SUMMARY)

low_ids = set(low["scan_frame_id"].astype(str)) if len(low) else set()
high_ids = set(high["scan_frame_id"].astype(str)) if len(high) else set()

def issue_type(scan_frame_id):
    s = str(scan_frame_id)
    if s in low_ids:
        return "low_bbox_count_warning"
    if s in high_ids:
        return "high_bbox_count_warning"
    return "normal_bbox_count"

def score_band(score):
    score = float(score)
    if score >= 0.70:
        return "very_high_score_ge_0_70"
    if score >= 0.50:
        return "high_score_ge_0_50"
    if score >= 0.35:
        return "medium_score_ge_0_35"
    if score >= 0.25:
        return "low_accepted_score_ge_0_25"
    return "below_primary_threshold"

dets_qc = dets.copy()
dets_qc["bbox_count_issue_type"] = dets_qc["scan_frame_id"].apply(issue_type)
dets_qc["detection_score_band"] = dets_qc["score"].apply(score_band)

dets_qc["recommended_use"] = "primary_detection_keep"
dets_qc.loc[
    dets_qc["bbox_count_issue_type"] == "high_bbox_count_warning",
    "recommended_use"
] = "keep_but_review_high_bbox_frame"

dets_qc.loc[
    dets_qc["bbox_count_issue_type"] == "low_bbox_count_warning",
    "recommended_use"
] = "keep_but_review_low_bbox_frame"

dets_qc["conservative_keep_score_ge_0_50"] = dets_qc["score"].astype(float) >= 0.50
dets_qc["conservative_keep_score_ge_0_70"] = dets_qc["score"].astype(float) >= 0.70

dets_qc["qc_note"] = ""
dets_qc.loc[
    dets_qc["bbox_count_issue_type"] == "low_bbox_count_warning",
    "qc_note"
] = "Frame has fewer than six accepted bboxes at threshold 0.25; possible occlusion/missed detection. Do not auto-fill with low-confidence boxes."

dets_qc.loc[
    dets_qc["bbox_count_issue_type"] == "high_bbox_count_warning",
    "qc_note"
] = "Frame has more than ten accepted bboxes at threshold 0.25; possible duplicate/false positive detections. Conservative score filtering may be used for analysis."

# Outputs.
qc_detections_path = OUT_FEAT / "week6_yolov8s_all_scanpoint_detections_with_qc_flags.csv"
safe_to_csv(dets_qc, qc_detections_path)

conservative_050 = dets_qc[dets_qc["conservative_keep_score_ge_0_50"]].copy()
conservative_070 = dets_qc[dets_qc["conservative_keep_score_ge_0_70"]].copy()

conservative_050_path = OUT_FEAT / "week6_yolov8s_conservative_detections_score_ge_0_50.csv"
conservative_070_path = OUT_FEAT / "week6_yolov8s_conservative_detections_score_ge_0_70.csv"

safe_to_csv(conservative_050, conservative_050_path)
safe_to_csv(conservative_070, conservative_070_path)

# Frame-level QC summary from detections.
frame_qc = (
    dets_qc.groupby(["scan_frame_id", "bbox_count_issue_type"])
    .agg(
        bbox_count=("det_id", "count"),
        mean_score=("score", "mean"),
        max_score=("score", "max"),
        score_ge_0_50_count=("conservative_keep_score_ge_0_50", "sum"),
        score_ge_0_70_count=("conservative_keep_score_ge_0_70", "sum"),
    )
    .reset_index()
)

if len(frame_summary):
    merge_cols = [c for c in ["scan_frame_id", "video_id", "timestamp", "video_match_status"] if c in frame_summary.columns]
    frame_qc = frame_qc.merge(frame_summary[merge_cols], on="scan_frame_id", how="left")

frame_qc_path = OUT_STATS / "week6_detection_qc_frame_summary.csv"
safe_to_csv(frame_qc, frame_qc_path)

issue_summary = (
    dets_qc.groupby(["bbox_count_issue_type", "detection_score_band"])
    .size()
    .reset_index(name="detection_count")
    .sort_values(["bbox_count_issue_type", "detection_score_band"])
)

issue_summary_path = OUT_STATS / "week6_detection_qc_issue_score_band_summary.csv"
safe_to_csv(issue_summary, issue_summary_path)

conservative_summary = pd.DataFrame([
    {
        "subset": "primary_score_ge_0_25",
        "detection_count": len(dets_qc),
        "frame_count_with_detections": dets_qc["scan_frame_id"].nunique(),
        "mean_bboxes_per_frame": round(dets_qc.groupby("scan_frame_id").size().mean(), 3),
        "low_bbox_warning_frames": len(low_ids),
        "high_bbox_warning_frames": len(high_ids),
    },
    {
        "subset": "conservative_score_ge_0_50",
        "detection_count": len(conservative_050),
        "frame_count_with_detections": conservative_050["scan_frame_id"].nunique(),
        "mean_bboxes_per_frame": round(conservative_050.groupby("scan_frame_id").size().mean(), 3),
        "low_bbox_warning_frames": "",
        "high_bbox_warning_frames": "",
    },
    {
        "subset": "conservative_score_ge_0_70",
        "detection_count": len(conservative_070),
        "frame_count_with_detections": conservative_070["scan_frame_id"].nunique(),
        "mean_bboxes_per_frame": round(conservative_070.groupby("scan_frame_id").size().mean(), 3),
        "low_bbox_warning_frames": "",
        "high_bbox_warning_frames": "",
    },
])

conservative_summary_path = OUT_STATS / "week6_detection_qc_conservative_subset_summary.csv"
safe_to_csv(conservative_summary, conservative_summary_path)

note_path = NOTES / "week6_detection_qc_flags_and_conservative_subset_notes.md"

with open(note_path, "w") as f:
    f.write("# Week 6 Detection QC Flags and Conservative Subset Notes\n\n")

    f.write("## Purpose\n\n")
    f.write(
        "This step adds QC flags to the YOLOv8-s detection outputs without overwriting or deleting the primary detector table. "
        "It also creates conservative score-filtered subsets for optional downstream analysis.\n\n"
    )

    f.write("## Decision\n\n")
    f.write(
        "The primary detector output remains the threshold 0.25 table. "
        "A global threshold change is not adopted because low-bbox frames would lose additional pigs at higher thresholds, while lower thresholds increase false-positive risk. "
        "Warning frames are instead explicitly flagged for visual QC and optional conservative analysis.\n\n"
    )

    f.write("## Outputs\n\n")
    f.write(f"- QC flagged detections: `{qc_detections_path}`\n")
    f.write(f"- Conservative score >= 0.50 subset: `{conservative_050_path}`\n")
    f.write(f"- Conservative score >= 0.70 subset: `{conservative_070_path}`\n")
    f.write(f"- Frame QC summary: `{frame_qc_path}`\n")
    f.write(f"- Issue/score-band summary: `{issue_summary_path}`\n")
    f.write(f"- Conservative subset summary: `{conservative_summary_path}`\n\n")

    f.write("## Conservative subset summary\n\n")
    f.write(conservative_summary.to_markdown(index=False))
    f.write("\n\n")

    f.write("## Issue and score-band summary\n\n")
    f.write(issue_summary.to_markdown(index=False))
    f.write("\n\n")

    f.write("## Interpretation\n\n")
    f.write(
        "The primary table should be used when recall is important. "
        "The conservative score >= 0.50 subset can be used for cleaner crop/embedding features when precision is more important. "
        "The score >= 0.70 subset is stricter and may miss occluded pigs. "
        "Low/high bbox-count frames should remain documented as detector QC limitations rather than treated as manual annotation errors.\n"
    )

print("Saved:")
print(qc_detections_path)
print(conservative_050_path)
print(conservative_070_path)
print(frame_qc_path)
print(issue_summary_path)
print(conservative_summary_path)
print(note_path)

print()
print("=== Conservative subset summary ===")
print(conservative_summary.to_string(index=False))

print()
print("=== Issue score-band summary ===")
print(issue_summary.to_string(index=False))
