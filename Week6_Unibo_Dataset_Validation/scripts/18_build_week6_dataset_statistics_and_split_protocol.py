from pathlib import Path
import csv
import pandas as pd


PROJECT_ROOT = Path.home() / "PigBench"
W6 = PROJECT_ROOT / "Week6_Unibo_Dataset_Validation"

OUT_STATS = W6 / "outputs/dataset_statistics"
OUT_GT = W6 / "outputs/unified_ground_truth"
OUT_FEAT = W6 / "outputs/feature_extractors"
OUT_VIS = W6 / "outputs/visual_label_check"
NOTES = W6 / "notes"

OUT_STATS.mkdir(parents=True, exist_ok=True)
OUT_GT.mkdir(parents=True, exist_ok=True)
NOTES.mkdir(parents=True, exist_ok=True)

VIDEO_INV = OUT_STATS / "unibo_candidate_raw_work_videos.csv"
NAMING = OUT_STATS / "unibo_raw_video_naming_pattern_analysis.csv"
GT_V2 = OUT_GT / "week6_unified_ground_truth_v2_with_recovered_videos.csv"
FRAMES = OUT_GT / "week6_scanpoint_frame_index.csv"
DETECTIONS = OUT_FEAT / "week6_yolov8s_all_scanpoint_detections.csv"
MARKERS = OUT_FEAT / "week6_crop_colour_marker_features.csv"
CANDIDATES = OUT_FEAT / "week6_candidate_bbox_to_colour_assignments.csv"


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


def pct(n, d):
    return round(float(n) / float(d) * 100, 2) if d else 0.0


video_inv = read_optional(VIDEO_INV)
naming = read_optional(NAMING)
gt = read_optional(GT_V2)
frames = read_optional(FRAMES)
dets = read_optional(DETECTIONS)
markers = read_optional(MARKERS)
candidates = read_optional(CANDIDATES)

# -----------------------------
# 1) Dataset overview metrics
# -----------------------------
metrics = []

if len(video_inv):
    metrics.extend([
        {"section": "raw_video_inventory", "metric": "raw_work_mp4_videos", "value": len(video_inv)},
        {"section": "raw_video_inventory", "metric": "raw_work_total_size_gb", "value": round(video_inv["size_mb"].sum() / 1024, 3) if "size_mb" in video_inv.columns else ""},
        {"section": "raw_video_inventory", "metric": "mean_video_duration_sec", "value": round(video_inv["duration_sec"].mean(), 2) if "duration_sec" in video_inv.columns else ""},
    ])

if len(naming):
    metrics.extend([
        {"section": "video_naming", "metric": "naming_families", "value": naming["naming_family"].nunique() if "naming_family" in naming.columns else ""},
        {"section": "video_naming", "metric": "camera_tokens", "value": naming["parsed_camera_id"].nunique() if "parsed_camera_id" in naming.columns else ""},
    ])

if len(gt):
    metrics.extend([
        {"section": "ground_truth", "metric": "unified_gt_records", "value": len(gt)},
        {"section": "ground_truth", "metric": "unique_video_slots", "value": gt["video_id"].nunique()},
        {"section": "ground_truth", "metric": "unique_colour_ids", "value": gt["colour_id"].nunique()},
        {"section": "ground_truth", "metric": "unique_behaviour_codes", "value": gt["behaviour_code"].nunique()},
        {"section": "ground_truth", "metric": "direct_tlc_records", "value": int((gt["video_match_status"] == "matched_tlc_hour_video").sum())},
        {"section": "ground_truth", "metric": "candidate_recovered_ctoken_records", "value": int((gt["video_match_status"] == "candidate_recovered_ctoken_video").sum())},
        {"section": "ground_truth", "metric": "records_with_video_path", "value": int((gt["video_path"].fillna("").astype(str) != "").sum())},
        {"section": "ground_truth", "metric": "bbox_available_in_gt", "value": int(gt["bbox_available"].sum()) if "bbox_available" in gt.columns else 0},
    ])

if len(frames):
    metrics.extend([
        {"section": "scanpoint_frames", "metric": "scanpoint_frames", "value": len(frames)},
        {"section": "scanpoint_frames", "metric": "ok_extracted_frames", "value": int((frames["extraction_status"] == "ok").sum())},
    ])

if len(dets):
    det_per_frame = dets.groupby("scan_frame_id").size()
    metrics.extend([
        {"section": "detections", "metric": "total_yolov8s_bboxes", "value": len(dets)},
        {"section": "detections", "metric": "mean_bboxes_per_frame", "value": round(det_per_frame.mean(), 3)},
        {"section": "detections", "metric": "min_bboxes_per_frame", "value": int(det_per_frame.min())},
        {"section": "detections", "metric": "max_bboxes_per_frame", "value": int(det_per_frame.max())},
        {"section": "detections", "metric": "mean_detection_score", "value": round(dets["score"].mean(), 4)},
        {"section": "detections", "metric": "median_detection_score", "value": round(dets["score"].median(), 4)},
    ])

if len(markers):
    metrics.extend([
        {"section": "marker_candidates", "metric": "analysed_bbox_crops", "value": len(markers)},
        {"section": "marker_candidates", "metric": "high_marker_candidates", "value": int((markers["marker_confidence"] == "high").sum())},
        {"section": "marker_candidates", "metric": "medium_marker_candidates", "value": int((markers["marker_confidence"] == "medium").sum())},
        {"section": "marker_candidates", "metric": "low_marker_candidates", "value": int((markers["marker_confidence"] == "low").sum())},
        {"section": "marker_candidates", "metric": "no_marker_detected", "value": int((markers["marker_confidence"] == "none").sum())},
    ])

if len(candidates):
    metrics.append({"section": "marker_candidates", "metric": "medium_or_high_candidate_assignments", "value": len(candidates)})

metrics_df = pd.DataFrame(metrics)
metrics_path = OUT_STATS / "week6_final_dataset_overview_metrics.csv"
safe_to_csv(metrics_df, metrics_path)

# -----------------------------
# 2) Distribution tables
# -----------------------------
if len(gt):
    behaviour_dist = (
        gt.groupby(["behaviour_code", "behaviour_label"])
        .size()
        .reset_index(name="count")
        .sort_values("count", ascending=False)
    )
    behaviour_dist["percentage"] = behaviour_dist["count"].apply(lambda x: pct(x, len(gt)))
else:
    behaviour_dist = pd.DataFrame()

behaviour_path = OUT_STATS / "week6_final_behaviour_distribution.csv"
safe_to_csv(behaviour_dist, behaviour_path)

if len(behaviour_dist):
    rare = behaviour_dist[(behaviour_dist["count"] < 10) | (behaviour_dist["percentage"] < 3)].copy()
else:
    rare = pd.DataFrame()

rare_path = OUT_STATS / "week6_final_rare_behaviour_classes.csv"
safe_to_csv(rare, rare_path)

if len(gt):
    colour_dist = (
        gt.groupby(["colour_id", "colour_raw"])
        .size()
        .reset_index(name="count")
        .sort_values("colour_id")
    )
else:
    colour_dist = pd.DataFrame()

colour_path = OUT_STATS / "week6_final_colour_distribution.csv"
safe_to_csv(colour_dist, colour_path)

if len(gt):
    hourly_dist = (
        gt.groupby(["hour_start", "hour_end", "video_id", "video_match_status"])
        .agg(
            label_count=("record_id", "count"),
            unique_colours=("colour_id", "nunique"),
            unique_behaviours=("behaviour_code", "nunique"),
        )
        .reset_index()
        .sort_values("hour_start")
    )
else:
    hourly_dist = pd.DataFrame()

hourly_path = OUT_STATS / "week6_final_hourly_annotation_distribution.csv"
safe_to_csv(hourly_dist, hourly_path)

if len(dets):
    detection_frame_summary = (
        dets.groupby(["scan_frame_id", "video_match_status"])
        .agg(
            bbox_count=("det_id", "count"),
            mean_score=("score", "mean"),
            max_score=("score", "max"),
            mean_bbox_area=("bbox_area", "mean"),
        )
        .reset_index()
    )
else:
    detection_frame_summary = pd.DataFrame()

detection_frame_summary_path = OUT_STATS / "week6_final_detection_frame_summary.csv"
safe_to_csv(detection_frame_summary, detection_frame_summary_path)

if len(markers):
    marker_summary = (
        markers.groupby(["best_marker_colour", "marker_confidence"])
        .agg(
            detection_count=("det_id", "count"),
            mean_marker_score=("best_marker_score", "mean"),
            mean_margin=("marker_score_margin", "mean"),
            mean_detection_score=("score", "mean"),
        )
        .reset_index()
        .sort_values(["marker_confidence", "detection_count"], ascending=[True, False])
    )
else:
    marker_summary = pd.DataFrame()

marker_summary_path = OUT_STATS / "week6_final_marker_candidate_distribution.csv"
safe_to_csv(marker_summary, marker_summary_path)

# -----------------------------
# 3) Split protocol
# -----------------------------
# Split by video/hour group to avoid frame-level leakage.
# 12 videos/hours, each has 36 labels.
# Train/val/test all include at least one direct TLC and one candidate recovered c-token hour where possible.
split_hour_map = {
    "07:00": "train",
    "08:00": "train",
    "09:00": "val",
    "10:00": "train",
    "11:00": "train",
    "12:00": "test",
    "13:00": "train",
    "14:00": "train",
    "15:00": "val",
    "16:00": "train",
    "17:00": "train",
    "18:00": "test",
}

if len(gt):
    gt_split = gt.copy()
    gt_split["split"] = gt_split["hour_start"].map(split_hour_map).fillna("unassigned")
    gt_split["split_unit"] = gt_split["video_id"]
    gt_split["split_policy"] = "video_hour_level_no_frame_leakage"

    split_gt_path = OUT_GT / "week6_unified_ground_truth_v2_with_split.csv"
    safe_to_csv(gt_split, split_gt_path)

    split_video_protocol = (
        gt_split.groupby(["split", "video_id", "hour_start", "hour_end", "video_match_status", "video_mapping_confidence"])
        .agg(
            label_count=("record_id", "count"),
            unique_colours=("colour_id", "nunique"),
            unique_behaviours=("behaviour_code", "nunique"),
        )
        .reset_index()
        .sort_values(["split", "hour_start"])
    )

    split_video_path = OUT_STATS / "week6_split_protocol_video_level.csv"
    safe_to_csv(split_video_protocol, split_video_path)

    split_summary = (
        gt_split.groupby("split")
        .agg(
            video_units=("video_id", "nunique"),
            label_count=("record_id", "count"),
            unique_colours=("colour_id", "nunique"),
            unique_behaviours=("behaviour_code", "nunique"),
            direct_tlc_records=("video_match_status", lambda s: int((s == "matched_tlc_hour_video").sum())),
            candidate_ctoken_records=("video_match_status", lambda s: int((s == "candidate_recovered_ctoken_video").sum())),
        )
        .reset_index()
    )

    split_summary["label_percentage"] = split_summary["label_count"].apply(lambda x: pct(x, len(gt_split)))
    split_summary_path = OUT_STATS / "week6_split_protocol_summary.csv"
    safe_to_csv(split_summary, split_summary_path)

    split_behaviour = (
        gt_split.groupby(["split", "behaviour_code", "behaviour_label"])
        .size()
        .reset_index(name="count")
        .sort_values(["split", "count"], ascending=[True, False])
    )

    split_behaviour_path = OUT_STATS / "week6_split_protocol_behaviour_distribution.csv"
    safe_to_csv(split_behaviour, split_behaviour_path)
else:
    gt_split = pd.DataFrame()
    split_gt_path = OUT_GT / "week6_unified_ground_truth_v2_with_split.csv"
    split_video_protocol = pd.DataFrame()
    split_video_path = OUT_STATS / "week6_split_protocol_video_level.csv"
    split_summary = pd.DataFrame()
    split_summary_path = OUT_STATS / "week6_split_protocol_summary.csv"
    split_behaviour = pd.DataFrame()
    split_behaviour_path = OUT_STATS / "week6_split_protocol_behaviour_distribution.csv"

# -----------------------------
# 4) Notes/report
# -----------------------------
note_path = NOTES / "week6_dataset_statistics_and_split_protocol.md"

with open(note_path, "w") as f:
    f.write("# Week 6 Dataset Statistics and Split Protocol\n\n")

    f.write("## Purpose\n\n")
    f.write(
        "This note consolidates Week 6 dataset statistics after raw video inventory, Excel ground-truth parsing, "
        "video linkage, scanpoint frame extraction, YOLOv8-s bbox detection, and crop-based colour-marker analysis.\n\n"
    )

    f.write("## Dataset overview metrics\n\n")
    f.write(metrics_df.to_markdown(index=False) if len(metrics_df) else "No metrics available.")
    f.write("\n\n")

    f.write("## Behaviour class distribution\n\n")
    f.write(behaviour_dist.to_markdown(index=False) if len(behaviour_dist) else "No behaviour distribution available.")
    f.write("\n\n")

    f.write("## Rare behaviour classes\n\n")
    f.write(rare.to_markdown(index=False) if len(rare) else "No rare classes under the selected threshold.")
    f.write("\n\n")

    f.write("## Colour ID distribution\n\n")
    f.write(colour_dist.to_markdown(index=False) if len(colour_dist) else "No colour distribution available.")
    f.write("\n\n")

    f.write("## Hourly annotation distribution\n\n")
    f.write(hourly_dist.to_markdown(index=False) if len(hourly_dist) else "No hourly distribution available.")
    f.write("\n\n")

    f.write("## Detection frame summary\n\n")
    f.write(detection_frame_summary.describe(include='all').to_markdown() if len(detection_frame_summary) else "No detection frame summary available.")
    f.write("\n\n")

    f.write("## Marker candidate distribution\n\n")
    f.write(marker_summary.to_markdown(index=False) if len(marker_summary) else "No marker candidate distribution available.")
    f.write("\n\n")

    f.write("## Split protocol\n\n")
    f.write(
        "The recommended split is video/hour-level, not frame-level. All labels from the same hourly video remain in the same split. "
        "This avoids leakage where frames from the same video hour appear in both training and evaluation sets.\n\n"
    )

    f.write("### Split summary\n\n")
    f.write(split_summary.to_markdown(index=False) if len(split_summary) else "No split summary available.")
    f.write("\n\n")

    f.write("### Video-level split table\n\n")
    f.write(split_video_protocol.to_markdown(index=False) if len(split_video_protocol) else "No video-level split available.")
    f.write("\n\n")

    f.write("## Important limitations\n\n")
    f.write(
        "- Current labels come from one Excel sheet, one day, one camera/pen context. This is useful for Week 6 validation and prototyping, but not enough for a final generalization benchmark.\n"
        "- The 15:00-19:00 video linkage is candidate recovered c-token mapping and should remain marked as medium confidence until visually confirmed.\n"
        "- Bounding boxes are detector outputs, not manual ground-truth boxes.\n"
        "- Crop marker features provide candidate bbox-to-colour association only. Red is ambiguous between red_neck and red_tail, and no_color cannot be recovered by marker detection.\n"
        "- Rare classes such as drinking, walking, aggressive interaction, and sitting inactive need special handling or grouping for future classification experiments.\n\n"
    )

    f.write("## Interpretation\n\n")
    f.write(
        "At this stage, the Week 6 dataset is suitable for dataset validation, visualization demos, bbox extraction, feature extractor testing, "
        "and preliminary split-protocol design. It is not yet a fully identity-resolved frame-level behaviour dataset. "
        "The next technical step is to build feature extractor comparison tables from trajectory/bbox/crop/marker/embedding-style descriptors.\n"
    )

print("Saved:")
print(metrics_path)
print(behaviour_path)
print(rare_path)
print(colour_path)
print(hourly_path)
print(detection_frame_summary_path)
print(marker_summary_path)
print(split_gt_path)
print(split_video_path)
print(split_summary_path)
print(split_behaviour_path)
print(note_path)

print()
print("=== Dataset overview metrics ===")
print(metrics_df.to_string(index=False) if len(metrics_df) else "No metrics.")

print()
print("=== Split summary ===")
print(split_summary.to_string(index=False) if len(split_summary) else "No split summary.")

print()
print("=== Rare classes ===")
print(rare.to_string(index=False) if len(rare) else "No rare classes.")
