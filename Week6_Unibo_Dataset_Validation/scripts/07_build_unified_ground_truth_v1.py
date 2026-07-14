from pathlib import Path
import json
import csv
import pandas as pd


PROJECT_ROOT = Path.home() / "PigBench"
W6 = PROJECT_ROOT / "Week6_Unibo_Dataset_Validation"
OUT_GT = W6 / "outputs/unified_ground_truth"
OUT_STATS = W6 / "outputs/dataset_statistics"
NOTES = W6 / "notes"

OUT_GT.mkdir(parents=True, exist_ok=True)
OUT_STATS.mkdir(parents=True, exist_ok=True)
NOTES.mkdir(parents=True, exist_ok=True)

EXCEL_LONG = OUT_GT / "work_unibo_excel_annotations_long.csv"


def safe_to_csv(df, path):
    df.to_csv(
        path,
        index=False,
        quoting=csv.QUOTE_ALL,
        escapechar="\\",
        lineterminator="\n"
    )


if not EXCEL_LONG.exists():
    raise FileNotFoundError(EXCEL_LONG)

src = pd.read_csv(EXCEL_LONG)

# Build unified table matching Week 6 task sheet fields.
unified = pd.DataFrame()

unified["dataset_name"] = "Unibo"
unified["record_id"] = ["unibo_gt_%06d" % i for i in range(len(src))]

unified["video_id"] = src["matched_video_filename"].fillna("").astype(str).str.replace(".mp4", "", regex=False)
unified.loc[unified["video_id"] == "", "video_id"] = (
    src["camera_id"].astype(str)
    + "_"
    + src["pen_id"].astype(str)
    + "_"
    + src["hour_start"].astype(str).str.replace(":", "", regex=False)
    + "_"
    + src["hour_end"].astype(str).str.replace(":", "", regex=False)
    + "_unmatched_video"
)

unified["video_path"] = src["matched_video_path"].fillna("")
unified["video_match_status"] = src["matched_video_status"].fillna("")

unified["crate_id"] = src["pen_id"].fillna("")
unified["pen_id"] = src["pen_id"].fillna("")
unified["camera_id"] = src["camera_id"].fillna("")

unified["date"] = src["date"].fillna("")
unified["hour_start"] = src["hour_start"].fillna("")
unified["hour_end"] = src["hour_end"].fillna("")
unified["timestamp"] = src["observation_time"].fillna("")
unified["timestamp_end_nominal"] = src["nominal_next_observation_time"].fillna("")

unified["frame_index"] = src["frame_index_estimate"].fillna("")
unified["timestamp_sec_in_video"] = src["observation_minute_in_hour"].fillna("") * 60

unified["pig_id"] = src["pig_id"].fillna("")
unified["colour_id"] = src["colour_id"].fillna("")
unified["colour_raw"] = src["colour_raw"].fillna("")

unified["bbox_x1"] = ""
unified["bbox_y1"] = ""
unified["bbox_x2"] = ""
unified["bbox_y2"] = ""
unified["bbox_xyxy"] = ""
unified["bbox_available"] = False
unified["bbox_source"] = ""

unified["behaviour_code"] = src["behaviour_code"].fillna("")
unified["behaviour_label"] = src["behaviour_label"].fillna("")
unified["label_source"] = src["label_source"].fillna("manual_excel_scan_sampling")
unified["annotation_granularity"] = "10_minute_scan_sampling"
unified["annotation_interval_note"] = "One manual ethological scan-sampling label at a 10-minute observation point; not continuous frame-level behaviour."

unified["source_file"] = src["source_file"].fillna("")
unified["source_sheet"] = src["sheet"].fillna("")
unified["source_excel_cell"] = src["excel_cell"].fillna("")
unified["quality_flag"] = "label_only_no_bbox"
unified.loc[unified["video_match_status"] == "matched_tlc_hour_video", "quality_flag"] = "label_with_video_match_no_bbox"
unified.loc[unified["video_match_status"] != "matched_tlc_hour_video", "quality_flag"] = "label_without_video_match_no_bbox"

unified["notes"] = src["notes"].fillna("")

unified_csv = OUT_GT / "week6_unified_ground_truth_v1.csv"
unified_json = OUT_GT / "week6_unified_ground_truth_v1.json"

safe_to_csv(unified, unified_csv)
unified_json.write_text(json.dumps(unified.to_dict(orient="records"), indent=2, ensure_ascii=False))

# Summary.
summary_rows = [
    {"metric": "total_unified_records", "value": len(unified)},
    {"metric": "unique_videos_or_video_slots", "value": unified["video_id"].nunique()},
    {"metric": "records_with_matched_video", "value": int((unified["video_match_status"] == "matched_tlc_hour_video").sum())},
    {"metric": "records_without_matched_video", "value": int((unified["video_match_status"] != "matched_tlc_hour_video").sum())},
    {"metric": "bbox_available_records", "value": int(unified["bbox_available"].sum())},
    {"metric": "unique_pigs_colours", "value": unified["colour_id"].nunique()},
    {"metric": "unique_behaviour_codes", "value": unified["behaviour_code"].nunique()},
    {"metric": "annotation_granularity", "value": "10_minute_scan_sampling"},
]

summary = pd.DataFrame(summary_rows)
summary_path = OUT_GT / "week6_unified_ground_truth_v1_summary.csv"
safe_to_csv(summary, summary_path)

behaviour_dist = (
    unified.groupby(["behaviour_code", "behaviour_label"])
    .size()
    .reset_index(name="count")
    .sort_values("count", ascending=False)
)
behaviour_dist["percentage"] = (behaviour_dist["count"] / len(unified) * 100).round(2)

# Rare class threshold: < 10 records or < 3%.
rare = behaviour_dist[(behaviour_dist["count"] < 10) | (behaviour_dist["percentage"] < 3)].copy()

behaviour_dist_path = OUT_STATS / "week6_unified_behaviour_class_distribution.csv"
rare_path = OUT_STATS / "week6_unified_rare_behaviour_classes.csv"

safe_to_csv(behaviour_dist, behaviour_dist_path)
safe_to_csv(rare, rare_path)

colour_dist = (
    unified.groupby(["colour_id", "colour_raw"])
    .size()
    .reset_index(name="count")
    .sort_values("colour_id")
)

colour_dist_path = OUT_STATS / "week6_unified_colour_distribution.csv"
safe_to_csv(colour_dist, colour_dist_path)

video_dist = (
    unified.groupby(["video_id", "video_match_status", "video_path"])
    .size()
    .reset_index(name="label_count")
    .sort_values(["video_match_status", "video_id"])
)

video_dist_path = OUT_STATS / "week6_unified_video_label_distribution.csv"
safe_to_csv(video_dist, video_dist_path)

missing_ambiguous = pd.DataFrame([
    {
        "issue_type": "missing_bbox",
        "count": int((unified["bbox_available"] == False).sum()),
        "interpretation": "Manual Excel annotations provide behaviour labels but not bounding boxes; bbox must be linked later from detector/tracker outputs."
    },
    {
        "issue_type": "unmatched_video",
        "count": int((unified["video_match_status"] != "matched_tlc_hour_video").sum()),
        "interpretation": "Labels from 15:00-19:00 currently have no matching TLC hourly video filename; candidate c-token videos must be inspected before linking."
    },
    {
        "issue_type": "rare_classes",
        "count": len(rare),
        "interpretation": "Some behaviour classes have very few labels and may be unsuitable for train/test split without grouping or special handling."
    },
])

missing_path = OUT_STATS / "week6_missing_ambiguous_label_report.csv"
safe_to_csv(missing_ambiguous, missing_path)

note_path = NOTES / "week6_unified_ground_truth_v1_notes.md"

with open(note_path, "w") as f:
    f.write("# Week 6 Unified Ground Truth v1 Notes\n\n")

    f.write("## Purpose\n\n")
    f.write(
        "This table converts the parsed Unibo Excel scan-sampling annotations into the unified Week 6 ground-truth format. "
        "It is label-only at this stage: behaviour labels are available, but bounding boxes are not yet linked.\n\n"
    )

    f.write("## Summary\n\n")
    f.write(summary.to_markdown(index=False))
    f.write("\n\n")

    f.write("## Behaviour class distribution\n\n")
    f.write(behaviour_dist.to_markdown(index=False))
    f.write("\n\n")

    f.write("## Rare behaviour classes\n\n")
    f.write(rare.to_markdown(index=False) if len(rare) else "No rare classes under the selected threshold.")
    f.write("\n\n")

    f.write("## Colour distribution\n\n")
    f.write(colour_dist.to_markdown(index=False))
    f.write("\n\n")

    f.write("## Missing / ambiguous label report\n\n")
    f.write(missing_ambiguous.to_markdown(index=False))
    f.write("\n\n")

    f.write("## Interpretation\n\n")
    f.write(
        "The unified table is suitable as a first machine-learning label source, but it should be interpreted as scan-sampling labels rather than dense frame-level labels. "
        "For visual verification and feature extraction, matched TLC video records can be used immediately. "
        "Unmatched afternoon labels should only be linked after confirming which raw videos correspond to those hours.\n"
    )

print("Saved:")
print(unified_csv)
print(unified_json)
print(summary_path)
print(behaviour_dist_path)
print(rare_path)
print(colour_dist_path)
print(video_dist_path)
print(missing_path)
print(note_path)

print()
print("=== Unified GT summary ===")
print(summary.to_string(index=False))

print()
print("=== Behaviour distribution ===")
print(behaviour_dist.to_string(index=False))

print()
print("=== Rare classes ===")
print(rare.to_string(index=False))

print()
print("=== Missing / ambiguous report ===")
print(missing_ambiguous.to_string(index=False))

print()
print("=== First 20 unified rows ===")
cols = [
    "record_id",
    "video_id",
    "timestamp",
    "frame_index",
    "pig_id",
    "colour_id",
    "behaviour_code",
    "behaviour_label",
    "video_match_status",
    "quality_flag",
]
print(unified[cols].head(20).to_string(index=False))
