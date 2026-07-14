from pathlib import Path
import csv
import pandas as pd


PROJECT_ROOT = Path.home() / "PigBench"
W6 = PROJECT_ROOT / "Week6_Unibo_Dataset_Validation"

OUT_GT = W6 / "outputs/unified_ground_truth"
OUT_STATS = W6 / "outputs/dataset_statistics"
NOTES = W6 / "notes"

GT_V2 = OUT_GT / "week6_unified_ground_truth_v2_with_recovered_videos.csv"


def safe_to_csv(df, path):
    df.to_csv(
        path,
        index=False,
        quoting=csv.QUOTE_ALL,
        escapechar="\\",
        lineterminator="\n"
    )


gt = pd.read_csv(GT_V2)

valid_video_statuses = [
    "matched_tlc_hour_video",
    "candidate_recovered_ctoken_video",
]

still_unmatched = int((~gt["video_match_status"].isin(valid_video_statuses)).sum())

summary = pd.DataFrame([
    {"metric": "total_records", "value": len(gt)},
    {"metric": "direct_tlc_video_matched_records", "value": int((gt["video_match_status"] == "matched_tlc_hour_video").sum())},
    {"metric": "candidate_recovered_ctoken_video_records", "value": int((gt["video_match_status"] == "candidate_recovered_ctoken_video").sum())},
    {"metric": "still_unmatched_video_records", "value": still_unmatched},
    {"metric": "records_with_any_video_path", "value": int((gt["video_path"].fillna("").astype(str) != "").sum())},
    {"metric": "bbox_available_records", "value": int(gt["bbox_available"].sum()) if "bbox_available" in gt.columns else 0},
])

summary_path = OUT_GT / "week6_unified_ground_truth_v2_summary.csv"
safe_to_csv(summary, summary_path)

video_match_dist = (
    gt.groupby(["video_match_status", "video_mapping_confidence"])
    .size()
    .reset_index(name="count")
    .sort_values(["video_match_status", "video_mapping_confidence"])
)

quality_dist = (
    gt.groupby("quality_flag")
    .size()
    .reset_index(name="count")
    .sort_values("count", ascending=False)
)

note_path = NOTES / "week6_unified_ground_truth_v2_recovered_video_notes.md"

with open(note_path, "w") as f:
    f.write("# Week 6 Unified Ground Truth v2 with Recovered c-token Videos\n\n")

    f.write("## Purpose\n\n")
    f.write(
        "This version updates the unified ground-truth table by linking the previously unmatched 15:00-19:00 labels "
        "to candidate c-token videos. The mapping is based on TLC-to-c-token visual similarity and should be treated as candidate evidence until visually confirmed.\n\n"
    )

    f.write("## Summary\n\n")
    f.write(summary.to_markdown(index=False))
    f.write("\n\n")

    f.write("## Video match status distribution\n\n")
    f.write(video_match_dist.to_markdown(index=False))
    f.write("\n\n")

    f.write("## Quality flag distribution\n\n")
    f.write(quality_dist.to_markdown(index=False))
    f.write("\n\n")

    f.write("## Interpretation\n\n")
    f.write(
        "All 432 labels now have a video path. The 07:00-15:00 labels are direct TLC filename matches and have high confidence. "
        "The 15:00-19:00 labels are recovered candidate c-token matches and have medium confidence until the contact sheet is visually inspected. "
        "Bounding boxes are still not linked at this stage.\n"
    )

print("Fixed summary:")
print(summary.to_string(index=False))
print()
print("Updated note:")
print(note_path)
