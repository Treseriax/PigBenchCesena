from pathlib import Path
import csv
import json
import pandas as pd


PROJECT_ROOT = Path.home() / "PigBench"
W6 = PROJECT_ROOT / "Week6_Unibo_Dataset_Validation"

OUT_GT = W6 / "outputs/unified_ground_truth"
OUT_STATS = W6 / "outputs/dataset_statistics"
NOTES = W6 / "notes"

OUT_GT.mkdir(parents=True, exist_ok=True)
OUT_STATS.mkdir(parents=True, exist_ok=True)
NOTES.mkdir(parents=True, exist_ok=True)

GT_V1 = OUT_GT / "week6_unified_ground_truth_v1.csv"
RECOVERED = OUT_STATS / "tlc_unmatched_hours_recovered_ctoken_candidates.csv"
VIDEO_NAMING = OUT_STATS / "unibo_raw_video_naming_pattern_analysis.csv"


def safe_to_csv(df, path):
    df.to_csv(
        path,
        index=False,
        quoting=csv.QUOTE_ALL,
        escapechar="\\",
        lineterminator="\n"
    )


def normalize_hour_to_hms(x):
    s = str(x).strip()
    if not s or s.lower() == "nan":
        return ""
    parts = s.split(":")
    if len(parts) == 2:
        return f"{int(parts[0]):02d}:{int(parts[1]):02d}:00"
    if len(parts) == 3:
        return f"{int(parts[0]):02d}:{int(parts[1]):02d}:{int(parts[2]):02d}"
    return s


def strip_mp4(filename):
    s = str(filename)
    return s[:-4] if s.lower().endswith(".mp4") else s


if not GT_V1.exists():
    raise FileNotFoundError(GT_V1)

if not RECOVERED.exists():
    raise FileNotFoundError(RECOVERED)

gt = pd.read_csv(GT_V1)
rec = pd.read_csv(RECOVERED)

if VIDEO_NAMING.exists():
    vids = pd.read_csv(VIDEO_NAMING)
else:
    vids = pd.DataFrame()

# Build recovered mapping by hour.
rec_map = {}

for _, r in rec.iterrows():
    hour = normalize_hour_to_hms(r.get("target_hour_start", ""))
    path = str(r.get("recommended_video_path", "")).strip()
    filename = str(r.get("recommended_video_filename", "")).strip()
    camera = str(r.get("recommended_ctoken_camera", "")).strip()
    status = str(r.get("status", "")).strip()

    if path and status == "candidate_recovered_video":
        rec_map[hour] = {
            "path": path,
            "filename": filename,
            "camera": camera,
            "status": status,
        }

# Add fields if missing.
for col in [
    "video_mapping_source",
    "video_mapping_confidence",
    "video_mapping_note",
    "recovered_ctoken_camera",
]:
    if col not in gt.columns:
        gt[col] = ""

# Existing matched TLC rows.
tlc_mask = gt["video_match_status"].astype(str).eq("matched_tlc_hour_video")
gt.loc[tlc_mask, "video_mapping_source"] = "direct_tlc_filename_match"
gt.loc[tlc_mask, "video_mapping_confidence"] = "high"
gt.loc[tlc_mask, "video_mapping_note"] = "Video matched directly from TLC1 B1 hourly filename."

# Update unmatched rows if recovered mapping exists.
updated_count = 0
still_unmatched_count = 0

for idx, row in gt.iterrows():
    status = str(row.get("video_match_status", ""))
    if status == "matched_tlc_hour_video":
        continue

    hour_hms = normalize_hour_to_hms(row.get("hour_start", ""))

    if hour_hms in rec_map:
        info = rec_map[hour_hms]

        gt.at[idx, "video_id"] = strip_mp4(info["filename"])
        gt.at[idx, "video_path"] = info["path"]
        gt.at[idx, "video_match_status"] = "candidate_recovered_ctoken_video"
        gt.at[idx, "recovered_ctoken_camera"] = info["camera"]
        gt.at[idx, "video_mapping_source"] = "tlc_to_ctoken_similarity_candidate"
        gt.at[idx, "video_mapping_confidence"] = "medium_needs_visual_confirmation"
        gt.at[idx, "video_mapping_note"] = (
            "Originally unmatched TLC label. Linked to c-token video using camera-level TLC/c-token similarity; "
            "contact sheet should be visually inspected before treating this as confirmed."
        )
        gt.at[idx, "quality_flag"] = "label_with_candidate_recovered_video_no_bbox"

        # Estimate frame index if empty.
        timestamp_sec = row.get("timestamp_sec_in_video", "")
        current_frame = row.get("frame_index", "")

        if pd.isna(current_frame) or str(current_frame).strip() == "":
            try:
                ts = float(timestamp_sec)
                fps = 25.0

                if not vids.empty:
                    vrow = vids[vids["absolute_path"].astype(str) == info["path"]]
                    if len(vrow):
                        fps = float(vrow.iloc[0]["fps"])

                gt.at[idx, "frame_index"] = int(round(ts * fps))
            except Exception:
                pass

        updated_count += 1

    else:
        gt.at[idx, "video_mapping_source"] = "unmatched_after_recovery_attempt"
        gt.at[idx, "video_mapping_confidence"] = "none"
        gt.at[idx, "video_mapping_note"] = "No recovered c-token video candidate found for this hour."
        still_unmatched_count += 1

# Recompute summary.
v2_csv = OUT_GT / "week6_unified_ground_truth_v2_with_recovered_videos.csv"
v2_json = OUT_GT / "week6_unified_ground_truth_v2_with_recovered_videos.json"

safe_to_csv(gt, v2_csv)
v2_json.write_text(json.dumps(gt.to_dict(orient="records"), indent=2, ensure_ascii=False))

summary = pd.DataFrame([
    {"metric": "total_records", "value": len(gt)},
    {"metric": "direct_tlc_video_matched_records", "value": int((gt["video_match_status"] == "matched_tlc_hour_video").sum())},
    {"metric": "candidate_recovered_ctoken_video_records", "value": int((gt["video_match_status"] == "candidate_recovered_ctoken_video").sum())},
    {"metric": "still_unmatched_video_records", "value": int(~gt["video_match_status"].isin(["matched_tlc_hour_video", "candidate_recovered_ctoken_video"]).sum())},
    {"metric": "records_with_any_video_path", "value": int((gt["video_path"].fillna("").astype(str) != "").sum())},
    {"metric": "bbox_available_records", "value": int(gt["bbox_available"].sum()) if "bbox_available" in gt.columns else 0},
    {"metric": "updated_in_this_step", "value": updated_count},
    {"metric": "still_unmatched_in_this_step", "value": still_unmatched_count},
])

summary_path = OUT_GT / "week6_unified_ground_truth_v2_summary.csv"
safe_to_csv(summary, summary_path)

video_match_dist = (
    gt.groupby(["video_match_status", "video_mapping_confidence"])
    .size()
    .reset_index(name="count")
    .sort_values(["video_match_status", "video_mapping_confidence"])
)

video_match_dist_path = OUT_STATS / "week6_v2_video_match_status_distribution.csv"
safe_to_csv(video_match_dist, video_match_dist_path)

video_label_dist = (
    gt.groupby(["video_id", "video_match_status", "video_path"])
    .size()
    .reset_index(name="label_count")
    .sort_values(["video_match_status", "video_id"])
)

video_label_dist_path = OUT_STATS / "week6_v2_video_label_distribution.csv"
safe_to_csv(video_label_dist, video_label_dist_path)

quality_dist = (
    gt.groupby("quality_flag")
    .size()
    .reset_index(name="count")
    .sort_values("count", ascending=False)
)

quality_dist_path = OUT_STATS / "week6_v2_quality_flag_distribution.csv"
safe_to_csv(quality_dist, quality_dist_path)

behaviour_by_video_status = (
    gt.groupby(["video_match_status", "behaviour_code", "behaviour_label"])
    .size()
    .reset_index(name="count")
    .sort_values(["video_match_status", "count"], ascending=[True, False])
)

behaviour_by_video_status_path = OUT_STATS / "week6_v2_behaviour_by_video_match_status.csv"
safe_to_csv(behaviour_by_video_status, behaviour_by_video_status_path)

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

    f.write("## Video label distribution\n\n")
    f.write(video_label_dist.to_markdown(index=False))
    f.write("\n\n")

    f.write("## Interpretation\n\n")
    f.write(
        "All 432 labels now have a video path, but the confidence differs. "
        "The 07:00-15:00 labels are direct TLC filename matches and have high confidence. "
        "The 15:00-19:00 labels are recovered candidate c-token matches and have medium confidence until the contact sheet is visually inspected. "
        "Bounding boxes are still not linked at this stage.\n"
    )

print("Saved:")
print(v2_csv)
print(v2_json)
print(summary_path)
print(video_match_dist_path)
print(video_label_dist_path)
print(quality_dist_path)
print(behaviour_by_video_status_path)
print(note_path)

print()
print("=== V2 summary ===")
print(summary.to_string(index=False))

print()
print("=== Video match status distribution ===")
print(video_match_dist.to_string(index=False))

print()
print("=== Quality flag distribution ===")
print(quality_dist.to_string(index=False))

print()
print("=== First 20 recovered candidate rows ===")
cols = [
    "record_id",
    "video_id",
    "video_path",
    "timestamp",
    "frame_index",
    "pig_id",
    "behaviour_code",
    "video_match_status",
    "video_mapping_confidence",
    "quality_flag",
]
rec_rows = gt[gt["video_match_status"] == "candidate_recovered_ctoken_video"]
print(rec_rows[cols].head(20).to_string(index=False) if len(rec_rows) else "No recovered rows.")
