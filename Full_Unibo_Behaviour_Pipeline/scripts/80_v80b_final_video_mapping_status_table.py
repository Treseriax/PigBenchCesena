from pathlib import Path
from datetime import datetime, timedelta
import re
import pandas as pd

F = Path.home() / "PigBench" / "Full_Unibo_Behaviour_Pipeline"
O = F / "outputs/v80_final_project_completion/01_dataset_json_export"
O.mkdir(parents=True, exist_ok=True)

mapping_path = F / "outputs/v78k8_full_video_manual_overlay_mapping/Full_Unibo_Full_Video_Manual_Overlay_Mapping/v78k8_FULL_VIDEO_MAPPING_FINAL.csv"
ready_path = F / "outputs/v78k9c_annotation_video_join_corrected/v78k9c_ready_annotation_video_rows.csv"
clips_path = F / "outputs/v79g8_identity_ready_moving_bbox_dataset/v79g8_identity_ready_clip_dataset.csv"
plan_path = F / "outputs/v79b_tracking_manifest_audit_and_run_plan/v79b_tracking_run_plan_36_videos.csv"

m = pd.read_csv(mapping_path).fillna("")
ready = pd.read_csv(ready_path).fillna("")
clips = pd.read_csv(clips_path).fillna("")
plan = pd.read_csv(plan_path).fillna("")

def end_plus_one_hour(x):
    try:
        t = datetime.strptime(str(x), "%H:%M:%S")
        return (t + timedelta(hours=1)).strftime("%H:%M:%S")
    except Exception:
        return ""

def friendly_interval(filename):
    s = str(filename)
    match = re.search(r"(\d{3,4})\s*-\s*(\d{3,4})", s)
    if not match:
        return "", ""

    def fmt(z):
        z = str(z).zfill(4)
        return z[:2] + ":" + z[2:] + ":00"

    return fmt(match.group(1)), fmt(match.group(2))

readyg = (
    ready.groupby("matched_video_id", dropna=False)
    .agg(
        matched_annotation_rows=("annotation_window_id", "count"),
        matched_behaviour_labels=("behaviour_label", lambda x: ";".join(sorted(set(map(str, x))))),
        gt_dates=("date", lambda x: ";".join(sorted(set(map(str, x))))),
        gt_sources=("sheet_name", lambda x: ";".join(sorted(set(map(str, x))))),
    )
    .reset_index()
    .rename(columns={"matched_video_id": "video_id"})
)

clipg = (
    clips.groupby("video_id", dropna=False)
    .agg(
        clip_rows=("clip_id", "count"),
        identity_ready_full6_clips=(
            "identity_dataset_status",
            lambda x: int((x == "IDENTITY_READY_FULL_6_CANDIDATES").sum()),
        ),
        caution_low_candidate_clips=(
            "identity_dataset_status",
            lambda x: int((x != "IDENTITY_READY_FULL_6_CANDIDATES").sum()),
        ),
        clip_behaviour_labels=("behaviour_label", lambda x: ";".join(sorted(set(map(str, x))))),
        identity_colours=("identity_colour", lambda x: ";".join(sorted(set(map(str, x))))),
    )
    .reset_index()
)

pl = plan[
    ["video_id", "date", "tlc_camera", "room_pen", "clip_count", "priority"]
].rename(
    columns={
        "date": "plan_date",
        "tlc_camera": "plan_tlc_camera",
        "room_pen": "plan_room_pen",
        "clip_count": "tracking_plan_clip_count",
        "priority": "tracking_priority",
    }
)

out = (
    m.merge(readyg, on="video_id", how="left")
    .merge(clipg, on="video_id", how="left")
    .merge(pl, on="video_id", how="left")
    .fillna("")
)

starts = []
ends = []
dates = []
intervals = []
notes = []

for _, r in out.iterrows():
    filename_start = str(r.get("filename_start_time", ""))
    friendly_start, friendly_end = friendly_interval(r.get("video_filename", ""))

    start_time = filename_start if filename_start else friendly_start
    end_time = end_plus_one_hour(filename_start) if filename_start else friendly_end

    recording_date = str(r.get("filename_date", ""))
    if not recording_date:
        recording_date = str(r.get("plan_date", ""))

    starts.append(start_time)
    ends.append(end_time)
    dates.append(recording_date)
    intervals.append((start_time + "-" + end_time) if start_time and end_time else "")

    row_notes = []

    if str(r.get("manual_review_status", "")) != "RESOLVED":
        row_notes.append("manual_mapping_not_resolved")

    if not str(r.get("manual_tlc_camera", "")):
        row_notes.append("missing_manual_tlc_camera")

    if not str(r.get("manual_room_pen", "")):
        row_notes.append("missing_manual_room_pen")

    if not recording_date:
        row_notes.append("missing_recording_date")

    if not start_time:
        row_notes.append("missing_recording_time_interval")

    if str(r.get("clip_rows", "")) == "":
        row_notes.append("no_matched_behaviour_clips_in_current_validated_subset")

    notes.append(";".join(row_notes) if row_notes else "ok")

out["recording_date"] = dates
out["recording_start_time"] = starts
out["recording_end_time"] = ends
out["recording_time_interval"] = intervals
out["camera_identifier"] = out["manual_tlc_camera"]
out["pen_identifier"] = out["manual_room_pen"]

out["ground_truth_source"] = out.apply(
    lambda r: r["gt_sources"]
    if str(r["gt_sources"])
    else "no_matched_gt_source_in_current_validated_subset",
    axis=1,
)

out["mapping_status"] = out.apply(
    lambda r: "MAPPING_RESOLVED"
    if str(r["manual_review_status"]) == "RESOLVED"
    and str(r["manual_tlc_camera"])
    and str(r["manual_room_pen"])
    else "MAPPING_NEEDS_REVIEW",
    axis=1,
)

out["annotation_status"] = out.apply(
    lambda r: "MATCHED_BEHAVIOUR_CLIPS_AVAILABLE"
    if str(r["clip_rows"])
    else "MAPPED_BUT_NO_MATCHED_BEHAVIOUR_CLIPS",
    axis=1,
)

out["tracking_subset_status"] = out.apply(
    lambda r: "IN_36_VIDEO_TRACKING_SUBSET"
    if str(r.get("tracking_plan_clip_count", ""))
    else "NOT_IN_36_VIDEO_TRACKING_SUBSET",
    axis=1,
)

out["inconsistency_notes"] = notes

cols = [
    "video_id",
    "video_index",
    "video_filename",
    "video_path",
    "video_type",
    "filename_c_code",
    "recording_date",
    "recording_start_time",
    "recording_end_time",
    "recording_time_interval",
    "camera_identifier",
    "pen_identifier",
    "manual_overlay_text",
    "manual_review_status",
    "manual_confidence",
    "mapping_status",
    "ground_truth_source",
    "annotation_status",
    "matched_annotation_rows",
    "clip_rows",
    "identity_ready_full6_clips",
    "caution_low_candidate_clips",
    "matched_behaviour_labels",
    "identity_colours",
    "tracking_subset_status",
    "tracking_plan_clip_count",
    "tracking_priority",
    "inconsistency_notes",
]

output_path = O / "v80b_final_video_mapping_status_table.csv"
out[cols].to_csv(output_path, index=False)

print("rows", len(out))
print("=== mapping_status ===")
print(out["mapping_status"].value_counts().to_string())
print("=== annotation_status ===")
print(out["annotation_status"].value_counts().to_string())
print("=== tracking_subset_status ===")
print(out["tracking_subset_status"].value_counts().to_string())
print("saved", output_path)
