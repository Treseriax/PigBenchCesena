from pathlib import Path
from datetime import datetime
import json
import re
import pandas as pd

F = Path.home() / "PigBench" / "Full_Unibo_Behaviour_Pipeline"
O = F / "outputs/v80_final_project_completion/01_dataset_json_export"
JSON_DIR = O / "json_annotations_all_84_videos"
JSON_DIR.mkdir(parents=True, exist_ok=True)

status_path = O / "v80b_final_video_mapping_status_table.csv"
clips_path = F / "outputs/v79g8_identity_ready_moving_bbox_dataset/v79g8_identity_ready_clip_dataset.csv"

status = pd.read_csv(status_path).fillna("")
clips = pd.read_csv(clips_path).fillna("")

SCHEMA_VERSION = "unibo_pig_behaviour_annotation_v1"

def safe_name(s):
    s = str(s)
    s = re.sub(r"[^A-Za-z0-9_.-]+", "_", s)
    return s.strip("_")

def split_semicolon(x):
    x = str(x)
    if not x:
        return []
    return [p for p in x.split(";") if p]

def to_float(x):
    try:
        return float(x)
    except Exception:
        return None

def to_int_or_none(x):
    try:
        if str(x) == "":
            return None
        return int(float(x))
    except Exception:
        return None

manifest_rows = []

for _, video in status.iterrows():
    video_id = str(video["video_id"])
    vclips = clips[clips["video_id"].astype(str) == video_id].copy()

    annotations = []

    for _, c in vclips.iterrows():
        start_sec = to_float(c.get("clip_start_sec_in_video", ""))
        end_sec = to_float(c.get("clip_end_sec_in_video", ""))

        ann = {
            "clip_id": str(c.get("clip_id", "")),
            "annotation_window_id": str(c.get("annotation_window_id", "")),
            "window_id": str(c.get("window_id", "")),
            "start_sec": start_sec,
            "end_sec": end_sec,
            "duration_sec": (end_sec - start_sec) if start_sec is not None and end_sec is not None else None,
            "behaviour_label": str(c.get("behaviour_label", "")),
            "identity_colour": str(c.get("identity_colour", "")),
            "candidate_tracklets": split_semicolon(c.get("top6_candidate_tracklets", "")),
            "candidate_tracklet_count": to_int_or_none(c.get("top6_candidate_count", "")),
            "window_assignment_readiness": str(c.get("window_assignment_readiness", "")),
            "identity_dataset_status": str(c.get("identity_dataset_status", "")),
            "usable_for_identity_training": str(c.get("usable_for_identity_training", "")).lower() == "true",
            "usable_for_behaviour_clip_dataset": str(c.get("usable_for_behaviour_clip_dataset", "")).lower() == "true",
            "bbox_source": "roi_filtered_top6_candidate_tracklets",
            "claim_boundary": "candidate_tracklets_not_final_colour_to_track_assignment",
        }
        annotations.append(ann)

    has_labels = len(annotations) > 0

    payload = {
        "schema_version": SCHEMA_VERSION,
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "video": {
            "video_id": video_id,
            "video_filename": str(video["video_filename"]),
            "video_path": str(video["video_path"]),
            "video_type": str(video["video_type"]),
            "camera_identifier": str(video["camera_identifier"]),
            "pen_identifier": str(video["pen_identifier"]),
            "recording_date": str(video["recording_date"]),
            "recording_start_time": str(video["recording_start_time"]),
            "recording_end_time": str(video["recording_end_time"]),
            "recording_time_interval": str(video["recording_time_interval"]),
            "manual_overlay_text": str(video["manual_overlay_text"]),
            "manual_review_status": str(video["manual_review_status"]),
            "manual_confidence": str(video["manual_confidence"]),
            "mapping_status": str(video["mapping_status"]),
            "annotation_status": str(video["annotation_status"]),
            "tracking_subset_status": str(video["tracking_subset_status"]),
            "ground_truth_source": str(video["ground_truth_source"]),
            "inconsistency_notes": str(video["inconsistency_notes"]),
        },
        "annotation_file_type": "behaviour_annotations" if has_labels else "metadata_status_only_no_behaviour_labels",
        "annotations": annotations,
        "summary": {
            "annotation_count": len(annotations),
            "has_behaviour_annotations": has_labels,
            "matched_annotation_rows": to_int_or_none(video.get("matched_annotation_rows", "")),
            "clip_rows": to_int_or_none(video.get("clip_rows", "")),
            "identity_ready_full6_clips": to_int_or_none(video.get("identity_ready_full6_clips", "")),
            "caution_low_candidate_clips": to_int_or_none(video.get("caution_low_candidate_clips", "")),
            "behaviour_labels": split_semicolon(video.get("matched_behaviour_labels", "")),
            "identity_colours": split_semicolon(video.get("identity_colours", "")),
        },
        "claim_boundary": (
            "label_bearing_json_for_validated_matched_subset"
            if has_labels
            else "metadata_only_json_no_behaviour_completion_claim"
        ),
    }

    json_name = safe_name(video_id + "__" + str(video["video_filename"]).replace(".mp4", "")) + ".json"
    json_path = JSON_DIR / json_name

    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)

    manifest_rows.append(
        {
            "video_id": video_id,
            "video_filename": str(video["video_filename"]),
            "json_path": str(json_path),
            "annotation_file_type": payload["annotation_file_type"],
            "annotation_count": len(annotations),
            "has_behaviour_annotations": has_labels,
            "mapping_status": str(video["mapping_status"]),
            "annotation_status": str(video["annotation_status"]),
            "tracking_subset_status": str(video["tracking_subset_status"]),
            "claim_boundary": payload["claim_boundary"],
        }
    )

manifest = pd.DataFrame(manifest_rows)
manifest.to_csv(O / "v80c_json_export_manifest.csv", index=False)

print("json files", len(manifest))
print(manifest["annotation_file_type"].value_counts().to_string())
print("annotations total", int(manifest["annotation_count"].sum()))
print("saved", JSON_DIR)
