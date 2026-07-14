from pathlib import Path
import json
import pandas as pd


TRACKS_CSV = Path(
    "Week3_Behaviour_Dataset/outputs/scan_window_aligned/tracking_with_time/"
    "all_scan_windows_bytetrack_tracks_with_excel_window.csv"
)

LABELS_CSV = Path(
    "Week3_Behaviour_Dataset/outputs/scan_window_aligned/behaviour_labels/"
    "scan_window_behaviour_labels.csv"
)

MAPPING_CSV = Path(
    "Week3_Behaviour_Dataset/outputs/scan_window_aligned/identity_mapping/"
    "dominant_track_id_to_colour_mapping_table.csv"
)

OUT_DIR = Path("Week3_Behaviour_Dataset/outputs/scan_window_aligned/json")
OUT_DIR.mkdir(parents=True, exist_ok=True)

tracks = pd.read_csv(TRACKS_CSV)
labels = pd.read_csv(LABELS_CSV)

# Mapping table may still be unfilled. We support both filled and unfilled cases.
mapping = pd.read_csv(MAPPING_CSV)

# Normalize empty cells
for col in [
    "assigned_colour",
    "assigned_pig_id",
    "assigned_behaviour_code",
    "assigned_behaviour_label",
    "identity_confidence",
    "use_in_final_json",
    "notes",
]:
    if col in mapping.columns:
        mapping[col] = mapping[col].fillna("").astype(str)

# Fast lookup: segment -> list of available Excel labels
segment_label_lookup = {}

for seg, seg_labels in labels.groupby("segment_id"):
    segment_label_lookup[seg] = []
    for _, r in seg_labels.iterrows():
        segment_label_lookup[seg].append({
            "pig_id": r["pig_id"],
            "colour": r["colour"],
            "behaviour_code": r["behaviour_code"],
            "behaviour_label": r["behaviour_label"],
            "source": "excel_scan_sampling_ground_truth"
        })

# Fast lookup: segment + track_id -> mapping row
mapping_lookup = {}

for _, r in mapping.iterrows():
    seg = str(r["segment_id"])
    tid = int(r["track_id"])

    assigned_colour = str(r.get("assigned_colour", "")).strip()
    assigned_pig_id = str(r.get("assigned_pig_id", "")).strip()
    assigned_code = str(r.get("assigned_behaviour_code", "")).strip()
    assigned_label = str(r.get("assigned_behaviour_label", "")).strip()
    confidence = str(r.get("identity_confidence", "")).strip()
    use_flag = str(r.get("use_in_final_json", "")).strip().lower()
    notes = str(r.get("notes", "")).strip()

    mapping_lookup[(seg, tid)] = {
        "assigned_colour": assigned_colour,
        "assigned_pig_id": assigned_pig_id,
        "assigned_behaviour_code": assigned_code,
        "assigned_behaviour_label": assigned_label,
        "identity_confidence": confidence,
        "use_in_final_json": use_flag,
        "notes": notes
    }

# Build JSON grouped by segment and frame
dataset = {
    "dataset_name": "Week3_corrected_scan_window_behaviour_annotations",
    "source_video": "Unibo_scan_windows_clean.mp4",
    "method_note": (
        "The source video is a manually concatenated scan-window screen recording. "
        "Therefore, video elapsed time is not treated as continuous camera time. "
        "Each segment is manually mapped to its corresponding Excel scan-sampling interval."
    ),
    "annotation_policy": {
        "identity_verified_rows": (
            "If assigned_colour and assigned_pig_id are filled in the mapping table, "
            "the corresponding Excel behaviour label is attached to the track."
        ),
        "identity_unverified_rows": (
            "If identity mapping is not verified, the track keeps identity_status='unverified' "
            "and receives the segment-level behaviour candidate list instead of a forced label."
        )
    },
    "segments": []
}

for seg, seg_tracks in tracks.groupby("segment_id"):
    seg = str(seg)
    seg_tracks = seg_tracks.sort_values(["frame", "track_id"])

    first_row = seg_tracks.iloc[0]

    segment_obj = {
        "segment_id": seg,
        "excel_interval_start": str(first_row["excel_interval_start"]),
        "excel_interval_end": str(first_row["excel_interval_end"]),
        "source_video_segment_start_sec": float(first_row["segment_start_sec"]),
        "source_video_segment_end_sec": float(first_row["segment_end_sec"]),
        "available_excel_behaviour_labels": segment_label_lookup.get(seg, []),
        "frames": []
    }

    for frame_idx, frame_df in seg_tracks.groupby("frame"):
        frame_obj = {
            "frame": int(frame_idx),
            "segment_time_sec": float(frame_df["segment_time_sec"].iloc[0]),
            "video_time_sec": float(frame_df["video_time_sec"].iloc[0]),
            "excel_interval_start": str(frame_df["excel_interval_start"].iloc[0]),
            "excel_interval_end": str(frame_df["excel_interval_end"].iloc[0]),
            "tracks": []
        }

        for _, r in frame_df.iterrows():
            tid = int(r["track_id"])
            m = mapping_lookup.get((seg, tid), None)

            identity_status = "unverified"
            assigned_colour = None
            assigned_pig_id = None
            behaviour = {
                "behaviour_code": None,
                "behaviour_label": "identity_not_verified",
                "source": "not_assigned_without_verified_track_to_colour_mapping",
                "confidence": "not_applicable"
            }

            if m:
                use_flag = m.get("use_in_final_json", "")
                assigned_colour_candidate = m.get("assigned_colour", "")
                assigned_pig_candidate = m.get("assigned_pig_id", "")
                assigned_code = m.get("assigned_behaviour_code", "")
                assigned_label = m.get("assigned_behaviour_label", "")

                if assigned_colour_candidate and assigned_pig_candidate and assigned_label and use_flag in ["yes", "y", "true", "1"]:
                    identity_status = "manually_verified"
                    assigned_colour = assigned_colour_candidate
                    assigned_pig_id = assigned_pig_candidate
                    behaviour = {
                        "behaviour_code": assigned_code,
                        "behaviour_label": assigned_label,
                        "source": "excel_scan_sampling_ground_truth_after_manual_identity_mapping",
                        "confidence": m.get("identity_confidence", "manual")
                    }

            track_obj = {
                "track_id": tid,
                "bbox_xyxy": [
                    float(r["x1"]),
                    float(r["y1"]),
                    float(r["x2"]),
                    float(r["y2"]),
                ],
                "bbox_xywh": [
                    float(r["x1"]),
                    float(r["y1"]),
                    float(r["x2"] - r["x1"]),
                    float(r["y2"] - r["y1"]),
                ],
                "centroid": [
                    float(r["cx"]),
                    float(r["cy"]),
                ],
                "score": float(r["score"]),
                "identity_status": identity_status,
                "assigned_colour": assigned_colour,
                "assigned_pig_id": assigned_pig_id,
                "behaviour": behaviour,
                "available_segment_behaviour_labels": segment_label_lookup.get(seg, [])
            }

            frame_obj["tracks"].append(track_obj)

        segment_obj["frames"].append(frame_obj)

    dataset["segments"].append(segment_obj)

out_json = OUT_DIR / "corrected_scan_window_behaviour_annotations.json"

with open(out_json, "w") as f:
    json.dump(dataset, f, indent=2)

# Also create a compact summary
summary_rows = []

for seg_obj in dataset["segments"]:
    num_frames = len(seg_obj["frames"])
    num_tracks = sum(len(fr["tracks"]) for fr in seg_obj["frames"])
    verified = sum(
        1
        for fr in seg_obj["frames"]
        for tr in fr["tracks"]
        if tr["identity_status"] == "manually_verified"
    )
    unverified = num_tracks - verified

    summary_rows.append({
        "segment_id": seg_obj["segment_id"],
        "excel_interval_start": seg_obj["excel_interval_start"],
        "excel_interval_end": seg_obj["excel_interval_end"],
        "frames": num_frames,
        "track_instances": num_tracks,
        "verified_track_instances": verified,
        "unverified_track_instances": unverified
    })

summary_df = pd.DataFrame(summary_rows)
summary_csv = OUT_DIR / "corrected_scan_window_json_summary.csv"
summary_df.to_csv(summary_csv, index=False)

print("Saved JSON:", out_json)
print("Saved summary:", summary_csv)
print()
print(summary_df.to_string(index=False))
