import json
from pathlib import Path
import pandas as pd


TRACKS_CSV = Path("Week3_Behaviour_Dataset/outputs/tracking/UniboVid2_sample_bytetrack/UniboVid2_sample_bytetrack_tracks_with_time.csv")
OUT_JSON = Path("Week3_Behaviour_Dataset/outputs/json/UniboVid2_sample_behaviour_annotations.json")
OUT_SCHEMA = Path("Week3_Behaviour_Dataset/outputs/json/annotation_schema_description.json")

OUT_JSON.parent.mkdir(parents=True, exist_ok=True)

df = pd.read_csv(TRACKS_CSV)

video_info = {
    "video_id": "UniboVid2_sample",
    "source_video": "UniboVid2.mp4",
    "sequence_dir": "Week3_Behaviour_Dataset/data/sequences/UniboVid2_sample",
    "tracking_csv": str(TRACKS_CSV),
    "fps": 25.091418,
    "estimated_start_time": "09:04:38",
    "estimated_end_time": "09:04:49.916",
    "timestamp_source": "estimated_from_camera_overlay_and_fps",
    "annotation_status": "prototype_no_exact_excel_overlap"
}

protocol = {
    "behaviour_source": "Excel scan-sampling sheet",
    "scan_sampling_description": "Behaviour observations are recorded during 10-second intervals every 10 minutes.",
    "available_excel_windows_example": [
        "09:00:00-09:00:10",
        "09:10:00-09:10:10",
        "09:20:00-09:20:10",
        "09:30:00-09:30:10",
        "09:40:00-09:40:10",
        "09:50:00-09:50:10"
    ],
    "current_clip_overlap": False,
    "current_clip_note": (
        "The current MP4 sample covers approximately 09:04:38-09:04:49.916, "
        "so exact behaviour labels from the Excel scan-sampling windows cannot be assigned."
    )
}

frames = []

for frame_idx, frame_df in df.groupby("frame"):
    frame_df = frame_df.sort_values("track_id")

    timestamp = str(frame_df["estimated_timestamp"].iloc[0])
    time_from_start = float(frame_df["time_from_video_start_sec"].iloc[0])

    pigs = []

    for _, row in frame_df.iterrows():
        pig_entry = {
            "track_id": int(row["track_id"]),
            "pig_id": None,
            "colour": None,
            "identity_status": "unassigned_track_id",
            "bbox_xyxy": [
                round(float(row["x1"]), 3),
                round(float(row["y1"]), 3),
                round(float(row["x2"]), 3),
                round(float(row["y2"]), 3)
            ],
            "bbox_xywh": [
                round(float(row["x1"]), 3),
                round(float(row["y1"]), 3),
                round(float(row["x2"] - row["x1"]), 3),
                round(float(row["y2"] - row["y1"]), 3)
            ],
            "centroid": [
                round(float(row["cx"]), 3),
                round(float(row["cy"]), 3)
            ],
            "detection_score": round(float(row["score"]), 4),
            "behaviour": {
                "code": None,
                "label": "unknown_no_scan_overlap",
                "source": "not_assigned",
                "confidence": 0.0,
                "note": "No exact overlap between this video timestamp and Excel scan-sampling intervals."
            }
        }

        pigs.append(pig_entry)

    frames.append({
        "frame_index": int(frame_idx),
        "timestamp": timestamp,
        "time_from_video_start_sec": round(time_from_start, 6),
        "pigs": pigs
    })

annotation = {
    "video": video_info,
    "annotation_protocol": protocol,
    "frames": frames
}

schema_description = {
    "top_level": {
        "video": "Metadata about the clip, FPS, source video, and timestamp assumptions.",
        "annotation_protocol": "Description of scan-sampling protocol and overlap status.",
        "frames": "List of frame-level annotations."
    },
    "frame_fields": {
        "frame_index": "1-based frame index in the extracted sequence.",
        "timestamp": "Estimated real-world timestamp derived from camera overlay and FPS.",
        "time_from_video_start_sec": "Elapsed time from the start of the extracted sample.",
        "pigs": "List of tracked pigs visible in the frame."
    },
    "pig_fields": {
        "track_id": "Tracker-generated identity.",
        "pig_id": "Animal identity from colour marker or manual mapping, if available.",
        "colour": "Colour marker identity, if manually assigned.",
        "identity_status": "Status of identity assignment.",
        "bbox_xyxy": "Bounding box as x1, y1, x2, y2.",
        "bbox_xywh": "Bounding box as x, y, width, height.",
        "centroid": "Bounding box centre point.",
        "detection_score": "Detector confidence score.",
        "behaviour": "Behaviour label object."
    }
}

with open(OUT_JSON, "w") as f:
    json.dump(annotation, f, indent=2)

with open(OUT_SCHEMA, "w") as f:
    json.dump(schema_description, f, indent=2)

print("Input tracks:", TRACKS_CSV)
print("Output annotation JSON:", OUT_JSON)
print("Output schema description:", OUT_SCHEMA)
print("Frames written:", len(frames))
print("Total pig/frame entries:", sum(len(frame["pigs"]) for frame in frames))
print("Unique track IDs:", df["track_id"].nunique())

print()
print("Example first frame:")
print(json.dumps(frames[0], indent=2)[:3000])
