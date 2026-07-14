from pathlib import Path
import csv
import cv2
import pandas as pd


PROJECT_ROOT = Path.home() / "PigBench"
W6 = PROJECT_ROOT / "Week6_Unibo_Dataset_Validation"

GT = W6 / "outputs/unified_ground_truth/week6_unified_ground_truth_v2_with_recovered_videos.csv"
OUT_FRAMES = W6 / "outputs/visual_label_check/all_scanpoint_frames"
OUT_GT = W6 / "outputs/unified_ground_truth"
NOTES = W6 / "notes"

OUT_FRAMES.mkdir(parents=True, exist_ok=True)
OUT_GT.mkdir(parents=True, exist_ok=True)
NOTES.mkdir(parents=True, exist_ok=True)


def safe_to_csv(df, path):
    df.to_csv(
        path,
        index=False,
        quoting=csv.QUOTE_ALL,
        escapechar="\\",
        lineterminator="\n"
    )


def safe_name(x):
    return (
        str(x)
        .replace("/", "_")
        .replace(" ", "_")
        .replace(":", "")
        .replace("-", "")
        .replace("T", "_")
    )


def read_frame(video_path, frame_index):
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        return None, "video_open_failed", {}

    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = float(cap.get(cv2.CAP_PROP_FPS))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    if total <= 0:
        cap.release()
        return None, "frame_count_unavailable", {
            "frame_count": total,
            "fps": fps,
            "width": width,
            "height": height,
        }

    idx = max(0, min(int(frame_index), total - 1))
    cap.set(cv2.CAP_PROP_POS_FRAMES, idx)

    ok, frame = cap.read()
    cap.release()

    meta = {
        "frame_count": total,
        "fps": fps,
        "width": width,
        "height": height,
        "used_frame_index": idx,
    }

    if not ok:
        return None, "frame_read_failed", meta

    return frame, "ok", meta


gt = pd.read_csv(GT)

# One group = one observation timestamp / video frame.
group_cols = [
    "video_id",
    "video_path",
    "timestamp",
    "frame_index",
    "timestamp_sec_in_video",
    "video_match_status",
    "video_mapping_confidence",
]

rows = []
label_rows = []

for keys, group in gt.groupby(group_cols, dropna=False):
    (
        video_id,
        video_path,
        timestamp,
        frame_index,
        timestamp_sec_in_video,
        video_match_status,
        video_mapping_confidence,
    ) = keys

    if pd.isna(video_path) or not str(video_path).strip():
        status = "missing_video_path"
        out_path = ""
        meta = {}
    elif not Path(str(video_path)).exists():
        status = "video_path_does_not_exist"
        out_path = ""
        meta = {}
    else:
        frame, status, meta = read_frame(str(video_path), frame_index)

        if status == "ok":
            frame_name = (
                f"{safe_name(timestamp)}"
                f"__{safe_name(video_id)}"
                f"__frame_{int(float(frame_index)):06d}.jpg"
            )
            out_path = OUT_FRAMES / frame_name
            cv2.imwrite(str(out_path), frame)
        else:
            out_path = ""

    frame_record_id = f"scanframe_{len(rows):04d}"

    rows.append({
        "scan_frame_id": frame_record_id,
        "video_id": video_id,
        "video_path": video_path,
        "timestamp": timestamp,
        "timestamp_sec_in_video": timestamp_sec_in_video,
        "frame_index": frame_index,
        "extraction_status": status,
        "frame_image_path": str(out_path),
        "video_match_status": video_match_status,
        "video_mapping_confidence": video_mapping_confidence,
        "num_labels_at_frame": len(group),
        "video_fps": meta.get("fps", ""),
        "video_frame_count": meta.get("frame_count", ""),
        "video_width": meta.get("width", ""),
        "video_height": meta.get("height", ""),
        "used_frame_index": meta.get("used_frame_index", ""),
    })

    for _, r in group.iterrows():
        label_rows.append({
            "scan_frame_id": frame_record_id,
            "record_id": r["record_id"],
            "video_id": r["video_id"],
            "timestamp": r["timestamp"],
            "frame_index": r["frame_index"],
            "pig_id": r["pig_id"],
            "colour_id": r["colour_id"],
            "colour_raw": r["colour_raw"],
            "behaviour_code": r["behaviour_code"],
            "behaviour_label": r["behaviour_label"],
            "frame_image_path": str(out_path),
            "bbox_available": False,
            "bbox_assignment_status": "pending_detector_or_tracker",
        })

frames_df = pd.DataFrame(rows)
labels_df = pd.DataFrame(label_rows)

frames_path = OUT_GT / "week6_scanpoint_frame_index.csv"
labels_path = OUT_GT / "week6_scanpoint_frame_labels_long.csv"

safe_to_csv(frames_df, frames_path)
safe_to_csv(labels_df, labels_path)

status_summary = (
    frames_df.groupby(["extraction_status", "video_match_status"])
    .size()
    .reset_index(name="frame_count")
    .sort_values(["extraction_status", "video_match_status"])
)

status_path = OUT_GT / "week6_scanpoint_frame_extraction_status_summary.csv"
safe_to_csv(status_summary, status_path)

note_path = NOTES / "week6_scanpoint_frame_extraction_notes.md"

with open(note_path, "w") as f:
    f.write("# Week 6 Scanpoint Frame Extraction Notes\n\n")

    f.write("## Purpose\n\n")
    f.write(
        "This step extracts one video frame for each unique manual Excel scan-sampling timestamp. "
        "These frames are the basis for visual quality control, detector bbox extraction, colour-marker analysis, and segmentation tests.\n\n"
    )

    f.write("## Outputs\n\n")
    f.write(f"- Frame directory: `{OUT_FRAMES}`\n")
    f.write(f"- Scan frame index: `{frames_path}`\n")
    f.write(f"- Long label table per scan frame: `{labels_path}`\n")
    f.write(f"- Extraction status summary: `{status_path}`\n\n")

    f.write("## Extraction status summary\n\n")
    f.write(status_summary.to_markdown(index=False))
    f.write("\n\n")

    f.write("## Interpretation\n\n")
    f.write(
        "Each extracted frame should have six manual pig-colour labels from the Excel sheet. "
        "Bounding boxes are still pending and will be added after running or linking a pig detector/tracker.\n"
    )

print("Saved:")
print(frames_path)
print(labels_path)
print(status_path)
print(note_path)
print(OUT_FRAMES)

print()
print("=== Frame extraction status summary ===")
print(status_summary.to_string(index=False))

print()
print("=== First 20 scan frames ===")
print(frames_df.head(20).to_string(index=False))
