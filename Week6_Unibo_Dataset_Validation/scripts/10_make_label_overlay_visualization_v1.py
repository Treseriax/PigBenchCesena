from pathlib import Path
import csv
import cv2
import numpy as np
import pandas as pd


PROJECT_ROOT = Path.home() / "PigBench"
W6 = PROJECT_ROOT / "Week6_Unibo_Dataset_Validation"

GT = W6 / "outputs/unified_ground_truth/week6_unified_ground_truth_v2_with_recovered_videos.csv"
OUT_VIS = W6 / "outputs/visual_label_check"
NOTES = W6 / "notes"

FRAME_DIR = OUT_VIS / "label_overlay_frames_v1"
VIDEO_DIR = OUT_VIS / "label_overlay_videos_v1"

FRAME_DIR.mkdir(parents=True, exist_ok=True)
VIDEO_DIR.mkdir(parents=True, exist_ok=True)
NOTES.mkdir(parents=True, exist_ok=True)


def safe_to_csv(df, path):
    df.to_csv(
        path,
        index=False,
        quoting=csv.QUOTE_ALL,
        escapechar="\\",
        lineterminator="\n"
    )


def read_frame(video_path, frame_index):
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        return None, "video_open_failed"

    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if total <= 0:
        cap.release()
        return None, "frame_count_unavailable"

    frame_index = max(0, min(int(frame_index), total - 1))
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_index)

    ok, frame = cap.read()
    cap.release()

    if not ok:
        return None, "frame_read_failed"

    return frame, "ok"


def put_text(img, text, x, y, scale=0.5, thickness=1):
    cv2.putText(
        img,
        str(text),
        (x, y),
        cv2.FONT_HERSHEY_SIMPLEX,
        scale,
        (0, 0, 0),
        thickness,
        cv2.LINE_AA,
    )


def make_overlay(frame, group, title):
    frame_h, frame_w = frame.shape[:2]

    target_w = 704
    scale = target_w / frame_w
    target_h = int(frame_h * scale)
    frame_resized = cv2.resize(frame, (target_w, target_h), interpolation=cv2.INTER_AREA)

    panel_w = 720
    panel = np.full((target_h, panel_w, 3), 245, dtype=np.uint8)

    y = 30
    put_text(panel, title, 15, y, 0.55, 1)
    y += 35

    video_id = group["video_id"].iloc[0]
    timestamp = group["timestamp"].iloc[0]
    frame_index = group["frame_index"].iloc[0]
    status = group["video_match_status"].iloc[0]
    confidence = group["video_mapping_confidence"].iloc[0]

    put_text(panel, f"video_id: {video_id}", 15, y)
    y += 25
    put_text(panel, f"timestamp: {timestamp}", 15, y)
    y += 25
    put_text(panel, f"frame_index: {frame_index}", 15, y)
    y += 25
    put_text(panel, f"video_match: {status}", 15, y)
    y += 25
    put_text(panel, f"confidence: {confidence}", 15, y)
    y += 35

    put_text(panel, "Manual scan-sampling labels:", 15, y, 0.55, 1)
    y += 30

    for _, r in group.sort_values("colour_id").iterrows():
        line1 = f"{r['colour_id']}: {r['behaviour_code']}"
        line2 = f"  {r['behaviour_label']}"
        put_text(panel, line1, 25, y, 0.55, 1)
        y += 22
        put_text(panel, line2[:75], 25, y, 0.43, 1)
        y += 30

    y += 10
    put_text(panel, "Note: bbox not linked yet.", 15, min(y, target_h - 45), 0.5, 1)
    put_text(panel, "Next step: detector/tracker bbox association.", 15, min(y + 25, target_h - 20), 0.5, 1)

    combined = np.hstack([frame_resized, panel])
    return combined


if not GT.exists():
    raise FileNotFoundError(GT)

gt = pd.read_csv(GT)

# Choose one scan point per available hour: minute 0 for all 12 videos.
sample = gt[gt["timestamp_sec_in_video"].astype(float).eq(0.0)].copy()

# There should be six rows per hour because six pigs.
group_cols = ["video_id", "video_path", "timestamp", "frame_index"]

summary_rows = []
written_frames = []

for keys, group in sample.groupby(group_cols):
    video_id, video_path, timestamp, frame_index = keys

    if not video_path or not Path(video_path).exists():
        summary_rows.append({
            "video_id": video_id,
            "timestamp": timestamp,
            "frame_index": frame_index,
            "status": "video_path_missing",
            "output_frame": "",
        })
        continue

    frame, status = read_frame(video_path, frame_index)

    if status != "ok":
        summary_rows.append({
            "video_id": video_id,
            "timestamp": timestamp,
            "frame_index": frame_index,
            "status": status,
            "output_frame": "",
        })
        continue

    safe_video_id = str(video_id).replace(" ", "_").replace("/", "_").replace(":", "_")
    safe_ts = str(timestamp).replace(":", "").replace("-", "").replace("T", "_")
    out_frame = FRAME_DIR / f"{safe_ts}_{safe_video_id}_labels.jpg"

    title = "Week 6 Unibo label overlay v1"
    overlay = make_overlay(frame, group, title)
    cv2.imwrite(str(out_frame), overlay)

    written_frames.append(out_frame)

    summary_rows.append({
        "video_id": video_id,
        "timestamp": timestamp,
        "frame_index": frame_index,
        "status": "written",
        "output_frame": str(out_frame),
        "num_labels": len(group),
        "video_match_status": group["video_match_status"].iloc[0],
    })

summary = pd.DataFrame(summary_rows)
summary_path = OUT_VIS / "label_overlay_v1_frame_summary.csv"
safe_to_csv(summary, summary_path)

# Make a simple slideshow video from generated frames.
if written_frames:
    imgs = [cv2.imread(str(p)) for p in written_frames]
    imgs = [im for im in imgs if im is not None]

    if imgs:
        h = max(im.shape[0] for im in imgs)
        w = max(im.shape[1] for im in imgs)

        padded_imgs = []
        for im in imgs:
            pad_bottom = h - im.shape[0]
            pad_right = w - im.shape[1]
            if pad_bottom or pad_right:
                im = cv2.copyMakeBorder(
                    im,
                    0,
                    pad_bottom,
                    0,
                    pad_right,
                    cv2.BORDER_CONSTANT,
                    value=(255, 255, 255),
                )
            padded_imgs.append(im)

        video_path = VIDEO_DIR / "week6_label_overlay_v1_scanpoint_slideshow.mp4"
        writer = cv2.VideoWriter(
            str(video_path),
            cv2.VideoWriter_fourcc(*"mp4v"),
            1.0,
            (w, h),
        )

        for im in padded_imgs:
            # hold each frame for 2 seconds
            writer.write(im)
            writer.write(im)

        writer.release()
    else:
        video_path = ""
else:
    video_path = ""

note_path = NOTES / "week6_label_overlay_visualization_v1_notes.md"

with open(note_path, "w") as f:
    f.write("# Week 6 Label Overlay Visualization v1 Notes\n\n")

    f.write("## Purpose\n\n")
    f.write(
        "This visualization checks whether the unified ground-truth labels are correctly linked to video frames. "
        "Because bounding boxes are not yet linked, this v1 visualization displays the frame and a side panel with pig colour IDs and manual behaviour labels.\n\n"
    )

    f.write("## Outputs\n\n")
    f.write(f"- Frame directory: `{FRAME_DIR}`\n")
    f.write(f"- Slideshow video: `{video_path}`\n")
    f.write(f"- Summary CSV: `{summary_path}`\n\n")

    f.write("## Summary\n\n")
    if len(summary):
        f.write(summary.to_markdown(index=False))
    else:
        f.write("No frames generated.")
    f.write("\n\n")

    f.write("## Interpretation\n\n")
    f.write(
        "The generated frames provide a visual check for timestamp/video matching. "
        "This is not yet the final bbox+label visualization requested by the task sheet. "
        "The next step is to link detector/tracker bounding boxes to these scan-sampling timestamps and then render bbox + colour ID + behaviour label overlays.\n"
    )

print("Saved:")
print(summary_path)
print(note_path)
print(video_path)

print()
print("=== Label overlay v1 summary ===")
print(summary.to_string(index=False))

print()
print("=== Generated frames ===")
for p in written_frames[:20]:
    print(p)
