from pathlib import Path
import csv
import cv2
import numpy as np
import pandas as pd


PROJECT_ROOT = Path.home() / "PigBench"
W6 = PROJECT_ROOT / "Week6_Unibo_Dataset_Validation"

DETECTIONS = W6 / "outputs/feature_extractors/week6_yolov8s_all_scanpoint_detections.csv"
FRAME_INDEX = W6 / "outputs/unified_ground_truth/week6_scanpoint_frame_index.csv"
LABELS = W6 / "outputs/unified_ground_truth/week6_scanpoint_frame_labels_long.csv"

OUT_VIS = W6 / "outputs/visual_label_check/bbox_label_overlay_v2"
OUT_VIDEO = W6 / "outputs/visual_label_check/bbox_label_overlay_v2_video"
OUT_SUMMARY = W6 / "outputs/visual_label_check"
NOTES = W6 / "notes"

OUT_VIS.mkdir(parents=True, exist_ok=True)
OUT_VIDEO.mkdir(parents=True, exist_ok=True)
OUT_SUMMARY.mkdir(parents=True, exist_ok=True)
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


def put_text(img, text, x, y, scale=0.48, thickness=1):
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


def draw_detection_boxes(img, dets):
    out = img.copy()

    for _, r in dets.iterrows():
        x1, y1, x2, y2 = int(r["x1"]), int(r["y1"]), int(r["x2"]), int(r["y2"])
        score = float(r["score"])
        det_id = int(r["det_id"])

        cv2.rectangle(out, (x1, y1), (x2, y2), (0, 255, 0), 2)

        label = f"D{det_id} {score:.2f}"
        cv2.putText(
            out,
            label,
            (x1, max(18, y1 - 5)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            (0, 255, 0),
            2,
            cv2.LINE_AA,
        )

    return out


def make_panel(frame_row, label_group, det_group, panel_h):
    panel_w = 780
    panel = np.full((panel_h, panel_w, 3), 245, dtype=np.uint8)

    y = 28

    put_text(panel, "Week 6 bbox + manual label overlay v2", 15, y, 0.55, 1)
    y += 32

    put_text(panel, f"scan_frame_id: {frame_row['scan_frame_id']}", 15, y)
    y += 23
    put_text(panel, f"video_id: {frame_row['video_id']}", 15, y)
    y += 23
    put_text(panel, f"timestamp: {frame_row['timestamp']}", 15, y)
    y += 23
    put_text(panel, f"frame_index: {frame_row['frame_index']}", 15, y)
    y += 23
    put_text(panel, f"video_match: {frame_row['video_match_status']}", 15, y)
    y += 30

    put_text(panel, "Detector output:", 15, y, 0.52, 1)
    y += 25
    put_text(panel, f"bbox count: {len(det_group)}", 25, y)
    y += 22

    if len(det_group):
        put_text(panel, f"max score: {det_group['score'].max():.3f}", 25, y)
        y += 22
        put_text(panel, f"mean score: {det_group['score'].mean():.3f}", 25, y)
    else:
        put_text(panel, "no detections above threshold", 25, y)

    y += 35

    put_text(panel, "Manual Excel scan labels:", 15, y, 0.52, 1)
    y += 27

    for _, r in label_group.sort_values("colour_id").iterrows():
        put_text(panel, f"{r['colour_id']}: {r['behaviour_code']}", 25, y, 0.50, 1)
        y += 21
        put_text(panel, f"  {str(r['behaviour_label'])[:80]}", 25, y, 0.40, 1)
        y += 28

    y += 8
    put_text(panel, "Important:", 15, min(y, panel_h - 70), 0.50, 1)
    put_text(panel, "bbox = detector pig instance", 25, min(y + 22, panel_h - 48), 0.43, 1)
    put_text(panel, "colour/behaviour = manual Excel label", 25, min(y + 43, panel_h - 25), 0.43, 1)
    put_text(panel, "bbox-to-colour assignment is pending.", 25, min(y + 64, panel_h - 5), 0.43, 1)

    return panel


if not DETECTIONS.exists():
    raise FileNotFoundError(DETECTIONS)
if not FRAME_INDEX.exists():
    raise FileNotFoundError(FRAME_INDEX)
if not LABELS.exists():
    raise FileNotFoundError(LABELS)

dets = pd.read_csv(DETECTIONS)
frames = pd.read_csv(FRAME_INDEX)
labels = pd.read_csv(LABELS)

written = []
summary_rows = []

for _, frame_row in frames.iterrows():
    scan_frame_id = frame_row["scan_frame_id"]
    image_path = Path(frame_row["frame_image_path"])

    if not image_path.exists():
        summary_rows.append({
            "scan_frame_id": scan_frame_id,
            "status": "missing_frame_image",
            "output_path": "",
            "bbox_count": 0,
            "manual_label_count": 0,
        })
        continue

    img = cv2.imread(str(image_path))

    if img is None:
        summary_rows.append({
            "scan_frame_id": scan_frame_id,
            "status": "frame_read_failed",
            "output_path": "",
            "bbox_count": 0,
            "manual_label_count": 0,
        })
        continue

    det_group = dets[dets["scan_frame_id"] == scan_frame_id].copy()
    label_group = labels[labels["scan_frame_id"] == scan_frame_id].copy()

    img_boxes = draw_detection_boxes(img, det_group)

    # Resize frame to a consistent width.
    target_w = 704
    h, w = img_boxes.shape[:2]
    scale = target_w / w
    target_h = int(h * scale)
    img_boxes = cv2.resize(img_boxes, (target_w, target_h), interpolation=cv2.INTER_AREA)

    panel = make_panel(frame_row, label_group, det_group, target_h)

    combined = np.hstack([img_boxes, panel])

    out_name = f"{scan_frame_id}__{safe_name(frame_row['timestamp'])}__bbox_label_overlay_v2.jpg"
    out_path = OUT_VIS / out_name

    cv2.imwrite(str(out_path), combined)
    written.append(out_path)

    summary_rows.append({
        "scan_frame_id": scan_frame_id,
        "status": "written",
        "output_path": str(out_path),
        "bbox_count": len(det_group),
        "manual_label_count": len(label_group),
        "video_id": frame_row["video_id"],
        "timestamp": frame_row["timestamp"],
        "frame_index": frame_row["frame_index"],
        "video_match_status": frame_row["video_match_status"],
        "max_detection_score": float(det_group["score"].max()) if len(det_group) else "",
        "mean_detection_score": float(det_group["score"].mean()) if len(det_group) else "",
    })

summary = pd.DataFrame(summary_rows)

summary_path = OUT_SUMMARY / "bbox_label_overlay_v2_summary.csv"
safe_to_csv(summary, summary_path)

# Slideshow video.
video_path = OUT_VIDEO / "week6_bbox_label_overlay_v2_slideshow.mp4"

if written:
    imgs = [cv2.imread(str(p)) for p in written]
    imgs = [im for im in imgs if im is not None]

    if imgs:
        h = max(im.shape[0] for im in imgs)
        w = max(im.shape[1] for im in imgs)

        padded = []

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

            padded.append(im)

        writer = cv2.VideoWriter(
            str(video_path),
            cv2.VideoWriter_fourcc(*"mp4v"),
            1.0,
            (w, h),
        )

        for im in padded:
            writer.write(im)
            writer.write(im)

        writer.release()

# Summary tables.
if len(summary):
    status_summary = (
        summary.groupby(["status", "video_match_status"])
        .agg(
            frame_count=("scan_frame_id", "count"),
            total_bboxes=("bbox_count", "sum"),
            mean_bboxes_per_frame=("bbox_count", "mean"),
            min_bboxes=("bbox_count", "min"),
            max_bboxes=("bbox_count", "max"),
            total_manual_labels=("manual_label_count", "sum"),
        )
        .reset_index()
    )
else:
    status_summary = pd.DataFrame()

status_path = OUT_SUMMARY / "bbox_label_overlay_v2_status_summary.csv"
safe_to_csv(status_summary, status_path)

note_path = NOTES / "week6_bbox_label_overlay_visualization_v2_notes.md"

with open(note_path, "w") as f:
    f.write("# Week 6 Bbox + Label Overlay Visualization v2 Notes\n\n")

    f.write("## Purpose\n\n")
    f.write(
        "This visualization combines YOLOv8-s pig detections with manual Excel scan-sampling behaviour labels. "
        "It provides a visual quality-control bridge between video frames, detector bboxes, pig colour IDs, and behaviour labels.\n\n"
    )

    f.write("## Outputs\n\n")
    f.write(f"- Overlay image directory: `{OUT_VIS}`\n")
    f.write(f"- Slideshow video: `{video_path}`\n")
    f.write(f"- Overlay summary CSV: `{summary_path}`\n")
    f.write(f"- Status summary CSV: `{status_path}`\n\n")

    f.write("## Status summary\n\n")
    f.write(status_summary.to_markdown(index=False) if len(status_summary) else "No status summary.")
    f.write("\n\n")

    f.write("## Interpretation\n\n")
    f.write(
        "The detector bboxes are available for all scanpoint frames. "
        "The manual labels are shown as a side panel because the exact bbox-to-colour-ID association has not yet been solved. "
        "The next step is crop-based colour-marker analysis, where each detected pig crop is scored for green/blue/purple/red marker evidence and then linked to manual colour IDs when possible.\n"
    )

print("Saved:")
print(summary_path)
print(status_path)
print(video_path)
print(note_path)
print(OUT_VIS)

print()
print("=== Bbox + label overlay v2 status summary ===")
print(status_summary.to_string(index=False) if len(status_summary) else "No status summary.")

print()
print("=== Generated overlay count ===")
print(len(written))

print()
print("=== First 10 overlays ===")
for p in written[:10]:
    print(p)
