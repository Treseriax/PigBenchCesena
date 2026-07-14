from pathlib import Path
import csv
import cv2
import numpy as np
import pandas as pd


PROJECT_ROOT = Path.home() / "PigBench"
W6 = PROJECT_ROOT / "Week6_Unibo_Dataset_Validation"

MARKERS = W6 / "outputs/feature_extractors/week6_crop_colour_marker_features.csv"
FRAME_INDEX = W6 / "outputs/unified_ground_truth/week6_scanpoint_frame_index.csv"
LABELS = W6 / "outputs/unified_ground_truth/week6_scanpoint_frame_labels_long.csv"

OUT_VIS = W6 / "outputs/visual_label_check/marker_bbox_overlay_v3"
OUT_VIDEO = W6 / "outputs/visual_label_check/marker_bbox_overlay_v3_video"
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


def put_text(img, text, x, y, scale=0.46, thickness=1):
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


def marker_to_draw_colour(marker, confidence):
    # BGR colors for OpenCV visualization.
    if confidence not in ["high", "medium"]:
        return (180, 180, 180)

    if marker == "green":
        return (0, 220, 0)
    if marker == "blue":
        return (255, 80, 0)
    if marker == "purple":
        return (180, 0, 180)
    if marker == "red":
        return (0, 0, 255)

    return (180, 180, 180)


def marker_display_text(row):
    conf = str(row.get("marker_confidence", "none"))
    marker = str(row.get("best_marker_colour", "no_marker_detected"))

    if conf in ["high", "medium"]:
        if marker == "red":
            return f"{marker}? {conf}"
        return f"{marker} {conf}"

    return "no marker"


def draw_marker_boxes(img, dets):
    out = img.copy()

    for _, r in dets.iterrows():
        x1, y1, x2, y2 = int(r["x1"]), int(r["y1"]), int(r["x2"]), int(r["y2"])
        det_id = int(r["det_id"])
        score = float(r["score"])
        conf = str(r["marker_confidence"])
        marker = str(r["best_marker_colour"])

        colour = marker_to_draw_colour(marker, conf)
        thickness = 3 if conf == "high" else 2

        cv2.rectangle(out, (x1, y1), (x2, y2), colour, thickness)

        label = f"D{det_id} {score:.2f} {marker_display_text(r)}"
        cv2.putText(
            out,
            label,
            (x1, max(18, y1 - 5)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.43,
            colour,
            2,
            cv2.LINE_AA,
        )

    return out


def make_panel(frame_row, label_group, marker_group, panel_h):
    panel_w = 840
    panel = np.full((panel_h, panel_w, 3), 245, dtype=np.uint8)

    y = 28

    put_text(panel, "Week 6 marker-bbox overlay v3", 15, y, 0.56, 1)
    y += 32

    put_text(panel, f"scan_frame_id: {frame_row['scan_frame_id']}", 15, y)
    y += 23
    put_text(panel, f"video_id: {frame_row['video_id']}", 15, y)
    y += 23
    put_text(panel, f"timestamp: {frame_row['timestamp']}", 15, y)
    y += 23
    put_text(panel, f"video_match: {frame_row['video_match_status']}", 15, y)
    y += 31

    high_count = int((marker_group["marker_confidence"] == "high").sum())
    med_count = int((marker_group["marker_confidence"] == "medium").sum())
    low_count = int((marker_group["marker_confidence"] == "low").sum())
    none_count = int((marker_group["marker_confidence"] == "none").sum())

    put_text(panel, "Detector + crop-marker candidates:", 15, y, 0.52, 1)
    y += 25
    put_text(panel, f"bbox count: {len(marker_group)}", 25, y)
    y += 22
    put_text(panel, f"high/medium marker: {high_count + med_count}  (high={high_count}, medium={med_count})", 25, y)
    y += 22
    put_text(panel, f"low/no marker: {low_count + none_count}  (low={low_count}, none={none_count})", 25, y)
    y += 32

    if len(marker_group):
        compact = marker_group[marker_group["marker_confidence"].isin(["high", "medium"])].copy()
        compact = compact.sort_values(["marker_confidence", "best_marker_colour", "best_marker_score"], ascending=[True, True, False])

        put_text(panel, "Candidate marker bboxes:", 15, y, 0.52, 1)
        y += 24

        if len(compact):
            for _, r in compact.head(10).iterrows():
                marker = r["best_marker_colour"]
                if marker == "red":
                    marker = "red? neck/tail"
                put_text(
                    panel,
                    f"D{int(r['det_id'])}: {marker}, {r['marker_confidence']}, marker={float(r['best_marker_score']):.3f}, det={float(r['score']):.2f}",
                    25,
                    y,
                    0.41,
                    1,
                )
                y += 21
        else:
            put_text(panel, "No high/medium marker candidates in this frame.", 25, y, 0.43, 1)
            y += 22

    y += 18
    put_text(panel, "Manual Excel labels:", 15, y, 0.52, 1)
    y += 26

    for _, r in label_group.sort_values("colour_id").iterrows():
        put_text(panel, f"{r['colour_id']}: {r['behaviour_code']}", 25, y, 0.49, 1)
        y += 21
        put_text(panel, f"  {str(r['behaviour_label'])[:82]}", 25, y, 0.39, 1)
        y += 26

    y_note = max(y + 8, panel_h - 86)
    put_text(panel, "Interpretation:", 15, min(y_note, panel_h - 66), 0.49, 1)
    put_text(panel, "Marker colour is candidate only, not final identity GT.", 25, min(y_note + 22, panel_h - 44), 0.40, 1)
    put_text(panel, "Red marker is ambiguous: red_neck or red_tail.", 25, min(y_note + 43, panel_h - 23), 0.40, 1)
    put_text(panel, "No-color pig cannot be marker-detected.", 25, min(y_note + 64, panel_h - 4), 0.40, 1)

    return panel


if not MARKERS.exists():
    raise FileNotFoundError(MARKERS)
if not FRAME_INDEX.exists():
    raise FileNotFoundError(FRAME_INDEX)
if not LABELS.exists():
    raise FileNotFoundError(LABELS)

markers = pd.read_csv(MARKERS)
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
        })
        continue

    img = cv2.imread(str(image_path))

    if img is None:
        summary_rows.append({
            "scan_frame_id": scan_frame_id,
            "status": "frame_read_failed",
            "output_path": "",
        })
        continue

    marker_group = markers[markers["scan_frame_id"] == scan_frame_id].copy()
    label_group = labels[labels["scan_frame_id"] == scan_frame_id].copy()

    img_boxes = draw_marker_boxes(img, marker_group)

    target_w = 704
    h, w = img_boxes.shape[:2]
    scale = target_w / w
    target_h = int(h * scale)
    img_boxes = cv2.resize(img_boxes, (target_w, target_h), interpolation=cv2.INTER_AREA)

    panel = make_panel(frame_row, label_group, marker_group, target_h)

    combined = np.hstack([img_boxes, panel])

    out_name = f"{scan_frame_id}__{safe_name(frame_row['timestamp'])}__marker_bbox_overlay_v3.jpg"
    out_path = OUT_VIS / out_name

    cv2.imwrite(str(out_path), combined)
    written.append(out_path)

    summary_rows.append({
        "scan_frame_id": scan_frame_id,
        "status": "written",
        "output_path": str(out_path),
        "video_id": frame_row["video_id"],
        "timestamp": frame_row["timestamp"],
        "video_match_status": frame_row["video_match_status"],
        "bbox_count": len(marker_group),
        "manual_label_count": len(label_group),
        "high_marker_candidates": int((marker_group["marker_confidence"] == "high").sum()),
        "medium_marker_candidates": int((marker_group["marker_confidence"] == "medium").sum()),
        "low_marker_candidates": int((marker_group["marker_confidence"] == "low").sum()),
        "no_marker": int((marker_group["marker_confidence"] == "none").sum()),
        "green_candidates": int(((marker_group["best_marker_colour"] == "green") & marker_group["marker_confidence"].isin(["high", "medium"])).sum()),
        "blue_candidates": int(((marker_group["best_marker_colour"] == "blue") & marker_group["marker_confidence"].isin(["high", "medium"])).sum()),
        "purple_candidates": int(((marker_group["best_marker_colour"] == "purple") & marker_group["marker_confidence"].isin(["high", "medium"])).sum()),
        "red_candidates": int(((marker_group["best_marker_colour"] == "red") & marker_group["marker_confidence"].isin(["high", "medium"])).sum()),
    })

summary = pd.DataFrame(summary_rows)

summary_path = OUT_SUMMARY / "marker_bbox_overlay_v3_summary.csv"
safe_to_csv(summary, summary_path)

if len(summary):
    status_summary = (
        summary.groupby(["status", "video_match_status"])
        .agg(
            frame_count=("scan_frame_id", "count"),
            total_bboxes=("bbox_count", "sum"),
            total_manual_labels=("manual_label_count", "sum"),
            high_marker_candidates=("high_marker_candidates", "sum"),
            medium_marker_candidates=("medium_marker_candidates", "sum"),
            low_marker_candidates=("low_marker_candidates", "sum"),
            no_marker=("no_marker", "sum"),
            green_candidates=("green_candidates", "sum"),
            blue_candidates=("blue_candidates", "sum"),
            purple_candidates=("purple_candidates", "sum"),
            red_candidates=("red_candidates", "sum"),
        )
        .reset_index()
    )
else:
    status_summary = pd.DataFrame()

status_path = OUT_SUMMARY / "marker_bbox_overlay_v3_status_summary.csv"
safe_to_csv(status_summary, status_path)

video_path = OUT_VIDEO / "week6_marker_bbox_overlay_v3_slideshow.mp4"

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

note_path = NOTES / "week6_marker_bbox_overlay_visualization_v3_notes.md"

with open(note_path, "w") as f:
    f.write("# Week 6 Marker-Bbox Overlay Visualization v3 Notes\n\n")

    f.write("## Purpose\n\n")
    f.write(
        "This visualization overlays YOLOv8-s pig bounding boxes and crop-based colour-marker candidates on all 72 scanpoint frames. "
        "The side panel preserves the manual Excel colour IDs and behaviour labels for visual comparison.\n\n"
    )

    f.write("## Outputs\n\n")
    f.write(f"- Overlay image directory: `{OUT_VIS}`\n")
    f.write(f"- Slideshow video: `{video_path}`\n")
    f.write(f"- Summary CSV: `{summary_path}`\n")
    f.write(f"- Status summary CSV: `{status_path}`\n\n")

    f.write("## Status summary\n\n")
    f.write(status_summary.to_markdown(index=False) if len(status_summary) else "No status summary.")
    f.write("\n\n")

    f.write("## Interpretation\n\n")
    f.write(
        "The visualization should be used for visual quality control. "
        "Green, blue, purple, and red marker detections are candidate bbox-to-colour links. "
        "Red remains ambiguous because the manual labels distinguish red_neck and red_tail, while HSV marker detection only detects red. "
        "No-color pigs cannot be recovered through colour-marker evidence. "
        "Therefore v3 is a strong visual QC artifact but not final identity-resolved ground truth.\n"
    )

print("Saved:")
print(summary_path)
print(status_path)
print(video_path)
print(note_path)
print(OUT_VIS)

print()
print("=== Marker bbox overlay v3 status summary ===")
print(status_summary.to_string(index=False) if len(status_summary) else "No status summary.")

print()
print("=== Generated overlay count ===")
print(len(written))

print()
print("=== First 10 overlays ===")
for p in written[:10]:
    print(p)
