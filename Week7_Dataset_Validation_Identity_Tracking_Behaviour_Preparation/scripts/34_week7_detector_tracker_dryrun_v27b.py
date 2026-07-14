from pathlib import Path
from datetime import datetime
import csv
import math
import re
import traceback
import os
os.environ["CUDA_VISIBLE_DEVICES"] = ""

import cv2
import numpy as np
import pandas as pd
from PIL import Image, ImageDraw, ImageFont

import torch
from mmdet.apis import init_detector, inference_detector


ROOT = Path.home() / "PigBench"
W7 = ROOT / "Week7_Dataset_Validation_Identity_Tracking_Behaviour_Preparation"

SELECTED_CLIPS = W7 / "outputs" / "detector_tracker_preflight_v27a" / "week7_detector_tracker_preflight_v27a_selected_dryrun_clips.csv"

CONFIG = ROOT / "detection" / "configs" / "yolov8" / "yolov8_s.py"
CHECKPOINT = ROOT / "detection" / "data" / "pretrained_weights" / "yolov8_pigs" / "yolov8_s.pth"

OUT_ROOT = W7 / "outputs" / "detector_tracker_dryrun_v27b"
FRAME_ROOT = OUT_ROOT / "annotated_sampled_frames"
CONTACT_ROOT = OUT_ROOT / "contact_sheets"

for p in [OUT_ROOT, FRAME_ROOT, CONTACT_ROOT]:
    p.mkdir(parents=True, exist_ok=True)

OUT_DETECTIONS = OUT_ROOT / "week7_detector_tracker_dryrun_v27b_detections.csv"
OUT_TRACKS = OUT_ROOT / "week7_detector_tracker_dryrun_v27b_tracks.csv"
OUT_CLIP_SUMMARY = OUT_ROOT / "week7_detector_tracker_dryrun_v27b_clip_summary.csv"
OUT_FRAME_SUMMARY = OUT_ROOT / "week7_detector_tracker_dryrun_v27b_frame_summary.csv"
OUT_CONTACT_INDEX = OUT_ROOT / "week7_detector_tracker_dryrun_v27b_contact_sheet_index.csv"
OUT_ISSUES = OUT_ROOT / "week7_detector_tracker_dryrun_v27b_issues.csv"
OUT_SUMMARY = OUT_ROOT / "week7_detector_tracker_dryrun_v27b_summary.csv"
OUT_README = OUT_ROOT / "README_detector_tracker_dryrun_v27b.md"
OUT_NOTE = W7 / "notes" / "week7_detector_tracker_dryrun_v27b_notes.md"

CONF_THRES = 0.25
IOU_TRACK_THRES = 0.35
SAMPLE_EVERY_N_FRAMES = 10
MAX_SAMPLED_FRAMES_PER_CLIP = 40


def safe_to_csv(df, path):
    df.to_csv(
        path,
        index=False,
        quoting=csv.QUOTE_ALL,
        escapechar="\\",
        lineterminator="\n",
    )


def clean(v):
    if pd.isna(v):
        return ""
    return str(v).strip()


def slug(s):
    s = clean(s)
    s = re.sub(r"[^A-Za-z0-9_.-]+", "_", s)
    s = re.sub(r"_+", "_", s).strip("_")
    return s or "unknown"


def load_font(size=14):
    try:
        return ImageFont.truetype("DejaVuSans.ttf", size)
    except Exception:
        return ImageFont.load_default()


def choose_device():
    # CPU-safe dry-run.
    # The server currently exposes CUDA, but the runtime is missing libnvrtc/cuDNN linkage.
    # For this 5-clip dry-run, CPU is safer and sufficient.
    return "cpu"


def extract_predictions(result):
    """
    Supports MMDetection 3.x DetDataSample output.
    Returns list of dicts with bbox, score, label.
    """
    preds = []

    if hasattr(result, "pred_instances"):
        inst = result.pred_instances

        bboxes = inst.bboxes.detach().cpu().numpy() if hasattr(inst, "bboxes") else np.zeros((0, 4))
        scores = inst.scores.detach().cpu().numpy() if hasattr(inst, "scores") else np.zeros((len(bboxes),))
        labels = inst.labels.detach().cpu().numpy() if hasattr(inst, "labels") else np.zeros((len(bboxes),), dtype=int)

        for bbox, score, label in zip(bboxes, scores, labels):
            score = float(score)
            if score < CONF_THRES:
                continue

            x1, y1, x2, y2 = [float(x) for x in bbox]

            preds.append({
                "x1": x1,
                "y1": y1,
                "x2": x2,
                "y2": y2,
                "score": score,
                "label": int(label),
            })

        return preds

    # Fallback for older list/tuple style outputs.
    if isinstance(result, (list, tuple)):
        for class_id, arr in enumerate(result):
            arr = np.asarray(arr)
            if arr.ndim != 2 or arr.shape[1] < 5:
                continue

            for row in arr:
                score = float(row[4])
                if score < CONF_THRES:
                    continue

                preds.append({
                    "x1": float(row[0]),
                    "y1": float(row[1]),
                    "x2": float(row[2]),
                    "y2": float(row[3]),
                    "score": score,
                    "label": int(class_id),
                })

    return preds


def bbox_iou(a, b):
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b

    ix1 = max(ax1, bx1)
    iy1 = max(ay1, by1)
    ix2 = min(ax2, bx2)
    iy2 = min(ay2, by2)

    iw = max(0.0, ix2 - ix1)
    ih = max(0.0, iy2 - iy1)
    inter = iw * ih

    area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)

    union = area_a + area_b - inter

    if union <= 0:
        return 0.0

    return inter / union


class SimpleIoUTracker:
    def __init__(self, iou_threshold=0.35, max_missing=2):
        self.iou_threshold = iou_threshold
        self.max_missing = max_missing
        self.next_track_id = 1
        self.tracks = {}

    def update(self, detections):
        """
        detections: list of dict with x1,y1,x2,y2,score,label.
        Returns detections enriched with track_id.
        """
        assigned_det = set()
        assigned_tracks = set()

        track_ids = list(self.tracks.keys())

        pairs = []
        for ti in track_ids:
            tb = self.tracks[ti]["bbox"]
            for di, det in enumerate(detections):
                db = [det["x1"], det["y1"], det["x2"], det["y2"]]
                iou = bbox_iou(tb, db)
                pairs.append((iou, ti, di))

        pairs.sort(reverse=True, key=lambda x: x[0])

        output = []

        for iou, ti, di in pairs:
            if iou < self.iou_threshold:
                continue
            if ti in assigned_tracks or di in assigned_det:
                continue

            det = detections[di].copy()
            det["track_id"] = ti
            det["track_iou"] = iou

            self.tracks[ti]["bbox"] = [det["x1"], det["y1"], det["x2"], det["y2"]]
            self.tracks[ti]["missing"] = 0

            assigned_tracks.add(ti)
            assigned_det.add(di)
            output.append(det)

        for di, det in enumerate(detections):
            if di in assigned_det:
                continue

            ti = self.next_track_id
            self.next_track_id += 1

            det = det.copy()
            det["track_id"] = ti
            det["track_iou"] = ""

            self.tracks[ti] = {
                "bbox": [det["x1"], det["y1"], det["x2"], det["y2"]],
                "missing": 0,
            }

            assigned_tracks.add(ti)
            output.append(det)

        for ti in list(self.tracks.keys()):
            if ti not in assigned_tracks:
                self.tracks[ti]["missing"] += 1

            if self.tracks[ti]["missing"] > self.max_missing:
                del self.tracks[ti]

        return output


def draw_annotations(frame_bgr, detections, title):
    img = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    pil = Image.fromarray(img)
    draw = ImageDraw.Draw(pil)

    font = load_font(14)
    small = load_font(12)

    # Title panel.
    draw.rectangle((0, 0, pil.width, 34), fill=(255, 255, 255))
    draw.text((8, 8), title, fill=(0, 0, 0), font=small)

    for det in detections:
        x1, y1, x2, y2 = [int(round(det[k])) for k in ["x1", "y1", "x2", "y2"]]
        track_id = det.get("track_id", "")
        score = det.get("score", 0.0)

        # Use deterministic visual colour by track id.
        tid = int(track_id) if track_id != "" else 0
        colour = (
            int((37 * tid) % 255),
            int((97 * tid) % 255),
            int((157 * tid) % 255),
        )

        for k in range(3):
            draw.rectangle((x1-k, y1-k, x2+k, y2+k), outline=colour)

        label = f"T{track_id} pig {score:.2f}"
        try:
            tb = draw.textbbox((x1, max(35, y1 - 20)), label, font=small)
            draw.rectangle((tb[0], tb[1], tb[2] + 4, tb[3] + 4), fill=(255, 255, 255))
        except Exception:
            pass

        draw.text((x1 + 2, max(35, y1 - 18)), label, fill=colour, font=small)

    return cv2.cvtColor(np.asarray(pil), cv2.COLOR_RGB2BGR)


def make_contact_sheet(rows, out_path, title, cols=4, thumb_w=300, thumb_h=190):
    if not rows:
        return False

    font = load_font(12)
    title_font = load_font(16)

    pad = 8
    title_h = 40
    cell_w = thumb_w + 2 * pad
    cell_h = thumb_h + 60 + 2 * pad
    sheet_rows = math.ceil(len(rows) / cols)

    sheet = Image.new("RGB", (cols * cell_w, title_h + sheet_rows * cell_h), "white")
    draw = ImageDraw.Draw(sheet)

    draw.text((pad, pad), title, fill=(0, 0, 0), font=title_font)

    for i, row in enumerate(rows):
        r = i // cols
        c = i % cols

        x0 = c * cell_w + pad
        y0 = title_h + r * cell_h + pad

        p = Path(row["annotated_frame_path"])

        try:
            im = Image.open(p).convert("RGB")
            im.thumbnail((thumb_w, thumb_h))
            bg = Image.new("RGB", (thumb_w, thumb_h), "white")
            bg.paste(im, ((thumb_w - im.width) // 2, (thumb_h - im.height) // 2))
            sheet.paste(bg, (x0, y0))
        except Exception:
            bg = Image.new("RGB", (thumb_w, thumb_h), "lightgray")
            dd = ImageDraw.Draw(bg)
            dd.text((8, 8), "LOAD ERROR", fill=(0, 0, 0), font=font)
            sheet.paste(bg, (x0, y0))

        label = (
            f"{row.get('scan_frame_id', '')} | f={row.get('sampled_frame_index', '')}\n"
            f"det={row.get('detections_in_frame', '')} tracks={row.get('active_track_ids', '')}"
        )

        draw.text((x0, y0 + thumb_h + 4), label, fill=(0, 0, 0), font=font)

    sheet.save(out_path, quality=95)
    return True


issues = []
detection_rows = []
track_rows = []
frame_summary_rows = []
clip_summary_rows = []
contact_rows = []

selected = pd.read_csv(SELECTED_CLIPS)

device = choose_device()
model = None
model_device_used = device

try:
    model = init_detector(str(CONFIG), str(CHECKPOINT), device=device)
except Exception as e:
    issues.append({
        "clip_id": "",
        "scan_frame_id": "",
        "issue_type": "model_init_failed_primary_device",
        "issue_detail": f"device={device}; {repr(e)}",
    })

    if device != "cpu":
        try:
            model = init_detector(str(CONFIG), str(CHECKPOINT), device="cpu")
            model_device_used = "cpu"
        except Exception as e2:
            issues.append({
                "clip_id": "",
                "scan_frame_id": "",
                "issue_type": "model_init_failed_cpu_fallback",
                "issue_detail": repr(e2),
            })

if model is not None:
    for _, clip_row in selected.iterrows():
        scan_frame_id = clean(clip_row.get("scan_frame_id", ""))
        video_id = clean(clip_row.get("video_id", ""))
        selection_reason = clean(clip_row.get("selection_reason", ""))
        behaviour_codes = clean(clip_row.get("behaviour_codes_present", ""))
        clip_path = Path(clean(clip_row.get("clip_path", "")))

        clip_id = f"{slug(scan_frame_id)}__{slug(selection_reason)}"

        if not clip_path.exists():
            issues.append({
                "clip_id": clip_id,
                "scan_frame_id": scan_frame_id,
                "issue_type": "clip_missing",
                "issue_detail": str(clip_path),
            })
            continue

        cap = cv2.VideoCapture(str(clip_path))

        if not cap.isOpened():
            issues.append({
                "clip_id": clip_id,
                "scan_frame_id": scan_frame_id,
                "issue_type": "clip_open_failed",
                "issue_detail": str(clip_path),
            })
            continue

        fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        if fps <= 0:
            fps = 25.0

        sampled_indices = list(range(0, total_frames, SAMPLE_EVERY_N_FRAMES))
        sampled_indices = sampled_indices[:MAX_SAMPLED_FRAMES_PER_CLIP]

        tracker = SimpleIoUTracker(iou_threshold=IOU_TRACK_THRES, max_missing=2)

        clip_detection_count = 0
        clip_sampled_count = 0
        clip_track_ids = set()

        annotated_rows_for_sheet = []

        for frame_idx in sampled_indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
            ok, frame = cap.read()

            if not ok or frame is None:
                issues.append({
                    "clip_id": clip_id,
                    "scan_frame_id": scan_frame_id,
                    "issue_type": "frame_read_failed",
                    "issue_detail": f"frame_idx={frame_idx}",
                })
                continue

            try:
                result = inference_detector(model, frame)
                detections = extract_predictions(result)
            except Exception as e:
                issues.append({
                    "clip_id": clip_id,
                    "scan_frame_id": scan_frame_id,
                    "issue_type": "inference_failed",
                    "issue_detail": f"frame_idx={frame_idx}; {repr(e)}",
                })
                continue

            tracked = tracker.update(detections)

            active_ids = sorted(set(int(x["track_id"]) for x in tracked if "track_id" in x))
            clip_track_ids.update(active_ids)

            frame_time_sec = frame_idx / float(fps)

            for det_id, det in enumerate(tracked):
                row = {
                    "clip_id": clip_id,
                    "scan_frame_id": scan_frame_id,
                    "video_id": video_id,
                    "selection_reason": selection_reason,
                    "behaviour_codes_present": behaviour_codes,
                    "clip_path": str(clip_path),
                    "sampled_frame_index": int(frame_idx),
                    "sampled_frame_time_sec": round(frame_time_sec, 4),
                    "det_id_in_frame": int(det_id),
                    "track_id": int(det["track_id"]),
                    "track_iou": det.get("track_iou", ""),
                    "label": int(det.get("label", 0)),
                    "score": round(float(det.get("score", 0.0)), 6),
                    "x1": round(float(det["x1"]), 4),
                    "y1": round(float(det["y1"]), 4),
                    "x2": round(float(det["x2"]), 4),
                    "y2": round(float(det["y2"]), 4),
                }
                detection_rows.append(row)
                track_rows.append(row)

            annotated = draw_annotations(
                frame,
                tracked,
                f"{scan_frame_id} | {selection_reason} | frame {frame_idx} | det={len(tracked)}",
            )

            clip_frame_dir = FRAME_ROOT / clip_id
            clip_frame_dir.mkdir(parents=True, exist_ok=True)

            annotated_path = clip_frame_dir / f"{clip_id}__frame_{frame_idx:06d}.jpg"
            cv2.imwrite(str(annotated_path), annotated)

            frame_summary_row = {
                "clip_id": clip_id,
                "scan_frame_id": scan_frame_id,
                "video_id": video_id,
                "selection_reason": selection_reason,
                "sampled_frame_index": int(frame_idx),
                "sampled_frame_time_sec": round(frame_time_sec, 4),
                "detections_in_frame": int(len(tracked)),
                "active_track_ids": " | ".join(map(str, active_ids)),
                "annotated_frame_path": str(annotated_path),
            }

            frame_summary_rows.append(frame_summary_row)
            annotated_rows_for_sheet.append(frame_summary_row)

            clip_detection_count += len(tracked)
            clip_sampled_count += 1

        cap.release()

        if annotated_rows_for_sheet:
            sheet_path = CONTACT_ROOT / f"{clip_id}_contact_sheet.jpg"
            made = make_contact_sheet(
                annotated_rows_for_sheet,
                sheet_path,
                f"v27b detector/tracker dry-run | {scan_frame_id}",
            )

            if made:
                contact_rows.append({
                    "clip_id": clip_id,
                    "scan_frame_id": scan_frame_id,
                    "selection_reason": selection_reason,
                    "contact_sheet_path": str(sheet_path),
                    "sampled_frames": len(annotated_rows_for_sheet),
                })

        clip_summary_rows.append({
            "clip_id": clip_id,
            "scan_frame_id": scan_frame_id,
            "video_id": video_id,
            "selection_reason": selection_reason,
            "behaviour_codes_present": behaviour_codes,
            "clip_path": str(clip_path),
            "fps": round(float(fps), 4),
            "total_frames": int(total_frames),
            "width": int(width),
            "height": int(height),
            "sampled_frames_requested": int(len(sampled_indices)),
            "sampled_frames_processed": int(clip_sampled_count),
            "detections_total": int(clip_detection_count),
            "unique_track_ids": int(len(clip_track_ids)),
            "track_ids": " | ".join(map(str, sorted(clip_track_ids))),
            "mean_detections_per_sampled_frame": round(clip_detection_count / clip_sampled_count, 4) if clip_sampled_count else 0,
            "contact_sheet_created": bool(len(annotated_rows_for_sheet) > 0),
        })


detections_df = pd.DataFrame(detection_rows)
tracks_df = pd.DataFrame(track_rows)
frame_summary = pd.DataFrame(frame_summary_rows)
clip_summary = pd.DataFrame(clip_summary_rows)
contact_index = pd.DataFrame(contact_rows)
issues_df = pd.DataFrame(issues, columns=["clip_id", "scan_frame_id", "issue_type", "issue_detail"])

safe_to_csv(detections_df, OUT_DETECTIONS)
safe_to_csv(tracks_df, OUT_TRACKS)
safe_to_csv(frame_summary, OUT_FRAME_SUMMARY)
safe_to_csv(clip_summary, OUT_CLIP_SUMMARY)
safe_to_csv(contact_index, OUT_CONTACT_INDEX)
safe_to_csv(issues_df, OUT_ISSUES)

summary = pd.DataFrame([{
    "selected_clips": int(len(selected)),
    "processed_clips": int(len(clip_summary)),
    "sample_every_n_frames": int(SAMPLE_EVERY_N_FRAMES),
    "max_sampled_frames_per_clip": int(MAX_SAMPLED_FRAMES_PER_CLIP),
    "confidence_threshold": float(CONF_THRES),
    "iou_track_threshold": float(IOU_TRACK_THRES),
    "model_config": str(CONFIG),
    "model_checkpoint": str(CHECKPOINT),
    "model_device_used": model_device_used if model is not None else "",
    "sampled_frames_processed": int(len(frame_summary)),
    "detections_total": int(len(detections_df)),
    "contact_sheets_created": int(len(contact_index)),
    "issue_count": int(len(issues_df)),
    "ready_for_visual_review": bool(len(clip_summary) > 0 and len(frame_summary) > 0),
    "ready_for_v28_full_clip_tracking": bool(len(issues_df) == 0 and len(clip_summary) == len(selected) and len(detections_df) > 0),
    "generated_at": datetime.now().isoformat(timespec="seconds"),
}])

safe_to_csv(summary, OUT_SUMMARY)

ready_v28 = bool(summary.iloc[0]["ready_for_v28_full_clip_tracking"])

OUT_README.write_text(
    "# Week 7 Detector / Tracker Dry-run v27b\n\n"
    "## Purpose\n\n"
    "This step runs YOLOv8-s pig detection and a simple IoU tracker on five representative 10-second clips only. "
    "It is a dry-run before full clip tracking.\n\n"
    "## Method\n\n"
    "- Detector: PigBench YOLOv8-s official checkpoint.\n"
    "- Sampling: every 10th frame, up to 40 sampled frames per clip.\n"
    "- Tracking: simple greedy IoU association.\n"
    "- This is not final tracking; it is a QA/preflight dry-run.\n\n"
    "## Outputs\n\n"
    "- `week7_detector_tracker_dryrun_v27b_detections.csv`\n"
    "- `week7_detector_tracker_dryrun_v27b_tracks.csv`\n"
    "- `week7_detector_tracker_dryrun_v27b_clip_summary.csv`\n"
    "- `week7_detector_tracker_dryrun_v27b_frame_summary.csv`\n"
    "- `contact_sheets/`\n"
)

OUT_NOTE.write_text(
    "# Week 7 Detector / Tracker Dry-run v27b\n\n"
    "## Purpose\n\n"
    "This step runs detector and simple IoU tracking on five representative clips before full tracking.\n\n"
    "## Summary\n\n"
    f"- Selected clips: `{int(summary.iloc[0]['selected_clips'])}`\n"
    f"- Processed clips: `{int(summary.iloc[0]['processed_clips'])}`\n"
    f"- Sampled frames processed: `{int(summary.iloc[0]['sampled_frames_processed'])}`\n"
    f"- Detections total: `{int(summary.iloc[0]['detections_total'])}`\n"
    f"- Contact sheets created: `{int(summary.iloc[0]['contact_sheets_created'])}`\n"
    f"- Issue count: `{int(summary.iloc[0]['issue_count'])}`\n"
    f"- Model device used: `{summary.iloc[0]['model_device_used']}`\n"
    f"- Ready for visual review: `{bool(summary.iloc[0]['ready_for_visual_review'])}`\n"
    f"- Ready for v28 full clip tracking: `{ready_v28}`\n\n"
    "## Outputs\n\n"
    f"- Summary: `{OUT_SUMMARY}`\n"
    f"- Clip summary: `{OUT_CLIP_SUMMARY}`\n"
    f"- Frame summary: `{OUT_FRAME_SUMMARY}`\n"
    f"- Detections: `{OUT_DETECTIONS}`\n"
    f"- Tracks: `{OUT_TRACKS}`\n"
    f"- Contact sheets: `{CONTACT_ROOT}`\n"
    f"- Issues: `{OUT_ISSUES}`\n"
)

print("Saved:")
print(OUT_DETECTIONS)
print(OUT_TRACKS)
print(OUT_CLIP_SUMMARY)
print(OUT_FRAME_SUMMARY)
print(OUT_CONTACT_INDEX)
print(OUT_ISSUES)
print(OUT_SUMMARY)
print(OUT_README)
print(OUT_NOTE)

print()
print("=== v27b detector/tracker dry-run summary ===")
print(summary.to_string(index=False))

print()
print("=== v27b clip summary ===")
if len(clip_summary):
    print(clip_summary.to_string(index=False))
else:
    print("No clip summary generated.")

print()
print("=== v27b issues ===")
if len(issues_df):
    print(issues_df.head(30).to_string(index=False))
else:
    print("No issues found.")
