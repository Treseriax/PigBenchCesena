from pathlib import Path
from datetime import datetime
import os
os.environ["CUDA_VISIBLE_DEVICES"] = ""

import csv
import json
import math
import re

import cv2
import numpy as np
import pandas as pd
from PIL import Image, ImageDraw, ImageFont

import torch
from mmdet.apis import init_detector, inference_detector


ROOT = Path.home() / "PigBench"
W7 = ROOT / "Week7_Dataset_Validation_Identity_Tracking_Behaviour_Preparation"

SELECTED_CLIPS = W7 / "outputs" / "detector_tracker_preflight_v27a" / "week7_detector_tracker_preflight_v27a_selected_dryrun_clips.csv"
POLYGONS = W7 / "outputs" / "target_pen_polygon_roi_v28b" / "week7_target_pen_polygon_roi_v28b_polygons.csv"
V28C_CLIP_SUMMARY = W7 / "outputs" / "polygon_filtered_retracking_v28c" / "week7_polygon_filtered_retracking_v28c_clip_summary.csv"

CONFIG = ROOT / "detection" / "configs" / "yolov8" / "yolov8_s.py"
CHECKPOINT = ROOT / "detection" / "data" / "pretrained_weights" / "yolov8_pigs" / "yolov8_s.pth"

OUT_ROOT = W7 / "outputs" / "dense_polygon_filtered_tracking_v28d"
FRAME_ROOT = OUT_ROOT / "annotated_dense_polygon_frames"
CONTACT_ROOT = OUT_ROOT / "contact_sheets"

for p in [OUT_ROOT, FRAME_ROOT, CONTACT_ROOT]:
    p.mkdir(parents=True, exist_ok=True)

OUT_DETECTIONS = OUT_ROOT / "week7_dense_polygon_filtered_tracking_v28d_detections.csv"
OUT_TRACKS = OUT_ROOT / "week7_dense_polygon_filtered_tracking_v28d_tracks.csv"
OUT_FRAME_SUMMARY = OUT_ROOT / "week7_dense_polygon_filtered_tracking_v28d_frame_summary.csv"
OUT_CLIP_SUMMARY = OUT_ROOT / "week7_dense_polygon_filtered_tracking_v28d_clip_summary.csv"
OUT_COMPARISON = OUT_ROOT / "week7_dense_polygon_filtered_tracking_v28d_compare_to_v28c.csv"
OUT_DECISION = OUT_ROOT / "week7_dense_polygon_filtered_tracking_v28d_decision_summary.csv"
OUT_CONTACT_INDEX = OUT_ROOT / "week7_dense_polygon_filtered_tracking_v28d_contact_sheet_index.csv"
OUT_LIMITATIONS = OUT_ROOT / "week7_dense_polygon_filtered_tracking_v28d_limitations.md"
OUT_ISSUES = OUT_ROOT / "week7_dense_polygon_filtered_tracking_v28d_issues.csv"
OUT_README = OUT_ROOT / "README_dense_polygon_filtered_tracking_v28d.md"
OUT_NOTE = W7 / "notes" / "week7_dense_polygon_filtered_tracking_v28d_notes.md"

CONF_THRES = 0.25
DENSE_SAMPLE_EVERY_N_FRAMES = 5
MAX_SAMPLED_FRAMES_PER_CLIP = 80

# Lower than v27b/v28c because denser sampling should allow smoother association.
IOU_TRACK_THRES = 0.30
MAX_MISSING = 5


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


def to_num(v, default=np.nan):
    try:
        x = pd.to_numeric(pd.Series([v]), errors="coerce").iloc[0]
        if pd.isna(x):
            return default
        return float(x)
    except Exception:
        return default


def slug(s):
    s = clean(s)
    s = re.sub(r"[^A-Za-z0-9_.-]+", "_", s)
    s = re.sub(r"_+", "_", s).strip("_")
    return s or "unknown"


def load_font(size=13):
    try:
        return ImageFont.truetype("DejaVuSans.ttf", size)
    except Exception:
        return ImageFont.load_default()


def point_in_poly(x, y, poly):
    if not poly or len(poly) < 3:
        return False

    inside = False
    j = len(poly) - 1

    for i in range(len(poly)):
        xi, yi = float(poly[i]["x"]), float(poly[i]["y"])
        xj, yj = float(poly[j]["x"]), float(poly[j]["y"])

        intersect = ((yi > y) != (yj > y)) and (
            x < (xj - xi) * (y - yi) / ((yj - yi) + 1e-12) + xi
        )

        if intersect:
            inside = not inside

        j = i

    return inside


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


class DenseIoUTracker:
    def __init__(self, iou_threshold=0.30, max_missing=5):
        self.iou_threshold = iou_threshold
        self.max_missing = max_missing
        self.next_track_id = 1
        self.tracks = {}

    def update(self, detections):
        assigned_det = set()
        assigned_tracks = set()
        output = []

        pairs = []

        for tid, tr in self.tracks.items():
            tb = tr["bbox"]

            for di, det in enumerate(detections):
                db = [det["x1"], det["y1"], det["x2"], det["y2"]]
                iou = bbox_iou(tb, db)
                pairs.append((iou, tid, di))

        pairs.sort(reverse=True, key=lambda x: x[0])

        for iou, tid, di in pairs:
            if iou < self.iou_threshold:
                continue
            if tid in assigned_tracks or di in assigned_det:
                continue

            det = detections[di].copy()
            det["dense_track_id"] = tid
            det["dense_track_iou"] = round(float(iou), 6)

            self.tracks[tid]["bbox"] = [det["x1"], det["y1"], det["x2"], det["y2"]]
            self.tracks[tid]["missing"] = 0
            self.tracks[tid]["hits"] += 1

            assigned_tracks.add(tid)
            assigned_det.add(di)
            output.append(det)

        for di, det in enumerate(detections):
            if di in assigned_det:
                continue

            tid = self.next_track_id
            self.next_track_id += 1

            det = det.copy()
            det["dense_track_id"] = tid
            det["dense_track_iou"] = ""

            self.tracks[tid] = {
                "bbox": [det["x1"], det["y1"], det["x2"], det["y2"]],
                "missing": 0,
                "hits": 1,
            }

            assigned_tracks.add(tid)
            output.append(det)

        for tid in list(self.tracks.keys()):
            if tid not in assigned_tracks:
                self.tracks[tid]["missing"] += 1

            if self.tracks[tid]["missing"] > self.max_missing:
                del self.tracks[tid]

        return output


def extract_predictions(result):
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
                "center_x": (x1 + x2) / 2.0,
                "center_y": (y1 + y2) / 2.0,
            })

    return preds


def draw_frame(frame_bgr, detections, removed_dets, polygon, title):
    rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    im = Image.fromarray(rgb)
    draw = ImageDraw.Draw(im)

    font = load_font(13)
    small = load_font(11)

    if polygon and len(polygon) >= 3:
        pts = [(float(p["x"]), float(p["y"])) for p in polygon]
        draw.line(pts + [pts[0]], fill=(0, 220, 0), width=4)

    draw.rectangle((0, 0, im.width, 44), fill=(255, 255, 255))
    draw.text((8, 8), title, fill=(0, 0, 0), font=font)

    # Removed detections as thin grey boxes for QA.
    for det in removed_dets:
        x1, y1, x2, y2 = [int(round(float(det[k]))) for k in ["x1", "y1", "x2", "y2"]]
        draw.rectangle((x1, y1, x2, y2), outline=(150, 150, 150), width=1)

    for det in detections:
        x1, y1, x2, y2 = [int(round(float(det[k]))) for k in ["x1", "y1", "x2", "y2"]]
        tid = int(det["dense_track_id"])
        score = float(det["score"])

        colour = (
            int((53 * tid) % 255),
            int((101 * tid) % 255),
            int((173 * tid) % 255),
        )

        for k in range(3):
            draw.rectangle((x1 - k, y1 - k, x2 + k, y2 + k), outline=colour)

        label = f"D-T{tid} {score:.2f}"

        try:
            tb = draw.textbbox((x1, max(46, y1 - 18)), label, font=small)
            draw.rectangle((tb[0], tb[1], tb[2] + 4, tb[3] + 4), fill=(255, 255, 255))
        except Exception:
            pass

        draw.text((x1 + 2, max(46, y1 - 17)), label, fill=colour, font=small)

    return cv2.cvtColor(np.asarray(im), cv2.COLOR_RGB2BGR)


def make_contact_sheet(rows, out_path, title, cols=4, thumb_w=300, thumb_h=190):
    if not rows:
        return False

    font = load_font(11)
    title_font = load_font(16)

    pad = 8
    title_h = 40
    cell_w = thumb_w + 2 * pad
    cell_h = thumb_h + 66 + 2 * pad
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
            f"{row.get('scan_frame_id', '')} | f={row.get('frame_index', '')}\n"
            f"kept={row.get('kept_detections', '')} removed={row.get('removed_detections', '')}\n"
            f"tracks={row.get('active_dense_track_ids', '')}"
        )

        draw.text((x0, y0 + thumb_h + 4), label, fill=(0, 0, 0), font=font)

    sheet.save(out_path, quality=95)
    return True


def classify_count(mean_kept, expected):
    if expected <= 0:
        return "not_applicable"

    ratio = mean_kept / expected

    if ratio < 0.60:
        return "possible_under_detection_dense"
    if ratio > 1.50:
        return "possible_over_detection_dense"
    if ratio > 1.25:
        return "mild_over_detection_dense"
    return "reasonable_dense"


def classify_fragmentation(unique_tracks, expected):
    if expected <= 0:
        return "not_applicable"

    ratio = unique_tracks / expected

    if ratio >= 4.0:
        return "high_fragmentation_dense"
    if ratio >= 2.5:
        return "moderate_fragmentation_dense"
    if ratio >= 1.5:
        return "mild_fragmentation_dense"
    return "low_fragmentation_dense"


issues = []
det_rows = []
track_rows = []
frame_rows = []
clip_rows = []
contact_rows = []

required = [SELECTED_CLIPS, POLYGONS, CONFIG, CHECKPOINT]

for p in required:
    if not Path(p).exists():
        issues.append({
            "clip_id": "",
            "scan_frame_id": "",
            "issue_type": "missing_required_file",
            "issue_detail": str(p),
        })

if issues:
    issues_df = pd.DataFrame(issues)
    safe_to_csv(issues_df, OUT_ISSUES)
    print(issues_df.to_string(index=False))
    raise SystemExit(1)

selected = pd.read_csv(SELECTED_CLIPS)
poly_df = pd.read_csv(POLYGONS)

for df in [selected, poly_df]:
    if "scan_frame_id" in df.columns:
        df["scan_frame_id"] = df["scan_frame_id"].fillna("").astype(str).str.strip()

polygons = {}

for _, r in poly_df.iterrows():
    sid = clean(r["scan_frame_id"])

    if clean(r.get("status", "")) != "confirmed":
        issues.append({
            "clip_id": "",
            "scan_frame_id": sid,
            "issue_type": "polygon_not_confirmed",
            "issue_detail": clean(r.get("status", "")),
        })
        continue

    try:
        polygons[sid] = json.loads(r["polygon_points_json"])
    except Exception as e:
        issues.append({
            "clip_id": "",
            "scan_frame_id": sid,
            "issue_type": "polygon_parse_failed",
            "issue_detail": repr(e),
        })

print("Loading detector on CPU...")
model = init_detector(str(CONFIG), str(CHECKPOINT), device="cpu")
print("Detector loaded.")

for _, clip_info in selected.iterrows():
    sid = clean(clip_info["scan_frame_id"])
    clip_path = Path(clean(clip_info["clip_path"]))
    clip_id = f"{slug(sid)}__dense_polygon"

    video_id = clean(clip_info.get("video_id", ""))
    selection_reason = clean(clip_info.get("selection_reason", ""))
    behaviour_codes = clean(clip_info.get("behaviour_codes_present", ""))
    expected = int(to_num(clip_info.get("pig_rows", clip_info.get("pig_rows_numeric", 0)), 0))

    polygon = polygons.get(sid, [])

    if not clip_path.exists():
        issues.append({
            "clip_id": clip_id,
            "scan_frame_id": sid,
            "issue_type": "clip_missing",
            "issue_detail": str(clip_path),
        })
        continue

    if not polygon:
        issues.append({
            "clip_id": clip_id,
            "scan_frame_id": sid,
            "issue_type": "polygon_missing",
            "issue_detail": sid,
        })
        continue

    cap = cv2.VideoCapture(str(clip_path))

    if not cap.isOpened():
        issues.append({
            "clip_id": clip_id,
            "scan_frame_id": sid,
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

    sampled_indices = list(range(0, total_frames, DENSE_SAMPLE_EVERY_N_FRAMES))
    sampled_indices = sampled_indices[:MAX_SAMPLED_FRAMES_PER_CLIP]

    tracker = DenseIoUTracker(IOU_TRACK_THRES, MAX_MISSING)

    clip_kept = 0
    clip_removed = 0
    clip_above_conf = 0
    clip_track_ids = set()
    track_frame_counts = {}

    annotated_for_sheet = []

    for frame_idx in sampled_indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(frame_idx))
        ok, frame = cap.read()

        if not ok or frame is None:
            issues.append({
                "clip_id": clip_id,
                "scan_frame_id": sid,
                "issue_type": "frame_read_failed",
                "issue_detail": f"frame_idx={frame_idx}",
            })
            continue

        try:
            result = inference_detector(model, frame)
            preds = extract_predictions(result)
        except Exception as e:
            issues.append({
                "clip_id": clip_id,
                "scan_frame_id": sid,
                "issue_type": "inference_failed",
                "issue_detail": f"frame_idx={frame_idx}; {repr(e)}",
            })
            continue

        kept = []
        removed = []

        for det in preds:
            inside = point_in_poly(det["center_x"], det["center_y"], polygon)

            row_base = {
                "clip_id": clip_id,
                "scan_frame_id": sid,
                "video_id": video_id,
                "selection_reason": selection_reason,
                "behaviour_codes_present": behaviour_codes,
                "clip_path": str(clip_path),
                "frame_index": int(frame_idx),
                "frame_time_sec": round(frame_idx / float(fps), 4),
                "score": round(float(det["score"]), 6),
                "label": int(det["label"]),
                "x1": round(float(det["x1"]), 4),
                "y1": round(float(det["y1"]), 4),
                "x2": round(float(det["x2"]), 4),
                "y2": round(float(det["y2"]), 4),
                "center_x": round(float(det["center_x"]), 4),
                "center_y": round(float(det["center_y"]), 4),
                "inside_manual_polygon_roi": bool(inside),
                "confidence_threshold": CONF_THRES,
            }

            det_rows.append(row_base)

            if inside:
                kept.append({
                    **det,
                    "clip_id": clip_id,
                    "scan_frame_id": sid,
                    "video_id": video_id,
                    "selection_reason": selection_reason,
                    "behaviour_codes_present": behaviour_codes,
                    "clip_path": str(clip_path),
                    "frame_index": int(frame_idx),
                    "frame_time_sec": round(frame_idx / float(fps), 4),
                })
            else:
                removed.append(det)

        tracked = tracker.update(kept)

        active_ids = sorted(set(int(x["dense_track_id"]) for x in tracked))
        clip_track_ids.update(active_ids)

        for tid in active_ids:
            track_frame_counts[tid] = track_frame_counts.get(tid, 0) + 1

        for det_id, tr in enumerate(tracked):
            tr_row = {
                "clip_id": clip_id,
                "scan_frame_id": sid,
                "video_id": video_id,
                "selection_reason": selection_reason,
                "behaviour_codes_present": behaviour_codes,
                "clip_path": str(clip_path),
                "frame_index": int(frame_idx),
                "frame_time_sec": round(frame_idx / float(fps), 4),
                "det_id_in_frame": int(det_id),
                "dense_track_id": int(tr["dense_track_id"]),
                "dense_track_iou": tr.get("dense_track_iou", ""),
                "label": int(tr.get("label", 0)),
                "score": round(float(tr.get("score", 0.0)), 6),
                "x1": round(float(tr["x1"]), 4),
                "y1": round(float(tr["y1"]), 4),
                "x2": round(float(tr["x2"]), 4),
                "y2": round(float(tr["y2"]), 4),
                "center_x": round(float(tr["center_x"]), 4),
                "center_y": round(float(tr["center_y"]), 4),
            }
            track_rows.append(tr_row)

        annotated = draw_frame(
            frame,
            tracked,
            removed,
            polygon,
            f"{sid} dense polygon | f={frame_idx} | kept={len(tracked)} removed={len(removed)}",
        )

        clip_frame_dir = FRAME_ROOT / slug(clip_id)
        clip_frame_dir.mkdir(parents=True, exist_ok=True)

        annotated_path = clip_frame_dir / f"{slug(clip_id)}__frame_{frame_idx:06d}_dense_polygon.jpg"
        cv2.imwrite(str(annotated_path), annotated)

        frame_row = {
            "clip_id": clip_id,
            "scan_frame_id": sid,
            "video_id": video_id,
            "selection_reason": selection_reason,
            "frame_index": int(frame_idx),
            "frame_time_sec": round(frame_idx / float(fps), 4),
            "detections_above_confidence": int(len(preds)),
            "kept_detections": int(len(tracked)),
            "removed_detections": int(len(removed)),
            "active_dense_track_ids": " | ".join(map(str, active_ids)),
            "annotated_frame_path": str(annotated_path),
        }

        frame_rows.append(frame_row)
        annotated_for_sheet.append(frame_row)

        clip_above_conf += len(preds)
        clip_kept += len(tracked)
        clip_removed += len(removed)

    cap.release()

    # Contact sheets split into chunks.
    for part_idx in range(0, len(annotated_for_sheet), 24):
        chunk = annotated_for_sheet[part_idx:part_idx + 24]
        sheet_path = CONTACT_ROOT / f"{slug(clip_id)}_dense_polygon_contact_sheet_part_{part_idx // 24 + 1:02d}.jpg"

        made = make_contact_sheet(
            chunk,
            sheet_path,
            f"v28d dense polygon tracking | {sid} | part {part_idx // 24 + 1}",
        )

        if made:
            contact_rows.append({
                "clip_id": clip_id,
                "scan_frame_id": sid,
                "selection_reason": selection_reason,
                "contact_sheet_path": str(sheet_path),
                "sampled_frames": len(chunk),
                "part": part_idx // 24 + 1,
            })

    sampled_processed = len(annotated_for_sheet)
    mean_kept = clip_kept / sampled_processed if sampled_processed else 0.0
    unique_tracks = len(clip_track_ids)

    frag_ratio = unique_tracks / expected if expected > 0 else np.nan
    count_ratio = mean_kept / expected if expected > 0 else np.nan

    lifetimes = list(track_frame_counts.values())
    median_lifetime = float(np.median(lifetimes)) if lifetimes else 0.0
    mean_lifetime = float(np.mean(lifetimes)) if lifetimes else 0.0

    long_tracks_ge_5 = sum(1 for x in lifetimes if x >= 5)
    long_tracks_ge_10 = sum(1 for x in lifetimes if x >= 10)
    long_tracks_ge_20 = sum(1 for x in lifetimes if x >= 20)

    clip_rows.append({
        "clip_id": clip_id,
        "scan_frame_id": sid,
        "video_id": video_id,
        "selection_reason": selection_reason,
        "behaviour_codes_present": behaviour_codes,
        "expected_target_pig_rows": expected,
        "fps": round(float(fps), 4),
        "total_frames": int(total_frames),
        "sample_every_n_frames": DENSE_SAMPLE_EVERY_N_FRAMES,
        "sampled_frames_processed": sampled_processed,
        "detections_above_confidence_total": int(clip_above_conf),
        "kept_detections_total": int(clip_kept),
        "removed_detections_total": int(clip_removed),
        "mean_kept_detections_per_frame": round(mean_kept, 4),
        "unique_dense_track_ids": int(unique_tracks),
        "dense_track_ids": " | ".join(map(str, sorted(clip_track_ids))),
        "fragmentation_ratio_unique_tracks_over_expected": round(frag_ratio, 4) if not pd.isna(frag_ratio) else "",
        "count_ratio_mean_kept_over_expected": round(count_ratio, 4) if not pd.isna(count_ratio) else "",
        "count_status": classify_count(mean_kept, expected),
        "fragmentation_status": classify_fragmentation(unique_tracks, expected),
        "track_lifetime_mean_frames": round(mean_lifetime, 4),
        "track_lifetime_median_frames": round(median_lifetime, 4),
        "long_tracks_ge_5_frames": int(long_tracks_ge_5),
        "long_tracks_ge_10_frames": int(long_tracks_ge_10),
        "long_tracks_ge_20_frames": int(long_tracks_ge_20),
        "contact_sheet_parts": int(math.ceil(len(annotated_for_sheet) / 24)) if annotated_for_sheet else 0,
    })

det_df = pd.DataFrame(det_rows)
tracks_df = pd.DataFrame(track_rows)
frame_summary = pd.DataFrame(frame_rows)
clip_summary = pd.DataFrame(clip_rows)
contact_index = pd.DataFrame(contact_rows)
issues_df = pd.DataFrame(issues, columns=["clip_id", "scan_frame_id", "issue_type", "issue_detail"])

safe_to_csv(det_df, OUT_DETECTIONS)
safe_to_csv(tracks_df, OUT_TRACKS)
safe_to_csv(frame_summary, OUT_FRAME_SUMMARY)
safe_to_csv(clip_summary, OUT_CLIP_SUMMARY)
safe_to_csv(contact_index, OUT_CONTACT_INDEX)
safe_to_csv(issues_df, OUT_ISSUES)

# Compare with v28c sparse polygon-filtered results.
comparison_rows = []

if V28C_CLIP_SUMMARY.exists() and len(clip_summary):
    v28c = pd.read_csv(V28C_CLIP_SUMMARY)

    for df in [v28c, clip_summary]:
        if "scan_frame_id" in df.columns:
            df["scan_frame_id"] = df["scan_frame_id"].fillna("").astype(str).str.strip()

    for _, r in clip_summary.iterrows():
        sid = clean(r["scan_frame_id"])
        old = v28c[v28c["scan_frame_id"] == sid]

        old_unique = ""
        old_mean = ""
        old_frag = ""
        old_count_status = ""
        old_frag_status = ""

        if len(old):
            old_row = old.iloc[0]
            old_unique = old_row.get("unique_polygon_track_ids", "")
            old_mean = old_row.get("mean_kept_detections_per_frame", "")
            old_frag = old_row.get("fragmentation_ratio_unique_tracks_over_expected", "")
            old_count_status = old_row.get("count_status", "")
            old_frag_status = old_row.get("fragmentation_status", "")

        comparison_rows.append({
            "scan_frame_id": sid,
            "selection_reason": clean(r["selection_reason"]),
            "expected_target_pig_rows": r["expected_target_pig_rows"],
            "v28c_sparse_mean_kept": old_mean,
            "v28d_dense_mean_kept": r["mean_kept_detections_per_frame"],
            "v28c_sparse_unique_tracks": old_unique,
            "v28d_dense_unique_tracks": r["unique_dense_track_ids"],
            "v28c_sparse_fragmentation_ratio": old_frag,
            "v28d_dense_fragmentation_ratio": r["fragmentation_ratio_unique_tracks_over_expected"],
            "v28c_count_status": old_count_status,
            "v28d_count_status": r["count_status"],
            "v28c_fragmentation_status": old_frag_status,
            "v28d_fragmentation_status": r["fragmentation_status"],
        })

comparison = pd.DataFrame(comparison_rows)
safe_to_csv(comparison, OUT_COMPARISON)

if len(clip_summary):
    mod_high_frag = int(clip_summary["fragmentation_status"].isin([
        "moderate_fragmentation_dense",
        "high_fragmentation_dense",
    ]).sum())
    high_frag = int(clip_summary["fragmentation_status"].eq("high_fragmentation_dense").sum())
    count_warn = int(~clip_summary["count_status"].eq("reasonable_dense").sum())
else:
    mod_high_frag = 0
    high_frag = 0
    count_warn = 0

decision = pd.DataFrame([{
    "v28d_decision": "dense_polygon_filtered_tracking_dryrun_completed",
    "detector": "YOLOv8-s PigBench official checkpoint",
    "device": "cpu",
    "confidence_threshold": CONF_THRES,
    "sample_every_n_frames": DENSE_SAMPLE_EVERY_N_FRAMES,
    "iou_track_threshold": IOU_TRACK_THRES,
    "max_missing": MAX_MISSING,
    "processed_clips": int(len(clip_summary)),
    "sampled_frames_processed": int(len(frame_summary)),
    "detections_above_confidence_total": int(len(det_df)),
    "kept_detections_total": int(clip_summary["kept_detections_total"].sum()) if len(clip_summary) else 0,
    "removed_detections_total": int(clip_summary["removed_detections_total"].sum()) if len(clip_summary) else 0,
    "contact_sheet_parts_created": int(len(contact_index)),
    "issue_count": int(len(issues_df)),
    "clips_with_moderate_or_high_fragmentation_dense": mod_high_frag,
    "clips_with_high_fragmentation_dense": high_frag,
    "clips_with_count_warning_dense": count_warn,
    "ready_for_visual_review": bool(len(contact_index) > 0 and len(issues_df) == 0),
    "ready_for_v29_tracker_strategy_decision": bool(len(clip_summary) > 0 and len(issues_df) == 0),
    "recommended_next_step": "visual review; if fragmentation persists in crowded clips, move to improved tracker/colour-constrained association",
    "generated_at": datetime.now().isoformat(timespec="seconds"),
}])

safe_to_csv(decision, OUT_DECISION)

OUT_LIMITATIONS.write_text(
    "# Week 7 v28d Dense Polygon-filtered Tracking Limitations\n\n"
    "## What this step does\n\n"
    "v28d re-runs detector inference more densely on the five representative clips, applies manually confirmed polygon ROI filtering, and performs baseline IoU tracking.\n\n"
    "## What this step does not do\n\n"
    "It is still not final identity tracking. It is a dense dry-run to test whether sparse sampling was the main cause of fragmentation.\n\n"
    "## Important caveats\n\n"
    "1. Simple IoU association can still fragment identities in crowded or occluded scenes.\n"
    "2. Polygon ROI reduces target-pen leakage but cannot solve occlusion or identity switches alone.\n"
    "3. If crowded clips remain fragmented, v29 should evaluate improved tracker strategies.\n"
    "4. The final identity layer should eventually use visual marker colour constraints, not tracker ID alone.\n"
)

OUT_README.write_text(
    "# Week 7 Dense Polygon-filtered Tracking v28d\n\n"
    "## Purpose\n\n"
    "This step performs a denser polygon-filtered tracking dry-run to test whether increased temporal sampling improves ID stability.\n\n"
    "## Outputs\n\n"
    "- `week7_dense_polygon_filtered_tracking_v28d_detections.csv`\n"
    "- `week7_dense_polygon_filtered_tracking_v28d_tracks.csv`\n"
    "- `week7_dense_polygon_filtered_tracking_v28d_frame_summary.csv`\n"
    "- `week7_dense_polygon_filtered_tracking_v28d_clip_summary.csv`\n"
    "- `week7_dense_polygon_filtered_tracking_v28d_compare_to_v28c.csv`\n"
    "- `contact_sheets/`\n"
)

OUT_NOTE.write_text(
    "# Week 7 Dense Polygon-filtered Tracking v28d\n\n"
    "## Purpose\n\n"
    "This step re-runs detector inference more densely, applies manual polygon ROI filtering, and performs baseline IoU tracking.\n\n"
    "## Summary\n\n"
    f"- Processed clips: `{int(decision.iloc[0]['processed_clips'])}`\n"
    f"- Sampled frames processed: `{int(decision.iloc[0]['sampled_frames_processed'])}`\n"
    f"- Kept detections: `{int(decision.iloc[0]['kept_detections_total'])}`\n"
    f"- Removed detections: `{int(decision.iloc[0]['removed_detections_total'])}`\n"
    f"- Contact sheet parts: `{int(decision.iloc[0]['contact_sheet_parts_created'])}`\n"
    f"- Issue count: `{int(decision.iloc[0]['issue_count'])}`\n"
    f"- Clips with moderate/high dense fragmentation: `{int(decision.iloc[0]['clips_with_moderate_or_high_fragmentation_dense'])}`\n"
    f"- Ready for visual review: `{bool(decision.iloc[0]['ready_for_visual_review'])}`\n"
    f"- Ready for v29 tracker strategy decision: `{bool(decision.iloc[0]['ready_for_v29_tracker_strategy_decision'])}`\n\n"
    "## Outputs\n\n"
    f"- Decision summary: `{OUT_DECISION}`\n"
    f"- Clip summary: `{OUT_CLIP_SUMMARY}`\n"
    f"- Comparison to v28c: `{OUT_COMPARISON}`\n"
    f"- Tracks: `{OUT_TRACKS}`\n"
    f"- Contact sheets: `{CONTACT_ROOT}`\n"
    f"- Issues: `{OUT_ISSUES}`\n"
)

print("Saved:")
print(OUT_DETECTIONS)
print(OUT_TRACKS)
print(OUT_FRAME_SUMMARY)
print(OUT_CLIP_SUMMARY)
print(OUT_COMPARISON)
print(OUT_DECISION)
print(OUT_CONTACT_INDEX)
print(OUT_LIMITATIONS)
print(OUT_ISSUES)
print(OUT_README)
print(OUT_NOTE)

print()
print("=== v28d decision ===")
print(decision.to_string(index=False))

print()
print("=== v28d clip summary ===")
if len(clip_summary):
    print(clip_summary.to_string(index=False))
else:
    print("No clip summary.")

print()
print("=== v28d compare to v28c ===")
if len(comparison):
    print(comparison.to_string(index=False))
else:
    print("No comparison generated.")

print()
print("=== v28d issues ===")
if len(issues_df):
    print(issues_df.to_string(index=False))
else:
    print("No issues found.")
