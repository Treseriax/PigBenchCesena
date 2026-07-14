from pathlib import Path
from datetime import datetime
import csv
import json
import math
import re

import cv2
import numpy as np
import pandas as pd
from PIL import Image, ImageDraw, ImageFont


ROOT = Path.home() / "PigBench"
W7 = ROOT / "Week7_Dataset_Validation_Identity_Tracking_Behaviour_Preparation"

V27B_ROOT = W7 / "outputs" / "detector_tracker_dryrun_v27b"
V28B_ROOT = W7 / "outputs" / "target_pen_polygon_roi_v28b"

V27B_DETECTIONS = V27B_ROOT / "week7_detector_tracker_dryrun_v27b_detections.csv"
V27B_CLIP_SUMMARY = V27B_ROOT / "week7_detector_tracker_dryrun_v27b_clip_summary.csv"
V28B_POLYGONS = V28B_ROOT / "week7_target_pen_polygon_roi_v28b_polygons.csv"

OUT_ROOT = W7 / "outputs" / "polygon_filtered_retracking_v28c"
FRAME_ROOT = OUT_ROOT / "annotated_polygon_filtered_frames"
CONTACT_ROOT = OUT_ROOT / "contact_sheets"

for p in [OUT_ROOT, FRAME_ROOT, CONTACT_ROOT]:
    p.mkdir(parents=True, exist_ok=True)

OUT_FILTERED_DET = OUT_ROOT / "week7_polygon_filtered_retracking_v28c_filtered_detections.csv"
OUT_TRACKS = OUT_ROOT / "week7_polygon_filtered_retracking_v28c_tracks.csv"
OUT_FRAME_SUMMARY = OUT_ROOT / "week7_polygon_filtered_retracking_v28c_frame_summary.csv"
OUT_CLIP_SUMMARY = OUT_ROOT / "week7_polygon_filtered_retracking_v28c_clip_summary.csv"
OUT_DECISION = OUT_ROOT / "week7_polygon_filtered_retracking_v28c_decision_summary.csv"
OUT_CONTACT_INDEX = OUT_ROOT / "week7_polygon_filtered_retracking_v28c_contact_sheet_index.csv"
OUT_LIMITATIONS = OUT_ROOT / "week7_polygon_filtered_retracking_v28c_limitations.md"
OUT_ISSUES = OUT_ROOT / "week7_polygon_filtered_retracking_v28c_issues.csv"
OUT_README = OUT_ROOT / "README_polygon_filtered_retracking_v28c.md"
OUT_NOTE = W7 / "notes" / "week7_polygon_filtered_retracking_v28c_notes.md"

CONF_THRES = 0.25
IOU_TRACK_THRES = 0.35


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
    """
    Ray casting point-in-polygon.
    poly: list of dicts [{"x":..., "y":...}]
    """
    if not poly or len(poly) < 3:
        return False

    inside = False
    n = len(poly)
    j = n - 1

    for i in range(n):
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


class SimpleIoUTracker:
    def __init__(self, iou_threshold=0.35, max_missing=2):
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
            det["track_id_polygon"] = tid
            det["track_iou_polygon"] = iou

            self.tracks[tid]["bbox"] = [det["x1"], det["y1"], det["x2"], det["y2"]]
            self.tracks[tid]["missing"] = 0

            assigned_tracks.add(tid)
            assigned_det.add(di)
            output.append(det)

        for di, det in enumerate(detections):
            if di in assigned_det:
                continue

            tid = self.next_track_id
            self.next_track_id += 1

            det = det.copy()
            det["track_id_polygon"] = tid
            det["track_iou_polygon"] = ""

            self.tracks[tid] = {
                "bbox": [det["x1"], det["y1"], det["x2"], det["y2"]],
                "missing": 0,
            }

            assigned_tracks.add(tid)
            output.append(det)

        for tid in list(self.tracks.keys()):
            if tid not in assigned_tracks:
                self.tracks[tid]["missing"] += 1

            if self.tracks[tid]["missing"] > self.max_missing:
                del self.tracks[tid]

        return output


def draw_frame(frame_bgr, detections, polygon, title):
    rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    im = Image.fromarray(rgb)
    draw = ImageDraw.Draw(im)

    font = load_font(13)
    small = load_font(11)

    # Polygon.
    if polygon and len(polygon) >= 3:
        pts = [(float(p["x"]), float(p["y"])) for p in polygon]
        draw.line(pts + [pts[0]], fill=(0, 220, 0), width=4)

    # Title.
    draw.rectangle((0, 0, im.width, 42), fill=(255, 255, 255))
    draw.text((8, 8), title, fill=(0, 0, 0), font=font)

    for det in detections:
        x1, y1, x2, y2 = [int(round(float(det[k]))) for k in ["x1", "y1", "x2", "y2"]]
        tid = int(det["track_id_polygon"])
        score = float(det["score"])

        colour = (
            int((41 * tid) % 255),
            int((113 * tid) % 255),
            int((177 * tid) % 255),
        )

        for k in range(3):
            draw.rectangle((x1-k, y1-k, x2+k, y2+k), outline=colour)

        label = f"P-T{tid} {score:.2f}"

        try:
            tb = draw.textbbox((x1, max(44, y1 - 18)), label, font=small)
            draw.rectangle((tb[0], tb[1], tb[2] + 4, tb[3] + 4), fill=(255, 255, 255))
        except Exception:
            pass

        draw.text((x1 + 2, max(44, y1 - 17)), label, fill=colour, font=small)

    return cv2.cvtColor(np.asarray(im), cv2.COLOR_RGB2BGR)


def make_contact_sheet(rows, out_path, title, cols=4, thumb_w=300, thumb_h=190):
    if not rows:
        return False

    font = load_font(11)
    title_font = load_font(16)

    pad = 8
    title_h = 40
    cell_w = thumb_w + 2 * pad
    cell_h = thumb_h + 62 + 2 * pad
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
            f"kept={row.get('kept_detections_in_frame', '')} removed={row.get('removed_detections_in_frame', '')}\n"
            f"tracks={row.get('active_polygon_track_ids', '')}"
        )

        draw.text((x0, y0 + thumb_h + 4), label, fill=(0, 0, 0), font=font)

    sheet.save(out_path, quality=95)
    return True


issues = []
filtered_rows = []
track_rows = []
frame_rows = []
clip_rows = []
contact_rows = []

required = [V27B_DETECTIONS, V27B_CLIP_SUMMARY, V28B_POLYGONS]

for p in required:
    if not p.exists():
        issues.append({
            "clip_id": "",
            "scan_frame_id": "",
            "issue_type": "missing_required_file",
            "issue_detail": str(p),
        })

if issues:
    issues_df = pd.DataFrame(issues)
    safe_to_csv(issues_df, OUT_ISSUES)
    print("Missing required files.")
    print(issues_df.to_string(index=False))
    raise SystemExit(1)

det = pd.read_csv(V27B_DETECTIONS)
clip_summary = pd.read_csv(V27B_CLIP_SUMMARY)
poly_df = pd.read_csv(V28B_POLYGONS)

for df in [det, clip_summary, poly_df]:
    if "scan_frame_id" in df.columns:
        df["scan_frame_id"] = df["scan_frame_id"].fillna("").astype(str).str.strip()
    if "clip_id" in df.columns:
        df["clip_id"] = df["clip_id"].fillna("").astype(str).str.strip()

for c in ["x1", "y1", "x2", "y2", "score", "sampled_frame_index"]:
    if c in det.columns:
        det[c] = pd.to_numeric(det[c], errors="coerce")

det["center_x"] = (det["x1"] + det["x2"]) / 2.0
det["center_y"] = (det["y1"] + det["y2"]) / 2.0

# Load confirmed polygons.
polygons = {}

for _, r in poly_df.iterrows():
    sid = clean(r["scan_frame_id"])
    status = clean(r.get("status", ""))

    if status != "confirmed":
        issues.append({
            "clip_id": "",
            "scan_frame_id": sid,
            "issue_type": "polygon_not_confirmed",
            "issue_detail": f"status={status}",
        })
        continue

    try:
        pts = json.loads(r["polygon_points_json"])
    except Exception as e:
        issues.append({
            "clip_id": "",
            "scan_frame_id": sid,
            "issue_type": "polygon_json_parse_failed",
            "issue_detail": repr(e),
        })
        continue

    if len(pts) < 3:
        issues.append({
            "clip_id": "",
            "scan_frame_id": sid,
            "issue_type": "polygon_too_few_points",
            "issue_detail": str(pts),
        })
        continue

    polygons[sid] = pts

# Filter detections.
det["inside_manual_polygon_roi"] = False

for idx, r in det.iterrows():
    sid = clean(r["scan_frame_id"])
    poly = polygons.get(sid)

    if not poly:
        continue

    det.at[idx, "inside_manual_polygon_roi"] = point_in_poly(float(r["center_x"]), float(r["center_y"]), poly)

det["passes_confidence"] = det["score"] >= CONF_THRES
det["passes_polygon_filter"] = det["passes_confidence"] & det["inside_manual_polygon_roi"]

# Save detection-level filter table.
det_out = det.copy()
safe_to_csv(det_out, OUT_FILTERED_DET)

# Re-track per clip using only filtered detections.
for clip_id, cg in det.groupby("clip_id", sort=True):
    cg = cg.sort_values("sampled_frame_index").copy()

    first = cg.iloc[0]
    sid = clean(first["scan_frame_id"])
    video_id = clean(first.get("video_id", ""))
    selection_reason = clean(first.get("selection_reason", ""))
    behaviour_codes = clean(first.get("behaviour_codes_present", ""))
    clip_path = Path(clean(first.get("clip_path", "")))
    poly = polygons.get(sid, [])

    if not clip_path.exists():
        issues.append({
            "clip_id": clip_id,
            "scan_frame_id": sid,
            "issue_type": "clip_missing",
            "issue_detail": str(clip_path),
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
    if fps <= 0:
        fps = 25.0

    tracker = SimpleIoUTracker(IOU_TRACK_THRES, max_missing=2)

    clip_kept = 0
    clip_removed = 0
    clip_tracks = set()
    clip_frame_count = 0
    annotated_for_sheet = []

    for frame_idx, fg in cg.groupby("sampled_frame_index", sort=True):
        frame_idx = int(frame_idx)

        above_conf = fg[fg["passes_confidence"] == True].copy()
        kept = above_conf[above_conf["passes_polygon_filter"] == True].copy()
        removed = above_conf[above_conf["passes_polygon_filter"] == False].copy()

        dets_for_tracker = []

        for _, rr in kept.iterrows():
            dets_for_tracker.append({
                "clip_id": clip_id,
                "scan_frame_id": sid,
                "video_id": video_id,
                "selection_reason": selection_reason,
                "behaviour_codes_present": behaviour_codes,
                "clip_path": str(clip_path),
                "sampled_frame_index": frame_idx,
                "sampled_frame_time_sec": round(frame_idx / float(fps), 4),
                "original_track_id_v27b": clean(rr.get("track_id", "")),
                "label": int(to_num(rr.get("label", 0), 0)),
                "score": float(rr["score"]),
                "x1": float(rr["x1"]),
                "y1": float(rr["y1"]),
                "x2": float(rr["x2"]),
                "y2": float(rr["y2"]),
                "center_x": float(rr["center_x"]),
                "center_y": float(rr["center_y"]),
            })

        tracked = tracker.update(dets_for_tracker)

        active_ids = sorted(set(int(x["track_id_polygon"]) for x in tracked if "track_id_polygon" in x))
        clip_tracks.update(active_ids)

        for det_id, tr in enumerate(tracked):
            row = tr.copy()
            row["det_id_in_frame_after_polygon"] = det_id
            row["confidence_threshold"] = CONF_THRES
            row["iou_track_threshold"] = IOU_TRACK_THRES
            filtered_rows.append(row)
            track_rows.append(row)

        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ok, frame = cap.read()

        if ok and frame is not None:
            annotated = draw_frame(
                frame,
                tracked,
                poly,
                f"{sid} polygon-filtered | f={frame_idx} | kept={len(tracked)} removed={len(removed)}",
            )

            clip_dir = FRAME_ROOT / slug(clip_id)
            clip_dir.mkdir(parents=True, exist_ok=True)

            annotated_path = clip_dir / f"{slug(clip_id)}__frame_{frame_idx:06d}_polygon_filtered.jpg"
            cv2.imwrite(str(annotated_path), annotated)

            frame_row = {
                "clip_id": clip_id,
                "scan_frame_id": sid,
                "video_id": video_id,
                "selection_reason": selection_reason,
                "sampled_frame_index": frame_idx,
                "sampled_frame_time_sec": round(frame_idx / float(fps), 4),
                "detections_above_confidence": int(len(above_conf)),
                "kept_detections_in_frame": int(len(tracked)),
                "removed_detections_in_frame": int(len(removed)),
                "active_polygon_track_ids": " | ".join(map(str, active_ids)),
                "annotated_frame_path": str(annotated_path),
            }

            frame_rows.append(frame_row)
            annotated_for_sheet.append(frame_row)
        else:
            issues.append({
                "clip_id": clip_id,
                "scan_frame_id": sid,
                "issue_type": "frame_read_failed",
                "issue_detail": f"frame_idx={frame_idx}",
            })

        clip_kept += len(tracked)
        clip_removed += len(removed)
        clip_frame_count += 1

    cap.release()

    if annotated_for_sheet:
        sheet_path = CONTACT_ROOT / f"{slug(clip_id)}_polygon_filtered_contact_sheet.jpg"
        made = make_contact_sheet(
            annotated_for_sheet,
            sheet_path,
            f"v28c polygon-filtered re-tracking | {sid}",
        )

        if made:
            contact_rows.append({
                "clip_id": clip_id,
                "scan_frame_id": sid,
                "selection_reason": selection_reason,
                "contact_sheet_path": str(sheet_path),
                "sampled_frames": len(annotated_for_sheet),
            })

    expected = 0
    match = clip_summary[clip_summary["clip_id"] == clip_id]

    if len(match):
        # Original v27b does not include expected pigs. Use estimated mean from selected if unavailable.
        # We use behaviour context only for QA, not final correctness.
        expected = 0

    # Derive expected target pig rows from v28b polygons table.
    poly_match = poly_df[poly_df["scan_frame_id"] == sid]
    if len(poly_match):
        expected = int(to_num(poly_match.iloc[0].get("expected_pig_rows", 0), 0))

    mean_kept = clip_kept / clip_frame_count if clip_frame_count else 0.0
    frag_ratio = len(clip_tracks) / expected if expected > 0 else np.nan
    count_ratio = mean_kept / expected if expected > 0 else np.nan

    if expected <= 0:
        count_status = "not_applicable"
    elif count_ratio < 0.60:
        count_status = "possible_under_detection_after_polygon_filter"
    elif count_ratio > 1.50:
        count_status = "possible_over_detection_after_polygon_filter"
    elif count_ratio > 1.25:
        count_status = "mild_over_detection_after_polygon_filter"
    else:
        count_status = "reasonable_after_polygon_filter"

    if expected <= 0:
        frag_status = "not_applicable"
    elif frag_ratio >= 4.0:
        frag_status = "high_fragmentation_after_polygon_filter"
    elif frag_ratio >= 2.5:
        frag_status = "moderate_fragmentation_after_polygon_filter"
    elif frag_ratio >= 1.5:
        frag_status = "mild_fragmentation_after_polygon_filter"
    else:
        frag_status = "low_fragmentation_after_polygon_filter"

    clip_rows.append({
        "clip_id": clip_id,
        "scan_frame_id": sid,
        "video_id": video_id,
        "selection_reason": selection_reason,
        "behaviour_codes_present": behaviour_codes,
        "expected_target_pig_rows": expected,
        "sampled_frames_processed": clip_frame_count,
        "kept_detections_total": int(clip_kept),
        "removed_detections_total": int(clip_removed),
        "mean_kept_detections_per_frame": round(mean_kept, 4),
        "unique_polygon_track_ids": int(len(clip_tracks)),
        "polygon_track_ids": " | ".join(map(str, sorted(clip_tracks))),
        "fragmentation_ratio_unique_tracks_over_expected": round(frag_ratio, 4) if not pd.isna(frag_ratio) else "",
        "count_ratio_mean_kept_over_expected": round(count_ratio, 4) if not pd.isna(count_ratio) else "",
        "count_status": count_status,
        "fragmentation_status": frag_status,
        "contact_sheet_created": bool(len(annotated_for_sheet) > 0),
    })

filtered_df = pd.DataFrame(filtered_rows)
tracks_df = pd.DataFrame(track_rows)
frame_summary = pd.DataFrame(frame_rows)
clip_summary_out = pd.DataFrame(clip_rows)
contact_index = pd.DataFrame(contact_rows)
issues_df = pd.DataFrame(issues, columns=["clip_id", "scan_frame_id", "issue_type", "issue_detail"])

safe_to_csv(filtered_df, OUT_TRACKS)
safe_to_csv(tracks_df, OUT_TRACKS)
safe_to_csv(frame_summary, OUT_FRAME_SUMMARY)
safe_to_csv(clip_summary_out, OUT_CLIP_SUMMARY)
safe_to_csv(contact_index, OUT_CONTACT_INDEX)
safe_to_csv(issues_df, OUT_ISSUES)

if len(clip_summary_out):
    moderate_high_frag = int(clip_summary_out["fragmentation_status"].isin([
        "moderate_fragmentation_after_polygon_filter",
        "high_fragmentation_after_polygon_filter",
    ]).sum())

    over_count = int(clip_summary_out["count_status"].isin([
        "possible_over_detection_after_polygon_filter",
        "mild_over_detection_after_polygon_filter",
    ]).sum())

    under_count = int(clip_summary_out["count_status"].eq("possible_under_detection_after_polygon_filter").sum())
else:
    moderate_high_frag = 0
    over_count = 0
    under_count = 0

decision = pd.DataFrame([{
    "v28c_decision": "polygon_filter_reduces_leakage_but_tracking_quality_requires_review",
    "confidence_threshold": CONF_THRES,
    "iou_track_threshold": IOU_TRACK_THRES,
    "processed_clips": int(len(clip_summary_out)),
    "sampled_frames_processed": int(len(frame_summary)),
    "kept_detections_total": int(len(filtered_df)),
    "removed_detections_total": int(frame_summary["removed_detections_in_frame"].sum()) if len(frame_summary) else 0,
    "contact_sheets_created": int(len(contact_index)),
    "issue_count": int(len(issues_df)),
    "clips_with_moderate_or_high_fragmentation_after_polygon": moderate_high_frag,
    "clips_with_over_count_warning_after_polygon": over_count,
    "clips_with_under_count_warning_after_polygon": under_count,
    "ready_for_visual_review": bool(len(contact_index) > 0 and len(issues_df) == 0),
    "ready_for_v28d_dense_polygon_filtered_tracking": bool(len(issues_df) == 0 and len(clip_summary_out) > 0),
    "recommended_next_step": "visual review of polygon-filtered contact sheets, then dense polygon-filtered tracking if leakage is acceptable",
    "generated_at": datetime.now().isoformat(timespec="seconds"),
}])

safe_to_csv(decision, OUT_DECISION)

OUT_LIMITATIONS.write_text(
    "# Week 7 v28c Polygon-filtered Re-tracking Limitations\n\n"
    "## What this step does\n\n"
    "v28c applies manually confirmed polygon ROI filtering to v27b detector outputs and re-runs simple IoU association on the filtered detections.\n\n"
    "## What this step does not do\n\n"
    "v28c does not re-run detector inference on every frame and does not claim final stable identity tracking.\n\n"
    "## Important caveats\n\n"
    "1. The detections are inherited from v27b sampled frames.\n"
    "2. This is still sparse-frame tracking, not dense video tracking.\n"
    "3. The simple IoU tracker remains a baseline.\n"
    "4. Polygon filtering should reduce target-pen leakage, but tracking stability still needs visual review.\n"
    "5. v28d should run denser polygon-filtered tracking if v28c contact sheets look acceptable.\n"
)

OUT_README.write_text(
    "# Week 7 Polygon-filtered Re-tracking v28c\n\n"
    "## Purpose\n\n"
    "This step applies manually confirmed polygon ROI filtering to v27b detections and re-runs simple IoU tracking.\n\n"
    "## Outputs\n\n"
    "- `week7_polygon_filtered_retracking_v28c_filtered_detections.csv`\n"
    "- `week7_polygon_filtered_retracking_v28c_tracks.csv`\n"
    "- `week7_polygon_filtered_retracking_v28c_frame_summary.csv`\n"
    "- `week7_polygon_filtered_retracking_v28c_clip_summary.csv`\n"
    "- `contact_sheets/`\n"
    "- `week7_polygon_filtered_retracking_v28c_decision_summary.csv`\n"
)

OUT_NOTE.write_text(
    "# Week 7 Polygon-filtered Re-tracking v28c\n\n"
    "## Purpose\n\n"
    "This step filters v27b detections using v28b confirmed polygon ROIs and re-runs simple IoU tracking.\n\n"
    "## Summary\n\n"
    f"- Processed clips: `{int(decision.iloc[0]['processed_clips'])}`\n"
    f"- Sampled frames processed: `{int(decision.iloc[0]['sampled_frames_processed'])}`\n"
    f"- Kept detections total: `{int(decision.iloc[0]['kept_detections_total'])}`\n"
    f"- Removed detections total: `{int(decision.iloc[0]['removed_detections_total'])}`\n"
    f"- Contact sheets created: `{int(decision.iloc[0]['contact_sheets_created'])}`\n"
    f"- Issue count: `{int(decision.iloc[0]['issue_count'])}`\n"
    f"- Ready for visual review: `{bool(decision.iloc[0]['ready_for_visual_review'])}`\n"
    f"- Ready for v28d dense polygon-filtered tracking: `{bool(decision.iloc[0]['ready_for_v28d_dense_polygon_filtered_tracking'])}`\n\n"
    "## Outputs\n\n"
    f"- Decision summary: `{OUT_DECISION}`\n"
    f"- Clip summary: `{OUT_CLIP_SUMMARY}`\n"
    f"- Frame summary: `{OUT_FRAME_SUMMARY}`\n"
    f"- Tracks: `{OUT_TRACKS}`\n"
    f"- Contact sheets: `{CONTACT_ROOT}`\n"
    f"- Issues: `{OUT_ISSUES}`\n"
)

print("Saved:")
print(OUT_FILTERED_DET)
print(OUT_TRACKS)
print(OUT_FRAME_SUMMARY)
print(OUT_CLIP_SUMMARY)
print(OUT_DECISION)
print(OUT_CONTACT_INDEX)
print(OUT_LIMITATIONS)
print(OUT_ISSUES)
print(OUT_README)
print(OUT_NOTE)

print()
print("=== v28c decision ===")
print(decision.to_string(index=False))

print()
print("=== v28c clip summary ===")
if len(clip_summary_out):
    print(clip_summary_out.to_string(index=False))
else:
    print("No clip summary.")

print()
print("=== v28c issues ===")
if len(issues_df):
    print(issues_df.to_string(index=False))
else:
    print("No issues found.")
