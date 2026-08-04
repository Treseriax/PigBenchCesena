from pathlib import Path
from datetime import datetime
import json
import csv
import math
import pandas as pd
import numpy as np


ROOT = Path.home() / "PigBench"
W8 = ROOT / "Week8_Ground_Truth_Validation_Behaviour_Dataset_Preparation"

V45_CLIP_JSON = W8 / "outputs" / "propagated_ground_truth" / "v45_label_propagation" / "week8_v45_clip_level_ground_truth.json"
V45_ANCHORS = W8 / "outputs" / "propagated_ground_truth" / "v45_label_propagation" / "week8_v45_clip_object_propagated_annotations.csv"
V51C_TRACKS = W8 / "outputs" / "v51c_nms_corrected_dense_tracking" / "week8_v51c_nms_dense_tracking_rows.csv"
V51D_CLIP_RECALL = W8 / "outputs" / "v51d_dense_tracking_qa_comparison" / "week8_v51d_clip_recall_by_source.csv"

OUT = W8 / "outputs" / "v52a3_constrained_detector_linked_pilot"
GALLERY = OUT / "constrained_detector_linked_gallery"
REPORTS = W8 / "reports"
NOTES = W8 / "notes"
PROGRESS = W8 / "progress"

for p in [OUT, GALLERY, REPORTS, NOTES, PROGRESS]:
    p.mkdir(parents=True, exist_ok=True)

OUT_TRACKS = OUT / "week8_v52a3_constrained_detector_linked_rows.csv"
OUT_CLIP_SUMMARY = OUT / "week8_v52a3_clip_summary.csv"
OUT_OBJECT_SUMMARY = OUT / "week8_v52a3_object_summary.csv"
OUT_GALLERY_INDEX = OUT / "week8_v52a3_gallery_index.csv"
OUT_GALLERY_HTML = OUT / "index.html"
OUT_DECISION = OUT / "week8_v52a3_decision_summary.csv"
OUT_ISSUES = OUT / "week8_v52a3_issues.csv"
OUT_REPORT = REPORTS / "week8_v52a3_constrained_detector_linked_pilot_report.md"
OUT_NOTE = NOTES / "week8_v52a3_constrained_detector_linked_pilot_notes.md"
OUT_PROGRESS = PROGRESS / "week8_experiment_progress_log.csv"


PARAMS = {
    "max_center_dist": 72.0,
    "min_iou": 0.06,
    "max_gap": 10,
    "min_area_ratio": 0.35,
    "max_area_ratio": 3.20,
    "track_id_bonus": 0.55,
    "colour_bonus_weight": 0.85,
}


def safe_to_csv(df, path):
    df.to_csv(path, index=False, quoting=csv.QUOTE_ALL, escapechar="\\", lineterminator="\n")


def clean_str(x):
    if pd.isna(x):
        return ""
    s = str(x).strip()
    if s.lower() in ["nan", "none", "null"]:
        return ""
    return s


def to_float(x, default=None):
    try:
        if pd.isna(x):
            return default
        return float(x)
    except Exception:
        return default


def to_int(x, default=None):
    try:
        if pd.isna(x):
            return default
        return int(round(float(x)))
    except Exception:
        return default


def iou_xyxy(a, b):
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
    denom = area_a + area_b - inter
    if denom <= 0:
        return 0.0
    return inter / denom


def center_distance(a, b):
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    acx = (ax1 + ax2) / 2.0
    acy = (ay1 + ay2) / 2.0
    bcx = (bx1 + bx2) / 2.0
    bcy = (by1 + by2) / 2.0
    return math.sqrt((acx - bcx) ** 2 + (acy - bcy) ** 2)


def area_xyxy(b):
    x1, y1, x2, y2 = b
    return max(1.0, (x2 - x1) * (y2 - y1))


def get_center(b):
    x1, y1, x2, y2 = b
    return (x1 + x2) / 2.0, (y1 + y2) / 2.0


def shift_box(b, vx, vy):
    x1, y1, x2, y2 = b
    return [x1 + vx, y1 + vy, x2 + vx, y2 + vy]


def get_anchor_frame(clip):
    fps = to_float(clip.get("fps_used"), 25.0) or 25.0
    duration = to_float(clip.get("duration_sec"), 10.0) or 10.0
    scan = clean_str(clip.get("scan_frame_id"))
    if scan in ["scanframe_0000", "scanframe_0006", "scanframe_0012", "scanframe_0018"]:
        anchor_rel_sec = 0.0
    else:
        anchor_rel_sec = duration / 2.0
    return int(round(anchor_rel_sec * fps)), anchor_rel_sec


def bbox_from_anchor(row):
    return [float(row["bbox_x1"]), float(row["bbox_y1"]), float(row["bbox_x2"]), float(row["bbox_y2"])]


def bbox_from_det(row):
    return [float(row["x1"]), float(row["y1"]), float(row["x2"]), float(row["y2"])]


def read_all_frames(video_path):
    import cv2
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        cap.release()
        return [], {}
    fps = float(cap.get(cv2.CAP_PROP_FPS))
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    frames = []
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        frames.append(frame)
    cap.release()
    return frames, {"fps": fps, "frame_count": frame_count, "width": width, "height": height, "frames_read": len(frames)}


def marker_colour_score(frame, box, colour):
    import cv2
    colour = clean_str(colour).lower()
    if colour in ["", "unknown", "no_color", "not_visible", "uncertain", "unassigned"]:
        return 0.0

    h, w = frame.shape[:2]
    x1, y1, x2, y2 = box
    x1 = max(0, min(w - 1, int(round(x1))))
    y1 = max(0, min(h - 1, int(round(y1))))
    x2 = max(0, min(w - 1, int(round(x2))))
    y2 = max(0, min(h - 1, int(round(y2))))
    if x2 <= x1 or y2 <= y1:
        return 0.0

    crop = frame[y1:y2, x1:x2]
    if crop.size == 0:
        return 0.0

    hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
    H = hsv[:, :, 0]
    S = hsv[:, :, 1]
    V = hsv[:, :, 2]
    sat = S > 45
    bright = V > 45

    if colour == "red":
        mask = ((H <= 12) | (H >= 168)) & sat & bright
    elif colour == "pink":
        mask = (H >= 145) & (H <= 175) & sat & bright
    elif colour == "purple":
        mask = (H >= 125) & (H <= 160) & sat & bright
    elif colour == "blue":
        mask = (H >= 95) & (H <= 130) & sat & bright
    elif colour == "green":
        mask = (H >= 35) & (H <= 85) & sat & bright
    elif colour == "cyan":
        mask = (H >= 80) & (H <= 105) & sat & bright
    else:
        return 0.0

    return min(1.0, float(mask.mean()) * 12.0)


def choose_matches(states, candidates, frame, params):
    proposals = []

    for obj_id, st in states.items():
        if st["dead"]:
            continue

        pred_box = shift_box(st["bbox"], st["vx"], st["vy"])
        prev_area = area_xyxy(st["bbox"])
        target_colour = st["visual_marker_colour"]
        previous_track = st["source_detector_track_id"]

        for ci, cand in candidates.iterrows():
            cbox = bbox_from_det(cand)
            val_iou = iou_xyxy(pred_box, cbox)
            dist = center_distance(pred_box, cbox)
            area_ratio = area_xyxy(cbox) / prev_area

            if area_ratio < params["min_area_ratio"] or area_ratio > params["max_area_ratio"]:
                continue

            det_track = clean_str(cand.get("track_id"))
            same_track = previous_track != "" and det_track == previous_track
            colour_score = marker_colour_score(frame, cbox, target_colour)
            area_penalty = abs(math.log(max(0.05, min(20.0, area_ratio))))

            eligible = val_iou >= params["min_iou"] or dist <= params["max_center_dist"] or same_track
            if not eligible:
                continue

            score = (
                4.20 * val_iou
                - 0.013 * dist
                - 0.38 * area_penalty
                + 0.25 * float(cand.get("score", 0.0))
                + params["colour_bonus_weight"] * colour_score
                + (params["track_id_bonus"] if same_track else 0.0)
            )

            proposals.append({
                "obj_id": obj_id,
                "candidate_index": ci,
                "score": score,
                "iou": val_iou,
                "center_distance": dist,
                "area_ratio": area_ratio,
                "colour_score": colour_score,
            })

    proposals = sorted(proposals, key=lambda x: x["score"], reverse=True)
    assigned_obj = set()
    assigned_cand = set()
    assignments = {}

    for p in proposals:
        if p["obj_id"] in assigned_obj or p["candidate_index"] in assigned_cand:
            continue
        assigned_obj.add(p["obj_id"])
        assigned_cand.add(p["candidate_index"])
        assignments[p["obj_id"]] = p

    return assignments


def init_states(anchors_df):
    states = {}
    for _, a in anchors_df.iterrows():
        obj_id = clean_str(a.get("final_box_id"))
        states[obj_id] = {
            "bbox": bbox_from_anchor(a),
            "vx": 0.0,
            "vy": 0.0,
            "gap": 0,
            "dead": False,
            "anchor": a.to_dict(),
            "source_detector_track_id": "",
            "visual_marker_colour": clean_str(a.get("visual_marker_colour")),
        }
    return states


def make_row(scan, clip, obj_id, st, fi, direction, status, draw_ok, match=None, det_row=None):
    a = st["anchor"]
    box = st["bbox"]
    if match is None:
        match = {}

    return {
        "dataset_version": "week8_v52a3_constrained_detector_linked_pilot",
        "scan_frame_id": scan,
        "video_id": clean_str(clip.get("video_id")),
        "clip_path": clean_str(clip.get("clip_path")),
        "frame_index_in_clip": int(fi),
        "final_box_id": obj_id,
        "behaviour_pig_id": clean_str(a.get("behaviour_pig_id")),
        "visual_marker_colour": clean_str(a.get("visual_marker_colour")),
        "behaviour_code": clean_str(a.get("behaviour_code")),
        "anchor_frame_index": int(st.get("anchor_frame_index", -1)),
        "tracking_direction": direction,
        "x1": float(box[0]),
        "y1": float(box[1]),
        "x2": float(box[2]),
        "y2": float(box[3]),
        "trajectory_status": status,
        "draw_ok": bool(draw_ok),
        "missed_count": int(st["gap"]),
        "source_detector_track_id": clean_str(det_row.get("track_id")) if det_row is not None else st.get("source_detector_track_id", ""),
        "source_detector_score": to_float(det_row.get("score")) if det_row is not None else None,
        "match_iou": float(match.get("iou", 0.0)),
        "match_center_distance": match.get("center_distance"),
        "area_ratio_vs_previous": float(match.get("area_ratio", 1.0)),
        "marker_colour_score": float(match.get("colour_score", 0.0)),
        "method": "gt_seeded_detector_linked_constrained",
    }


def propagate(scan, clip, anchors_df, det_df, frames, meta, direction, params):
    anchor_frame, _ = get_anchor_frame(clip)
    anchor_frame = max(0, min(meta["frames_read"] - 1, anchor_frame))
    states = init_states(anchors_df)

    for st in states.values():
        st["anchor_frame_index"] = anchor_frame

    if direction == "forward":
        frame_indices = list(range(anchor_frame + 1, meta["frames_read"]))
    else:
        frame_indices = list(range(anchor_frame - 1, -1, -1))

    det_by_frame = {int(k): v.copy() for k, v in det_df.groupby("frame_index_in_clip")} if len(det_df) else {}
    rows = []

    for fi in frame_indices:
        cand = det_by_frame.get(fi, pd.DataFrame())
        assignments = choose_matches(states, cand, frames[fi], params) if len(cand) else {}

        for obj_id, st in states.items():
            if st["dead"]:
                continue

            if obj_id in assignments:
                m = assignments[obj_id]
                drow = cand.loc[m["candidate_index"]]
                old_cx, old_cy = get_center(st["bbox"])
                new_box = bbox_from_det(drow)
                new_cx, new_cy = get_center(new_box)

                st["vx"] = 0.65 * st["vx"] + 0.35 * (new_cx - old_cx)
                st["vy"] = 0.65 * st["vy"] + 0.35 * (new_cy - old_cy)
                st["bbox"] = new_box
                st["gap"] = 0
                st["source_detector_track_id"] = clean_str(drow.get("track_id"))

                rows.append(make_row(scan, clip, obj_id, st, fi, direction, "detector_linked", True, m, drow))
            else:
                st["gap"] += 1
                st["bbox"] = shift_box(st["bbox"], st["vx"], st["vy"])

                if st["gap"] <= params["max_gap"]:
                    rows.append(make_row(scan, clip, obj_id, st, fi, direction, "short_gap_predicted", True))
                else:
                    rows.append(make_row(scan, clip, obj_id, st, fi, direction, "missing_long_gap", False))

    return rows


def build_tracks(scan, clip, anchors_df, det_df, frames, meta, params):
    anchor_frame, _ = get_anchor_frame(clip)
    anchor_frame = max(0, min(meta["frames_read"] - 1, anchor_frame))
    rows = []

    for _, a in anchors_df.iterrows():
        box = bbox_from_anchor(a)
        rows.append({
            "dataset_version": "week8_v52a3_constrained_detector_linked_pilot",
            "scan_frame_id": scan,
            "video_id": clean_str(clip.get("video_id")),
            "clip_path": clean_str(clip.get("clip_path")),
            "frame_index_in_clip": int(anchor_frame),
            "final_box_id": clean_str(a.get("final_box_id")),
            "behaviour_pig_id": clean_str(a.get("behaviour_pig_id")),
            "visual_marker_colour": clean_str(a.get("visual_marker_colour")),
            "behaviour_code": clean_str(a.get("behaviour_code")),
            "anchor_frame_index": int(anchor_frame),
            "tracking_direction": "anchor",
            "x1": float(box[0]),
            "y1": float(box[1]),
            "x2": float(box[2]),
            "y2": float(box[3]),
            "trajectory_status": "anchor_gt",
            "draw_ok": True,
            "missed_count": 0,
            "source_detector_track_id": "",
            "source_detector_score": None,
            "match_iou": 1.0,
            "match_center_distance": 0.0,
            "area_ratio_vs_previous": 1.0,
            "marker_colour_score": 1.0,
            "method": "gt_seeded_detector_linked_constrained",
        })

    rows.extend(propagate(scan, clip, anchors_df, det_df, frames, meta, "forward", params))
    rows.extend(propagate(scan, clip, anchors_df, det_df, frames, meta, "backward", params))
    return rows


def draw_gallery(scan, clip, tracks_df, frames, out_path):
    import cv2
    if len(frames) == 0:
        return False, "no_frames"

    sample_frames = sorted(set([0, len(frames) // 2, len(frames) - 1]))
    panels = []

    colour_map = {
        "blue": (255, 80, 30),
        "green": (60, 220, 60),
        "cyan": (255, 220, 0),
        "red": (40, 40, 255),
        "pink": (220, 80, 255),
        "purple": (180, 70, 220),
        "unknown": (180, 180, 180),
        "not_visible": (130, 130, 130),
        "uncertain": (0, 220, 255),
        "unassigned": (160, 160, 160),
    }

    for fi in sample_frames:
        img = frames[fi].copy()
        h, w = img.shape[:2]
        g = tracks_df[(tracks_df["frame_index_in_clip"] == fi) & (tracks_df["draw_ok"] == True)].copy()

        for _, r in g.iterrows():
            x1 = to_int(r.get("x1"))
            y1 = to_int(r.get("y1"))
            x2 = to_int(r.get("x2"))
            y2 = to_int(r.get("y2"))

            if None in [x1, y1, x2, y2]:
                continue

            colour = clean_str(r.get("visual_marker_colour")) or "unknown"
            bgr = colour_map.get(colour, (255, 255, 255))
            status = clean_str(r.get("trajectory_status"))
            thickness = 3 if status == "anchor_gt" else 2 if status == "detector_linked" else 1

            cv2.rectangle(img, (x1, y1), (x2, y2), bgr, thickness)
            label = f"{clean_str(r.get('behaviour_pig_id'))}/{colour}/{status}"
            cv2.putText(img, label, (x1, max(18, y1 - 6)), cv2.FONT_HERSHEY_SIMPLEX, 0.38, bgr, 1)

        cv2.rectangle(img, (0, 0), (w, 55), (0, 0, 0), -1)
        cv2.putText(img, f"{scan} constrained GT-seeded detector-linked frame={fi}", (10, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.58, (255, 255, 255), 2)
        cv2.putText(img, "stricter gates + detector track consistency + marker colour score", (10, 47), cv2.FONT_HERSHEY_SIMPLEX, 0.40, (230, 230, 230), 1)

        panels.append(img)

    combined = np.hstack(panels)
    ok = cv2.imwrite(str(out_path), combined)
    return bool(ok), "saved" if ok else "write_failed"


issues = []

for p in [V45_CLIP_JSON, V45_ANCHORS, V51C_TRACKS]:
    if not p.exists():
        issues.append({
            "item": str(p),
            "issue_type": "hard_missing_required_input",
            "issue_detail": "Required input missing.",
            "severity": "hard",
        })

if issues:
    issues_df = pd.DataFrame(issues)
    safe_to_csv(issues_df, OUT_ISSUES)
    decision = pd.DataFrame([{
        "v52a3_decision": "constrained_detector_linked_pilot_blocked_missing_inputs",
        "hard_issue_count": int((issues_df["severity"] == "hard").sum()),
        "ready_for_v52b_full_constrained_detector_linked_tracking": False,
        "generated_at": datetime.now().isoformat(timespec="seconds"),
    }])
    safe_to_csv(decision, OUT_DECISION)
    print(decision.to_string(index=False))
    raise SystemExit(1)


clips = json.loads(V45_CLIP_JSON.read_text()).get("clips", [])
clip_map = {clean_str(c.get("scan_frame_id")): c for c in clips}

anchors = pd.read_csv(V45_ANCHORS)
det = pd.read_csv(V51C_TRACKS)

anchors["scan_frame_id"] = anchors["scan_frame_id"].astype(str).map(clean_str)
det["scan_frame_id"] = det["scan_frame_id"].astype(str).map(clean_str)

for c in ["bbox_x1", "bbox_y1", "bbox_x2", "bbox_y2"]:
    anchors[c] = pd.to_numeric(anchors[c], errors="coerce")

for c in ["frame_index_in_clip", "x1", "y1", "x2", "y2", "score", "video_frame_count"]:
    if c in det.columns:
        det[c] = pd.to_numeric(det[c], errors="coerce")

anchor_by_scan = {str(k): v.copy() for k, v in anchors.groupby("scan_frame_id")}
det_by_scan = {str(k): v.copy() for k, v in det.groupby("scan_frame_id")}

selected_scans = []

if V51D_CLIP_RECALL.exists():
    cr = pd.read_csv(V51D_CLIP_RECALL)
    nms = cr[cr["source_name"] == "v51c_nms_corrected_dense"].copy()
    hard = nms.sort_values("strong_moderate_recall").head(10)["scan_frame_id"].tolist()
    for s in hard:
        if s not in selected_scans:
            selected_scans.append(s)

for s in list(clip_map.keys())[:6]:
    if s not in selected_scans:
        selected_scans.append(s)

selected_scans = selected_scans[:14]

all_rows = []
clip_summary_rows = []
object_summary_rows = []
gallery_rows = []

for idx, scan in enumerate(selected_scans, start=1):
    print(f"[{idx}/{len(selected_scans)}] {scan}")

    clip = clip_map.get(scan)
    an = anchor_by_scan.get(scan, pd.DataFrame())
    dg = det_by_scan.get(scan, pd.DataFrame())

    if clip is None or len(an) == 0:
        issues.append({
            "item": scan,
            "issue_type": "warning_missing_clip_or_anchor",
            "issue_detail": "Missing clip metadata or anchors.",
            "severity": "warning",
        })
        continue

    frames, meta = read_all_frames(Path(clean_str(clip.get("clip_path"))))

    if len(frames) == 0:
        issues.append({
            "item": scan,
            "issue_type": "warning_video_read_failed",
            "issue_detail": "Could not read clip frames.",
            "severity": "warning",
        })
        continue

    rows = build_tracks(scan, clip, an, dg, frames, meta, PARAMS)
    clip_df = pd.DataFrame(rows)
    all_rows.extend(rows)

    frame_count = int(clip_df["frame_index_in_clip"].nunique()) if len(clip_df) else 0
    expected_rows = int(len(an) * frame_count)
    actual_rows = int(len(clip_df))
    detector_rows = int((clip_df["trajectory_status"] == "detector_linked").sum())
    carried_rows = int((clip_df["trajectory_status"] == "short_gap_predicted").sum())
    missing_rows = int((clip_df["trajectory_status"] == "missing_long_gap").sum())
    anchor_rows = int((clip_df["trajectory_status"] == "anchor_gt").sum())
    draw_rows = int((clip_df["draw_ok"] == True).sum())

    clip_summary_rows.append({
        "scan_frame_id": scan,
        "video_id": clean_str(clip.get("video_id")),
        "anchor_object_count": int(len(an)),
        "frame_count_linked": frame_count,
        "expected_object_frame_rows": expected_rows,
        "actual_object_frame_rows": actual_rows,
        "anchor_rows": anchor_rows,
        "detector_linked_rows": detector_rows,
        "short_gap_predicted_rows": carried_rows,
        "missing_long_gap_rows": missing_rows,
        "draw_ok_rows": draw_rows,
        "detector_linked_ratio": float(detector_rows / max(1, actual_rows)),
        "missing_ratio": float(missing_rows / max(1, actual_rows)),
        "draw_ok_ratio": float(draw_rows / max(1, actual_rows)),
        "status": "processed",
    })

    for obj_id, g in clip_df.groupby("final_box_id"):
        total = len(g)

        object_summary_rows.append({
            "scan_frame_id": scan,
            "final_box_id": obj_id,
            "behaviour_pig_id": clean_str(g["behaviour_pig_id"].iloc[0]),
            "visual_marker_colour": clean_str(g["visual_marker_colour"].iloc[0]),
            "behaviour_code": clean_str(g["behaviour_code"].iloc[0]),
            "rows_total": int(total),
            "detector_linked_rows": int((g["trajectory_status"] == "detector_linked").sum()),
            "short_gap_predicted_rows": int((g["trajectory_status"] == "short_gap_predicted").sum()),
            "missing_long_gap_rows": int((g["trajectory_status"] == "missing_long_gap").sum()),
            "draw_ok_rows": int((g["draw_ok"] == True).sum()),
            "detector_linked_ratio": float((g["trajectory_status"] == "detector_linked").sum() / max(1, total)),
            "missing_ratio": float((g["trajectory_status"] == "missing_long_gap").sum() / max(1, total)),
        })

    out_img = GALLERY / f"{scan}_constrained_detector_linked.jpg"
    ok, msg = draw_gallery(scan, clip, clip_df, frames, out_img)

    gallery_rows.append({
        "scan_frame_id": scan,
        "overlay_path": str(out_img),
        "saved": bool(ok),
        "message": msg,
        "anchor_object_count": int(len(an)),
        "detector_linked_ratio": float(detector_rows / max(1, actual_rows)),
        "missing_ratio": float(missing_rows / max(1, actual_rows)),
        "draw_ok_ratio": float(draw_rows / max(1, actual_rows)),
    })


tracks_df = pd.DataFrame(all_rows)
clip_summary = pd.DataFrame(clip_summary_rows)
object_summary = pd.DataFrame(object_summary_rows)
gallery_index = pd.DataFrame(gallery_rows)

safe_to_csv(tracks_df, OUT_TRACKS)
safe_to_csv(clip_summary, OUT_CLIP_SUMMARY)
safe_to_csv(object_summary, OUT_OBJECT_SUMMARY)
safe_to_csv(gallery_index, OUT_GALLERY_INDEX)

html = []
html.append("<!doctype html><html><head><meta charset='utf-8'>")
html.append("<title>Week 8 v52a3 Constrained Detector-linked Pilot</title>")
html.append("<style>body{font-family:Arial;background:#111;color:#eee;margin:20px}.grid{display:grid;grid-template-columns:repeat(auto-fill,minmax(760px,1fr));gap:18px}.card{background:#1e1e1e;border:1px solid #333;border-radius:10px;padding:12px}img{width:100%;border:1px solid #444;border-radius:6px}.meta{font-size:13px;color:#bbb;line-height:1.4}.top{background:#2b2b2b;padding:12px;border-radius:10px;margin-bottom:18px}</style></head><body>")
html.append("<div class='top'><h1>Week 8 v52a3 Constrained GT-seeded Detector-linked Tracking Pilot</h1>")
html.append("<p>Stricter association with motion prediction, detector-track consistency and marker-colour score. Long gaps are not drawn.</p></div><div class='grid'>")

for _, r in gallery_index.iterrows():
    if not bool(r["saved"]):
        continue

    rel = Path(r["overlay_path"]).relative_to(OUT)
    html.append("<div class='card'>")
    html.append(f"<h3>{r['scan_frame_id']}</h3><img src='{rel.as_posix()}'>")
    html.append("<div class='meta'>")
    html.append(f"anchor objects: {r['anchor_object_count']}<br>")
    html.append(f"detector linked ratio: {float(r['detector_linked_ratio']):.3f}<br>")
    html.append(f"missing ratio: {float(r['missing_ratio']):.3f}<br>")
    html.append(f"draw ok ratio: {float(r['draw_ok_ratio']):.3f}<br>")
    html.append("</div></div>")

html.append("</div></body></html>")
OUT_GALLERY_HTML.write_text("\n".join(html))

pilot_clip_count = int(len(clip_summary))
pilot_object_count = int(len(object_summary))
tracking_rows_total = int(len(tracks_df))
mean_detector_linked_ratio = float(clip_summary["detector_linked_ratio"].mean()) if len(clip_summary) else 0.0
mean_missing_ratio = float(clip_summary["missing_ratio"].mean()) if len(clip_summary) else 1.0
mean_draw_ok_ratio = float(clip_summary["draw_ok_ratio"].mean()) if len(clip_summary) else 0.0

if tracking_rows_total == 0:
    issues.append({
        "item": "constrained_detector_linked_rows",
        "issue_type": "hard_no_rows",
        "issue_detail": "No constrained linked rows produced.",
        "severity": "hard",
    })

if mean_missing_ratio > 0.35:
    issues.append({
        "item": "constrained_detector_linked_missing",
        "issue_type": "warning_high_missing_ratio",
        "issue_detail": f"Mean missing ratio is {mean_missing_ratio:.4f}.",
        "severity": "warning",
    })

issues_df = pd.DataFrame(issues, columns=["item", "issue_type", "issue_detail", "severity"])
hard_issues = issues_df[issues_df["severity"] == "hard"] if len(issues_df) else pd.DataFrame()
warnings = issues_df[issues_df["severity"] == "warning"] if len(issues_df) else pd.DataFrame()
ready = len(hard_issues) == 0

decision = pd.DataFrame([{
    "v52a3_decision": "constrained_detector_linked_pilot_completed" if ready else "constrained_detector_linked_pilot_blocked",
    "pilot_clip_count": pilot_clip_count,
    "pilot_object_count": pilot_object_count,
    "tracking_rows_total": tracking_rows_total,
    "mean_detector_linked_ratio": mean_detector_linked_ratio,
    "mean_missing_ratio": mean_missing_ratio,
    "mean_draw_ok_ratio": mean_draw_ok_ratio,
    "gallery_overlay_count": int(len(gallery_index)),
    "max_center_dist": PARAMS["max_center_dist"],
    "min_iou": PARAMS["min_iou"],
    "max_gap": PARAMS["max_gap"],
    "min_area_ratio": PARAMS["min_area_ratio"],
    "max_area_ratio": PARAMS["max_area_ratio"],
    "hard_issue_count": int(len(hard_issues)),
    "warning_count": int(len(warnings)),
    "issue_count": int(len(issues_df)),
    "ready_for_v52b_full_constrained_detector_linked_tracking": bool(ready),
    "generated_at": datetime.now().isoformat(timespec="seconds"),
}])

safe_to_csv(issues_df, OUT_ISSUES)
safe_to_csv(decision, OUT_DECISION)

OUT_REPORT.write_text(
    "Week 8 v52a3 Constrained Detector-linked Tracking Pilot Report\n\n"
    f"Decision: {decision.iloc[0]['v52a3_decision']}\n"
    f"Pilot clips: {pilot_clip_count}\n"
    f"Pilot objects: {pilot_object_count}\n"
    f"Tracking rows total: {tracking_rows_total}\n"
    f"Mean detector-linked ratio: {mean_detector_linked_ratio:.4f}\n"
    f"Mean missing ratio: {mean_missing_ratio:.4f}\n"
    f"Mean draw-ok ratio: {mean_draw_ok_ratio:.4f}\n"
    f"Hard issues: {len(hard_issues)}\n"
    f"Warnings: {len(warnings)}\n\n"
    "Interpretation:\n"
    "This pilot constrains v52a2 association using motion prediction, detector-track continuity, marker-colour evidence and stricter gates.\n"
)

OUT_NOTE.write_text(
    "# Week 8 v52a3 Constrained GT-seeded Detector-linked Tracking Pilot\n\n"
    "## Summary\n\n"
    f"- v52a3 decision: {decision.iloc[0]['v52a3_decision']}\n"
    f"- Pilot clips: {pilot_clip_count}\n"
    f"- Pilot objects: {pilot_object_count}\n"
    f"- Tracking rows total: {tracking_rows_total}\n"
    f"- Mean detector-linked ratio: {mean_detector_linked_ratio:.4f}\n"
    f"- Mean missing ratio: {mean_missing_ratio:.4f}\n"
    f"- Mean draw-ok ratio: {mean_draw_ok_ratio:.4f}\n"
    f"- Hard issue count: {len(hard_issues)}\n"
    f"- Warning count: {len(warnings)}\n"
    f"- Ready for v52b full constrained detector-linked tracking: {ready}\n\n"
    "## Interpretation\n\n"
    "v52a2 improved coverage but still showed wrong links and drift. v52a3 uses stricter association and colour/track continuity constraints.\n\n"
    f"Gallery: {OUT_GALLERY_HTML}\n"
)

progress_row = pd.DataFrame([{
    "date": datetime.now().date().isoformat(),
    "stage": "v52a3",
    "task_name": "Constrained GT-seeded detector-linked tracking pilot",
    "status": "PASS" if ready else "BLOCKED",
    "input_summary": str(V51C_TRACKS),
    "output_summary": str(OUT),
    "hard_issues": int(len(hard_issues)),
    "warnings": int(len(warnings)),
    "next_action": "v52b full constrained detector-linked tracking" if ready else "Review v52a3 issues.",
}])

if OUT_PROGRESS.exists():
    old = pd.read_csv(OUT_PROGRESS)
    progress = pd.concat([old, progress_row], ignore_index=True)
else:
    progress = progress_row

safe_to_csv(progress, OUT_PROGRESS)

print("Saved:")
print(OUT_TRACKS)
print(OUT_CLIP_SUMMARY)
print(OUT_OBJECT_SUMMARY)
print(OUT_GALLERY_INDEX)
print(OUT_GALLERY_HTML)
print(OUT_DECISION)
print(OUT_ISSUES)
print(OUT_REPORT)
print(OUT_NOTE)

print()
print("=== v52a3 decision ===")
print(decision.to_string(index=False))

print()
print("=== v52a3 issues ===")
if len(issues_df):
    print(issues_df.to_string(index=False))
else:
    print("No issues found.")

print()
print("=== v52a3 clip summary ===")
print(clip_summary.to_string(index=False))
