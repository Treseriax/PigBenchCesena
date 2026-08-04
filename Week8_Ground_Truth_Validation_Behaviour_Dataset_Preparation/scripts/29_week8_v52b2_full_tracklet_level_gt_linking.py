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

OUT = W8 / "outputs" / "v52b2_tracklet_level_gt_linking_full"
GALLERY = OUT / "tracklet_level_gt_linking_gallery"
REPORTS = W8 / "reports"
NOTES = W8 / "notes"
PROGRESS = W8 / "progress"

for p in [OUT, GALLERY, REPORTS, NOTES, PROGRESS]:
    p.mkdir(parents=True, exist_ok=True)

OUT_ROWS = OUT / "week8_v52b2_tracklet_level_gt_linking_rows.csv"
OUT_CLIP_SUMMARY = OUT / "week8_v52b2_clip_summary.csv"
OUT_OBJECT_SUMMARY = OUT / "week8_v52b2_object_summary.csv"
OUT_ANCHOR_MATCHES = OUT / "week8_v52b2_anchor_tracklet_matches.csv"
OUT_GALLERY_INDEX = OUT / "week8_v52b2_gallery_index.csv"
OUT_GALLERY_HTML = OUT / "index.html"
OUT_DECISION = OUT / "week8_v52b2_decision_summary.csv"
OUT_ISSUES = OUT / "week8_v52b2_issues.csv"
OUT_REPORT = REPORTS / "week8_v52b2_tracklet_level_gt_linking_full_report.md"
OUT_NOTE = NOTES / "week8_v52b2_tracklet_level_gt_linking_full_notes.md"
OUT_PROGRESS = PROGRESS / "week8_experiment_progress_log.csv"


PARAMS = {
    "anchor_window": 12,
    "min_anchor_iou": 0.08,
    "max_anchor_center_dist": 115.0,
    "max_short_gap": 18,
    "min_tracklet_length": 4,
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


def get_anchor_frame(clip):
    fps = to_float(clip.get("fps_used"), 25.0) or 25.0
    duration = to_float(clip.get("duration_sec"), 10.0) or 10.0
    scan = clean_str(clip.get("scan_frame_id"))
    if scan in ["scanframe_0000", "scanframe_0006", "scanframe_0012", "scanframe_0018"]:
        anchor_rel_sec = 0.0
    else:
        anchor_rel_sec = duration / 2.0
    return int(round(anchor_rel_sec * fps)), anchor_rel_sec


def bbox_anchor(row):
    return [float(row["bbox_x1"]), float(row["bbox_y1"]), float(row["bbox_x2"]), float(row["bbox_y2"])]


def bbox_track(row):
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


def find_best_tracklet_for_anchor(anchor_box, anchor_frame, det_df, used_track_ids, params):
    if len(det_df) == 0:
        return None

    w = params["anchor_window"]
    near = det_df[
        (det_df["frame_index_in_clip"] >= anchor_frame - w)
        & (det_df["frame_index_in_clip"] <= anchor_frame + w)
    ].copy()

    if len(near) == 0:
        return None

    proposals = []

    for track_id, g in near.groupby("track_id"):
        track_id = clean_str(track_id)
        if track_id in used_track_ids:
            continue

        full_g = det_df[det_df["track_id"].astype(str).map(clean_str) == track_id].copy()
        if len(full_g) < params["min_tracklet_length"]:
            continue

        best_iou = 0.0
        best_dist = 999999.0
        best_frame_delta = 999999
        best_score = -999999.0
        best_row = None

        for _, r in g.iterrows():
            box = bbox_track(r)
            val_iou = iou_xyxy(anchor_box, box)
            dist = center_distance(anchor_box, box)
            frame_delta = abs(int(r["frame_index_in_clip"]) - int(anchor_frame))

            if val_iou < params["min_anchor_iou"] and dist > params["max_anchor_center_dist"]:
                continue

            score = 4.0 * val_iou - 0.012 * dist - 0.025 * frame_delta + 0.20 * float(r.get("score", 0.0))

            if score > best_score:
                best_score = score
                best_iou = val_iou
                best_dist = dist
                best_frame_delta = frame_delta
                best_row = r

        if best_row is not None:
            proposals.append({
                "track_id": track_id,
                "best_score": best_score,
                "best_iou": best_iou,
                "best_center_distance": best_dist,
                "best_frame_delta": best_frame_delta,
                "tracklet_length": len(full_g),
                "tracklet_start": int(full_g["frame_index_in_clip"].min()),
                "tracklet_end": int(full_g["frame_index_in_clip"].max()),
            })

    if not proposals:
        return None

    proposals = sorted(proposals, key=lambda x: x["best_score"], reverse=True)
    return proposals[0]


def interpolate_box(prev_box, next_box, alpha):
    return [
        prev_box[0] * (1 - alpha) + next_box[0] * alpha,
        prev_box[1] * (1 - alpha) + next_box[1] * alpha,
        prev_box[2] * (1 - alpha) + next_box[2] * alpha,
        prev_box[3] * (1 - alpha) + next_box[3] * alpha,
    ]


def build_rows_for_object(scan, clip, anchor_obj, anchor_frame, frame_count, det_df, chosen_track_id, match_info, params):
    anchor_box = bbox_anchor(anchor_obj)
    obj_id = clean_str(anchor_obj.get("final_box_id"))

    base = {
        "dataset_version": "week8_v52b2_tracklet_level_gt_linking_full",
        "scan_frame_id": scan,
        "video_id": clean_str(clip.get("video_id")),
        "clip_path": clean_str(clip.get("clip_path")),
        "final_box_id": obj_id,
        "behaviour_pig_id": clean_str(anchor_obj.get("behaviour_pig_id")),
        "visual_marker_colour": clean_str(anchor_obj.get("visual_marker_colour")),
        "behaviour_code": clean_str(anchor_obj.get("behaviour_code")),
        "anchor_frame_index": int(anchor_frame),
        "selected_tracklet_id": chosen_track_id if chosen_track_id is not None else "",
        "method": "gt_seeded_tracklet_level_linking",
    }

    rows = []

    if chosen_track_id is None:
        for fi in range(frame_count):
            draw = fi == anchor_frame
            box = anchor_box
            rows.append({
                **base,
                "frame_index_in_clip": int(fi),
                "x1": box[0], "y1": box[1], "x2": box[2], "y2": box[3],
                "trajectory_status": "anchor_only" if draw else "missing_no_tracklet",
                "draw_ok": bool(draw),
                "source_detector_track_id": "",
                "source_detector_score": None,
                "anchor_match_iou": None,
                "anchor_match_center_distance": None,
                "anchor_match_frame_delta": None,
            })
        return rows

    tg = det_df[det_df["track_id"].astype(str).map(clean_str) == clean_str(chosen_track_id)].copy()
    tg = tg.sort_values("frame_index_in_clip")

    by_frame = {}
    for _, r in tg.iterrows():
        by_frame[int(r["frame_index_in_clip"])] = r

    available_frames = sorted(by_frame.keys())

    for fi in range(frame_count):
        if fi == anchor_frame:
            box = anchor_box
            rows.append({
                **base,
                "frame_index_in_clip": int(fi),
                "x1": box[0], "y1": box[1], "x2": box[2], "y2": box[3],
                "trajectory_status": "anchor_gt",
                "draw_ok": True,
                "source_detector_track_id": clean_str(chosen_track_id),
                "source_detector_score": None,
                "anchor_match_iou": match_info["best_iou"],
                "anchor_match_center_distance": match_info["best_center_distance"],
                "anchor_match_frame_delta": match_info["best_frame_delta"],
            })
            continue

        if fi in by_frame:
            r = by_frame[fi]
            box = bbox_track(r)
            rows.append({
                **base,
                "frame_index_in_clip": int(fi),
                "x1": box[0], "y1": box[1], "x2": box[2], "y2": box[3],
                "trajectory_status": "tracklet_detection",
                "draw_ok": True,
                "source_detector_track_id": clean_str(chosen_track_id),
                "source_detector_score": to_float(r.get("score")),
                "anchor_match_iou": match_info["best_iou"],
                "anchor_match_center_distance": match_info["best_center_distance"],
                "anchor_match_frame_delta": match_info["best_frame_delta"],
            })
            continue

        prev_frames = [x for x in available_frames if x < fi]
        next_frames = [x for x in available_frames if x > fi]

        if prev_frames and next_frames:
            pf = max(prev_frames)
            nf = min(next_frames)
            gap = nf - pf

            if gap <= params["max_short_gap"]:
                pbox = bbox_track(by_frame[pf])
                nbox = bbox_track(by_frame[nf])
                alpha = (fi - pf) / max(1, gap)
                box = interpolate_box(pbox, nbox, alpha)
                status = "short_gap_interpolated"
                draw = True
            else:
                box = anchor_box
                status = "missing_tracklet_gap"
                draw = False

        else:
            box = anchor_box
            status = "missing_outside_tracklet_span"
            draw = False

        rows.append({
            **base,
            "frame_index_in_clip": int(fi),
            "x1": box[0], "y1": box[1], "x2": box[2], "y2": box[3],
            "trajectory_status": status,
            "draw_ok": bool(draw),
            "source_detector_track_id": clean_str(chosen_track_id),
            "source_detector_score": None,
            "anchor_match_iou": match_info["best_iou"],
            "anchor_match_center_distance": match_info["best_center_distance"],
            "anchor_match_frame_delta": match_info["best_frame_delta"],
        })

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
            thickness = 3 if status == "anchor_gt" else 2 if status == "tracklet_detection" else 1

            cv2.rectangle(img, (x1, y1), (x2, y2), bgr, thickness)
            label = f"{clean_str(r.get('behaviour_pig_id'))}/{colour}/{status}"
            cv2.putText(img, label, (x1, max(18, y1 - 6)), cv2.FONT_HERSHEY_SIMPLEX, 0.38, bgr, 1)

        cv2.rectangle(img, (0, 0), (w, 55), (0, 0, 0), -1)
        cv2.putText(img, f"{scan} tracklet-level GT linking frame={fi}", (10, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.60, (255, 255, 255), 2)
        cv2.putText(img, "anchor GT selects stable detector tracklet; no frame-by-frame identity jumping", (10, 47), cv2.FONT_HERSHEY_SIMPLEX, 0.40, (230, 230, 230), 1)
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
        "v52b2_decision": "tracklet_level_gt_linking_full_blocked_missing_inputs",
        "hard_issue_count": int((issues_df["severity"] == "hard").sum()),
        "ready_for_v52b_full_tracklet_level_linking": False,
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
det["track_id"] = det["track_id"].astype(str).map(clean_str)

for c in ["bbox_x1", "bbox_y1", "bbox_x2", "bbox_y2"]:
    anchors[c] = pd.to_numeric(anchors[c], errors="coerce")

for c in ["frame_index_in_clip", "x1", "y1", "x2", "y2", "score", "video_frame_count"]:
    if c in det.columns:
        det[c] = pd.to_numeric(det[c], errors="coerce")

anchor_by_scan = {str(k): v.copy() for k, v in anchors.groupby("scan_frame_id")}
det_by_scan = {str(k): v.copy() for k, v in det.groupby("scan_frame_id")}

selected_scans = list(clip_map.keys())

all_rows = []
anchor_match_rows = []
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
            "issue_detail": "Could not read frames.",
            "severity": "warning",
        })
        continue

    frame_count = len(frames)
    anchor_frame, _ = get_anchor_frame(clip)
    anchor_frame = max(0, min(frame_count - 1, anchor_frame))

    used_track_ids = set()
    clip_rows = []

    anchor_candidates = []

    for _, a in an.iterrows():
        abox = bbox_anchor(a)
        best = find_best_tracklet_for_anchor(abox, anchor_frame, dg, used_track_ids, PARAMS)

        if best is not None:
            anchor_candidates.append((best["best_score"], clean_str(a.get("final_box_id")), a, best))
        else:
            anchor_candidates.append((-999999.0, clean_str(a.get("final_box_id")), a, None))

    anchor_candidates = sorted(anchor_candidates, key=lambda x: x[0], reverse=True)

    for _, obj_id, a, best in anchor_candidates:
        if best is not None and best["track_id"] not in used_track_ids:
            chosen_track_id = best["track_id"]
            used_track_ids.add(chosen_track_id)
            match_info = best
            match_status = "matched_tracklet"
        else:
            chosen_track_id = None
            match_info = {
                "best_iou": None,
                "best_center_distance": None,
                "best_frame_delta": None,
                "tracklet_length": 0,
                "best_score": None,
                "track_id": "",
                "tracklet_start": None,
                "tracklet_end": None,
            }
            match_status = "no_unique_tracklet"

        anchor_match_rows.append({
            "scan_frame_id": scan,
            "final_box_id": clean_str(a.get("final_box_id")),
            "behaviour_pig_id": clean_str(a.get("behaviour_pig_id")),
            "visual_marker_colour": clean_str(a.get("visual_marker_colour")),
            "behaviour_code": clean_str(a.get("behaviour_code")),
            "chosen_tracklet_id": chosen_track_id if chosen_track_id is not None else "",
            "match_status": match_status,
            "best_score": match_info.get("best_score"),
            "best_iou": match_info.get("best_iou"),
            "best_center_distance": match_info.get("best_center_distance"),
            "best_frame_delta": match_info.get("best_frame_delta"),
            "tracklet_length": match_info.get("tracklet_length"),
            "tracklet_start": match_info.get("tracklet_start"),
            "tracklet_end": match_info.get("tracklet_end"),
        })

        obj_rows = build_rows_for_object(scan, clip, a, anchor_frame, frame_count, dg, chosen_track_id, match_info, PARAMS)
        clip_rows.extend(obj_rows)

    clip_df = pd.DataFrame(clip_rows)
    all_rows.extend(clip_rows)

    expected = int(len(an) * frame_count)
    actual = int(len(clip_df))
    tracklet_detection = int((clip_df["trajectory_status"] == "tracklet_detection").sum())
    interp = int((clip_df["trajectory_status"] == "short_gap_interpolated").sum())
    missing = int(clip_df["trajectory_status"].astype(str).str.startswith("missing").sum())
    anchor_rows = int((clip_df["trajectory_status"] == "anchor_gt").sum())
    draw_rows = int((clip_df["draw_ok"] == True).sum())
    matched_objects = int((pd.DataFrame(anchor_match_rows).query("scan_frame_id == @scan")["match_status"] == "matched_tracklet").sum())

    clip_summary_rows.append({
        "scan_frame_id": scan,
        "video_id": clean_str(clip.get("video_id")),
        "anchor_object_count": int(len(an)),
        "matched_object_count": matched_objects,
        "frame_count": int(frame_count),
        "expected_object_frame_rows": expected,
        "actual_object_frame_rows": actual,
        "anchor_rows": anchor_rows,
        "tracklet_detection_rows": tracklet_detection,
        "short_gap_interpolated_rows": interp,
        "missing_rows": missing,
        "draw_ok_rows": draw_rows,
        "matched_object_ratio": float(matched_objects / max(1, len(an))),
        "tracklet_detection_ratio": float(tracklet_detection / max(1, actual)),
        "missing_ratio": float(missing / max(1, actual)),
        "draw_ok_ratio": float(draw_rows / max(1, actual)),
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
            "selected_tracklet_id": clean_str(g["selected_tracklet_id"].iloc[0]),
            "rows_total": int(total),
            "tracklet_detection_rows": int((g["trajectory_status"] == "tracklet_detection").sum()),
            "short_gap_interpolated_rows": int((g["trajectory_status"] == "short_gap_interpolated").sum()),
            "missing_rows": int(g["trajectory_status"].astype(str).str.startswith("missing").sum()),
            "draw_ok_rows": int((g["draw_ok"] == True).sum()),
            "draw_ok_ratio": float((g["draw_ok"] == True).sum() / max(1, total)),
        })

    out_img = GALLERY / f"{scan}_tracklet_level_gt_linking.jpg"
    ok, msg = draw_gallery(scan, clip, clip_df, frames, out_img)

    gallery_rows.append({
        "scan_frame_id": scan,
        "overlay_path": str(out_img),
        "saved": bool(ok),
        "message": msg,
        "anchor_object_count": int(len(an)),
        "matched_object_ratio": float(matched_objects / max(1, len(an))),
        "missing_ratio": float(missing / max(1, actual)),
        "draw_ok_ratio": float(draw_rows / max(1, actual)),
    })


rows_df = pd.DataFrame(all_rows)
clip_summary = pd.DataFrame(clip_summary_rows)
object_summary = pd.DataFrame(object_summary_rows)
anchor_matches = pd.DataFrame(anchor_match_rows)
gallery_index = pd.DataFrame(gallery_rows)

safe_to_csv(rows_df, OUT_ROWS)
safe_to_csv(clip_summary, OUT_CLIP_SUMMARY)
safe_to_csv(object_summary, OUT_OBJECT_SUMMARY)
safe_to_csv(anchor_matches, OUT_ANCHOR_MATCHES)
safe_to_csv(gallery_index, OUT_GALLERY_INDEX)

html = []
html.append("<!doctype html><html><head><meta charset='utf-8'>")
html.append("<title>Week 8 v52b2 Tracklet-level GT Linking Full</title>")
html.append("<style>body{font-family:Arial;background:#111;color:#eee;margin:20px}.grid{display:grid;grid-template-columns:repeat(auto-fill,minmax(760px,1fr));gap:18px}.card{background:#1e1e1e;border:1px solid #333;border-radius:10px;padding:12px}img{width:100%;border:1px solid #444;border-radius:6px}.meta{font-size:13px;color:#bbb;line-height:1.4}.top{background:#2b2b2b;padding:12px;border-radius:10px;margin-bottom:18px}</style></head><body>")
html.append("<div class='top'><h1>Week 8 v52b2 Tracklet-level GT Linking Full</h1>")
html.append("<p>Anchor GT boxes select detector tracklets. This avoids frame-by-frame identity jumps.</p></div><div class='grid'>")

for _, r in gallery_index.iterrows():
    if not bool(r["saved"]):
        continue
    rel = Path(r["overlay_path"]).relative_to(OUT)
    html.append("<div class='card'>")
    html.append(f"<h3>{r['scan_frame_id']}</h3><img src='{rel.as_posix()}'>")
    html.append("<div class='meta'>")
    html.append(f"anchor objects: {r['anchor_object_count']}<br>")
    html.append(f"matched object ratio: {float(r['matched_object_ratio']):.3f}<br>")
    html.append(f"missing ratio: {float(r['missing_ratio']):.3f}<br>")
    html.append(f"draw ok ratio: {float(r['draw_ok_ratio']):.3f}<br>")
    html.append("</div></div>")

html.append("</div></body></html>")
OUT_GALLERY_HTML.write_text("\n".join(html))

full_clip_count = int(len(clip_summary))
full_object_count = int(len(object_summary))
tracking_rows_total = int(len(rows_df))
mean_matched_object_ratio = float(clip_summary["matched_object_ratio"].mean()) if len(clip_summary) else 0.0
mean_missing_ratio = float(clip_summary["missing_ratio"].mean()) if len(clip_summary) else 1.0
mean_draw_ok_ratio = float(clip_summary["draw_ok_ratio"].mean()) if len(clip_summary) else 0.0
mean_tracklet_detection_ratio = float(clip_summary["tracklet_detection_ratio"].mean()) if len(clip_summary) else 0.0

if tracking_rows_total == 0:
    issues.append({
        "item": "tracklet_level_rows",
        "issue_type": "hard_no_rows",
        "issue_detail": "No tracklet-level GT linking rows produced.",
        "severity": "hard",
    })

if mean_matched_object_ratio < 0.60:
    issues.append({
        "item": "tracklet_anchor_matching",
        "issue_type": "warning_low_mean_matched_object_ratio",
        "issue_detail": f"Mean matched object ratio is {mean_matched_object_ratio:.4f}.",
        "severity": "warning",
    })

issues_df = pd.DataFrame(issues, columns=["item", "issue_type", "issue_detail", "severity"])
hard_issues = issues_df[issues_df["severity"] == "hard"] if len(issues_df) else pd.DataFrame()
warnings = issues_df[issues_df["severity"] == "warning"] if len(issues_df) else pd.DataFrame()
ready = len(hard_issues) == 0

decision = pd.DataFrame([{
    "v52b2_decision": "tracklet_level_gt_linking_full_completed" if ready else "tracklet_level_gt_linking_full_blocked",
    "full_clip_count": full_clip_count,
    "full_object_count": full_object_count,
    "tracking_rows_total": tracking_rows_total,
    "mean_matched_object_ratio": mean_matched_object_ratio,
    "mean_tracklet_detection_ratio": mean_tracklet_detection_ratio,
    "mean_missing_ratio": mean_missing_ratio,
    "mean_draw_ok_ratio": mean_draw_ok_ratio,
    "gallery_overlay_count": int(len(gallery_index)),
    "anchor_window": PARAMS["anchor_window"],
    "min_anchor_iou": PARAMS["min_anchor_iou"],
    "max_anchor_center_dist": PARAMS["max_anchor_center_dist"],
    "max_short_gap": PARAMS["max_short_gap"],
    "hard_issue_count": int(len(hard_issues)),
    "warning_count": int(len(warnings)),
    "issue_count": int(len(issues_df)),
    "ready_for_v52b_full_tracklet_level_linking": bool(ready),
    "generated_at": datetime.now().isoformat(timespec="seconds"),
}])

safe_to_csv(issues_df, OUT_ISSUES)
safe_to_csv(decision, OUT_DECISION)

OUT_REPORT.write_text(
    "Week 8 v52b2 Tracklet-level GT Linking Full Report\n\n"
    f"Decision: {decision.iloc[0]['v52b2_decision']}\n"
    f"Full clips: {full_clip_count}\n"
    f"Full objects: {full_object_count}\n"
    f"Tracking rows total: {tracking_rows_total}\n"
    f"Mean matched object ratio: {mean_matched_object_ratio:.4f}\n"
    f"Mean tracklet detection ratio: {mean_tracklet_detection_ratio:.4f}\n"
    f"Mean missing ratio: {mean_missing_ratio:.4f}\n"
    f"Mean draw-ok ratio: {mean_draw_ok_ratio:.4f}\n"
    f"Hard issues: {len(hard_issues)}\n"
    f"Warnings: {len(warnings)}\n\n"
    "Interpretation:\n"
    "This full links anchor GT objects to stable detector tracklets instead of performing frame-by-frame greedy reassociation.\n"
)

OUT_NOTE.write_text(
    "# Week 8 v52b2 Tracklet-level GT Linking Full\n\n"
    "## Summary\n\n"
    f"- v52b2 decision: {decision.iloc[0]['v52b2_decision']}\n"
    f"- Full clips: {full_clip_count}\n"
    f"- Full objects: {full_object_count}\n"
    f"- Tracking rows total: {tracking_rows_total}\n"
    f"- Mean matched object ratio: {mean_matched_object_ratio:.4f}\n"
    f"- Mean tracklet detection ratio: {mean_tracklet_detection_ratio:.4f}\n"
    f"- Mean missing ratio: {mean_missing_ratio:.4f}\n"
    f"- Mean draw-ok ratio: {mean_draw_ok_ratio:.4f}\n"
    f"- Hard issue count: {len(hard_issues)}\n"
    f"- Warning count: {len(warnings)}\n"
    f"- Ready for v52b full tracklet-level linking: {ready}\n\n"
    "## Interpretation\n\n"
    "v52a2/v52a3 can jump identity frame-by-frame. v52b2 instead locks each anchor GT object onto a selected detector tracklet, improving identity stability at the cost of possible missing gaps.\n\n"
    f"Gallery: {OUT_GALLERY_HTML}\n"
)

progress_row = pd.DataFrame([{
    "date": datetime.now().date().isoformat(),
    "stage": "v52b2",
    "task_name": "Tracklet-level GT linking full",
    "status": "PASS" if ready else "BLOCKED",
    "input_summary": str(V51C_TRACKS),
    "output_summary": str(OUT),
    "hard_issues": int(len(hard_issues)),
    "warnings": int(len(warnings)),
    "next_action": "Compare v52a2/v52a3/v52b2 and choose final tracking strategy" if ready else "Review v52b2 issues.",
}])

if OUT_PROGRESS.exists():
    old = pd.read_csv(OUT_PROGRESS)
    progress = pd.concat([old, progress_row], ignore_index=True)
else:
    progress = progress_row

safe_to_csv(progress, OUT_PROGRESS)

print("Saved:")
print(OUT_ROWS)
print(OUT_CLIP_SUMMARY)
print(OUT_OBJECT_SUMMARY)
print(OUT_ANCHOR_MATCHES)
print(OUT_GALLERY_INDEX)
print(OUT_GALLERY_HTML)
print(OUT_DECISION)
print(OUT_ISSUES)
print(OUT_REPORT)
print(OUT_NOTE)

print()
print("=== v52b2 decision ===")
print(decision.to_string(index=False))

print()
print("=== v52b2 issues ===")
if len(issues_df):
    print(issues_df.to_string(index=False))
else:
    print("No issues found.")

print()
print("=== v52b2 clip summary ===")
print(clip_summary.to_string(index=False))
