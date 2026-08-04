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

OUT = W8 / "outputs" / "v52a2_gt_seeded_detector_linked_pilot"
GALLERY = OUT / "gt_seeded_detector_linked_gallery"
REPORTS = W8 / "reports"
NOTES = W8 / "notes"
PROGRESS = W8 / "progress"

for p in [OUT, GALLERY, REPORTS, NOTES, PROGRESS]:
    p.mkdir(parents=True, exist_ok=True)

OUT_TRACKS = OUT / "week8_v52a2_gt_seeded_detector_linked_rows.csv"
OUT_CLIP_SUMMARY = OUT / "week8_v52a2_clip_summary.csv"
OUT_OBJECT_SUMMARY = OUT / "week8_v52a2_object_summary.csv"
OUT_GALLERY_INDEX = OUT / "week8_v52a2_gallery_index.csv"
OUT_GALLERY_HTML = OUT / "index.html"
OUT_DECISION = OUT / "week8_v52a2_decision_summary.csv"
OUT_ISSUES = OUT / "week8_v52a2_issues.csv"
OUT_REPORT = REPORTS / "week8_v52a2_gt_seeded_detector_linked_pilot_report.md"
OUT_NOTE = NOTES / "week8_v52a2_gt_seeded_detector_linked_pilot_notes.md"
OUT_PROGRESS = PROGRESS / "week8_experiment_progress_log.csv"


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
    return [
        float(row["bbox_x1"]),
        float(row["bbox_y1"]),
        float(row["bbox_x2"]),
        float(row["bbox_y2"]),
    ]


def bbox_from_det(row):
    return [
        float(row["x1"]),
        float(row["y1"]),
        float(row["x2"]),
        float(row["y2"]),
    ]


def choose_matches(states, candidates, frame_index, max_center_dist=95.0, min_iou=0.03):
    proposals = []

    for obj_id, st in states.items():
        if st["status"] == "dead":
            continue

        prev_box = st["bbox"]
        prev_area = area_xyxy(prev_box)

        for ci, cand in candidates.iterrows():
            cbox = bbox_from_det(cand)
            val_iou = iou_xyxy(prev_box, cbox)
            dist = center_distance(prev_box, cbox)
            area_ratio = area_xyxy(cbox) / prev_area

            area_penalty = abs(math.log(max(0.05, min(20.0, area_ratio))))
            score = (
                3.0 * val_iou
                - 0.010 * dist
                - 0.30 * area_penalty
                + 0.20 * float(cand.get("score", 0.0))
            )

            eligible = (
                val_iou >= min_iou
                or dist <= max_center_dist
            )

            if eligible:
                proposals.append({
                    "obj_id": obj_id,
                    "candidate_index": ci,
                    "score": score,
                    "iou": val_iou,
                    "center_distance": dist,
                    "area_ratio": area_ratio,
                })

    proposals = sorted(proposals, key=lambda x: x["score"], reverse=True)

    assigned_obj = set()
    assigned_cand = set()
    assignments = {}

    for p in proposals:
        obj_id = p["obj_id"]
        ci = p["candidate_index"]

        if obj_id in assigned_obj or ci in assigned_cand:
            continue

        assigned_obj.add(obj_id)
        assigned_cand.add(ci)
        assignments[obj_id] = p

    return assignments


def propagate_direction(scan, clip, anchors_df, det_df, direction, max_gap=18):
    anchor_frame, anchor_rel_sec = get_anchor_frame(clip)

    frame_count = int(det_df["video_frame_count"].dropna().max()) if len(det_df) and "video_frame_count" in det_df.columns else int(anchor_frame + 126)
    frame_count = max(frame_count, anchor_frame + 1)

    if direction == "forward":
        frames = list(range(anchor_frame + 1, frame_count))
    else:
        frames = list(range(anchor_frame - 1, -1, -1))

    states = {}
    rows = []

    for _, a in anchors_df.iterrows():
        obj_id = clean_str(a.get("final_box_id"))
        seed_box = bbox_from_anchor(a)

        states[obj_id] = {
            "bbox": seed_box,
            "missed": 0,
            "status": "active",
            "anchor": a.to_dict(),
        }

    det_by_frame = {int(k): v.copy() for k, v in det_df.groupby("frame_index_in_clip")} if len(det_df) else {}

    for fi in frames:
        cand = det_by_frame.get(fi, pd.DataFrame())
        assignments = choose_matches(states, cand, fi)

        for obj_id, st in states.items():
            a = st["anchor"]

            if obj_id in assignments:
                p = assignments[obj_id]
                drow = cand.loc[p["candidate_index"]]
                new_box = bbox_from_det(drow)
                st["bbox"] = new_box
                st["missed"] = 0

                status = "detector_linked"
                source_det_track_id = clean_str(drow.get("track_id"))
                source_det_score = to_float(drow.get("score"))
                match_iou = p["iou"]
                match_dist = p["center_distance"]
                area_ratio = p["area_ratio"]
                draw_ok = True

            else:
                st["missed"] += 1
                new_box = st["bbox"]

                source_det_track_id = ""
                source_det_score = None
                match_iou = 0.0
                match_dist = None
                area_ratio = 1.0

                if st["missed"] <= max_gap:
                    status = "short_gap_carried"
                    draw_ok = True
                else:
                    status = "missing_long_gap"
                    draw_ok = False

            rows.append({
                "dataset_version": "week8_v52a2_gt_seeded_detector_linked_pilot",
                "scan_frame_id": scan,
                "video_id": clean_str(clip.get("video_id")),
                "clip_path": clean_str(clip.get("clip_path")),
                "frame_index_in_clip": int(fi),
                "timestamp_sec_in_clip": None,
                "final_box_id": obj_id,
                "behaviour_pig_id": clean_str(a.get("behaviour_pig_id")),
                "visual_marker_colour": clean_str(a.get("visual_marker_colour")),
                "behaviour_code": clean_str(a.get("behaviour_code")),
                "anchor_frame_index": int(anchor_frame),
                "tracking_direction": direction,
                "x1": float(new_box[0]),
                "y1": float(new_box[1]),
                "x2": float(new_box[2]),
                "y2": float(new_box[3]),
                "trajectory_status": status,
                "draw_ok": bool(draw_ok),
                "missed_count": int(st["missed"]),
                "source_detector_track_id": source_det_track_id,
                "source_detector_score": source_det_score,
                "match_iou": float(match_iou),
                "match_center_distance": match_dist,
                "area_ratio_vs_previous": float(area_ratio),
                "method": "gt_seeded_detector_linked",
            })

    return rows


def build_gt_seeded_linked_tracks(scan, clip, anchors_df, det_df):
    anchor_frame, anchor_rel_sec = get_anchor_frame(clip)

    if len(det_df) and "video_frame_count" in det_df.columns:
        frame_count = int(det_df["video_frame_count"].dropna().max())
    else:
        frame_count = anchor_frame + 126

    frame_count = max(frame_count, anchor_frame + 1)

    rows = []

    for _, a in anchors_df.iterrows():
        seed_box = bbox_from_anchor(a)

        rows.append({
            "dataset_version": "week8_v52a2_gt_seeded_detector_linked_pilot",
            "scan_frame_id": scan,
            "video_id": clean_str(clip.get("video_id")),
            "clip_path": clean_str(clip.get("clip_path")),
            "frame_index_in_clip": int(anchor_frame),
            "timestamp_sec_in_clip": None,
            "final_box_id": clean_str(a.get("final_box_id")),
            "behaviour_pig_id": clean_str(a.get("behaviour_pig_id")),
            "visual_marker_colour": clean_str(a.get("visual_marker_colour")),
            "behaviour_code": clean_str(a.get("behaviour_code")),
            "anchor_frame_index": int(anchor_frame),
            "tracking_direction": "anchor",
            "x1": float(seed_box[0]),
            "y1": float(seed_box[1]),
            "x2": float(seed_box[2]),
            "y2": float(seed_box[3]),
            "trajectory_status": "anchor_gt",
            "draw_ok": True,
            "missed_count": 0,
            "source_detector_track_id": "",
            "source_detector_score": None,
            "match_iou": 1.0,
            "match_center_distance": 0.0,
            "area_ratio_vs_previous": 1.0,
            "method": "gt_seeded_detector_linked",
        })

    rows.extend(propagate_direction(scan, clip, anchors_df, det_df, "forward"))
    rows.extend(propagate_direction(scan, clip, anchors_df, det_df, "backward"))

    return rows


def read_frame(video_path, frame_index):
    import cv2

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        cap.release()
        return None

    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    frame_index = max(0, min(total - 1, int(frame_index)))
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
    ok, frame = cap.read()
    cap.release()

    if not ok:
        return None
    return frame


def draw_gallery(scan, clip, tracks_df, out_path):
    import cv2

    clip_path = clean_str(clip.get("clip_path"))

    frame_count = int(tracks_df["frame_index_in_clip"].max()) + 1 if len(tracks_df) else 1
    sample_frames = sorted(set([0, frame_count // 2, max(0, frame_count - 1)]))

    colour_map = {
        "blue": (255, 80, 30),
        "green": (60, 220, 60),
        "cyan": (255, 220, 0),
        "red": (40, 40, 255),
        "red_neck": (40, 40, 255),
        "red_tail": (220, 80, 255),
        "pink": (220, 80, 255),
        "purple": (180, 70, 220),
        "unknown": (180, 180, 180),
        "not_visible": (130, 130, 130),
        "uncertain": (0, 220, 255),
        "unassigned": (160, 160, 160),
    }

    panels = []

    for fi in sample_frames:
        img = read_frame(clip_path, fi)
        if img is None:
            continue

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

            if status == "anchor_gt":
                thickness = 3
            elif status == "detector_linked":
                thickness = 2
            else:
                thickness = 1

            cv2.rectangle(img, (x1, y1), (x2, y2), bgr, thickness)

            label = f"{clean_str(r.get('behaviour_pig_id'))}/{colour}/{status}"
            cv2.putText(img, label, (x1, max(18, y1 - 6)), cv2.FONT_HERSHEY_SIMPLEX, 0.40, bgr, 1)

        cv2.rectangle(img, (0, 0), (w, 55), (0, 0, 0), -1)
        cv2.putText(img, f"{scan} GT-seeded detector-linked frame={fi}", (10, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.62, (255, 255, 255), 2)
        cv2.putText(img, "anchor GT + v51c dense detections, missing shown by absence", (10, 47), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (230, 230, 230), 1)

        panels.append(img)

    if not panels:
        return False, "no_panels"

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
        "v52a2_decision": "gt_seeded_detector_linked_pilot_blocked_missing_inputs",
        "hard_issue_count": int((issues_df["severity"] == "hard").sum()),
        "ready_for_v52b_full_gt_seeded_detector_linked_tracking": False,
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
            "issue_type": "warning_missing_clip_or_anchors",
            "issue_detail": "Missing clip metadata or anchor boxes.",
            "severity": "warning",
        })
        continue

    rows = build_gt_seeded_linked_tracks(scan, clip, an, dg)
    clip_df = pd.DataFrame(rows)
    all_rows.extend(rows)

    frame_count = int(clip_df["frame_index_in_clip"].nunique()) if len(clip_df) else 0
    expected_rows = int(len(an) * frame_count)
    actual_rows = int(len(clip_df))
    linked_rows = int((clip_df["trajectory_status"] == "detector_linked").sum())
    carried_rows = int((clip_df["trajectory_status"] == "short_gap_carried").sum())
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
        "detector_linked_rows": linked_rows,
        "short_gap_carried_rows": carried_rows,
        "missing_long_gap_rows": missing_rows,
        "draw_ok_rows": draw_rows,
        "detector_linked_ratio": float(linked_rows / max(1, actual_rows)),
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
            "short_gap_carried_rows": int((g["trajectory_status"] == "short_gap_carried").sum()),
            "missing_long_gap_rows": int((g["trajectory_status"] == "missing_long_gap").sum()),
            "draw_ok_rows": int((g["draw_ok"] == True).sum()),
            "detector_linked_ratio": float((g["trajectory_status"] == "detector_linked").sum() / max(1, total)),
            "missing_ratio": float((g["trajectory_status"] == "missing_long_gap").sum() / max(1, total)),
        })

    out_img = GALLERY / f"{scan}_gt_seeded_detector_linked.jpg"
    ok, msg = draw_gallery(scan, clip, clip_df, out_img)

    gallery_rows.append({
        "scan_frame_id": scan,
        "overlay_path": str(out_img),
        "saved": bool(ok),
        "message": msg,
        "anchor_object_count": int(len(an)),
        "detector_linked_ratio": float(linked_rows / max(1, actual_rows)),
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
html.append("<!doctype html>")
html.append("<html><head><meta charset='utf-8'>")
html.append("<title>Week 8 v52a2 GT-seeded Detector-linked Pilot</title>")
html.append("<style>")
html.append("body{font-family:Arial;background:#111;color:#eee;margin:20px}")
html.append(".grid{display:grid;grid-template-columns:repeat(auto-fill,minmax(760px,1fr));gap:18px}")
html.append(".card{background:#1e1e1e;border:1px solid #333;border-radius:10px;padding:12px}")
html.append("img{width:100%;border:1px solid #444;border-radius:6px}")
html.append(".meta{font-size:13px;color:#bbb;line-height:1.4}")
html.append(".top{background:#2b2b2b;padding:12px;border-radius:10px;margin-bottom:18px}")
html.append("</style></head><body>")
html.append("<div class='top'>")
html.append("<h1>Week 8 v52a2 GT-seeded Detector-linked Tracking Pilot</h1>")
html.append("<p>One trajectory per anchor GT pig, linked through v51c dense detections. Long missing gaps are not drawn.</p>")
html.append("</div><div class='grid'>")

for _, r in gallery_index.iterrows():
    if not bool(r["saved"]):
        continue
    rel = Path(r["overlay_path"]).relative_to(OUT)
    html.append("<div class='card'>")
    html.append(f"<h3>{r['scan_frame_id']}</h3>")
    html.append(f"<img src='{rel.as_posix()}'>")
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
        "item": "gt_seeded_detector_linked_rows",
        "issue_type": "hard_no_rows",
        "issue_detail": "No GT-seeded detector-linked rows produced.",
        "severity": "hard",
    })

if mean_missing_ratio > 0.35:
    issues.append({
        "item": "gt_seeded_detector_linked_missing",
        "issue_type": "warning_high_missing_ratio",
        "issue_detail": f"Mean missing ratio is {mean_missing_ratio:.4f}.",
        "severity": "warning",
    })

issues_df = pd.DataFrame(issues, columns=["item", "issue_type", "issue_detail", "severity"])
hard_issues = issues_df[issues_df["severity"] == "hard"] if len(issues_df) else pd.DataFrame()
warnings = issues_df[issues_df["severity"] == "warning"] if len(issues_df) else pd.DataFrame()

ready = len(hard_issues) == 0

decision = pd.DataFrame([{
    "v52a2_decision": "gt_seeded_detector_linked_pilot_completed" if ready else "gt_seeded_detector_linked_pilot_blocked",
    "pilot_clip_count": pilot_clip_count,
    "pilot_object_count": pilot_object_count,
    "tracking_rows_total": tracking_rows_total,
    "mean_detector_linked_ratio": mean_detector_linked_ratio,
    "mean_missing_ratio": mean_missing_ratio,
    "mean_draw_ok_ratio": mean_draw_ok_ratio,
    "gallery_overlay_count": int(len(gallery_index)),
    "hard_issue_count": int(len(hard_issues)),
    "warning_count": int(len(warnings)),
    "issue_count": int(len(issues_df)),
    "ready_for_v52b_full_gt_seeded_detector_linked_tracking": bool(ready),
    "generated_at": datetime.now().isoformat(timespec="seconds"),
}])

safe_to_csv(issues_df, OUT_ISSUES)
safe_to_csv(decision, OUT_DECISION)

OUT_REPORT.write_text(
    "Week 8 v52a2 GT-seeded Detector-linked Tracking Pilot Report\n\n"
    f"Decision: {decision.iloc[0]['v52a2_decision']}\n"
    f"Pilot clips: {pilot_clip_count}\n"
    f"Pilot objects: {pilot_object_count}\n"
    f"Tracking rows total: {tracking_rows_total}\n"
    f"Mean detector-linked ratio: {mean_detector_linked_ratio:.4f}\n"
    f"Mean missing ratio: {mean_missing_ratio:.4f}\n"
    f"Mean draw-ok ratio: {mean_draw_ok_ratio:.4f}\n"
    f"Hard issues: {len(hard_issues)}\n"
    f"Warnings: {len(warnings)}\n\n"
    "Interpretation:\n"
    "This pilot replaces pure OpenCV tracking with anchor-seeded association through v51c dense detections.\n"
)

OUT_NOTE.write_text(
    "# Week 8 v52a2 GT-seeded Detector-linked Tracking Pilot\n\n"
    "## Summary\n\n"
    f"- v52a2 decision: {decision.iloc[0]['v52a2_decision']}\n"
    f"- Pilot clips: {pilot_clip_count}\n"
    f"- Pilot objects: {pilot_object_count}\n"
    f"- Tracking rows total: {tracking_rows_total}\n"
    f"- Mean detector-linked ratio: {mean_detector_linked_ratio:.4f}\n"
    f"- Mean missing ratio: {mean_missing_ratio:.4f}\n"
    f"- Mean draw-ok ratio: {mean_draw_ok_ratio:.4f}\n"
    f"- Hard issue count: {len(hard_issues)}\n"
    f"- Warning count: {len(warnings)}\n"
    f"- Ready for v52b full GT-seeded detector-linked tracking: {ready}\n\n"
    "## Interpretation\n\n"
    "v52a failed because OpenCV tracker propagation did not work. "
    "v52a2 instead uses anchor GT identities and links them through v51c dense detections frame by frame.\n\n"
    f"Gallery: {OUT_GALLERY_HTML}\n"
)

progress_row = pd.DataFrame([{
    "date": datetime.now().date().isoformat(),
    "stage": "v52a2",
    "task_name": "GT-seeded detector-linked tracking pilot",
    "status": "PASS" if ready else "BLOCKED",
    "input_summary": str(V51C_TRACKS),
    "output_summary": str(OUT),
    "hard_issues": int(len(hard_issues)),
    "warnings": int(len(warnings)),
    "next_action": "v52b full GT-seeded detector-linked tracking" if ready else "Review pilot issues.",
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
print("=== v52a2 decision ===")
print(decision.to_string(index=False))

print()
print("=== v52a2 issues ===")
if len(issues_df):
    print(issues_df.to_string(index=False))
else:
    print("No issues found.")

print()
print("=== v52a2 clip summary ===")
print(clip_summary.to_string(index=False))
