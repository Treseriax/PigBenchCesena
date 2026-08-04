from pathlib import Path
from datetime import datetime
import json
import csv
import pandas as pd
import numpy as np


ROOT = Path.home() / "PigBench"
W8 = ROOT / "Week8_Ground_Truth_Validation_Behaviour_Dataset_Preparation"

V45_CLIP_JSON = W8 / "outputs" / "propagated_ground_truth" / "v45_label_propagation" / "week8_v45_clip_level_ground_truth.json"
V45_ANCHORS = W8 / "outputs" / "propagated_ground_truth" / "v45_label_propagation" / "week8_v45_clip_object_propagated_annotations.csv"
V50C_RAW_TRACKS = W8 / "outputs" / "v50c_full_72_clip_cpu_tracking" / "week8_v50c_full_tracking_rows.csv"
V50E_ROI = W8 / "outputs" / "v50e_roi_filtered_tracking_overlay" / "week8_v50e_video_level_roi_from_anchor_boxes.csv"
V50E_FILTERED_TRACKS = W8 / "outputs" / "v50e_roi_filtered_tracking_overlay" / "week8_v50e_roi_filtered_tracking_rows.csv"
V50E_DECISION = W8 / "outputs" / "v50e_roi_filtered_tracking_overlay" / "week8_v50e_decision_summary.csv"

OUT = W8 / "outputs" / "v50f_anchor_tracking_recall_audit"
GALLERY = OUT / "anchor_tracking_recall_gallery"
REPORTS = W8 / "reports"
NOTES = W8 / "notes"
PROGRESS = W8 / "progress"

for p in [OUT, GALLERY, REPORTS, NOTES, PROGRESS]:
    p.mkdir(parents=True, exist_ok=True)

OUT_OBJECT_AUDIT = OUT / "week8_v50f_anchor_object_tracking_recall_audit.csv"
OUT_CLIP_SUMMARY = OUT / "week8_v50f_clip_tracking_recall_summary.csv"
OUT_REVIEW_QUEUE = OUT / "week8_v50f_missed_anchor_review_queue.csv"
OUT_GALLERY_INDEX = OUT / "week8_v50f_gallery_index.csv"
OUT_GALLERY_HTML = OUT / "index.html"
OUT_DECISION = OUT / "week8_v50f_decision_summary.csv"
OUT_ISSUES = OUT / "week8_v50f_issues.csv"
OUT_REPORT = REPORTS / "week8_v50f_anchor_tracking_recall_audit_report.md"
OUT_NOTE = NOTES / "week8_v50f_anchor_tracking_recall_audit_notes.md"
OUT_PROGRESS = PROGRESS / "week8_experiment_progress_log.csv"


def safe_to_csv(df, path):
    df.to_csv(
        path,
        index=False,
        quoting=csv.QUOTE_ALL,
        escapechar="\\",
        lineterminator="\n",
    )


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


def nearest_frame(frames, target):
    frames = [int(x) for x in frames if pd.notna(x)]
    if not frames:
        return None
    arr = np.array(sorted(frames))
    return int(arr[np.argmin(np.abs(arr - int(target)))])


def get_anchor_frame(clip):
    fps = to_float(clip.get("fps_used"), 25.0) or 25.0
    duration = to_float(clip.get("duration_sec"), 10.0) or 10.0
    scan = clean_str(clip.get("scan_frame_id"))

    if scan in ["scanframe_0000", "scanframe_0006", "scanframe_0012", "scanframe_0018"]:
        anchor_rel_sec = 0.0
    else:
        anchor_rel_sec = duration / 2.0

    return int(round(anchor_rel_sec * fps)), anchor_rel_sec


def center_inside_roi(row):
    cx = (row["x1"] + row["x2"]) / 2.0
    cy = (row["y1"] + row["y2"]) / 2.0
    return (
        cx >= row["roi_x1"]
        and cx <= row["roi_x2"]
        and cy >= row["roi_y1"]
        and cy <= row["roi_y2"]
    )


def bbox_overlap_ratio_with_roi(row):
    x1 = max(row["x1"], row["roi_x1"])
    y1 = max(row["y1"], row["roi_y1"])
    x2 = min(row["x2"], row["roi_x2"])
    y2 = min(row["y2"], row["roi_y2"])

    iw = max(0.0, x2 - x1)
    ih = max(0.0, y2 - y1)
    inter = iw * ih

    area = max(0.0, row["x2"] - row["x1"]) * max(0.0, row["y2"] - row["y1"])
    if area <= 0:
        return 0.0
    return inter / area


def reconstruct_roi_keep(raw_tracks, roi):
    t = raw_tracks.copy()
    r = roi.copy()

    t["video_id_clean"] = t["video_id"].astype(str).map(clean_str)
    r["video_id_clean"] = r["video_id"].astype(str).map(clean_str)

    t = t.merge(
        r[["video_id_clean", "roi_x1", "roi_y1", "roi_x2", "roi_y2"]],
        on="video_id_clean",
        how="left",
    )

    t["track_center_x"] = (t["x1"] + t["x2"]) / 2.0
    t["track_center_y"] = (t["y1"] + t["y2"]) / 2.0

    t["roi_center_inside"] = t.apply(
        lambda row: False if pd.isna(row.get("roi_x1")) else center_inside_roi(row),
        axis=1,
    )

    t["roi_bbox_overlap_ratio"] = t.apply(
        lambda row: 0.0 if pd.isna(row.get("roi_x1")) else bbox_overlap_ratio_with_roi(row),
        axis=1,
    )

    t["roi_keep_reconstructed"] = t["roi_center_inside"] | (t["roi_bbox_overlap_ratio"] >= 0.50)
    t.loc[t["roi_x1"].isna(), "roi_keep_reconstructed"] = True

    return t


def best_match(anchor_box, candidate_tracks):
    best_iou = 0.0
    best_track_id = ""
    best_score = None

    for _, tr in candidate_tracks.iterrows():
        tbox = [
            to_float(tr.get("x1")),
            to_float(tr.get("y1")),
            to_float(tr.get("x2")),
            to_float(tr.get("y2")),
        ]

        if not all(v is not None for v in tbox):
            continue

        val = iou_xyxy(anchor_box, tbox)
        if val > best_iou:
            best_iou = val
            best_track_id = clean_str(tr.get("track_id"))
            best_score = to_float(tr.get("score"))

    return best_iou, best_track_id, best_score


def draw_gallery_image(scan, clip, anchors_df, raw_frame, kept_frame, audit_df, roi_row, out_path):
    try:
        import cv2
    except Exception:
        return False, "cv2_not_available"

    clip_path = Path(clean_str(clip.get("clip_path")))
    cap = cv2.VideoCapture(str(clip_path))

    if not cap.isOpened():
        cap.release()
        return False, "video_open_failed"

    anchor_frame, _ = get_anchor_frame(clip)

    if len(raw_frame):
        draw_frame = int(raw_frame["frame_index_in_clip"].iloc[0])
    elif len(kept_frame):
        draw_frame = int(kept_frame["frame_index_in_clip"].iloc[0])
    else:
        draw_frame = anchor_frame

    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    draw_frame = max(0, min(frame_count - 1, draw_frame))

    cap.set(cv2.CAP_PROP_POS_FRAMES, draw_frame)
    ok, frame = cap.read()
    cap.release()

    if not ok or frame is None:
        return False, "frame_read_failed"

    h, w = frame.shape[:2]

    if roi_row is not None:
        rx1 = to_int(roi_row.get("roi_x1"))
        ry1 = to_int(roi_row.get("roi_y1"))
        rx2 = to_int(roi_row.get("roi_x2"))
        ry2 = to_int(roi_row.get("roi_y2"))
        if None not in [rx1, ry1, rx2, ry2]:
            cv2.rectangle(frame, (rx1, ry1), (rx2, ry2), (0, 255, 255), 2)
            cv2.putText(frame, "ROI", (rx1 + 5, max(18, ry1 - 6)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2)

    # Raw removed/outside tracks: red.
    if len(raw_frame):
        outside = raw_frame[raw_frame["roi_keep_reconstructed"] == False]
        for _, tr in outside.iterrows():
            x1 = to_int(tr.get("x1"))
            y1 = to_int(tr.get("y1"))
            x2 = to_int(tr.get("x2"))
            y2 = to_int(tr.get("y2"))
            if None in [x1, y1, x2, y2]:
                continue
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 1)

    # Kept tracking: white thick.
    for _, tr in kept_frame.iterrows():
        x1 = to_int(tr.get("x1"))
        y1 = to_int(tr.get("y1"))
        x2 = to_int(tr.get("x2"))
        y2 = to_int(tr.get("y2"))
        if None in [x1, y1, x2, y2]:
            continue
        label = f"T{clean_str(tr.get('track_id'))}"

        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 0), 4)
        cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 255, 255), 2)
        cv2.putText(frame, label, (x1, min(h - 5, y2 + 15)), cv2.FONT_HERSHEY_SIMPLEX, 0.43, (0, 0, 0), 3)
        cv2.putText(frame, label, (x1, min(h - 5, y2 + 15)), cv2.FONT_HERSHEY_SIMPLEX, 0.43, (255, 255, 255), 1)

    # Anchor GT on top, status-coded.
    status_colour = {
        "filtered_strong_match": (40, 220, 40),
        "filtered_moderate_match": (0, 180, 255),
        "raw_match_removed_by_roi": (0, 140, 255),
        "detector_or_sampling_miss": (255, 0, 255),
        "weak_filtered_match": (180, 180, 180),
    }

    audit_by_box = {clean_str(r["final_box_id"]): r for _, r in audit_df.iterrows()}

    for _, a in anchors_df.iterrows():
        x1 = to_int(a.get("bbox_x1"))
        y1 = to_int(a.get("bbox_y1"))
        x2 = to_int(a.get("bbox_x2"))
        y2 = to_int(a.get("bbox_y2"))
        if None in [x1, y1, x2, y2]:
            continue

        box_id = clean_str(a.get("final_box_id"))
        ar = audit_by_box.get(box_id, {})
        status = clean_str(ar.get("tracking_recall_status")) or "unknown"
        bgr = status_colour.get(status, (200, 200, 200))

        cv2.rectangle(frame, (x1, y1), (x2, y2), bgr, 3)
        label = f"GT {clean_str(a.get('behaviour_pig_id'))}/{clean_str(a.get('visual_marker_colour'))}/{status}"
        cv2.putText(frame, label, (x1, max(20, y1 - 7)), cv2.FONT_HERSHEY_SIMPLEX, 0.42, bgr, 2)

    cv2.rectangle(frame, (0, 0), (w, 78), (0, 0, 0), -1)
    cv2.putText(frame, f"{scan} anchor-vs-tracking recall audit", (12, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 2)
    cv2.putText(frame, "green=strong, orange=moderate, yellow=raw-only/ROI issue, magenta=miss", (12, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.47, (230, 230, 230), 1)
    cv2.putText(frame, f"draw_frame={draw_frame}, anchor_frame={anchor_frame}", (12, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.43, (230, 230, 230), 1)

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    saved = cv2.imwrite(str(out_path), frame)

    return bool(saved), "saved" if saved else "write_failed"


issues = []

required = [V45_CLIP_JSON, V45_ANCHORS, V50C_RAW_TRACKS, V50E_ROI, V50E_FILTERED_TRACKS, V50E_DECISION]
for p in required:
    if not p.exists():
        issues.append({
            "item": str(p),
            "issue_type": "hard_missing_required_input",
            "issue_detail": "Required artifact missing.",
            "severity": "hard",
        })

if issues:
    issues_df = pd.DataFrame(issues)
    safe_to_csv(issues_df, OUT_ISSUES)
    decision = pd.DataFrame([{
        "v50f_decision": "anchor_tracking_recall_audit_blocked_missing_inputs",
        "hard_issue_count": int((issues_df["severity"] == "hard").sum()),
        "ready_for_v51_tracking_integrated_visualizer": False,
        "generated_at": datetime.now().isoformat(timespec="seconds"),
    }])
    safe_to_csv(decision, OUT_DECISION)
    print(decision.to_string(index=False))
    raise SystemExit(1)


clip_data = json.loads(V45_CLIP_JSON.read_text())
clips = clip_data.get("clips", [])
clip_map = {clean_str(c.get("scan_frame_id")): c for c in clips}

anchors = pd.read_csv(V45_ANCHORS)
raw_tracks = pd.read_csv(V50C_RAW_TRACKS)
filtered_tracks = pd.read_csv(V50E_FILTERED_TRACKS)
roi = pd.read_csv(V50E_ROI)

for df, cols in [
    (anchors, ["bbox_x1", "bbox_y1", "bbox_x2", "bbox_y2"]),
    (raw_tracks, ["frame_index_in_clip", "x1", "y1", "x2", "y2", "score"]),
    (filtered_tracks, ["frame_index_in_clip", "x1", "y1", "x2", "y2", "score"]),
]:
    for c in cols:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")

raw_tracks_roi = reconstruct_roi_keep(raw_tracks, roi)

anchor_by_scan = {str(k): v.copy() for k, v in anchors.groupby("scan_frame_id")}
raw_by_scan = {str(k): v.copy() for k, v in raw_tracks_roi.groupby("scan_frame_id")}
filtered_by_scan = {str(k): v.copy() for k, v in filtered_tracks.groupby("scan_frame_id")}

object_rows = []

for scan, clip in clip_map.items():
    anchor_frame, anchor_rel_sec = get_anchor_frame(clip)

    an = anchor_by_scan.get(scan, pd.DataFrame())
    raw = raw_by_scan.get(scan, pd.DataFrame())
    filt = filtered_by_scan.get(scan, pd.DataFrame())

    raw_nearest = nearest_frame(raw["frame_index_in_clip"].dropna().unique().tolist(), anchor_frame) if len(raw) else None
    filt_nearest = nearest_frame(filt["frame_index_in_clip"].dropna().unique().tolist(), anchor_frame) if len(filt) else None

    raw_frame = raw[raw["frame_index_in_clip"].astype("Int64") == raw_nearest].copy() if raw_nearest is not None else pd.DataFrame()
    filt_frame = filt[filt["frame_index_in_clip"].astype("Int64") == filt_nearest].copy() if filt_nearest is not None else pd.DataFrame()

    raw_kept_frame = raw_frame[raw_frame["roi_keep_reconstructed"] == True].copy() if len(raw_frame) else pd.DataFrame()
    raw_removed_frame = raw_frame[raw_frame["roi_keep_reconstructed"] == False].copy() if len(raw_frame) else pd.DataFrame()

    for _, a in an.iterrows():
        abox = [
            to_float(a.get("bbox_x1")),
            to_float(a.get("bbox_y1")),
            to_float(a.get("bbox_x2")),
            to_float(a.get("bbox_y2")),
        ]

        if not all(v is not None for v in abox):
            continue

        filtered_iou, filtered_track_id, filtered_score = best_match(abox, filt_frame)
        raw_iou, raw_track_id, raw_score = best_match(abox, raw_frame)
        raw_kept_iou, raw_kept_track_id, raw_kept_score = best_match(abox, raw_kept_frame)
        raw_removed_iou, raw_removed_track_id, raw_removed_score = best_match(abox, raw_removed_frame)

        if filtered_iou >= 0.50:
            status = "filtered_strong_match"
        elif filtered_iou >= 0.30:
            status = "filtered_moderate_match"
        elif raw_removed_iou >= 0.30:
            status = "raw_match_removed_by_roi"
        elif raw_iou >= 0.10:
            status = "weak_filtered_match"
        else:
            status = "detector_or_sampling_miss"

        object_rows.append({
            "scan_frame_id": scan,
            "video_id": clean_str(clip.get("video_id")),
            "final_box_id": clean_str(a.get("final_box_id")),
            "behaviour_pig_id": clean_str(a.get("behaviour_pig_id")),
            "visual_marker_colour": clean_str(a.get("visual_marker_colour")),
            "behaviour_code": clean_str(a.get("behaviour_code")),
            "anchor_frame_expected": int(anchor_frame),
            "raw_nearest_frame": raw_nearest,
            "filtered_nearest_frame": filt_nearest,
            "raw_frame_delta": None if raw_nearest is None else int(raw_nearest - anchor_frame),
            "filtered_frame_delta": None if filt_nearest is None else int(filt_nearest - anchor_frame),
            "raw_boxes_at_frame": int(len(raw_frame)),
            "filtered_boxes_at_frame": int(len(filt_frame)),
            "raw_kept_boxes_at_frame": int(len(raw_kept_frame)),
            "raw_removed_boxes_at_frame": int(len(raw_removed_frame)),
            "best_filtered_iou": float(filtered_iou),
            "best_filtered_track_id": filtered_track_id,
            "best_filtered_score": filtered_score,
            "best_raw_iou": float(raw_iou),
            "best_raw_track_id": raw_track_id,
            "best_raw_score": raw_score,
            "best_raw_removed_iou": float(raw_removed_iou),
            "best_raw_removed_track_id": raw_removed_track_id,
            "tracking_recall_status": status,
        })

object_audit = pd.DataFrame(object_rows)
safe_to_csv(object_audit, OUT_OBJECT_AUDIT)

# Clip summary.
clip_rows = []
for scan, g in object_audit.groupby("scan_frame_id"):
    total = len(g)
    strong = int((g["tracking_recall_status"] == "filtered_strong_match").sum())
    moderate = int((g["tracking_recall_status"] == "filtered_moderate_match").sum())
    raw_removed = int((g["tracking_recall_status"] == "raw_match_removed_by_roi").sum())
    weak = int((g["tracking_recall_status"] == "weak_filtered_match").sum())
    miss = int((g["tracking_recall_status"] == "detector_or_sampling_miss").sum())

    filtered_recall = (strong + moderate) / total if total else 0.0
    raw_recall = (strong + moderate + raw_removed + weak) / total if total else 0.0

    if filtered_recall >= 0.85:
        status = "good_filtered_tracking_recall"
    elif filtered_recall >= 0.65:
        status = "moderate_filtered_tracking_recall"
    else:
        status = "low_filtered_tracking_recall_review"

    clip_rows.append({
        "scan_frame_id": scan,
        "video_id": clean_str(g["video_id"].iloc[0]),
        "anchor_object_count": int(total),
        "filtered_strong_match": strong,
        "filtered_moderate_match": moderate,
        "raw_match_removed_by_roi": raw_removed,
        "weak_filtered_match": weak,
        "detector_or_sampling_miss": miss,
        "filtered_anchor_recall_strong_or_moderate": float(filtered_recall),
        "raw_anchor_recall_with_weak_or_roi_removed": float(raw_recall),
        "clip_recall_status": status,
    })

clip_summary = pd.DataFrame(clip_rows)
safe_to_csv(clip_summary, OUT_CLIP_SUMMARY)

review_queue = object_audit[
    object_audit["tracking_recall_status"].isin([
        "raw_match_removed_by_roi",
        "weak_filtered_match",
        "detector_or_sampling_miss",
    ])
].copy()

review_queue["manual_review_status"] = ""
review_queue["manual_review_note"] = ""
safe_to_csv(review_queue, OUT_REVIEW_QUEUE)

# Gallery for first clips and worst recall clips.
selected_scans = []
for s in list(clip_map.keys())[:6]:
    selected_scans.append(s)

worst_scans = clip_summary.sort_values(
    ["filtered_anchor_recall_strong_or_moderate", "anchor_object_count"],
    ascending=[True, False],
)["scan_frame_id"].head(14).tolist()

for s in worst_scans:
    if s not in selected_scans:
        selected_scans.append(s)

gallery_rows = []

roi_by_video = {clean_str(k): v.iloc[0].to_dict() for k, v in roi.groupby("video_id")}

for scan in selected_scans:
    clip = clip_map.get(scan)
    if clip is None:
        continue

    video_id = clean_str(clip.get("video_id"))
    roi_row = roi_by_video.get(video_id)

    anchor_frame, _ = get_anchor_frame(clip)
    raw = raw_by_scan.get(scan, pd.DataFrame())
    filt = filtered_by_scan.get(scan, pd.DataFrame())
    an = anchor_by_scan.get(scan, pd.DataFrame())
    audit = object_audit[object_audit["scan_frame_id"] == scan].copy()

    raw_nearest = nearest_frame(raw["frame_index_in_clip"].dropna().unique().tolist(), anchor_frame) if len(raw) else None
    filt_nearest = nearest_frame(filt["frame_index_in_clip"].dropna().unique().tolist(), anchor_frame) if len(filt) else None

    raw_frame = raw[raw["frame_index_in_clip"].astype("Int64") == raw_nearest].copy() if raw_nearest is not None else pd.DataFrame()
    filt_frame = filt[filt["frame_index_in_clip"].astype("Int64") == filt_nearest].copy() if filt_nearest is not None else pd.DataFrame()

    out_path = GALLERY / f"{scan}_anchor_tracking_recall.jpg"

    ok, msg = draw_gallery_image(
        scan=scan,
        clip=clip,
        anchors_df=an,
        raw_frame=raw_frame,
        kept_frame=filt_frame,
        audit_df=audit,
        roi_row=roi_row,
        out_path=out_path,
    )

    cs = clip_summary[clip_summary["scan_frame_id"] == scan].iloc[0].to_dict()

    gallery_rows.append({
        "scan_frame_id": scan,
        "video_id": video_id,
        "overlay_path": str(out_path),
        "saved": bool(ok),
        "message": msg,
        "filtered_recall": cs["filtered_anchor_recall_strong_or_moderate"],
        "raw_recall": cs["raw_anchor_recall_with_weak_or_roi_removed"],
        "clip_recall_status": cs["clip_recall_status"],
        "anchor_object_count": cs["anchor_object_count"],
        "detector_or_sampling_miss": cs["detector_or_sampling_miss"],
        "raw_match_removed_by_roi": cs["raw_match_removed_by_roi"],
    })

gallery_index = pd.DataFrame(gallery_rows)
safe_to_csv(gallery_index, OUT_GALLERY_INDEX)

# HTML.
html = []
html.append("<!doctype html>")
html.append("<html><head><meta charset='utf-8'>")
html.append("<title>Week 8 v50f Anchor Tracking Recall Gallery</title>")
html.append("<style>")
html.append("body{font-family:Arial;background:#111;color:#eee;margin:20px}")
html.append(".grid{display:grid;grid-template-columns:repeat(auto-fill,minmax(410px,1fr));gap:16px}")
html.append(".card{background:#1e1e1e;border:1px solid #333;border-radius:10px;padding:12px}")
html.append("img{width:100%;border:1px solid #444;border-radius:6px}")
html.append(".meta{font-size:13px;color:#bbb;line-height:1.4}")
html.append(".top{background:#2b2b2b;padding:12px;border-radius:10px;margin-bottom:18px}")
html.append("</style></head><body>")
html.append("<div class='top'>")
html.append("<h1>Week 8 v50f Anchor GT vs Tracking Recall Gallery</h1>")
html.append("<p>White boxes = ROI-filtered tracking. Red boxes = raw tracks removed by ROI. Anchor GT boxes are status-coloured.</p>")
html.append("<p>green=strong match, orange=moderate match, yellow=raw-only/ROI issue, magenta=miss.</p>")
html.append("<p>Track IDs are not final pig identities.</p>")
html.append("</div><div class='grid'>")

for _, r in gallery_index.iterrows():
    if not bool(r["saved"]):
        continue
    rel = Path(r["overlay_path"]).relative_to(OUT)
    html.append("<div class='card'>")
    html.append(f"<h3>{r['scan_frame_id']}</h3>")
    html.append(f"<img src='{rel.as_posix()}'>")
    html.append("<div class='meta'>")
    html.append(f"video: {r['video_id']}<br>")
    html.append(f"filtered recall: {float(r['filtered_recall']):.3f}<br>")
    html.append(f"raw recall: {float(r['raw_recall']):.3f}<br>")
    html.append(f"status: {r['clip_recall_status']}<br>")
    html.append(f"anchor objects: {r['anchor_object_count']}<br>")
    html.append(f"misses: {r['detector_or_sampling_miss']}<br>")
    html.append(f"ROI-removed matches: {r['raw_match_removed_by_roi']}<br>")
    html.append("</div></div>")

html.append("</div></body></html>")
OUT_GALLERY_HTML.write_text("\n".join(html))

# Decision.
total_anchor_objects = int(len(object_audit))
strong_count = int((object_audit["tracking_recall_status"] == "filtered_strong_match").sum())
moderate_count = int((object_audit["tracking_recall_status"] == "filtered_moderate_match").sum())
roi_removed_count = int((object_audit["tracking_recall_status"] == "raw_match_removed_by_roi").sum())
weak_count = int((object_audit["tracking_recall_status"] == "weak_filtered_match").sum())
miss_count = int((object_audit["tracking_recall_status"] == "detector_or_sampling_miss").sum())

filtered_recall_total = (strong_count + moderate_count) / total_anchor_objects if total_anchor_objects else 0.0
raw_recall_total = (strong_count + moderate_count + roi_removed_count + weak_count) / total_anchor_objects if total_anchor_objects else 0.0

low_recall_clip_count = int((clip_summary["clip_recall_status"] == "low_filtered_tracking_recall_review").sum())
moderate_recall_clip_count = int((clip_summary["clip_recall_status"] == "moderate_filtered_tracking_recall").sum())
good_recall_clip_count = int((clip_summary["clip_recall_status"] == "good_filtered_tracking_recall").sum())

if miss_count > 0:
    issues.append({
        "item": "anchor_tracking_recall",
        "issue_type": "warning_anchor_objects_without_tracking_match",
        "issue_detail": f"{miss_count} anchor GT objects have no useful raw/filtered tracking match at the nearest sampled anchor frame.",
        "severity": "warning",
    })

if roi_removed_count > 0:
    issues.append({
        "item": "roi_filter_false_removal_possible",
        "issue_type": "warning_raw_tracking_match_removed_by_roi",
        "issue_detail": f"{roi_removed_count} anchor GT objects have raw tracking overlap but were removed by ROI filtering.",
        "severity": "warning",
    })

if low_recall_clip_count > 0:
    issues.append({
        "item": "clip_level_tracking_recall",
        "issue_type": "warning_low_recall_clips",
        "issue_detail": f"{low_recall_clip_count} clips have low filtered tracking recall and need visual review.",
        "severity": "warning",
    })

issues_df = pd.DataFrame(issues, columns=["item", "issue_type", "issue_detail", "severity"])
hard_issues = issues_df[issues_df["severity"] == "hard"] if len(issues_df) else pd.DataFrame()
warnings = issues_df[issues_df["severity"] == "warning"] if len(issues_df) else pd.DataFrame()

# Ready for v51 as an interface layer, not as a final identity/complete-tracking claim.
ready_for_v51 = len(hard_issues) == 0

decision = pd.DataFrame([{
    "v50f_decision": "anchor_tracking_recall_audit_completed_use_tracking_as_optional_layer" if ready_for_v51 else "anchor_tracking_recall_audit_blocked",
    "anchor_object_count": int(total_anchor_objects),
    "filtered_strong_match_count": int(strong_count),
    "filtered_moderate_match_count": int(moderate_count),
    "raw_match_removed_by_roi_count": int(roi_removed_count),
    "weak_filtered_match_count": int(weak_count),
    "detector_or_sampling_miss_count": int(miss_count),
    "filtered_anchor_recall_total": float(filtered_recall_total),
    "raw_anchor_recall_total_with_weak_or_roi_removed": float(raw_recall_total),
    "good_recall_clip_count": int(good_recall_clip_count),
    "moderate_recall_clip_count": int(moderate_recall_clip_count),
    "low_recall_clip_count": int(low_recall_clip_count),
    "review_queue_rows": int(len(review_queue)),
    "gallery_overlay_count": int(len(gallery_index)),
    "hard_issue_count": int(len(hard_issues)),
    "warning_count": int(len(warnings)),
    "issue_count": int(len(issues_df)),
    "tracking_ids_final_identity_claim": False,
    "tracking_boxes_complete_gt_claim": False,
    "ready_for_v51_tracking_integrated_visualizer": bool(ready_for_v51),
    "generated_at": datetime.now().isoformat(timespec="seconds"),
}])

safe_to_csv(issues_df, OUT_ISSUES)
safe_to_csv(decision, OUT_DECISION)

report = (
    "Week 8 v50f Anchor Tracking Recall Audit Report\n\n"
    f"Decision: {decision.iloc[0]['v50f_decision']}\n"
    f"Anchor objects: {total_anchor_objects}\n"
    f"Filtered strong matches: {strong_count}\n"
    f"Filtered moderate matches: {moderate_count}\n"
    f"Raw matches removed by ROI: {roi_removed_count}\n"
    f"Weak filtered matches: {weak_count}\n"
    f"Detector/sampling misses: {miss_count}\n"
    f"Filtered anchor recall: {filtered_recall_total:.4f}\n"
    f"Raw anchor recall with weak/ROI-removed: {raw_recall_total:.4f}\n"
    f"Low recall clips: {low_recall_clip_count}\n"
    f"Review queue rows: {len(review_queue)}\n"
    f"Hard issues: {len(hard_issues)}\n"
    f"Warnings: {len(warnings)}\n\n"
    "Interpretation:\n"
    "This audit checks whether every anchor GT pig has a nearby tracking box. Tracking can be integrated into the visualizer as an optional full-clip layer, but it must not be treated as complete ground truth or final pig identity.\n\n"
    "Next:\n"
    "Use v51 to build a tracking-integrated visualizer with clear layer labels: Anchor GT, ROI-filtered tracking, missed-anchor warnings, and manual review notes.\n"
)
OUT_REPORT.write_text(report)

OUT_NOTE.write_text(
    "# Week 8 v50f Anchor Tracking Recall Audit\n\n"
    "## Summary\n\n"
    f"- v50f decision: {decision.iloc[0]['v50f_decision']}\n"
    f"- Anchor objects: {total_anchor_objects}\n"
    f"- Filtered strong matches: {strong_count}\n"
    f"- Filtered moderate matches: {moderate_count}\n"
    f"- Raw matches removed by ROI: {roi_removed_count}\n"
    f"- Weak filtered matches: {weak_count}\n"
    f"- Detector/sampling misses: {miss_count}\n"
    f"- Filtered anchor recall: {filtered_recall_total:.4f}\n"
    f"- Raw anchor recall with weak/ROI-removed: {raw_recall_total:.4f}\n"
    f"- Good recall clips: {good_recall_clip_count}\n"
    f"- Moderate recall clips: {moderate_recall_clip_count}\n"
    f"- Low recall clips: {low_recall_clip_count}\n"
    f"- Review queue rows: {len(review_queue)}\n"
    f"- Hard issue count: {len(hard_issues)}\n"
    f"- Warning count: {len(warnings)}\n"
    f"- Ready for v51 tracking-integrated visualizer: {ready_for_v51}\n\n"
    "## Important limitation\n\n"
    "Tracking boxes should be shown as an optional visualization layer, not as complete ground truth and not as final pig identity.\n\n"
    f"Gallery: {OUT_GALLERY_HTML}\n"
)

progress_row = pd.DataFrame([{
    "date": datetime.now().date().isoformat(),
    "stage": "v50f",
    "task_name": "Anchor GT vs tracking recall audit",
    "status": "PASS" if ready_for_v51 else "BLOCKED",
    "input_summary": str(V50E_FILTERED_TRACKS),
    "output_summary": str(OUT),
    "hard_issues": int(len(hard_issues)),
    "warnings": int(len(warnings)),
    "next_action": "v51 tracking-integrated visualizer with limitations" if ready_for_v51 else "Resolve recall audit hard issues.",
}])

if OUT_PROGRESS.exists():
    old = pd.read_csv(OUT_PROGRESS)
    progress = pd.concat([old, progress_row], ignore_index=True)
else:
    progress = progress_row

safe_to_csv(progress, OUT_PROGRESS)

print("Saved:")
print(OUT_OBJECT_AUDIT)
print(OUT_CLIP_SUMMARY)
print(OUT_REVIEW_QUEUE)
print(OUT_GALLERY_INDEX)
print(OUT_GALLERY_HTML)
print(OUT_DECISION)
print(OUT_ISSUES)
print(OUT_REPORT)
print(OUT_NOTE)

print()
print("=== v50f decision ===")
print(decision.to_string(index=False))

print()
print("=== v50f issues ===")
if len(issues_df):
    print(issues_df.to_string(index=False))
else:
    print("No issues found.")

print()
print("=== recall status counts ===")
print(object_audit["tracking_recall_status"].value_counts(dropna=False).to_string())

print()
print("=== clip recall status counts ===")
print(clip_summary["clip_recall_status"].value_counts(dropna=False).to_string())

print()
print("=== worst recall clips ===")
print(clip_summary.sort_values("filtered_anchor_recall_strong_or_moderate").head(15).to_string(index=False))
