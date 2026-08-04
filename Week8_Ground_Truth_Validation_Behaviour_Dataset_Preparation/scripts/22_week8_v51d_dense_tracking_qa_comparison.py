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

V51B_TRACKS = W8 / "outputs" / "v51b_dense_full_72_tracking" / "week8_v51b_dense_full_tracking_rows.csv"
V51B_SUMMARY = W8 / "outputs" / "v51b_dense_full_72_tracking" / "week8_v51b_dense_clip_tracking_summary.csv"

V51C_TRACKS = W8 / "outputs" / "v51c_nms_corrected_dense_tracking" / "week8_v51c_nms_dense_tracking_rows.csv"
V51C_SUMMARY = W8 / "outputs" / "v51c_nms_corrected_dense_tracking" / "week8_v51c_nms_dense_clip_summary.csv"
V51C_DECISION = W8 / "outputs" / "v51c_nms_corrected_dense_tracking" / "week8_v51c_decision_summary.csv"

OUT = W8 / "outputs" / "v51d_dense_tracking_qa_comparison"
GALLERY = OUT / "comparison_gallery"
REPORTS = W8 / "reports"
NOTES = W8 / "notes"
PROGRESS = W8 / "progress"

for p in [OUT, GALLERY, REPORTS, NOTES, PROGRESS]:
    p.mkdir(parents=True, exist_ok=True)

OUT_SOURCE_SUMMARY = OUT / "week8_v51d_tracking_source_summary.csv"
OUT_OBJECT_RECALL = OUT / "week8_v51d_anchor_object_recall_by_source.csv"
OUT_CLIP_RECALL = OUT / "week8_v51d_clip_recall_by_source.csv"
OUT_SELECTED = OUT / "week8_v51d_selected_tracking_source.csv"
OUT_GALLERY_INDEX = OUT / "week8_v51d_comparison_gallery_index.csv"
OUT_GALLERY_HTML = OUT / "index.html"
OUT_DECISION = OUT / "week8_v51d_decision_summary.csv"
OUT_ISSUES = OUT / "week8_v51d_issues.csv"
OUT_REPORT = REPORTS / "week8_v51d_dense_tracking_qa_comparison_report.md"
OUT_NOTE = NOTES / "week8_v51d_dense_tracking_qa_comparison_notes.md"
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


def nearest_frame(frames, target):
    vals = [int(x) for x in frames if pd.notna(x)]
    if not vals:
        return None
    arr = np.array(sorted(vals))
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


def best_match(anchor_box, track_frame):
    best_iou = 0.0
    best_track_id = ""
    best_score = None

    for _, tr in track_frame.iterrows():
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


def recall_status(best_iou):
    if best_iou >= 0.50:
        return "strong"
    if best_iou >= 0.30:
        return "moderate"
    if best_iou >= 0.10:
        return "weak"
    return "miss"


def evaluate_source(source_name, tracks, summary, anchors, clip_map):
    tracks = tracks.copy()
    anchors = anchors.copy()

    tracks["scan_frame_id"] = tracks["scan_frame_id"].astype(str).map(clean_str)
    tracks["frame_index_in_clip"] = pd.to_numeric(tracks["frame_index_in_clip"], errors="coerce")
    tracks["track_id"] = tracks["track_id"].astype(str).map(clean_str)

    for c in ["x1", "y1", "x2", "y2", "score"]:
        tracks[c] = pd.to_numeric(tracks[c], errors="coerce")

    anchors["scan_frame_id"] = anchors["scan_frame_id"].astype(str).map(clean_str)
    for c in ["bbox_x1", "bbox_y1", "bbox_x2", "bbox_y2"]:
        anchors[c] = pd.to_numeric(anchors[c], errors="coerce")

    track_by_scan = {str(k): v.copy() for k, v in tracks.groupby("scan_frame_id")}
    anchor_by_scan = {str(k): v.copy() for k, v in anchors.groupby("scan_frame_id")}

    object_rows = []
    clip_rows = []

    for scan, clip in clip_map.items():
        anchor_frame, anchor_rel_sec = get_anchor_frame(clip)
        an = anchor_by_scan.get(scan, pd.DataFrame())
        tr = track_by_scan.get(scan, pd.DataFrame())

        if len(tr):
            nf = nearest_frame(tr["frame_index_in_clip"].dropna().unique().tolist(), anchor_frame)
            tr_frame = tr[tr["frame_index_in_clip"].astype("Int64") == int(nf)].copy() if nf is not None else pd.DataFrame()
        else:
            nf = None
            tr_frame = pd.DataFrame()

        duplicate_pressure = len(tr_frame) / max(1, len(an)) if len(an) else 0.0

        strong = 0
        moderate = 0
        weak = 0
        miss = 0

        for _, a in an.iterrows():
            abox = [
                to_float(a.get("bbox_x1")),
                to_float(a.get("bbox_y1")),
                to_float(a.get("bbox_x2")),
                to_float(a.get("bbox_y2")),
            ]

            if not all(v is not None for v in abox):
                continue

            best_iou, best_track_id, best_score = best_match(abox, tr_frame)
            status = recall_status(best_iou)

            if status == "strong":
                strong += 1
            elif status == "moderate":
                moderate += 1
            elif status == "weak":
                weak += 1
            else:
                miss += 1

            object_rows.append({
                "source_name": source_name,
                "scan_frame_id": scan,
                "video_id": clean_str(clip.get("video_id")),
                "final_box_id": clean_str(a.get("final_box_id")),
                "behaviour_pig_id": clean_str(a.get("behaviour_pig_id")),
                "visual_marker_colour": clean_str(a.get("visual_marker_colour")),
                "behaviour_code": clean_str(a.get("behaviour_code")),
                "anchor_frame_expected": anchor_frame,
                "nearest_tracking_frame": nf,
                "frame_delta": None if nf is None else int(nf - anchor_frame),
                "track_boxes_at_nearest_frame": int(len(tr_frame)),
                "duplicate_pressure": float(duplicate_pressure),
                "best_track_id": best_track_id,
                "best_score": best_score,
                "best_iou": float(best_iou),
                "match_status": status,
            })

        total = strong + moderate + weak + miss
        strong_moderate_recall = (strong + moderate) / total if total else 0.0
        weak_inclusive_recall = (strong + moderate + weak) / total if total else 0.0

        if strong_moderate_recall >= 0.85:
            clip_status = "good"
        elif strong_moderate_recall >= 0.65:
            clip_status = "moderate"
        else:
            clip_status = "low_review"

        clip_rows.append({
            "source_name": source_name,
            "scan_frame_id": scan,
            "video_id": clean_str(clip.get("video_id")),
            "anchor_object_count": int(total),
            "track_boxes_at_anchor_frame": int(len(tr_frame)),
            "duplicate_pressure": float(duplicate_pressure),
            "strong_count": int(strong),
            "moderate_count": int(moderate),
            "weak_count": int(weak),
            "miss_count": int(miss),
            "strong_moderate_recall": float(strong_moderate_recall),
            "weak_inclusive_recall": float(weak_inclusive_recall),
            "clip_recall_status": clip_status,
        })

    object_df = pd.DataFrame(object_rows)
    clip_df = pd.DataFrame(clip_rows)

    total_objects = len(object_df)
    strong_total = int((object_df["match_status"] == "strong").sum())
    moderate_total = int((object_df["match_status"] == "moderate").sum())
    weak_total = int((object_df["match_status"] == "weak").sum())
    miss_total = int((object_df["match_status"] == "miss").sum())

    strong_moderate_recall = (strong_total + moderate_total) / total_objects if total_objects else 0.0
    weak_inclusive_recall = (strong_total + moderate_total + weak_total) / total_objects if total_objects else 0.0
    miss_rate = miss_total / total_objects if total_objects else 1.0

    rows_total = int(len(tracks))
    unique_track_sum = int(tracks.groupby("scan_frame_id")["track_id"].nunique().sum()) if len(tracks) else 0
    avg_duplicate_pressure = float(clip_df["duplicate_pressure"].mean()) if len(clip_df) else 0.0
    median_duplicate_pressure = float(clip_df["duplicate_pressure"].median()) if len(clip_df) else 0.0
    low_recall_clips = int((clip_df["clip_recall_status"] == "low_review").sum())

    source_summary = {
        "source_name": source_name,
        "tracking_rows_total": rows_total,
        "unique_track_total_by_clip_sum": unique_track_sum,
        "anchor_object_count": int(total_objects),
        "strong_match_count": strong_total,
        "moderate_match_count": moderate_total,
        "weak_match_count": weak_total,
        "miss_count": miss_total,
        "strong_moderate_recall": float(strong_moderate_recall),
        "weak_inclusive_recall": float(weak_inclusive_recall),
        "miss_rate": float(miss_rate),
        "avg_duplicate_pressure": avg_duplicate_pressure,
        "median_duplicate_pressure": median_duplicate_pressure,
        "low_recall_clip_count": low_recall_clips,
    }

    return source_summary, object_df, clip_df


def draw_comparison(scan, clip, anchors_df, raw_tracks, nms_tracks, raw_obj, nms_obj, out_path):
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
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    frame_idx = max(0, min(frame_count - 1, anchor_frame))
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
    ok, frame = cap.read()
    cap.release()

    if not ok or frame is None:
        return False, "frame_read_failed"

    def nearest_tracks_at_anchor(df):
        if len(df) == 0:
            return pd.DataFrame()
        frames = df["frame_index_in_clip"].dropna().astype(int).unique().tolist()
        nf = nearest_frame(frames, anchor_frame)
        if nf is None:
            return pd.DataFrame()
        return df[df["frame_index_in_clip"].astype("Int64") == int(nf)].copy()

    raw_frame = nearest_tracks_at_anchor(raw_tracks)
    nms_frame = nearest_tracks_at_anchor(nms_tracks)

    def render_panel(base, source_label, tracks_frame, obj_eval):
        img = base.copy()
        h, w = img.shape[:2]

        # Draw tracking boxes first.
        for _, tr in tracks_frame.iterrows():
            x1 = to_int(tr.get("x1"))
            y1 = to_int(tr.get("y1"))
            x2 = to_int(tr.get("x2"))
            y2 = to_int(tr.get("y2"))
            if None in [x1, y1, x2, y2]:
                continue
            cv2.rectangle(img, (x1, y1), (x2, y2), (255, 255, 255), 1)
            cv2.putText(img, f"T{clean_str(tr.get('track_id'))}", (x1, min(h - 4, y2 + 12)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.36, (255, 255, 255), 1)

        status_bgr = {
            "strong": (40, 220, 40),
            "moderate": (0, 180, 255),
            "weak": (180, 180, 180),
            "miss": (255, 0, 255),
        }

        eval_by_box = {clean_str(r["final_box_id"]): r for _, r in obj_eval.iterrows()}

        for _, a in anchors_df.iterrows():
            x1 = to_int(a.get("bbox_x1"))
            y1 = to_int(a.get("bbox_y1"))
            x2 = to_int(a.get("bbox_x2"))
            y2 = to_int(a.get("bbox_y2"))
            if None in [x1, y1, x2, y2]:
                continue

            box_id = clean_str(a.get("final_box_id"))
            status = clean_str(eval_by_box.get(box_id, {}).get("match_status")) or "miss"
            bgr = status_bgr.get(status, (200, 200, 200))

            cv2.rectangle(img, (x1, y1), (x2, y2), bgr, 3)
            cv2.putText(img, status, (x1, max(18, y1 - 7)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, bgr, 2)

        cv2.rectangle(img, (0, 0), (w, 54), (0, 0, 0), -1)
        cv2.putText(img, source_label, (10, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 2)
        cv2.putText(img, "green=strong orange=moderate gray=weak magenta=miss",
                    (10, 46), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (230, 230, 230), 1)
        return img

    raw_panel = render_panel(frame, "v51b raw dense", raw_frame, raw_obj)
    nms_panel = render_panel(frame, "v51c NMS-corrected dense", nms_frame, nms_obj)

    combined = np.hstack([raw_panel, nms_panel])
    ok = cv2.imwrite(str(out_path), combined)
    return bool(ok), "saved" if ok else "write_failed"


issues = []

required = [V45_CLIP_JSON, V45_ANCHORS, V51B_TRACKS, V51B_SUMMARY, V51C_TRACKS, V51C_SUMMARY, V51C_DECISION]
for p in required:
    if not p.exists():
        issues.append({
            "item": str(p),
            "issue_type": "hard_missing_required_input",
            "issue_detail": "Required artifact is missing.",
            "severity": "hard",
        })

if issues:
    issues_df = pd.DataFrame(issues)
    safe_to_csv(issues_df, OUT_ISSUES)
    decision = pd.DataFrame([{
        "v51d_decision": "dense_tracking_qa_blocked_missing_inputs",
        "hard_issue_count": int((issues_df["severity"] == "hard").sum()),
        "ready_for_v52_tracking_integrated_visualizer": False,
        "generated_at": datetime.now().isoformat(timespec="seconds"),
    }])
    safe_to_csv(decision, OUT_DECISION)
    print(decision.to_string(index=False))
    raise SystemExit(1)


clips = json.loads(V45_CLIP_JSON.read_text()).get("clips", [])
clip_map = {clean_str(c.get("scan_frame_id")): c for c in clips}

anchors = pd.read_csv(V45_ANCHORS)
v51b_tracks = pd.read_csv(V51B_TRACKS)
v51b_summary = pd.read_csv(V51B_SUMMARY)
v51c_tracks = pd.read_csv(V51C_TRACKS)
v51c_summary = pd.read_csv(V51C_SUMMARY)

source_summaries = []
object_dfs = []
clip_dfs = []

for name, tr, sm in [
    ("v51b_raw_dense", v51b_tracks, v51b_summary),
    ("v51c_nms_corrected_dense", v51c_tracks, v51c_summary),
]:
    ss, odf, cdf = evaluate_source(name, tr, sm, anchors, clip_map)
    source_summaries.append(ss)
    object_dfs.append(odf)
    clip_dfs.append(cdf)

source_summary = pd.DataFrame(source_summaries)
object_recall = pd.concat(object_dfs, ignore_index=True)
clip_recall = pd.concat(clip_dfs, ignore_index=True)

safe_to_csv(source_summary, OUT_SOURCE_SUMMARY)
safe_to_csv(object_recall, OUT_OBJECT_RECALL)
safe_to_csv(clip_recall, OUT_CLIP_RECALL)

b = source_summary[source_summary["source_name"] == "v51b_raw_dense"].iloc[0].to_dict()
c = source_summary[source_summary["source_name"] == "v51c_nms_corrected_dense"].iloc[0].to_dict()

recall_drop = float(b["strong_moderate_recall"] - c["strong_moderate_recall"])
weak_recall_drop = float(b["weak_inclusive_recall"] - c["weak_inclusive_recall"])
row_reduction = int(b["tracking_rows_total"] - c["tracking_rows_total"])
track_reduction = int(b["unique_track_total_by_clip_sum"] - c["unique_track_total_by_clip_sum"])
duplicate_reduction = float(b["avg_duplicate_pressure"] - c["avg_duplicate_pressure"])

# Selection rule:
# Prefer NMS if it reduces duplicate/track burden while keeping recall close.
# If NMS loses too much recall, keep raw dense for maximum recall.
if (
    recall_drop <= 0.03
    and weak_recall_drop <= 0.03
    and row_reduction >= 0
    and duplicate_reduction >= -0.10
):
    selected_name = "v51c_nms_corrected_dense"
    selected_reason = "NMS-corrected dense tracking keeps recall close while reducing detections/tracks."
    selected_tracks_path = str(V51C_TRACKS)
elif c["strong_moderate_recall"] >= b["strong_moderate_recall"]:
    selected_name = "v51c_nms_corrected_dense"
    selected_reason = "NMS-corrected dense tracking has equal or better strong/moderate recall."
    selected_tracks_path = str(V51C_TRACKS)
else:
    selected_name = "v51b_raw_dense"
    selected_reason = "Raw dense tracking preserves higher recall; NMS loses too much anchor recall."
    selected_tracks_path = str(V51B_TRACKS)

selected_row = source_summary[source_summary["source_name"] == selected_name].iloc[0].to_dict()

selected = pd.DataFrame([{
    "selected_tracking_source": selected_name,
    "selected_tracks_path": selected_tracks_path,
    "selection_reason": selected_reason,
    "selected_strong_moderate_recall": float(selected_row["strong_moderate_recall"]),
    "selected_weak_inclusive_recall": float(selected_row["weak_inclusive_recall"]),
    "selected_miss_rate": float(selected_row["miss_rate"]),
    "selected_avg_duplicate_pressure": float(selected_row["avg_duplicate_pressure"]),
    "selected_tracking_rows_total": int(selected_row["tracking_rows_total"]),
    "selected_unique_track_total_by_clip_sum": int(selected_row["unique_track_total_by_clip_sum"]),
    "v51b_strong_moderate_recall": float(b["strong_moderate_recall"]),
    "v51c_strong_moderate_recall": float(c["strong_moderate_recall"]),
    "recall_drop_v51b_minus_v51c": recall_drop,
    "weak_recall_drop_v51b_minus_v51c": weak_recall_drop,
    "row_reduction_v51b_minus_v51c": row_reduction,
    "track_reduction_v51b_minus_v51c": track_reduction,
    "duplicate_pressure_reduction_v51b_minus_v51c": duplicate_reduction,
    "tracking_ids_final_identity_claim": False,
    "tracking_boxes_complete_gt_claim": False,
}])

safe_to_csv(selected, OUT_SELECTED)

# Gallery selection: first clips + worst selected-source clips + biggest disagreement clips.
selected_clip = clip_recall[clip_recall["source_name"] == selected_name].copy()
worst = selected_clip.sort_values("strong_moderate_recall").head(10)["scan_frame_id"].tolist()
firsts = list(clip_map.keys())[:6]

comparison = clip_recall.pivot(index="scan_frame_id", columns="source_name", values="strong_moderate_recall").reset_index()
comparison["abs_recall_diff"] = (
    comparison["v51b_raw_dense"] - comparison["v51c_nms_corrected_dense"]
).abs()
diffs = comparison.sort_values("abs_recall_diff", ascending=False).head(8)["scan_frame_id"].tolist()

selected_scans = []
for s in firsts + worst + diffs:
    if s not in selected_scans:
        selected_scans.append(s)

anchor_by_scan = {str(k): v.copy() for k, v in anchors.groupby("scan_frame_id")}
raw_by_scan = {str(k): v.copy() for k, v in v51b_tracks.groupby("scan_frame_id")}
nms_by_scan = {str(k): v.copy() for k, v in v51c_tracks.groupby("scan_frame_id")}

gallery_rows = []

for scan in selected_scans:
    clip = clip_map.get(scan)
    if clip is None:
        continue

    an = anchor_by_scan.get(scan, pd.DataFrame())
    raw = raw_by_scan.get(scan, pd.DataFrame())
    nms = nms_by_scan.get(scan, pd.DataFrame())

    raw_obj = object_recall[
        (object_recall["source_name"] == "v51b_raw_dense")
        & (object_recall["scan_frame_id"] == scan)
    ].copy()

    nms_obj = object_recall[
        (object_recall["source_name"] == "v51c_nms_corrected_dense")
        & (object_recall["scan_frame_id"] == scan)
    ].copy()

    out_img = GALLERY / f"{scan}_v51b_vs_v51c.jpg"

    ok, msg = draw_comparison(
        scan=scan,
        clip=clip,
        anchors_df=an,
        raw_tracks=raw,
        nms_tracks=nms,
        raw_obj=raw_obj,
        nms_obj=nms_obj,
        out_path=out_img,
    )

    raw_clip_row = clip_recall[
        (clip_recall["source_name"] == "v51b_raw_dense")
        & (clip_recall["scan_frame_id"] == scan)
    ].iloc[0].to_dict()

    nms_clip_row = clip_recall[
        (clip_recall["source_name"] == "v51c_nms_corrected_dense")
        & (clip_recall["scan_frame_id"] == scan)
    ].iloc[0].to_dict()

    gallery_rows.append({
        "scan_frame_id": scan,
        "overlay_path": str(out_img),
        "saved": bool(ok),
        "message": msg,
        "v51b_recall": raw_clip_row["strong_moderate_recall"],
        "v51c_recall": nms_clip_row["strong_moderate_recall"],
        "v51b_duplicate_pressure": raw_clip_row["duplicate_pressure"],
        "v51c_duplicate_pressure": nms_clip_row["duplicate_pressure"],
    })

gallery_index = pd.DataFrame(gallery_rows)
safe_to_csv(gallery_index, OUT_GALLERY_INDEX)

# HTML gallery.
html = []
html.append("<!doctype html>")
html.append("<html><head><meta charset='utf-8'>")
html.append("<title>Week 8 v51d Dense Tracking Comparison</title>")
html.append("<style>")
html.append("body{font-family:Arial;background:#111;color:#eee;margin:20px}")
html.append(".grid{display:grid;grid-template-columns:repeat(auto-fill,minmax(680px,1fr));gap:18px}")
html.append(".card{background:#1e1e1e;border:1px solid #333;border-radius:10px;padding:12px}")
html.append("img{width:100%;border:1px solid #444;border-radius:6px}")
html.append(".meta{font-size:13px;color:#bbb;line-height:1.4}")
html.append(".top{background:#2b2b2b;padding:12px;border-radius:10px;margin-bottom:18px}")
html.append("</style></head><body>")
html.append("<div class='top'>")
html.append("<h1>Week 8 v51d Dense Tracking QA Comparison</h1>")
html.append("<p>Left = v51b raw dense. Right = v51c NMS-corrected dense.</p>")
html.append("<p>White boxes = tracking boxes. Anchor GT boxes are status-coloured: green strong, orange moderate, gray weak, magenta miss.</p>")
html.append(f"<p>Selected source: <b>{selected_name}</b>. Reason: {selected_reason}</p>")
html.append("</div><div class='grid'>")

for _, r in gallery_index.iterrows():
    if not bool(r["saved"]):
        continue
    rel = Path(r["overlay_path"]).relative_to(OUT)
    html.append("<div class='card'>")
    html.append(f"<h3>{r['scan_frame_id']}</h3>")
    html.append(f"<img src='{rel.as_posix()}'>")
    html.append("<div class='meta'>")
    html.append(f"v51b recall: {float(r['v51b_recall']):.3f}<br>")
    html.append(f"v51c recall: {float(r['v51c_recall']):.3f}<br>")
    html.append(f"v51b duplicate pressure: {float(r['v51b_duplicate_pressure']):.3f}<br>")
    html.append(f"v51c duplicate pressure: {float(r['v51c_duplicate_pressure']):.3f}<br>")
    html.append("</div></div>")

html.append("</div></body></html>")
OUT_GALLERY_HTML.write_text("\n".join(html))

# Issues.
if float(selected_row["strong_moderate_recall"]) < 0.75:
    issues.append({
        "item": "selected_tracking_recall",
        "issue_type": "warning_selected_tracking_recall_below_0_75",
        "issue_detail": f"Selected strong/moderate recall is {float(selected_row['strong_moderate_recall']):.4f}.",
        "severity": "warning",
    })

if float(selected_row["avg_duplicate_pressure"]) > 2.0:
    issues.append({
        "item": "selected_duplicate_pressure",
        "issue_type": "warning_selected_duplicate_pressure_high",
        "issue_detail": f"Selected average duplicate pressure is {float(selected_row['avg_duplicate_pressure']):.4f}.",
        "severity": "warning",
    })

issues.append({
    "item": "tracking_identity_claim",
    "issue_type": "info_no_final_identity_claim",
    "issue_detail": "Tracking IDs are not final pig identities; identity still requires colour/anchor review.",
    "severity": "info",
})

issues_df = pd.DataFrame(issues, columns=["item", "issue_type", "issue_detail", "severity"])
hard_issues = issues_df[issues_df["severity"] == "hard"] if len(issues_df) else pd.DataFrame()
warnings = issues_df[issues_df["severity"] == "warning"] if len(issues_df) else pd.DataFrame()
infos = issues_df[issues_df["severity"] == "info"] if len(issues_df) else pd.DataFrame()

ready = len(hard_issues) == 0

decision = pd.DataFrame([{
    "v51d_decision": "dense_tracking_qa_completed_selected_source_ready_for_visualizer" if ready else "dense_tracking_qa_blocked",
    "selected_tracking_source": selected_name,
    "selected_tracks_path": selected_tracks_path,
    "selection_reason": selected_reason,
    "v51b_strong_moderate_recall": float(b["strong_moderate_recall"]),
    "v51c_strong_moderate_recall": float(c["strong_moderate_recall"]),
    "v51b_weak_inclusive_recall": float(b["weak_inclusive_recall"]),
    "v51c_weak_inclusive_recall": float(c["weak_inclusive_recall"]),
    "v51b_tracking_rows": int(b["tracking_rows_total"]),
    "v51c_tracking_rows": int(c["tracking_rows_total"]),
    "v51b_unique_track_sum": int(b["unique_track_total_by_clip_sum"]),
    "v51c_unique_track_sum": int(c["unique_track_total_by_clip_sum"]),
    "row_reduction_v51b_minus_v51c": int(row_reduction),
    "track_reduction_v51b_minus_v51c": int(track_reduction),
    "gallery_overlay_count": int(len(gallery_index)),
    "hard_issue_count": int(len(hard_issues)),
    "warning_count": int(len(warnings)),
    "info_count": int(len(infos)),
    "issue_count": int(len(issues_df)),
    "tracking_ids_final_identity_claim": False,
    "tracking_boxes_complete_gt_claim": False,
    "ready_for_v52_tracking_integrated_visualizer": bool(ready),
    "generated_at": datetime.now().isoformat(timespec="seconds"),
}])

safe_to_csv(issues_df, OUT_ISSUES)
safe_to_csv(decision, OUT_DECISION)

report = (
    "Week 8 v51d Dense Tracking QA Comparison Report\n\n"
    f"Decision: {decision.iloc[0]['v51d_decision']}\n"
    f"Selected source: {selected_name}\n"
    f"Selection reason: {selected_reason}\n\n"
    f"v51b strong/moderate recall: {float(b['strong_moderate_recall']):.4f}\n"
    f"v51c strong/moderate recall: {float(c['strong_moderate_recall']):.4f}\n"
    f"v51b weak-inclusive recall: {float(b['weak_inclusive_recall']):.4f}\n"
    f"v51c weak-inclusive recall: {float(c['weak_inclusive_recall']):.4f}\n"
    f"v51b tracking rows: {int(b['tracking_rows_total'])}\n"
    f"v51c tracking rows: {int(c['tracking_rows_total'])}\n"
    f"v51b unique track sum: {int(b['unique_track_total_by_clip_sum'])}\n"
    f"v51c unique track sum: {int(c['unique_track_total_by_clip_sum'])}\n"
    f"Row reduction: {row_reduction}\n"
    f"Track reduction: {track_reduction}\n\n"
    "Interpretation:\n"
    "This QA compares raw dense tracking and NMS-corrected dense tracking against anchor GT objects. "
    "The selected source should be used in the next visualizer, but tracking IDs are not final pig identities.\n"
)

OUT_REPORT.write_text(report)

OUT_NOTE.write_text(
    "# Week 8 v51d Dense Tracking QA Comparison\n\n"
    "## Summary\n\n"
    f"- v51d decision: {decision.iloc[0]['v51d_decision']}\n"
    f"- Selected tracking source: {selected_name}\n"
    f"- Selection reason: {selected_reason}\n"
    f"- v51b strong/moderate recall: {float(b['strong_moderate_recall']):.4f}\n"
    f"- v51c strong/moderate recall: {float(c['strong_moderate_recall']):.4f}\n"
    f"- v51b weak-inclusive recall: {float(b['weak_inclusive_recall']):.4f}\n"
    f"- v51c weak-inclusive recall: {float(c['weak_inclusive_recall']):.4f}\n"
    f"- v51b tracking rows: {int(b['tracking_rows_total'])}\n"
    f"- v51c tracking rows: {int(c['tracking_rows_total'])}\n"
    f"- v51b unique track sum: {int(b['unique_track_total_by_clip_sum'])}\n"
    f"- v51c unique track sum: {int(c['unique_track_total_by_clip_sum'])}\n"
    f"- Row reduction: {row_reduction}\n"
    f"- Track reduction: {track_reduction}\n"
    f"- Hard issue count: {len(hard_issues)}\n"
    f"- Warning count: {len(warnings)}\n"
    f"- Ready for v52 visualizer: {ready}\n\n"
    "## Important limitation\n\n"
    "Tracking IDs are not final pig identities. The selected tracking source is for improved full-clip visualization and review.\n"
    f"\nGallery: {OUT_GALLERY_HTML}\n"
)

progress_row = pd.DataFrame([{
    "date": datetime.now().date().isoformat(),
    "stage": "v51d",
    "task_name": "Dense tracking QA comparison and source selection",
    "status": "PASS" if ready else "BLOCKED",
    "input_summary": str(V51C_TRACKS),
    "output_summary": str(OUT),
    "hard_issues": int(len(hard_issues)),
    "warnings": int(len(warnings)),
    "next_action": "v52 tracking-integrated visualizer using selected source" if ready else "Resolve dense tracking QA hard issues.",
}])

if OUT_PROGRESS.exists():
    old = pd.read_csv(OUT_PROGRESS)
    progress = pd.concat([old, progress_row], ignore_index=True)
else:
    progress = progress_row

safe_to_csv(progress, OUT_PROGRESS)

print("Saved:")
print(OUT_SOURCE_SUMMARY)
print(OUT_OBJECT_RECALL)
print(OUT_CLIP_RECALL)
print(OUT_SELECTED)
print(OUT_GALLERY_INDEX)
print(OUT_GALLERY_HTML)
print(OUT_DECISION)
print(OUT_ISSUES)
print(OUT_REPORT)
print(OUT_NOTE)

print()
print("=== v51d decision ===")
print(decision.to_string(index=False))

print()
print("=== v51d source summary ===")
print(source_summary.to_string(index=False))

print()
print("=== v51d selected ===")
print(selected.to_string(index=False))

print()
print("=== v51d issues ===")
print(issues_df.to_string(index=False))
