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
V45_CLIP_OBJECTS = W8 / "outputs" / "propagated_ground_truth" / "v45_label_propagation" / "week8_v45_clip_object_propagated_annotations.csv"
V49C_DECISION = W8 / "outputs" / "v49c_anchor_validation_verdict" / "week8_v49c_decision_summary.csv"
V50C_TRACKS = W8 / "outputs" / "v50c_full_72_clip_cpu_tracking" / "week8_v50c_full_tracking_rows.csv"
V50C_SUMMARY = W8 / "outputs" / "v50c_full_72_clip_cpu_tracking" / "week8_v50c_clip_tracking_summary.csv"
V50C_DECISION = W8 / "outputs" / "v50c_full_72_clip_cpu_tracking" / "week8_v50c_decision_summary.csv"

OUT = W8 / "outputs" / "v50d_tracking_qa_visual_diagnostics"
OVERLAYS = OUT / "tracking_anchor_overlay_gallery"
REPORTS = W8 / "reports"
NOTES = W8 / "notes"
PROGRESS = W8 / "progress"

for p in [OUT, OVERLAYS, REPORTS, NOTES, PROGRESS]:
    p.mkdir(parents=True, exist_ok=True)

OUT_COVERAGE_QA = OUT / "week8_v50d_tracking_coverage_qa.csv"
OUT_BBOX_QA = OUT / "week8_v50d_tracking_bbox_qa.csv"
OUT_TRACK_FRAGMENTATION = OUT / "week8_v50d_track_fragmentation_summary.csv"
OUT_ANCHOR_MATCH = OUT / "week8_v50d_anchor_tracking_match_qa.csv"
OUT_OVERLAY_INDEX = OUT / "week8_v50d_overlay_gallery_index.csv"
OUT_GALLERY = OUT / "index.html"
OUT_DECISION = OUT / "week8_v50d_decision_summary.csv"
OUT_ISSUES = OUT / "week8_v50d_issues.csv"
OUT_REPORT = REPORTS / "week8_v50d_tracking_qa_visual_diagnostics_report.md"
OUT_NOTE = NOTES / "week8_v50d_tracking_qa_visual_diagnostics_notes.md"
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


def get_clip_anchor_frame(clip):
    fps = to_float(clip.get("fps_used"), 25.0) or 25.0
    duration = to_float(clip.get("duration_sec"), 10.0) or 10.0
    anchor_rel_sec = duration / 2.0

    scan = clean_str(clip.get("scan_frame_id"))
    if scan in ["scanframe_0000", "scanframe_0006", "scanframe_0012", "scanframe_0018"]:
        anchor_rel_sec = 0.0

    return int(round(anchor_rel_sec * fps)), anchor_rel_sec


def nearest_available_frame(track_frames, target_frame):
    if len(track_frames) == 0:
        return None
    arr = np.array(sorted(track_frames), dtype=float)
    idx = int(np.argmin(np.abs(arr - target_frame)))
    return int(arr[idx])


def draw_overlay(scan, clip_path, frame_idx, anchor_objects, tracking_rows, out_path):
    try:
        import cv2
    except Exception:
        return False, "cv2_not_available"

    cap = cv2.VideoCapture(str(clip_path))
    if not cap.isOpened():
        cap.release()
        return False, "video_open_failed"

    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if frame_count <= 0:
        cap.release()
        return False, "invalid_frame_count"

    frame_idx = max(0, min(frame_count - 1, int(frame_idx)))
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
    ok, frame = cap.read()
    cap.release()

    if not ok or frame is None:
        return False, "frame_read_failed"

    h, w = frame.shape[:2]

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

    for obj in anchor_objects:
        x1 = to_float(obj.get("bbox_x1"))
        y1 = to_float(obj.get("bbox_y1"))
        x2 = to_float(obj.get("bbox_x2"))
        y2 = to_float(obj.get("bbox_y2"))
        if any(v is None for v in [x1, y1, x2, y2]):
            continue

        colour = clean_str(obj.get("visual_marker_colour")) or "unknown"
        bgr = colour_map.get(colour, (255, 255, 255))

        p1 = (int(round(x1)), int(round(y1)))
        p2 = (int(round(x2)), int(round(y2)))
        cv2.rectangle(frame, p1, p2, bgr, 3)

        label = f"ANCHOR {clean_str(obj.get('behaviour_pig_id'))}/{colour}/{clean_str(obj.get('behaviour_code'))}"
        y_text = max(22, p1[1] - 8)
        cv2.putText(frame, label, (p1[0], y_text), cv2.FONT_HERSHEY_SIMPLEX, 0.48, bgr, 2, cv2.LINE_AA)

    for _, tr in tracking_rows.iterrows():
        x1 = to_float(tr.get("x1"))
        y1 = to_float(tr.get("y1"))
        x2 = to_float(tr.get("x2"))
        y2 = to_float(tr.get("y2"))
        if any(v is None for v in [x1, y1, x2, y2]):
            continue

        p1 = (int(round(x1)), int(round(y1)))
        p2 = (int(round(x2)), int(round(y2)))
        cv2.rectangle(frame, p1, p2, (255, 255, 255), 1)

        label = f"T{clean_str(tr.get('track_id'))} {to_float(tr.get('score'), 0.0):.2f}"
        y_text = min(h - 5, p2[1] + 14)
        cv2.putText(frame, label, (p1[0], y_text), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (255, 255, 255), 1, cv2.LINE_AA)

    cv2.rectangle(frame, (0, 0), (w, 62), (0, 0, 0), -1)
    cv2.putText(frame, f"{scan} tracking QA frame={frame_idx}", (12, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.68, (255, 255, 255), 2, cv2.LINE_AA)
    cv2.putText(frame, "Anchor GT boxes = coloured thick rectangles | Tracking detections = thin white rectangles", (12, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (230, 230, 230), 1, cv2.LINE_AA)

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    ok = cv2.imwrite(str(out_path), frame)
    return bool(ok), "saved" if ok else "write_failed"


issues = []

required = [V45_CLIP_JSON, V45_CLIP_OBJECTS, V49C_DECISION, V50C_TRACKS, V50C_SUMMARY, V50C_DECISION]
for p in required:
    if not p.exists():
        issues.append({
            "item": str(p),
            "issue_type": "hard_missing_required_input",
            "issue_detail": "Required previous-stage artifact missing.",
            "severity": "hard",
        })

if issues:
    issues_df = pd.DataFrame(issues)
    safe_to_csv(issues_df, OUT_ISSUES)
    decision = pd.DataFrame([{
        "v50d_decision": "tracking_qa_blocked_missing_inputs",
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

anchors = pd.read_csv(V45_CLIP_OBJECTS)
tracks = pd.read_csv(V50C_TRACKS)
summary = pd.read_csv(V50C_SUMMARY)
v50c_decision = pd.read_csv(V50C_DECISION)

# BBox QA
bbox = tracks.copy()
for col in ["x1", "y1", "x2", "y2", "video_width", "video_height"]:
    bbox[col] = pd.to_numeric(bbox[col], errors="coerce")

bbox["bbox_width"] = bbox["x2"] - bbox["x1"]
bbox["bbox_height"] = bbox["y2"] - bbox["y1"]
bbox["bbox_valid_basic"] = (
    bbox["x1"].notna()
    & bbox["y1"].notna()
    & bbox["x2"].notna()
    & bbox["y2"].notna()
    & (bbox["bbox_width"] > 0)
    & (bbox["bbox_height"] > 0)
)
bbox["bbox_inside_video"] = (
    bbox["bbox_valid_basic"]
    & (bbox["x1"] >= 0)
    & (bbox["y1"] >= 0)
    & (bbox["x2"] <= bbox["video_width"])
    & (bbox["y2"] <= bbox["video_height"])
)

bbox_qa = pd.DataFrame([{
    "tracking_rows_total": int(len(bbox)),
    "bbox_valid_basic_rows": int(bbox["bbox_valid_basic"].sum()),
    "bbox_inside_video_rows": int(bbox["bbox_inside_video"].sum()),
    "invalid_basic_bbox_rows": int((~bbox["bbox_valid_basic"]).sum()),
    "bbox_outside_video_rows": int((~bbox["bbox_inside_video"]).sum()),
    "min_x1": float(bbox["x1"].min()),
    "min_y1": float(bbox["y1"].min()),
    "max_x2": float(bbox["x2"].max()),
    "max_y2": float(bbox["y2"].max()),
}])
safe_to_csv(bbox_qa, OUT_BBOX_QA)

# Coverage QA
coverage_rows = []
for _, s in summary.iterrows():
    scan = clean_str(s.get("scan_frame_id"))
    g = tracks[tracks["scan_frame_id"].astype(str) == scan]
    expected_processed = to_int(s.get("processed_frames"), 0) or 0
    unique_frames = int(g["frame_index_in_clip"].nunique()) if len(g) else 0
    unique_tracks = int(g["track_id"].nunique()) if len(g) else 0
    rows = int(len(g))
    detections_per_sampled_frame = rows / unique_frames if unique_frames > 0 else 0.0

    if rows == 0:
        status = "no_tracking_rows"
    elif unique_frames >= expected_processed * 0.90:
        status = "full_sampled_frame_coverage"
    elif unique_frames >= expected_processed * 0.50:
        status = "partial_sampled_frame_coverage"
    else:
        status = "low_sampled_frame_coverage"

    coverage_rows.append({
        "scan_frame_id": scan,
        "video_id": clean_str(s.get("video_id")),
        "processed_frames_expected": expected_processed,
        "tracking_unique_frames": unique_frames,
        "tracking_rows": rows,
        "unique_tracks": unique_tracks,
        "detections_per_sampled_frame": detections_per_sampled_frame,
        "coverage_status": status,
    })

coverage_qa = pd.DataFrame(coverage_rows)
safe_to_csv(coverage_qa, OUT_COVERAGE_QA)

# Fragmentation summary
frag_rows = []
for scan, g in tracks.groupby("scan_frame_id"):
    frame_count = int(g["frame_index_in_clip"].nunique())
    track_count = int(g["track_id"].nunique())
    rows = int(len(g))
    mean_track_length = float(g.groupby("track_id")["frame_index_in_clip"].nunique().mean()) if track_count > 0 else 0.0
    median_track_length = float(g.groupby("track_id")["frame_index_in_clip"].nunique().median()) if track_count > 0 else 0.0
    short_tracks = int((g.groupby("track_id")["frame_index_in_clip"].nunique() <= 2).sum()) if track_count > 0 else 0
    fragmentation_ratio = track_count / 6.0

    if track_count <= 10:
        fragmentation_status = "low_to_moderate"
    elif track_count <= 20:
        fragmentation_status = "moderate"
    else:
        fragmentation_status = "high_fragmentation_review_needed"

    frag_rows.append({
        "scan_frame_id": scan,
        "tracking_rows": rows,
        "sampled_frames": frame_count,
        "unique_tracks": track_count,
        "mean_track_length_sampled_frames": mean_track_length,
        "median_track_length_sampled_frames": median_track_length,
        "short_tracks_len_le_2": short_tracks,
        "fragmentation_ratio_vs_6_pigs": fragmentation_ratio,
        "fragmentation_status": fragmentation_status,
    })

fragmentation = pd.DataFrame(frag_rows).sort_values(["fragmentation_status", "unique_tracks"], ascending=[False, False])
safe_to_csv(fragmentation, OUT_TRACK_FRAGMENTATION)

# Anchor/tracking match
anchor_match_rows = []
anchor_by_scan = {str(k): v.copy() for k, v in anchors.groupby("scan_frame_id")}
tracks_by_scan = {str(k): v.copy() for k, v in tracks.groupby("scan_frame_id")}

for scan, clip in clip_map.items():
    anchor_frame, anchor_rel_sec = get_clip_anchor_frame(clip)
    tr = tracks_by_scan.get(scan, pd.DataFrame())
    an = anchor_by_scan.get(scan, pd.DataFrame())

    if len(tr):
        available_frames = sorted([int(x) for x in tr["frame_index_in_clip"].dropna().unique().tolist()])
        nearest_frame = nearest_available_frame(available_frames, anchor_frame)
        tr_anchor = tr[tr["frame_index_in_clip"].astype(int) == int(nearest_frame)] if nearest_frame is not None else pd.DataFrame()
    else:
        nearest_frame = None
        tr_anchor = pd.DataFrame()

    for _, a in an.iterrows():
        abox = [
            to_float(a.get("bbox_x1")),
            to_float(a.get("bbox_y1")),
            to_float(a.get("bbox_x2")),
            to_float(a.get("bbox_y2")),
        ]

        best_iou = 0.0
        best_track_id = ""
        best_score = None

        if all(v is not None for v in abox) and len(tr_anchor):
            for _, t in tr_anchor.iterrows():
                tbox = [
                    to_float(t.get("x1")),
                    to_float(t.get("y1")),
                    to_float(t.get("x2")),
                    to_float(t.get("y2")),
                ]
                if not all(v is not None for v in tbox):
                    continue
                val = iou_xyxy(abox, tbox)
                if val > best_iou:
                    best_iou = val
                    best_track_id = clean_str(t.get("track_id"))
                    best_score = to_float(t.get("score"))

        if best_iou >= 0.50:
            match_status = "strong_match"
        elif best_iou >= 0.30:
            match_status = "moderate_match"
        elif best_iou > 0.0:
            match_status = "weak_match"
        else:
            match_status = "no_overlap"

        anchor_match_rows.append({
            "scan_frame_id": scan,
            "final_box_id": clean_str(a.get("final_box_id")),
            "behaviour_pig_id": clean_str(a.get("behaviour_pig_id")),
            "visual_marker_colour": clean_str(a.get("visual_marker_colour")),
            "behaviour_code": clean_str(a.get("behaviour_code")),
            "anchor_frame_expected": int(anchor_frame),
            "nearest_tracking_frame": nearest_frame,
            "frame_delta": None if nearest_frame is None else int(nearest_frame - anchor_frame),
            "tracking_boxes_at_nearest_frame": int(len(tr_anchor)),
            "best_track_id": best_track_id,
            "best_track_score": best_score,
            "best_iou": float(best_iou),
            "match_status": match_status,
        })

anchor_match = pd.DataFrame(anchor_match_rows)
safe_to_csv(anchor_match, OUT_ANCHOR_MATCH)

# Diagnostic overlays: choose high fragmentation + first clips + low match clips
scan_candidates = []
first_scans = list(clip_map.keys())[:6]
high_frag_scans = fragmentation[fragmentation["fragmentation_status"] == "high_fragmentation_review_needed"]["scan_frame_id"].head(6).tolist()

low_match_scan_counts = (
    anchor_match[anchor_match["match_status"].isin(["weak_match", "no_overlap"])]
    .groupby("scan_frame_id")
    .size()
    .sort_values(ascending=False)
    .head(6)
    .index
    .tolist()
)

for s in first_scans + high_frag_scans + low_match_scan_counts:
    if s not in scan_candidates:
        scan_candidates.append(s)

overlay_rows = []

for scan in scan_candidates:
    clip = clip_map.get(scan)
    if clip is None:
        continue

    anchor_frame, anchor_rel_sec = get_clip_anchor_frame(clip)
    tr = tracks_by_scan.get(scan, pd.DataFrame())

    if len(tr):
        available_frames = sorted([int(x) for x in tr["frame_index_in_clip"].dropna().unique().tolist()])
        nearest_frame = nearest_available_frame(available_frames, anchor_frame)
        tr_frame = tr[tr["frame_index_in_clip"].astype(int) == int(nearest_frame)] if nearest_frame is not None else pd.DataFrame()
    else:
        nearest_frame = anchor_frame
        tr_frame = pd.DataFrame()

    an = anchor_by_scan.get(scan, pd.DataFrame())
    out_path = OVERLAYS / f"{scan}_anchor_tracking_overlay.jpg"

    ok, msg = draw_overlay(
        scan=scan,
        clip_path=clean_str(clip.get("clip_path")),
        frame_idx=nearest_frame if nearest_frame is not None else anchor_frame,
        anchor_objects=an.to_dict("records"),
        tracking_rows=tr_frame,
        out_path=out_path,
    )

    overlay_rows.append({
        "scan_frame_id": scan,
        "overlay_path": str(out_path),
        "saved": bool(ok),
        "message": msg,
        "anchor_frame_expected": int(anchor_frame),
        "nearest_tracking_frame": nearest_frame,
        "tracking_boxes_drawn": int(len(tr_frame)),
        "anchor_boxes_drawn": int(len(an)),
    })

overlay_index = pd.DataFrame(overlay_rows)
safe_to_csv(overlay_index, OUT_OVERLAY_INDEX)

# HTML gallery
html = []
html.append("<!doctype html>")
html.append("<html><head><meta charset='utf-8'>")
html.append("<title>Week 8 v50d Tracking QA Gallery</title>")
html.append("<style>")
html.append("body{font-family:Arial;background:#111;color:#eee;margin:20px}")
html.append(".grid{display:grid;grid-template-columns:repeat(auto-fill,minmax(380px,1fr));gap:16px}")
html.append(".card{background:#1e1e1e;border:1px solid #333;border-radius:10px;padding:12px}")
html.append("img{width:100%;border:1px solid #444;border-radius:6px}")
html.append(".meta{font-size:13px;color:#bbb;line-height:1.4}")
html.append(".top{background:#2b2b2b;padding:12px;border-radius:10px;margin-bottom:18px}")
html.append("</style></head><body>")
html.append("<div class='top'>")
html.append("<h1>Week 8 v50d Tracking QA Gallery</h1>")
html.append("<p>Coloured thick boxes = anchor GT. Thin white boxes = tracking detections at nearest sampled frame.</p>")
html.append("</div>")
html.append("<div class='grid'>")

for _, r in overlay_index.iterrows():
    if not bool(r.get("saved")):
        continue
    rel = Path(r["overlay_path"]).relative_to(OUT)
    html.append("<div class='card'>")
    html.append(f"<h3>{r['scan_frame_id']}</h3>")
    html.append(f"<img src='{rel.as_posix()}'>")
    html.append("<div class='meta'>")
    html.append(f"anchor frame expected: {r['anchor_frame_expected']}<br>")
    html.append(f"nearest tracking frame: {r['nearest_tracking_frame']}<br>")
    html.append(f"anchor boxes: {r['anchor_boxes_drawn']}<br>")
    html.append(f"tracking boxes: {r['tracking_boxes_drawn']}<br>")
    html.append("</div></div>")

html.append("</div></body></html>")
OUT_GALLERY.write_text("\n".join(html))

# Issues and decision
invalid_tracking_bbox_rows = int(bbox_qa.iloc[0]["invalid_basic_bbox_rows"])
outside_tracking_bbox_rows = int(bbox_qa.iloc[0]["bbox_outside_video_rows"])
no_tracking_clip_count = int((coverage_qa["coverage_status"] == "no_tracking_rows").sum())
low_coverage_clip_count = int((coverage_qa["coverage_status"] == "low_sampled_frame_coverage").sum())
high_frag_count = int((fragmentation["fragmentation_status"] == "high_fragmentation_review_needed").sum())

strong_match_count = int((anchor_match["match_status"] == "strong_match").sum())
moderate_match_count = int((anchor_match["match_status"] == "moderate_match").sum())
weak_match_count = int((anchor_match["match_status"] == "weak_match").sum())
no_overlap_count = int((anchor_match["match_status"] == "no_overlap").sum())

if invalid_tracking_bbox_rows > 0:
    issues.append({
        "item": "tracking_bbox_basic_validity",
        "issue_type": "hard_invalid_tracking_bbox_geometry",
        "issue_detail": f"{invalid_tracking_bbox_rows} tracking rows have invalid bbox geometry.",
        "severity": "hard",
    })

if outside_tracking_bbox_rows > 0:
    issues.append({
        "item": "tracking_bbox_bounds",
        "issue_type": "warning_tracking_bbox_outside_video",
        "issue_detail": f"{outside_tracking_bbox_rows} tracking rows are outside video bounds.",
        "severity": "warning",
    })

if no_tracking_clip_count > 0:
    issues.append({
        "item": "tracking_clip_coverage",
        "issue_type": "hard_no_tracking_rows_for_some_clips",
        "issue_detail": f"{no_tracking_clip_count} clips have no tracking rows.",
        "severity": "hard",
    })

if low_coverage_clip_count > 0:
    issues.append({
        "item": "tracking_sampled_frame_coverage",
        "issue_type": "warning_low_sampled_frame_coverage",
        "issue_detail": f"{low_coverage_clip_count} clips have low sampled-frame coverage.",
        "severity": "warning",
    })

if high_frag_count > 0:
    issues.append({
        "item": "tracking_fragmentation",
        "issue_type": "warning_high_track_fragmentation",
        "issue_detail": f"{high_frag_count} clips show high fragmentation. Track IDs must not be treated as final pig identities.",
        "severity": "warning",
    })

if no_overlap_count > 0:
    issues.append({
        "item": "anchor_tracking_iou",
        "issue_type": "warning_anchor_boxes_without_tracking_overlap",
        "issue_detail": f"{no_overlap_count} anchor objects have no overlap with tracking boxes at nearest sampled anchor frame.",
        "severity": "warning",
    })

issues_df = pd.DataFrame(issues, columns=["item", "issue_type", "issue_detail", "severity"])
hard_issues = issues_df[issues_df["severity"] == "hard"] if len(issues_df) else pd.DataFrame()
warnings = issues_df[issues_df["severity"] == "warning"] if len(issues_df) else pd.DataFrame()

ready_for_v51 = len(hard_issues) == 0

decision = pd.DataFrame([{
    "v50d_decision": "tracking_qa_completed_ready_for_interface_integration_with_limitations" if ready_for_v51 else "tracking_qa_blocked",
    "clip_count": int(len(coverage_qa)),
    "tracking_rows_total": int(len(tracks)),
    "sampled_frame_coverage_full_or_partial_clips": int((coverage_qa["coverage_status"].isin(["full_sampled_frame_coverage", "partial_sampled_frame_coverage"])).sum()),
    "no_tracking_clip_count": int(no_tracking_clip_count),
    "invalid_tracking_bbox_rows": int(invalid_tracking_bbox_rows),
    "outside_tracking_bbox_rows": int(outside_tracking_bbox_rows),
    "high_fragmentation_clip_count": int(high_frag_count),
    "anchor_objects_total": int(len(anchor_match)),
    "anchor_strong_match_count": int(strong_match_count),
    "anchor_moderate_match_count": int(moderate_match_count),
    "anchor_weak_match_count": int(weak_match_count),
    "anchor_no_overlap_count": int(no_overlap_count),
    "overlay_gallery_count": int(len(overlay_index)),
    "hard_issue_count": int(len(hard_issues)),
    "warning_count": int(len(warnings)),
    "issue_count": int(len(issues_df)),
    "tracking_ids_final_identity_claim": False,
    "ready_for_v51_tracking_integrated_visualizer": bool(ready_for_v51),
    "generated_at": datetime.now().isoformat(timespec="seconds"),
}])

safe_to_csv(issues_df, OUT_ISSUES)
safe_to_csv(decision, OUT_DECISION)

report = (
    "Week 8 v50d Tracking QA and Visual Diagnostics Report\n\n"
    f"Decision: {decision.iloc[0]['v50d_decision']}\n"
    f"Clip count: {len(coverage_qa)}\n"
    f"Tracking rows total: {len(tracks)}\n"
    f"No-tracking clips: {no_tracking_clip_count}\n"
    f"Invalid tracking bbox rows: {invalid_tracking_bbox_rows}\n"
    f"Outside-video tracking bbox rows: {outside_tracking_bbox_rows}\n"
    f"High-fragmentation clips: {high_frag_count}\n"
    f"Anchor strong matches: {strong_match_count}\n"
    f"Anchor moderate matches: {moderate_match_count}\n"
    f"Anchor weak matches: {weak_match_count}\n"
    f"Anchor no-overlap matches: {no_overlap_count}\n"
    f"Hard issues: {len(hard_issues)}\n"
    f"Warnings: {len(warnings)}\n\n"
    "Interpretation:\n"
    "The v50c tracking output provides full 72-clip coverage and can be used for a tracking-integrated visualizer, but track IDs must not be treated as final pig identities. Fragmentation and anchor matching should guide review and identity refinement.\n\n"
    "Next:\n"
    "Proceed to v51 tracking-integrated visualizer if no hard issues are present.\n"
)
OUT_REPORT.write_text(report)

OUT_NOTE.write_text(
    "# Week 8 v50d Tracking QA and Visual Diagnostics\n\n"
    "## Summary\n\n"
    f"- v50d decision: {decision.iloc[0]['v50d_decision']}\n"
    f"- Clip count: {len(coverage_qa)}\n"
    f"- Tracking rows total: {len(tracks)}\n"
    f"- No-tracking clips: {no_tracking_clip_count}\n"
    f"- Invalid tracking bbox rows: {invalid_tracking_bbox_rows}\n"
    f"- Outside-video tracking bbox rows: {outside_tracking_bbox_rows}\n"
    f"- High-fragmentation clips: {high_frag_count}\n"
    f"- Anchor strong matches: {strong_match_count}\n"
    f"- Anchor moderate matches: {moderate_match_count}\n"
    f"- Anchor weak matches: {weak_match_count}\n"
    f"- Anchor no-overlap matches: {no_overlap_count}\n"
    f"- Hard issue count: {len(hard_issues)}\n"
    f"- Warning count: {len(warnings)}\n"
    f"- Ready for v51 tracking-integrated visualizer: {ready_for_v51}\n\n"
    "## Important limitation\n\n"
    "Tracking boxes can improve full-clip visualization, but track IDs are not final pig identities. "
    "Identity assignment still needs colour/anchor-based review.\n\n"
    f"Gallery: {OUT_GALLERY}\n"
)

progress_row = pd.DataFrame([{
    "date": datetime.now().date().isoformat(),
    "stage": "v50d",
    "task_name": "Tracking QA and visual diagnostics",
    "status": "PASS" if ready_for_v51 else "BLOCKED",
    "input_summary": str(V50C_TRACKS),
    "output_summary": str(OUT),
    "hard_issues": int(len(hard_issues)),
    "warnings": int(len(warnings)),
    "next_action": "v51 tracking-integrated visualizer" if ready_for_v51 else "Resolve tracking QA hard issues.",
}])

if OUT_PROGRESS.exists():
    old = pd.read_csv(OUT_PROGRESS)
    progress = pd.concat([old, progress_row], ignore_index=True)
else:
    progress = progress_row

safe_to_csv(progress, OUT_PROGRESS)

print("Saved:")
print(OUT_COVERAGE_QA)
print(OUT_BBOX_QA)
print(OUT_TRACK_FRAGMENTATION)
print(OUT_ANCHOR_MATCH)
print(OUT_OVERLAY_INDEX)
print(OUT_GALLERY)
print(OUT_DECISION)
print(OUT_ISSUES)
print(OUT_REPORT)
print(OUT_NOTE)

print()
print("=== v50d decision ===")
print(decision.to_string(index=False))

print()
print("=== v50d issues ===")
if len(issues_df):
    print(issues_df.to_string(index=False))
else:
    print("No issues found.")

print()
print("=== coverage status counts ===")
print(coverage_qa["coverage_status"].value_counts(dropna=False).to_string())

print()
print("=== fragmentation status counts ===")
print(fragmentation["fragmentation_status"].value_counts(dropna=False).to_string())

print()
print("=== anchor match status counts ===")
print(anchor_match["match_status"].value_counts(dropna=False).to_string())
