from pathlib import Path
from datetime import datetime
import json
import csv
import pandas as pd
import numpy as np


ROOT = Path.home() / "PigBench"
W8 = ROOT / "Week8_Ground_Truth_Validation_Behaviour_Dataset_Preparation"

V45_CLIP_JSON = W8 / "outputs" / "propagated_ground_truth" / "v45_label_propagation" / "week8_v45_clip_level_ground_truth.json"
V45_CLIP_OBJECTS = W8 / "outputs" / "propagated_ground_truth" / "v45_label_propagation" / "week8_v45_clip_object_propagated_annotations.csv"
V50C_TRACKS = W8 / "outputs" / "v50c_full_72_clip_cpu_tracking" / "week8_v50c_full_tracking_rows.csv"
V50D_DECISION = W8 / "outputs" / "v50d_tracking_qa_visual_diagnostics" / "week8_v50d_decision_summary.csv"

OUT = W8 / "outputs" / "v50e_roi_filtered_tracking_overlay"
GALLERY_DIR = OUT / "roi_filtered_overlay_gallery"
REPORTS = W8 / "reports"
NOTES = W8 / "notes"
PROGRESS = W8 / "progress"

for p in [OUT, GALLERY_DIR, REPORTS, NOTES, PROGRESS]:
    p.mkdir(parents=True, exist_ok=True)

OUT_ROI = OUT / "week8_v50e_video_level_roi_from_anchor_boxes.csv"
OUT_FILTERED_TRACKS = OUT / "week8_v50e_roi_filtered_tracking_rows.csv"
OUT_TRACK_QA = OUT / "week8_v50e_roi_filter_tracking_qa.csv"
OUT_GALLERY_INDEX = OUT / "week8_v50e_gallery_index.csv"
OUT_GALLERY_HTML = OUT / "index.html"
OUT_DECISION = OUT / "week8_v50e_decision_summary.csv"
OUT_ISSUES = OUT / "week8_v50e_issues.csv"
OUT_REPORT = REPORTS / "week8_v50e_roi_filtered_tracking_overlay_report.md"
OUT_NOTE = NOTES / "week8_v50e_roi_filtered_tracking_overlay_notes.md"
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


def draw_overlay(scan, clip, anchor_df, raw_tracks, filtered_tracks, roi_row, out_path):
    try:
        import cv2
    except Exception:
        return False, "cv2_not_available"

    clip_path = Path(clean_str(clip.get("clip_path")))
    cap = cv2.VideoCapture(str(clip_path))
    if not cap.isOpened():
        cap.release()
        return False, "video_open_failed"

    anchor_frame, anchor_rel_sec = get_anchor_frame(clip)

    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if frame_count <= 0:
        cap.release()
        return False, "invalid_frame_count"

    anchor_frame = max(0, min(frame_count - 1, anchor_frame))
    cap.set(cv2.CAP_PROP_POS_FRAMES, anchor_frame)
    ok, frame = cap.read()
    cap.release()

    if not ok or frame is None:
        return False, "frame_read_failed"

    h, w = frame.shape[:2]

    raw_at_frame = raw_tracks[raw_tracks["frame_index_in_clip"].astype(int) == int(anchor_frame)].copy()
    filtered_at_frame = filtered_tracks[filtered_tracks["frame_index_in_clip"].astype(int) == int(anchor_frame)].copy()

    # If exact anchor frame is not sampled due stride, use nearest sampled frame.
    if len(raw_at_frame) == 0 and len(raw_tracks):
        frames = np.array(sorted(raw_tracks["frame_index_in_clip"].dropna().astype(int).unique().tolist()))
        nearest = int(frames[np.argmin(np.abs(frames - anchor_frame))])
        raw_at_frame = raw_tracks[raw_tracks["frame_index_in_clip"].astype(int) == nearest].copy()
        filtered_at_frame = filtered_tracks[filtered_tracks["frame_index_in_clip"].astype(int) == nearest].copy()
        draw_frame = nearest
    else:
        draw_frame = anchor_frame

    # Draw ROI rectangle first.
    rx1 = int(round(roi_row["roi_x1"]))
    ry1 = int(round(roi_row["roi_y1"]))
    rx2 = int(round(roi_row["roi_x2"]))
    ry2 = int(round(roi_row["roi_y2"]))
    cv2.rectangle(frame, (rx1, ry1), (rx2, ry2), (0, 255, 255), 2)
    cv2.putText(frame, "ROI", (rx1 + 5, max(20, ry1 - 8)), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 255), 2, cv2.LINE_AA)

    # Draw anchor GT boxes thin coloured.
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

    for _, a in anchor_df.iterrows():
        x1 = to_int(a.get("bbox_x1"))
        y1 = to_int(a.get("bbox_y1"))
        x2 = to_int(a.get("bbox_x2"))
        y2 = to_int(a.get("bbox_y2"))
        if None in [x1, y1, x2, y2]:
            continue

        colour = clean_str(a.get("visual_marker_colour")) or "unknown"
        bgr = colour_map.get(colour, (255, 255, 255))

        cv2.rectangle(frame, (x1, y1), (x2, y2), bgr, 1)
        label = f"GT {clean_str(a.get('behaviour_pig_id'))}/{colour}/{clean_str(a.get('behaviour_code'))}"
        cv2.putText(frame, label, (x1, max(18, y1 - 6)), cv2.FONT_HERSHEY_SIMPLEX, 0.42, bgr, 1, cv2.LINE_AA)

    # Draw raw outside-ROI tracks in red dashed-like simple boxes.
    raw_outside = raw_at_frame[raw_at_frame["roi_keep"] == False] if len(raw_at_frame) else pd.DataFrame()
    for _, t in raw_outside.iterrows():
        x1 = to_int(t.get("x1"))
        y1 = to_int(t.get("y1"))
        x2 = to_int(t.get("x2"))
        y2 = to_int(t.get("y2"))
        if None in [x1, y1, x2, y2]:
            continue

        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 1)
        label = f"OUT T{clean_str(t.get('track_id'))}"
        cv2.putText(frame, label, (x1, min(h - 5, y2 + 14)), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (0, 0, 255), 1, cv2.LINE_AA)

    # Draw kept tracking boxes last, thick white with black shadow.
    for _, t in filtered_at_frame.iterrows():
        x1 = to_int(t.get("x1"))
        y1 = to_int(t.get("y1"))
        x2 = to_int(t.get("x2"))
        y2 = to_int(t.get("y2"))
        if None in [x1, y1, x2, y2]:
            continue

        label = f"T{clean_str(t.get('track_id'))} {to_float(t.get('score'), 0.0):.2f}"

        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 0), 4)
        cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 255, 255), 2)

        text_y = min(h - 5, y2 + 16)
        cv2.putText(frame, label, (x1, text_y), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 0), 3, cv2.LINE_AA)
        cv2.putText(frame, label, (x1, text_y), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1, cv2.LINE_AA)

    # Header.
    cv2.rectangle(frame, (0, 0), (w, 70), (0, 0, 0), -1)
    cv2.putText(frame, f"{scan} ROI-filtered tracking QA", (12, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.68, (255, 255, 255), 2, cv2.LINE_AA)
    cv2.putText(frame, f"draw_frame={draw_frame} | GT thin colour | kept tracks thick white | outside ROI red | ROI yellow", (12, 52), cv2.FONT_HERSHEY_SIMPLEX, 0.47, (230, 230, 230), 1, cv2.LINE_AA)

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    saved = cv2.imwrite(str(out_path), frame)
    return bool(saved), "saved" if saved else "write_failed"


issues = []

required = [V45_CLIP_JSON, V45_CLIP_OBJECTS, V50C_TRACKS, V50D_DECISION]
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
        "v50e_decision": "roi_filtered_tracking_blocked_missing_inputs",
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

for col in ["bbox_x1", "bbox_y1", "bbox_x2", "bbox_y2"]:
    anchors[col] = pd.to_numeric(anchors[col], errors="coerce")

for col in ["x1", "y1", "x2", "y2", "video_width", "video_height", "frame_index_in_clip"]:
    tracks[col] = pd.to_numeric(tracks[col], errors="coerce")

# Build video-level ROI from anchor boxes, because each video/camera session can have slightly different valid pen area.
roi_rows = []
margin = 35

for video_id, g in anchors.groupby("video_id"):
    video_id = clean_str(video_id)

    # Find video dimensions from tracks for this video.
    tg = tracks[tracks["video_id"].astype(str).map(clean_str) == video_id]
    video_w = int(tg["video_width"].dropna().iloc[0]) if len(tg) and tg["video_width"].notna().any() else 704
    video_h = int(tg["video_height"].dropna().iloc[0]) if len(tg) and tg["video_height"].notna().any() else 576

    x1 = max(0, float(g["bbox_x1"].min()) - margin)
    y1 = max(0, float(g["bbox_y1"].min()) - margin)
    x2 = min(video_w, float(g["bbox_x2"].max()) + margin)
    y2 = min(video_h, float(g["bbox_y2"].max()) + margin)

    roi_rows.append({
        "video_id": video_id,
        "roi_x1": x1,
        "roi_y1": y1,
        "roi_x2": x2,
        "roi_y2": y2,
        "roi_margin_px": margin,
        "video_width": video_w,
        "video_height": video_h,
        "anchor_box_count": int(len(g)),
        "roi_source": "video_level_union_of_anchor_gt_boxes_expanded",
    })

roi = pd.DataFrame(roi_rows)
safe_to_csv(roi, OUT_ROI)

tracks_roi = tracks.merge(roi, on="video_id", how="left", suffixes=("", "_roi"))

missing_roi_rows = int(tracks_roi["roi_x1"].isna().sum())

if missing_roi_rows > 0:
    issues.append({
        "item": "roi_assignment",
        "issue_type": "warning_tracking_rows_missing_video_level_roi",
        "issue_detail": f"{missing_roi_rows} tracking rows could not be assigned to a video-level ROI.",
        "severity": "warning",
    })

# If missing ROI, keep those rows for safety but flag them.
tracks_roi["track_center_x"] = (tracks_roi["x1"] + tracks_roi["x2"]) / 2.0
tracks_roi["track_center_y"] = (tracks_roi["y1"] + tracks_roi["y2"]) / 2.0

tracks_roi["roi_center_inside"] = tracks_roi.apply(
    lambda r: False if pd.isna(r.get("roi_x1")) else center_inside_roi(r),
    axis=1,
)

tracks_roi["roi_bbox_overlap_ratio"] = tracks_roi.apply(
    lambda r: 0.0 if pd.isna(r.get("roi_x1")) else bbox_overlap_ratio_with_roi(r),
    axis=1,
)

# Keep if center is inside ROI or at least 50% of bbox area overlaps ROI.
tracks_roi["roi_keep"] = tracks_roi["roi_center_inside"] | (tracks_roi["roi_bbox_overlap_ratio"] >= 0.50)

# Missing ROI rows are kept but flagged, to avoid accidental deletion.
tracks_roi.loc[tracks_roi["roi_x1"].isna(), "roi_keep"] = True
tracks_roi["roi_filter_version"] = "v50e_video_level_anchor_union_margin35"
tracks_roi["tracking_identity_claim"] = False

safe_to_csv(tracks_roi[tracks_roi["roi_keep"] == True], OUT_FILTERED_TRACKS)

# QA by clip.
qa_rows = []
for scan, clip in clip_map.items():
    g = tracks_roi[tracks_roi["scan_frame_id"].astype(str).map(clean_str) == scan]
    if len(g) == 0:
        qa_rows.append({
            "scan_frame_id": scan,
            "video_id": clean_str(clip.get("video_id")),
            "raw_tracking_rows": 0,
            "kept_tracking_rows": 0,
            "removed_outside_roi_rows": 0,
            "removed_ratio": 0.0,
            "raw_unique_tracks": 0,
            "kept_unique_tracks": 0,
            "qa_status": "no_tracking_rows",
        })
        continue

    kept = g[g["roi_keep"] == True]
    removed = g[g["roi_keep"] == False]

    removed_ratio = len(removed) / len(g) if len(g) else 0.0

    if removed_ratio == 0:
        status = "no_roi_removal"
    elif removed_ratio <= 0.15:
        status = "minor_roi_removal"
    elif removed_ratio <= 0.40:
        status = "moderate_roi_removal_review"
    else:
        status = "high_roi_removal_review"

    qa_rows.append({
        "scan_frame_id": scan,
        "video_id": clean_str(clip.get("video_id")),
        "raw_tracking_rows": int(len(g)),
        "kept_tracking_rows": int(len(kept)),
        "removed_outside_roi_rows": int(len(removed)),
        "removed_ratio": float(removed_ratio),
        "raw_unique_tracks": int(g["track_id"].nunique()),
        "kept_unique_tracks": int(kept["track_id"].nunique()) if len(kept) else 0,
        "qa_status": status,
    })

qa = pd.DataFrame(qa_rows)
safe_to_csv(qa, OUT_TRACK_QA)

# Gallery: first clips + highest removal review clips.
selected_scans = []
for s in list(clip_map.keys())[:6]:
    selected_scans.append(s)

for s in qa.sort_values("removed_ratio", ascending=False)["scan_frame_id"].head(10).tolist():
    if s not in selected_scans:
        selected_scans.append(s)

gallery_rows = []
anchor_by_scan = {str(k): v.copy() for k, v in anchors.groupby("scan_frame_id")}

for scan in selected_scans:
    clip = clip_map.get(scan)
    if clip is None:
        continue

    video_id = clean_str(clip.get("video_id"))
    roi_match = roi[roi["video_id"].astype(str).map(clean_str) == video_id]
    if len(roi_match) == 0:
        continue

    roi_row = roi_match.iloc[0].to_dict()
    anchor_df = anchor_by_scan.get(scan, pd.DataFrame())
    raw_tracks = tracks_roi[tracks_roi["scan_frame_id"].astype(str).map(clean_str) == scan].copy()
    filtered_tracks = raw_tracks[raw_tracks["roi_keep"] == True].copy()

    out_path = GALLERY_DIR / f"{scan}_roi_filtered_overlay.jpg"

    ok, msg = draw_overlay(
        scan=scan,
        clip=clip,
        anchor_df=anchor_df,
        raw_tracks=raw_tracks,
        filtered_tracks=filtered_tracks,
        roi_row=roi_row,
        out_path=out_path,
    )

    qrow = qa[qa["scan_frame_id"] == scan].iloc[0].to_dict()

    gallery_rows.append({
        "scan_frame_id": scan,
        "video_id": video_id,
        "overlay_path": str(out_path),
        "saved": bool(ok),
        "message": msg,
        "raw_tracking_rows": qrow["raw_tracking_rows"],
        "kept_tracking_rows": qrow["kept_tracking_rows"],
        "removed_outside_roi_rows": qrow["removed_outside_roi_rows"],
        "removed_ratio": qrow["removed_ratio"],
        "qa_status": qrow["qa_status"],
    })

gallery = pd.DataFrame(gallery_rows)
safe_to_csv(gallery, OUT_GALLERY_INDEX)

# HTML gallery.
html = []
html.append("<!doctype html>")
html.append("<html><head><meta charset='utf-8'>")
html.append("<title>Week 8 v50e ROI-filtered Tracking Gallery</title>")
html.append("<style>")
html.append("body{font-family:Arial;background:#111;color:#eee;margin:20px}")
html.append(".grid{display:grid;grid-template-columns:repeat(auto-fill,minmax(390px,1fr));gap:16px}")
html.append(".card{background:#1e1e1e;border:1px solid #333;border-radius:10px;padding:12px}")
html.append("img{width:100%;border:1px solid #444;border-radius:6px}")
html.append(".meta{font-size:13px;color:#bbb;line-height:1.4}")
html.append(".top{background:#2b2b2b;padding:12px;border-radius:10px;margin-bottom:18px}")
html.append("</style></head><body>")
html.append("<div class='top'>")
html.append("<h1>Week 8 v50e ROI-filtered Tracking Gallery</h1>")
html.append("<p>GT anchor boxes = thin coloured. Kept tracking boxes = thick white. Outside-ROI detections = red. ROI = yellow.</p>")
html.append("<p>Track IDs are not final pig identities.</p>")
html.append("</div>")
html.append("<div class='grid'>")

for _, r in gallery.iterrows():
    if not bool(r.get("saved")):
        continue
    rel = Path(r["overlay_path"]).relative_to(OUT)
    html.append("<div class='card'>")
    html.append(f"<h3>{r['scan_frame_id']}</h3>")
    html.append(f"<img src='{rel.as_posix()}'>")
    html.append("<div class='meta'>")
    html.append(f"video: {r['video_id']}<br>")
    html.append(f"raw rows: {r['raw_tracking_rows']}<br>")
    html.append(f"kept rows: {r['kept_tracking_rows']}<br>")
    html.append(f"removed outside ROI: {r['removed_outside_roi_rows']}<br>")
    html.append(f"removed ratio: {float(r['removed_ratio']):.3f}<br>")
    html.append(f"QA status: {r['qa_status']}<br>")
    html.append("</div></div>")

html.append("</div></body></html>")
OUT_GALLERY_HTML.write_text("\n".join(html))

# Issues and decision.
total_raw_rows = int(len(tracks_roi))
total_kept_rows = int((tracks_roi["roi_keep"] == True).sum())
total_removed_rows = int((tracks_roi["roi_keep"] == False).sum())
removed_ratio_total = total_removed_rows / total_raw_rows if total_raw_rows else 0.0

clips_high_removal = int((qa["qa_status"] == "high_roi_removal_review").sum())
clips_moderate_removal = int((qa["qa_status"] == "moderate_roi_removal_review").sum())

if total_kept_rows == 0:
    issues.append({
        "item": "roi_filtered_tracking",
        "issue_type": "hard_no_rows_after_roi_filter",
        "issue_detail": "ROI filter removed all tracking rows.",
        "severity": "hard",
    })

if clips_high_removal > 0:
    issues.append({
        "item": "roi_filter_clip_review",
        "issue_type": "warning_high_roi_removal_clips",
        "issue_detail": f"{clips_high_removal} clips have high ROI removal ratio and need visual review.",
        "severity": "warning",
    })

if clips_moderate_removal > 0:
    issues.append({
        "item": "roi_filter_clip_review",
        "issue_type": "warning_moderate_roi_removal_clips",
        "issue_detail": f"{clips_moderate_removal} clips have moderate ROI removal ratio and need visual review.",
        "severity": "warning",
    })

issues_df = pd.DataFrame(issues, columns=["item", "issue_type", "issue_detail", "severity"])
hard_issues = issues_df[issues_df["severity"] == "hard"] if len(issues_df) else pd.DataFrame()
warnings = issues_df[issues_df["severity"] == "warning"] if len(issues_df) else pd.DataFrame()

ready_for_v51 = len(hard_issues) == 0

decision = pd.DataFrame([{
    "v50e_decision": "roi_filtered_tracking_overlay_completed" if ready_for_v51 else "roi_filtered_tracking_overlay_blocked",
    "raw_tracking_rows": int(total_raw_rows),
    "kept_tracking_rows": int(total_kept_rows),
    "removed_outside_roi_rows": int(total_removed_rows),
    "removed_ratio_total": float(removed_ratio_total),
    "clip_count": int(len(qa)),
    "clips_high_roi_removal_review": int(clips_high_removal),
    "clips_moderate_roi_removal_review": int(clips_moderate_removal),
    "gallery_overlay_count": int(len(gallery)),
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
    "Week 8 v50e ROI-filtered Tracking Overlay Report\n\n"
    f"Decision: {decision.iloc[0]['v50e_decision']}\n"
    f"Raw tracking rows: {total_raw_rows}\n"
    f"Kept tracking rows: {total_kept_rows}\n"
    f"Removed outside ROI rows: {total_removed_rows}\n"
    f"Removed ratio total: {removed_ratio_total:.4f}\n"
    f"High ROI-removal clips: {clips_high_removal}\n"
    f"Moderate ROI-removal clips: {clips_moderate_removal}\n"
    f"Hard issues: {len(hard_issues)}\n"
    f"Warnings: {len(warnings)}\n\n"
    "Interpretation:\n"
    "ROI filtering reduces detections outside the annotated pen area. The filter is derived from video-level anchor GT boxes expanded by a conservative margin. "
    "This should be treated as a QA filter, not a final biological identity assignment.\n\n"
    "Visualization update:\n"
    "The new overlay draws anchor GT boxes thin, kept tracking boxes thick white, outside-ROI detections red, and the ROI boundary yellow.\n\n"
    "Next:\n"
    "Proceed to v51 tracking-integrated visualizer if no hard issues are present.\n"
)

OUT_REPORT.write_text(report)

OUT_NOTE.write_text(
    "# Week 8 v50e ROI-filtered Tracking Overlay\n\n"
    "## Summary\n\n"
    f"- v50e decision: {decision.iloc[0]['v50e_decision']}\n"
    f"- Raw tracking rows: {total_raw_rows}\n"
    f"- Kept tracking rows: {total_kept_rows}\n"
    f"- Removed outside ROI rows: {total_removed_rows}\n"
    f"- Removed ratio total: {removed_ratio_total:.4f}\n"
    f"- High ROI-removal clips: {clips_high_removal}\n"
    f"- Moderate ROI-removal clips: {clips_moderate_removal}\n"
    f"- Hard issue count: {len(hard_issues)}\n"
    f"- Warning count: {len(warnings)}\n"
    f"- Ready for v51 tracking-integrated visualizer: {ready_for_v51}\n\n"
    "## Important limitation\n\n"
    "ROI-filtered tracking boxes improve visualization, but track IDs are still not final pig identities.\n\n"
    f"Gallery: {OUT_GALLERY_HTML}\n"
)

progress_row = pd.DataFrame([{
    "date": datetime.now().date().isoformat(),
    "stage": "v50e",
    "task_name": "ROI-filtered tracking overlay",
    "status": "PASS" if ready_for_v51 else "BLOCKED",
    "input_summary": str(V50C_TRACKS),
    "output_summary": str(OUT),
    "hard_issues": int(len(hard_issues)),
    "warnings": int(len(warnings)),
    "next_action": "v51 tracking-integrated visualizer" if ready_for_v51 else "Review ROI filter hard issues.",
}])

if OUT_PROGRESS.exists():
    old = pd.read_csv(OUT_PROGRESS)
    progress = pd.concat([old, progress_row], ignore_index=True)
else:
    progress = progress_row

safe_to_csv(progress, OUT_PROGRESS)

print("Saved:")
print(OUT_ROI)
print(OUT_FILTERED_TRACKS)
print(OUT_TRACK_QA)
print(OUT_GALLERY_INDEX)
print(OUT_GALLERY_HTML)
print(OUT_DECISION)
print(OUT_ISSUES)
print(OUT_REPORT)
print(OUT_NOTE)

print()
print("=== v50e decision ===")
print(decision.to_string(index=False))

print()
print("=== v50e issues ===")
if len(issues_df):
    print(issues_df.to_string(index=False))
else:
    print("No issues found.")

print()
print("=== ROI filter QA status counts ===")
print(qa["qa_status"].value_counts(dropna=False).to_string())

print()
print("=== highest removal clips ===")
print(qa.sort_values("removed_ratio", ascending=False).head(15).to_string(index=False))
