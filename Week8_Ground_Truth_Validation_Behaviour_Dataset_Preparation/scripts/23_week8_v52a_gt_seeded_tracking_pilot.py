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
V51D_CLIP_RECALL = W8 / "outputs" / "v51d_dense_tracking_qa_comparison" / "week8_v51d_clip_recall_by_source.csv"

OUT = W8 / "outputs" / "v52a_gt_seeded_tracking_pilot"
GALLERY = OUT / "gt_seeded_tracking_gallery"
REPORTS = W8 / "reports"
NOTES = W8 / "notes"
PROGRESS = W8 / "progress"

for p in [OUT, GALLERY, REPORTS, NOTES, PROGRESS]:
    p.mkdir(parents=True, exist_ok=True)

OUT_TRACKS = OUT / "week8_v52a_gt_seeded_tracking_rows.csv"
OUT_CLIP_SUMMARY = OUT / "week8_v52a_gt_seeded_clip_summary.csv"
OUT_OBJECT_SUMMARY = OUT / "week8_v52a_gt_seeded_object_summary.csv"
OUT_GALLERY_INDEX = OUT / "week8_v52a_gallery_index.csv"
OUT_GALLERY_HTML = OUT / "index.html"
OUT_DECISION = OUT / "week8_v52a_decision_summary.csv"
OUT_ISSUES = OUT / "week8_v52a_issues.csv"
OUT_REPORT = REPORTS / "week8_v52a_gt_seeded_tracking_pilot_report.md"
OUT_NOTE = NOTES / "week8_v52a_gt_seeded_tracking_pilot_notes.md"
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


def get_anchor_frame(clip):
    fps = to_float(clip.get("fps_used"), 25.0) or 25.0
    duration = to_float(clip.get("duration_sec"), 10.0) or 10.0
    scan = clean_str(clip.get("scan_frame_id"))

    if scan in ["scanframe_0000", "scanframe_0006", "scanframe_0012", "scanframe_0018"]:
        anchor_rel_sec = 0.0
    else:
        anchor_rel_sec = duration / 2.0

    return int(round(anchor_rel_sec * fps)), anchor_rel_sec


def create_tracker():
    import cv2

    # Prefer CSRT if available, then KCF, then MIL.
    constructors = [
        ("CSRT", lambda: cv2.legacy.TrackerCSRT_create()),
        ("KCF", lambda: cv2.legacy.TrackerKCF_create()),
        ("CSRT", lambda: cv2.TrackerCSRT_create()),
        ("KCF", lambda: cv2.TrackerKCF_create()),
        ("MIL", lambda: cv2.TrackerMIL_create()),
    ]

    last_error = None
    for name, ctor in constructors:
        try:
            tracker = ctor()
            return name, tracker
        except Exception as e:
            last_error = str(e)

    raise RuntimeError("No supported OpenCV tracker found. Last error: " + str(last_error))


def bbox_xyxy_to_xywh(x1, y1, x2, y2, width, height):
    x1 = max(0.0, min(float(width - 1), float(x1)))
    y1 = max(0.0, min(float(height - 1), float(y1)))
    x2 = max(0.0, min(float(width - 1), float(x2)))
    y2 = max(0.0, min(float(height - 1), float(y2)))

    if x2 <= x1:
        x2 = min(float(width - 1), x1 + 2.0)
    if y2 <= y1:
        y2 = min(float(height - 1), y1 + 2.0)

    return (x1, y1, x2 - x1, y2 - y1)


def bbox_xywh_to_xyxy(b):
    x, y, w, h = b
    return float(x), float(y), float(x + w), float(y + h)


def read_frames(video_path):
    import cv2

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        cap.release()
        return None, {}

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

    meta = {
        "fps": fps,
        "frame_count": frame_count,
        "width": width,
        "height": height,
        "frames_read": len(frames),
    }

    return frames, meta


def track_one_object(frames, meta, anchor_frame, anchor_obj):
    width = meta["width"]
    height = meta["height"]
    fps = meta["fps"] if meta["fps"] else 25.0
    n = len(frames)

    x1 = to_float(anchor_obj.get("bbox_x1"))
    y1 = to_float(anchor_obj.get("bbox_y1"))
    x2 = to_float(anchor_obj.get("bbox_x2"))
    y2 = to_float(anchor_obj.get("bbox_y2"))

    init_xywh = bbox_xyxy_to_xywh(x1, y1, x2, y2, width, height)
    init_area = max(1.0, init_xywh[2] * init_xywh[3])

    rows = []

    # Anchor row.
    rows.append({
        "frame_index_in_clip": anchor_frame,
        "timestamp_sec_in_clip": anchor_frame / fps,
        "x1": x1,
        "y1": y1,
        "x2": x2,
        "y2": y2,
        "tracking_ok": True,
        "tracking_direction": "anchor",
        "area_ratio_vs_anchor": 1.0,
        "status": "anchor_gt",
    })

    # Forward tracking.
    try:
        tracker_name, tracker = create_tracker()
        tracker.init(frames[anchor_frame], tuple(init_xywh))

        for fi in range(anchor_frame + 1, n):
            ok, box = tracker.update(frames[fi])
            bx1, by1, bx2, by2 = bbox_xywh_to_xyxy(box)
            area = max(1.0, (bx2 - bx1) * (by2 - by1))
            area_ratio = area / init_area

            if not ok:
                status = "tracker_failed"
            elif bx2 <= 0 or by2 <= 0 or bx1 >= width or by1 >= height:
                status = "bbox_outside_frame"
            elif area_ratio < 0.20 or area_ratio > 5.00:
                status = "area_drift_warning"
            else:
                status = "tracked"

            rows.append({
                "frame_index_in_clip": fi,
                "timestamp_sec_in_clip": fi / fps,
                "x1": bx1,
                "y1": by1,
                "x2": bx2,
                "y2": by2,
                "tracking_ok": bool(ok),
                "tracking_direction": "forward",
                "area_ratio_vs_anchor": area_ratio,
                "status": status,
            })
    except Exception as e:
        rows.append({
            "frame_index_in_clip": anchor_frame,
            "timestamp_sec_in_clip": anchor_frame / fps,
            "x1": x1,
            "y1": y1,
            "x2": x2,
            "y2": y2,
            "tracking_ok": False,
            "tracking_direction": "forward",
            "area_ratio_vs_anchor": 1.0,
            "status": "forward_exception_" + str(e)[:80],
        })

    # Backward tracking by using reversed frames up to anchor.
    try:
        tracker_name_back, tracker_back = create_tracker()
        tracker_back.init(frames[anchor_frame], tuple(init_xywh))

        for fi in range(anchor_frame - 1, -1, -1):
            ok, box = tracker_back.update(frames[fi])
            bx1, by1, bx2, by2 = bbox_xywh_to_xyxy(box)
            area = max(1.0, (bx2 - bx1) * (by2 - by1))
            area_ratio = area / init_area

            if not ok:
                status = "tracker_failed"
            elif bx2 <= 0 or by2 <= 0 or bx1 >= width or by1 >= height:
                status = "bbox_outside_frame"
            elif area_ratio < 0.20 or area_ratio > 5.00:
                status = "area_drift_warning"
            else:
                status = "tracked"

            rows.append({
                "frame_index_in_clip": fi,
                "timestamp_sec_in_clip": fi / fps,
                "x1": bx1,
                "y1": by1,
                "x2": bx2,
                "y2": by2,
                "tracking_ok": bool(ok),
                "tracking_direction": "backward",
                "area_ratio_vs_anchor": area_ratio,
                "status": status,
            })
    except Exception as e:
        rows.append({
            "frame_index_in_clip": anchor_frame,
            "timestamp_sec_in_clip": anchor_frame / fps,
            "x1": x1,
            "y1": y1,
            "x2": x2,
            "y2": y2,
            "tracking_ok": False,
            "tracking_direction": "backward",
            "area_ratio_vs_anchor": 1.0,
            "status": "backward_exception_" + str(e)[:80],
        })

    for r in rows:
        r["tracker_name"] = tracker_name if "tracker_name" in locals() else "unknown"

    return rows


def draw_gallery_image(scan, clip, frames, meta, anchors_df, tracks_df, out_path):
    import cv2

    sample_frames = [0, meta["frames_read"] // 2, max(0, meta["frames_read"] - 1)]
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

        g = tracks_df[tracks_df["frame_index_in_clip"] == fi].copy()

        for _, r in g.iterrows():
            x1 = to_int(r.get("x1"))
            y1 = to_int(r.get("y1"))
            x2 = to_int(r.get("x2"))
            y2 = to_int(r.get("y2"))
            if None in [x1, y1, x2, y2]:
                continue

            colour = clean_str(r.get("visual_marker_colour")) or "unknown"
            bgr = colour_map.get(colour, (255, 255, 255))

            status = clean_str(r.get("status"))
            thickness = 3 if status in ["anchor_gt", "tracked"] else 1

            cv2.rectangle(img, (x1, y1), (x2, y2), bgr, thickness)
            label = f"{clean_str(r.get('behaviour_pig_id'))}/{colour}/{clean_str(r.get('behaviour_code'))}"
            cv2.putText(img, label, (x1, max(18, y1 - 6)), cv2.FONT_HERSHEY_SIMPLEX, 0.42, bgr, 1)

        cv2.rectangle(img, (0, 0), (w, 55), (0, 0, 0), -1)
        cv2.putText(img, f"{scan} GT-seeded tracking frame={fi}", (10, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.62, (255, 255, 255), 2)
        cv2.putText(img, "one trajectory initialized from each anchor GT box", (10, 47), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (230, 230, 230), 1)

        panels.append(img)

    combined = np.hstack(panels)
    ok = cv2.imwrite(str(out_path), combined)
    return bool(ok), "saved" if ok else "write_failed"


issues = []

for p in [V45_CLIP_JSON, V45_ANCHORS]:
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
        "v52a_decision": "gt_seeded_tracking_pilot_blocked_missing_inputs",
        "hard_issue_count": int((issues_df["severity"] == "hard").sum()),
        "ready_for_v52b_full_gt_seeded_tracking": False,
        "generated_at": datetime.now().isoformat(timespec="seconds"),
    }])
    safe_to_csv(decision, OUT_DECISION)
    print(decision.to_string(index=False))
    raise SystemExit(1)


clips = json.loads(V45_CLIP_JSON.read_text()).get("clips", [])
clip_map = {clean_str(c.get("scan_frame_id")): c for c in clips}

anchors = pd.read_csv(V45_ANCHORS)
for c in ["bbox_x1", "bbox_y1", "bbox_x2", "bbox_y2"]:
    anchors[c] = pd.to_numeric(anchors[c], errors="coerce")

anchor_by_scan = {str(k): v.copy() for k, v in anchors.groupby("scan_frame_id")}

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
    clip = clip_map.get(scan)
    if clip is None:
        continue

    print(f"[{idx}/{len(selected_scans)}] {scan}")

    frames, meta = read_frames(Path(clean_str(clip.get("clip_path"))))
    if frames is None or len(frames) == 0:
        issues.append({
            "item": scan,
            "issue_type": "warning_video_open_failed",
            "issue_detail": "Could not open/read clip video.",
            "severity": "warning",
        })
        continue

    anchor_frame, anchor_rel_sec = get_anchor_frame(clip)
    anchor_frame = max(0, min(len(frames) - 1, anchor_frame))

    an = anchor_by_scan.get(scan, pd.DataFrame())
    clip_rows = []

    for _, obj in an.iterrows():
        rows = track_one_object(frames, meta, anchor_frame, obj)

        for r in rows:
            r.update({
                "dataset_version": "week8_v52a_gt_seeded_tracking_pilot",
                "scan_frame_id": scan,
                "video_id": clean_str(clip.get("video_id")),
                "clip_path": clean_str(clip.get("clip_path")),
                "final_box_id": clean_str(obj.get("final_box_id")),
                "behaviour_pig_id": clean_str(obj.get("behaviour_pig_id")),
                "visual_marker_colour": clean_str(obj.get("visual_marker_colour")),
                "behaviour_code": clean_str(obj.get("behaviour_code")),
                "anchor_frame_index": int(anchor_frame),
                "anchor_rel_sec": float(anchor_rel_sec),
                "video_width": int(meta["width"]),
                "video_height": int(meta["height"]),
                "video_fps": float(meta["fps"]),
                "video_frame_count": int(meta["frame_count"]),
                "frames_read": int(meta["frames_read"]),
            })

        clip_rows.extend(rows)

        obj_df = pd.DataFrame(rows)
        object_summary_rows.append({
            "scan_frame_id": scan,
            "video_id": clean_str(clip.get("video_id")),
            "final_box_id": clean_str(obj.get("final_box_id")),
            "behaviour_pig_id": clean_str(obj.get("behaviour_pig_id")),
            "visual_marker_colour": clean_str(obj.get("visual_marker_colour")),
            "behaviour_code": clean_str(obj.get("behaviour_code")),
            "anchor_frame_index": int(anchor_frame),
            "frames_expected": int(len(frames)),
            "frames_output": int(obj_df["frame_index_in_clip"].nunique()),
            "tracked_status_rows": int((obj_df["status"] == "tracked").sum()),
            "anchor_rows": int((obj_df["status"] == "anchor_gt").sum()),
            "failure_rows": int((obj_df["status"].astype(str).str.contains("failed|exception|outside|drift", case=False, regex=True)).sum()),
            "coverage_ratio": float(obj_df["frame_index_in_clip"].nunique() / max(1, len(frames))),
            "tracker_name": clean_str(obj_df["tracker_name"].iloc[0]) if len(obj_df) else "",
        })

    clip_df = pd.DataFrame(clip_rows)
    all_rows.extend(clip_rows)

    expected_rows = len(an) * len(frames)
    actual_rows = len(clip_df)
    failure_rows = int((clip_df["status"].astype(str).str.contains("failed|exception|outside|drift", case=False, regex=True)).sum()) if len(clip_df) else 0
    coverage_ratio = actual_rows / max(1, expected_rows)

    clip_summary_rows.append({
        "scan_frame_id": scan,
        "video_id": clean_str(clip.get("video_id")),
        "anchor_object_count": int(len(an)),
        "frames_read": int(len(frames)),
        "expected_object_frame_rows": int(expected_rows),
        "actual_object_frame_rows": int(actual_rows),
        "coverage_ratio": float(coverage_ratio),
        "failure_rows": int(failure_rows),
        "failure_ratio": float(failure_rows / max(1, actual_rows)),
        "anchor_frame_index": int(anchor_frame),
        "status": "processed",
    })

    out_img = GALLERY / f"{scan}_gt_seeded_tracking_pilot.jpg"
    ok, msg = draw_gallery_image(scan, clip, frames, meta, an, clip_df, out_img)

    gallery_rows.append({
        "scan_frame_id": scan,
        "overlay_path": str(out_img),
        "saved": bool(ok),
        "message": msg,
        "anchor_object_count": int(len(an)),
        "frames_read": int(len(frames)),
        "coverage_ratio": float(coverage_ratio),
        "failure_ratio": float(failure_rows / max(1, actual_rows)),
    })


tracks_df = pd.DataFrame(all_rows)
clip_summary = pd.DataFrame(clip_summary_rows)
object_summary = pd.DataFrame(object_summary_rows)
gallery_index = pd.DataFrame(gallery_rows)

safe_to_csv(tracks_df, OUT_TRACKS)
safe_to_csv(clip_summary, OUT_CLIP_SUMMARY)
safe_to_csv(object_summary, OUT_OBJECT_SUMMARY)
safe_to_csv(gallery_index, OUT_GALLERY_INDEX)

# HTML gallery.
html = []
html.append("<!doctype html>")
html.append("<html><head><meta charset='utf-8'>")
html.append("<title>Week 8 v52a GT-seeded Tracking Pilot</title>")
html.append("<style>")
html.append("body{font-family:Arial;background:#111;color:#eee;margin:20px}")
html.append(".grid{display:grid;grid-template-columns:repeat(auto-fill,minmax(760px,1fr));gap:18px}")
html.append(".card{background:#1e1e1e;border:1px solid #333;border-radius:10px;padding:12px}")
html.append("img{width:100%;border:1px solid #444;border-radius:6px}")
html.append(".meta{font-size:13px;color:#bbb;line-height:1.4}")
html.append(".top{background:#2b2b2b;padding:12px;border-radius:10px;margin-bottom:18px}")
html.append("</style></head><body>")
html.append("<div class='top'>")
html.append("<h1>Week 8 v52a GT-seeded Tracking Pilot</h1>")
html.append("<p>Each trajectory is initialized from the anchor GT box. Panels show start/middle/end frames.</p>")
html.append("<p>This is a pilot for replacing pure detector-driven tracking with anchor-seeded propagation.</p>")
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
    html.append(f"frames read: {r['frames_read']}<br>")
    html.append(f"coverage ratio: {float(r['coverage_ratio']):.3f}<br>")
    html.append(f"failure ratio: {float(r['failure_ratio']):.3f}<br>")
    html.append("</div></div>")

html.append("</div></body></html>")
OUT_GALLERY_HTML.write_text("\n".join(html))

total_clips = int(len(clip_summary))
total_objects = int(len(object_summary))
total_rows = int(len(tracks_df))
mean_coverage = float(clip_summary["coverage_ratio"].mean()) if len(clip_summary) else 0.0
mean_failure = float(clip_summary["failure_ratio"].mean()) if len(clip_summary) else 1.0
full_coverage_clips = int((clip_summary["coverage_ratio"] >= 0.99).sum()) if len(clip_summary) else 0

if total_rows == 0:
    issues.append({
        "item": "gt_seeded_tracks",
        "issue_type": "hard_no_tracking_rows",
        "issue_detail": "GT-seeded tracking produced zero rows.",
        "severity": "hard",
    })

if mean_failure > 0.20:
    issues.append({
        "item": "gt_seeded_tracker_failure",
        "issue_type": "warning_high_mean_failure_ratio",
        "issue_detail": f"Mean failure ratio is {mean_failure:.4f}.",
        "severity": "warning",
    })

issues_df = pd.DataFrame(issues, columns=["item", "issue_type", "issue_detail", "severity"])
hard_issues = issues_df[issues_df["severity"] == "hard"] if len(issues_df) else pd.DataFrame()
warnings = issues_df[issues_df["severity"] == "warning"] if len(issues_df) else pd.DataFrame()

ready = len(hard_issues) == 0

decision = pd.DataFrame([{
    "v52a_decision": "gt_seeded_tracking_pilot_completed" if ready else "gt_seeded_tracking_pilot_blocked",
    "pilot_clip_count": total_clips,
    "pilot_object_count": total_objects,
    "tracking_rows_total": total_rows,
    "mean_clip_coverage_ratio": mean_coverage,
    "mean_failure_ratio": mean_failure,
    "full_coverage_clip_count": full_coverage_clips,
    "gallery_overlay_count": int(len(gallery_index)),
    "hard_issue_count": int(len(hard_issues)),
    "warning_count": int(len(warnings)),
    "issue_count": int(len(issues_df)),
    "ready_for_v52b_full_gt_seeded_tracking": bool(ready),
    "generated_at": datetime.now().isoformat(timespec="seconds"),
}])

safe_to_csv(issues_df, OUT_ISSUES)
safe_to_csv(decision, OUT_DECISION)

OUT_REPORT.write_text(
    "Week 8 v52a GT-seeded Tracking Pilot Report\n\n"
    f"Decision: {decision.iloc[0]['v52a_decision']}\n"
    f"Pilot clips: {total_clips}\n"
    f"Pilot objects: {total_objects}\n"
    f"Tracking rows total: {total_rows}\n"
    f"Mean coverage ratio: {mean_coverage:.4f}\n"
    f"Mean failure ratio: {mean_failure:.4f}\n"
    f"Full coverage clips: {full_coverage_clips}\n"
    f"Hard issues: {len(hard_issues)}\n"
    f"Warnings: {len(warnings)}\n\n"
    "Interpretation:\n"
    "This pilot initializes one tracker from each anchor GT box and propagates it through the 10-second clip. "
    "It directly targets the detector-miss problem observed in v51d.\n"
)

OUT_NOTE.write_text(
    "# Week 8 v52a GT-seeded Tracking Pilot\n\n"
    "## Summary\n\n"
    f"- v52a decision: {decision.iloc[0]['v52a_decision']}\n"
    f"- Pilot clips: {total_clips}\n"
    f"- Pilot objects: {total_objects}\n"
    f"- Tracking rows total: {total_rows}\n"
    f"- Mean coverage ratio: {mean_coverage:.4f}\n"
    f"- Mean failure ratio: {mean_failure:.4f}\n"
    f"- Full coverage clips: {full_coverage_clips}\n"
    f"- Hard issue count: {len(hard_issues)}\n"
    f"- Warning count: {len(warnings)}\n"
    f"- Ready for v52b full GT-seeded tracking: {ready}\n\n"
    "## Interpretation\n\n"
    "This approach starts from anchor GT boxes instead of relying only on detector boxes. "
    "It is the correct direction if the goal is to maximize tracking coverage for annotated pigs.\n\n"
    f"Gallery: {OUT_GALLERY_HTML}\n"
)

progress_row = pd.DataFrame([{
    "date": datetime.now().date().isoformat(),
    "stage": "v52a",
    "task_name": "GT-seeded tracking pilot",
    "status": "PASS" if ready else "BLOCKED",
    "input_summary": str(V45_ANCHORS),
    "output_summary": str(OUT),
    "hard_issues": int(len(hard_issues)),
    "warnings": int(len(warnings)),
    "next_action": "v52b full GT-seeded tracking" if ready else "Resolve GT-seeded tracking hard issues.",
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
print("=== v52a decision ===")
print(decision.to_string(index=False))

print()
print("=== v52a issues ===")
if len(issues_df):
    print(issues_df.to_string(index=False))
else:
    print("No issues found.")

print()
print("=== v52a clip summary ===")
print(clip_summary.to_string(index=False))
