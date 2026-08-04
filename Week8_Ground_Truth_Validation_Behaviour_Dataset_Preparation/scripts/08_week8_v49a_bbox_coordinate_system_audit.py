from pathlib import Path
from datetime import datetime
import json
import csv
import math
import pandas as pd
import numpy as np


ROOT = Path.home() / "PigBench"
W8 = ROOT / "Week8_Ground_Truth_Validation_Behaviour_Dataset_Preparation"

LOCKED_JSON = W8 / "outputs" / "v43_setup_input_audit" / "week8_v43c_final_locked_inputs.json"
V45_CLIP_JSON = W8 / "outputs" / "propagated_ground_truth" / "v45_label_propagation" / "week8_v45_clip_level_ground_truth.json"
V45_CLIP_OBJECT_CSV = W8 / "outputs" / "propagated_ground_truth" / "v45_label_propagation" / "week8_v45_clip_object_propagated_annotations.csv"

OUT = W8 / "outputs" / "v49a_bbox_coordinate_system_audit"
PREVIEWS = OUT / "preview_overlays"
VALIDATION = W8 / "validation"
REPORTS = W8 / "reports"
NOTES = W8 / "notes"
PROGRESS = W8 / "progress"

for p in [OUT, PREVIEWS, VALIDATION, REPORTS, NOTES, PROGRESS]:
    p.mkdir(parents=True, exist_ok=True)

OUT_CLIP_AUDIT = OUT / "week8_v49a_clip_dimension_bbox_audit.csv"
OUT_OBJECT_AUDIT = OUT / "week8_v49a_object_bbox_bounds_audit.csv"
OUT_DIAGNOSIS = OUT / "week8_v49a_resolution_diagnosis.csv"
OUT_PREVIEW_INDEX = OUT / "week8_v49a_preview_overlay_index.csv"
OUT_DECISION = OUT / "week8_v49a_decision_summary.csv"
OUT_ISSUES = OUT / "week8_v49a_issues.csv"
OUT_REPORT = REPORTS / "week8_v49a_bbox_coordinate_system_audit_report.md"
OUT_NOTE = NOTES / "week8_v49a_bbox_coordinate_system_audit_notes.md"
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


def read_locked():
    return json.loads(Path(LOCKED_JSON).read_text())


def path_for(locked, key):
    return Path(locked["locked_inputs"][key]["path"])


def get_video_info(path):
    info = {
        "exists": False,
        "open_ok": False,
        "width": None,
        "height": None,
        "fps": None,
        "frame_count": None,
        "duration_sec": None,
    }

    p = Path(path)
    info["exists"] = p.exists()

    if not p.exists():
        return info

    try:
        import cv2
        cap = cv2.VideoCapture(str(p))
        if not cap.isOpened():
            cap.release()
            return info

        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = float(cap.get(cv2.CAP_PROP_FPS))
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        cap.release()

        info["open_ok"] = True
        info["width"] = width if width > 0 else None
        info["height"] = height if height > 0 else None
        info["fps"] = fps if fps > 0 else None
        info["frame_count"] = frame_count if frame_count > 0 else None
        if fps > 0 and frame_count > 0:
            info["duration_sec"] = frame_count / fps
    except Exception:
        pass

    return info


def get_image_info(path):
    info = {
        "exists": False,
        "open_ok": False,
        "width": None,
        "height": None,
    }

    p = Path(path)
    info["exists"] = p.exists()

    if not p.exists():
        return info

    try:
        import cv2
        img = cv2.imread(str(p))
        if img is None:
            return info
        h, w = img.shape[:2]
        info["open_ok"] = True
        info["width"] = int(w)
        info["height"] = int(h)
    except Exception:
        pass

    return info


def bbox_inside(x1, y1, x2, y2, width, height):
    if width is None or height is None:
        return False
    if any(v is None for v in [x1, y1, x2, y2]):
        return False
    return x1 >= 0 and y1 >= 0 and x2 <= width and y2 <= height and x2 > x1 and y2 > y1


def bbox_valid_basic(x1, y1, x2, y2):
    if any(v is None for v in [x1, y1, x2, y2]):
        return False
    return x2 > x1 and y2 > y1


def draw_preview(clip_path, objects, out_path, title, scale_x=1.0, scale_y=1.0, anchor_ratio=0.5):
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
        frame_idx = 0
    else:
        frame_idx = max(0, min(frame_count - 1, int(round(frame_count * anchor_ratio))))

    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
    ok, frame = cap.read()
    cap.release()

    if not ok or frame is None:
        return False, "frame_read_failed"

    h, w = frame.shape[:2]

    colours = {
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

    for obj in objects:
        x1 = to_float(obj.get("bbox_x1", obj.get("bbox_xyxy", [None])[0] if obj.get("bbox_xyxy") else None))
        y1 = to_float(obj.get("bbox_y1", obj.get("bbox_xyxy", [None, None])[1] if obj.get("bbox_xyxy") else None))
        x2 = to_float(obj.get("bbox_x2", obj.get("bbox_xyxy", [None, None, None])[2] if obj.get("bbox_xyxy") else None))
        y2 = to_float(obj.get("bbox_y2", obj.get("bbox_xyxy", [None, None, None, None])[3] if obj.get("bbox_xyxy") else None))

        if any(v is None for v in [x1, y1, x2, y2]):
            continue

        x1 = int(round(x1 * scale_x))
        y1 = int(round(y1 * scale_y))
        x2 = int(round(x2 * scale_x))
        y2 = int(round(y2 * scale_y))

        colour = clean_str(obj.get("visual_marker_colour", "unknown"))
        bgr = colours.get(colour, (255, 255, 255))

        cv2.rectangle(frame, (x1, y1), (x2, y2), bgr, 3)

        label = f"{clean_str(obj.get('behaviour_pig_id'))}/{colour}/{clean_str(obj.get('behaviour_code'))}"
        label_y = max(20, y1 - 8)
        cv2.putText(frame, label, (x1, label_y), cv2.FONT_HERSHEY_SIMPLEX, 0.55, bgr, 2, cv2.LINE_AA)

    cv2.putText(frame, title, (15, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (255, 255, 255), 2, cv2.LINE_AA)

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    ok = cv2.imwrite(str(out_path), frame)
    return bool(ok), "saved" if ok else "write_failed"


issues = []

for required_path in [LOCKED_JSON, V45_CLIP_JSON, V45_CLIP_OBJECT_CSV]:
    if not required_path.exists():
        issues.append({
            "item": str(required_path),
            "issue_type": "hard_missing_required_input",
            "issue_detail": "Required input file is missing.",
            "severity": "hard",
        })

if issues:
    issues_df = pd.DataFrame(issues)
    safe_to_csv(issues_df, OUT_ISSUES)
    decision = pd.DataFrame([{
        "v49a_decision": "bbox_coordinate_audit_blocked_missing_inputs",
        "hard_issue_count": len(issues_df[issues_df["severity"] == "hard"]),
        "warning_count": len(issues_df[issues_df["severity"] == "warning"]),
        "ready_for_v49b_bbox_fix": False,
        "generated_at": datetime.now().isoformat(timespec="seconds"),
    }])
    safe_to_csv(decision, OUT_DECISION)
    print(decision.to_string(index=False))
    raise SystemExit(1)


locked = read_locked()
bf_path = path_for(locked, "behaviour_fusion_box_level_429")
clip_index_path = path_for(locked, "clip_extraction_index_72")

bf = pd.read_csv(bf_path)
clip_index = pd.read_csv(clip_index_path)
clip_obj = pd.read_csv(V45_CLIP_OBJECT_CSV)
clip_json = json.loads(V45_CLIP_JSON.read_text())
clips_json = clip_json.get("clips", [])

# Build maps
clip_index_map = {}
for _, r in clip_index.iterrows():
    scan = clean_str(r.get("scan_frame_id"))
    clip_index_map[scan] = r.to_dict()

bf_by_scan = {str(k): v.copy() for k, v in bf.groupby("scan_frame_id")}
clip_obj_by_scan = {str(k): v.copy() for k, v in clip_obj.groupby("scan_frame_id")}

clip_rows = []
object_rows = []
diagnosis_rows = []
preview_rows = []

for clip in clips_json:
    scan = clean_str(clip.get("scan_frame_id"))
    clip_path = clean_str(clip.get("clip_path"))
    preview_path = clean_str(clip.get("preview_frame_path"))

    ci = clip_index_map.get(scan, {})
    source_video_path = clean_str(ci.get("source_video_path", clip.get("source_video_path", "")))
    center_frame_index = to_int(ci.get("center_frame_index"), None)
    center_sec = to_float(ci.get("center_sec"), None)
    start_sec = to_float(ci.get("start_sec"), clip.get("start_sec", None))
    end_sec = to_float(ci.get("end_sec"), clip.get("end_sec", None))

    video_info = get_video_info(clip_path)
    preview_info = get_image_info(preview_path)

    source_frame_paths = []
    if scan in bf_by_scan and "frame_image_path" in bf_by_scan[scan].columns:
        source_frame_paths = [clean_str(x) for x in bf_by_scan[scan]["frame_image_path"].dropna().unique().tolist() if clean_str(x)]

    source_frame_info = {"exists": False, "open_ok": False, "width": None, "height": None}
    if source_frame_paths:
        source_frame_info = get_image_info(source_frame_paths[0])

    objs = clip_obj_by_scan.get(scan, pd.DataFrame())
    object_count = len(objs)

    video_w = video_info["width"]
    video_h = video_info["height"]

    inside_count = 0
    outside_count = 0
    basic_valid_count = 0
    max_x2 = None
    max_y2 = None

    for _, obj in objs.iterrows():
        x1 = to_float(obj.get("bbox_x1"))
        y1 = to_float(obj.get("bbox_y1"))
        x2 = to_float(obj.get("bbox_x2"))
        y2 = to_float(obj.get("bbox_y2"))

        valid_basic = bbox_valid_basic(x1, y1, x2, y2)
        inside = bbox_inside(x1, y1, x2, y2, video_w, video_h)

        if valid_basic:
            basic_valid_count += 1
        if inside:
            inside_count += 1
        else:
            outside_count += 1

        max_x2 = x2 if max_x2 is None else max(max_x2, x2 if x2 is not None else max_x2)
        max_y2 = y2 if max_y2 is None else max(max_y2, y2 if y2 is not None else max_y2)

        object_rows.append({
            "scan_frame_id": scan,
            "final_box_id": clean_str(obj.get("final_box_id")),
            "video_width": video_w,
            "video_height": video_h,
            "bbox_x1": x1,
            "bbox_y1": y1,
            "bbox_x2": x2,
            "bbox_y2": y2,
            "bbox_width": None if x1 is None or x2 is None else x2 - x1,
            "bbox_height": None if y1 is None or y2 is None else y2 - y1,
            "bbox_valid_basic": bool(valid_basic),
            "bbox_inside_clip_video": bool(inside),
            "visual_marker_colour": clean_str(obj.get("visual_marker_colour")),
            "behaviour_pig_id": clean_str(obj.get("behaviour_pig_id")),
            "behaviour_code": clean_str(obj.get("behaviour_code")),
            "clip_path": clip_path,
        })

    x_ratio = None
    y_ratio = None
    if video_w and max_x2:
        x_ratio = max_x2 / video_w
    if video_h and max_y2:
        y_ratio = max_y2 / video_h

    likely_issue = "unknown"
    suggested_action = "manual_review"

    source_w = source_frame_info.get("width") or preview_info.get("width")
    source_h = source_frame_info.get("height") or preview_info.get("height")

    scale_x_candidate = None
    scale_y_candidate = None

    if video_w and video_h and source_w and source_h:
        scale_x_candidate = video_w / source_w
        scale_y_candidate = video_h / source_h

    if object_count > 0 and outside_count == 0:
        likely_issue = "bbox_coordinates_fit_clip_video"
        suggested_action = "anchor_frame_visual_check"
    elif object_count > 0 and outside_count > 0 and source_w and source_h and max_x2 and max_y2 and max_x2 <= source_w * 1.05 and max_y2 <= source_h * 1.05 and (source_w != video_w or source_h != video_h):
        likely_issue = "probable_resolution_mismatch_source_frame_to_clip_video"
        suggested_action = "test_scaled_bbox_overlay"
    elif object_count > 0 and outside_count > 0 and x_ratio is not None and y_ratio is not None:
        likely_issue = "bbox_outside_clip_video_bounds"
        suggested_action = "inspect_coordinate_origin_and_resolution"
    elif object_count == 0:
        likely_issue = "no_objects_for_clip"
        suggested_action = "inspect_annotation_mapping"

    clip_rows.append({
        "scan_frame_id": scan,
        "clip_path": clip_path,
        "clip_exists": Path(clip_path).exists(),
        "video_open_ok": video_info["open_ok"],
        "video_width": video_w,
        "video_height": video_h,
        "video_fps": video_info["fps"],
        "video_frame_count": video_info["frame_count"],
        "video_duration_sec": video_info["duration_sec"],
        "preview_frame_path": preview_path,
        "preview_open_ok": preview_info["open_ok"],
        "preview_width": preview_info["width"],
        "preview_height": preview_info["height"],
        "source_frame_path_example": source_frame_paths[0] if source_frame_paths else "",
        "source_frame_open_ok": source_frame_info["open_ok"],
        "source_frame_width": source_frame_info["width"],
        "source_frame_height": source_frame_info["height"],
        "center_frame_index": center_frame_index,
        "center_sec": center_sec,
        "start_sec": start_sec,
        "end_sec": end_sec,
        "object_count": object_count,
        "bbox_basic_valid_count": basic_valid_count,
        "bbox_inside_clip_video_count": inside_count,
        "bbox_outside_clip_video_count": outside_count,
        "bbox_max_x2": max_x2,
        "bbox_max_y2": max_y2,
        "bbox_max_x2_over_video_width": x_ratio,
        "bbox_max_y2_over_video_height": y_ratio,
        "scale_x_candidate_source_to_clip": scale_x_candidate,
        "scale_y_candidate_source_to_clip": scale_y_candidate,
        "likely_issue": likely_issue,
        "suggested_action": suggested_action,
    })

    diagnosis_rows.append({
        "scan_frame_id": scan,
        "likely_issue": likely_issue,
        "suggested_action": suggested_action,
        "object_count": object_count,
        "bbox_outside_clip_video_count": outside_count,
        "video_size": f"{video_w}x{video_h}",
        "preview_size": f"{preview_info.get('width')}x{preview_info.get('height')}",
        "source_frame_size": f"{source_frame_info.get('width')}x{source_frame_info.get('height')}",
        "scale_x_candidate_source_to_clip": scale_x_candidate,
        "scale_y_candidate_source_to_clip": scale_y_candidate,
    })

clip_audit = pd.DataFrame(clip_rows)
object_audit = pd.DataFrame(object_rows)
diagnosis = pd.DataFrame(diagnosis_rows)

# Diagnostic preview overlays for selected clips:
# prioritize clips with outside boxes, then a few apparently valid clips.
selected_scans = []
outside_scans = clip_audit[clip_audit["bbox_outside_clip_video_count"] > 0]["scan_frame_id"].head(12).tolist()
valid_scans = clip_audit[clip_audit["bbox_outside_clip_video_count"] == 0]["scan_frame_id"].head(4).tolist()
selected_scans = outside_scans + [s for s in valid_scans if s not in outside_scans]

for scan in selected_scans:
    row = clip_audit[clip_audit["scan_frame_id"] == scan].iloc[0].to_dict()
    objs = clip_obj_by_scan.get(scan, pd.DataFrame()).to_dict("records")
    clip_path = row["clip_path"]

    out_as_is = PREVIEWS / f"{scan}_anchor_overlay_as_is.jpg"
    ok1, msg1 = draw_preview(
        clip_path,
        objs,
        out_as_is,
        f"{scan} as-is bbox",
        scale_x=1.0,
        scale_y=1.0,
        anchor_ratio=0.5,
    )

    scale_x = row.get("scale_x_candidate_source_to_clip")
    scale_y = row.get("scale_y_candidate_source_to_clip")

    out_scaled = ""
    ok2 = False
    msg2 = "not_attempted"

    if scale_x and scale_y and scale_x > 0 and scale_y > 0 and (abs(scale_x - 1.0) > 0.01 or abs(scale_y - 1.0) > 0.01):
        out_scaled_path = PREVIEWS / f"{scan}_anchor_overlay_scaled_candidate.jpg"
        ok2, msg2 = draw_preview(
            clip_path,
            objs,
            out_scaled_path,
            f"{scan} scaled candidate",
            scale_x=scale_x,
            scale_y=scale_y,
            anchor_ratio=0.5,
        )
        out_scaled = str(out_scaled_path)

    preview_rows.append({
        "scan_frame_id": scan,
        "as_is_preview_path": str(out_as_is),
        "as_is_saved": bool(ok1),
        "as_is_message": msg1,
        "scaled_candidate_preview_path": out_scaled,
        "scaled_candidate_saved": bool(ok2),
        "scaled_candidate_message": msg2,
        "scale_x_candidate": scale_x,
        "scale_y_candidate": scale_y,
        "likely_issue": row.get("likely_issue"),
    })

preview_index = pd.DataFrame(preview_rows)

# Issue creation
total_objects = len(object_audit)
outside_objects = int((object_audit["bbox_inside_clip_video"] == False).sum()) if total_objects else 0
invalid_basic_objects = int((object_audit["bbox_valid_basic"] == False).sum()) if total_objects else 0

clips_with_outside = int((clip_audit["bbox_outside_clip_video_count"] > 0).sum()) if len(clip_audit) else 0
clips_probable_resolution_mismatch = int((clip_audit["likely_issue"] == "probable_resolution_mismatch_source_frame_to_clip_video").sum()) if len(clip_audit) else 0
clips_fit = int((clip_audit["likely_issue"] == "bbox_coordinates_fit_clip_video").sum()) if len(clip_audit) else 0

if outside_objects > 0:
    issues.append({
        "item": "bbox_outside_clip_video_bounds",
        "issue_type": "warning_bbox_coordinates_outside_video",
        "issue_detail": f"{outside_objects}/{total_objects} objects have bbox coordinates outside clip video bounds.",
        "severity": "warning",
    })

if invalid_basic_objects > 0:
    issues.append({
        "item": "bbox_basic_validity",
        "issue_type": "hard_invalid_bbox_geometry",
        "issue_detail": f"{invalid_basic_objects}/{total_objects} objects have invalid bbox geometry.",
        "severity": "hard",
    })

if clips_probable_resolution_mismatch > 0:
    issues.append({
        "item": "resolution_mismatch",
        "issue_type": "warning_probable_resolution_mismatch",
        "issue_detail": f"{clips_probable_resolution_mismatch} clips likely need source-to-clip bbox scaling.",
        "severity": "warning",
    })

issues_df = pd.DataFrame(issues, columns=["item", "issue_type", "issue_detail", "severity"])

hard_issues = issues_df[issues_df["severity"] == "hard"] if len(issues_df) else pd.DataFrame()
warnings = issues_df[issues_df["severity"] == "warning"] if len(issues_df) else pd.DataFrame()

ready_for_fix = len(hard_issues) == 0

decision = pd.DataFrame([{
    "v49a_decision": "bbox_coordinate_system_audit_completed" if ready_for_fix else "bbox_coordinate_system_audit_blocked",
    "clip_count": int(len(clip_audit)),
    "object_count": int(total_objects),
    "clips_with_bbox_outside_video": int(clips_with_outside),
    "objects_with_bbox_outside_video": int(outside_objects),
    "objects_with_invalid_basic_bbox": int(invalid_basic_objects),
    "clips_probable_resolution_mismatch": int(clips_probable_resolution_mismatch),
    "clips_bbox_fit_video": int(clips_fit),
    "preview_overlay_count": int(len(preview_index)),
    "hard_issue_count": int(len(hard_issues)),
    "warning_count": int(len(warnings)),
    "issue_count": int(len(issues_df)),
    "ready_for_v49b_bbox_coordinate_fix": bool(ready_for_fix),
    "generated_at": datetime.now().isoformat(timespec="seconds"),
}])

safe_to_csv(clip_audit, OUT_CLIP_AUDIT)
safe_to_csv(object_audit, OUT_OBJECT_AUDIT)
safe_to_csv(diagnosis, OUT_DIAGNOSIS)
safe_to_csv(preview_index, OUT_PREVIEW_INDEX)
safe_to_csv(issues_df, OUT_ISSUES)
safe_to_csv(decision, OUT_DECISION)

report = f"""Week 8 v49a BBox Coordinate System Audit Report

Decision:
v49a decision: {decision.iloc[0]["v49a_decision"]}
Clip count: {len(clip_audit)}
Object count: {total_objects}
Clips with bbox outside video: {clips_with_outside}
Objects with bbox outside video: {outside_objects}
Objects with invalid basic bbox: {invalid_basic_objects}
Clips probable resolution mismatch: {clips_probable_resolution_mismatch}
Clips where bbox coordinates fit video: {clips_fit}
Hard issue count: {len(hard_issues)}
Warning count: {len(warnings)}

Interpretation:
If many boxes are outside the video frame, this is not merely motion drift. It indicates a coordinate-system or resolution mismatch between the annotation boxes and the clip video display resolution, or incorrect source association.

Outputs:
Clip audit: {OUT_CLIP_AUDIT}
Object audit: {OUT_OBJECT_AUDIT}
Resolution diagnosis: {OUT_DIAGNOSIS}
Preview overlay index: {OUT_PREVIEW_INDEX}
Preview overlays folder: {PREVIEWS}
Issues: {OUT_ISSUES}

Next step:
Inspect preview overlays. If scaled candidate overlays align better than as-is overlays, v49b should implement bbox coordinate scaling for the interface. If both fail, the source box/frame mapping must be reviewed before using tracking-refined boxes.
"""
OUT_REPORT.write_text(report)

OUT_NOTE.write_text(
    "# Week 8 v49a BBox Coordinate System Audit\n\n"
    "## Summary\n\n"
    f"- v49a decision: {decision.iloc[0]['v49a_decision']}\n"
    f"- Clip count: {len(clip_audit)}\n"
    f"- Object count: {total_objects}\n"
    f"- Clips with bbox outside video: {clips_with_outside}\n"
    f"- Objects with bbox outside video: {outside_objects}\n"
    f"- Objects with invalid basic bbox: {invalid_basic_objects}\n"
    f"- Clips probable resolution mismatch: {clips_probable_resolution_mismatch}\n"
    f"- Clips where bbox coordinates fit video: {clips_fit}\n"
    f"- Hard issue count: {len(hard_issues)}\n"
    f"- Warning count: {len(warnings)}\n"
    f"- Ready for v49b bbox coordinate fix: {ready_for_fix}\n\n"
    "## Interpretation\n\n"
    "If boxes are outside the video frame, this is stronger than normal motion drift. "
    "It suggests a coordinate-system or resolution mismatch that must be fixed before trusting visual identity/behaviour review.\n\n"
    "## Outputs\n\n"
    f"- Clip audit: {OUT_CLIP_AUDIT}\n"
    f"- Object audit: {OUT_OBJECT_AUDIT}\n"
    f"- Resolution diagnosis: {OUT_DIAGNOSIS}\n"
    f"- Preview index: {OUT_PREVIEW_INDEX}\n"
    f"- Preview folder: {PREVIEWS}\n"
)

progress_row = pd.DataFrame([{
    "date": datetime.now().date().isoformat(),
    "stage": "v49a",
    "task_name": "BBox coordinate system and anchor-frame audit",
    "status": "PASS" if ready_for_fix else "BLOCKED",
    "input_summary": str(V45_CLIP_OBJECT_CSV),
    "output_summary": str(OUT),
    "hard_issues": int(len(hard_issues)),
    "warnings": int(len(warnings)),
    "next_action": "v49b bbox coordinate fix or source mapping review",
}])

if OUT_PROGRESS.exists():
    old = pd.read_csv(OUT_PROGRESS)
    progress = pd.concat([old, progress_row], ignore_index=True)
else:
    progress = progress_row

safe_to_csv(progress, OUT_PROGRESS)

print("Saved:")
print(OUT_CLIP_AUDIT)
print(OUT_OBJECT_AUDIT)
print(OUT_DIAGNOSIS)
print(OUT_PREVIEW_INDEX)
print(OUT_DECISION)
print(OUT_ISSUES)
print(OUT_REPORT)
print(OUT_NOTE)

print()
print("=== v49a decision ===")
print(decision.to_string(index=False))

print()
print("=== v49a issues ===")
if len(issues_df):
    print(issues_df.to_string(index=False))
else:
    print("No issues found.")

print()
print("=== v49a diagnosis head ===")
print(diagnosis.head(20).to_string(index=False))

print()
print("=== v49a preview index ===")
if len(preview_index):
    print(preview_index.to_string(index=False))
else:
    print("No preview overlays generated.")
