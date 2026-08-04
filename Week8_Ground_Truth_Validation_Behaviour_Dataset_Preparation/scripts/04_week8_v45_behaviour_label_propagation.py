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

OUT = W8 / "outputs" / "propagated_ground_truth" / "v45_label_propagation"
OUT.mkdir(parents=True, exist_ok=True)

VALIDATION = W8 / "validation"
VALIDATION.mkdir(parents=True, exist_ok=True)

REPORTS = W8 / "reports"
REPORTS.mkdir(parents=True, exist_ok=True)

NOTES = W8 / "notes"
NOTES.mkdir(parents=True, exist_ok=True)

PROGRESS = W8 / "progress"
PROGRESS.mkdir(parents=True, exist_ok=True)

OUT_CLIP_OBJECT_CSV = OUT / "week8_v45_clip_object_propagated_annotations.csv"
OUT_FRAME_OBJECT_CSV = OUT / "week8_v45_frame_object_propagated_annotations.csv"
OUT_FRAME_JSONL = OUT / "week8_v45_frame_level_ground_truth.jsonl"
OUT_CLIP_JSON = OUT / "week8_v45_clip_level_ground_truth.json"
OUT_CLIP_SUMMARY = OUT / "week8_v45_clip_propagation_summary.csv"
OUT_QA = OUT / "week8_v45_propagation_qa_summary.csv"
OUT_ISSUES = OUT / "week8_v45_propagation_issues.csv"
OUT_DECISION = OUT / "week8_v45_decision_summary.csv"
OUT_REPORT = REPORTS / "week8_v45_behaviour_label_propagation_report.md"
OUT_NOTE = NOTES / "week8_v45_behaviour_label_propagation_notes.md"
OUT_PROGRESS = PROGRESS / "week8_experiment_progress_log.csv"


def safe_to_csv(df, path):
    df.to_csv(
        path,
        index=False,
        quoting=csv.QUOTE_ALL,
        escapechar="\\",
        lineterminator="\n",
    )


def clean_value(x):
    if pd.isna(x):
        return None
    if isinstance(x, np.integer):
        return int(x)
    if isinstance(x, np.floating):
        if math.isnan(float(x)):
            return None
        return float(x)
    if isinstance(x, np.bool_):
        return bool(x)
    return x


def clean_str(x):
    if pd.isna(x):
        return ""
    s = str(x).strip()
    if s.lower() in ["nan", "none", "null"]:
        return ""
    return s


def clean_record(d):
    return {k: clean_value(v) for k, v in d.items()}


def read_locked():
    return json.loads(Path(LOCKED_JSON).read_text())


def path_for(locked, key):
    return Path(locked["locked_inputs"][key]["path"])


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


def bool_from_value(x):
    s = clean_str(x).lower()
    return s in ["true", "1", "yes", "y"]


def get_video_info(path):
    info = {
        "cv2_available": False,
        "video_open_ok": False,
        "actual_frame_count": None,
        "actual_fps": None,
        "actual_duration_sec": None,
        "video_width": None,
        "video_height": None,
        "video_info_source": "fallback",
    }

    try:
        import cv2
        info["cv2_available"] = True
    except Exception:
        return info

    p = str(path)
    cap = cv2.VideoCapture(p)
    if not cap.isOpened():
        cap.release()
        return info

    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = float(cap.get(cv2.CAP_PROP_FPS))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    cap.release()

    info["video_open_ok"] = True
    info["actual_frame_count"] = frame_count if frame_count > 0 else None
    info["actual_fps"] = fps if fps and fps > 0 else None
    info["actual_duration_sec"] = (frame_count / fps) if frame_count > 0 and fps and fps > 0 else None
    info["video_width"] = width if width > 0 else None
    info["video_height"] = height if height > 0 else None
    info["video_info_source"] = "cv2"
    return info


def object_flags(row):
    flags = []

    behaviour_code = clean_str(row.get("behaviour_code"))
    visual_colour = clean_str(row.get("visual_marker_colour_v18c"))
    identity_status = clean_str(row.get("final_identity_status_v17")).lower()

    if not behaviour_code:
        flags.append("missing_behaviour_label")

    if not visual_colour:
        flags.append("missing_colour_identity")

    if visual_colour in ["unknown", "not_visible", "uncertain", "unassigned"]:
        flags.append("non_training_colour_identity")

    if "not_visible" in identity_status:
        flags.append("not_visible")

    if "uncertain" in identity_status:
        flags.append("uncertain_identity")

    if bool_from_value(row.get("has_matched_behaviour_label")) is False:
        flags.append("behaviour_not_matched")

    if bool_from_value(row.get("ready_for_behaviour_model_training")) is False:
        flags.append("not_training_ready")

    return sorted(set(flags))


def markdown_table(df, max_rows=50):
    if df is None or len(df) == 0:
        return "_No rows._"
    d = df.head(max_rows)
    cols = list(d.columns)
    out = []
    out.append("| " + " | ".join(cols) + " |")
    out.append("| " + " | ".join(["---"] * len(cols)) + " |")
    for _, r in d.iterrows():
        vals = []
        for c in cols:
            x = str(r.get(c, ""))
            x = x.replace("\n", " ").replace("|", "/")
            vals.append(x)
        out.append("| " + " | ".join(vals) + " |")
    return "\n".join(out)


locked = read_locked()

bf_path = path_for(locked, "behaviour_fusion_box_level_429")
tr_path = path_for(locked, "training_ready_primary_split_374")
clip_path = path_for(locked, "clip_extraction_index_72")
clip_level_path = path_for(locked, "clip_level_multilabel_index_72")

bf = pd.read_csv(bf_path)
tr = pd.read_csv(tr_path)
clips = pd.read_csv(clip_path)
clip_level = pd.read_csv(clip_level_path)

issues = []
clip_object_rows = []
frame_object_rows = []
clip_summary_rows = []
clip_json_records = []

# Maps from clip-level table
clip_level_map = {}
for _, r in clip_level.iterrows():
    rr = clean_record(r.to_dict())
    scan = clean_str(rr.get("scan_frame_id"))
    clip_level_map[scan] = rr

# Behaviour fusion rows grouped by scan_frame_id
bf_by_scan = {
    str(k): v.copy()
    for k, v in bf.groupby("scan_frame_id")
}

frame_jsonl_f = open(OUT_FRAME_JSONL, "w", encoding="utf-8")

total_frames_generated = 0
total_frame_object_rows = 0
total_clip_object_rows = 0

for _, clip_row in clips.iterrows():
    c = clean_record(clip_row.to_dict())
    scan_frame_id = clean_str(c.get("scan_frame_id"))
    clip_level_row = clip_level_map.get(scan_frame_id, {})

    fps_used = to_float(c.get("fps_used"), 25.0) or 25.0
    start_sec = to_float(c.get("start_sec"), 0.0) or 0.0
    end_sec = to_float(c.get("end_sec"), start_sec + 10.0) or (start_sec + 10.0)
    duration_sec = to_float(c.get("duration_sec"), end_sec - start_sec) or (end_sec - start_sec)
    start_frame_adjusted = to_int(c.get("start_frame_adjusted"), int(round(start_sec * fps_used))) or int(round(start_sec * fps_used))

    clip_file = Path(clean_str(c.get("clip_path")))
    clip_exists = clip_file.exists()

    video_info = get_video_info(clip_file) if clip_exists else {
        "cv2_available": False,
        "video_open_ok": False,
        "actual_frame_count": None,
        "actual_fps": None,
        "actual_duration_sec": None,
        "video_width": None,
        "video_height": None,
        "video_info_source": "missing_clip",
    }

    fallback_frame_count = max(1, int(round(duration_sec * fps_used)))
    frame_count = video_info.get("actual_frame_count") or fallback_frame_count
    actual_fps = video_info.get("actual_fps") or fps_used

    if not clip_exists:
        issues.append({
            "scan_frame_id": scan_frame_id,
            "item": "clip_path",
            "issue_type": "hard_missing_clip_file",
            "issue_detail": clean_str(c.get("clip_path")),
            "severity": "hard",
        })

    if frame_count <= 0:
        issues.append({
            "scan_frame_id": scan_frame_id,
            "item": "frame_count",
            "issue_type": "hard_invalid_frame_count",
            "issue_detail": f"frame_count={frame_count}",
            "severity": "hard",
        })

    if not video_info.get("video_open_ok"):
        issues.append({
            "scan_frame_id": scan_frame_id,
            "item": "video_open",
            "issue_type": "warning_video_open_fallback_used",
            "issue_detail": f"clip_exists={clip_exists}; generated_frame_count={frame_count}",
            "severity": "warning",
        })

    objects_df = bf_by_scan.get(scan_frame_id, pd.DataFrame(columns=bf.columns))
    object_count = len(objects_df)

    if object_count == 0:
        issues.append({
            "scan_frame_id": scan_frame_id,
            "item": "objects",
            "issue_type": "warning_no_objects_for_clip",
            "issue_detail": "No behaviour_fusion rows for this scan_frame_id.",
            "severity": "warning",
        })

    split = clean_str(clip_level_row.get("split"))
    if not split:
        split = "unmapped"

    behaviour_set = clean_str(clip_level_row.get("behaviour_set")) or clean_str(c.get("behaviour_codes_present"))

    clip_objects = []

    for _, obj_row in objects_df.iterrows():
        obj = clean_record(obj_row.to_dict())

        final_box_id = clean_str(obj.get("final_box_id"))
        behaviour_code = clean_str(obj.get("behaviour_code"))
        behaviour_label = clean_str(obj.get("behaviour_label")) or behaviour_code
        visual_colour = clean_str(obj.get("visual_marker_colour_v18c")) or clean_str(obj.get("final_colour_identity_v17")) or "unknown"
        behaviour_pig_id = clean_str(obj.get("behaviour_pig_id_v18c")) or clean_str(obj.get("pig_id"))
        identity_status = clean_str(obj.get("final_identity_status_v17"))
        flags = object_flags(obj)

        x1 = to_float(obj.get("x1"))
        y1 = to_float(obj.get("y1"))
        x2 = to_float(obj.get("x2"))
        y2 = to_float(obj.get("y2"))

        has_behaviour = bool(behaviour_code)
        is_training_ready = bool_from_value(obj.get("ready_for_behaviour_model_training"))

        clip_object_record = {
            "dataset_version": "week8_v45",
            "scan_frame_id": scan_frame_id,
            "video_id": clean_str(c.get("video_id")),
            "clip_path": clean_str(c.get("clip_path")),
            "source_video_path": clean_str(c.get("source_video_path")),
            "split": split,
            "start_sec": start_sec,
            "end_sec": end_sec,
            "duration_sec": duration_sec,
            "fps_used": fps_used,
            "generated_frame_count": int(frame_count),
            "final_box_id": final_box_id,
            "bbox_x1": x1,
            "bbox_y1": y1,
            "bbox_x2": x2,
            "bbox_y2": y2,
            "bbox_source": "scanpoint_anchor_repeated_across_interval",
            "visual_marker_colour": visual_colour,
            "behaviour_pig_id": behaviour_pig_id,
            "behaviour_code": behaviour_code,
            "behaviour_label": behaviour_label,
            "identity_status": identity_status,
            "has_behaviour_label": bool(has_behaviour),
            "is_training_ready": bool(is_training_ready),
            "label_source": "propagated_from_10_second_observation_window",
            "validation_status": "unchecked",
            "validation_flags": "|".join(flags),
            "propagation_scope": "full_annotated_clip_interval",
        }

        clip_object_rows.append(clip_object_record)
        clip_objects.append(clip_object_record)

    for frame_idx in range(int(frame_count)):
        source_frame_index = int(start_frame_adjusted + frame_idx)
        timestamp_sec = float(start_sec + frame_idx / fps_used)

        frame_objects_json = []

        for obj in clip_objects:
            frame_object_record = {
                "dataset_version": "week8_v45",
                "scan_frame_id": scan_frame_id,
                "video_id": obj["video_id"],
                "clip_path": obj["clip_path"],
                "split": split,
                "frame_index_in_clip": int(frame_idx),
                "source_frame_index": source_frame_index,
                "timestamp_sec": timestamp_sec,
                "final_box_id": obj["final_box_id"],
                "bbox_x1": obj["bbox_x1"],
                "bbox_y1": obj["bbox_y1"],
                "bbox_x2": obj["bbox_x2"],
                "bbox_y2": obj["bbox_y2"],
                "bbox_source": obj["bbox_source"],
                "visual_marker_colour": obj["visual_marker_colour"],
                "behaviour_pig_id": obj["behaviour_pig_id"],
                "behaviour_code": obj["behaviour_code"],
                "behaviour_label": obj["behaviour_label"],
                "identity_status": obj["identity_status"],
                "has_behaviour_label": obj["has_behaviour_label"],
                "is_training_ready": obj["is_training_ready"],
                "label_source": obj["label_source"],
                "validation_status": obj["validation_status"],
                "validation_flags": obj["validation_flags"],
                "propagation_scope": obj["propagation_scope"],
            }

            frame_object_rows.append(frame_object_record)
            frame_objects_json.append({
                "final_box_id": obj["final_box_id"],
                "bbox_xyxy": [obj["bbox_x1"], obj["bbox_y1"], obj["bbox_x2"], obj["bbox_y2"]],
                "visual_marker_colour": obj["visual_marker_colour"],
                "behaviour_pig_id": obj["behaviour_pig_id"],
                "behaviour_code": obj["behaviour_code"],
                "behaviour_label": obj["behaviour_label"],
                "identity_status": obj["identity_status"],
                "label_source": obj["label_source"],
                "validation_status": obj["validation_status"],
                "validation_flags": obj["validation_flags"].split("|") if obj["validation_flags"] else [],
            })

        frame_jsonl_f.write(json.dumps({
            "dataset_version": "week8_v45",
            "scan_frame_id": scan_frame_id,
            "video_id": clean_str(c.get("video_id")),
            "clip_path": clean_str(c.get("clip_path")),
            "frame_index_in_clip": int(frame_idx),
            "source_frame_index": source_frame_index,
            "timestamp_sec": timestamp_sec,
            "objects": frame_objects_json,
        }) + "\n")

    total_frames_generated += int(frame_count)
    total_clip_object_rows += int(object_count)
    total_frame_object_rows += int(frame_count) * int(object_count)

    has_missing_behaviour = any(not bool(clean_str(x.get("behaviour_code"))) for _, x in objects_df.iterrows()) if object_count > 0 else False
    training_ready_count = int(sum(bool_from_value(x.get("ready_for_behaviour_model_training")) for _, x in objects_df.iterrows())) if object_count > 0 else 0

    clip_summary_rows.append({
        "scan_frame_id": scan_frame_id,
        "video_id": clean_str(c.get("video_id")),
        "clip_path": clean_str(c.get("clip_path")),
        "clip_exists": bool(clip_exists),
        "video_open_ok": bool(video_info.get("video_open_ok")),
        "video_info_source": video_info.get("video_info_source"),
        "fps_used": fps_used,
        "actual_fps": video_info.get("actual_fps"),
        "start_sec": start_sec,
        "end_sec": end_sec,
        "duration_sec": duration_sec,
        "generated_frame_count": int(frame_count),
        "actual_frame_count": video_info.get("actual_frame_count"),
        "object_count": int(object_count),
        "training_ready_object_count": int(training_ready_count),
        "frame_object_rows_generated": int(frame_count) * int(object_count),
        "split": split,
        "behaviour_set": behaviour_set,
        "has_missing_behaviour": bool(has_missing_behaviour),
        "propagation_status": "propagated",
    })

    clip_json_records.append({
        "scan_frame_id": scan_frame_id,
        "video_id": clean_str(c.get("video_id")),
        "source_video_path": clean_str(c.get("source_video_path")),
        "clip_path": clean_str(c.get("clip_path")),
        "preview_frame_path": clean_str(c.get("preview_frame_path")),
        "split": split,
        "start_sec": start_sec,
        "end_sec": end_sec,
        "duration_sec": duration_sec,
        "fps_used": fps_used,
        "generated_frame_count": int(frame_count),
        "behaviour_set": behaviour_set,
        "objects": [
            {
                "final_box_id": o["final_box_id"],
                "bbox_xyxy": [o["bbox_x1"], o["bbox_y1"], o["bbox_x2"], o["bbox_y2"]],
                "bbox_source": o["bbox_source"],
                "visual_marker_colour": o["visual_marker_colour"],
                "behaviour_pig_id": o["behaviour_pig_id"],
                "behaviour_code": o["behaviour_code"],
                "behaviour_label": o["behaviour_label"],
                "identity_status": o["identity_status"],
                "has_behaviour_label": o["has_behaviour_label"],
                "is_training_ready": o["is_training_ready"],
                "label_source": o["label_source"],
                "validation_status": o["validation_status"],
                "validation_flags": o["validation_flags"].split("|") if o["validation_flags"] else [],
            }
            for o in clip_objects
        ],
    })

frame_jsonl_f.close()

clip_object_df = pd.DataFrame(clip_object_rows)
frame_object_df = pd.DataFrame(frame_object_rows)
clip_summary_df = pd.DataFrame(clip_summary_rows)

# QA metrics
clip_count = len(clips)
processed_clip_count = len(clip_summary_df)
clips_with_objects = int((clip_summary_df["object_count"] > 0).sum()) if len(clip_summary_df) else 0
clips_without_objects = int((clip_summary_df["object_count"] == 0).sum()) if len(clip_summary_df) else 0
clips_with_missing_behaviour = int(clip_summary_df["has_missing_behaviour"].sum()) if "has_missing_behaviour" in clip_summary_df.columns else 0
clips_unmapped = int((clip_summary_df["split"] == "unmapped").sum()) if "split" in clip_summary_df.columns else 0

objects_total = len(clip_object_df)
objects_with_behaviour = int(clip_object_df["has_behaviour_label"].sum()) if len(clip_object_df) else 0
objects_training_ready = int(clip_object_df["is_training_ready"].sum()) if len(clip_object_df) else 0
objects_missing_behaviour = int(objects_total - objects_with_behaviour)

frame_objects_total = len(frame_object_df)
frame_objects_with_behaviour = int(frame_object_df["has_behaviour_label"].sum()) if len(frame_object_df) else 0
frame_objects_training_ready = int(frame_object_df["is_training_ready"].sum()) if len(frame_object_df) else 0
frame_objects_missing_behaviour = int(frame_objects_total - frame_objects_with_behaviour)

# Add informational issues for expected unknown/unmapped cases
if objects_missing_behaviour > 0:
    issues.append({
        "scan_frame_id": "ALL",
        "item": "missing_behaviour_objects",
        "issue_type": "info_missing_behaviour_preserved",
        "issue_detail": f"{objects_missing_behaviour} clip-level objects have no behaviour label and are preserved for validation.",
        "severity": "info",
    })

if clips_unmapped > 0:
    issues.append({
        "scan_frame_id": "ALL",
        "item": "unmapped_clips",
        "issue_type": "info_unmapped_clips_preserved",
        "issue_detail": f"{clips_unmapped} clips have split=unmapped and are preserved for validation.",
        "severity": "info",
    })

issues_df = pd.DataFrame(issues, columns=["scan_frame_id", "item", "issue_type", "issue_detail", "severity"])

hard_issues = issues_df[issues_df["severity"] == "hard"] if len(issues_df) else pd.DataFrame()
warnings = issues_df[issues_df["severity"] == "warning"] if len(issues_df) else pd.DataFrame()
infos = issues_df[issues_df["severity"] == "info"] if len(issues_df) else pd.DataFrame()

ready_for_v46 = len(hard_issues) == 0

qa_df = pd.DataFrame([{
    "metric": "clip_count_input",
    "value": clip_count,
}, {
    "metric": "processed_clip_count",
    "value": processed_clip_count,
}, {
    "metric": "clips_with_objects",
    "value": clips_with_objects,
}, {
    "metric": "clips_without_objects",
    "value": clips_without_objects,
}, {
    "metric": "clips_with_missing_behaviour",
    "value": clips_with_missing_behaviour,
}, {
    "metric": "clips_unmapped",
    "value": clips_unmapped,
}, {
    "metric": "total_frames_generated",
    "value": total_frames_generated,
}, {
    "metric": "clip_object_rows",
    "value": objects_total,
}, {
    "metric": "clip_objects_with_behaviour",
    "value": objects_with_behaviour,
}, {
    "metric": "clip_objects_training_ready",
    "value": objects_training_ready,
}, {
    "metric": "clip_objects_missing_behaviour",
    "value": objects_missing_behaviour,
}, {
    "metric": "frame_object_rows",
    "value": frame_objects_total,
}, {
    "metric": "frame_objects_with_behaviour",
    "value": frame_objects_with_behaviour,
}, {
    "metric": "frame_objects_training_ready",
    "value": frame_objects_training_ready,
}, {
    "metric": "frame_objects_missing_behaviour",
    "value": frame_objects_missing_behaviour,
}])

decision = pd.DataFrame([{
    "v45_decision": "behaviour_label_propagation_completed" if ready_for_v46 else "behaviour_label_propagation_blocked",
    "clip_count_input": int(clip_count),
    "processed_clip_count": int(processed_clip_count),
    "total_frames_generated": int(total_frames_generated),
    "clip_object_rows": int(objects_total),
    "frame_object_rows": int(frame_objects_total),
    "clip_objects_with_behaviour": int(objects_with_behaviour),
    "clip_objects_training_ready": int(objects_training_ready),
    "clip_objects_missing_behaviour": int(objects_missing_behaviour),
    "clips_unmapped": int(clips_unmapped),
    "hard_issue_count": int(len(hard_issues)),
    "warning_count": int(len(warnings)),
    "info_count": int(len(infos)),
    "issue_count": int(len(issues_df)),
    "label_source": "propagated_from_10_second_observation_window",
    "bbox_source": "scanpoint_anchor_repeated_across_interval",
    "ready_for_v46_propagation_qa": bool(ready_for_v46),
    "generated_at": datetime.now().isoformat(timespec="seconds"),
}])

clip_level_json = {
    "dataset_version": "week8_v45",
    "created_at": datetime.now().isoformat(timespec="seconds"),
    "annotation_protocol": {
        "observation_window_sec": 10,
        "behaviour_label_scope": "full_annotated_clip_interval",
        "propagation_rule": "propagate_pig_behaviour_to_every_frame_in_interval",
        "label_source": "propagated_from_10_second_observation_window",
        "bbox_source": "scanpoint_anchor_repeated_across_interval",
    },
    "summary": decision.iloc[0].to_dict(),
    "clips": clip_json_records,
}

OUT_CLIP_JSON.write_text(json.dumps(clip_level_json, indent=2))

safe_to_csv(clip_object_df, OUT_CLIP_OBJECT_CSV)
safe_to_csv(frame_object_df, OUT_FRAME_OBJECT_CSV)
safe_to_csv(clip_summary_df, OUT_CLIP_SUMMARY)
safe_to_csv(qa_df, OUT_QA)
safe_to_csv(issues_df, OUT_ISSUES)
safe_to_csv(decision, OUT_DECISION)

report = f"""# Week 8 v45 Behaviour Label Propagation Report

## Decision

- v45 decision: {decision.iloc[0]["v45_decision"]}
- Input clips: {clip_count}
- Processed clips: {processed_clip_count}
- Total frames generated: {total_frames_generated}
- Clip-object rows: {objects_total}
- Frame-object rows: {frame_objects_total}
- Clip objects with behaviour labels: {objects_with_behaviour}
- Clip objects training-ready: {objects_training_ready}
- Clip objects missing behaviour: {objects_missing_behaviour}
- Unmapped clips: {clips_unmapped}
- Hard issue count: {len(hard_issues)}
- Warning count: {len(warnings)}
- Ready for v46 propagation QA: {ready_for_v46}

## Propagation rule

Each annotated scanpoint defines a 10-second observation interval. Pig-level behaviour labels are propagated to every frame within that interval.

## Important scope note

The current v45 geometry source is scanpoint-anchor boxes repeated across the interval. This is sufficient for label propagation and validation indexing. Tracking-refined per-frame boxes can be integrated later through the Week 8 identity/tracking improvement stage.

## Outputs

- Clip-object annotations: {OUT_CLIP_OBJECT_CSV}
- Frame-object annotations: {OUT_FRAME_OBJECT_CSV}
- Frame-level JSONL: {OUT_FRAME_JSONL}
- Clip-level JSON: {OUT_CLIP_JSON}
- Clip propagation summary: {OUT_CLIP_SUMMARY}
- QA summary: {OUT_QA}
- Issues: {OUT_ISSUES}

## Next step

v46 should perform propagation QA and consistency checks before the visualization interface is built.
"""

OUT_REPORT.write_text(report)

OUT_NOTE.write_text(
    "# Week 8 v45 Behaviour Label Propagation\n\n"
    "## Summary\n\n"
    f"- v45 decision: {decision.iloc[0]['v45_decision']}\n"
    f"- Processed clips: {processed_clip_count} / {clip_count}\n"
    f"- Total frames generated: {total_frames_generated}\n"
    f"- Clip-object rows: {objects_total}\n"
    f"- Frame-object rows: {frame_objects_total}\n"
    f"- Clip objects with behaviour: {objects_with_behaviour}\n"
    f"- Clip objects training-ready: {objects_training_ready}\n"
    f"- Clip objects missing behaviour: {objects_missing_behaviour}\n"
    f"- Unmapped clips: {clips_unmapped}\n"
    f"- Hard issue count: {len(hard_issues)}\n"
    f"- Warning count: {len(warnings)}\n"
    f"- Ready for v46 propagation QA: {ready_for_v46}\n\n"
    "## Key scope note\n\n"
    "Behaviour labels are propagated over the full annotated 10-second interval. "
    "Bounding boxes are scanpoint-anchor boxes repeated across the interval in v45; "
    "tracking-refined boxes can be integrated in a later identity/tracking refinement step.\n\n"
    "## Outputs\n\n"
    f"- Clip-object CSV: {OUT_CLIP_OBJECT_CSV}\n"
    f"- Frame-object CSV: {OUT_FRAME_OBJECT_CSV}\n"
    f"- Frame-level JSONL: {OUT_FRAME_JSONL}\n"
    f"- Clip-level JSON: {OUT_CLIP_JSON}\n"
    f"- QA summary: {OUT_QA}\n"
    f"- Issues: {OUT_ISSUES}\n"
)

progress_row = pd.DataFrame([{
    "date": datetime.now().date().isoformat(),
    "stage": "v45",
    "task_name": "Behaviour label propagation across full 10-second clips",
    "status": "PASS" if ready_for_v46 else "BLOCKED",
    "input_summary": str(LOCKED_JSON),
    "output_summary": str(OUT),
    "hard_issues": int(len(hard_issues)),
    "warnings": int(len(warnings)),
    "next_action": "v46 propagation QA and consistency validation" if ready_for_v46 else "Resolve propagation hard issues.",
}])

if OUT_PROGRESS.exists():
    old = pd.read_csv(OUT_PROGRESS)
    progress = pd.concat([old, progress_row], ignore_index=True)
else:
    progress = progress_row

safe_to_csv(progress, OUT_PROGRESS)

print("Saved:")
print(OUT_CLIP_OBJECT_CSV)
print(OUT_FRAME_OBJECT_CSV)
print(OUT_FRAME_JSONL)
print(OUT_CLIP_JSON)
print(OUT_CLIP_SUMMARY)
print(OUT_QA)
print(OUT_DECISION)
print(OUT_ISSUES)
print(OUT_REPORT)
print(OUT_NOTE)

print()
print("=== v45 decision ===")
print(decision.to_string(index=False))

print()
print("=== v45 QA ===")
print(qa_df.to_string(index=False))

print()
print("=== v45 issues ===")
if len(issues_df):
    print(issues_df.to_string(index=False))
else:
    print("No issues found.")
