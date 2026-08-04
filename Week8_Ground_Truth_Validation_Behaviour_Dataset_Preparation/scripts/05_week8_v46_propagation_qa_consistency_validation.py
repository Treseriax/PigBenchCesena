from pathlib import Path
from datetime import datetime
import json
import csv
import pandas as pd
import numpy as np


ROOT = Path.home() / "PigBench"
W8 = ROOT / "Week8_Ground_Truth_Validation_Behaviour_Dataset_Preparation"

V45 = W8 / "outputs" / "propagated_ground_truth" / "v45_label_propagation"

CLIP_OBJECT_CSV = V45 / "week8_v45_clip_object_propagated_annotations.csv"
FRAME_OBJECT_CSV = V45 / "week8_v45_frame_object_propagated_annotations.csv"
FRAME_JSONL = V45 / "week8_v45_frame_level_ground_truth.jsonl"
CLIP_JSON = V45 / "week8_v45_clip_level_ground_truth.json"
CLIP_SUMMARY = V45 / "week8_v45_clip_propagation_summary.csv"
V45_DECISION = V45 / "week8_v45_decision_summary.csv"
V45_QA = V45 / "week8_v45_propagation_qa_summary.csv"

OUT = W8 / "outputs" / "propagated_ground_truth" / "v46_propagation_qa"
OUT.mkdir(parents=True, exist_ok=True)

VALIDATION = W8 / "validation"
VALIDATION.mkdir(parents=True, exist_ok=True)

REPORTS = W8 / "reports"
REPORTS.mkdir(parents=True, exist_ok=True)

NOTES = W8 / "notes"
NOTES.mkdir(parents=True, exist_ok=True)

PROGRESS = W8 / "progress"
PROGRESS.mkdir(parents=True, exist_ok=True)

OUT_CONSISTENCY = OUT / "week8_v46_propagation_consistency_checks.csv"
OUT_OBJECT_COUNTS = OUT / "week8_v46_object_frame_count_consistency.csv"
OUT_CLIP_COUNTS = OUT / "week8_v46_clip_count_consistency.csv"
OUT_LABEL_STABILITY = OUT / "week8_v46_label_stability_by_object.csv"
OUT_TIMESTAMP_QA = OUT / "week8_v46_timestamp_consistency.csv"
OUT_BBOX_QA = OUT / "week8_v46_bbox_consistency.csv"
OUT_JSONL_QA = OUT / "week8_v46_jsonl_consistency.csv"
OUT_VALIDATION_ISSUES = VALIDATION / "week8_v46_propagation_validation_issues.csv"
OUT_DECISION = OUT / "week8_v46_decision_summary.csv"
OUT_REPORT = REPORTS / "week8_v46_propagation_qa_consistency_report.md"
OUT_NOTE = NOTES / "week8_v46_propagation_qa_notes.md"
OUT_PROGRESS = PROGRESS / "week8_experiment_progress_log.csv"


def safe_to_csv(df, path):
    df.to_csv(
        path,
        index=False,
        quoting=csv.QUOTE_ALL,
        escapechar="\\",
        lineterminator="\n",
    )


def read_csv(path):
    return pd.read_csv(path)


def add_issue(issues, check_name, item, issue_type, detail, severity):
    issues.append({
        "check_name": check_name,
        "item": item,
        "issue_type": issue_type,
        "issue_detail": detail,
        "severity": severity,
    })


def clean_str(x):
    if pd.isna(x):
        return ""
    s = str(x).strip()
    if s.lower() in ["nan", "none", "null"]:
        return ""
    return s


def bool_series(s):
    if s.dtype == bool:
        return s
    return s.astype(str).str.lower().isin(["true", "1", "yes", "y"])


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


issues = []

required_files = [
    CLIP_OBJECT_CSV,
    FRAME_OBJECT_CSV,
    FRAME_JSONL,
    CLIP_JSON,
    CLIP_SUMMARY,
    V45_DECISION,
    V45_QA,
]

for p in required_files:
    if not p.exists():
        add_issue(
            issues,
            "required_files",
            str(p),
            "hard_missing_required_v45_output",
            "Required v45 output file is missing.",
            "hard",
        )

if any(not p.exists() for p in required_files):
    issues_df = pd.DataFrame(issues)
    safe_to_csv(issues_df, OUT_VALIDATION_ISSUES)
    decision = pd.DataFrame([{
        "v46_decision": "propagation_qa_blocked_missing_files",
        "hard_issue_count": int((issues_df["severity"] == "hard").sum()),
        "warning_count": int((issues_df["severity"] == "warning").sum()),
        "info_count": int((issues_df["severity"] == "info").sum()),
        "ready_for_v47_visualization_interface": False,
        "generated_at": datetime.now().isoformat(timespec="seconds"),
    }])
    safe_to_csv(decision, OUT_DECISION)
    print(decision.to_string(index=False))
    raise SystemExit(1)


clip_obj = read_csv(CLIP_OBJECT_CSV)
frame_obj = read_csv(FRAME_OBJECT_CSV)
clip_summary = read_csv(CLIP_SUMMARY)
v45_decision = read_csv(V45_DECISION)
v45_qa = read_csv(V45_QA)

# 1. Global count checks
expected = {
    "processed_clip_count": 72,
    "clip_object_rows": 429,
    "clip_objects_with_behaviour": 374,
    "clip_objects_training_ready": 374,
    "clip_objects_missing_behaviour": 55,
}

actual = {
    "processed_clip_count": int(clip_summary["scan_frame_id"].nunique()),
    "clip_object_rows": int(len(clip_obj)),
    "clip_objects_with_behaviour": int(bool_series(clip_obj["has_behaviour_label"]).sum()),
    "clip_objects_training_ready": int(bool_series(clip_obj["is_training_ready"]).sum()),
    "clip_objects_missing_behaviour": int((~bool_series(clip_obj["has_behaviour_label"])).sum()),
}

consistency_rows = []
for metric, exp in expected.items():
    got = actual.get(metric)
    passed = got == exp
    consistency_rows.append({
        "check_name": "global_expected_count",
        "metric": metric,
        "expected": exp,
        "actual": got,
        "passed": bool(passed),
        "severity_if_failed": "hard",
    })
    if not passed:
        add_issue(
            issues,
            "global_expected_count",
            metric,
            "hard_global_count_mismatch",
            f"Expected {exp}, got {got}.",
            "hard",
        )

# Frame object count expected from clip summary
expected_frame_object_rows = int((clip_summary["generated_frame_count"] * clip_summary["object_count"]).sum())
actual_frame_object_rows = len(frame_obj)

consistency_rows.append({
    "check_name": "frame_object_total",
    "metric": "frame_object_rows",
    "expected": expected_frame_object_rows,
    "actual": actual_frame_object_rows,
    "passed": bool(expected_frame_object_rows == actual_frame_object_rows),
    "severity_if_failed": "hard",
})

if expected_frame_object_rows != actual_frame_object_rows:
    add_issue(
        issues,
        "frame_object_total",
        "frame_object_rows",
        "hard_frame_object_total_mismatch",
        f"Expected {expected_frame_object_rows}, got {actual_frame_object_rows}.",
        "hard",
    )

# 2. Per-clip consistency
clip_count_rows = []
frame_group = frame_obj.groupby("scan_frame_id").size().to_dict()
clip_obj_group = clip_obj.groupby("scan_frame_id").size().to_dict()

for _, r in clip_summary.iterrows():
    scan = clean_str(r["scan_frame_id"])
    frame_count = int(r["generated_frame_count"])
    object_count = int(r["object_count"])
    expected_rows = frame_count * object_count
    actual_rows = int(frame_group.get(scan, 0))
    actual_clip_objects = int(clip_obj_group.get(scan, 0))

    row = {
        "scan_frame_id": scan,
        "generated_frame_count": frame_count,
        "object_count": object_count,
        "expected_frame_object_rows": expected_rows,
        "actual_frame_object_rows": actual_rows,
        "actual_clip_object_rows": actual_clip_objects,
        "frame_object_count_passed": bool(expected_rows == actual_rows),
        "clip_object_count_passed": bool(object_count == actual_clip_objects),
        "split": clean_str(r.get("split", "")),
        "clip_exists": bool(r.get("clip_exists")),
        "video_open_ok": bool(r.get("video_open_ok")),
    }
    clip_count_rows.append(row)

    if expected_rows != actual_rows:
        add_issue(
            issues,
            "per_clip_frame_object_count",
            scan,
            "hard_clip_frame_object_count_mismatch",
            f"Expected {expected_rows}, got {actual_rows}.",
            "hard",
        )

    if object_count != actual_clip_objects:
        add_issue(
            issues,
            "per_clip_object_count",
            scan,
            "hard_clip_object_count_mismatch",
            f"Expected {object_count}, got {actual_clip_objects}.",
            "hard",
        )

clip_count_df = pd.DataFrame(clip_count_rows)

# 3. Per-object propagation consistency
object_key_cols = ["scan_frame_id", "final_box_id"]
object_counts = frame_obj.groupby(object_key_cols).size().reset_index(name="actual_frame_rows")
clip_expected = clip_obj[["scan_frame_id", "final_box_id", "generated_frame_count", "behaviour_code", "visual_marker_colour", "is_training_ready", "has_behaviour_label"]].copy()

object_consistency = clip_expected.merge(
    object_counts,
    on=object_key_cols,
    how="left",
)

object_consistency["actual_frame_rows"] = object_consistency["actual_frame_rows"].fillna(0).astype(int)
object_consistency["expected_frame_rows"] = object_consistency["generated_frame_count"].astype(int)
object_consistency["propagation_count_passed"] = object_consistency["actual_frame_rows"] == object_consistency["expected_frame_rows"]

for _, r in object_consistency[~object_consistency["propagation_count_passed"]].iterrows():
    add_issue(
        issues,
        "per_object_propagation_count",
        f"{r['scan_frame_id']}::{r['final_box_id']}",
        "hard_object_propagation_count_mismatch",
        f"Expected {r['expected_frame_rows']}, got {r['actual_frame_rows']}.",
        "hard",
    )

# 4. Label stability across frames
stability_rows = []
for (scan, box), g in frame_obj.groupby(["scan_frame_id", "final_box_id"]):
    behaviour_unique = sorted([x for x in g["behaviour_code"].dropna().astype(str).unique().tolist() if x and x != "nan"])
    colour_unique = sorted([x for x in g["visual_marker_colour"].dropna().astype(str).unique().tolist() if x and x != "nan"])
    pig_unique = sorted([x for x in g["behaviour_pig_id"].dropna().astype(str).unique().tolist() if x and x != "nan"])
    bbox_unique_count = g[["bbox_x1", "bbox_y1", "bbox_x2", "bbox_y2"]].drop_duplicates().shape[0]

    stable = len(behaviour_unique) <= 1 and len(colour_unique) <= 1 and len(pig_unique) <= 1

    stability_rows.append({
        "scan_frame_id": scan,
        "final_box_id": box,
        "frame_rows": len(g),
        "behaviour_unique_count": len(behaviour_unique),
        "behaviour_values": "|".join(behaviour_unique),
        "colour_unique_count": len(colour_unique),
        "colour_values": "|".join(colour_unique),
        "behaviour_pig_id_unique_count": len(pig_unique),
        "behaviour_pig_id_values": "|".join(pig_unique),
        "bbox_unique_count": bbox_unique_count,
        "label_stability_passed": bool(stable),
        "bbox_repeated_as_expected": bool(bbox_unique_count == 1),
    })

    if not stable:
        add_issue(
            issues,
            "label_stability",
            f"{scan}::{box}",
            "hard_label_not_stable_across_propagated_frames",
            f"behaviour={behaviour_unique}; colour={colour_unique}; pig_id={pig_unique}",
            "hard",
        )

label_stability_df = pd.DataFrame(stability_rows)

# 5. Timestamp consistency
timestamp_rows = []
for scan, g in frame_obj.groupby("scan_frame_id"):
    gf = g[["frame_index_in_clip", "timestamp_sec"]].drop_duplicates().sort_values("frame_index_in_clip")
    frame_idx_min = int(gf["frame_index_in_clip"].min()) if len(gf) else None
    frame_idx_max = int(gf["frame_index_in_clip"].max()) if len(gf) else None
    unique_frame_count = int(gf["frame_index_in_clip"].nunique())
    timestamp_monotonic = bool(gf["timestamp_sec"].is_monotonic_increasing)
    duplicate_frame_timestamps = int(gf.duplicated(subset=["frame_index_in_clip"]).sum())

    timestamp_rows.append({
        "scan_frame_id": scan,
        "unique_frame_count": unique_frame_count,
        "frame_index_min": frame_idx_min,
        "frame_index_max": frame_idx_max,
        "timestamp_monotonic_increasing": timestamp_monotonic,
        "duplicate_frame_timestamp_rows": duplicate_frame_timestamps,
    })

    if not timestamp_monotonic:
        add_issue(
            issues,
            "timestamp_consistency",
            scan,
            "hard_non_monotonic_timestamps",
            "timestamp_sec is not monotonic increasing.",
            "hard",
        )

timestamp_df = pd.DataFrame(timestamp_rows)

# 6. BBox consistency
bbox_df = clip_obj.copy()
for col in ["bbox_x1", "bbox_y1", "bbox_x2", "bbox_y2"]:
    bbox_df[col] = pd.to_numeric(bbox_df[col], errors="coerce")

bbox_df["bbox_width"] = bbox_df["bbox_x2"] - bbox_df["bbox_x1"]
bbox_df["bbox_height"] = bbox_df["bbox_y2"] - bbox_df["bbox_y1"]
bbox_df["bbox_valid"] = (
    bbox_df["bbox_x1"].notna()
    & bbox_df["bbox_y1"].notna()
    & bbox_df["bbox_x2"].notna()
    & bbox_df["bbox_y2"].notna()
    & (bbox_df["bbox_width"] > 0)
    & (bbox_df["bbox_height"] > 0)
)

bbox_qa = bbox_df[[
    "scan_frame_id",
    "final_box_id",
    "bbox_x1",
    "bbox_y1",
    "bbox_x2",
    "bbox_y2",
    "bbox_width",
    "bbox_height",
    "bbox_valid",
    "visual_marker_colour",
    "behaviour_code",
]]

for _, r in bbox_qa[~bbox_qa["bbox_valid"]].iterrows():
    add_issue(
        issues,
        "bbox_validity",
        f"{r['scan_frame_id']}::{r['final_box_id']}",
        "hard_invalid_bbox",
        f"bbox=({r['bbox_x1']},{r['bbox_y1']},{r['bbox_x2']},{r['bbox_y2']})",
        "hard",
    )

# 7. JSONL consistency
jsonl_line_count = 0
jsonl_bad_lines = 0
jsonl_object_rows = 0
jsonl_scan_ids = set()

with open(FRAME_JSONL, "r", encoding="utf-8") as f:
    for line in f:
        jsonl_line_count += 1
        try:
            obj = json.loads(line)
            jsonl_scan_ids.add(str(obj.get("scan_frame_id")))
            jsonl_object_rows += len(obj.get("objects", []))
        except Exception:
            jsonl_bad_lines += 1

jsonl_expected_lines = int(clip_summary["generated_frame_count"].sum())
jsonl_expected_objects = len(frame_obj)

jsonl_qa = pd.DataFrame([{
    "jsonl_line_count": jsonl_line_count,
    "expected_frame_line_count": jsonl_expected_lines,
    "jsonl_line_count_passed": bool(jsonl_line_count == jsonl_expected_lines),
    "jsonl_bad_lines": jsonl_bad_lines,
    "jsonl_object_rows": jsonl_object_rows,
    "expected_object_rows": jsonl_expected_objects,
    "jsonl_object_rows_passed": bool(jsonl_object_rows == jsonl_expected_objects),
    "jsonl_unique_scan_ids": len(jsonl_scan_ids),
    "expected_unique_scan_ids": int(clip_summary["scan_frame_id"].nunique()),
}])

if jsonl_line_count != jsonl_expected_lines:
    add_issue(
        issues,
        "jsonl_consistency",
        "jsonl_line_count",
        "hard_jsonl_line_count_mismatch",
        f"Expected {jsonl_expected_lines}, got {jsonl_line_count}.",
        "hard",
    )

if jsonl_bad_lines > 0:
    add_issue(
        issues,
        "jsonl_consistency",
        "jsonl_bad_lines",
        "hard_jsonl_parse_errors",
        f"{jsonl_bad_lines} JSONL lines could not be parsed.",
        "hard",
    )

if jsonl_object_rows != jsonl_expected_objects:
    add_issue(
        issues,
        "jsonl_consistency",
        "jsonl_object_rows",
        "hard_jsonl_object_count_mismatch",
        f"Expected {jsonl_expected_objects}, got {jsonl_object_rows}.",
        "hard",
    )

# 8. Expected info conditions preserved
missing_objects = int((~bool_series(clip_obj["has_behaviour_label"])).sum())
unmapped_clips = int((clip_summary["split"].astype(str) == "unmapped").sum())

if missing_objects == 55:
    add_issue(
        issues,
        "expected_preserved_cases",
        "missing_behaviour_objects",
        "info_expected_missing_behaviour_preserved",
        "55 missing/not-visible/unknown clip-level objects are preserved for validation.",
        "info",
    )
else:
    add_issue(
        issues,
        "expected_preserved_cases",
        "missing_behaviour_objects",
        "warning_unexpected_missing_behaviour_count",
        f"Expected 55, got {missing_objects}.",
        "warning",
    )

if unmapped_clips == 2:
    add_issue(
        issues,
        "expected_preserved_cases",
        "unmapped_clips",
        "info_expected_unmapped_clips_preserved",
        "2 unmapped clips are preserved for validation.",
        "info",
    )
else:
    add_issue(
        issues,
        "expected_preserved_cases",
        "unmapped_clips",
        "warning_unexpected_unmapped_clip_count",
        f"Expected 2, got {unmapped_clips}.",
        "warning",
    )

issues_df = pd.DataFrame(issues, columns=["check_name", "item", "issue_type", "issue_detail", "severity"])

hard_issues = issues_df[issues_df["severity"] == "hard"] if len(issues_df) else pd.DataFrame()
warnings = issues_df[issues_df["severity"] == "warning"] if len(issues_df) else pd.DataFrame()
infos = issues_df[issues_df["severity"] == "info"] if len(issues_df) else pd.DataFrame()

ready_for_v47 = len(hard_issues) == 0

# Summary checks
global_passed = all(r["passed"] for r in consistency_rows)
clip_counts_passed = bool(clip_count_df["frame_object_count_passed"].all() and clip_count_df["clip_object_count_passed"].all())
object_counts_passed = bool(object_consistency["propagation_count_passed"].all())
label_stability_passed = bool(label_stability_df["label_stability_passed"].all())
bbox_passed = bool(bbox_qa["bbox_valid"].all())
timestamp_passed = bool(timestamp_df["timestamp_monotonic_increasing"].all())
jsonl_passed = bool(jsonl_qa.iloc[0]["jsonl_line_count_passed"] and jsonl_qa.iloc[0]["jsonl_object_rows_passed"] and jsonl_qa.iloc[0]["jsonl_bad_lines"] == 0)

decision = pd.DataFrame([{
    "v46_decision": "propagation_qa_passed" if ready_for_v47 else "propagation_qa_blocked",
    "global_counts_passed": bool(global_passed),
    "clip_count_consistency_passed": bool(clip_counts_passed),
    "object_frame_count_consistency_passed": bool(object_counts_passed),
    "label_stability_passed": bool(label_stability_passed),
    "bbox_validity_passed": bool(bbox_passed),
    "timestamp_consistency_passed": bool(timestamp_passed),
    "jsonl_consistency_passed": bool(jsonl_passed),
    "clip_count": int(clip_summary["scan_frame_id"].nunique()),
    "clip_object_rows": int(len(clip_obj)),
    "frame_object_rows": int(len(frame_obj)),
    "jsonl_line_count": int(jsonl_line_count),
    "missing_behaviour_objects_preserved": int(missing_objects),
    "unmapped_clips_preserved": int(unmapped_clips),
    "hard_issue_count": int(len(hard_issues)),
    "warning_count": int(len(warnings)),
    "info_count": int(len(infos)),
    "issue_count": int(len(issues_df)),
    "ready_for_v47_visualization_interface": bool(ready_for_v47),
    "generated_at": datetime.now().isoformat(timespec="seconds"),
}])

safe_to_csv(pd.DataFrame(consistency_rows), OUT_CONSISTENCY)
safe_to_csv(object_consistency, OUT_OBJECT_COUNTS)
safe_to_csv(clip_count_df, OUT_CLIP_COUNTS)
safe_to_csv(label_stability_df, OUT_LABEL_STABILITY)
safe_to_csv(timestamp_df, OUT_TIMESTAMP_QA)
safe_to_csv(bbox_qa, OUT_BBOX_QA)
safe_to_csv(jsonl_qa, OUT_JSONL_QA)
safe_to_csv(issues_df, OUT_VALIDATION_ISSUES)
safe_to_csv(decision, OUT_DECISION)

report = f"""# Week 8 v46 Propagation QA and Consistency Report

## Decision

- v46 decision: {decision.iloc[0]["v46_decision"]}
- Ready for v47 visualization interface: {ready_for_v47}
- Hard issue count: {len(hard_issues)}
- Warning count: {len(warnings)}
- Info count: {len(infos)}

## QA results

| Check | Passed |
|---|---:|
| Global counts | {global_passed} |
| Per-clip frame/object counts | {clip_counts_passed} |
| Per-object propagation counts | {object_counts_passed} |
| Label stability across propagated frames | {label_stability_passed} |
| BBox validity | {bbox_passed} |
| Timestamp consistency | {timestamp_passed} |
| JSONL consistency | {jsonl_passed} |

## Main counts

- Clip count: {int(clip_summary["scan_frame_id"].nunique())}
- Clip-object rows: {len(clip_obj)}
- Frame-object rows: {len(frame_obj)}
- JSONL frame lines: {jsonl_line_count}
- Missing behaviour objects preserved: {missing_objects}
- Unmapped clips preserved: {unmapped_clips}

## Interpretation

The propagated ground-truth dataset is internally consistent if all hard checks pass. Missing behaviour objects and unmapped clips are expected preserved cases, not failures.

## Scope note

Bounding boxes in v45/v46 are scanpoint-anchor boxes repeated across the interval. This is appropriate for validating label propagation and building the visual interface. Tracking-refined per-frame boxes remain a later refinement step.

## Outputs

- Consistency checks: {OUT_CONSISTENCY}
- Object/frame count consistency: {OUT_OBJECT_COUNTS}
- Clip count consistency: {OUT_CLIP_COUNTS}
- Label stability: {OUT_LABEL_STABILITY}
- Timestamp QA: {OUT_TIMESTAMP_QA}
- BBox QA: {OUT_BBOX_QA}
- JSONL QA: {OUT_JSONL_QA}
- Validation issues: {OUT_VALIDATION_ISSUES}
"""

OUT_REPORT.write_text(report)

OUT_NOTE.write_text(
    "# Week 8 v46 Propagation QA\n\n"
    "## Summary\n\n"
    f"- v46 decision: {decision.iloc[0]['v46_decision']}\n"
    f"- Global counts passed: {global_passed}\n"
    f"- Clip count consistency passed: {clip_counts_passed}\n"
    f"- Object-frame count consistency passed: {object_counts_passed}\n"
    f"- Label stability passed: {label_stability_passed}\n"
    f"- BBox validity passed: {bbox_passed}\n"
    f"- Timestamp consistency passed: {timestamp_passed}\n"
    f"- JSONL consistency passed: {jsonl_passed}\n"
    f"- Hard issue count: {len(hard_issues)}\n"
    f"- Warning count: {len(warnings)}\n"
    f"- Ready for v47 visualization interface: {ready_for_v47}\n\n"
    "## Expected preserved cases\n\n"
    f"- Missing behaviour objects preserved: {missing_objects}\n"
    f"- Unmapped clips preserved: {unmapped_clips}\n\n"
    "## Outputs\n\n"
    f"- Decision: {OUT_DECISION}\n"
    f"- Validation issues: {OUT_VALIDATION_ISSUES}\n"
    f"- Report: {OUT_REPORT}\n"
)

progress_row = pd.DataFrame([{
    "date": datetime.now().date().isoformat(),
    "stage": "v46",
    "task_name": "Propagation QA and consistency validation",
    "status": "PASS" if ready_for_v47 else "BLOCKED",
    "input_summary": str(V45),
    "output_summary": str(OUT),
    "hard_issues": int(len(hard_issues)),
    "warnings": int(len(warnings)),
    "next_action": "v47 behaviour dataset visualization interface MVP" if ready_for_v47 else "Resolve propagation QA hard issues.",
}])

if OUT_PROGRESS.exists():
    old = pd.read_csv(OUT_PROGRESS)
    progress = pd.concat([old, progress_row], ignore_index=True)
else:
    progress = progress_row

safe_to_csv(progress, OUT_PROGRESS)

print("Saved:")
print(OUT_CONSISTENCY)
print(OUT_OBJECT_COUNTS)
print(OUT_CLIP_COUNTS)
print(OUT_LABEL_STABILITY)
print(OUT_TIMESTAMP_QA)
print(OUT_BBOX_QA)
print(OUT_JSONL_QA)
print(OUT_VALIDATION_ISSUES)
print(OUT_DECISION)
print(OUT_REPORT)
print(OUT_NOTE)

print()
print("=== v46 decision ===")
print(decision.to_string(index=False))

print()
print("=== v46 consistency checks ===")
print(pd.DataFrame(consistency_rows).to_string(index=False))

print()
print("=== v46 issues ===")
if len(issues_df):
    print(issues_df.to_string(index=False))
else:
    print("No issues found.")
