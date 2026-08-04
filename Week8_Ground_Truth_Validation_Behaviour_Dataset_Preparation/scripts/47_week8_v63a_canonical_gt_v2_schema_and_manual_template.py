from pathlib import Path
from datetime import datetime
import csv
import json
import re
import pandas as pd


ROOT = Path.home() / "PigBench"
W8 = ROOT / "Week8_Ground_Truth_Validation_Behaviour_Dataset_Preparation"

V62B2_OBS = W8 / "outputs" / "v62b2_corrected_canonical_excel_observation_extraction" / "week8_v62b2_corrected_canonical_colour_behaviour_observations.csv"
V45_FRAME_OBJECTS = W8 / "outputs" / "propagated_ground_truth" / "v45_label_propagation" / "week8_v45_frame_object_propagated_annotations.csv"
V45_CLIP_OBJECTS = W8 / "outputs" / "propagated_ground_truth" / "v45_label_propagation" / "week8_v45_clip_object_propagated_annotations.csv"
MANUAL_NOTES = W8 / "validation" / "week8_v52d_visual_validation_notes.csv"

OUT = W8 / "outputs" / "v63a_canonical_gt_v2_schema"
DOCS = W8 / "docs"
NOTES = W8 / "notes"
REPORTS = W8 / "reports"
PROGRESS = W8 / "progress"
VALIDATION = W8 / "validation"

for p in [OUT, DOCS, NOTES, REPORTS, PROGRESS, VALIDATION]:
    p.mkdir(parents=True, exist_ok=True)

OUT_TARGETS = OUT / "week8_v63a_canonical_gt_v2_targets.csv"
OUT_CANDIDATE_BOXES = OUT / "week8_v63a_existing_anchor_candidate_boxes.csv"
OUT_SCANFRAME_STATUS = OUT / "week8_v63a_scanframe_gt_v2_initial_status.csv"
OUT_UI_JSON = OUT / "week8_v63a_visualizer_data.json"
OUT_DECISION = OUT / "week8_v63a_decision_summary.csv"
OUT_ISSUES = OUT / "week8_v63a_issues.csv"

MANUAL_TEMPLATE = VALIDATION / "week8_v63a_manual_gt_v2_assignment_template.csv"

OUT_SCHEMA_DOC = DOCS / "week8_v63a_canonical_gt_v2_schema.md"
OUT_NOTE = NOTES / "week8_v63a_canonical_gt_v2_schema_notes.md"
OUT_REPORT = REPORTS / "week8_v63a_canonical_gt_v2_schema_report.md"
OUT_PROGRESS = PROGRESS / "week8_experiment_progress_log.csv"


EXPECTED_COLOURS = ["blue", "green", "no_colour", "purple", "red_neck", "red_tail"]


def safe_to_csv(df, path):
    df.to_csv(path, index=False, quoting=csv.QUOTE_ALL, escapechar="\\", lineterminator="\n")


def clean_str(x):
    if pd.isna(x):
        return ""
    s = str(x).strip()
    if s.lower() in ["nan", "none", "null"]:
        return ""
    return s


def scan_num(s):
    m = re.search(r"(\d+)$", clean_str(s))
    return int(m.group(1)) if m else 10**9


def norm_bool(x):
    s = clean_str(x).lower()
    return s in ["true", "1", "yes", "y"]


def severity_rank(sev):
    sev = clean_str(sev).lower()
    if sev == "critical":
        return 4
    if sev == "major":
        return 3
    if sev == "minor":
        return 2
    if sev == "info":
        return 1
    return 0


def aggregate_notes(notes):
    if notes is None or len(notes) == 0:
        return pd.DataFrame(columns=[
            "scan_frame_id",
            "manual_note_count",
            "manual_issue_types",
            "manual_max_severity",
            "manual_note_summary",
            "has_manual_gt_anchor_problem",
            "has_manual_tracking_problem",
            "has_manual_major_or_critical",
        ])

    n = notes.copy()
    for c in ["scan_frame_id", "issue_type", "severity", "note"]:
        if c in n.columns:
            n[c] = n[c].map(clean_str)

    rows = []
    for scan, g in n.groupby("scan_frame_id"):
        issue_types = sorted(set(g["issue_type"].tolist()))
        severities = sorted(set(g["severity"].tolist()))
        max_sev = ""
        if len(g):
            max_sev = max(g["severity"].tolist(), key=severity_rank)

        note_text = " | ".join([clean_str(x) for x in g["note"].tolist() if clean_str(x)])
        note_low = note_text.lower()

        has_gt_anchor_problem = (
            "wrong gt anchor" in note_low
            or "missing a ground truth anchor" in note_low
            or "missing ground truth anchor" in note_low
        )

        has_tracking_problem = any(x in issue_types for x in [
            "missing_pig",
            "bbox_wrong",
            "identity_switch",
            "identity_uncertain",
            "false_positive",
        ])

        has_major_or_critical = any(s in ["major", "critical"] for s in severities)

        rows.append({
            "scan_frame_id": scan,
            "manual_note_count": int(len(g)),
            "manual_issue_types": ";".join(issue_types),
            "manual_max_severity": max_sev,
            "manual_note_summary": note_text,
            "has_manual_gt_anchor_problem": bool(has_gt_anchor_problem),
            "has_manual_tracking_problem": bool(has_tracking_problem),
            "has_manual_major_or_critical": bool(has_major_or_critical),
        })

    return pd.DataFrame(rows)


issues = []

for p in [V62B2_OBS, V45_CLIP_OBJECTS]:
    if not p.exists():
        issues.append({
            "item": str(p),
            "issue_type": "hard_missing_required_input",
            "issue_detail": "Required v63a input is missing.",
            "severity": "hard",
        })

if issues:
    issues_df = pd.DataFrame(issues)
    safe_to_csv(issues_df, OUT_ISSUES)
    decision = pd.DataFrame([{
        "v63a_decision": "canonical_gt_v2_schema_blocked",
        "hard_issue_count": int((issues_df["severity"] == "hard").sum()),
        "ready_for_v63b_manual_gt_v2_visualizer": False,
        "generated_at": datetime.now().isoformat(timespec="seconds"),
    }])
    safe_to_csv(decision, OUT_DECISION)
    print(decision.to_string(index=False))
    raise SystemExit(1)


obs = pd.read_csv(V62B2_OBS)
clip_objects = pd.read_csv(V45_CLIP_OBJECTS)

if MANUAL_NOTES.exists():
    manual_notes = pd.read_csv(MANUAL_NOTES)
else:
    manual_notes = pd.DataFrame()

note_summary = aggregate_notes(manual_notes)

# Existing anchor candidate boxes.
# v45 clip object file already has one row per object if available; if it is duplicated, group defensively.
candidate_cols = [
    "scan_frame_id",
    "video_id",
    "clip_path",
    "source_video_path",
    "start_sec",
    "end_sec",
    "duration_sec",
    "fps_used",
    "generated_frame_count",
    "final_box_id",
    "bbox_x1",
    "bbox_y1",
    "bbox_x2",
    "bbox_y2",
    "bbox_source",
    "visual_marker_colour",
    "identity_status",
    "validation_status",
    "validation_flags",
]

existing_cols = [c for c in candidate_cols if c in clip_objects.columns]
boxes = clip_objects[existing_cols].copy()

for c in boxes.columns:
    boxes[c] = boxes[c].map(clean_str) if boxes[c].dtype == object else boxes[c]

# Deduplicate at anchor-object level.
key_cols = ["scan_frame_id", "final_box_id"]
boxes = boxes.drop_duplicates(key_cols).copy()

boxes = boxes.rename(columns={
    "visual_marker_colour": "deprecated_previous_visual_marker_colour",
    "identity_status": "deprecated_previous_identity_status",
    "validation_status": "previous_validation_status",
    "validation_flags": "previous_validation_flags",
})

boxes["candidate_box_id"] = boxes["final_box_id"].astype(str)
boxes["candidate_box_source"] = "existing_v45_anchor_candidate_not_final_gt"
boxes["canonical_status"] = "candidate_only_requires_manual_assignment"

bbox_cols = ["bbox_x1", "bbox_y1", "bbox_x2", "bbox_y2"]
for c in bbox_cols:
    if c in boxes.columns:
        boxes[c] = pd.to_numeric(boxes[c], errors="coerce")

if all(c in boxes.columns for c in bbox_cols):
    boxes["bbox_width"] = boxes["bbox_x2"] - boxes["bbox_x1"]
    boxes["bbox_height"] = boxes["bbox_y2"] - boxes["bbox_y1"]
    boxes["bbox_area"] = boxes["bbox_width"] * boxes["bbox_height"]

safe_to_csv(boxes, OUT_CANDIDATE_BOXES)

# Candidate box compact summary per scanframe.
box_summary_rows = []
for scan, g in boxes.groupby("scan_frame_id"):
    candidate_ids = [clean_str(x) for x in g["candidate_box_id"].tolist()]
    prev_colours = []
    if "deprecated_previous_visual_marker_colour" in g.columns:
        prev_colours = [clean_str(x) for x in g["deprecated_previous_visual_marker_colour"].tolist()]

    box_summary_rows.append({
        "scan_frame_id": scan,
        "existing_candidate_box_count": int(len(g)),
        "existing_candidate_box_ids": ";".join(candidate_ids),
        "deprecated_previous_visual_marker_colours": ";".join(prev_colours),
    })

box_summary = pd.DataFrame(box_summary_rows)

# Canonical targets.
obs = obs.copy()
obs["scan_order"] = obs["scan_frame_id"].map(scan_num)

for c in ["scan_frame_id", "video_id", "canonical_colour_label_norm", "canonical_colour_label_raw", "behaviour_code"]:
    if c in obs.columns:
        obs[c] = obs[c].map(clean_str)

targets = obs.copy()
targets["canonical_gt_object_id"] = (
    targets["scan_frame_id"].astype(str)
    + "__"
    + targets["canonical_colour_label_norm"].astype(str)
)

targets = targets.merge(box_summary, on="scan_frame_id", how="left")
targets = targets.merge(note_summary, on="scan_frame_id", how="left")

fill_cols = [
    "existing_candidate_box_count",
    "existing_candidate_box_ids",
    "deprecated_previous_visual_marker_colours",
    "manual_note_count",
    "manual_issue_types",
    "manual_max_severity",
    "manual_note_summary",
]
for c in fill_cols:
    if c in targets.columns:
        targets[c] = targets[c].fillna("")

for c in ["has_manual_gt_anchor_problem", "has_manual_tracking_problem", "has_manual_major_or_critical"]:
    if c in targets.columns:
        targets[c] = targets[c].fillna(False).astype(bool)

targets["canonical_source_of_truth"] = "original_excel_v62b2"
targets["bbox_source_of_truth"] = "manual_gt_v2_pending"
targets["identity_assignment_source_of_truth"] = "manual_gt_v2_pending"
targets["classification_use_initial"] = "pending_manual_gt_v2_validation"

# Manual assignment fields.
manual_fields = {
    "manual_assigned_candidate_box_id": "",
    "manual_bbox_x1": "",
    "manual_bbox_y1": "",
    "manual_bbox_x2": "",
    "manual_bbox_y2": "",
    "manual_bbox_status": "",
    "manual_identity_status": "",
    "manual_colour_status": "",
    "manual_behaviour_status": "",
    "manual_gt_v2_status": "",
    "manual_classification_use": "",
    "manual_review_priority": "",
    "manual_reviewer_note": "",
    "manual_reviewed_by": "",
    "manual_reviewed_at": "",
}

for k, v in manual_fields.items():
    targets[k] = v

# Initial review priority from previous manual notes and box count.
def initial_priority(row):
    box_count = pd.to_numeric(row.get("existing_candidate_box_count", ""), errors="coerce")
    if bool(row.get("has_manual_gt_anchor_problem", False)):
        return "P0_gt_anchor_problem"
    if bool(row.get("has_manual_major_or_critical", False)):
        return "P1_major_or_critical_manual_issue"
    if pd.notna(box_count) and int(box_count) != 6:
        return "P1_candidate_box_count_not_six"
    if bool(row.get("has_manual_tracking_problem", False)):
        return "P2_tracking_or_identity_issue"
    return "P3_standard_review"

targets["initial_review_priority"] = targets.apply(initial_priority, axis=1)

target_cols = [
    "canonical_gt_object_id",
    "scan_frame_id",
    "video_id",
    "observation_offset_min",
    "canonical_colour_label_norm",
    "canonical_colour_label_raw",
    "behaviour_code",
    "behaviour_raw",
    "canonical_source_of_truth",
    "canonical_excel_path",
    "canonical_sheet",
    "excel_behaviour_row_0based",
    "excel_behaviour_col_0based",
    "excel_cell_context",
    "existing_candidate_box_count",
    "existing_candidate_box_ids",
    "deprecated_previous_visual_marker_colours",
    "manual_note_count",
    "manual_issue_types",
    "manual_max_severity",
    "manual_note_summary",
    "has_manual_gt_anchor_problem",
    "has_manual_tracking_problem",
    "has_manual_major_or_critical",
    "initial_review_priority",
    "bbox_source_of_truth",
    "identity_assignment_source_of_truth",
    "classification_use_initial",
    "manual_assigned_candidate_box_id",
    "manual_bbox_x1",
    "manual_bbox_y1",
    "manual_bbox_x2",
    "manual_bbox_y2",
    "manual_bbox_status",
    "manual_identity_status",
    "manual_colour_status",
    "manual_behaviour_status",
    "manual_gt_v2_status",
    "manual_classification_use",
    "manual_review_priority",
    "manual_reviewer_note",
    "manual_reviewed_by",
    "manual_reviewed_at",
]

target_cols = [c for c in target_cols if c in targets.columns]
targets = targets[target_cols].sort_values(
    ["scan_frame_id", "canonical_colour_label_norm"],
    key=lambda s: s.map(scan_num) if s.name == "scan_frame_id" else s
)

safe_to_csv(targets, OUT_TARGETS)
safe_to_csv(targets, MANUAL_TEMPLATE)

# Scanframe status table.
status_rows = []
for scan, g in targets.groupby("scan_frame_id"):
    colours = sorted(g["canonical_colour_label_norm"].astype(str).unique())
    candidate_count = clean_str(g["existing_candidate_box_count"].iloc[0])
    try:
        candidate_count_int = int(float(candidate_count))
    except Exception:
        candidate_count_int = 0

    note_count = clean_str(g["manual_note_count"].iloc[0])
    issue_types = clean_str(g["manual_issue_types"].iloc[0])
    max_sev = clean_str(g["manual_max_severity"].iloc[0])
    has_gt_anchor = bool(g["has_manual_gt_anchor_problem"].iloc[0])
    has_major = bool(g["has_manual_major_or_critical"].iloc[0])

    if has_gt_anchor:
        scan_status = "P0_gt_anchor_problem"
        classification_gate = "exclude_until_gt_anchor_corrected"
    elif has_major:
        scan_status = "P1_manual_major_or_critical_issue"
        classification_gate = "exclude_until_reviewed_and_corrected"
    elif candidate_count_int != 6:
        scan_status = "P1_candidate_box_count_not_six"
        classification_gate = "exclude_until_bbox_count_corrected"
    elif issue_types and issue_types != "bbox_ok":
        scan_status = "P2_review_required"
        classification_gate = "pending_manual_review"
    else:
        scan_status = "P3_standard_review"
        classification_gate = "pending_manual_review"

    status_rows.append({
        "scan_frame_id": scan,
        "video_id": g["video_id"].iloc[0],
        "canonical_expected_object_count": int(len(g)),
        "canonical_colour_count": int(len(colours)),
        "canonical_colours": ";".join(colours),
        "existing_candidate_box_count": candidate_count_int,
        "box_count_delta_existing_minus_expected": int(candidate_count_int - len(g)),
        "manual_note_count": note_count,
        "manual_issue_types": issue_types,
        "manual_max_severity": max_sev,
        "has_manual_gt_anchor_problem": has_gt_anchor,
        "has_manual_major_or_critical": has_major,
        "initial_scanframe_gt_v2_status": scan_status,
        "initial_classification_gate": classification_gate,
    })

scan_status = pd.DataFrame(status_rows).sort_values("scan_frame_id", key=lambda s: s.map(scan_num))
safe_to_csv(scan_status, OUT_SCANFRAME_STATUS)

# UI JSON for next visualizer.
ui = {
    "dataset_version": "v63a_canonical_gt_v2_schema",
    "created_at": datetime.now().isoformat(timespec="seconds"),
    "canonical_colour_labels": EXPECTED_COLOURS,
    "source_files": {
        "canonical_excel_observations": str(V62B2_OBS),
        "candidate_anchor_boxes": str(V45_CLIP_OBJECTS),
        "manual_notes": str(MANUAL_NOTES),
    },
    "scanframes": [],
}

clip_meta_cols = ["scan_frame_id", "video_id", "clip_path", "source_video_path", "start_sec", "end_sec", "duration_sec", "fps_used", "generated_frame_count"]
clip_meta = clip_objects[[c for c in clip_meta_cols if c in clip_objects.columns]].drop_duplicates("scan_frame_id")

for _, srow in scan_status.iterrows():
    scan = srow["scan_frame_id"]
    meta = clip_meta[clip_meta["scan_frame_id"].astype(str) == scan]
    meta_dict = meta.iloc[0].to_dict() if len(meta) else {}

    t_rows = targets[targets["scan_frame_id"] == scan].to_dict(orient="records")
    b_rows = boxes[boxes["scan_frame_id"].astype(str) == scan].to_dict(orient="records")

    ui["scanframes"].append({
        "scan_frame_id": scan,
        "video_id": srow["video_id"],
        "status": srow["initial_scanframe_gt_v2_status"],
        "classification_gate": srow["initial_classification_gate"],
        "clip_meta": {k: clean_str(v) for k, v in meta_dict.items()},
        "canonical_targets": t_rows,
        "candidate_boxes": b_rows,
    })

OUT_UI_JSON.write_text(json.dumps(ui, indent=2, ensure_ascii=False))

# QA / decision.
if len(targets) != 432:
    issues.append({
        "item": "canonical_targets",
        "issue_type": "hard_unexpected_target_row_count",
        "issue_detail": f"Expected 432 target rows, found {len(targets)}.",
        "severity": "hard",
    })

if targets["scan_frame_id"].nunique() != 72:
    issues.append({
        "item": "canonical_targets",
        "issue_type": "hard_unexpected_scanframe_count",
        "issue_detail": f"Expected 72 scanframes, found {targets['scan_frame_id'].nunique()}.",
        "severity": "hard",
    })

bad_colour_sets = scan_status[scan_status["canonical_colour_count"] != 6]
if len(bad_colour_sets):
    issues.append({
        "item": "scanframe_status",
        "issue_type": "hard_not_all_scanframes_have_six_colours",
        "issue_detail": f"{len(bad_colour_sets)} scanframes do not have six canonical colours.",
        "severity": "hard",
    })

candidate_incomplete = scan_status[scan_status["existing_candidate_box_count"] != 6]
if len(candidate_incomplete):
    issues.append({
        "item": "candidate_boxes",
        "issue_type": "warning_some_scanframes_candidate_box_count_not_six",
        "issue_detail": f"{len(candidate_incomplete)} scanframes have candidate anchor box count different from six.",
        "severity": "warning",
    })

manual_problem_scanframes = scan_status[
    scan_status["initial_scanframe_gt_v2_status"].isin([
        "P0_gt_anchor_problem",
        "P1_manual_major_or_critical_issue",
        "P1_candidate_box_count_not_six",
        "P2_review_required",
    ])
]

issues_df = pd.DataFrame(issues, columns=["item", "issue_type", "issue_detail", "severity"])
safe_to_csv(issues_df, OUT_ISSUES)

hard_issue_count = int((issues_df["severity"] == "hard").sum()) if len(issues_df) else 0
warning_count = int((issues_df["severity"] == "warning").sum()) if len(issues_df) else 0

decision = pd.DataFrame([{
    "v63a_decision": "canonical_gt_v2_schema_created" if hard_issue_count == 0 else "canonical_gt_v2_schema_blocked",
    "canonical_target_rows": int(len(targets)),
    "scanframe_count": int(targets["scan_frame_id"].nunique()),
    "candidate_anchor_box_rows": int(len(boxes)),
    "scanframes_with_candidate_box_count_not_six": int(len(candidate_incomplete)),
    "manual_problem_or_review_scanframes": int(len(manual_problem_scanframes)),
    "p0_gt_anchor_problem_scanframes": int((scan_status["initial_scanframe_gt_v2_status"] == "P0_gt_anchor_problem").sum()),
    "p1_major_or_box_count_scanframes": int(scan_status["initial_scanframe_gt_v2_status"].isin([
        "P1_manual_major_or_critical_issue",
        "P1_candidate_box_count_not_six",
    ]).sum()),
    "manual_template_path": str(MANUAL_TEMPLATE),
    "hard_issue_count": hard_issue_count,
    "warning_count": warning_count,
    "issue_count": int(len(issues_df)),
    "ready_for_v63b_manual_gt_v2_visualizer": bool(hard_issue_count == 0),
    "generated_at": datetime.now().isoformat(timespec="seconds"),
}])

safe_to_csv(decision, OUT_DECISION)

schema_doc = """# Week 8 v63a Canonical GT v2 Schema

## Purpose

This schema replaces the previous task-sheet-oriented GT assumption with a canonical manual GT v2 workflow.

## Source of truth hierarchy

1. Canonical colour and behaviour labels: original Excel annotation extracted in v62b2.
2. Candidate bounding boxes: existing v45 anchor boxes, not final GT.
3. Candidate tracking: helper layer only, not final GT.
4. Manual GT v2 assignment: final source for bbox-to-colour identity association.

## Manual GT v2 status values

Recommended values:

- gold_usable
- silver_usable_with_caution
- red_exclude
- fix_required
- unknown_pending_review

## Manual bbox status values

Recommended values:

- bbox_ok
- bbox_wrong
- bbox_missing
- bbox_extra_false_positive
- bbox_needs_manual_redraw
- bbox_uncertain

## Manual identity status values

Recommended values:

- identity_confirmed
- identity_uncertain
- identity_switch
- identity_not_visible
- identity_missing

## Classification use values

Recommended values:

- use_for_classification
- use_for_classification_with_caution
- exclude_from_classification
- pending_review
"""

OUT_SCHEMA_DOC.write_text(schema_doc)

OUT_NOTE.write_text(
    "# Week 8 v63a Canonical GT v2 Schema\n\n"
    f"- v63a decision: {decision.iloc[0]['v63a_decision']}\n"
    f"- Canonical target rows: {len(targets)}\n"
    f"- Scanframes: {targets['scan_frame_id'].nunique()}\n"
    f"- Candidate anchor boxes: {len(boxes)}\n"
    f"- Scanframes with candidate box count not six: {len(candidate_incomplete)}\n"
    f"- Manual problem/review scanframes: {len(manual_problem_scanframes)}\n"
    f"- Manual template: {MANUAL_TEMPLATE}\n"
    f"- Hard issues: {hard_issue_count}\n"
    f"- Warnings: {warning_count}\n"
    f"- Ready for v63b manual GT v2 visualizer: {bool(hard_issue_count == 0)}\n\n"
    "Important: candidate boxes are not final GT. The final identity-to-bbox assignment must be manually confirmed in v63b/v64.\n"
)

OUT_REPORT.write_text(
    "# Week 8 v63a Canonical GT v2 Schema Report\n\n"
    f"Decision: {decision.iloc[0]['v63a_decision']}\n\n"
    f"Canonical targets: `{OUT_TARGETS}`\n\n"
    f"Candidate boxes: `{OUT_CANDIDATE_BOXES}`\n\n"
    f"Scanframe status: `{OUT_SCANFRAME_STATUS}`\n\n"
    f"Manual template: `{MANUAL_TEMPLATE}`\n"
)

progress_row = pd.DataFrame([{
    "date": datetime.now().date().isoformat(),
    "stage": "v63a",
    "task_name": "Canonical GT v2 schema and manual assignment template",
    "status": "PASS" if hard_issue_count == 0 else "BLOCKED",
    "input_summary": str(V62B2_OBS),
    "output_summary": str(OUT),
    "hard_issues": hard_issue_count,
    "warnings": warning_count,
    "next_action": "v63b manual GT v2 visualizer with canonical colour/behaviour list.",
}])

if OUT_PROGRESS.exists():
    old_progress = pd.read_csv(OUT_PROGRESS)
    progress = pd.concat([old_progress, progress_row], ignore_index=True)
else:
    progress = progress_row

safe_to_csv(progress, OUT_PROGRESS)

print("Saved:")
print(OUT_TARGETS)
print(OUT_CANDIDATE_BOXES)
print(OUT_SCANFRAME_STATUS)
print(OUT_UI_JSON)
print(MANUAL_TEMPLATE)
print(OUT_DECISION)
print(OUT_ISSUES)
print(OUT_SCHEMA_DOC)
print(OUT_NOTE)

print()
print("=== v63a decision ===")
print(decision.to_string(index=False))

print()
print("=== scanframe status counts ===")
print(scan_status["initial_scanframe_gt_v2_status"].value_counts().to_string())

print()
print("=== candidate box count distribution ===")
print(scan_status["existing_candidate_box_count"].value_counts().sort_index().to_string())

print()
print("=== high priority scanframes ===")
print(scan_status[scan_status["initial_scanframe_gt_v2_status"].isin([
    "P0_gt_anchor_problem",
    "P1_manual_major_or_critical_issue",
    "P1_candidate_box_count_not_six",
])][[
    "scan_frame_id",
    "video_id",
    "existing_candidate_box_count",
    "manual_issue_types",
    "manual_max_severity",
    "initial_scanframe_gt_v2_status",
    "initial_classification_gate",
]].to_string(index=False))

print()
print("=== issues ===")
if len(issues_df):
    print(issues_df.to_string(index=False))
else:
    print("No issues found.")
