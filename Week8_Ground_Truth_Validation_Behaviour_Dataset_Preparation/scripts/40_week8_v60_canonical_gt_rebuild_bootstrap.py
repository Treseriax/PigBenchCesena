from pathlib import Path
from datetime import datetime
import csv
import pandas as pd


ROOT = Path.home() / "PigBench"
W8 = ROOT / "Week8_Ground_Truth_Validation_Behaviour_Dataset_Preparation"

MANUAL_NOTES = W8 / "validation" / "week8_v52d_visual_validation_notes.csv"

OUT = W8 / "outputs" / "v60_canonical_gt_rebuild_bootstrap"
DOCS = W8 / "docs"
REPORTS = W8 / "reports"
NOTES = W8 / "notes"
PROGRESS = W8 / "progress"

for p in [OUT, DOCS, REPORTS, NOTES, PROGRESS]:
    p.mkdir(parents=True, exist_ok=True)

OUT_DECISION = OUT / "week8_v60_decision_summary.csv"
OUT_ISSUES = OUT / "week8_v60_issues.csv"
OUT_MANUAL_NOTE_SUMMARY = OUT / "week8_v60_manual_note_summary.csv"
OUT_SCANFRAME_TRIAGE_PRELIM = OUT / "week8_v60_scanframe_preliminary_triage.csv"
OUT_DEPRECATED_OUTPUT_POLICY = OUT / "week8_v60_deprecated_output_policy.csv"

OUT_POLICY_DOC = DOCS / "week8_v60_canonical_gt_rebuild_policy.md"
OUT_REPORT = REPORTS / "week8_v60_canonical_gt_rebuild_bootstrap_report.md"
OUT_NOTE = NOTES / "week8_v60_canonical_gt_rebuild_bootstrap_notes.md"
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


issues = []

if not MANUAL_NOTES.exists():
    issues.append({
        "item": str(MANUAL_NOTES),
        "issue_type": "hard_missing_manual_notes",
        "issue_detail": "Manual validation notes CSV is missing.",
        "severity": "hard",
    })

if issues:
    issues_df = pd.DataFrame(issues)
    safe_to_csv(issues_df, OUT_ISSUES)
    decision = pd.DataFrame([{
        "v60_decision": "canonical_gt_rebuild_blocked_missing_manual_notes",
        "hard_issue_count": int((issues_df["severity"] == "hard").sum()),
        "ready_for_v61_annotation_source_inventory": False,
        "generated_at": datetime.now().isoformat(timespec="seconds"),
    }])
    safe_to_csv(decision, OUT_DECISION)
    print(decision.to_string(index=False))
    raise SystemExit(1)


notes = pd.read_csv(MANUAL_NOTES)

for col in ["scan_frame_id", "video_id", "issue_type", "severity", "note"]:
    if col in notes.columns:
        notes[col] = notes[col].astype(str).map(clean_str)

problem_issue_types = {
    "missing_pig",
    "bbox_wrong",
    "identity_switch",
    "identity_uncertain",
    "behaviour_uncertain",
    "false_positive",
    "other",
}

major_levels = {"major", "critical"}

summary_rows = []

issue_counts = notes["issue_type"].value_counts(dropna=False).reset_index()
issue_counts.columns = ["category", "count"]
issue_counts["summary_type"] = "issue_type"
summary_rows.append(issue_counts)

severity_counts = notes["severity"].value_counts(dropna=False).reset_index()
severity_counts.columns = ["category", "count"]
severity_counts["summary_type"] = "severity"
summary_rows.append(severity_counts)

note_summary = pd.concat(summary_rows, ignore_index=True)[["summary_type", "category", "count"]]
safe_to_csv(note_summary, OUT_MANUAL_NOTE_SUMMARY)

triage_rows = []

for scan, g in notes.groupby("scan_frame_id"):
    issue_types = set(g["issue_type"].tolist())
    severities = set(g["severity"].tolist())
    note_text = " | ".join([clean_str(x) for x in g["note"].tolist() if clean_str(x)])

    has_ok = "bbox_ok" in issue_types
    has_problem = bool(issue_types.intersection(problem_issue_types))
    has_major = bool(severities.intersection(major_levels))
    has_conflict = has_ok and has_problem

    if has_major or "identity_switch" in issue_types or "bbox_wrong" in issue_types:
        preliminary_status = "red_requires_gt_review"
        classification_use = "exclude_until_corrected"
    elif has_problem:
        preliminary_status = "yellow_needs_review"
        classification_use = "do_not_use_until_checked"
    elif has_ok:
        preliminary_status = "green_candidate"
        classification_use = "candidate_after_colour_gt_rebuild"
    else:
        preliminary_status = "unknown_needs_review"
        classification_use = "do_not_use_until_checked"

    if "Wrong GT Anchor" in note_text or "Missing a ground truth anchor" in note_text or "ground truth anchor" in note_text.lower():
        preliminary_status = "red_gt_anchor_problem"
        classification_use = "exclude_until_gt_anchor_corrected"

    triage_rows.append({
        "scan_frame_id": scan,
        "video_id": g["video_id"].iloc[-1] if "video_id" in g.columns else "",
        "note_count": int(len(g)),
        "issue_types": ";".join(sorted(issue_types)),
        "severities": ";".join(sorted(severities)),
        "has_ok_note": bool(has_ok),
        "has_problem_note": bool(has_problem),
        "has_major_or_critical": bool(has_major),
        "has_conflicting_notes": bool(has_conflict),
        "preliminary_status": preliminary_status,
        "classification_use": classification_use,
        "manual_note_summary": note_text,
    })

triage = pd.DataFrame(triage_rows).sort_values(["preliminary_status", "scan_frame_id"])
safe_to_csv(triage, OUT_SCANFRAME_TRIAGE_PRELIM)

deprecated_policy = pd.DataFrame([
    {
        "old_component": "v17/v18/v21 colour identity and behaviour fusion",
        "new_status": "deprecated_as_final_gt",
        "reason": "Colour labels must be rebuilt from the original annotation file, not from visual guesses or previous suggested mappings.",
        "allowed_use": "diagnostic/reference only",
    },
    {
        "old_component": "v52b3 hybrid tracking full",
        "new_status": "helper_layer_only",
        "reason": "Manual validation found missing pigs, identity switches and wrong boxes.",
        "allowed_use": "visual QA and candidate tracking support only",
    },
    {
        "old_component": "v53 optimistic validation report",
        "new_status": "superseded_by_canonical_gt_rebuild",
        "reason": "Extended manual notes revealed substantial GT/tracking issues.",
        "allowed_use": "historical report only; do not use as final claim",
    },
    {
        "old_component": "classification-ready 72-clip claim",
        "new_status": "revoked",
        "reason": "Full 72 clips are not reliable classification GT until GT v2 triage/correction is complete.",
        "allowed_use": "none",
    },
])
safe_to_csv(deprecated_policy, OUT_DEPRECATED_OUTPUT_POLICY)

issues_df = pd.DataFrame(issues, columns=["item", "issue_type", "issue_detail", "severity"])
safe_to_csv(issues_df, OUT_ISSUES)

total_notes = int(len(notes))
unique_scans_with_notes = int(notes["scan_frame_id"].nunique())
red_count = int(triage["preliminary_status"].str.startswith("red").sum())
yellow_count = int((triage["preliminary_status"] == "yellow_needs_review").sum())
green_count = int((triage["preliminary_status"] == "green_candidate").sum())
conflict_count = int(triage["has_conflicting_notes"].sum())

decision = pd.DataFrame([{
    "v60_decision": "canonical_gt_rebuild_started",
    "manual_note_count": total_notes,
    "scanframes_with_manual_notes": unique_scans_with_notes,
    "preliminary_green_candidate_count": green_count,
    "preliminary_yellow_needs_review_count": yellow_count,
    "preliminary_red_requires_review_count": red_count,
    "conflicting_note_scanframe_count": conflict_count,
    "old_classification_ready_claim": "revoked",
    "tracking_status": "helper_layer_only",
    "colour_label_source_of_truth": "original_annotation_file_required",
    "hard_issue_count": 0,
    "warning_count": 0,
    "issue_count": 0,
    "ready_for_v61_annotation_source_inventory": True,
    "generated_at": datetime.now().isoformat(timespec="seconds"),
}])
safe_to_csv(decision, OUT_DECISION)

policy_doc = f"""# Week 8 v60 Canonical GT Rebuild Policy

## Decision

The previous task-sheet-oriented pipeline is frozen. It remains useful as a diagnostic and visualization pipeline, but it is not accepted as final ground truth.

## Reason

Manual validation notes revealed substantial issues:

- missing pigs
- wrong bounding boxes
- identity switches
- possible wrong or missing GT anchors
- conflicting manual notes for some scanframes

Therefore, classification is paused until a canonical GT v2 is created.

## New source-of-truth rules

1. Original annotation file is the canonical source for pig colour / identity labels.
2. Visual colour guesses are not canonical.
3. Previous suggested colour mappings are deprecated until verified against the annotation file.
4. Tracking is a helper layer only.
5. Classification can only use clips marked GOLD or explicitly accepted SILVER in GT v2.
6. RED / FIX_REQUIRED clips are excluded from classification until corrected.

## GT v2 clip status

- GOLD: reliable bbox, identity, colour and behaviour.
- SILVER: usable with minor limitations.
- RED: not usable for classification.
- FIX_REQUIRED: potentially usable after manual correction.
- UNKNOWN: not yet reviewed.

## Next step

Run v61 annotation source inventory and identify the original annotation file containing pig IDs, colour labels and behaviour labels.
"""

OUT_POLICY_DOC.write_text(policy_doc)

OUT_REPORT.write_text(
    "# Week 8 v60 Canonical GT Rebuild Bootstrap Report\n\n"
    f"Decision: {decision.iloc[0]['v60_decision']}\n\n"
    f"- Manual notes: {total_notes}\n"
    f"- Scanframes with notes: {unique_scans_with_notes}\n"
    f"- Preliminary green candidates: {green_count}\n"
    f"- Preliminary yellow review clips: {yellow_count}\n"
    f"- Preliminary red / GT-review clips: {red_count}\n"
    f"- Conflicting-note scanframes: {conflict_count}\n\n"
    "Classification is paused until canonical GT v2 is built.\n"
)

OUT_NOTE.write_text(
    "# Week 8 v60 Canonical GT Rebuild Bootstrap\n\n"
    f"- v60 decision: {decision.iloc[0]['v60_decision']}\n"
    f"- Manual notes: {total_notes}\n"
    f"- Scanframes with manual notes: {unique_scans_with_notes}\n"
    f"- Green candidates: {green_count}\n"
    f"- Yellow needs review: {yellow_count}\n"
    f"- Red requires GT review: {red_count}\n"
    f"- Conflicting-note scanframes: {conflict_count}\n"
    "- Old classification-ready claim: revoked\n"
    "- Tracking status: helper layer only\n"
    "- Colour label source of truth: original annotation file required\n"
    "- Ready for v61 annotation source inventory: True\n"
)

progress_row = pd.DataFrame([{
    "date": datetime.now().date().isoformat(),
    "stage": "v60",
    "task_name": "Canonical GT rebuild bootstrap",
    "status": "PASS",
    "input_summary": str(MANUAL_NOTES),
    "output_summary": str(OUT),
    "hard_issues": 0,
    "warnings": 0,
    "next_action": "v61 annotation source inventory",
}])

if OUT_PROGRESS.exists():
    old = pd.read_csv(OUT_PROGRESS)
    progress = pd.concat([old, progress_row], ignore_index=True)
else:
    progress = progress_row

safe_to_csv(progress, OUT_PROGRESS)

print("Saved:")
print(OUT_DECISION)
print(OUT_MANUAL_NOTE_SUMMARY)
print(OUT_SCANFRAME_TRIAGE_PRELIM)
print(OUT_DEPRECATED_OUTPUT_POLICY)
print(OUT_POLICY_DOC)
print(OUT_REPORT)
print(OUT_NOTE)

print()
print("=== v60 decision ===")
print(decision.to_string(index=False))

print()
print("=== v60 manual note summary ===")
print(note_summary.to_string(index=False))

print()
print("=== v60 preliminary triage counts ===")
print(triage["preliminary_status"].value_counts().to_string())

print()
print("=== v60 red / review examples ===")
print(triage[triage["classification_use"].str.contains("exclude", na=False)][[
    "scan_frame_id",
    "video_id",
    "issue_types",
    "severities",
    "preliminary_status",
    "classification_use",
    "manual_note_summary",
]].head(25).to_string(index=False))
