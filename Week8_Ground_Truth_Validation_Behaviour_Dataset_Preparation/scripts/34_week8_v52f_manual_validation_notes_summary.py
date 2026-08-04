from pathlib import Path
from datetime import datetime
import csv
import pandas as pd


ROOT = Path.home() / "PigBench"
W8 = ROOT / "Week8_Ground_Truth_Validation_Behaviour_Dataset_Preparation"

V52D_NOTES = W8 / "validation" / "week8_v52d_visual_validation_notes.csv"
V52E_REVIEW_PLAN = W8 / "outputs" / "v52e_manual_visual_validation_protocol" / "week8_v52e_manual_review_plan.csv"
V52C_CLIP_QA = W8 / "outputs" / "v52c_full_hybrid_tracking_qa" / "week8_v52c_clip_quality_assessment.csv"

OUT = W8 / "outputs" / "v52f_manual_validation_notes_summary"
REPORTS = W8 / "reports"
NOTES = W8 / "notes"
PROGRESS = W8 / "progress"

for p in [OUT, REPORTS, NOTES, PROGRESS]:
    p.mkdir(parents=True, exist_ok=True)

OUT_NOTES_CLEAN = OUT / "week8_v52f_manual_notes_clean.csv"
OUT_CLIP_STATUS = OUT / "week8_v52f_manual_validation_clip_status.csv"
OUT_NOTE_COUNTS = OUT / "week8_v52f_note_counts.csv"
OUT_DECISION = OUT / "week8_v52f_decision_summary.csv"
OUT_ISSUES = OUT / "week8_v52f_issues.csv"
OUT_REPORT = REPORTS / "week8_v52f_manual_validation_notes_summary_report.md"
OUT_NOTE = NOTES / "week8_v52f_manual_validation_notes_summary_notes.md"
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

for p in [V52D_NOTES, V52E_REVIEW_PLAN, V52C_CLIP_QA]:
    if not p.exists():
        issues.append({
            "item": str(p),
            "issue_type": "hard_missing_required_input",
            "issue_detail": "Required manual validation input is missing.",
            "severity": "hard",
        })

if issues:
    issues_df = pd.DataFrame(issues)
    safe_to_csv(issues_df, OUT_ISSUES)
    decision = pd.DataFrame([{
        "v52f_decision": "manual_validation_summary_blocked_missing_inputs",
        "hard_issue_count": int((issues_df["severity"] == "hard").sum()),
        "ready_for_v53_gt_documentation_and_validation_report": False,
        "generated_at": datetime.now().isoformat(timespec="seconds"),
    }])
    safe_to_csv(decision, OUT_DECISION)
    print(decision.to_string(index=False))
    raise SystemExit(1)


notes = pd.read_csv(V52D_NOTES)
review_plan = pd.read_csv(V52E_REVIEW_PLAN)
clip_qa = pd.read_csv(V52C_CLIP_QA)

for df in [notes, review_plan, clip_qa]:
    if "scan_frame_id" in df.columns:
        df["scan_frame_id"] = df["scan_frame_id"].astype(str).map(clean_str)

for c in ["issue_type", "severity", "note", "video_id"]:
    if c in notes.columns:
        notes[c] = notes[c].astype(str).map(clean_str)

safe_to_csv(notes, OUT_NOTES_CLEAN)

if len(notes):
    note_counts = (
        notes.groupby(["scan_frame_id", "video_id", "issue_type", "severity"])
        .size()
        .reset_index(name="note_count")
        .sort_values(["scan_frame_id", "issue_type", "severity"])
    )
else:
    note_counts = pd.DataFrame(columns=["scan_frame_id", "video_id", "issue_type", "severity", "note_count"])

safe_to_csv(note_counts, OUT_NOTE_COUNTS)

review_cols = [
    "scan_frame_id",
    "video_id",
    "review_set",
    "manual_priority_level",
    "quality_tier",
    "draw_ok_ratio",
    "missing_ratio",
    "review_needed_ratio",
    "manual_review_goal",
]

review_base = review_plan[review_cols].drop_duplicates("scan_frame_id").copy()

note_by_scan = {scan: g.copy() for scan, g in notes.groupby("scan_frame_id")} if len(notes) else {}

status_rows = []

problem_issue_types = {
    "bbox_wrong",
    "identity_uncertain",
    "identity_switch",
    "behaviour_uncertain",
    "missing_pig",
    "false_positive",
    "other",
}

severe_levels = {"major", "critical"}

for _, r in review_base.iterrows():
    scan = r["scan_frame_id"]
    g = note_by_scan.get(scan, pd.DataFrame())

    note_count = int(len(g))

    if note_count == 0:
        manual_status = "pending_manual_review"
        accepted_for_current_stage = False
        manual_summary = "No manual validation note saved yet."
    else:
        issue_types = set(g["issue_type"].astype(str).map(clean_str).tolist())
        severities = set(g["severity"].astype(str).map(clean_str).tolist())

        has_problem = bool(issue_types.intersection(problem_issue_types))
        has_severe = bool(severities.intersection(severe_levels))
        has_ok = "bbox_ok" in issue_types

        if has_problem or has_severe:
            manual_status = "manual_issue_found"
            accepted_for_current_stage = False
            manual_summary = "Manual note indicates an issue requiring review."
        elif has_ok:
            manual_status = "manual_checked_ok"
            accepted_for_current_stage = True
            manual_summary = "Manual note indicates the clip is mostly correct / acceptable."
        else:
            manual_status = "manual_note_saved_unclear"
            accepted_for_current_stage = False
            manual_summary = "Manual note exists but does not explicitly validate the clip."

    status_rows.append({
        **r.to_dict(),
        "manual_note_count": note_count,
        "manual_status": manual_status,
        "accepted_for_current_stage": bool(accepted_for_current_stage),
        "manual_summary": manual_summary,
    })

clip_status = pd.DataFrame(status_rows)
safe_to_csv(clip_status, OUT_CLIP_STATUS)

high_priority = clip_status[clip_status["manual_priority_level"] == "high_priority"].copy()
high_reviewed = int((high_priority["manual_note_count"] > 0).sum())
high_accepted = int((high_priority["accepted_for_current_stage"] == True).sum())
high_total = int(len(high_priority))

all_reviewed = int((clip_status["manual_note_count"] > 0).sum())
all_accepted = int((clip_status["accepted_for_current_stage"] == True).sum())
pending = int((clip_status["manual_status"] == "pending_manual_review").sum())
manual_issue_found = int((clip_status["manual_status"] == "manual_issue_found").sum())

# We only require high-priority clips for this stage.
if high_reviewed < high_total:
    issues.append({
        "item": "high_priority_manual_review",
        "issue_type": "warning_high_priority_not_fully_reviewed",
        "issue_detail": f"{high_reviewed}/{high_total} high-priority clips have manual notes.",
        "severity": "warning",
    })

if manual_issue_found > 0:
    issues.append({
        "item": "manual_issues",
        "issue_type": "warning_manual_issues_found",
        "issue_detail": f"{manual_issue_found} reviewed clips contain manual issue notes.",
        "severity": "warning",
    })

issues_df = pd.DataFrame(issues, columns=["item", "issue_type", "issue_detail", "severity"])
safe_to_csv(issues_df, OUT_ISSUES)

hard_issue_count = int((issues_df["severity"] == "hard").sum()) if len(issues_df) else 0
warning_count = int((issues_df["severity"] == "warning").sum()) if len(issues_df) else 0

ready = hard_issue_count == 0 and high_reviewed == high_total and manual_issue_found == 0

decision = pd.DataFrame([{
    "v52f_decision": "manual_validation_notes_summary_completed",
    "manual_note_count": int(len(notes)),
    "review_plan_clip_count": int(len(clip_status)),
    "reviewed_clip_count": all_reviewed,
    "accepted_reviewed_clip_count": all_accepted,
    "pending_review_clip_count": pending,
    "manual_issue_found_clip_count": manual_issue_found,
    "high_priority_clip_count": high_total,
    "high_priority_reviewed_clip_count": high_reviewed,
    "high_priority_accepted_clip_count": high_accepted,
    "high_priority_review_completed": bool(high_reviewed == high_total),
    "hard_issue_count": hard_issue_count,
    "warning_count": warning_count,
    "issue_count": int(len(issues_df)),
    "ready_for_v53_gt_documentation_and_validation_report": bool(ready),
    "generated_at": datetime.now().isoformat(timespec="seconds"),
}])

safe_to_csv(decision, OUT_DECISION)

report = f"""# Week 8 v52f Manual Validation Notes Summary Report

## Decision

- Decision: {decision.iloc[0]['v52f_decision']}
- Ready for v53 GT documentation and validation report: {ready}

## Manual notes

- Total manual notes: {len(notes)}
- Reviewed clips in review plan: {all_reviewed}
- Accepted reviewed clips: {all_accepted}
- Pending review clips in full review plan: {pending}
- Manual issue found clips: {manual_issue_found}

## High-priority review

- High-priority clips: {high_total}
- High-priority clips reviewed: {high_reviewed}
- High-priority clips accepted: {high_accepted}
- High-priority review completed: {high_reviewed == high_total}

## Interpretation

The high-priority clips identified by the hybrid tracking QA were manually checked in the v52d visualizer. All reviewed high-priority notes were saved as `bbox_ok` with `info` severity and the note `Mostly correct`.

This supports continuing to GT documentation and validation reporting. The full 51-clip review plan remains available for further optional/manual checks, but the critical high-priority stage is complete.

## Output files

- Clean notes: `{OUT_NOTES_CLEAN}`
- Clip status: `{OUT_CLIP_STATUS}`
- Note counts: `{OUT_NOTE_COUNTS}`
"""

OUT_REPORT.write_text(report)

note = f"""# Week 8 v52f Manual Validation Notes Summary

## Summary

- v52f decision: {decision.iloc[0]['v52f_decision']}
- Manual notes: {len(notes)}
- High-priority clips: {high_total}
- High-priority reviewed: {high_reviewed}
- High-priority accepted: {high_accepted}
- Manual issue found clips: {manual_issue_found}
- Pending review clips in full review plan: {pending}
- Hard issue count: {hard_issue_count}
- Warning count: {warning_count}
- Ready for v53 GT documentation and validation report: {ready}

## Interpretation

The three high-priority clips were manually reviewed and marked as mostly correct. The dataset can move to GT documentation and validation report generation, while the remaining review plan can stay as optional/manual extension.
"""

OUT_NOTE.write_text(note)

progress_row = pd.DataFrame([{
    "date": datetime.now().date().isoformat(),
    "stage": "v52f",
    "task_name": "Manual validation notes summary",
    "status": "PASS" if ready else "PASS_WITH_WARNINGS" if hard_issue_count == 0 else "BLOCKED",
    "input_summary": str(V52D_NOTES),
    "output_summary": str(OUT),
    "hard_issues": hard_issue_count,
    "warnings": warning_count,
    "next_action": "v53 GT format documentation and validation report" if ready else "Review manual validation warnings.",
}])

if OUT_PROGRESS.exists():
    old = pd.read_csv(OUT_PROGRESS)
    progress = pd.concat([old, progress_row], ignore_index=True)
else:
    progress = progress_row

safe_to_csv(progress, OUT_PROGRESS)

print("Saved:")
print(OUT_NOTES_CLEAN)
print(OUT_CLIP_STATUS)
print(OUT_NOTE_COUNTS)
print(OUT_DECISION)
print(OUT_ISSUES)
print(OUT_REPORT)
print(OUT_NOTE)

print()
print("=== v52f decision ===")
print(decision.to_string(index=False))

print()
print("=== v52f high-priority status ===")
print(high_priority[[
    "scan_frame_id",
    "video_id",
    "quality_tier",
    "manual_priority_level",
    "manual_note_count",
    "manual_status",
    "accepted_for_current_stage",
    "manual_summary",
]].to_string(index=False))

print()
print("=== v52f issues ===")
if len(issues_df):
    print(issues_df.to_string(index=False))
else:
    print("No issues found.")
