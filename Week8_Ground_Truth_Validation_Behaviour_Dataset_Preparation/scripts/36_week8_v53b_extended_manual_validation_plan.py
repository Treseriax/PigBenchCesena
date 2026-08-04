from pathlib import Path
from datetime import datetime
import csv
import pandas as pd


ROOT = Path.home() / "PigBench"
W8 = ROOT / "Week8_Ground_Truth_Validation_Behaviour_Dataset_Preparation"

V52E_PLAN = W8 / "outputs" / "v52e_manual_visual_validation_protocol" / "week8_v52e_manual_review_plan.csv"
V52D_NOTES = W8 / "validation" / "week8_v52d_visual_validation_notes.csv"

OUT = W8 / "outputs" / "v53b_extended_manual_validation_plan"
NOTES = W8 / "notes"
REPORTS = W8 / "reports"
PROGRESS = W8 / "progress"

for p in [OUT, NOTES, REPORTS, PROGRESS]:
    p.mkdir(parents=True, exist_ok=True)

OUT_PLAN = OUT / "week8_v53b_extended_manual_validation_plan.csv"
OUT_CHECKLIST = OUT / "week8_v53b_extended_manual_validation_checklist.csv"
OUT_DECISION = OUT / "week8_v53b_decision_summary.csv"
OUT_ISSUES = OUT / "week8_v53b_issues.csv"
OUT_NOTE = NOTES / "week8_v53b_extended_manual_validation_plan_notes.md"
OUT_REPORT = REPORTS / "week8_v53b_extended_manual_validation_plan_report.md"
OUT_PROGRESS = PROGRESS / "week8_experiment_progress_log.csv"


def safe_to_csv(df, path):
    df.to_csv(path, index=False, quoting=csv.QUOTE_ALL, escapechar="\\", lineterminator="\n")


issues = []

if not V52E_PLAN.exists():
    issues.append({
        "item": str(V52E_PLAN),
        "issue_type": "hard_missing_review_plan",
        "issue_detail": "v52e manual review plan is missing.",
        "severity": "hard",
    })

if not V52D_NOTES.exists():
    issues.append({
        "item": str(V52D_NOTES),
        "issue_type": "hard_missing_manual_notes_csv",
        "issue_detail": "v52d manual notes CSV is missing.",
        "severity": "hard",
    })

if issues:
    issues_df = pd.DataFrame(issues)
    safe_to_csv(issues_df, OUT_ISSUES)
    decision = pd.DataFrame([{
        "v53b_decision": "extended_manual_validation_plan_blocked",
        "hard_issue_count": int((issues_df["severity"] == "hard").sum()),
        "ready_for_extended_manual_validation": False,
        "generated_at": datetime.now().isoformat(timespec="seconds"),
    }])
    safe_to_csv(decision, OUT_DECISION)
    print(decision.to_string(index=False))
    raise SystemExit(1)


plan = pd.read_csv(V52E_PLAN)
notes = pd.read_csv(V52D_NOTES)

already_reviewed = set(notes["scan_frame_id"].astype(str).tolist()) if len(notes) else set()

plan = plan.drop_duplicates("scan_frame_id").copy()
pending = plan[~plan["scan_frame_id"].astype(str).isin(already_reviewed)].copy()

for c in ["qa_priority_score", "draw_ok_ratio", "missing_ratio", "review_needed_ratio"]:
    if c in pending.columns:
        pending[c] = pd.to_numeric(pending[c], errors="coerce")

def pick(level=None, review_set=None, n=0):
    df = pending.copy()
    if level is not None:
        df = df[df["manual_priority_level"] == level]
    if review_set is not None:
        df = df[df["review_set"] == review_set]
    if "qa_priority_score" in df.columns:
        df = df.sort_values(["qa_priority_score", "missing_ratio", "review_needed_ratio"], ascending=[False, False, False])
    return df.head(n)

medium = pick(level="medium_priority", n=5)
normal = pick(level="normal_review", n=4)
strong = pick(review_set="strong_clip_spot_check", n=3)

selected = pd.concat([medium, normal, strong], ignore_index=True)
selected = selected.drop_duplicates("scan_frame_id").copy()

selected["v53b_review_group"] = ""
selected.loc[selected["scan_frame_id"].isin(medium["scan_frame_id"]), "v53b_review_group"] = "medium_priority_extension"
selected.loc[selected["scan_frame_id"].isin(normal["scan_frame_id"]), "v53b_review_group"] = "normal_review_extension"
selected.loc[selected["scan_frame_id"].isin(strong["scan_frame_id"]), "v53b_review_group"] = "strong_spotcheck_extension"

selected["recommended_inspection_times_sec"] = "0, 2.5, 5.0, 7.5, 9.5"
selected["save_note_policy"] = "Save one note per clip minimum. Use bbox_ok/info if acceptable; otherwise save the most relevant issue type."

safe_to_csv(selected, OUT_PLAN)

checklist_rows = []
for _, r in selected.iterrows():
    for t in [0.0, 2.5, 5.0, 7.5, 9.5]:
        checklist_rows.append({
            "scan_frame_id": r["scan_frame_id"],
            "video_id": r["video_id"],
            "v53b_review_group": r["v53b_review_group"],
            "manual_priority_level": r["manual_priority_level"],
            "target_time_sec": t,
            "check_bbox_alignment": "",
            "check_identity_consistency": "",
            "check_behaviour_plausibility": "",
            "check_fallback_if_visible": "",
            "note_saved": "",
            "manual_comment": "",
        })

checklist = pd.DataFrame(checklist_rows)
safe_to_csv(checklist, OUT_CHECKLIST)

issues_df = pd.DataFrame(issues, columns=["item", "issue_type", "issue_detail", "severity"])
safe_to_csv(issues_df, OUT_ISSUES)

decision = pd.DataFrame([{
    "v53b_decision": "extended_manual_validation_plan_created",
    "already_reviewed_clip_count": len(already_reviewed),
    "selected_clip_count": int(selected["scan_frame_id"].nunique()),
    "medium_priority_selected": int((selected["v53b_review_group"] == "medium_priority_extension").sum()),
    "normal_review_selected": int((selected["v53b_review_group"] == "normal_review_extension").sum()),
    "strong_spotcheck_selected": int((selected["v53b_review_group"] == "strong_spotcheck_extension").sum()),
    "checklist_rows": int(len(checklist)),
    "hard_issue_count": 0,
    "warning_count": 0,
    "issue_count": 0,
    "ready_for_extended_manual_validation": True,
    "generated_at": datetime.now().isoformat(timespec="seconds"),
}])

safe_to_csv(decision, OUT_DECISION)

note = f"""# Week 8 v53b Extended Manual Validation Plan

## Summary

- v53b decision: {decision.iloc[0]['v53b_decision']}
- Already reviewed clips: {len(already_reviewed)}
- Selected additional clips: {selected['scan_frame_id'].nunique()}
- Medium-priority selected: {(selected['v53b_review_group'] == 'medium_priority_extension').sum()}
- Normal-review selected: {(selected['v53b_review_group'] == 'normal_review_extension').sum()}
- Strong spot-check selected: {(selected['v53b_review_group'] == 'strong_spotcheck_extension').sum()}
- Ready for extended manual validation: True

## Instruction

Open the v52d visualizer and save one manual note for each selected clip. If the clip is acceptable, use:

- issue_type: bbox_ok
- severity: info
- note: Mostly correct

If an issue is visible, use the relevant issue type such as bbox_wrong, identity_uncertain, identity_switch, missing_pig, false_positive, or behaviour_uncertain.
"""

OUT_NOTE.write_text(note)

OUT_REPORT.write_text(
    "# Week 8 v53b Extended Manual Validation Plan Report\n\n"
    f"Decision: {decision.iloc[0]['v53b_decision']}\n\n"
    f"Selected additional clips: {selected['scan_frame_id'].nunique()}\n\n"
    "The goal is to strengthen the validation evidence before final packaging and classification baseline preparation.\n"
)

progress_row = pd.DataFrame([{
    "date": datetime.now().date().isoformat(),
    "stage": "v53b",
    "task_name": "Extended manual validation plan",
    "status": "PASS",
    "input_summary": str(V52E_PLAN),
    "output_summary": str(OUT),
    "hard_issues": 0,
    "warnings": 0,
    "next_action": "Review selected clips in v52d visualizer and save notes.",
}])

if OUT_PROGRESS.exists():
    old = pd.read_csv(OUT_PROGRESS)
    progress = pd.concat([old, progress_row], ignore_index=True)
else:
    progress = progress_row

safe_to_csv(progress, OUT_PROGRESS)

print("Saved:")
print(OUT_PLAN)
print(OUT_CHECKLIST)
print(OUT_DECISION)
print(OUT_ISSUES)
print(OUT_NOTE)

print()
print("=== v53b decision ===")
print(decision.to_string(index=False))

print()
print("=== selected clips for extended manual validation ===")
print(selected[[
    "scan_frame_id",
    "video_id",
    "v53b_review_group",
    "manual_priority_level",
    "quality_tier",
    "draw_ok_ratio",
    "missing_ratio",
    "review_needed_ratio",
]].to_string(index=False))
