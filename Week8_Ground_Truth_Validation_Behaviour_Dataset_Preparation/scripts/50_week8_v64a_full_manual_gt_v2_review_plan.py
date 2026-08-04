from pathlib import Path
from datetime import datetime
import csv
import pandas as pd


ROOT = Path.home() / "PigBench"
W8 = ROOT / "Week8_Ground_Truth_Validation_Behaviour_Dataset_Preparation"

V63A_STATUS = W8 / "outputs" / "v63a_canonical_gt_v2_schema" / "week8_v63a_scanframe_gt_v2_initial_status.csv"
V63C_SCANFRAME = W8 / "outputs" / "v63c_reviewed_assignment_consolidation" / "week8_v63c_scanframe_assignment_summary.csv"
ASSIGNMENTS = W8 / "validation" / "week8_v63b_manual_gt_v2_assignments.csv"

OUT = W8 / "outputs" / "v64a_full_manual_gt_v2_review_plan"
NOTES = W8 / "notes"
REPORTS = W8 / "reports"
PROGRESS = W8 / "progress"

for p in [OUT, NOTES, REPORTS, PROGRESS]:
    p.mkdir(parents=True, exist_ok=True)

OUT_PLAN = OUT / "week8_v64a_remaining_full_manual_gt_v2_review_plan.csv"
OUT_BATCHES = OUT / "week8_v64a_review_batches.csv"
OUT_DECISION = OUT / "week8_v64a_decision_summary.csv"
OUT_ISSUES = OUT / "week8_v64a_issues.csv"
OUT_NOTE = NOTES / "week8_v64a_full_manual_gt_v2_review_plan_notes.md"
OUT_REPORT = REPORTS / "week8_v64a_full_manual_gt_v2_review_plan_report.md"
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

for p in [V63A_STATUS, V63C_SCANFRAME, ASSIGNMENTS]:
    if not p.exists():
        issues.append({
            "item": str(p),
            "issue_type": "hard_missing_required_input",
            "issue_detail": "Required v64a input is missing.",
            "severity": "hard",
        })

if issues:
    issues_df = pd.DataFrame(issues)
    safe_to_csv(issues_df, OUT_ISSUES)
    decision = pd.DataFrame([{
        "v64a_decision": "full_manual_gt_v2_review_plan_blocked",
        "hard_issue_count": int((issues_df["severity"] == "hard").sum()),
        "ready_for_full_manual_review": False,
        "generated_at": datetime.now().isoformat(timespec="seconds"),
    }])
    safe_to_csv(decision, OUT_DECISION)
    print(decision.to_string(index=False))
    raise SystemExit(1)


status = pd.read_csv(V63A_STATUS).fillna("")
summary = pd.read_csv(V63C_SCANFRAME).fillna("")
assignments = pd.read_csv(ASSIGNMENTS).fillna("")

for df in [status, summary, assignments]:
    for c in df.columns:
        df[c] = df[c].map(clean_str)

merged = status.merge(
    summary[[
        "scan_frame_id",
        "reviewed_objects",
        "gold_usable_objects",
        "pending_or_fix_objects",
        "scanframe_gt_v2_decision",
        "classification_scope",
    ]],
    on="scan_frame_id",
    how="left",
)

for c in [
    "reviewed_objects",
    "gold_usable_objects",
    "pending_or_fix_objects",
    "scanframe_gt_v2_decision",
    "classification_scope",
]:
    merged[c] = merged[c].fillna("")

priority_rank = {
    "P0_gt_anchor_problem": 0,
    "P1_manual_major_or_critical_issue": 1,
    "P1_candidate_box_count_not_six": 2,
    "P2_review_required": 3,
    "P3_standard_review": 4,
}

def parse_int(x):
    try:
        return int(float(clean_str(x)))
    except Exception:
        return 0

merged["reviewed_objects_int"] = merged["reviewed_objects"].map(parse_int)
merged["gold_usable_objects_int"] = merged["gold_usable_objects"].map(parse_int)
merged["pending_or_fix_objects_int"] = merged["pending_or_fix_objects"].map(parse_int)
merged["priority_rank"] = merged["initial_scanframe_gt_v2_status"].map(lambda x: priority_rank.get(clean_str(x), 99))

def review_state(row):
    reviewed = int(row["reviewed_objects_int"])
    gold = int(row["gold_usable_objects_int"])
    pending = int(row["pending_or_fix_objects_int"])

    if reviewed == 0:
        return "not_started"
    if reviewed < 6:
        return "partial_review"
    if gold == 6:
        return "completed_full_gold"
    if reviewed == 6:
        return "completed_mixed"
    return "unknown"

merged["v64_review_state"] = merged.apply(review_state, axis=1)

# We still include already reviewed mixed clips at the end for possible second pass,
# but the main review plan first covers all not-started clips.
def required_action(row):
    state = row["v64_review_state"]
    status = clean_str(row["initial_scanframe_gt_v2_status"])

    if state == "not_started":
        return "review_all_6_canonical_objects"
    if state == "partial_review":
        return "finish_remaining_objects"
    if state == "completed_mixed":
        return "optional_second_pass_try_fix_pending_objects"
    if state == "completed_full_gold":
        return "done_no_action"
    return "inspect"

merged["required_action"] = merged.apply(required_action, axis=1)

plan = merged[merged["required_action"] != "done_no_action"].copy()

# Main order: not started/partial first, high risk first, then mixed second-pass.
state_rank = {
    "not_started": 0,
    "partial_review": 1,
    "completed_mixed": 2,
}
plan["state_rank"] = plan["v64_review_state"].map(lambda x: state_rank.get(x, 9))

plan = plan.sort_values(
    ["state_rank", "priority_rank", "scan_frame_id"],
    ascending=[True, True, True],
).reset_index(drop=True)

plan["review_order"] = range(1, len(plan) + 1)
plan["recommended_batch"] = ((plan["review_order"] - 1) // 10) + 1

out_cols = [
    "review_order",
    "recommended_batch",
    "scan_frame_id",
    "video_id",
    "existing_candidate_box_count",
    "manual_issue_types",
    "manual_max_severity",
    "initial_scanframe_gt_v2_status",
    "initial_classification_gate",
    "v64_review_state",
    "reviewed_objects",
    "gold_usable_objects",
    "pending_or_fix_objects",
    "required_action",
]

safe_to_csv(plan[out_cols], OUT_PLAN)

batch_rows = []
for batch, g in plan.groupby("recommended_batch"):
    batch_rows.append({
        "recommended_batch": int(batch),
        "scanframe_count": int(len(g)),
        "first_review_order": int(g["review_order"].min()),
        "last_review_order": int(g["review_order"].max()),
        "scanframes": ";".join(g["scan_frame_id"].tolist()),
        "status_mix": ";".join(sorted(set(g["initial_scanframe_gt_v2_status"].tolist()))),
        "action_mix": ";".join(sorted(set(g["required_action"].tolist()))),
    })

batches = pd.DataFrame(batch_rows)
safe_to_csv(batches, OUT_BATCHES)

issues_df = pd.DataFrame(issues, columns=["item", "issue_type", "issue_detail", "severity"])
safe_to_csv(issues_df, OUT_ISSUES)

not_started_count = int((merged["v64_review_state"] == "not_started").sum())
partial_count = int((merged["v64_review_state"] == "partial_review").sum())
mixed_count = int((merged["v64_review_state"] == "completed_mixed").sum())
full_gold_count = int((merged["v64_review_state"] == "completed_full_gold").sum())

decision = pd.DataFrame([{
    "v64a_decision": "full_manual_gt_v2_review_plan_created",
    "total_scanframes": int(len(merged)),
    "not_started_scanframes": not_started_count,
    "partial_review_scanframes": partial_count,
    "completed_mixed_scanframes": mixed_count,
    "completed_full_gold_scanframes": full_gold_count,
    "review_plan_scanframes": int(len(plan)),
    "main_remaining_unreviewed_scanframes": not_started_count + partial_count,
    "review_batch_count": int(batches["recommended_batch"].nunique()) if len(batches) else 0,
    "plan_path": str(OUT_PLAN),
    "batch_path": str(OUT_BATCHES),
    "hard_issue_count": 0,
    "warning_count": 0,
    "issue_count": 0,
    "ready_for_full_manual_review": True,
    "generated_at": datetime.now().isoformat(timespec="seconds"),
}])

safe_to_csv(decision, OUT_DECISION)

OUT_NOTE.write_text(
    "# Week 8 v64a Full Manual GT v2 Review Plan\n\n"
    f"- v64a decision: {decision.iloc[0]['v64a_decision']}\n"
    f"- Total scanframes: {len(merged)}\n"
    f"- Not started scanframes: {not_started_count}\n"
    f"- Partial review scanframes: {partial_count}\n"
    f"- Completed mixed scanframes: {mixed_count}\n"
    f"- Completed full-gold scanframes: {full_gold_count}\n"
    f"- Review plan scanframes: {len(plan)}\n"
    f"- Review batches: {len(batches)}\n"
    f"- Ready for full manual review: True\n\n"
    "Review rule: every scanframe must receive a decision for all six canonical Excel identities. Do not force gold if colour/identity is not visible. Mark missing/fix/red honestly.\n"
)

OUT_REPORT.write_text(
    "# Week 8 v64a Full Manual GT v2 Review Plan Report\n\n"
    f"Decision: {decision.iloc[0]['v64a_decision']}\n\n"
    f"Review plan: `{OUT_PLAN}`\n\n"
    f"Review batches: `{OUT_BATCHES}`\n"
)

progress_row = pd.DataFrame([{
    "date": datetime.now().date().isoformat(),
    "stage": "v64a",
    "task_name": "Full manual GT v2 review plan",
    "status": "PASS",
    "input_summary": str(V63C_SCANFRAME),
    "output_summary": str(OUT),
    "hard_issues": 0,
    "warnings": 0,
    "next_action": "Use v63b visualizer to review remaining scanframes batch by batch.",
}])

if OUT_PROGRESS.exists():
    old = pd.read_csv(OUT_PROGRESS)
    progress = pd.concat([old, progress_row], ignore_index=True)
else:
    progress = progress_row

safe_to_csv(progress, OUT_PROGRESS)

print("Saved:")
print(OUT_PLAN)
print(OUT_BATCHES)
print(OUT_DECISION)
print(OUT_NOTE)

print()
print("=== v64a decision ===")
print(decision.to_string(index=False))

print()
print("=== review batches ===")
print(batches.to_string(index=False))

print()
print("=== first 40 review items ===")
print(plan[out_cols].head(40).to_string(index=False))
