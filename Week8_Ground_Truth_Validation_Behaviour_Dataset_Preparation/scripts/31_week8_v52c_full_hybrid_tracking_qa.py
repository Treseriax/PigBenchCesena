from pathlib import Path
from datetime import datetime
import csv
import pandas as pd


ROOT = Path.home() / "PigBench"
W8 = ROOT / "Week8_Ground_Truth_Validation_Behaviour_Dataset_Preparation"

V52B3_ROWS = W8 / "outputs" / "v52b3_hybrid_tracking_full" / "week8_v52b3_hybrid_tracking_rows.csv"
V52B3_CLIP_SUMMARY = W8 / "outputs" / "v52b3_hybrid_tracking_full" / "week8_v52b3_clip_summary.csv"
V52B3_OBJECT_SUMMARY = W8 / "outputs" / "v52b3_hybrid_tracking_full" / "week8_v52b3_object_summary.csv"

OUT = W8 / "outputs" / "v52c_full_hybrid_tracking_qa"
REPORTS = W8 / "reports"
NOTES = W8 / "notes"
PROGRESS = W8 / "progress"

for p in [OUT, REPORTS, NOTES, PROGRESS]:
    p.mkdir(parents=True, exist_ok=True)

OUT_DECISION = OUT / "week8_v52c_decision_summary.csv"
OUT_ISSUES = OUT / "week8_v52c_issues.csv"
OUT_CLIP_QA = OUT / "week8_v52c_clip_quality_assessment.csv"
OUT_OBJECT_QA = OUT / "week8_v52c_object_quality_assessment.csv"
OUT_REVIEW_QUEUE = OUT / "week8_v52c_manual_review_queue.csv"
OUT_SUCCESS_CASES = OUT / "week8_v52c_successful_cases.csv"
OUT_CHALLENGING_CASES = OUT / "week8_v52c_challenging_cases.csv"
OUT_SOURCE_DISTRIBUTION = OUT / "week8_v52c_tracking_source_distribution.csv"
OUT_REPORT = REPORTS / "week8_v52c_full_hybrid_tracking_qa_report.md"
OUT_NOTE = NOTES / "week8_v52c_full_hybrid_tracking_qa_notes.md"
OUT_PROGRESS = PROGRESS / "week8_experiment_progress_log.csv"


def safe_to_csv(df, path):
    df.to_csv(path, index=False, quoting=csv.QUOTE_ALL, escapechar="\\", lineterminator="\n")


def quality_tier(row):
    draw = float(row["draw_ok_ratio"])
    stable = float(row["stable_tracklet_ratio"])
    missing = float(row["missing_ratio"])
    review = float(row["review_needed_ratio"])

    if draw >= 0.90 and stable >= 0.70 and missing <= 0.10 and review <= 0.25:
        return "strong_visual_tracking_support"
    if draw >= 0.75 and missing <= 0.25:
        return "usable_with_review"
    if draw >= 0.60 and missing <= 0.40:
        return "limited_review_required"
    return "challenging_low_tracking_support"


issues = []

for p in [V52B3_ROWS, V52B3_CLIP_SUMMARY, V52B3_OBJECT_SUMMARY]:
    if not p.exists():
        issues.append({
            "item": str(p),
            "issue_type": "hard_missing_required_input",
            "issue_detail": "Required v52b3 input is missing.",
            "severity": "hard",
        })

if issues:
    issues_df = pd.DataFrame(issues)
    safe_to_csv(issues_df, OUT_ISSUES)
    decision = pd.DataFrame([{
        "v52c_decision": "full_hybrid_tracking_qa_blocked_missing_inputs",
        "hard_issue_count": int((issues_df["severity"] == "hard").sum()),
        "ready_for_v52d_tracking_integrated_visualizer": False,
        "generated_at": datetime.now().isoformat(timespec="seconds"),
    }])
    safe_to_csv(decision, OUT_DECISION)
    print(decision.to_string(index=False))
    raise SystemExit(1)


rows = pd.read_csv(V52B3_ROWS)
clip = pd.read_csv(V52B3_CLIP_SUMMARY)
obj = pd.read_csv(V52B3_OBJECT_SUMMARY)

numeric_clip_cols = [
    "stable_tracklet_ratio",
    "recall_fallback_ratio",
    "missing_ratio",
    "draw_ok_ratio",
    "review_needed_ratio",
    "stable_tracklet_rows",
    "recall_fallback_rows",
    "missing_rows",
    "draw_ok_rows",
    "review_needed_rows",
    "total_object_frame_rows",
]

for c in numeric_clip_cols:
    if c in clip.columns:
        clip[c] = pd.to_numeric(clip[c], errors="coerce")

numeric_obj_cols = [
    "stable_tracklet_rows",
    "recall_fallback_rows",
    "missing_rows",
    "draw_ok_rows",
    "draw_ok_ratio",
    "review_needed_ratio",
]

for c in numeric_obj_cols:
    if c in obj.columns:
        obj[c] = pd.to_numeric(obj[c], errors="coerce")

clip["quality_tier"] = clip.apply(quality_tier, axis=1)

clip["qa_priority_score"] = (
    2.0 * clip["missing_ratio"].fillna(0)
    + 1.2 * clip["recall_fallback_ratio"].fillna(0)
    + 1.0 * clip["review_needed_ratio"].fillna(0)
    - 0.7 * clip["stable_tracklet_ratio"].fillna(0)
)

clip["manual_review_recommendation"] = clip["quality_tier"].map({
    "strong_visual_tracking_support": "spot_check_only",
    "usable_with_review": "normal_review",
    "limited_review_required": "priority_review",
    "challenging_low_tracking_support": "high_priority_manual_review",
})

safe_to_csv(clip, OUT_CLIP_QA)

obj["object_quality_tier"] = "usable"
obj.loc[obj["draw_ok_ratio"] < 0.60, "object_quality_tier"] = "weak_tracking_object"
obj.loc[obj["review_needed_ratio"] > 0.50, "object_quality_tier"] = "high_review_object"
obj.loc[obj["draw_ok_ratio"] >= 0.90, "object_quality_tier"] = "strong_tracking_object"

obj["object_review_priority_score"] = (
    1.5 * obj["review_needed_ratio"].fillna(0)
    + 1.5 * (1.0 - obj["draw_ok_ratio"].fillna(0))
)

safe_to_csv(obj, OUT_OBJECT_QA)

review_queue = clip[
    (clip["quality_tier"].isin(["limited_review_required", "challenging_low_tracking_support"]))
    | (clip["review_needed_ratio"] >= 0.35)
    | (clip["missing_ratio"] >= 0.20)
].copy()

review_queue = review_queue.sort_values(
    ["quality_tier", "qa_priority_score"],
    ascending=[True, False],
)

safe_to_csv(review_queue, OUT_REVIEW_QUEUE)

success_cases = clip[
    clip["quality_tier"] == "strong_visual_tracking_support"
].sort_values(
    ["draw_ok_ratio", "stable_tracklet_ratio"],
    ascending=[False, False],
).head(12)

challenging_cases = clip.sort_values(
    ["qa_priority_score", "missing_ratio", "review_needed_ratio"],
    ascending=[False, False, False],
).head(15)

safe_to_csv(success_cases, OUT_SUCCESS_CASES)
safe_to_csv(challenging_cases, OUT_CHALLENGING_CASES)

source_dist = rows["hybrid_source"].value_counts(dropna=False).reset_index()
source_dist.columns = ["hybrid_source", "row_count"]
source_dist["row_ratio"] = source_dist["row_count"] / max(1, len(rows))
safe_to_csv(source_dist, OUT_SOURCE_DISTRIBUTION)

tier_counts = clip["quality_tier"].value_counts().to_dict()
review_count = int(len(review_queue))
success_count = int(len(success_cases))
challenging_count = int(len(challenging_cases))

mean_stable = float(clip["stable_tracklet_ratio"].mean())
mean_fallback = float(clip["recall_fallback_ratio"].mean())
mean_missing = float(clip["missing_ratio"].mean())
mean_draw = float(clip["draw_ok_ratio"].mean())
mean_review = float(clip["review_needed_ratio"].mean())

hard_issues = []
warnings = []

if len(rows) != int(clip["total_object_frame_rows"].sum()):
    warnings.append({
        "item": "row_count_consistency",
        "issue_type": "warning_row_count_mismatch",
        "issue_detail": "Hybrid row count does not equal clip summary total row sum.",
        "severity": "warning",
    })

if mean_draw < 0.85:
    warnings.append({
        "item": "draw_ok_ratio",
        "issue_type": "warning_mean_draw_ok_below_0_85",
        "issue_detail": f"Mean draw-ok ratio is {mean_draw:.4f}.",
        "severity": "warning",
    })

if mean_missing > 0.20:
    warnings.append({
        "item": "missing_ratio",
        "issue_type": "warning_mean_missing_above_0_20",
        "issue_detail": f"Mean missing ratio is {mean_missing:.4f}.",
        "severity": "warning",
    })

issues_df = pd.DataFrame(hard_issues + warnings, columns=["item", "issue_type", "issue_detail", "severity"])
safe_to_csv(issues_df, OUT_ISSUES)

ready = len(hard_issues) == 0

decision = pd.DataFrame([{
    "v52c_decision": "full_hybrid_tracking_qa_completed",
    "clip_count": int(len(clip)),
    "object_count": int(obj[["scan_frame_id", "final_box_id"]].drop_duplicates().shape[0]),
    "tracking_rows_total": int(len(rows)),
    "mean_stable_tracklet_ratio": mean_stable,
    "mean_recall_fallback_ratio": mean_fallback,
    "mean_missing_ratio": mean_missing,
    "mean_draw_ok_ratio": mean_draw,
    "mean_review_needed_ratio": mean_review,
    "strong_clip_count": int(tier_counts.get("strong_visual_tracking_support", 0)),
    "usable_clip_count": int(tier_counts.get("usable_with_review", 0)),
    "limited_clip_count": int(tier_counts.get("limited_review_required", 0)),
    "challenging_clip_count": int(tier_counts.get("challenging_low_tracking_support", 0)),
    "manual_review_queue_clip_count": review_count,
    "successful_case_count": success_count,
    "challenging_case_count": challenging_count,
    "hard_issue_count": int(len(hard_issues)),
    "warning_count": int(len(warnings)),
    "issue_count": int(len(issues_df)),
    "ready_for_v52d_tracking_integrated_visualizer": bool(ready),
    "generated_at": datetime.now().isoformat(timespec="seconds"),
}])

safe_to_csv(decision, OUT_DECISION)

report = []
report.append("# Week 8 v52c Full Hybrid Tracking QA Report\n")
report.append("## Decision\n")
report.append(f"- Decision: {decision.iloc[0]['v52c_decision']}")
report.append(f"- Ready for v52d tracking-integrated visualizer: {ready}\n")
report.append("## Global metrics\n")
report.append(f"- Clip count: {len(clip)}")
report.append(f"- Tracking rows total: {len(rows)}")
report.append(f"- Mean stable tracklet ratio: {mean_stable:.4f}")
report.append(f"- Mean recall fallback ratio: {mean_fallback:.4f}")
report.append(f"- Mean missing ratio: {mean_missing:.4f}")
report.append(f"- Mean draw-ok ratio: {mean_draw:.4f}")
report.append(f"- Mean review-needed ratio: {mean_review:.4f}\n")
report.append("## Quality tiers\n")
for k, v in tier_counts.items():
    report.append(f"- {k}: {v}")
report.append("\n## Interpretation\n")
report.append(
    "The full hybrid tracking layer provides strong visual tracking support for many clips, "
    "while preserving fallback and missing cases explicitly for manual review. "
    "It should be used as a tracking-assisted validation layer, not as fully automatic ground-truth replacement."
)
report.append("\n## Output files\n")
report.append(f"- Clip QA: {OUT_CLIP_QA}")
report.append(f"- Object QA: {OUT_OBJECT_QA}")
report.append(f"- Manual review queue: {OUT_REVIEW_QUEUE}")
report.append(f"- Successful cases: {OUT_SUCCESS_CASES}")
report.append(f"- Challenging cases: {OUT_CHALLENGING_CASES}")

OUT_REPORT.write_text("\n".join(report))

note = []
note.append("# Week 8 v52c Full Hybrid Tracking QA\n")
note.append("## Summary\n")
note.append(f"- v52c decision: {decision.iloc[0]['v52c_decision']}")
note.append(f"- Clip count: {len(clip)}")
note.append(f"- Tracking rows total: {len(rows)}")
note.append(f"- Mean stable tracklet ratio: {mean_stable:.4f}")
note.append(f"- Mean recall fallback ratio: {mean_fallback:.4f}")
note.append(f"- Mean missing ratio: {mean_missing:.4f}")
note.append(f"- Mean draw-ok ratio: {mean_draw:.4f}")
note.append(f"- Mean review-needed ratio: {mean_review:.4f}")
note.append(f"- Manual review queue clips: {review_count}")
note.append(f"- Strong clips: {tier_counts.get('strong_visual_tracking_support', 0)}")
note.append(f"- Usable clips: {tier_counts.get('usable_with_review', 0)}")
note.append(f"- Limited clips: {tier_counts.get('limited_review_required', 0)}")
note.append(f"- Challenging clips: {tier_counts.get('challenging_low_tracking_support', 0)}")
note.append(f"- Hard issue count: {len(hard_issues)}")
note.append(f"- Warning count: {len(warnings)}")
note.append(f"- Ready for v52d tracking-integrated visualizer: {ready}\n")
note.append("## Interpretation\n")
note.append(
    "v52b3 is accepted as the full hybrid tracking-support layer. "
    "Stable tracklets are the primary evidence; recall fallback rows are visible but explicitly review-needed; missing rows are preserved."
)

OUT_NOTE.write_text("\n".join(note))

progress_row = pd.DataFrame([{
    "date": datetime.now().date().isoformat(),
    "stage": "v52c",
    "task_name": "Full hybrid tracking QA",
    "status": "PASS" if ready else "BLOCKED",
    "input_summary": str(V52B3_ROWS),
    "output_summary": str(OUT),
    "hard_issues": int(len(hard_issues)),
    "warnings": int(len(warnings)),
    "next_action": "v52d tracking-integrated visualizer with layer toggles" if ready else "Resolve v52c hard issues.",
}])

if OUT_PROGRESS.exists():
    old = pd.read_csv(OUT_PROGRESS)
    progress = pd.concat([old, progress_row], ignore_index=True)
else:
    progress = progress_row

safe_to_csv(progress, OUT_PROGRESS)

print("Saved:")
print(OUT_DECISION)
print(OUT_ISSUES)
print(OUT_CLIP_QA)
print(OUT_OBJECT_QA)
print(OUT_REVIEW_QUEUE)
print(OUT_SUCCESS_CASES)
print(OUT_CHALLENGING_CASES)
print(OUT_SOURCE_DISTRIBUTION)
print(OUT_REPORT)
print(OUT_NOTE)

print()
print("=== v52c decision ===")
print(decision.to_string(index=False))

print()
print("=== v52c issues ===")
if len(issues_df):
    print(issues_df.to_string(index=False))
else:
    print("No issues found.")

print()
print("=== v52c quality tier counts ===")
print(pd.Series(tier_counts).to_string())

print()
print("=== v52c challenging cases ===")
print(challenging_cases[[
    "scan_frame_id",
    "video_id",
    "stable_tracklet_ratio",
    "recall_fallback_ratio",
    "missing_ratio",
    "draw_ok_ratio",
    "review_needed_ratio",
    "quality_tier",
    "manual_review_recommendation",
]].to_string(index=False))
