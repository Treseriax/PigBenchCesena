from pathlib import Path
from datetime import datetime
import csv
import pandas as pd


ROOT = Path.home() / "PigBench"
W8 = ROOT / "Week8_Ground_Truth_Validation_Behaviour_Dataset_Preparation"

V52C_CLIP_QA = W8 / "outputs" / "v52c_full_hybrid_tracking_qa" / "week8_v52c_clip_quality_assessment.csv"
V52C_REVIEW_QUEUE = W8 / "outputs" / "v52c_full_hybrid_tracking_qa" / "week8_v52c_manual_review_queue.csv"
V52C_SUCCESS = W8 / "outputs" / "v52c_full_hybrid_tracking_qa" / "week8_v52c_successful_cases.csv"
V52C_CHALLENGING = W8 / "outputs" / "v52c_full_hybrid_tracking_qa" / "week8_v52c_challenging_cases.csv"
V52D_NOTES = W8 / "validation" / "week8_v52d_visual_validation_notes.csv"

OUT = W8 / "outputs" / "v52e_manual_visual_validation_protocol"
DOCS = W8 / "docs"
REPORTS = W8 / "reports"
NOTES = W8 / "notes"
PROGRESS = W8 / "progress"

for p in [OUT, DOCS, REPORTS, NOTES, PROGRESS]:
    p.mkdir(parents=True, exist_ok=True)

OUT_REVIEW_PLAN = OUT / "week8_v52e_manual_review_plan.csv"
OUT_CHECKLIST = OUT / "week8_v52e_visual_validation_checklist.csv"
OUT_SPOTCHECK = OUT / "week8_v52e_strong_clip_spotcheck_plan.csv"
OUT_PRIORITY = OUT / "week8_v52e_priority_review_order.csv"
OUT_DECISION = OUT / "week8_v52e_decision_summary.csv"
OUT_ISSUES = OUT / "week8_v52e_issues.csv"
OUT_PROTOCOL = DOCS / "week8_v52e_manual_visual_validation_protocol.md"
OUT_REPORT = REPORTS / "week8_v52e_manual_visual_validation_protocol_report.md"
OUT_NOTE = NOTES / "week8_v52e_manual_visual_validation_protocol_notes.md"
OUT_PROGRESS = PROGRESS / "week8_experiment_progress_log.csv"


def safe_to_csv(df, path):
    df.to_csv(path, index=False, quoting=csv.QUOTE_ALL, escapechar="\\", lineterminator="\n")


issues = []

for p in [V52C_CLIP_QA, V52C_REVIEW_QUEUE, V52C_SUCCESS, V52C_CHALLENGING]:
    if not p.exists():
        issues.append({
            "item": str(p),
            "issue_type": "hard_missing_required_input",
            "issue_detail": "Required v52c QA artifact is missing.",
            "severity": "hard",
        })

if issues:
    issues_df = pd.DataFrame(issues)
    safe_to_csv(issues_df, OUT_ISSUES)
    decision = pd.DataFrame([{
        "v52e_decision": "manual_validation_protocol_blocked_missing_inputs",
        "hard_issue_count": int((issues_df["severity"] == "hard").sum()),
        "ready_for_manual_visual_validation": False,
        "generated_at": datetime.now().isoformat(timespec="seconds"),
    }])
    safe_to_csv(decision, OUT_DECISION)
    print(decision.to_string(index=False))
    raise SystemExit(1)


clip_qa = pd.read_csv(V52C_CLIP_QA)
review_queue = pd.read_csv(V52C_REVIEW_QUEUE)
success = pd.read_csv(V52C_SUCCESS)
challenging = pd.read_csv(V52C_CHALLENGING)

for df in [clip_qa, review_queue, success, challenging]:
    for c in [
        "stable_tracklet_ratio",
        "recall_fallback_ratio",
        "missing_ratio",
        "draw_ok_ratio",
        "review_needed_ratio",
        "qa_priority_score",
    ]:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")


# Priority order: challenging/limited/review-heavy clips first.
priority = clip_qa.copy()
priority["manual_priority_level"] = "low_spot_check"

priority.loc[
    (priority["quality_tier"] == "challenging_low_tracking_support")
    | (priority["missing_ratio"] >= 0.35)
    | (priority["review_needed_ratio"] >= 0.55),
    "manual_priority_level"
] = "high_priority"

priority.loc[
    (priority["manual_priority_level"] != "high_priority")
    & (
        (priority["quality_tier"] == "limited_review_required")
        | (priority["missing_ratio"] >= 0.20)
        | (priority["review_needed_ratio"] >= 0.35)
    ),
    "manual_priority_level"
] = "medium_priority"

priority.loc[
    (priority["manual_priority_level"] == "low_spot_check")
    & (priority["quality_tier"] == "usable_with_review"),
    "manual_priority_level"
] = "normal_review"

priority["manual_review_goal"] = priority["manual_priority_level"].map({
    "high_priority": "Inspect full clip carefully; save notes for identity switch, missing pig, wrong/fallback box, and behaviour uncertainty.",
    "medium_priority": "Inspect start/middle/end and any fallback-heavy moments; save notes if identity/box/behaviour issue appears.",
    "normal_review": "Inspect start/middle/end; verify fallback boxes do not break identity.",
    "low_spot_check": "Spot-check only; verify strong tracking remains visually aligned.",
})

priority_order = {
    "high_priority": 0,
    "medium_priority": 1,
    "normal_review": 2,
    "low_spot_check": 3,
}

priority["priority_sort"] = priority["manual_priority_level"].map(priority_order).fillna(9)
priority = priority.sort_values(
    ["priority_sort", "qa_priority_score", "missing_ratio", "review_needed_ratio"],
    ascending=[True, False, False, False],
).drop(columns=["priority_sort"])

safe_to_csv(priority, OUT_PRIORITY)


# Review plan: all high/medium/normal review clips + 12 strong spot-checks.
review_plan_parts = []

main_review = priority[priority["manual_priority_level"].isin(["high_priority", "medium_priority", "normal_review"])].copy()
main_review["review_set"] = "main_review_queue"
review_plan_parts.append(main_review)

spotcheck = priority[
    (priority["manual_priority_level"] == "low_spot_check")
    & (priority["quality_tier"] == "strong_visual_tracking_support")
].copy().head(12)

spotcheck["review_set"] = "strong_clip_spot_check"
review_plan_parts.append(spotcheck)

review_plan = pd.concat(review_plan_parts, ignore_index=True)

review_plan["recommended_check_times_sec"] = "0, 2.5, 5.0, 7.5, 9.5"
review_plan["required_note_if"] = (
    "bbox_wrong; identity_switch; identity_uncertain; missing_pig; false_positive; behaviour_uncertain; fallback_box_unreliable"
)

safe_to_csv(review_plan, OUT_REVIEW_PLAN)
safe_to_csv(spotcheck, OUT_SPOTCHECK)


# Per-clip checklist.
checklist_rows = []

for _, r in review_plan.iterrows():
    scan = r["scan_frame_id"]
    video = r["video_id"]
    priority_level = r["manual_priority_level"]
    review_set = r["review_set"]

    for time_sec in [0.0, 2.5, 5.0, 7.5, 9.5]:
        checklist_rows.append({
            "scan_frame_id": scan,
            "video_id": video,
            "review_set": review_set,
            "manual_priority_level": priority_level,
            "target_time_sec": time_sec,
            "check_anchor_gt_alignment": "",
            "check_stable_tracklet_alignment": "",
            "check_recall_fallback_alignment": "",
            "check_identity_colour_consistency": "",
            "check_behaviour_label_plausibility": "",
            "check_missing_or_false_positive": "",
            "note_saved_in_v52d_interface": "",
            "manual_status": "",
            "manual_comment": "",
        })

checklist = pd.DataFrame(checklist_rows)
safe_to_csv(checklist, OUT_CHECKLIST)


# Prepare notes CSV if needed.
if not V52D_NOTES.exists():
    safe_to_csv(pd.DataFrame(columns=[
        "created_at",
        "scan_frame_id",
        "video_id",
        "time_sec",
        "frame_index_in_clip",
        "issue_type",
        "severity",
        "note",
    ]), V52D_NOTES)


high_count = int((review_plan["manual_priority_level"] == "high_priority").sum())
medium_count = int((review_plan["manual_priority_level"] == "medium_priority").sum())
normal_count = int((review_plan["manual_priority_level"] == "normal_review").sum())
spot_count = int((review_plan["review_set"] == "strong_clip_spot_check").sum())

issues_df = pd.DataFrame(issues, columns=["item", "issue_type", "issue_detail", "severity"])
safe_to_csv(issues_df, OUT_ISSUES)

decision = pd.DataFrame([{
    "v52e_decision": "manual_visual_validation_protocol_created",
    "total_clip_count": int(len(clip_qa)),
    "review_plan_clip_count": int(review_plan["scan_frame_id"].nunique()),
    "checklist_rows": int(len(checklist)),
    "high_priority_clip_count": high_count,
    "medium_priority_clip_count": medium_count,
    "normal_review_clip_count": normal_count,
    "strong_spotcheck_clip_count": spot_count,
    "manual_notes_csv": str(V52D_NOTES),
    "hard_issue_count": 0,
    "warning_count": 0,
    "issue_count": 0,
    "ready_for_manual_visual_validation": True,
    "generated_at": datetime.now().isoformat(timespec="seconds"),
}])

safe_to_csv(decision, OUT_DECISION)


protocol = f"""# Week 8 Manual Visual Validation Protocol

## Purpose

This protocol validates the 72 annotated observation clips before behaviour classification. The goal is not to annotate new raw videos. The goal is to verify the existing pig-colour-behaviour associations and the hybrid tracking support layer.

## Data scope

- Raw video archive is larger than the validated dataset.
- Week 8 validation uses the 72 annotated scanframe clips.
- Each scanframe corresponds to a 10-second observation window.
- Anchor GT remains the primary reference.
- Hybrid tracking is a validation support layer:
  - stable_tracklet = primary temporal evidence
  - recall_fallback = visible but review-needed
  - missing = preserved, not hidden

## Review priority

1. High priority clips:
   - challenging tracking support
   - high missing ratio
   - high review-needed ratio

2. Medium priority clips:
   - limited tracking support
   - moderate missing/review ratio

3. Normal review clips:
   - usable with review

4. Strong spot-check clips:
   - strong tracking support
   - only start/middle/end check required

## What to check in the v52d visualizer

For each priority clip:

- Anchor GT alignment at the annotation frame
- Stable tracklet boxes remain on the correct pig
- Recall fallback boxes are not misleading
- Missing pigs are correctly not drawn
- Colour identity remains plausible
- Behaviour label is plausible for the visible pig
- No obvious false positive outside the pen
- No identity switch between pigs

## Required note types

Use the v52d interface note panel with one of:

- bbox_ok
- bbox_wrong
- identity_uncertain
- identity_switch
- behaviour_uncertain
- missing_pig
- false_positive
- other

## Recommended inspection times

For each reviewed clip, inspect approximately:

- 0.0 s
- 2.5 s
- 5.0 s
- 7.5 s
- 9.5 s

If the clip has a visible issue, pause exactly where it occurs and save a note.

## Outputs

- Review plan: `{OUT_REVIEW_PLAN}`
- Checklist: `{OUT_CHECKLIST}`
- Strong spot-check plan: `{OUT_SPOTCHECK}`
- Priority order: `{OUT_PRIORITY}`
- Notes saved by interface: `{V52D_NOTES}`

## Classification policy

Classification can proceed after this validation phase as a baseline/pilot experiment. The dataset is not claimed to support a production-level final classifier. Rare behaviour classes should remain limited/report-only unless additional annotation is later requested.
"""

OUT_PROTOCOL.write_text(protocol)

OUT_REPORT.write_text(
    "# Week 8 v52e Manual Visual Validation Protocol Report\n\n"
    f"Decision: {decision.iloc[0]['v52e_decision']}\n\n"
    f"- Total clips: {len(clip_qa)}\n"
    f"- Review plan clips: {review_plan['scan_frame_id'].nunique()}\n"
    f"- Checklist rows: {len(checklist)}\n"
    f"- High priority clips: {high_count}\n"
    f"- Medium priority clips: {medium_count}\n"
    f"- Normal review clips: {normal_count}\n"
    f"- Strong spot-check clips: {spot_count}\n\n"
    "The protocol focuses on validating existing annotated clips, not adding new raw-video annotations.\n"
)

OUT_NOTE.write_text(
    "# Week 8 v52e Manual Visual Validation Protocol\n\n"
    "## Summary\n\n"
    f"- v52e decision: {decision.iloc[0]['v52e_decision']}\n"
    f"- Total clips: {len(clip_qa)}\n"
    f"- Review plan clips: {review_plan['scan_frame_id'].nunique()}\n"
    f"- Checklist rows: {len(checklist)}\n"
    f"- High priority clips: {high_count}\n"
    f"- Medium priority clips: {medium_count}\n"
    f"- Normal review clips: {normal_count}\n"
    f"- Strong spot-check clips: {spot_count}\n"
    f"- Manual notes CSV: {V52D_NOTES}\n"
    f"- Ready for manual visual validation: True\n\n"
    "## Next\n\n"
    "Use the v52d visualizer to review the priority clips and save notes. Start from the high-priority clips, then medium-priority, then spot-check strong clips.\n"
)

progress_row = pd.DataFrame([{
    "date": datetime.now().date().isoformat(),
    "stage": "v52e",
    "task_name": "Manual visual validation protocol",
    "status": "PASS",
    "input_summary": str(V52C_CLIP_QA),
    "output_summary": str(OUT),
    "hard_issues": 0,
    "warnings": 0,
    "next_action": "Manual validation in v52d visualizer, then v52f validation notes summary.",
}])

if OUT_PROGRESS.exists():
    old = pd.read_csv(OUT_PROGRESS)
    progress = pd.concat([old, progress_row], ignore_index=True)
else:
    progress = progress_row

safe_to_csv(progress, OUT_PROGRESS)

print("Saved:")
print(OUT_REVIEW_PLAN)
print(OUT_CHECKLIST)
print(OUT_SPOTCHECK)
print(OUT_PRIORITY)
print(OUT_DECISION)
print(OUT_ISSUES)
print(OUT_PROTOCOL)
print(OUT_REPORT)
print(OUT_NOTE)

print()
print("=== v52e decision ===")
print(decision.to_string(index=False))

print()
print("=== high priority clips ===")
print(review_plan[review_plan["manual_priority_level"] == "high_priority"][[
    "scan_frame_id",
    "video_id",
    "quality_tier",
    "draw_ok_ratio",
    "missing_ratio",
    "review_needed_ratio",
    "manual_review_recommendation",
]].to_string(index=False))

print()
print("=== review plan first 25 ===")
print(review_plan[[
    "scan_frame_id",
    "video_id",
    "review_set",
    "manual_priority_level",
    "quality_tier",
    "draw_ok_ratio",
    "missing_ratio",
    "review_needed_ratio",
]].head(25).to_string(index=False))
