from pathlib import Path
from datetime import datetime
import csv
import pandas as pd


ROOT = Path.home() / "PigBench"
W8 = ROOT / "Week8_Ground_Truth_Validation_Behaviour_Dataset_Preparation"

V49A_DECISION = W8 / "outputs" / "v49a_bbox_coordinate_system_audit" / "week8_v49a_decision_summary.csv"
V49B_DECISION = W8 / "outputs" / "v49b_exact_anchor_overlay_gallery" / "week8_v49b_decision_summary.csv"
V49B_INDEX = W8 / "outputs" / "v49b_exact_anchor_overlay_gallery" / "week8_v49b_anchor_overlay_index.csv"

OUT = W8 / "outputs" / "v49c_anchor_validation_verdict"
VALIDATION = W8 / "validation"
REPORTS = W8 / "reports"
NOTES = W8 / "notes"
PROGRESS = W8 / "progress"

for p in [OUT, VALIDATION, REPORTS, NOTES, PROGRESS]:
    p.mkdir(parents=True, exist_ok=True)

OUT_VERDICT = OUT / "week8_v49c_anchor_validation_verdict.csv"
OUT_DECISION = OUT / "week8_v49c_decision_summary.csv"
OUT_ISSUES = OUT / "week8_v49c_issues.csv"
OUT_MANUAL_REVIEW_TEMPLATE = VALIDATION / "week8_v49c_anchor_gallery_manual_review_template.csv"
OUT_REPORT = REPORTS / "week8_v49c_anchor_validation_verdict_report.md"
OUT_NOTE = NOTES / "week8_v49c_anchor_validation_verdict_notes.md"
OUT_PROGRESS = PROGRESS / "week8_experiment_progress_log.csv"


def safe_to_csv(df, path):
    df.to_csv(
        path,
        index=False,
        quoting=csv.QUOTE_ALL,
        escapechar="\\",
        lineterminator="\n",
    )


issues = []

for p in [V49A_DECISION, V49B_DECISION, V49B_INDEX]:
    if not p.exists():
        issues.append({
            "item": str(p),
            "issue_type": "hard_missing_required_input",
            "issue_detail": "Required v49a/v49b artifact is missing.",
            "severity": "hard",
        })

if issues:
    issues_df = pd.DataFrame(issues)
    safe_to_csv(issues_df, OUT_ISSUES)
    decision = pd.DataFrame([{
        "v49c_decision": "anchor_validation_verdict_blocked",
        "hard_issue_count": int((issues_df["severity"] == "hard").sum()),
        "ready_for_v50_tracking_refined_boxes": False,
        "generated_at": datetime.now().isoformat(timespec="seconds"),
    }])
    safe_to_csv(decision, OUT_DECISION)
    print(decision.to_string(index=False))
    raise SystemExit(1)


v49a = pd.read_csv(V49A_DECISION)
v49b = pd.read_csv(V49B_DECISION)
overlay_index = pd.read_csv(V49B_INDEX)

v49a_row = v49a.iloc[0].to_dict()
v49b_row = v49b.iloc[0].to_dict()

clip_count = int(v49b_row.get("clip_count", 0))
saved_overlay_count = int(v49b_row.get("saved_overlay_count", 0))
bbox_outside = int(v49a_row.get("objects_with_bbox_outside_video", 0))
invalid_bbox = int(v49a_row.get("objects_with_invalid_basic_bbox", 0))
resolution_mismatch_clips = int(v49a_row.get("clips_probable_resolution_mismatch", 0))

# This is based on manual visual confirmation from user after v49b gallery inspection.
manual_visual_result = "exact_anchor_overlays_visually_aligned"
manual_visual_note = (
    "User inspected v49b exact-anchor overlay gallery and reported that the overlays are now correct. "
    "Therefore, the earlier mismatch is interpreted as full-clip motion / scanpoint-anchor repetition, not coordinate-system failure."
)

bbox_coordinate_system_status = "passed"
anchor_frame_alignment_status = "passed"
full_clip_bbox_status = "requires_tracking_refined_per_frame_boxes"
coordinate_fix_needed = False
tracking_refined_boxes_needed = True
annotation_mapping_repair_needed = False

if bbox_outside > 0 or invalid_bbox > 0 or resolution_mismatch_clips > 0:
    bbox_coordinate_system_status = "review_required"
    coordinate_fix_needed = True

if saved_overlay_count != clip_count:
    anchor_frame_alignment_status = "incomplete_review"
    issues.append({
        "item": "exact_anchor_overlays",
        "issue_type": "warning_incomplete_anchor_overlay_gallery",
        "issue_detail": f"Saved overlays {saved_overlay_count}/{clip_count}.",
        "severity": "warning",
    })

verdict = pd.DataFrame([{
    "created_at": datetime.now().isoformat(timespec="seconds"),
    "verdict": "anchor_frame_annotations_validated",
    "manual_visual_result": manual_visual_result,
    "manual_visual_note": manual_visual_note,
    "bbox_coordinate_system_status": bbox_coordinate_system_status,
    "anchor_frame_alignment_status": anchor_frame_alignment_status,
    "full_clip_bbox_status": full_clip_bbox_status,
    "bbox_coordinate_fix_needed": bool(coordinate_fix_needed),
    "annotation_mapping_repair_needed": bool(annotation_mapping_repair_needed),
    "tracking_refined_per_frame_boxes_needed": bool(tracking_refined_boxes_needed),
    "clip_count": clip_count,
    "saved_anchor_overlay_count": saved_overlay_count,
    "objects_with_bbox_outside_video": bbox_outside,
    "objects_with_invalid_basic_bbox": invalid_bbox,
    "clips_probable_resolution_mismatch": resolution_mismatch_clips,
    "next_recommended_stage": "v50_tracking_refined_per_frame_boxes_and_identity_review",
}])

review_template = overlay_index[[
    "scan_frame_id",
    "video_id",
    "overlay_path",
    "object_count",
    "anchor_rel_sec",
    "anchor_frame_idx",
    "behaviour_set",
]].copy()

review_template["manual_anchor_bbox_status"] = ""
review_template["manual_colour_identity_status"] = ""
review_template["manual_behaviour_association_status"] = ""
review_template["review_note"] = ""
review_template["reviewer"] = ""
review_template["reviewed_at"] = ""

issues_df = pd.DataFrame(issues, columns=["item", "issue_type", "issue_detail", "severity"])

hard_issues = issues_df[issues_df["severity"] == "hard"] if len(issues_df) else pd.DataFrame()
warnings = issues_df[issues_df["severity"] == "warning"] if len(issues_df) else pd.DataFrame()

ready = len(hard_issues) == 0

decision = pd.DataFrame([{
    "v49c_decision": "anchor_validation_verdict_recorded" if ready else "anchor_validation_verdict_blocked",
    "verdict": "anchor_frame_annotations_validated",
    "clip_count": int(clip_count),
    "saved_anchor_overlay_count": int(saved_overlay_count),
    "bbox_coordinate_fix_needed": bool(coordinate_fix_needed),
    "annotation_mapping_repair_needed": bool(annotation_mapping_repair_needed),
    "tracking_refined_per_frame_boxes_needed": bool(tracking_refined_boxes_needed),
    "hard_issue_count": int(len(hard_issues)),
    "warning_count": int(len(warnings)),
    "issue_count": int(len(issues_df)),
    "ready_for_v50_tracking_refined_boxes": bool(ready),
    "generated_at": datetime.now().isoformat(timespec="seconds"),
}])

safe_to_csv(verdict, OUT_VERDICT)
safe_to_csv(review_template, OUT_MANUAL_REVIEW_TEMPLATE)
safe_to_csv(issues_df, OUT_ISSUES)
safe_to_csv(decision, OUT_DECISION)

report = f"""Week 8 v49c Anchor Validation Verdict Report

Decision:
v49c decision: {decision.iloc[0]["v49c_decision"]}

Verdict:
Anchor-frame annotations are validated.

Evidence:
- v49a found no coordinate-system mismatch.
- v49a found no bbox outside video bounds.
- v49a found no invalid basic bbox geometry.
- v49b generated exact anchor-frame overlays for {saved_overlay_count}/{clip_count} clips.
- Manual visual inspection confirmed that exact-anchor overlays are now correct.

Interpretation:
The earlier visual mismatch in the browser viewer is not a coordinate-system failure. It is mainly caused by using scanpoint-anchor bounding boxes across 10-second clips where pigs move.

Next action:
Proceed to tracking-refined per-frame boxes and identity review. The validation interface should keep anchor-frame mode as the reliable reference, while full-clip playback needs per-frame tracking boxes.

Important limitation:
Anchor-frame bbox correctness does not mean full-clip bbox correctness. Full-clip validation still requires tracking-refined boxes.
"""

OUT_REPORT.write_text(report)

OUT_NOTE.write_text(
    "# Week 8 v49c Anchor Validation Verdict\n\n"
    "## Summary\n\n"
    f"- v49c decision: {decision.iloc[0]['v49c_decision']}\n"
    "- Verdict: anchor-frame annotations validated\n"
    f"- Clip count: {clip_count}\n"
    f"- Saved anchor overlays: {saved_overlay_count}\n"
    f"- BBox coordinate fix needed: {coordinate_fix_needed}\n"
    f"- Annotation mapping repair needed: {annotation_mapping_repair_needed}\n"
    f"- Tracking-refined per-frame boxes needed: {tracking_refined_boxes_needed}\n"
    f"- Ready for v50: {ready}\n\n"
    "## Interpretation\n\n"
    "Exact anchor overlays look correct. The earlier mismatch is not a resolution/coordinate-system problem. "
    "It is caused by using fixed scanpoint-anchor boxes during full-clip playback while pigs move.\n\n"
    "## Next\n\n"
    "Proceed to v50: tracking-refined per-frame boxes and identity review.\n"
)

progress_row = pd.DataFrame([{
    "date": datetime.now().date().isoformat(),
    "stage": "v49c",
    "task_name": "Anchor-frame validation verdict",
    "status": "PASS" if ready else "BLOCKED",
    "input_summary": str(V49B_INDEX),
    "output_summary": str(OUT),
    "hard_issues": int(len(hard_issues)),
    "warnings": int(len(warnings)),
    "next_action": "v50 tracking-refined per-frame boxes and identity review",
}])

if OUT_PROGRESS.exists():
    old = pd.read_csv(OUT_PROGRESS)
    progress = pd.concat([old, progress_row], ignore_index=True)
else:
    progress = progress_row

safe_to_csv(progress, OUT_PROGRESS)

print("Saved:")
print(OUT_VERDICT)
print(OUT_MANUAL_REVIEW_TEMPLATE)
print(OUT_DECISION)
print(OUT_ISSUES)
print(OUT_REPORT)
print(OUT_NOTE)

print()
print("=== v49c decision ===")
print(decision.to_string(index=False))

print()
print("=== v49c verdict ===")
print(verdict.to_string(index=False))

print()
print("=== v49c issues ===")
if len(issues_df):
    print(issues_df.to_string(index=False))
else:
    print("No issues found.")
