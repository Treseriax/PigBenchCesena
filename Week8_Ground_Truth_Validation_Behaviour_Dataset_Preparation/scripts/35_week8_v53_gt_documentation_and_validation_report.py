from pathlib import Path
from datetime import datetime
import csv
import json
import pandas as pd


ROOT = Path.home() / "PigBench"
W8 = ROOT / "Week8_Ground_Truth_Validation_Behaviour_Dataset_Preparation"

V43C_DECISION = W8 / "outputs" / "v43c_final_input_lock" / "week8_v43c_decision_summary.csv"
V44_SCHEMA_DOC = W8 / "docs" / "week8_ground_truth_json_format_specification_v44.md"
V44_RULES_DOC = W8 / "docs" / "week8_annotation_rules_v44.md"

V45_CLIP_JSON = W8 / "outputs" / "propagated_ground_truth" / "v45_label_propagation" / "week8_v45_clip_level_ground_truth.json"
V45_CLIP_OBJECTS = W8 / "outputs" / "propagated_ground_truth" / "v45_label_propagation" / "week8_v45_clip_object_propagated_annotations.csv"

V46_DECISION = W8 / "outputs" / "propagated_ground_truth" / "v46_propagation_qa" / "week8_v46_decision_summary.csv"

V52B3_DECISION = W8 / "outputs" / "v52b3_hybrid_tracking_full" / "week8_v52b3_decision_summary.csv"
V52B3_CLIP_SUMMARY = W8 / "outputs" / "v52b3_hybrid_tracking_full" / "week8_v52b3_clip_summary.csv"
V52B3_TRACKS = W8 / "outputs" / "v52b3_hybrid_tracking_full" / "week8_v52b3_hybrid_tracking_rows.csv"

V52C_DECISION = W8 / "outputs" / "v52c_full_hybrid_tracking_qa" / "week8_v52c_decision_summary.csv"
V52C_CLIP_QA = W8 / "outputs" / "v52c_full_hybrid_tracking_qa" / "week8_v52c_clip_quality_assessment.csv"
V52C_REVIEW_QUEUE = W8 / "outputs" / "v52c_full_hybrid_tracking_qa" / "week8_v52c_manual_review_queue.csv"
V52C_SUCCESS = W8 / "outputs" / "v52c_full_hybrid_tracking_qa" / "week8_v52c_successful_cases.csv"
V52C_CHALLENGING = W8 / "outputs" / "v52c_full_hybrid_tracking_qa" / "week8_v52c_challenging_cases.csv"

V52D_NOTES = W8 / "validation" / "week8_v52d_visual_validation_notes.csv"
V52E_PROTOCOL = W8 / "docs" / "week8_v52e_manual_visual_validation_protocol.md"
V52F_DECISION = W8 / "outputs" / "v52f_manual_validation_notes_summary" / "week8_v52f_decision_summary.csv"
V52F_CLIP_STATUS = W8 / "outputs" / "v52f_manual_validation_notes_summary" / "week8_v52f_manual_validation_clip_status.csv"

OUT = W8 / "outputs" / "v53_gt_documentation_and_validation_report"
DOCS = W8 / "docs"
REPORTS = W8 / "reports"
NOTES = W8 / "notes"
PROGRESS = W8 / "progress"

for p in [OUT, DOCS, REPORTS, NOTES, PROGRESS]:
    p.mkdir(parents=True, exist_ok=True)

OUT_DATASET_SUMMARY = OUT / "week8_v53_dataset_summary_metrics.csv"
OUT_SOURCE_MANIFEST = OUT / "week8_v53_source_manifest.csv"
OUT_BEHAVIOUR_DISTRIBUTION = OUT / "week8_v53_behaviour_distribution.csv"
OUT_VIDEO_SCANFRAME_SUMMARY = OUT / "week8_v53_video_scanframe_summary.csv"
OUT_TRACKING_TIER_SUMMARY = OUT / "week8_v53_tracking_quality_tier_summary.csv"
OUT_MANUAL_VALIDATION_SUMMARY = OUT / "week8_v53_manual_validation_summary.csv"
OUT_LIMITATIONS = OUT / "week8_v53_limitations_and_claims.csv"
OUT_DECISION = OUT / "week8_v53_decision_summary.csv"
OUT_ISSUES = OUT / "week8_v53_issues.csv"

OUT_DATASET_DOC = DOCS / "week8_v53_ground_truth_dataset_documentation.md"
OUT_VALIDATION_DOC = DOCS / "week8_v53_validation_report.md"
OUT_REPORT = REPORTS / "week8_v53_gt_documentation_and_validation_report.md"
OUT_NOTE = NOTES / "week8_v53_gt_documentation_and_validation_report_notes.md"
OUT_PROGRESS = PROGRESS / "week8_experiment_progress_log.csv"


def safe_to_csv(df, path):
    df.to_csv(path, index=False, quoting=csv.QUOTE_ALL, escapechar="\\", lineterminator="\n")


def read_first_row(path):
    if path.exists():
        df = pd.read_csv(path)
        if len(df):
            return df.iloc[0].to_dict()
    return {}


def clean_str(x):
    if pd.isna(x):
        return ""
    s = str(x).strip()
    if s.lower() in ["nan", "none", "null"]:
        return ""
    return s


issues = []

required = [
    V45_CLIP_JSON,
    V45_CLIP_OBJECTS,
    V52B3_DECISION,
    V52B3_CLIP_SUMMARY,
    V52B3_TRACKS,
    V52C_DECISION,
    V52C_CLIP_QA,
    V52C_REVIEW_QUEUE,
    V52F_DECISION,
    V52F_CLIP_STATUS,
]

for p in required:
    if not p.exists():
        issues.append({
            "item": str(p),
            "issue_type": "hard_missing_required_input",
            "issue_detail": "Required v53 input is missing.",
            "severity": "hard",
        })

optional = [
    V43C_DECISION,
    V44_SCHEMA_DOC,
    V44_RULES_DOC,
    V46_DECISION,
    V52C_SUCCESS,
    V52C_CHALLENGING,
    V52D_NOTES,
    V52E_PROTOCOL,
]

for p in optional:
    if not p.exists():
        issues.append({
            "item": str(p),
            "issue_type": "warning_missing_optional_reference",
            "issue_detail": "Optional documentation/reference input is missing.",
            "severity": "warning",
        })

if any(i["severity"] == "hard" for i in issues):
    issues_df = pd.DataFrame(issues)
    safe_to_csv(issues_df, OUT_ISSUES)
    decision = pd.DataFrame([{
        "v53_decision": "gt_documentation_and_validation_report_blocked_missing_inputs",
        "hard_issue_count": int((issues_df["severity"] == "hard").sum()),
        "warning_count": int((issues_df["severity"] == "warning").sum()),
        "ready_for_v54_final_delivery_package": False,
        "generated_at": datetime.now().isoformat(timespec="seconds"),
    }])
    safe_to_csv(decision, OUT_DECISION)
    print(decision.to_string(index=False))
    raise SystemExit(1)


clip_data = json.loads(V45_CLIP_JSON.read_text())
clips = clip_data.get("clips", [])
clip_objects = pd.read_csv(V45_CLIP_OBJECTS)
b3_decision = read_first_row(V52B3_DECISION)
b3_clip = pd.read_csv(V52B3_CLIP_SUMMARY)
b3_tracks = pd.read_csv(V52B3_TRACKS)
c_decision = read_first_row(V52C_DECISION)
c_clip = pd.read_csv(V52C_CLIP_QA)
f_decision = read_first_row(V52F_DECISION)
f_status = pd.read_csv(V52F_CLIP_STATUS)

for df in [clip_objects, b3_clip, c_clip, f_status]:
    if "scan_frame_id" in df.columns:
        df["scan_frame_id"] = df["scan_frame_id"].astype(str).map(clean_str)

for col in ["behaviour_code", "visual_marker_colour", "behaviour_pig_id", "video_id"]:
    if col in clip_objects.columns:
        clip_objects[col] = clip_objects[col].astype(str).map(clean_str)

clip_count = len(clips)
scanframe_count = len({clean_str(c.get("scan_frame_id")) for c in clips})
source_video_count = len({clean_str(c.get("video_id")) for c in clips})
clip_object_count = len(clip_objects)

behaviour_col = "behaviour_code" if "behaviour_code" in clip_objects.columns else None
if behaviour_col:
    usable_behaviour_objects = int((clip_objects[behaviour_col].astype(str).map(clean_str) != "").sum())
    missing_behaviour_objects = int((clip_objects[behaviour_col].astype(str).map(clean_str) == "").sum())
else:
    usable_behaviour_objects = 0
    missing_behaviour_objects = clip_object_count

tracking_rows_total = int(len(b3_tracks))
mean_stable = float(b3_decision.get("mean_stable_tracklet_ratio", c_decision.get("mean_stable_tracklet_ratio", 0)))
mean_fallback = float(b3_decision.get("mean_recall_fallback_ratio", c_decision.get("mean_recall_fallback_ratio", 0)))
mean_missing = float(b3_decision.get("mean_missing_ratio", c_decision.get("mean_missing_ratio", 0)))
mean_draw = float(b3_decision.get("mean_draw_ok_ratio", c_decision.get("mean_draw_ok_ratio", 0)))
mean_review = float(b3_decision.get("mean_review_needed_ratio", c_decision.get("mean_review_needed_ratio", 0)))

strong_clip_count = int(c_decision.get("strong_clip_count", 0))
usable_clip_count = int(c_decision.get("usable_clip_count", 0))
limited_clip_count = int(c_decision.get("limited_clip_count", 0))
challenging_clip_count = int(c_decision.get("challenging_clip_count", 0))
manual_review_queue_clip_count = int(c_decision.get("manual_review_queue_clip_count", 0))

manual_note_count = int(f_decision.get("manual_note_count", 0))
high_priority_clip_count = int(f_decision.get("high_priority_clip_count", 0))
high_priority_reviewed = int(f_decision.get("high_priority_reviewed_clip_count", 0))
high_priority_accepted = int(f_decision.get("high_priority_accepted_clip_count", 0))
manual_issue_found = int(f_decision.get("manual_issue_found_clip_count", 0))

dataset_summary = pd.DataFrame([
    {"metric": "raw_dataset_scope", "value": "larger Unibo raw video archive; not all raw videos are part of validated GT dataset"},
    {"metric": "validated_scanframe_count", "value": scanframe_count},
    {"metric": "validated_clip_count", "value": clip_count},
    {"metric": "source_video_count_in_validated_dataset", "value": source_video_count},
    {"metric": "clip_duration_policy", "value": "10-second observation window per scanframe"},
    {"metric": "clip_object_count", "value": clip_object_count},
    {"metric": "usable_behaviour_labelled_objects", "value": usable_behaviour_objects},
    {"metric": "missing_or_unmapped_behaviour_objects", "value": missing_behaviour_objects},
    {"metric": "frame_object_tracking_rows", "value": tracking_rows_total},
    {"metric": "mean_stable_tracklet_ratio", "value": f"{mean_stable:.4f}"},
    {"metric": "mean_recall_fallback_ratio", "value": f"{mean_fallback:.4f}"},
    {"metric": "mean_missing_ratio", "value": f"{mean_missing:.4f}"},
    {"metric": "mean_draw_ok_ratio", "value": f"{mean_draw:.4f}"},
    {"metric": "mean_review_needed_ratio", "value": f"{mean_review:.4f}"},
    {"metric": "strong_clip_count", "value": strong_clip_count},
    {"metric": "usable_clip_count", "value": usable_clip_count},
    {"metric": "limited_clip_count", "value": limited_clip_count},
    {"metric": "challenging_clip_count", "value": challenging_clip_count},
    {"metric": "manual_review_queue_clip_count", "value": manual_review_queue_clip_count},
    {"metric": "manual_note_count", "value": manual_note_count},
    {"metric": "high_priority_clip_count", "value": high_priority_clip_count},
    {"metric": "high_priority_reviewed_clip_count", "value": high_priority_reviewed},
    {"metric": "high_priority_accepted_clip_count", "value": high_priority_accepted},
    {"metric": "manual_issue_found_clip_count", "value": manual_issue_found},
])
safe_to_csv(dataset_summary, OUT_DATASET_SUMMARY)

source_manifest_rows = []
for name, path, required_flag, description in [
    ("v45_clip_level_ground_truth_json", V45_CLIP_JSON, True, "Clip-level propagated GT metadata."),
    ("v45_clip_object_annotations", V45_CLIP_OBJECTS, True, "Clip-object anchor GT, identity and behaviour propagation reference."),
    ("v52b3_hybrid_tracking_rows", V52B3_TRACKS, True, "Full hybrid frame-object tracking support layer."),
    ("v52b3_clip_summary", V52B3_CLIP_SUMMARY, True, "Full hybrid tracking per-clip summary."),
    ("v52c_clip_quality_assessment", V52C_CLIP_QA, True, "Tracking QA and quality tier per clip."),
    ("v52c_manual_review_queue", V52C_REVIEW_QUEUE, True, "Manual review queue from tracking QA."),
    ("v52f_manual_validation_clip_status", V52F_CLIP_STATUS, True, "Manual validation result summary per reviewed clip."),
    ("v44_json_schema_doc", V44_SCHEMA_DOC, False, "GT JSON format documentation."),
    ("v52e_manual_validation_protocol", V52E_PROTOCOL, False, "Manual visual validation protocol."),
]:
    source_manifest_rows.append({
        "source_name": name,
        "path": str(path),
        "exists": bool(path.exists()),
        "required": required_flag,
        "description": description,
    })
safe_to_csv(pd.DataFrame(source_manifest_rows), OUT_SOURCE_MANIFEST)

if behaviour_col:
    behaviour_distribution = (
        clip_objects.assign(
            behaviour_code_clean=clip_objects[behaviour_col].astype(str).map(clean_str).replace("", "MISSING_OR_UNMAPPED")
        )
        .groupby("behaviour_code_clean")
        .size()
        .reset_index(name="clip_object_count")
        .sort_values("clip_object_count", ascending=False)
    )
else:
    behaviour_distribution = pd.DataFrame(columns=["behaviour_code_clean", "clip_object_count"])
safe_to_csv(behaviour_distribution, OUT_BEHAVIOUR_DISTRIBUTION)

video_scanframe_summary = (
    b3_clip.groupby("video_id")
    .agg(
        scanframe_count=("scan_frame_id", "count"),
        first_scanframe=("scan_frame_id", "min"),
        last_scanframe=("scan_frame_id", "max"),
        total_object_frame_rows=("total_object_frame_rows", "sum"),
        mean_draw_ok_ratio=("draw_ok_ratio", "mean"),
        mean_missing_ratio=("missing_ratio", "mean"),
        mean_review_needed_ratio=("review_needed_ratio", "mean"),
    )
    .reset_index()
    .sort_values("video_id")
)
safe_to_csv(video_scanframe_summary, OUT_VIDEO_SCANFRAME_SUMMARY)

tracking_tier_summary = (
    c_clip.groupby("quality_tier")
    .agg(
        clip_count=("scan_frame_id", "count"),
        mean_draw_ok_ratio=("draw_ok_ratio", "mean"),
        mean_missing_ratio=("missing_ratio", "mean"),
        mean_review_needed_ratio=("review_needed_ratio", "mean"),
    )
    .reset_index()
    .sort_values("clip_count", ascending=False)
)
safe_to_csv(tracking_tier_summary, OUT_TRACKING_TIER_SUMMARY)

manual_validation_summary = (
    f_status.groupby(["manual_priority_level", "manual_status", "accepted_for_current_stage"])
    .size()
    .reset_index(name="clip_count")
    .sort_values(["manual_priority_level", "manual_status"])
)
safe_to_csv(manual_validation_summary, OUT_MANUAL_VALIDATION_SUMMARY)

claims_limitations = pd.DataFrame([
    {
        "type": "claim",
        "statement": "The Week 8 dataset is a validated 72-scanframe / 72-clip observation-window dataset, not the full raw video archive.",
        "report_wording": "The validated dataset consists of annotated observation windows selected from the larger Unibo raw video archive."
    },
    {
        "type": "claim",
        "statement": "Behaviour labels are propagated over the corresponding 10-second observation window.",
        "report_wording": "Each behaviour annotation is treated as applying to the full 10-second observation interval."
    },
    {
        "type": "claim",
        "statement": "Hybrid tracking is a validation support layer.",
        "report_wording": "Stable tracklets are used as primary temporal evidence and recall fallback rows are explicitly flagged for review."
    },
    {
        "type": "claim",
        "statement": "High-priority tracking QA clips were manually reviewed and accepted for the current stage.",
        "report_wording": "The high-priority validation clips were checked in the visualizer and marked as mostly correct."
    },
    {
        "type": "limitation",
        "statement": "The hybrid tracking layer is not claimed as fully automatic tracking ground truth.",
        "report_wording": "The tracking output supports visual validation and should not be interpreted as a fully automatic replacement for manually verified tracking ground truth."
    },
    {
        "type": "limitation",
        "statement": "The full 84-video raw archive is not fully manually annotated for behaviour classification.",
        "report_wording": "The remaining raw videos are outside the validated behaviour GT scope unless additional manual annotation is later performed."
    },
    {
        "type": "limitation",
        "statement": "Rare behaviour classes remain limited.",
        "report_wording": "Rare behaviours should be treated carefully in classification experiments and may require grouping, exclusion or report-only handling."
    },
    {
        "type": "limitation",
        "statement": "Classification can proceed as baseline/pilot, not as a production-level final classifier claim.",
        "report_wording": "The prepared dataset is suitable for baseline classification experiments, but not for claiming a robust production behaviour classifier."
    },
])
safe_to_csv(claims_limitations, OUT_LIMITATIONS)

issues_df = pd.DataFrame(issues, columns=["item", "issue_type", "issue_detail", "severity"])
safe_to_csv(issues_df, OUT_ISSUES)

hard_issue_count = int((issues_df["severity"] == "hard").sum()) if len(issues_df) else 0
warning_count = int((issues_df["severity"] == "warning").sum()) if len(issues_df) else 0

ready = (
    hard_issue_count == 0
    and scanframe_count == 72
    and clip_count == 72
    and clip_object_count == 429
    and high_priority_reviewed == high_priority_clip_count
    and manual_issue_found == 0
)

decision = pd.DataFrame([{
    "v53_decision": "gt_documentation_and_validation_report_completed",
    "validated_scanframe_count": scanframe_count,
    "validated_clip_count": clip_count,
    "source_video_count": source_video_count,
    "clip_object_count": clip_object_count,
    "usable_behaviour_labelled_objects": usable_behaviour_objects,
    "missing_or_unmapped_behaviour_objects": missing_behaviour_objects,
    "tracking_rows_total": tracking_rows_total,
    "mean_draw_ok_ratio": mean_draw,
    "mean_missing_ratio": mean_missing,
    "strong_clip_count": strong_clip_count,
    "usable_clip_count": usable_clip_count,
    "limited_clip_count": limited_clip_count,
    "challenging_clip_count": challenging_clip_count,
    "high_priority_reviewed_clip_count": high_priority_reviewed,
    "high_priority_accepted_clip_count": high_priority_accepted,
    "manual_issue_found_clip_count": manual_issue_found,
    "hard_issue_count": hard_issue_count,
    "warning_count": warning_count,
    "issue_count": int(len(issues_df)),
    "ready_for_v54_final_delivery_package": bool(ready),
    "generated_at": datetime.now().isoformat(timespec="seconds"),
}])
safe_to_csv(decision, OUT_DECISION)

behaviour_table_md = behaviour_distribution.to_markdown(index=False)
video_table_md = video_scanframe_summary.to_markdown(index=False)
tier_table_md = tracking_tier_summary.to_markdown(index=False)
manual_table_md = manual_validation_summary.to_markdown(index=False)

dataset_doc = f"""# Week 8 Ground Truth Dataset Documentation

## Dataset scope

The Week 8 ground truth dataset is not the complete raw Unibo video archive. It is a validated subset built from annotated observation windows.

- Validated scanframes: {scanframe_count}
- Validated clips: {clip_count}
- Source videos represented in the validated subset: {source_video_count}
- Clip duration policy: 10-second observation window per scanframe
- Clip-object annotations: {clip_object_count}
- Usable behaviour-labelled objects: {usable_behaviour_objects}
- Missing or unmapped behaviour objects: {missing_behaviour_objects}

Each scanframe corresponds to one annotation reference point. A 10-second clip was extracted around that observation window, and the behaviour label is propagated across the interval for the corresponding pig identity.

## Main data levels

### Clip level

Each clip represents one annotated observation window.

Typical fields:

- scan_frame_id
- video_id
- clip_path
- frame_count
- fps_used
- observation interval metadata

### Clip-object level

Each clip-object row represents one pig/object at the annotation anchor level.

Typical fields:

- scan_frame_id
- final_box_id
- visual marker colour
- behaviour pig ID
- behaviour code
- anchor bounding box
- behaviour propagation source

### Frame-object tracking-support level

The full hybrid tracking-support layer expands clip-object annotations to frame-object rows.

- Frame-object rows: {tracking_rows_total}
- Stable tracklet ratio: {mean_stable:.4f}
- Recall fallback ratio: {mean_fallback:.4f}
- Missing ratio: {mean_missing:.4f}
- Draw-ok ratio: {mean_draw:.4f}
- Review-needed ratio: {mean_review:.4f}

Hybrid source interpretation:

- stable_tracklet: primary temporal tracking evidence
- recall_fallback: fallback detection-linked evidence; review-needed
- missing: no reliable box drawn; preserved explicitly

## Behaviour distribution

{behaviour_table_md}

## Source video / scanframe summary

{video_table_md}

## Classification usage policy

This dataset is suitable for baseline and pilot classification experiments after validation. It should not be used to claim a robust production-level behaviour classifier. Rare classes remain limited and should be handled carefully.
"""

OUT_DATASET_DOC.write_text(dataset_doc)

validation_doc = f"""# Week 8 Validation Report

## Validation status

- v53 decision: {decision.iloc[0]['v53_decision']}
- Ready for v54 final delivery package: {ready}

## Propagation and GT validation

The dataset uses a 10-second observation-window propagation policy. Behaviour labels are not treated as isolated single-frame labels; they apply to the observation interval.

## Hybrid tracking validation

The hybrid tracking layer was created to support visual validation of pig-colour-behaviour associations.

Global tracking-support metrics:

- Mean stable tracklet ratio: {mean_stable:.4f}
- Mean recall fallback ratio: {mean_fallback:.4f}
- Mean missing ratio: {mean_missing:.4f}
- Mean draw-ok ratio: {mean_draw:.4f}
- Mean review-needed ratio: {mean_review:.4f}

## Clip quality tiers

{tier_table_md}

## Manual validation

High-priority clips identified by automatic tracking QA were manually reviewed in the v52d visualizer.

- High-priority clips: {high_priority_clip_count}
- High-priority reviewed: {high_priority_reviewed}
- High-priority accepted: {high_priority_accepted}
- Manual issue found clips: {manual_issue_found}
- Manual notes saved: {manual_note_count}

Manual validation status summary:

{manual_table_md}

## Important limitations

The tracking output is not claimed as fully automatic ground-truth replacement. It is a tracking-assisted validation layer.

The full raw video archive is not fully manually annotated. The validated dataset covers the 72 annotated scanframe clips.

Rare behaviour classes remain limited. Classification should initially be presented as a baseline/pilot experiment.

## Report-ready claim

A validated behaviour-dataset preparation pipeline was completed for 72 annotated observation clips. The pipeline includes GT schema documentation, behaviour label propagation, hybrid tracking-assisted visual validation, quality-tier assessment, manual review of high-priority clips, and report-ready dataset documentation.
"""

OUT_VALIDATION_DOC.write_text(validation_doc)

report = f"""# Week 8 v53 GT Documentation and Validation Report

## Summary

- Decision: {decision.iloc[0]['v53_decision']}
- Validated scanframes: {scanframe_count}
- Validated clips: {clip_count}
- Source videos represented: {source_video_count}
- Clip-object annotations: {clip_object_count}
- Usable behaviour-labelled objects: {usable_behaviour_objects}
- Frame-object tracking rows: {tracking_rows_total}
- Mean draw-ok ratio: {mean_draw:.4f}
- Mean missing ratio: {mean_missing:.4f}
- High-priority clips reviewed: {high_priority_reviewed}/{high_priority_clip_count}
- Manual issue found clips: {manual_issue_found}
- Ready for v54 final delivery package: {ready}

## Generated documents

- Dataset documentation: `{OUT_DATASET_DOC}`
- Validation report: `{OUT_VALIDATION_DOC}`

## Generated tables

- Dataset summary metrics: `{OUT_DATASET_SUMMARY}`
- Source manifest: `{OUT_SOURCE_MANIFEST}`
- Behaviour distribution: `{OUT_BEHAVIOUR_DISTRIBUTION}`
- Video scanframe summary: `{OUT_VIDEO_SCANFRAME_SUMMARY}`
- Tracking quality tier summary: `{OUT_TRACKING_TIER_SUMMARY}`
- Manual validation summary: `{OUT_MANUAL_VALIDATION_SUMMARY}`
- Limitations and claims: `{OUT_LIMITATIONS}`

## Interpretation

The Week 8 dataset is ready for final packaging and classification-preparation steps. The correct claim is that the project has produced a validated observation-window dataset and a tracking-assisted visual validation layer, not a fully automatic production tracking or final behaviour-classification system.
"""

OUT_REPORT.write_text(report)

note = f"""# Week 8 v53 GT Documentation and Validation Report

## Summary

- v53 decision: {decision.iloc[0]['v53_decision']}
- Validated clips: {clip_count}
- Validated scanframes: {scanframe_count}
- Source videos represented: {source_video_count}
- Clip-object annotations: {clip_object_count}
- Usable behaviour-labelled objects: {usable_behaviour_objects}
- Frame-object tracking rows: {tracking_rows_total}
- Mean draw-ok ratio: {mean_draw:.4f}
- Mean missing ratio: {mean_missing:.4f}
- Strong clips: {strong_clip_count}
- Usable clips: {usable_clip_count}
- Limited clips: {limited_clip_count}
- Challenging clips: {challenging_clip_count}
- High-priority reviewed/accepted: {high_priority_reviewed}/{high_priority_accepted}
- Manual issue found clips: {manual_issue_found}
- Ready for v54 final delivery package: {ready}

## Next

Create the final Week 8 delivery package and audit it.
"""

OUT_NOTE.write_text(note)

progress_row = pd.DataFrame([{
    "date": datetime.now().date().isoformat(),
    "stage": "v53",
    "task_name": "GT documentation and validation report",
    "status": "PASS" if ready else "PASS_WITH_WARNINGS" if hard_issue_count == 0 else "BLOCKED",
    "input_summary": str(V52B3_TRACKS),
    "output_summary": str(OUT),
    "hard_issues": hard_issue_count,
    "warnings": warning_count,
    "next_action": "v54 final delivery package and audit" if ready else "Resolve v53 warnings before final packaging.",
}])

if OUT_PROGRESS.exists():
    old = pd.read_csv(OUT_PROGRESS)
    progress = pd.concat([old, progress_row], ignore_index=True)
else:
    progress = progress_row

safe_to_csv(progress, OUT_PROGRESS)

print("Saved:")
print(OUT_DATASET_SUMMARY)
print(OUT_SOURCE_MANIFEST)
print(OUT_BEHAVIOUR_DISTRIBUTION)
print(OUT_VIDEO_SCANFRAME_SUMMARY)
print(OUT_TRACKING_TIER_SUMMARY)
print(OUT_MANUAL_VALIDATION_SUMMARY)
print(OUT_LIMITATIONS)
print(OUT_DECISION)
print(OUT_ISSUES)
print(OUT_DATASET_DOC)
print(OUT_VALIDATION_DOC)
print(OUT_REPORT)
print(OUT_NOTE)

print()
print("=== v53 decision ===")
print(decision.to_string(index=False))

print()
print("=== v53 dataset summary ===")
print(dataset_summary.to_string(index=False))

print()
print("=== v53 tracking tier summary ===")
print(tracking_tier_summary.to_string(index=False))

print()
print("=== v53 issues ===")
if len(issues_df):
    print(issues_df.to_string(index=False))
else:
    print("No issues found.")
