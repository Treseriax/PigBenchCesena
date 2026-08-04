from pathlib import Path
from datetime import datetime
import csv
import json
import pandas as pd


ROOT = Path.home() / "PigBench"
W8 = ROOT / "Week8_Ground_Truth_Validation_Behaviour_Dataset_Preparation"

FINAL_DIR = W8 / "outputs" / "v64c_final_gt_v2_export" / "Week8_Final_GT_v2"

STRICT = FINAL_DIR / "week8_final_gt_v2_strict_gold_objects_for_classification.csv"
CAUTION = FINAL_DIR / "week8_final_gt_v2_caution_objects_for_analysis.csv"
NONUSABLE = FINAL_DIR / "week8_final_gt_v2_nonusable_fix_or_excluded_objects.csv"
ALL = FINAL_DIR / "week8_final_gt_v2_all_reviewed_objects.csv"
SCANFRAME = FINAL_DIR / "week8_final_gt_v2_scanframe_quality_summary.csv"
MANIFEST = FINAL_DIR / "week8_final_gt_v2_manifest.json"

OUT = W8 / "outputs" / "v65a_classification_readiness_audit"
NOTES = W8 / "notes"
REPORTS = W8 / "reports"
PROGRESS = W8 / "progress"

for p in [OUT, NOTES, REPORTS, PROGRESS]:
    p.mkdir(parents=True, exist_ok=True)

OUT_BEHAVIOUR = OUT / "week8_v65a_strict_gold_behaviour_distribution.csv"
OUT_COLOUR = OUT / "week8_v65a_strict_gold_colour_distribution.csv"
OUT_VIDEO = OUT / "week8_v65a_strict_gold_video_distribution.csv"
OUT_SCANFRAME = OUT / "week8_v65a_strict_gold_scanframe_distribution.csv"
OUT_SPLIT = OUT / "week8_v65a_recommended_scanframe_level_split.csv"
OUT_SPLIT_COUNTS = OUT / "week8_v65a_split_behaviour_counts.csv"
OUT_LIMITATIONS = OUT / "week8_v65a_classification_limitations.csv"
OUT_DECISION = OUT / "week8_v65a_decision_summary.csv"
OUT_ISSUES = OUT / "week8_v65a_issues.csv"
OUT_NOTE = NOTES / "week8_v65a_classification_readiness_audit_notes.md"
OUT_REPORT = REPORTS / "week8_v65a_classification_readiness_audit_report.md"
OUT_PROGRESS = PROGRESS / "week8_experiment_progress_log.csv"


def safe_to_csv(df, path):
    df.to_csv(path, index=False, quoting=csv.QUOTE_ALL, escapechar="\\", lineterminator="\n")


issues = []

for p in [STRICT, ALL, SCANFRAME, MANIFEST]:
    if not p.exists():
        issues.append({
            "item": str(p),
            "issue_type": "hard_missing_required_input",
            "issue_detail": "Required final GT v2 input is missing.",
            "severity": "hard",
        })

if issues:
    issues_df = pd.DataFrame(issues)
    safe_to_csv(issues_df, OUT_ISSUES)
    decision = pd.DataFrame([{
        "v65a_decision": "classification_readiness_audit_blocked",
        "hard_issue_count": int((issues_df["severity"] == "hard").sum()),
        "ready_for_v65b_dataset_materialization": False,
        "generated_at": datetime.now().isoformat(timespec="seconds"),
    }])
    safe_to_csv(decision, OUT_DECISION)
    print(decision.to_string(index=False))
    raise SystemExit(1)


strict = pd.read_csv(STRICT).fillna("")
all_df = pd.read_csv(ALL).fillna("")
scanframe = pd.read_csv(SCANFRAME).fillna("")

beh = (
    strict["behaviour_code"]
    .value_counts()
    .reset_index()
)
beh.columns = ["behaviour_code", "strict_gold_object_count"]
beh["percentage"] = (beh["strict_gold_object_count"] / len(strict) * 100).round(2)
safe_to_csv(beh, OUT_BEHAVIOUR)

colour = (
    strict["canonical_colour_label_norm"]
    .value_counts()
    .reset_index()
)
colour.columns = ["canonical_colour_label_norm", "strict_gold_object_count"]
colour["percentage"] = (colour["strict_gold_object_count"] / len(strict) * 100).round(2)
safe_to_csv(colour, OUT_COLOUR)

video = (
    strict.groupby("video_id")
    .agg(
        strict_gold_object_count=("canonical_gt_object_id", "count"),
        scanframe_count=("scan_frame_id", "nunique"),
        behaviour_codes=("behaviour_code", lambda x: ";".join(sorted(set(x.astype(str))))),
    )
    .reset_index()
)
safe_to_csv(video, OUT_VIDEO)

sf = (
    strict.groupby("scan_frame_id")
    .agg(
        video_id=("video_id", "first"),
        strict_gold_object_count=("canonical_gt_object_id", "count"),
        behaviour_codes=("behaviour_code", lambda x: ";".join(x.astype(str).tolist())),
    )
    .reset_index()
)
safe_to_csv(sf, OUT_SCANFRAME)

# Deterministic scanframe-level split.
# Split by scanframe, never by object, to avoid same-frame leakage.
scanframes = sorted(sf["scan_frame_id"].unique().tolist())

split_rows = []
for i, scan in enumerate(scanframes):
    if i % 10 in [0, 1]:
        split = "test"
    elif i % 10 in [2]:
        split = "val"
    else:
        split = "train"

    split_rows.append({
        "scan_frame_id": scan,
        "recommended_split": split,
    })

split_df = pd.DataFrame(split_rows)

strict_split = strict.merge(split_df, on="scan_frame_id", how="left")
safe_to_csv(strict_split, OUT_SPLIT)

split_counts = (
    strict_split.groupby(["recommended_split", "behaviour_code"])
    .size()
    .reset_index(name="count")
    .sort_values(["recommended_split", "behaviour_code"])
)
safe_to_csv(split_counts, OUT_SPLIT_COUNTS)

limitations = []

rare = beh[beh["strict_gold_object_count"] < 10]
for _, r in rare.iterrows():
    limitations.append({
        "limitation_type": "rare_behaviour_class",
        "item": r["behaviour_code"],
        "detail": f"Only {int(r['strict_gold_object_count'])} strict-gold objects. This class is not reliable for standalone classification metrics.",
        "severity": "warning",
    })

if len(strict) < 500:
    limitations.append({
        "limitation_type": "small_dataset",
        "item": "strict_gold_objects",
        "detail": f"Strict gold object count is {len(strict)}. Suitable for baseline/proof-of-concept, not production classifier.",
        "severity": "warning",
    })

if scanframe["classification_scope"].astype(str).str.contains("object_level").any():
    limitations.append({
        "limitation_type": "mixed_scanframes",
        "item": "object_level_gt",
        "detail": "Some scanframes are mixed object-level usable. Classification must use object-level rows, not whole clip labels blindly.",
        "severity": "warning",
    })

limitations_df = pd.DataFrame(limitations)
safe_to_csv(limitations_df, OUT_LIMITATIONS)

hard_issue_count = 0
warning_count = int((limitations_df["severity"] == "warning").sum()) if len(limitations_df) else 0

major_classes = int((beh["strict_gold_object_count"] >= 10).sum())
rare_classes = int((beh["strict_gold_object_count"] < 10).sum())

decision = pd.DataFrame([{
    "v65a_decision": "classification_readiness_audit_completed",
    "strict_gold_object_count": int(len(strict)),
    "strict_gold_scanframe_count": int(strict["scan_frame_id"].nunique()),
    "strict_gold_video_count": int(strict["video_id"].nunique()),
    "behaviour_class_count": int(strict["behaviour_code"].nunique()),
    "major_behaviour_classes_count_ge_10": major_classes,
    "rare_behaviour_classes_count_lt_10": rare_classes,
    "train_objects": int((strict_split["recommended_split"] == "train").sum()),
    "val_objects": int((strict_split["recommended_split"] == "val").sum()),
    "test_objects": int((strict_split["recommended_split"] == "test").sum()),
    "hard_issue_count": hard_issue_count,
    "warning_count": warning_count,
    "ready_for_v65b_dataset_materialization": True,
    "generated_at": datetime.now().isoformat(timespec="seconds"),
}])
safe_to_csv(decision, OUT_DECISION)

issues_df = pd.DataFrame(issues, columns=["item", "issue_type", "issue_detail", "severity"])
safe_to_csv(issues_df, OUT_ISSUES)

top_beh_text = beh.to_string(index=False)

OUT_NOTE.write_text(
    "# Week 8 v65a Classification Readiness Audit\n\n"
    f"- v65a decision: {decision.iloc[0]['v65a_decision']}\n"
    f"- Strict gold objects: {len(strict)}\n"
    f"- Strict gold scanframes: {strict['scan_frame_id'].nunique()}\n"
    f"- Behaviour classes: {strict['behaviour_code'].nunique()}\n"
    f"- Major classes with >=10 objects: {major_classes}\n"
    f"- Rare classes with <10 objects: {rare_classes}\n"
    f"- Train objects: {int((strict_split['recommended_split'] == 'train').sum())}\n"
    f"- Val objects: {int((strict_split['recommended_split'] == 'val').sum())}\n"
    f"- Test objects: {int((strict_split['recommended_split'] == 'test').sum())}\n"
    f"- Ready for v65b dataset materialization: True\n\n"
    "Important: this is suitable for a baseline/proof-of-concept classifier. It is not enough to claim a production-grade behaviour classifier.\n"
)

OUT_REPORT.write_text(
    "# Week 8 v65a Classification Readiness Audit Report\n\n"
    f"Decision: {decision.iloc[0]['v65a_decision']}\n\n"
    "## Behaviour distribution\n\n"
    f"```text\n{top_beh_text}\n```\n\n"
    f"Recommended split: `{OUT_SPLIT}`\n\n"
    f"Limitations: `{OUT_LIMITATIONS}`\n"
)

progress_row = pd.DataFrame([{
    "date": datetime.now().date().isoformat(),
    "stage": "v65a",
    "task_name": "Classification readiness audit",
    "status": "PASS",
    "input_summary": str(STRICT),
    "output_summary": str(OUT),
    "hard_issues": 0,
    "warnings": warning_count,
    "next_action": "Materialize strict-gold crop/metadata dataset for baseline classification.",
}])

if OUT_PROGRESS.exists():
    old = pd.read_csv(OUT_PROGRESS)
    progress = pd.concat([old, progress_row], ignore_index=True)
else:
    progress = progress_row

safe_to_csv(progress, OUT_PROGRESS)

print("Saved:")
print(OUT_BEHAVIOUR)
print(OUT_COLOUR)
print(OUT_VIDEO)
print(OUT_SCANFRAME)
print(OUT_SPLIT)
print(OUT_SPLIT_COUNTS)
print(OUT_LIMITATIONS)
print(OUT_DECISION)
print(OUT_NOTE)

print()
print("=== v65a decision ===")
print(decision.to_string(index=False))

print()
print("=== behaviour distribution ===")
print(beh.to_string(index=False))

print()
print("=== split behaviour counts ===")
print(split_counts.to_string(index=False))

print()
print("=== limitations ===")
if len(limitations_df):
    print(limitations_df.to_string(index=False))
else:
    print("No limitations found.")
