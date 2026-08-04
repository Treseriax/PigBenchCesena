from pathlib import Path
from datetime import datetime
import csv
import json
import hashlib
import zipfile
import pandas as pd


ROOT = Path.home() / "PigBench"
W8 = ROOT / "Week8_Ground_Truth_Validation_Behaviour_Dataset_Preparation"

FINAL_DIR = W8 / "outputs" / "v64c_final_gt_v2_export" / "Week8_Final_GT_v2"
V65A_DIR = W8 / "outputs" / "v65a_classification_readiness_audit"

ALL = FINAL_DIR / "week8_final_gt_v2_all_reviewed_objects.csv"
STRICT = FINAL_DIR / "week8_final_gt_v2_strict_gold_objects_for_classification.csv"
CAUTION = FINAL_DIR / "week8_final_gt_v2_caution_objects_for_analysis.csv"
NONUSABLE = FINAL_DIR / "week8_final_gt_v2_nonusable_fix_or_excluded_objects.csv"
SCANFRAME = FINAL_DIR / "week8_final_gt_v2_scanframe_quality_summary.csv"
COUNTS = FINAL_DIR / "week8_final_gt_v2_counts.csv"
MANIFEST = FINAL_DIR / "week8_final_gt_v2_manifest.json"

V65A_BEHAVIOUR = V65A_DIR / "week8_v65a_strict_gold_behaviour_distribution.csv"
V65A_LIMITATIONS = V65A_DIR / "week8_v65a_classification_limitations.csv"
V65A_DECISION = V65A_DIR / "week8_v65a_decision_summary.csv"

OUT = W8 / "outputs" / "v65b_gt_v2_solid_foundation_snapshot"
PACKAGE = OUT / "Week8_GT_v2_Solid_Foundation"
NOTES = W8 / "notes"
REPORTS = W8 / "reports"
PROGRESS = W8 / "progress"

for p in [OUT, PACKAGE, NOTES, REPORTS, PROGRESS]:
    p.mkdir(parents=True, exist_ok=True)

OUT_ALL = PACKAGE / "week8_gt_v2_solid_all_reviewed_objects.csv"
OUT_STRICT = PACKAGE / "week8_gt_v2_solid_strict_gold_objects_for_classification.csv"
OUT_CAUTION = PACKAGE / "week8_gt_v2_solid_caution_objects_for_analysis.csv"
OUT_NONUSABLE = PACKAGE / "week8_gt_v2_solid_nonusable_fix_or_excluded_objects.csv"
OUT_SCANFRAME = PACKAGE / "week8_gt_v2_solid_scanframe_quality_summary.csv"
OUT_BEHAVIOUR = PACKAGE / "week8_gt_v2_solid_behaviour_distribution.csv"
OUT_CLASS_POLICY = PACKAGE / "week8_gt_v2_solid_class_policy.csv"
OUT_TASK_COVERAGE = PACKAGE / "week8_gt_v2_solid_week8_task_coverage.csv"
OUT_QA = PACKAGE / "week8_gt_v2_solid_quality_checks.csv"
OUT_LIMITATIONS = PACKAGE / "week8_gt_v2_solid_limitations.csv"
OUT_MANIFEST = PACKAGE / "week8_gt_v2_solid_manifest.json"
OUT_README = PACKAGE / "README_Week8_GT_v2_Solid_Foundation.md"

OUT_DECISION = OUT / "week8_v65b_decision_summary.csv"
OUT_ISSUES = OUT / "week8_v65b_issues.csv"
OUT_NOTE = NOTES / "week8_v65b_gt_v2_solid_foundation_snapshot_notes.md"
OUT_REPORT = REPORTS / "week8_v65b_gt_v2_solid_foundation_snapshot_report.md"
OUT_PROGRESS = PROGRESS / "week8_experiment_progress_log.csv"
OUT_ZIP = OUT / "Week8_GT_v2_Solid_Foundation.zip"
OUT_SHA256 = OUT / "Week8_GT_v2_Solid_Foundation.sha256"


def safe_to_csv(df, path):
    df.to_csv(path, index=False, quoting=csv.QUOTE_ALL, escapechar="\\", lineterminator="\n")


def clean(x):
    if pd.isna(x):
        return ""
    s = str(x).strip()
    if s.lower() in ["nan", "none", "null"]:
        return ""
    return s


def sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


issues = []

required = [
    ALL,
    STRICT,
    CAUTION,
    NONUSABLE,
    SCANFRAME,
    COUNTS,
    MANIFEST,
    V65A_BEHAVIOUR,
    V65A_LIMITATIONS,
    V65A_DECISION,
]

for p in required:
    if not p.exists():
        issues.append({
            "item": str(p),
            "issue_type": "hard_missing_required_input",
            "issue_detail": "Required input is missing for v65b solid foundation snapshot.",
            "severity": "hard",
        })

if issues:
    issues_df = pd.DataFrame(issues)
    safe_to_csv(issues_df, OUT_ISSUES)
    decision = pd.DataFrame([{
        "v65b_decision": "gt_v2_solid_foundation_snapshot_blocked",
        "hard_issue_count": int((issues_df["severity"] == "hard").sum()),
        "ready_to_repeat_week8_tasks_on_solid_gt": False,
        "generated_at": datetime.now().isoformat(timespec="seconds"),
    }])
    safe_to_csv(decision, OUT_DECISION)
    print(decision.to_string(index=False))
    raise SystemExit(1)


all_df = pd.read_csv(ALL).fillna("")
strict = pd.read_csv(STRICT).fillna("")
caution = pd.read_csv(CAUTION).fillna("")
nonusable = pd.read_csv(NONUSABLE).fillna("")
scanframe = pd.read_csv(SCANFRAME).fillna("")
counts = pd.read_csv(COUNTS).fillna("")
behaviour = pd.read_csv(V65A_BEHAVIOUR).fillna("")
limitations = pd.read_csv(V65A_LIMITATIONS).fillna("")
v65a_decision = pd.read_csv(V65A_DECISION).fillna("")

for df in [all_df, strict, caution, nonusable, scanframe, counts, behaviour, limitations, v65a_decision]:
    for c in df.columns:
        df[c] = df[c].map(clean)

# Solid QA checks.
qa_rows = []

def add_check(name, expected, actual, passed, severity, detail):
    qa_rows.append({
        "check_name": name,
        "expected": str(expected),
        "actual": str(actual),
        "passed": bool(passed),
        "severity": severity,
        "detail": detail,
    })

add_check(
    "total_object_rows",
    432,
    len(all_df),
    len(all_df) == 432,
    "hard",
    "Canonical GT v2 should have 72 scanframes x 6 canonical identities.",
)

add_check(
    "all_rows_reviewed",
    432,
    int((all_df["manual_gt_v2_status"].astype(str).str.strip() != "").sum()),
    int((all_df["manual_gt_v2_status"].astype(str).str.strip() != "").sum()) == 432,
    "hard",
    "Every object row must have a manual GT v2 decision.",
)

add_check(
    "strict_plus_caution_plus_nonusable_equals_total",
    len(all_df),
    len(strict) + len(caution) + len(nonusable),
    len(strict) + len(caution) + len(nonusable) == len(all_df),
    "hard",
    "Final subsets must partition all reviewed rows.",
)

add_check(
    "strict_gold_count_dynamic",
    len(strict),
    len(strict),
    True,
    "warning",
    "Strict gold count follows the current audited GT state.",
)

add_check(
    "caution_count_dynamic",
    len(caution),
    len(caution),
    True,
    "warning",
    "Caution count follows the current audited GT state.",
)

add_check(
    "nonusable_count_dynamic",
    len(nonusable),
    len(nonusable),
    True,
    "warning",
    "Nonusable/fix/excluded count follows the current audited GT state.",
)

required_strict = (
    (strict["manual_gt_v2_status"] == "gold_usable")
    & (strict["manual_classification_use"] == "use_for_classification")
    & (strict["manual_bbox_status"] == "bbox_ok")
    & (strict["manual_identity_status"] == "identity_confirmed")
    & (strict["manual_assigned_candidate_box_id"].astype(str).str.strip() != "")
)

add_check(
    "strict_rows_follow_strict_rule",
    len(strict),
    int(required_strict.sum()),
    int(required_strict.sum()) == len(strict),
    "hard",
    "Strict rows must satisfy gold/use/bbox_ok/identity_confirmed/assigned_box rule.",
)

add_check(
    "scanframe_count",
    72,
    all_df["scan_frame_id"].nunique(),
    all_df["scan_frame_id"].nunique() == 72,
    "hard",
    "GT v2 scope should cover 72 annotated scanframes.",
)

add_check(
    "canonical_colour_count",
    6,
    all_df["canonical_colour_label_norm"].nunique(),
    all_df["canonical_colour_label_norm"].nunique() == 6,
    "hard",
    "GT v2 should contain six canonical identities per scanframe.",
)

qa = pd.DataFrame(qa_rows)
safe_to_csv(qa, OUT_QA)

hard_fail_count = int(((qa["severity"] == "hard") & (~qa["passed"])).sum())
warning_fail_count = int(((qa["severity"] == "warning") & (~qa["passed"])).sum())

if hard_fail_count > 0:
    issues.append({
        "item": "quality_checks",
        "issue_type": "hard_gt_v2_quality_check_failed",
        "issue_detail": f"{hard_fail_count} hard quality checks failed.",
        "severity": "hard",
    })

if warning_fail_count > 0:
    issues.append({
        "item": "quality_checks",
        "issue_type": "warning_gt_v2_quality_check_warning",
        "issue_detail": f"{warning_fail_count} warning-level quality checks failed.",
        "severity": "warning",
    })

# Class policy.
policy_rows = []
for _, r in behaviour.iterrows():
    code = r["behaviour_code"]
    count = int(float(r["strict_gold_object_count"]))
    pct = float(r["percentage"])

    if count >= 20:
        policy = "eligible_for_primary_baseline"
        recommendation = "Can be included in the main baseline, while reporting class imbalance."
    elif count >= 10:
        policy = "eligible_with_caution"
        recommendation = "Can be included, but metrics should be interpreted carefully."
    else:
        policy = "rare_class_not_reliable_for_standalone_metrics"
        recommendation = "Do not claim reliable standalone classification performance for this class."

    policy_rows.append({
        "behaviour_code": code,
        "strict_gold_object_count": count,
        "percentage": pct,
        "class_policy": policy,
        "recommendation": recommendation,
    })

class_policy = pd.DataFrame(policy_rows)
safe_to_csv(class_policy, OUT_CLASS_POLICY)

# Week 8 task coverage status.
task_rows = [
    {
        "week8_task": "Improve identity assignment",
        "status": "completed_as_manual_gt_v2_identity_assignment",
        "evidence": "All 432 canonical colour identities reviewed; strict/caution/nonusable categories produced.",
        "limitation": "This is not a claim of production-grade multi-object tracking over all raw videos.",
    },
    {
        "week8_task": "GT format documentation",
        "status": "completed",
        "evidence": "Final GT v2 manifest and README generated.",
        "limitation": "Documentation covers 72 annotated scanframe clips, not the entire raw Unibo archive.",
    },
    {
        "week8_task": "Behaviour label propagation",
        "status": "ready_to_repeat_on_solid_gt",
        "evidence": "Canonical Excel labels and object-level GT are finalized.",
        "limitation": "Previous propagation was based on earlier GT; repeat propagation should now use final GT v2.",
    },
    {
        "week8_task": "Dataset validation",
        "status": "completed_for_object_level_gt",
        "evidence": "v64b audit passed with zero hard inconsistencies.",
        "limitation": "Frame-level tracking propagation still needs final rerun if needed for temporal model.",
    },
    {
        "week8_task": "Behaviour dataset visualization interface",
        "status": "completed_for_manual_review",
        "evidence": "v63b visualizer used for all 72 scanframes.",
        "limitation": "Next visualizer should load final GT v2 categories directly.",
    },
    {
        "week8_task": "Classification baseline",
        "status": "ready_for_baseline_only",
        "evidence": "363 strict gold objects available.",
        "limitation": "Small and imbalanced dataset. Baseline/proof-of-concept only.",
    },
]

task_coverage = pd.DataFrame(task_rows)
safe_to_csv(task_coverage, OUT_TASK_COVERAGE)

# Copy final data into solid package.
safe_to_csv(all_df, OUT_ALL)
safe_to_csv(strict, OUT_STRICT)
safe_to_csv(caution, OUT_CAUTION)
safe_to_csv(nonusable, OUT_NONUSABLE)
safe_to_csv(scanframe, OUT_SCANFRAME)
safe_to_csv(behaviour, OUT_BEHAVIOUR)
safe_to_csv(limitations, OUT_LIMITATIONS)

manifest = {
    "dataset_version": "week8_gt_v2_solid_foundation_v65b",
    "created_at": datetime.now().isoformat(timespec="seconds"),
    "scope": {
        "annotated_scanframes": int(all_df["scan_frame_id"].nunique()),
        "canonical_object_rows": int(len(all_df)),
        "strict_gold_objects": int(len(strict)),
        "caution_objects": int(len(caution)),
        "nonusable_fix_or_excluded_objects": int(len(nonusable)),
        "strict_gold_scanframes": int(strict["scan_frame_id"].nunique()),
        "behaviour_classes": int(strict["behaviour_code"].nunique()),
    },
    "quality": {
        "hard_quality_failures": hard_fail_count,
        "warning_quality_failures": warning_fail_count,
        "ready_to_repeat_week8_tasks_on_solid_gt": hard_fail_count == 0,
        "ready_for_baseline_only": hard_fail_count == 0 and len(strict) > 0,
    },
    "usage_rules": {
        "classification_training_and_metrics": "Use strict gold objects only.",
        "caution_rows": "Use only for qualitative analysis unless explicitly stated.",
        "nonusable_rows": "Do not use for training or evaluation.",
        "clip_level_claims": "Only full strict-gold scanframes can support full-clip six-object claims.",
        "object_level_claims": "Mixed scanframes can support only object-level claims for strict gold rows.",
    },
    "important_limitations": [
        "This GT covers 72 annotated scanframe clips, not all raw Unibo videos.",
        "Behaviour distribution is imbalanced.",
        "Rare behaviours IA, DE, and BE are not reliable for standalone metrics.",
        "This supports a baseline/proof-of-concept classifier, not a production-grade behaviour classifier.",
        "Final behaviour propagation should be repeated from this solid GT if temporal/frame-level labels are required.",
    ],
    "files": {
        "all_reviewed_objects": OUT_ALL.name,
        "strict_gold_objects": OUT_STRICT.name,
        "caution_objects": OUT_CAUTION.name,
        "nonusable_objects": OUT_NONUSABLE.name,
        "scanframe_summary": OUT_SCANFRAME.name,
        "behaviour_distribution": OUT_BEHAVIOUR.name,
        "class_policy": OUT_CLASS_POLICY.name,
        "task_coverage": OUT_TASK_COVERAGE.name,
        "quality_checks": OUT_QA.name,
        "limitations": OUT_LIMITATIONS.name,
    },
}

OUT_MANIFEST.write_text(json.dumps(manifest, indent=2, ensure_ascii=False))

readme = f"""# Week 8 GT v2 Solid Foundation Snapshot

## Summary

This package freezes the current solid GT v2 foundation for the 72 annotated scanframe clips.

## Key numbers

- Total canonical object rows: {len(all_df)}
- Reviewed object rows: {int((all_df["manual_gt_v2_status"].astype(str).str.strip() != "").sum())}
- Strict gold objects for classification: {len(strict)}
- Caution objects for analysis: {len(caution)}
- Nonusable / fix / excluded objects: {len(nonusable)}
- Full strict-gold scanframes: {int((scanframe["scanframe_quality_decision"] == "full_strict_gold").sum())}
- Mixed object-level scanframes: {int(scanframe["scanframe_quality_decision"].astype(str).str.contains("mixed").sum())}
- No classification-usable scanframes: {int((scanframe["scanframe_quality_decision"] == "no_classification_usable_object").sum())}

## Usage

Use `week8_gt_v2_solid_strict_gold_objects_for_classification.csv` for any classification baseline.

Do not train or evaluate on caution or nonusable rows.

## Next step

Repeat the Week 8 downstream tasks on this solid GT base:

1. Rebuild propagated frame/object labels from final GT v2.
2. Update final visualizer to load final GT categories directly.
3. Rebuild validation reports.
4. Then run baseline classification only on strict gold rows.
"""

OUT_README.write_text(readme)

# Zip package.
if OUT_ZIP.exists():
    OUT_ZIP.unlink()

with zipfile.ZipFile(OUT_ZIP, "w", compression=zipfile.ZIP_DEFLATED) as z:
    for p in sorted(PACKAGE.rglob("*")):
        if p.is_file():
            z.write(p, p.relative_to(OUT))

zip_hash = sha256_file(OUT_ZIP)
OUT_SHA256.write_text(f"{zip_hash}  {OUT_ZIP.name}\n")

issues_df = pd.DataFrame(issues, columns=["item", "issue_type", "issue_detail", "severity"])
safe_to_csv(issues_df, OUT_ISSUES)

hard_issue_count = int((issues_df["severity"] == "hard").sum()) if len(issues_df) else 0
warning_count = int((issues_df["severity"] == "warning").sum()) if len(issues_df) else 0

decision = pd.DataFrame([{
    "v65b_decision": "gt_v2_solid_foundation_snapshot_completed" if hard_issue_count == 0 else "gt_v2_solid_foundation_snapshot_created_with_issues",
    "total_object_rows": int(len(all_df)),
    "reviewed_object_rows": int((all_df["manual_gt_v2_status"].astype(str).str.strip() != "").sum()),
    "strict_gold_objects": int(len(strict)),
    "caution_objects": int(len(caution)),
    "nonusable_fix_or_excluded_objects": int(len(nonusable)),
    "scanframe_count": int(all_df["scan_frame_id"].nunique()),
    "behaviour_class_count": int(strict["behaviour_code"].nunique()),
    "hard_quality_failures": hard_fail_count,
    "warning_quality_failures": warning_fail_count,
    "hard_issue_count": hard_issue_count,
    "warning_count": warning_count,
    "package_dir": str(PACKAGE),
    "zip_path": str(OUT_ZIP),
    "zip_sha256": zip_hash,
    "ready_to_repeat_week8_tasks_on_solid_gt": bool(hard_issue_count == 0),
    "ready_for_baseline_only": bool(hard_issue_count == 0 and len(strict) > 0),
    "generated_at": datetime.now().isoformat(timespec="seconds"),
}])

safe_to_csv(decision, OUT_DECISION)

OUT_NOTE.write_text(
    "# Week 8 v65b GT v2 Solid Foundation Snapshot\n\n"
    f"- v65b decision: {decision.iloc[0]['v65b_decision']}\n"
    f"- Total object rows: {len(all_df)}\n"
    f"- Reviewed object rows: {int((all_df['manual_gt_v2_status'].astype(str).str.strip() != '').sum())}\n"
    f"- Strict gold objects: {len(strict)}\n"
    f"- Caution objects: {len(caution)}\n"
    f"- Nonusable/fix/excluded objects: {len(nonusable)}\n"
    f"- Behaviour classes: {strict['behaviour_code'].nunique()}\n"
    f"- Hard quality failures: {hard_fail_count}\n"
    f"- Warning quality failures: {warning_fail_count}\n"
    f"- Package: {PACKAGE}\n"
    f"- Zip: {OUT_ZIP}\n"
    f"- SHA256: {zip_hash}\n"
    f"- Ready to repeat Week 8 tasks on solid GT: {bool(hard_issue_count == 0)}\n\n"
    "Next: rerun downstream Week 8 tasks using this GT v2 solid foundation instead of older draft GT outputs.\n"
)

OUT_REPORT.write_text(
    "# Week 8 v65b GT v2 Solid Foundation Snapshot Report\n\n"
    f"Decision: {decision.iloc[0]['v65b_decision']}\n\n"
    f"Package directory: {PACKAGE}\n\n"
    f"Zip: {OUT_ZIP}\n\n"
    f"SHA256: {zip_hash}\n\n"
    "This package is the current GT foundation before repeating Week 8 downstream tasks.\n"
)

progress_row = pd.DataFrame([{
    "date": datetime.now().date().isoformat(),
    "stage": "v65b",
    "task_name": "GT v2 solid foundation snapshot",
    "status": "PASS" if hard_issue_count == 0 else "NEEDS_FIX",
    "input_summary": str(FINAL_DIR),
    "output_summary": str(PACKAGE),
    "hard_issues": hard_issue_count,
    "warnings": warning_count,
    "next_action": "Repeat Week 8 propagation, visualizer, validation, and baseline preparation on solid GT v2.",
}])

if OUT_PROGRESS.exists():
    old = pd.read_csv(OUT_PROGRESS)
    progress = pd.concat([old, progress_row], ignore_index=True)
else:
    progress = progress_row

safe_to_csv(progress, OUT_PROGRESS)

print("Saved:")
print(PACKAGE)
print(OUT_QA)
print(OUT_CLASS_POLICY)
print(OUT_TASK_COVERAGE)
print(OUT_MANIFEST)
print(OUT_ZIP)
print(OUT_SHA256)
print(OUT_DECISION)
print(OUT_NOTE)

print()
print("=== v65b decision ===")
print(decision.to_string(index=False))

print()
print("=== quality checks ===")
print(qa.to_string(index=False))

print()
print("=== class policy ===")
print(class_policy.to_string(index=False))

print()
print("=== task coverage ===")
print(task_coverage.to_string(index=False))

print()
print("=== issues ===")
if len(issues_df):
    print(issues_df.to_string(index=False))
else:
    print("No issues found.")
