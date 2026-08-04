from pathlib import Path
from datetime import datetime
import csv
import json
import hashlib
import shutil
import zipfile
import pandas as pd


ROOT = Path.home() / "PigBench"
W8 = ROOT / "Week8_Ground_Truth_Validation_Behaviour_Dataset_Preparation"

V67 = W8 / "outputs" / "v67_week8_report_ready_package"
V67B = W8 / "outputs" / "v67b_independent_package_audit"

PKG_SRC = V67 / "Week8_Report_Ready_GT_v2_Package"
V67_ZIP = V67 / "Week8_Report_Ready_GT_v2_Package.zip"
V67_SHA = V67 / "Week8_Report_Ready_GT_v2_Package.sha256"
V67B_DECISION = V67B / "week8_v67b_decision_summary.csv"

OUT = W8 / "outputs" / "v67c_dataset_card_gt_documentation"
DOC_PKG = OUT / "Week8_GT_v2_Dataset_Documentation"
NOTES = W8 / "notes"
REPORTS = W8 / "reports"
PROGRESS = W8 / "progress"

for p in [OUT, DOC_PKG, NOTES, REPORTS, PROGRESS]:
    p.mkdir(parents=True, exist_ok=True)

DATA = PKG_SRC / "data"
AUDIT = PKG_SRC / "audit"
TRACKING = PKG_SRC / "tracking_helper"

ALL = DATA / "week8_final_gt_v2_all_reviewed_objects.csv"
STRICT = DATA / "week8_final_gt_v2_strict_gold_objects_for_classification.csv"
CAUTION = DATA / "week8_final_gt_v2_caution_objects_for_analysis.csv"
NONUSABLE = DATA / "week8_final_gt_v2_nonusable_fix_or_excluded_objects.csv"
SCANFRAME = DATA / "week8_final_gt_v2_scanframe_quality_summary.csv"
BEHAVIOUR = DATA / "week8_v65a_strict_gold_behaviour_distribution.csv"
LIMITATIONS = DATA / "week8_v65a_classification_limitations.csv"
V66A_SCAN = DATA / "week8_v66a_scanframe_propagation_summary.csv"
V66C_SCAN = TRACKING / "week8_v66c_scanframe_tracking_helper_summary.csv"

V65A_DECISION = AUDIT / "week8_v65a_decision_summary.csv"
V66A_DECISION = AUDIT / "week8_v66a_decision_summary.csv"
V66C_DECISION = AUDIT / "week8_v66c_decision_summary.csv"

OUT_DATASET_CARD = DOC_PKG / "DATASET_CARD_Week8_GT_v2.md"
OUT_GT_SCHEMA = DOC_PKG / "GT_FORMAT_AND_SCHEMA_Week8_GT_v2.md"
OUT_USAGE = DOC_PKG / "USAGE_AND_CLAIM_BOUNDARIES_Week8_GT_v2.md"
OUT_REPRO = DOC_PKG / "REPRODUCIBILITY_GUIDE_Week8_GT_v2.md"
OUT_README = DOC_PKG / "README_Week8_GT_v2_Dataset_Documentation.md"

OUT_KEY_NUMBERS = DOC_PKG / "week8_v67c_key_numbers.csv"
OUT_SCHEMA_INVENTORY = DOC_PKG / "week8_v67c_schema_inventory.csv"
OUT_CLAIM_CHECKS = DOC_PKG / "week8_v67c_claim_boundary_checks.csv"

OUT_DECISION = OUT / "week8_v67c_decision_summary.csv"
OUT_ISSUES = OUT / "week8_v67c_issues.csv"
OUT_ZIP = OUT / "Week8_GT_v2_Dataset_Documentation.zip"
OUT_SHA256 = OUT / "Week8_GT_v2_Dataset_Documentation.sha256"
OUT_NOTE = NOTES / "week8_v67c_dataset_card_gt_documentation_notes.md"
OUT_REPORT = REPORTS / "week8_v67c_dataset_card_gt_documentation_report.md"
OUT_PROGRESS = PROGRESS / "week8_experiment_progress_log.csv"


def safe_to_csv(df, path):
    df.to_csv(path, index=False, quoting=csv.QUOTE_ALL, escapechar="\\", lineterminator="\n")


def clean(x):
    if pd.isna(x):
        return ""
    s = str(x).strip()
    if s.lower() in ["nan", "none", "null"]:
        return ""
    return s


def read_csv_clean(path):
    df = pd.read_csv(path).fillna("")
    for c in df.columns:
        if df[c].dtype == object:
            df[c] = df[c].map(clean)
    return df


def sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def parse_sha(path):
    if not path.exists():
        return ""
    txt = path.read_text().strip()
    return txt.split()[0] if txt else ""


issues = []

required = [
    PKG_SRC,
    V67_ZIP,
    V67_SHA,
    V67B_DECISION,
    ALL,
    STRICT,
    CAUTION,
    NONUSABLE,
    SCANFRAME,
    BEHAVIOUR,
    LIMITATIONS,
    V66A_SCAN,
    V66C_SCAN,
    V65A_DECISION,
    V66A_DECISION,
    V66C_DECISION,
]

for p in required:
    if not p.exists():
        issues.append({
            "item": str(p),
            "issue_type": "hard_missing_required_input",
            "issue_detail": "Required input missing for v67c dataset documentation.",
            "severity": "hard",
        })

if issues:
    issues_df = pd.DataFrame(issues)
    safe_to_csv(issues_df, OUT_ISSUES)
    decision = pd.DataFrame([{
        "v67c_decision": "dataset_card_gt_documentation_blocked",
        "hard_issue_count": int((issues_df["severity"] == "hard").sum()),
        "ready_for_v68a_crop_dataset_materialization": False,
        "generated_at": datetime.now().isoformat(timespec="seconds"),
    }])
    safe_to_csv(decision, OUT_DECISION)
    print(decision.to_string(index=False))
    raise SystemExit(1)


v67b_decision = read_csv_clean(V67B_DECISION)
if len(v67b_decision) == 0 or v67b_decision.iloc[0].get("v67b_decision", "") != "independent_package_audit_passed":
    issues.append({
        "item": str(V67B_DECISION),
        "issue_type": "hard_v67b_not_passed",
        "issue_detail": "v67b independent package audit must pass before documentation is frozen.",
        "severity": "hard",
    })

all_gt = read_csv_clean(ALL)
strict = read_csv_clean(STRICT)
caution = read_csv_clean(CAUTION)
nonusable = read_csv_clean(NONUSABLE)
scanframe = read_csv_clean(SCANFRAME)
behaviour = read_csv_clean(BEHAVIOUR)
limitations = read_csv_clean(LIMITATIONS)
v66a_scan = read_csv_clean(V66A_SCAN)
v66c_scan = read_csv_clean(V66C_SCAN)
v65a_decision = read_csv_clean(V65A_DECISION)
v66a_decision = read_csv_clean(V66A_DECISION)
v66c_decision = read_csv_clean(V66C_DECISION)

total_objects = len(all_gt)
strict_objects = len(strict)
caution_objects = len(caution)
nonusable_objects = len(nonusable)
scanframes = all_gt["scan_frame_id"].nunique()
canonical_colours = sorted(all_gt["canonical_colour_label_norm"].unique().tolist())
behaviour_classes = strict["behaviour_code"].nunique()
source_videos = all_gt["video_id"].nunique()

frame_object_rows = int(pd.to_numeric(v66a_scan["frame_object_rows"], errors="coerce").fillna(0).sum())
strict_frame_object_rows = int(pd.to_numeric(v66a_scan["strict_gold_frame_object_rows"], errors="coerce").fillna(0).sum())
tracking_helper_scanframes = int((pd.to_numeric(v66c_scan["tracking_helper_total_rows"], errors="coerce").fillna(0) > 0).sum())

full_strict = int((scanframe["scanframe_quality_decision"] == "full_strict_gold").sum()) if "scanframe_quality_decision" in scanframe.columns else 0
mixed = int(scanframe["scanframe_quality_decision"].astype(str).str.contains("mixed").sum()) if "scanframe_quality_decision" in scanframe.columns else 0
no_usable = int((scanframe["scanframe_quality_decision"] == "no_classification_usable_object").sum()) if "scanframe_quality_decision" in scanframe.columns else 0

train_objects = v65a_decision.iloc[0].get("train_objects", "")
val_objects = v65a_decision.iloc[0].get("val_objects", "")
test_objects = v65a_decision.iloc[0].get("test_objects", "")

v67_zip_sha = parse_sha(V67_SHA)

key_numbers = pd.DataFrame([{
    "dataset_version": "Week8_GT_v2_v67c_documented",
    "v67_package_sha256": v67_zip_sha,
    "total_reviewed_objects": total_objects,
    "strict_gold_objects": strict_objects,
    "caution_objects": caution_objects,
    "nonusable_fix_or_excluded_objects": nonusable_objects,
    "scanframes": scanframes,
    "source_videos": source_videos,
    "canonical_colour_identity_count": len(canonical_colours),
    "canonical_colour_identities": ";".join(canonical_colours),
    "behaviour_classes_in_strict_gold": behaviour_classes,
    "full_strict_gold_scanframes": full_strict,
    "mixed_object_level_scanframes": mixed,
    "no_classification_usable_scanframes": no_usable,
    "frame_object_rows": frame_object_rows,
    "strict_gold_frame_object_rows": strict_frame_object_rows,
    "tracking_helper_scanframes": tracking_helper_scanframes,
    "train_objects": train_objects,
    "val_objects": val_objects,
    "test_objects": test_objects,
}])
safe_to_csv(key_numbers, OUT_KEY_NUMBERS)

schema_rows = []
tables = {
    "all_reviewed_objects": all_gt,
    "strict_gold_objects_for_classification": strict,
    "caution_objects_for_analysis": caution,
    "nonusable_fix_or_excluded_objects": nonusable,
    "scanframe_quality_summary": scanframe,
    "behaviour_distribution": behaviour,
    "classification_limitations": limitations,
    "propagation_scanframe_summary": v66a_scan,
    "tracking_helper_scanframe_summary": v66c_scan,
}

for table_name, df in tables.items():
    for col in df.columns:
        nonempty = int((df[col].astype(str).str.strip() != "").sum())
        unique = int(df[col].nunique(dropna=True))
        sample_values = ";".join(df[col].astype(str).drop_duplicates().head(5).tolist())
        schema_rows.append({
            "table_name": table_name,
            "column_name": col,
            "dtype": str(df[col].dtype),
            "nonempty_count": nonempty,
            "unique_count": unique,
            "sample_values": sample_values,
        })

schema_inventory = pd.DataFrame(schema_rows)
safe_to_csv(schema_inventory, OUT_SCHEMA_INVENTORY)

behaviour_table = behaviour.to_string(index=False)
limitations_table = limitations.to_string(index=False)
key_table = key_numbers.T.reset_index()
key_table.columns = ["item", "value"]
key_table_text = key_table.to_string(index=False)

dataset_card = f"""# Dataset Card: Week8 GT v2

## Dataset Name

Week8 GT v2 validated pig behaviour dataset foundation.

## Version

Week8_GT_v2_v67c_documented

## Package SHA256

{v67_zip_sha}

## Summary

This dataset card documents the validated Week 8 GT v2 package for the Unibo pig behaviour analysis workflow. The dataset covers 72 annotated scanframe clips and 432 canonical object-level annotation rows.

Manual GT v2 is the source of truth for bbox-to-identity assignment. Tracking is diagnostic helper only. This package is not the full raw Unibo archive.

## Key Numbers

{key_table_text}

## Canonical Identities

The six canonical colour identities are:

{", ".join(canonical_colours)}

These identities are derived from the original Excel annotation source and manually linked to visual/bbox evidence during GT v2 review.

## Behaviour Classes

The strict-gold subset contains {behaviour_classes} behaviour classes.

{behaviour_table}

## Intended Uses

- Dataset validation and reporting.
- Baseline/proof-of-concept behaviour classification using strict gold rows only.
- Error analysis and qualitative analysis using caution rows separately.
- Tracking diagnostic analysis at scanframe level.

## Out-of-Scope Uses

- Production-grade behaviour classification claims.
- Production-grade tracking claims.
- Full raw Unibo archive GT claims.
- Training or evaluating on caution/nonusable rows as if they were strict gold.
- Treating tracking helper outputs as identity source of truth.

## Data Source and Annotation Source

Colour and behaviour labels come from the canonical original Excel annotation source. Object-level bbox and identity assignment come from manual GT v2 validation and corrected bbox audit.

## Quality and Validation

The v67 package passed independent package audit in v67b. The audit verified ZIP integrity, SHA256 consistency, required files, subset partition, object counts, scanframe counts, strict rule compliance, propagation counts, tracking helper attachment, and claim boundaries.

## Known Limitations

{limitations_table}

## Claim Boundary

This dataset supports a baseline/proof-of-concept classifier only. It is not the full raw Unibo archive and does not support production-grade classification or tracking claims.
"""

OUT_DATASET_CARD.write_text(dataset_card)

schema_doc = f"""# GT Format and Schema Documentation

## Core Tables

The documentation package contains schema inventory for all important tables in:

- all reviewed object rows
- strict gold classification rows
- caution analysis rows
- nonusable/fix/excluded rows
- scanframe quality summary
- label propagation summary
- tracking helper summary

See:

week8_v67c_schema_inventory.csv

## Object-Level GT Rule

A row belongs to strict gold classification GT only if all of the following are true:

- manual_gt_v2_status = gold_usable
- manual_classification_use = use_for_classification
- manual_bbox_status = bbox_ok
- manual_identity_status = identity_confirmed
- manual_assigned_candidate_box_id is not empty

## Subset Definitions

Strict gold rows are used for training/evaluation. Caution rows are analysis-only. Nonusable/fix/excluded rows must not be used for training or evaluation.

## Canonical Object ID

canonical_gt_object_id uniquely identifies one canonical object row, usually combining scanframe and canonical colour identity.

## Scanframe Scope

Each scanframe contains six canonical colour identities. The dataset has {scanframes} scanframes and {total_objects} canonical object rows.

## Propagation Format

Final GT v2 labels are propagated across 10-second observation clips. BBox coordinates are static manual anchor boxes repeated across the clip for label representation. This is not a tracking-quality claim.

## Tracking Helper Format

Tracking helper is attached at scanframe level. Direct object-level tracking matches are not required and are not used as GT source of truth.
"""

OUT_GT_SCHEMA.write_text(schema_doc)

usage_doc = f"""# Usage and Claim Boundaries

## Correct Usage

Use strict gold rows only for classification training and evaluation.

Primary strict file:

week8_final_gt_v2_strict_gold_objects_for_classification.csv

## Caution Rows

Caution rows can be used for qualitative analysis, but they must not be mixed into strict train/eval metrics.

## Nonusable Rows

Nonusable, fix-required, or excluded rows must not be used for model training or model evaluation.

## Tracking Boundary

Tracking is diagnostic helper only. Manual GT v2 is the source of truth. Tracking helper output does not override manual GT v2 and does not establish production tracking quality.

## Classification Boundary

This dataset supports baseline/proof-of-concept behaviour classification only. It is not enough to claim a production-grade behaviour classifier.

## Archive Boundary

This package covers 72 annotated scanframe clips. It is not the full raw Unibo archive.

## Reporting Rules

Always report:

- total strict gold object count
- class imbalance
- rare class limitation
- macro-F1 and per-class metrics if classification is run
- confusion matrix
- split policy
- that caution and nonusable rows were excluded from strict training/evaluation
"""

OUT_USAGE.write_text(usage_doc)

repro_doc = f"""# Reproducibility Guide

## Final Package

ZIP:

{V67_ZIP}

SHA256:

{v67_zip_sha}

## Independent Audit

The package passed v67b independent audit.

## Rebuild Order

The GT foundation was rebuilt and audited in this order:

1. Canonical Excel observation extraction
2. Manual GT v2 assignment
3. Full manual GT audit
4. Final GT export
5. Classification readiness audit
6. Solid foundation snapshot
7. Final GT v2 label propagation
8. Final GT v2 inspection visualizer
9. Tracking helper reattachment
10. Report-ready package
11. Independent package audit
12. Dataset card and documentation

## Source-of-Truth Hierarchy

1. Original Excel annotation source for colour and behaviour.
2. Manual GT v2 review for bbox and identity assignment.
3. Tracking helper only for diagnostic support.

## Downstream Next Step

The next safe stage is v68a strict-gold anchor-frame crop dataset materialization.

Only strict gold rows should be used.
"""

OUT_REPRO.write_text(repro_doc)

readme = f"""# Week8 GT v2 Dataset Documentation

This documentation package accompanies the v67 report-ready GT package.

## Files

- DATASET_CARD_Week8_GT_v2.md
- GT_FORMAT_AND_SCHEMA_Week8_GT_v2.md
- USAGE_AND_CLAIM_BOUNDARIES_Week8_GT_v2.md
- REPRODUCIBILITY_GUIDE_Week8_GT_v2.md
- week8_v67c_key_numbers.csv
- week8_v67c_schema_inventory.csv
- week8_v67c_claim_boundary_checks.csv

## Main Rule

Manual GT v2 is the source of truth. Tracking is diagnostic helper only.

## Main Limitation

This dataset supports baseline/proof-of-concept classification only. It is not the full raw Unibo archive.
"""

OUT_README.write_text(readme)

combined = "\n".join([
    OUT_DATASET_CARD.read_text().lower(),
    OUT_GT_SCHEMA.read_text().lower(),
    OUT_USAGE.read_text().lower(),
    OUT_REPRO.read_text().lower(),
    OUT_README.read_text().lower(),
])

claim_phrases = [
    "manual gt v2 is the source of truth",
    "tracking is diagnostic helper only",
    "baseline/proof-of-concept",
    "not the full raw unibo archive",
    "strict gold rows only",
    "must not be used for training or evaluation",
    "production-grade",
]

claim_rows = []
for phrase in claim_phrases:
    present = phrase in combined
    claim_rows.append({
        "claim_boundary_phrase": phrase,
        "present": present,
    })
    if not present:
        issues.append({
            "item": phrase,
            "issue_type": "hard_missing_claim_boundary_phrase",
            "issue_detail": "Required claim boundary phrase missing from documentation.",
            "severity": "hard",
        })

claim_checks = pd.DataFrame(claim_rows)
safe_to_csv(claim_checks, OUT_CLAIM_CHECKS)

if total_objects != strict_objects + caution_objects + nonusable_objects:
    issues.append({
        "item": "subset_partition",
        "issue_type": "hard_subset_partition_failed",
        "issue_detail": "Strict + caution + nonusable does not equal all reviewed rows.",
        "severity": "hard",
    })

if scanframes != 72:
    issues.append({
        "item": "scanframe_count",
        "issue_type": "hard_scanframe_count_unexpected",
        "issue_detail": f"Expected 72 scanframes, found {scanframes}.",
        "severity": "hard",
    })

if strict_objects != 372:
    issues.append({
        "item": "strict_gold_count",
        "issue_type": "hard_strict_gold_count_unexpected",
        "issue_detail": f"Expected 372 strict gold rows, found {strict_objects}.",
        "severity": "hard",
    })

if tracking_helper_scanframes != 72:
    issues.append({
        "item": "tracking_helper_scanframes",
        "issue_type": "hard_tracking_helper_scanframe_count_unexpected",
        "issue_detail": f"Expected 72 tracking helper scanframes, found {tracking_helper_scanframes}.",
        "severity": "hard",
    })

issues_df = pd.DataFrame(issues, columns=["item", "issue_type", "issue_detail", "severity"])
safe_to_csv(issues_df, OUT_ISSUES)

hard_issue_count = int((issues_df["severity"] == "hard").sum()) if len(issues_df) else 0
warning_count = int((issues_df["severity"] == "warning").sum()) if len(issues_df) else 0
info_count = int((issues_df["severity"] == "info").sum()) if len(issues_df) else 0

if OUT_ZIP.exists():
    OUT_ZIP.unlink()

with zipfile.ZipFile(OUT_ZIP, "w", compression=zipfile.ZIP_DEFLATED) as z:
    for p in sorted(DOC_PKG.rglob("*")):
        if p.is_file():
            z.write(p, p.relative_to(OUT))

doc_zip_sha = sha256_file(OUT_ZIP)
OUT_SHA256.write_text(f"{doc_zip_sha}  {OUT_ZIP.name}\n")

decision = pd.DataFrame([{
    "v67c_decision": "dataset_card_gt_documentation_completed" if hard_issue_count == 0 else "dataset_card_gt_documentation_has_blocking_issues",
    "total_reviewed_objects": total_objects,
    "strict_gold_objects": strict_objects,
    "caution_objects": caution_objects,
    "nonusable_fix_or_excluded_objects": nonusable_objects,
    "scanframes": scanframes,
    "behaviour_classes": behaviour_classes,
    "frame_object_rows": frame_object_rows,
    "strict_gold_frame_object_rows": strict_frame_object_rows,
    "tracking_helper_scanframes": tracking_helper_scanframes,
    "documentation_package": str(DOC_PKG),
    "zip_path": str(OUT_ZIP),
    "zip_sha256": doc_zip_sha,
    "hard_issue_count": hard_issue_count,
    "warning_count": warning_count,
    "info_count": info_count,
    "issue_count": int(len(issues_df)),
    "ready_for_v68a_crop_dataset_materialization": bool(hard_issue_count == 0),
    "generated_at": datetime.now().isoformat(timespec="seconds"),
}])

safe_to_csv(decision, OUT_DECISION)

OUT_NOTE.write_text(
    "# Week 8 v67c Dataset Card and GT Documentation\n\n"
    f"- v67c decision: {decision.iloc[0]['v67c_decision']}\n"
    f"- Reviewed objects: {total_objects}\n"
    f"- Strict gold objects: {strict_objects}\n"
    f"- Caution objects: {caution_objects}\n"
    f"- Nonusable/fix/excluded objects: {nonusable_objects}\n"
    f"- Scanframes: {scanframes}\n"
    f"- Behaviour classes: {behaviour_classes}\n"
    f"- Tracking helper scanframes: {tracking_helper_scanframes}\n"
    f"- Hard issues: {hard_issue_count}\n"
    f"- Documentation package: {DOC_PKG}\n"
    f"- Zip: {OUT_ZIP}\n"
    f"- SHA256: {doc_zip_sha}\n"
    f"- Ready for v68a crop dataset materialization: {bool(hard_issue_count == 0)}\n\n"
    "The documentation explicitly states source-of-truth hierarchy, usage boundaries, claim boundaries, and limitations.\n"
)

OUT_REPORT.write_text(
    "# Week 8 v67c Dataset Card and GT Documentation Report\n\n"
    f"Decision: {decision.iloc[0]['v67c_decision']}\n\n"
    f"Documentation package: {DOC_PKG}\n\n"
    f"ZIP: {OUT_ZIP}\n\n"
    f"SHA256: {doc_zip_sha}\n"
)

progress_row = pd.DataFrame([{
    "date": datetime.now().date().isoformat(),
    "stage": "v67c",
    "task_name": "Dataset card and GT documentation",
    "status": "PASS" if hard_issue_count == 0 else "NEEDS_FIX",
    "input_summary": str(PKG_SRC),
    "output_summary": str(DOC_PKG),
    "hard_issues": hard_issue_count,
    "warnings": warning_count,
    "next_action": "Start v68a strict-gold anchor-frame crop dataset materialization." if hard_issue_count == 0 else "Fix documentation issues before v68a.",
}])

if OUT_PROGRESS.exists():
    old = pd.read_csv(OUT_PROGRESS)
    progress = pd.concat([old, progress_row], ignore_index=True)
else:
    progress = progress_row

safe_to_csv(progress, OUT_PROGRESS)

print("Saved:")
print(OUT_DATASET_CARD)
print(OUT_GT_SCHEMA)
print(OUT_USAGE)
print(OUT_REPRO)
print(OUT_KEY_NUMBERS)
print(OUT_SCHEMA_INVENTORY)
print(OUT_CLAIM_CHECKS)
print(OUT_ZIP)
print(OUT_SHA256)
print(OUT_DECISION)
print(OUT_NOTE)

print()
print("=== v67c decision ===")
print(decision.to_string(index=False))

print()
print("=== key numbers ===")
print(key_numbers.to_string(index=False))

print()
print("=== claim checks ===")
print(claim_checks.to_string(index=False))

print()
print("=== issues ===")
if len(issues_df):
    print(issues_df.to_string(index=False))
else:
    print("No issues found.")
