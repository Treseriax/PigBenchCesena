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

ASSIGNMENTS = W8 / "validation" / "week8_v63b_manual_gt_v2_assignments.csv"
V62B2_OBS = W8 / "outputs" / "v62b2_corrected_canonical_excel_observation_extraction" / "week8_v62b2_corrected_canonical_colour_behaviour_observations.csv"
V64B = W8 / "outputs" / "v64b_full_manual_gt_v2_audit"

STRICT_GOLD = V64B / "week8_v64b_strict_gold_usable_object_gt.csv"
CAUTION = V64B / "week8_v64b_caution_usable_object_gt.csv"
NONUSABLE = V64B / "week8_v64b_nonusable_or_fix_required_object_gt.csv"
SCANFRAME_QA = V64B / "week8_v64b_scanframe_quality_summary.csv"
COUNTS = V64B / "week8_v64b_counts.csv"
DECISION64B = V64B / "week8_v64b_decision_summary.csv"
ISSUES64B = V64B / "week8_v64b_issues.csv"
INCONSISTENCIES64B = V64B / "week8_v64b_assignment_inconsistencies.csv"

OUT = W8 / "outputs" / "v64c_final_gt_v2_export"
EXPORT = OUT / "Week8_Final_GT_v2"
NOTES = W8 / "notes"
REPORTS = W8 / "reports"
PROGRESS = W8 / "progress"

for p in [OUT, EXPORT, NOTES, REPORTS, PROGRESS]:
    p.mkdir(parents=True, exist_ok=True)

OUT_ALL = EXPORT / "week8_final_gt_v2_all_reviewed_objects.csv"
OUT_STRICT = EXPORT / "week8_final_gt_v2_strict_gold_objects_for_classification.csv"
OUT_CAUTION = EXPORT / "week8_final_gt_v2_caution_objects_for_analysis.csv"
OUT_NONUSABLE = EXPORT / "week8_final_gt_v2_nonusable_fix_or_excluded_objects.csv"
OUT_SCANFRAME = EXPORT / "week8_final_gt_v2_scanframe_quality_summary.csv"
OUT_COUNTS = EXPORT / "week8_final_gt_v2_counts.csv"
OUT_MANIFEST = EXPORT / "week8_final_gt_v2_manifest.json"
OUT_README = EXPORT / "README_Week8_Final_GT_v2.md"

OUT_DECISION = OUT / "week8_v64c_decision_summary.csv"
OUT_ISSUES = OUT / "week8_v64c_issues.csv"
OUT_NOTE = NOTES / "week8_v64c_final_gt_v2_export_notes.md"
OUT_REPORT = REPORTS / "week8_v64c_final_gt_v2_export_report.md"
OUT_PROGRESS = PROGRESS / "week8_experiment_progress_log.csv"
OUT_ZIP = OUT / "Week8_Final_GT_v2_Export.zip"
OUT_SHA256 = OUT / "Week8_Final_GT_v2_Export.sha256"


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
    ASSIGNMENTS,
    V62B2_OBS,
    STRICT_GOLD,
    CAUTION,
    NONUSABLE,
    SCANFRAME_QA,
    COUNTS,
    DECISION64B,
    ISSUES64B,
    INCONSISTENCIES64B,
]

for p in required:
    if not p.exists():
        issues.append({
            "item": str(p),
            "issue_type": "hard_missing_required_input",
            "issue_detail": "Required input for final GT v2 export is missing.",
            "severity": "hard",
        })

if issues:
    issues_df = pd.DataFrame(issues)
    safe_to_csv(issues_df, OUT_ISSUES)
    decision = pd.DataFrame([{
        "v64c_decision": "final_gt_v2_export_blocked",
        "hard_issue_count": int((issues_df["severity"] == "hard").sum()),
        "ready_for_classification_baseline": False,
        "generated_at": datetime.now().isoformat(timespec="seconds"),
    }])
    safe_to_csv(decision, OUT_DECISION)
    print(decision.to_string(index=False))
    raise SystemExit(1)


assignments = pd.read_csv(ASSIGNMENTS).fillna("")
strict = pd.read_csv(STRICT_GOLD).fillna("")
caution = pd.read_csv(CAUTION).fillna("")
nonusable = pd.read_csv(NONUSABLE).fillna("")
scanframe = pd.read_csv(SCANFRAME_QA).fillna("")
counts = pd.read_csv(COUNTS).fillna("")
decision64b = pd.read_csv(DECISION64B).fillna("")
issues64b = pd.read_csv(ISSUES64B).fillna("")
incons64b = pd.read_csv(INCONSISTENCIES64B).fillna("")

for df in [assignments, strict, caution, nonusable, scanframe, counts, decision64b, issues64b, incons64b]:
    for c in df.columns:
        df[c] = df[c].map(clean)

if len(decision64b) == 0 or decision64b.iloc[0].get("ready_for_v64c_final_gt_v2_export", "False") != "True":
    issues.append({
        "item": str(DECISION64B),
        "issue_type": "hard_v64b_not_ready",
        "issue_detail": "v64b decision does not mark ready_for_v64c_final_gt_v2_export=True.",
        "severity": "hard",
    })

if len(assignments) != 432:
    issues.append({
        "item": str(ASSIGNMENTS),
        "issue_type": "hard_unexpected_assignment_row_count",
        "issue_detail": f"Expected 432 rows, found {len(assignments)}.",
        "severity": "hard",
    })

reviewed_rows = int((assignments["manual_gt_v2_status"].astype(str).str.strip() != "").sum())
if reviewed_rows != 432:
    issues.append({
        "item": str(ASSIGNMENTS),
        "issue_type": "hard_not_all_rows_reviewed",
        "issue_detail": f"Expected 432 reviewed rows, found {reviewed_rows}.",
        "severity": "hard",
    })

# Dynamic post-adjust GT: strict count is taken from the current v64b audit output.
# No fixed 363-object assumption here.

hard_incons = incons64b[incons64b.get("audit_severity", "") == "hard"] if len(incons64b) else pd.DataFrame()
if len(hard_incons):
    issues.append({
        "item": str(INCONSISTENCIES64B),
        "issue_type": "hard_remaining_assignment_inconsistencies",
        "issue_detail": f"{len(hard_incons)} hard inconsistencies remain.",
        "severity": "hard",
    })

# Build final all-reviewed table with explicit export category.
all_df = assignments.copy()

strict_keys = set(zip(strict["scan_frame_id"], strict["canonical_colour_label_norm"])) if len(strict) else set()
caution_keys = set(zip(caution["scan_frame_id"], caution["canonical_colour_label_norm"])) if len(caution) else set()

def export_category(row):
    key = (row["scan_frame_id"], row["canonical_colour_label_norm"])
    if key in strict_keys:
        return "strict_gold_classification"
    if key in caution_keys:
        return "caution_analysis"
    return "nonusable_fix_or_excluded"

all_df["final_gt_v2_export_category"] = all_df.apply(export_category, axis=1)

all_df["final_classification_use"] = all_df["final_gt_v2_export_category"].map({
    "strict_gold_classification": "use_for_classification",
    "caution_analysis": "analysis_only_with_caution",
    "nonusable_fix_or_excluded": "do_not_use_for_classification",
})

all_df["final_gt_v2_version"] = "week8_final_gt_v2_v64c"
all_df["final_gt_v2_created_at"] = datetime.now().isoformat(timespec="seconds")

safe_to_csv(all_df, OUT_ALL)
safe_to_csv(strict, OUT_STRICT)
safe_to_csv(caution, OUT_CAUTION)
safe_to_csv(nonusable, OUT_NONUSABLE)
safe_to_csv(scanframe, OUT_SCANFRAME)
safe_to_csv(counts, OUT_COUNTS)

summary = {
    "dataset_version": "week8_final_gt_v2_v64c",
    "created_at": datetime.now().isoformat(timespec="seconds"),
    "scope": {
        "scanframes": int(all_df["scan_frame_id"].nunique()),
        "objects_total": int(len(all_df)),
        "reviewed_objects": reviewed_rows,
        "canonical_colour_labels": sorted(all_df["canonical_colour_label_norm"].unique().tolist()),
        "canonical_behaviour_source": "original_excel_v62b2",
        "bbox_identity_source": "manual_gt_v2_assignment_v63b_reviewed_and_audited_v64b",
    },
    "classification_subsets": {
        "strict_gold_objects_for_classification": int(len(strict)),
        "caution_objects_for_analysis": int(len(caution)),
        "nonusable_fix_or_excluded_objects": int(len(nonusable)),
    },
    "scanframe_quality": {
        "full_strict_gold_scanframes": int((scanframe["scanframe_quality_decision"] == "full_strict_gold").sum()),
        "full_usable_with_caution_scanframes": int((scanframe["scanframe_quality_decision"] == "full_usable_with_caution").sum()),
        "mixed_object_level_scanframes": int(scanframe["scanframe_quality_decision"].astype(str).str.contains("mixed").sum()),
        "no_classification_usable_scanframes": int((scanframe["scanframe_quality_decision"] == "no_classification_usable_object").sum()),
    },
    "source_files": {
        "manual_assignments": str(ASSIGNMENTS),
        "canonical_excel_observations": str(V62B2_OBS),
        "v64b_audit_decision": str(DECISION64B),
        "v64b_counts": str(COUNTS),
    },
    "export_files": {
        "all_reviewed_objects": str(OUT_ALL),
        "strict_gold_objects": str(OUT_STRICT),
        "caution_objects": str(OUT_CAUTION),
        "nonusable_objects": str(OUT_NONUSABLE),
        "scanframe_quality_summary": str(OUT_SCANFRAME),
        "counts": str(OUT_COUNTS),
    },
    "usage_rules": {
        "strict_classification_gt": "Use only rows from week8_final_gt_v2_strict_gold_objects_for_classification.csv.",
        "caution_rows": "Caution rows are not part of strict classification GT; they may be used for qualitative analysis only.",
        "nonusable_rows": "Nonusable/fix/excluded rows must not be used for classification training or metrics.",
    },
}

OUT_MANIFEST.write_text(json.dumps(summary, indent=2, ensure_ascii=False))

readme = f"""# Week 8 Final GT v2 Export

## Decision

This export contains the manually reviewed and audited Week 8 canonical GT v2 dataset.

## Scope

- Scanframes reviewed: {summary['scope']['scanframes']}
- Object rows reviewed: {summary['scope']['objects_total']}
- Strict gold classification objects: {summary['classification_subsets']['strict_gold_objects_for_classification']}
- Caution analysis objects: {summary['classification_subsets']['caution_objects_for_analysis']}
- Nonusable / fix / excluded objects: {summary['classification_subsets']['nonusable_fix_or_excluded_objects']}

## Source-of-truth hierarchy

1. Colour and behaviour labels come from the original Excel annotation extracted in v62b2.
2. Bounding-box-to-identity assignments come from manual GT v2 review in v63b.
3. v64b audit verifies that there are no hard consistency errors.
4. Strict classification GT uses only rows with:
   - manual_gt_v2_status = gold_usable
   - manual_classification_use = use_for_classification
   - manual_bbox_status = bbox_ok
   - manual_identity_status = identity_confirmed
   - manual_assigned_candidate_box_id is not empty

## Files

- `week8_final_gt_v2_all_reviewed_objects.csv`
- `week8_final_gt_v2_strict_gold_objects_for_classification.csv`
- `week8_final_gt_v2_caution_objects_for_analysis.csv`
- `week8_final_gt_v2_nonusable_fix_or_excluded_objects.csv`
- `week8_final_gt_v2_scanframe_quality_summary.csv`
- `week8_final_gt_v2_counts.csv`
- `week8_final_gt_v2_manifest.json`

## Important limitation

This is a curated object-level GT export for the 72 annotated scanframe clips. It is not a claim that the full raw Unibo video archive has complete manual GT.
"""
OUT_README.write_text(readme)

# Zip export folder.
if OUT_ZIP.exists():
    OUT_ZIP.unlink()

with zipfile.ZipFile(OUT_ZIP, "w", compression=zipfile.ZIP_DEFLATED) as z:
    for p in sorted(EXPORT.rglob("*")):
        if p.is_file():
            z.write(p, p.relative_to(OUT))

zip_hash = sha256_file(OUT_ZIP)
OUT_SHA256.write_text(f"{zip_hash}  {OUT_ZIP.name}\n")

issues_df = pd.DataFrame(issues, columns=["item", "issue_type", "issue_detail", "severity"])
safe_to_csv(issues_df, OUT_ISSUES)

hard_issue_count = int((issues_df["severity"] == "hard").sum()) if len(issues_df) else 0
warning_count = int((issues_df["severity"] == "warning").sum()) if len(issues_df) else 0

decision = pd.DataFrame([{
    "v64c_decision": "final_gt_v2_export_completed" if hard_issue_count == 0 else "final_gt_v2_export_created_with_blocking_issues",
    "total_reviewed_object_rows": int(len(all_df)),
    "strict_gold_objects_for_classification": int(len(strict)),
    "caution_objects_for_analysis": int(len(caution)),
    "nonusable_fix_or_excluded_objects": int(len(nonusable)),
    "scanframe_count": int(all_df["scan_frame_id"].nunique()),
    "full_strict_gold_scanframes": summary["scanframe_quality"]["full_strict_gold_scanframes"],
    "full_usable_with_caution_scanframes": summary["scanframe_quality"]["full_usable_with_caution_scanframes"],
    "mixed_object_level_scanframes": summary["scanframe_quality"]["mixed_object_level_scanframes"],
    "no_classification_usable_scanframes": summary["scanframe_quality"]["no_classification_usable_scanframes"],
    "export_dir": str(EXPORT),
    "zip_path": str(OUT_ZIP),
    "zip_sha256": zip_hash,
    "hard_issue_count": hard_issue_count,
    "warning_count": warning_count,
    "issue_count": int(len(issues_df)),
    "ready_for_classification_baseline": bool(hard_issue_count == 0 and len(strict) > 0),
    "generated_at": datetime.now().isoformat(timespec="seconds"),
}])

safe_to_csv(decision, OUT_DECISION)

OUT_NOTE.write_text(
    "# Week 8 v64c Final GT v2 Export\n\n"
    f"- v64c decision: {decision.iloc[0]['v64c_decision']}\n"
    f"- Total reviewed object rows: {len(all_df)}\n"
    f"- Strict gold objects for classification: {len(strict)}\n"
    f"- Caution objects for analysis: {len(caution)}\n"
    f"- Nonusable/fix/excluded objects: {len(nonusable)}\n"
    f"- Scanframes: {all_df['scan_frame_id'].nunique()}\n"
    f"- Full strict-gold scanframes: {summary['scanframe_quality']['full_strict_gold_scanframes']}\n"
    f"- Mixed object-level scanframes: {summary['scanframe_quality']['mixed_object_level_scanframes']}\n"
    f"- Export dir: {EXPORT}\n"
    f"- Zip: {OUT_ZIP}\n"
    f"- SHA256: {zip_hash}\n"
    f"- Ready for classification baseline: {decision.iloc[0]['ready_for_classification_baseline']}\n\n"
    "Use the strict gold CSV for classification baseline. Do not train/evaluate on caution or nonusable rows unless explicitly doing qualitative/error analysis.\n"
)

OUT_REPORT.write_text(
    "# Week 8 v64c Final GT v2 Export Report\n\n"
    f"Decision: {decision.iloc[0]['v64c_decision']}\n\n"
    f"Export directory: `{EXPORT}`\n\n"
    f"Zip: `{OUT_ZIP}`\n\n"
    f"SHA256: `{zip_hash}`\n"
)

progress_row = pd.DataFrame([{
    "date": datetime.now().date().isoformat(),
    "stage": "v64c",
    "task_name": "Final GT v2 export",
    "status": "PASS" if hard_issue_count == 0 else "NEEDS_FIX",
    "input_summary": str(ASSIGNMENTS),
    "output_summary": str(EXPORT),
    "hard_issues": hard_issue_count,
    "warnings": warning_count,
    "next_action": "Run classification baseline on strict gold object GT.",
}])

if OUT_PROGRESS.exists():
    old = pd.read_csv(OUT_PROGRESS)
    progress = pd.concat([old, progress_row], ignore_index=True)
else:
    progress = progress_row

safe_to_csv(progress, OUT_PROGRESS)

print("Saved:")
print(EXPORT)
print(OUT_ALL)
print(OUT_STRICT)
print(OUT_CAUTION)
print(OUT_NONUSABLE)
print(OUT_SCANFRAME)
print(OUT_COUNTS)
print(OUT_MANIFEST)
print(OUT_README)
print(OUT_ZIP)
print(OUT_SHA256)
print(OUT_DECISION)
print(OUT_NOTE)

print()
print("=== v64c decision ===")
print(decision.to_string(index=False))

print()
print("=== manifest summary ===")
print(json.dumps(summary, indent=2, ensure_ascii=False))

print()
print("=== issues ===")
if len(issues_df):
    print(issues_df.to_string(index=False))
else:
    print("No issues found.")
