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

OUT = W8 / "outputs" / "v67_week8_report_ready_package"
PKG = OUT / "Week8_Report_Ready_GT_v2_Package"
DATA = PKG / "data"
DOCS = PKG / "docs"
VIS = PKG / "visualizer"
AUDIT = PKG / "audit"
TRACKING = PKG / "tracking_helper"
NOTES_DIR = W8 / "notes"
REPORTS_DIR = W8 / "reports"
PROGRESS_DIR = W8 / "progress"

for p in [OUT, PKG, DATA, DOCS, VIS, AUDIT, TRACKING, NOTES_DIR, REPORTS_DIR, PROGRESS_DIR]:
    p.mkdir(parents=True, exist_ok=True)

V64B = W8 / "outputs" / "v64b_full_manual_gt_v2_audit"
V64C = W8 / "outputs" / "v64c_final_gt_v2_export"
V65A = W8 / "outputs" / "v65a_classification_readiness_audit"
V65B = W8 / "outputs" / "v65b_gt_v2_solid_foundation_snapshot"
V66A = W8 / "outputs" / "v66a_final_gt_v2_label_propagation"
V66B = W8 / "outputs" / "v66b_final_gt_v2_inspection_visualizer"
V66C = W8 / "outputs" / "v66c_tracking_helper_reattachment"

FINAL_GT = V64C / "Week8_Final_GT_v2"

REQ = {
    "v64b_decision": V64B / "week8_v64b_decision_summary.csv",
    "v64b_issues": V64B / "week8_v64b_issues.csv",
    "v64c_decision": V64C / "week8_v64c_decision_summary.csv",
    "v64c_issues": V64C / "week8_v64c_issues.csv",
    "v65a_decision": V65A / "week8_v65a_decision_summary.csv",
    "v65a_behaviour": V65A / "week8_v65a_strict_gold_behaviour_distribution.csv",
    "v65a_limitations": V65A / "week8_v65a_classification_limitations.csv",
    "v65b_decision": V65B / "week8_v65b_decision_summary.csv",
    "v65b_issues": V65B / "week8_v65b_issues.csv",
    "v66a_decision": V66A / "week8_v66a_decision_summary.csv",
    "v66a_issues": V66A / "week8_v66a_issues.csv",
    "v66b_decision": V66B / "week8_v66b_decision_summary.csv",
    "v66b_issues": V66B / "week8_v66b_issues.csv",
    "v66c_decision": V66C / "week8_v66c_decision_summary.csv",
    "v66c_issues": V66C / "week8_v66c_issues.csv",
    "final_gt_all": FINAL_GT / "week8_final_gt_v2_all_reviewed_objects.csv",
    "final_gt_strict": FINAL_GT / "week8_final_gt_v2_strict_gold_objects_for_classification.csv",
    "final_gt_caution": FINAL_GT / "week8_final_gt_v2_caution_objects_for_analysis.csv",
    "final_gt_nonusable": FINAL_GT / "week8_final_gt_v2_nonusable_fix_or_excluded_objects.csv",
    "final_gt_scanframe": FINAL_GT / "week8_final_gt_v2_scanframe_quality_summary.csv",
    "final_gt_manifest": FINAL_GT / "week8_final_gt_v2_manifest.json",
    "v66a_objects": V66A / "week8_v66a_final_gt_v2_object_table.csv",
    "v66a_scanframes": V66A / "week8_v66a_scanframe_propagation_summary.csv",
    "v66a_qa": V66A / "week8_v66a_propagation_quality_checks.csv",
    "v66a_behaviour_category": V66A / "week8_v66a_behaviour_category_distribution.csv",
    "v66b_json": V66B / "week8_v66b_final_gt_v2_visualizer_data.json",
    "v66c_scanframe": V66C / "week8_v66c_scanframe_tracking_helper_summary.csv",
    "v66c_object": V66C / "week8_v66c_final_gt_v2_object_tracking_helper_summary.csv",
    "v66c_inventory": V66C / "week8_v66c_tracking_helper_file_inventory.csv",
}

OUT_DECISION = OUT / "week8_v67_decision_summary.csv"
OUT_ISSUES = OUT / "week8_v67_issues.csv"
OUT_MANIFEST = PKG / "week8_v67_manifest.json"
OUT_README = PKG / "README_Week8_Report_Ready_GT_v2_Package.md"
OUT_FINAL_REPORT = PKG / "Week8_Final_GT_v2_Report.md"
OUT_HASHES = PKG / "week8_v67_file_hashes.csv"
OUT_ZIP = OUT / "Week8_Report_Ready_GT_v2_Package.zip"
OUT_SHA256 = OUT / "Week8_Report_Ready_GT_v2_Package.sha256"
OUT_NOTE = NOTES_DIR / "week8_v67_report_ready_package_notes.md"
OUT_REPORT = REPORTS_DIR / "week8_v67_report_ready_package_report.md"
OUT_PROGRESS = PROGRESS_DIR / "week8_experiment_progress_log.csv"


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


def copy_file(src, dst_dir, new_name=None):
    dst_dir.mkdir(parents=True, exist_ok=True)
    dst = dst_dir / (new_name or src.name)
    shutil.copy2(src, dst)
    return dst


def read_csv_clean(path):
    df = pd.read_csv(path).fillna("")
    for c in df.columns:
        if df[c].dtype == object:
            df[c] = df[c].map(clean)
    return df


def bool_true(value):
    return str(value).strip().lower() == "true"


issues = []

for name, path in REQ.items():
    if not path.exists():
        issues.append({
            "item": name,
            "issue_type": "hard_missing_required_file",
            "issue_detail": str(path),
            "severity": "hard",
        })

if issues:
    issues_df = pd.DataFrame(issues)
    safe_to_csv(issues_df, OUT_ISSUES)
    decision = pd.DataFrame([{
        "v67_decision": "report_ready_package_blocked",
        "hard_issue_count": int((issues_df["severity"] == "hard").sum()),
        "ready_for_delivery": False,
        "generated_at": datetime.now().isoformat(timespec="seconds"),
    }])
    safe_to_csv(decision, OUT_DECISION)
    print(decision.to_string(index=False))
    raise SystemExit(1)


v64b_d = read_csv_clean(REQ["v64b_decision"])
v64c_d = read_csv_clean(REQ["v64c_decision"])
v65a_d = read_csv_clean(REQ["v65a_decision"])
v65b_d = read_csv_clean(REQ["v65b_decision"])
v66a_d = read_csv_clean(REQ["v66a_decision"])
v66b_d = read_csv_clean(REQ["v66b_decision"])
v66c_d = read_csv_clean(REQ["v66c_decision"])

readiness_checks = [
    ("v64b_ready", bool_true(v64b_d.iloc[0].get("ready_for_v64c_final_gt_v2_export", ""))),
    ("v64c_ready", bool_true(v64c_d.iloc[0].get("ready_for_classification_baseline", ""))),
    ("v65a_ready", bool_true(v65a_d.iloc[0].get("ready_for_v65b_dataset_materialization", ""))),
    ("v65b_ready", bool_true(v65b_d.iloc[0].get("ready_to_repeat_week8_tasks_on_solid_gt", ""))),
    ("v66a_ready", bool_true(v66a_d.iloc[0].get("ready_for_v66b_final_gt_visualizer", ""))),
    ("v66b_ready", bool_true(v66b_d.iloc[0].get("ready_for_v66c_tracking_helper_reattachment", ""))),
    ("v66c_ready", bool_true(v66c_d.iloc[0].get("ready_for_v67_report_ready_package", ""))),
]

for name, passed in readiness_checks:
    if not passed:
        issues.append({
            "item": name,
            "issue_type": "hard_readiness_check_failed",
            "issue_detail": f"{name} is not ready.",
            "severity": "hard",
        })

all_gt = read_csv_clean(REQ["final_gt_all"])
strict = read_csv_clean(REQ["final_gt_strict"])
caution = read_csv_clean(REQ["final_gt_caution"])
nonusable = read_csv_clean(REQ["final_gt_nonusable"])
scanframe = read_csv_clean(REQ["final_gt_scanframe"])
behaviour = read_csv_clean(REQ["v65a_behaviour"])
limitations = read_csv_clean(REQ["v65a_limitations"])
v66a_objects = read_csv_clean(REQ["v66a_objects"])
v66a_scan = read_csv_clean(REQ["v66a_scanframes"])
v66c_scan = read_csv_clean(REQ["v66c_scanframe"])

strict_count = len(strict)
caution_count = len(caution)
nonusable_count = len(nonusable)
total_count = len(all_gt)
scan_count = all_gt["scan_frame_id"].nunique()

if "scanframe_quality_decision" in scanframe.columns:
    full_strict = int((scanframe["scanframe_quality_decision"] == "full_strict_gold").sum())
    mixed = int(scanframe["scanframe_quality_decision"].astype(str).str.contains("mixed").sum())
    no_usable = int((scanframe["scanframe_quality_decision"] == "no_classification_usable_object").sum())
else:
    full_strict = int(v64c_d.iloc[0].get("full_strict_gold_scanframes", "0"))
    mixed = int(v64c_d.iloc[0].get("mixed_object_level_scanframes", "0"))
    no_usable = int(v64c_d.iloc[0].get("no_classification_usable_scanframes", "0"))

copy_map = [
    (REQ["final_gt_all"], DATA),
    (REQ["final_gt_strict"], DATA),
    (REQ["final_gt_caution"], DATA),
    (REQ["final_gt_nonusable"], DATA),
    (REQ["final_gt_scanframe"], DATA),
    (REQ["final_gt_manifest"], DATA),
    (REQ["v65a_behaviour"], DATA),
    (REQ["v65a_limitations"], DATA),
    (REQ["v66a_objects"], DATA),
    (REQ["v66a_scanframes"], DATA),
    (REQ["v66a_qa"], AUDIT),
    (REQ["v66a_behaviour_category"], DATA),
    (REQ["v66b_json"], VIS),
    (REQ["v66c_scanframe"], TRACKING),
    (REQ["v66c_object"], TRACKING),
    (REQ["v66c_inventory"], TRACKING),
]

for src, dst in copy_map:
    copy_file(src, dst)

for key in [
    "v64b_decision", "v64b_issues",
    "v64c_decision", "v64c_issues",
    "v65a_decision",
    "v65b_decision", "v65b_issues",
    "v66a_decision", "v66a_issues",
    "v66b_decision", "v66b_issues",
    "v66c_decision", "v66c_issues",
]:
    copy_file(REQ[key], AUDIT)

server = W8 / "interface" / "week8_visualizer_server_v66b.py"
static = W8 / "interface" / "static_v66b"

if server.exists():
    copy_file(server, VIS)
else:
    issues.append({
        "item": str(server),
        "issue_type": "warning_missing_visualizer_server",
        "issue_detail": "v66b server not found.",
        "severity": "warning",
    })

if static.exists():
    static_dst = VIS / "static_v66b"
    if static_dst.exists():
        shutil.rmtree(static_dst)
    shutil.copytree(static, static_dst)
else:
    issues.append({
        "item": str(static),
        "issue_type": "warning_missing_visualizer_static",
        "issue_detail": "v66b static directory not found.",
        "severity": "warning",
    })

qa_rows = []


def add_qa(name, expected, actual, passed, severity, detail):
    qa_rows.append({
        "check_name": name,
        "expected": str(expected),
        "actual": str(actual),
        "passed": bool(passed),
        "severity": severity,
        "detail": detail,
    })


add_qa("total_reviewed_objects", 432, total_count, total_count == 432, "hard", "Final GT should contain all 432 reviewed object rows.")
add_qa("subset_partition", total_count, strict_count + caution_count + nonusable_count, strict_count + caution_count + nonusable_count == total_count, "hard", "Strict/caution/nonusable subsets must partition all reviewed rows.")
add_qa("scanframes", 72, scan_count, scan_count == 72, "hard", "All 72 scanframes should be present.")
add_qa("strict_gold_objects_current", strict_count, strict_count, strict_count > 0, "hard", "Strict gold subset must be non-empty.")
add_qa("caution_objects_current", caution_count, caution_count, True, "info", "Caution count is documented.")
add_qa("nonusable_objects_current", nonusable_count, nonusable_count, True, "info", "Nonusable count is documented.")
add_qa("v66a_frame_object_rows", ">0", v66a_d.iloc[0].get("frame_object_rows", "0"), int(v66a_d.iloc[0].get("frame_object_rows", "0")) > 0, "hard", "Frame-object propagation must exist.")
tracking_helper_scanframes_value = int(float(str(v66c_d.iloc[0].get("scanframes_with_tracking_helper", "0")).strip() or 0))
add_qa("tracking_helper_scanframes", 72, tracking_helper_scanframes_value, tracking_helper_scanframes_value == 72, "hard", "Tracking helper should be attached at scanframe level to all 72 scanframes.")
add_qa("tracking_direct_object_matches", "0 allowed", v66c_d.iloc[0].get("direct_object_tracking_matches", "0"), True, "info", "Direct object matches are not required; tracking is diagnostic helper only.")

hard_stage_issues = 0
for d in [v64b_d, v64c_d, v65b_d, v66a_d, v66b_d, v66c_d]:
    try:
        hard_stage_issues += int(d.iloc[0].get("hard_issue_count", "0"))
    except Exception:
        pass

add_qa("stage_hard_issues_zero", 0, hard_stage_issues, hard_stage_issues == 0, "hard", "All critical stages should have zero hard issues.")

qa_df = pd.DataFrame(qa_rows)
safe_to_csv(qa_df, AUDIT / "week8_v67_report_ready_package_quality_checks.csv")

hard_q_fail = int(((qa_df["severity"] == "hard") & (~qa_df["passed"])).sum())
if hard_q_fail:
    issues.append({
        "item": "v67_quality_checks",
        "issue_type": "hard_package_quality_check_failed",
        "issue_detail": f"{hard_q_fail} hard package quality checks failed.",
        "severity": "hard",
    })

beh_table = behaviour.to_string(index=False)
lim_table = limitations.to_string(index=False)

report = f"""# Week 8 Final GT v2 Report-Ready Package

## Executive Summary

This package freezes the Week 8 validated GT v2 dataset after manual review, bbox correction, final propagation, visualizer refresh, and tracking-helper reattachment.

## Final GT v2 Status

- Reviewed canonical object rows: {total_count}
- Annotated scanframes: {scan_count}
- Strict gold objects for classification: {strict_count}
- Caution objects for qualitative analysis: {caution_count}
- Nonusable / fix / excluded objects: {nonusable_count}
- Full strict-gold scanframes: {full_strict}
- Mixed object-level scanframes: {mixed}
- No classification-usable scanframes: {no_usable}

## Source-of-Truth Hierarchy

1. Colour and behaviour labels come from the original Excel annotation source.
2. BBox-to-identity assignments come from manual GT v2 review.
3. Corrected manual bbox rows are accepted only after audit.
4. Tracking is attached only as a diagnostic helper and does not override manual GT v2.

## Propagation

Final GT v2 labels were propagated across the 10-second observation clips. The propagated bbox coordinates are static manual anchor boxes repeated across the clip for label representation. This is not a tracking-quality claim.

## Tracking Helper

Tracking helper files were inventoried and attached at scanframe level. Direct canonical object matches were not available from v52 tracking files, so tracking is used as scanframe-level diagnostic support only.

## Classification Readiness

The strict gold subset supports a baseline/proof-of-concept classifier only. It does not support a production-grade behaviour classifier claim.

## Behaviour Distribution

{beh_table}

## Limitations

{lim_table}

## Usage Rules

- Use week8_final_gt_v2_strict_gold_objects_for_classification.csv for classification training/evaluation.
- Use caution rows only for qualitative analysis.
- Do not use nonusable/fix/excluded rows for training or evaluation.
- Do not claim production-grade tracking or production-grade behaviour classification.
- Report class imbalance and rare-class limitations.

## Final Claim Boundary

This deliverable demonstrates validated dataset preparation and GT foundation building for 72 annotated scanframe clips. It is not the full raw Unibo archive and does not claim complete GT for the full raw Unibo archive.
"""

OUT_FINAL_REPORT.write_text(report)

readme = f"""# Week8 Report-Ready GT v2 Package

## Contents

- data/: final GT v2 data tables and propagation summaries
- audit/: decision summaries, issue files, and quality checks
- visualizer/: final GT v2 inspection visualizer data and server files
- tracking_helper/: tracking helper reattachment summaries
- Week8_Final_GT_v2_Report.md: report-ready summary

## Key Numbers

- Reviewed objects: {total_count}
- Strict gold objects: {strict_count}
- Caution objects: {caution_count}
- Nonusable/fix/excluded objects: {nonusable_count}
- Scanframes: {scan_count}
- Tracking helper scanframes: {v66c_d.iloc[0].get("scanframes_with_tracking_helper", "0")}

## Main Rule

Manual GT v2 is the source of truth. Tracking is diagnostic helper only.
"""

OUT_README.write_text(readme)

manifest = {
    "package_version": "week8_report_ready_gt_v2_v67",
    "created_at": datetime.now().isoformat(timespec="seconds"),
    "key_numbers": {
        "reviewed_objects": int(total_count),
        "scanframes": int(scan_count),
        "strict_gold_objects": int(strict_count),
        "caution_objects": int(caution_count),
        "nonusable_fix_or_excluded_objects": int(nonusable_count),
        "full_strict_gold_scanframes": int(full_strict),
        "mixed_object_level_scanframes": int(mixed),
        "no_classification_usable_scanframes": int(no_usable),
        "tracking_csv_files_found": int(v66c_d.iloc[0].get("tracking_csv_files_found", "0")),
        "tracking_helper_scanframes": int(v66c_d.iloc[0].get("scanframes_with_tracking_helper", "0")),
        "direct_object_tracking_matches": int(v66c_d.iloc[0].get("direct_object_tracking_matches", "0")),
    },
    "claim_boundaries": {
        "classification": "baseline/proof-of-concept only",
        "tracking": "diagnostic helper only",
        "gt_scope": "72 annotated scanframe clips, not the full raw Unibo archive",
    },
    "source_of_truth": {
        "colour_behaviour": "canonical Excel annotation",
        "bbox_identity": "manual GT v2 review and corrected bbox audit",
        "tracking": "helper only, not source of truth",
    },
}

OUT_MANIFEST.write_text(json.dumps(manifest, indent=2, ensure_ascii=False))

hash_rows = []
for p in sorted(PKG.rglob("*")):
    if p.is_file():
        rel = str(p.relative_to(PKG))
        # Do not include the hash manifest itself in its own manifest.
        # Otherwise the file changes after writing and creates a self-hash mismatch.
        if rel == OUT_HASHES.name:
            continue
        hash_rows.append({
            "relative_path": rel,
            "size_bytes": p.stat().st_size,
            "sha256": sha256_file(p),
        })

hash_df = pd.DataFrame(hash_rows)
safe_to_csv(hash_df, OUT_HASHES)

if OUT_ZIP.exists():
    OUT_ZIP.unlink()

with zipfile.ZipFile(OUT_ZIP, "w", compression=zipfile.ZIP_DEFLATED) as z:
    for p in sorted(PKG.rglob("*")):
        if p.is_file():
            z.write(p, p.relative_to(OUT))

zip_hash = sha256_file(OUT_ZIP)
OUT_SHA256.write_text(f"{zip_hash}  {OUT_ZIP.name}\n")

issues_df = pd.DataFrame(issues, columns=["item", "issue_type", "issue_detail", "severity"])
safe_to_csv(issues_df, OUT_ISSUES)

hard_issue_count = int((issues_df["severity"] == "hard").sum()) if len(issues_df) else 0
warning_count = int((issues_df["severity"] == "warning").sum()) if len(issues_df) else 0
info_count = int((issues_df["severity"] == "info").sum()) if len(issues_df) else 0

decision = pd.DataFrame([{
    "v67_decision": "week8_report_ready_package_completed" if hard_issue_count == 0 else "week8_report_ready_package_has_blocking_issues",
    "reviewed_objects": int(total_count),
    "strict_gold_objects": int(strict_count),
    "caution_objects": int(caution_count),
    "nonusable_fix_or_excluded_objects": int(nonusable_count),
    "scanframes": int(scan_count),
    "full_strict_gold_scanframes": int(full_strict),
    "mixed_object_level_scanframes": int(mixed),
    "no_classification_usable_scanframes": int(no_usable),
    "tracking_csv_files_found": int(v66c_d.iloc[0].get("tracking_csv_files_found", "0")),
    "tracking_helper_scanframes": int(v66c_d.iloc[0].get("scanframes_with_tracking_helper", "0")),
    "direct_object_tracking_matches": int(v66c_d.iloc[0].get("direct_object_tracking_matches", "0")),
    "package_dir": str(PKG),
    "zip_path": str(OUT_ZIP),
    "zip_sha256": zip_hash,
    "hard_issue_count": hard_issue_count,
    "warning_count": warning_count,
    "info_count": info_count,
    "issue_count": int(len(issues_df)),
    "ready_for_delivery": bool(hard_issue_count == 0),
    "ready_for_v68_baseline_classification": bool(hard_issue_count == 0),
    "generated_at": datetime.now().isoformat(timespec="seconds"),
}])

safe_to_csv(decision, OUT_DECISION)

OUT_NOTE.write_text(
    "# Week 8 v67 Report-Ready Package\n\n"
    f"- v67 decision: {decision.iloc[0]['v67_decision']}\n"
    f"- Reviewed objects: {total_count}\n"
    f"- Strict gold objects: {strict_count}\n"
    f"- Caution objects: {caution_count}\n"
    f"- Nonusable/fix/excluded objects: {nonusable_count}\n"
    f"- Scanframes: {scan_count}\n"
    f"- Full strict-gold scanframes: {full_strict}\n"
    f"- Mixed object-level scanframes: {mixed}\n"
    f"- Tracking helper scanframes: {v66c_d.iloc[0].get('scanframes_with_tracking_helper', '0')}\n"
    f"- Hard issues: {hard_issue_count}\n"
    f"- Package: {PKG}\n"
    f"- Zip: {OUT_ZIP}\n"
    f"- SHA256: {zip_hash}\n"
    f"- Ready for delivery: {bool(hard_issue_count == 0)}\n\n"
    "Manual GT v2 is the source of truth. Tracking remains diagnostic helper only.\n"
)

OUT_REPORT.write_text(
    "# Week 8 v67 Report-Ready Package Report\n\n"
    f"Decision: {decision.iloc[0]['v67_decision']}\n\n"
    f"Package directory: {PKG}\n\n"
    f"Zip: {OUT_ZIP}\n\n"
    f"SHA256: {zip_hash}\n"
)

progress_row = pd.DataFrame([{
    "date": datetime.now().date().isoformat(),
    "stage": "v67",
    "task_name": "Week8 report-ready GT v2 package",
    "status": "PASS" if hard_issue_count == 0 else "NEEDS_FIX",
    "input_summary": str(W8 / "outputs"),
    "output_summary": str(PKG),
    "hard_issues": hard_issue_count,
    "warnings": warning_count,
    "next_action": "Optional: run v68 baseline classification on strict gold subset only.",
}])

if OUT_PROGRESS.exists():
    old = pd.read_csv(OUT_PROGRESS)
    progress = pd.concat([old, progress_row], ignore_index=True)
else:
    progress = progress_row

safe_to_csv(progress, OUT_PROGRESS)

print("Saved:")
print(PKG)
print(OUT_README)
print(OUT_FINAL_REPORT)
print(OUT_MANIFEST)
print(OUT_HASHES)
print(OUT_ZIP)
print(OUT_SHA256)
print(OUT_DECISION)
print(OUT_NOTE)

print()
print("=== v67 decision ===")
print(decision.to_string(index=False))

print()
print("=== quality checks ===")
print(qa_df.to_string(index=False))

print()
print("=== issues ===")
if len(issues_df):
    print(issues_df.to_string(index=False))
else:
    print("No issues found.")
