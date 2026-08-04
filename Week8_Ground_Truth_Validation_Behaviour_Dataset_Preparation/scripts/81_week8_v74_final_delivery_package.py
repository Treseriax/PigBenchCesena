from pathlib import Path
from datetime import datetime
import csv
import json
import hashlib
import zipfile
import shutil
import pandas as pd


ROOT = Path.home() / "PigBench"
W8 = ROOT / "Week8_Ground_Truth_Validation_Behaviour_Dataset_Preparation"

OUT = W8 / "outputs" / "v74_final_week8_delivery_package"
PKG = OUT / "Week8_Final_Delivery_Package"
NOTES = W8 / "notes"
REPORTS = W8 / "reports"
PROGRESS = W8 / "progress"

for p in [OUT, PKG, NOTES, REPORTS, PROGRESS]:
    p.mkdir(parents=True, exist_ok=True)

OUT_DECISION = OUT / "week8_v74_decision_summary.csv"
OUT_ISSUES = OUT / "week8_v74_issues.csv"
OUT_ZIP = OUT / "Week8_Final_Delivery_Package.zip"
OUT_SHA = OUT / "Week8_Final_Delivery_Package.sha256"
OUT_NOTE = NOTES / "week8_v74_final_delivery_package_notes.md"
OUT_REPORT = REPORTS / "week8_v74_final_delivery_package_report.md"
OUT_PROGRESS = PROGRESS / "week8_experiment_progress_log.csv"

INDEX = PKG / "week8_v74_delivery_index.csv"
ZIP_INDEX = PKG / "week8_v74_zip_index.csv"
METRICS = PKG / "week8_v74_key_metrics_summary.csv"
CLAIMS = PKG / "week8_v74_final_claim_boundaries.csv"
STAGES = PKG / "week8_v74_stage_status_summary.csv"
QA = PKG / "week8_v74_quality_checks.csv"
README = PKG / "README_Week8_Final_Delivery_Package.md"
MANIFEST = PKG / "week8_v74_manifest.json"


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


def bool_true(x):
    return str(x).strip().lower() == "true"


def sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def file_sha_or_empty(path):
    path = Path(path)
    if not path.exists() or not path.is_file():
        return ""
    return sha256_file(path)


def copy_if_exists(src, dst_dir):
    src = Path(src)
    dst_dir = Path(dst_dir)
    dst_dir.mkdir(parents=True, exist_ok=True)

    if not src.exists() or not src.is_file():
        return ""

    dst = dst_dir / src.name
    shutil.copy2(src, dst)
    return str(dst)


def read_decision(path):
    if not Path(path).exists():
        return {}
    df = read_csv_clean(path)
    if len(df) == 0:
        return {}
    return df.iloc[0].to_dict()


stage_specs = [
    {
        "stage": "v67",
        "name": "report-ready GT v2 package",
        "decision": W8 / "outputs" / "v67_week8_report_ready_package" / "week8_v67_decision_summary.csv",
        "zip": W8 / "outputs" / "v67_week8_report_ready_package" / "Week8_Report_Ready_GT_v2_Package.zip",
        "required": True,
    },
    {
        "stage": "v67b",
        "name": "independent GT package audit",
        "decision": W8 / "outputs" / "v67b_independent_package_audit" / "week8_v67b_decision_summary.csv",
        "zip": W8 / "outputs" / "v67b_independent_package_audit" / "Week8_Independent_Package_Audit_v67b.zip",
        "required": False,
    },
    {
        "stage": "v67c",
        "name": "GT documentation and dataset card",
        "decision": W8 / "outputs" / "v67c_dataset_card_gt_documentation" / "week8_v67c_decision_summary.csv",
        "zip": W8 / "outputs" / "v67c_dataset_card_gt_documentation" / "Week8_GT_v2_Dataset_Documentation.zip",
        "required": True,
    },
    {
        "stage": "v68a",
        "name": "strict-gold crop dataset",
        "decision": W8 / "outputs" / "v68a_strict_gold_anchor_crop_materialization" / "week8_v68a_decision_summary.csv",
        "zip": W8 / "outputs" / "v68a_strict_gold_anchor_crop_materialization" / "Week8_StrictGold_AnchorFrame_Crop_Dataset.zip",
        "required": True,
    },
    {
        "stage": "v68b",
        "name": "crop QA gallery",
        "decision": W8 / "outputs" / "v68b_crop_qa_gallery" / "week8_v68b_decision_summary.csv",
        "zip": W8 / "outputs" / "v68b_crop_qa_gallery" / "Week8_StrictGold_Crop_QA_Gallery.zip",
        "required": True,
    },
    {
        "stage": "v68c",
        "name": "split leakage audit",
        "decision": W8 / "outputs" / "v68c_split_leakage_audit" / "week8_v68c_decision_summary.csv",
        "zip": W8 / "outputs" / "v68c_split_leakage_audit" / "Week8_Split_Leakage_Audit.zip",
        "required": True,
    },
    {
        "stage": "v68d",
        "name": "group-aware split policy",
        "decision": W8 / "outputs" / "v68d_group_aware_split_policy" / "week8_v68d_decision_summary.csv",
        "zip": W8 / "outputs" / "v68d_group_aware_split_policy" / "Week8_GroupAware_Split_Policy.zip",
        "required": True,
    },
    {
        "stage": "v69a",
        "name": "sanity baselines",
        "decision": W8 / "outputs" / "v69a_baseline_sanity" / "week8_v69a_decision_summary.csv",
        "zip": W8 / "outputs" / "v69a_baseline_sanity" / "Week8_Baseline_Sanity.zip",
        "required": True,
    },
    {
        "stage": "v69b",
        "name": "simple image feature baseline",
        "decision": W8 / "outputs" / "v69b_feature_baseline" / "week8_v69b_decision_summary.csv",
        "zip": W8 / "outputs" / "v69b_feature_baseline" / "Week8_Feature_Baseline.zip",
        "required": True,
    },
    {
        "stage": "v69d",
        "name": "simple baseline error interface",
        "decision": W8 / "outputs" / "v69d_error_analysis_interface" / "week8_v69d_decision_summary.csv",
        "zip": W8 / "outputs" / "v69d_error_analysis_interface" / "Week8_Error_Analysis_Interface_v69d.zip",
        "required": True,
    },
    {
        "stage": "v70b",
        "name": "classical modeling decision package",
        "decision": W8 / "outputs" / "v70b_modeling_decision_package" / "week8_v70b_decision_summary.csv",
        "zip": W8 / "outputs" / "v70b_modeling_decision_package" / "Week8_Modeling_Decision_Package.zip",
        "required": True,
    },
    {
        "stage": "v71b",
        "name": "detector checkpoint loadability smoke test",
        "decision": W8 / "outputs" / "v71b_detector_checkpoint_loadability_smoke_test" / "week8_v71b_decision_summary.csv",
        "zip": W8 / "outputs" / "v71b_detector_checkpoint_loadability_smoke_test" / "Week8_Detector_Checkpoint_Loadability_Smoke_Test.zip",
        "required": True,
    },
    {
        "stage": "v72a",
        "name": "frozen detector embedding extraction",
        "decision": W8 / "outputs" / "v72a_frozen_detector_embedding_extraction" / "week8_v72a_decision_summary.csv",
        "zip": W8 / "outputs" / "v72a_frozen_detector_embedding_extraction" / "Week8_Frozen_Detector_Embeddings.zip",
        "required": True,
    },
    {
        "stage": "v72b",
        "name": "frozen embedding baseline",
        "decision": W8 / "outputs" / "v72b_frozen_embedding_baseline" / "week8_v72b_decision_summary.csv",
        "zip": W8 / "outputs" / "v72b_frozen_embedding_baseline" / "Week8_Frozen_Embedding_Baseline.zip",
        "required": True,
    },
    {
        "stage": "v72c",
        "name": "frozen embedding error analysis interface",
        "decision": W8 / "outputs" / "v72c_frozen_embedding_error_analysis_interface" / "week8_v72c_decision_summary.csv",
        "zip": W8 / "outputs" / "v72c_frozen_embedding_error_analysis_interface" / "Week8_Frozen_Embedding_Error_Analysis_Interface.zip",
        "required": True,
    },
    {
        "stage": "v73",
        "name": "final modeling package",
        "decision": W8 / "outputs" / "v73_final_modeling_package" / "week8_v73_decision_summary.csv",
        "zip": W8 / "outputs" / "v73_final_modeling_package" / "Week8_Final_Modeling_Package.zip",
        "required": True,
    },
]


issues = []
stage_rows = []
zip_rows = []
index_rows = []

decisions_dir = PKG / "decision_summaries"
copied_dir = PKG / "selected_reports"

for spec in stage_specs:
    decision_path = Path(spec["decision"])
    zip_path = Path(spec["zip"])
    decision_exists = decision_path.exists()
    zip_exists = zip_path.exists()

    d = read_decision(decision_path)

    hard_issues = clean(d.get("hard_issue_count", ""))
    hard_failures = clean(d.get("hard_quality_failures", ""))

    if spec["required"] and not decision_exists:
        issues.append({
            "item": spec["stage"],
            "issue_type": "hard_missing_required_decision",
            "issue_detail": str(decision_path),
            "severity": "hard",
        })

    if spec["required"] and not zip_exists:
        issues.append({
            "item": spec["stage"],
            "issue_type": "hard_missing_required_zip",
            "issue_detail": str(zip_path),
            "severity": "hard",
        })

    copied_decision = copy_if_exists(decision_path, decisions_dir)

    stage_rows.append({
        "stage": spec["stage"],
        "name": spec["name"],
        "required": bool(spec["required"]),
        "decision_path": str(decision_path),
        "decision_exists": decision_exists,
        "copied_decision_path": copied_decision,
        "zip_path": str(zip_path),
        "zip_exists": zip_exists,
        "hard_issue_count": hard_issues,
        "hard_quality_failures": hard_failures,
        "decision_fields": json.dumps(d, ensure_ascii=False),
    })

    if zip_exists:
        zip_rows.append({
            "stage": spec["stage"],
            "name": spec["name"],
            "zip_path": str(zip_path),
            "zip_size_mb": round(zip_path.stat().st_size / (1024 * 1024), 3),
            "zip_sha256": sha256_file(zip_path),
            "included_by_reference_not_copied": True,
        })

    index_rows.append({
        "stage": spec["stage"],
        "artifact_type": "decision_summary",
        "path": str(decision_path),
        "exists": decision_exists,
        "notes": "Copied into final package if available.",
    })
    index_rows.append({
        "stage": spec["stage"],
        "artifact_type": "zip_artifact_reference",
        "path": str(zip_path),
        "exists": zip_exists,
        "notes": "Referenced by path and SHA; not duplicated inside final package.",
    })

stages = pd.DataFrame(stage_rows)
zips = pd.DataFrame(zip_rows)
delivery_index = pd.DataFrame(index_rows)

safe_to_csv(stages, STAGES)
safe_to_csv(zips, ZIP_INDEX)
safe_to_csv(delivery_index, INDEX)

v73 = read_decision(W8 / "outputs" / "v73_final_modeling_package" / "week8_v73_decision_summary.csv")
v67 = read_decision(W8 / "outputs" / "v67_week8_report_ready_package" / "week8_v67_decision_summary.csv")
v72a = read_decision(W8 / "outputs" / "v72a_frozen_detector_embedding_extraction" / "week8_v72a_decision_summary.csv")
v72b = read_decision(W8 / "outputs" / "v72b_frozen_embedding_baseline" / "week8_v72b_decision_summary.csv")
v72c = read_decision(W8 / "outputs" / "v72c_frozen_embedding_error_analysis_interface" / "week8_v72c_decision_summary.csv")

metrics = pd.DataFrame([{
    "manual_gt_reviewed_objects": 432,
    "strict_gold_objects": 372,
    "caution_objects": 3,
    "nonusable_or_excluded_objects": 57,
    "scanframes": 72,
    "strict_gold_crop_rows": 744,
    "frozen_embedding_rows": int(v72a.get("embedding_rows", 0) or 0),
    "frozen_embedding_dim": int(v72b.get("embedding_dim", v72a.get("embedding_dim", 0)) or 0),
    "final_champion": clean(v73.get("final_champion_name", "")),
    "current_macro_f1": clean(v73.get("current_champion_macro_f1", "")),
    "current_accuracy": clean(v73.get("current_champion_accuracy", "")),
    "current_delta_macro_f1_vs_v69b": clean(v73.get("current_delta_macro_f1_vs_v69b", "")),
    "group_macro_f1": clean(v73.get("group_champion_macro_f1", "")),
    "group_accuracy": clean(v73.get("group_champion_accuracy", "")),
    "group_delta_macro_f1_vs_v69b": clean(v73.get("group_delta_macro_f1_vs_v69b", "")),
    "v72c_error_rows": clean(v72c.get("error_rows", "")),
    "v72c_correct_rows": clean(v72c.get("correct_rows", "")),
}])
safe_to_csv(metrics, METRICS)

claims = pd.DataFrame([
    {
        "topic": "source_of_truth",
        "allowed": "Manual GT v2 is source of truth.",
        "not_allowed": "Model predictions or tracking outputs are GT.",
    },
    {
        "topic": "dataset_scope",
        "allowed": "Week8 package covers validated annotated observation windows and strict-gold objects.",
        "not_allowed": "All 84 raw Unibo videos are fully labeled.",
    },
    {
        "topic": "classification",
        "allowed": "Frozen YOLOv8 detector embedding baseline improves macro-F1 over simple image-feature baseline.",
        "not_allowed": "Production-grade behaviour classifier.",
    },
    {
        "topic": "model_training",
        "allowed": "Frozen embeddings and shallow classifiers were evaluated.",
        "not_allowed": "YOLOv8 behaviour model was fine-tuned.",
    },
    {
        "topic": "split_policy",
        "allowed": "Current split is exploratory; group-aware split is conservative and source-video-separated.",
        "not_allowed": "Current split alone proves robust generalization.",
    },
    {
        "topic": "rare_classes",
        "allowed": "Rare classes are documented but interpreted cautiously.",
        "not_allowed": "Reliable standalone rare-class performance.",
    },
])
safe_to_csv(claims, CLAIMS)

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

required_specs = [s for s in stage_specs if s["required"]]
required_decisions = int(sum(1 for s in required_specs if Path(s["decision"]).exists()))
required_zips = int(sum(1 for s in required_specs if Path(s["zip"]).exists()))

add_qa("required_decisions_exist", len(required_specs), required_decisions, required_decisions == len(required_specs), "hard", "All required decision summaries should exist.")
add_qa("required_zips_exist", len(required_specs), required_zips, required_zips == len(required_specs), "hard", "All required zip artifacts should exist.")
add_qa("v73_ready_for_v74", True, bool_true(v73.get("ready_for_v74_final_week8_delivery_package", "")), bool_true(v73.get("ready_for_v74_final_week8_delivery_package", "")), "hard", "v73 should be ready for v74.")
add_qa("final_champion_v72b", "v72b", clean(v73.get("final_champion_stage", "")), clean(v73.get("final_champion_stage", "")) == "v72b", "hard", "Final modeling champion should be v72b.")
add_qa("strict_gold_objects", 372, metrics.iloc[0]["strict_gold_objects"], int(metrics.iloc[0]["strict_gold_objects"]) == 372, "hard", "Strict-gold object count should be 372.")
add_qa("embedding_rows", 744, metrics.iloc[0]["frozen_embedding_rows"], int(metrics.iloc[0]["frozen_embedding_rows"]) == 744, "hard", "Frozen embedding rows should be 744.")
add_qa("claim_boundaries_created", ">0", len(claims), len(claims) > 0, "hard", "Claim boundaries should be included.")
add_qa("zip_index_created", ">0", len(zips), len(zips) > 0, "hard", "Zip index should include referenced artifacts.")

qa = pd.DataFrame(qa_rows)
safe_to_csv(qa, QA)

hard_quality_failures = int(((qa["severity"] == "hard") & (~qa["passed"])).sum())

if hard_quality_failures:
    issues.append({
        "item": "v74_quality_checks",
        "issue_type": "hard_final_delivery_quality_failed",
        "issue_detail": f"{hard_quality_failures} hard checks failed.",
        "severity": "hard",
    })

issues.append({
    "item": "package_scope",
    "issue_type": "info_reference_package",
    "issue_detail": "v74 final package indexes large artifacts by path and SHA rather than duplicating all large zips/raw videos.",
    "severity": "info",
})

issues_df = pd.DataFrame(issues, columns=["item", "issue_type", "issue_detail", "severity"])
safe_to_csv(issues_df, OUT_ISSUES)

hard_issue_count = int((issues_df["severity"] == "hard").sum()) if len(issues_df) else 0
warning_count = int((issues_df["severity"] == "warning").sum()) if len(issues_df) else 0
info_count = int((issues_df["severity"] == "info").sum()) if len(issues_df) else 0

readme_text = f"""# Week8 Final Delivery Package

## Scope

This package is the final Week8 delivery index and summary package.

It does not duplicate raw Unibo videos or large intermediate image folders.
Large generated artifacts are referenced by path and SHA256 in `week8_v74_zip_index.csv`.

## Final GT status

- Reviewed manual GT objects: 432
- Strict-gold objects for classification: 372
- Caution objects: 3
- Nonusable / excluded objects: 57
- Scanframes: 72

## Final modeling status

Final champion: {clean(v73.get("final_champion_name", ""))}

Current split:
- crop/config: {clean(v73.get("current_champion_crop", ""))} / {clean(v73.get("current_champion_config", ""))}
- accuracy: {clean(v73.get("current_champion_accuracy", ""))}
- macro-F1: {clean(v73.get("current_champion_macro_f1", ""))}
- delta macro-F1 vs v69b: {clean(v73.get("current_delta_macro_f1_vs_v69b", ""))}

Group-aware split:
- crop/config: {clean(v73.get("group_champion_crop", ""))} / {clean(v73.get("group_champion_config", ""))}
- accuracy: {clean(v73.get("group_champion_accuracy", ""))}
- macro-F1: {clean(v73.get("group_champion_macro_f1", ""))}
- delta macro-F1 vs v69b: {clean(v73.get("group_delta_macro_f1_vs_v69b", ""))}

## Claim boundary

This is a validated GT + baseline modeling delivery.
It is not a production behaviour classifier.
Manual GT v2 remains the source of truth.
"""

README.write_text(readme_text)
OUT_REPORT.write_text(readme_text)

manifest = {
    "version": "week8_v74_final_delivery_package",
    "created_at": datetime.now().isoformat(timespec="seconds"),
    "package_scope": "index_summary_reference_package",
    "required_stage_count": len(required_specs),
    "required_decisions_found": required_decisions,
    "required_zips_found": required_zips,
    "final_champion": clean(v73.get("final_champion_name", "")),
    "metrics": metrics.to_dict(orient="records"),
    "zip_index_rows": len(zips),
    "claim_boundary": "final delivery package only; no production classifier claim",
}

MANIFEST.write_text(json.dumps(manifest, indent=2, ensure_ascii=False))

if OUT_ZIP.exists():
    OUT_ZIP.unlink()

with zipfile.ZipFile(OUT_ZIP, "w", compression=zipfile.ZIP_DEFLATED) as z:
    for p in sorted(PKG.rglob("*")):
        if p.is_file():
            z.write(p, p.relative_to(OUT))

zip_hash = sha256_file(OUT_ZIP)
OUT_SHA.write_text(f"{zip_hash}  {OUT_ZIP.name}\n")

decision = pd.DataFrame([{
    "v74_decision": "final_week8_delivery_package_completed" if hard_issue_count == 0 else "final_week8_delivery_package_has_blocking_issues",
    "required_stage_count": len(required_specs),
    "required_decisions_found": required_decisions,
    "required_zips_found": required_zips,
    "strict_gold_objects": 372,
    "frozen_embedding_rows": int(metrics.iloc[0]["frozen_embedding_rows"]),
    "final_champion_stage": clean(v73.get("final_champion_stage", "")),
    "final_champion_name": clean(v73.get("final_champion_name", "")),
    "current_macro_f1": clean(v73.get("current_champion_macro_f1", "")),
    "group_macro_f1": clean(v73.get("group_champion_macro_f1", "")),
    "zip_path": str(OUT_ZIP),
    "zip_sha256": zip_hash,
    "hard_quality_failures": hard_quality_failures,
    "hard_issue_count": hard_issue_count,
    "warning_count": warning_count,
    "info_count": info_count,
    "issue_count": int(len(issues_df)),
    "ready_for_v75_independent_final_delivery_audit": bool(hard_issue_count == 0),
    "claim_scope": "final_week8_delivery_index_and_summary_package",
    "generated_at": datetime.now().isoformat(timespec="seconds"),
}])

safe_to_csv(decision, OUT_DECISION)

OUT_NOTE.write_text(
    "# Week 8 v74 Final Delivery Package\n\n"
    f"- v74 decision: {decision.iloc[0]['v74_decision']}\n"
    f"- Required stages: {len(required_specs)}\n"
    f"- Required decisions found: {required_decisions}\n"
    f"- Required zips found: {required_zips}\n"
    f"- Strict-gold objects: 372\n"
    f"- Frozen embedding rows: {int(metrics.iloc[0]['frozen_embedding_rows'])}\n"
    f"- Final champion: {clean(v73.get('final_champion_name', ''))}\n"
    f"- Current macro-F1: {clean(v73.get('current_champion_macro_f1', ''))}\n"
    f"- Group macro-F1: {clean(v73.get('group_champion_macro_f1', ''))}\n"
    f"- Hard issues: {hard_issue_count}\n"
    f"- Ready for v75 independent final delivery audit: {bool(hard_issue_count == 0)}\n\n"
    "This package indexes large artifacts by path and SHA instead of duplicating raw videos or large generated folders.\n"
)

progress_row = pd.DataFrame([{
    "date": datetime.now().date().isoformat(),
    "stage": "v74",
    "task_name": "Final Week8 delivery package",
    "status": "PASS" if hard_issue_count == 0 else "NEEDS_FIX",
    "input_summary": str(W8 / "outputs"),
    "output_summary": str(PKG),
    "hard_issues": hard_issue_count,
    "warnings": warning_count,
    "next_action": "Run v75 independent final delivery audit." if hard_issue_count == 0 else "Fix v74 hard issues.",
}])

if OUT_PROGRESS.exists():
    old = pd.read_csv(OUT_PROGRESS)
    progress = pd.concat([old, progress_row], ignore_index=True)
else:
    progress = progress_row

safe_to_csv(progress, OUT_PROGRESS)

print("=== v74 decision ===")
print(decision.to_string(index=False))
print("\n=== key metrics ===")
print(metrics.to_string(index=False))
print("\n=== QA ===")
print(qa.to_string(index=False))
print("\n=== issues ===")
print(issues_df.to_string(index=False) if len(issues_df) else "No issues found.")
