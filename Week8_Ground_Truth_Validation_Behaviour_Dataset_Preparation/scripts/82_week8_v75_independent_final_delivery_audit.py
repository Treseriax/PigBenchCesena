from pathlib import Path
from datetime import datetime
import csv
import json
import hashlib
import zipfile
import pandas as pd


ROOT = Path.home() / "PigBench"
W8 = ROOT / "Week8_Ground_Truth_Validation_Behaviour_Dataset_Preparation"

V74 = W8 / "outputs" / "v74_final_week8_delivery_package"
V74_DECISION = V74 / "week8_v74_decision_summary.csv"
V74_ZIP = V74 / "Week8_Final_Delivery_Package.zip"
V74_SHA = V74 / "Week8_Final_Delivery_Package.sha256"
V74_PKG = V74 / "Week8_Final_Delivery_Package"

ZIP_INDEX = V74_PKG / "week8_v74_zip_index.csv"
KEY_METRICS = V74_PKG / "week8_v74_key_metrics_summary.csv"
CLAIMS = V74_PKG / "week8_v74_final_claim_boundaries.csv"
QA74 = V74_PKG / "week8_v74_quality_checks.csv"
MANIFEST74 = V74_PKG / "week8_v74_manifest.json"

OUT = W8 / "outputs" / "v75_independent_final_delivery_audit"
PKG = OUT / "Week8_Independent_Final_Delivery_Audit"
NOTES = W8 / "notes"
REPORTS = W8 / "reports"
PROGRESS = W8 / "progress"

for p in [OUT, PKG, NOTES, REPORTS, PROGRESS]:
    p.mkdir(parents=True, exist_ok=True)

OUT_ZIP_AUDIT = PKG / "week8_v75_referenced_zip_integrity_audit.csv"
OUT_PACKAGE_AUDIT = PKG / "week8_v75_final_package_content_audit.csv"
OUT_METRIC_AUDIT = PKG / "week8_v75_final_metric_consistency_audit.csv"
OUT_CLAIM_AUDIT = PKG / "week8_v75_claim_boundary_audit.csv"
OUT_QA = PKG / "week8_v75_quality_checks.csv"
OUT_README = PKG / "README_Week8_Independent_Final_Delivery_Audit.md"
OUT_MANIFEST = PKG / "week8_v75_manifest.json"

OUT_DECISION = OUT / "week8_v75_decision_summary.csv"
OUT_ISSUES = OUT / "week8_v75_issues.csv"
OUT_ZIP = OUT / "Week8_Independent_Final_Delivery_Audit.zip"
OUT_SHA = OUT / "Week8_Independent_Final_Delivery_Audit.sha256"
OUT_NOTE = NOTES / "week8_v75_independent_final_delivery_audit_notes.md"
OUT_REPORT = REPORTS / "week8_v75_independent_final_delivery_audit_report.md"
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


def bool_true(x):
    return str(x).strip().lower() == "true"


def sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def fnum(x):
    try:
        return float(x)
    except Exception:
        return 0.0


issues = []

required_inputs = [
    V74_DECISION,
    V74_ZIP,
    V74_SHA,
    ZIP_INDEX,
    KEY_METRICS,
    CLAIMS,
    QA74,
    MANIFEST74,
]

for p in required_inputs:
    if not Path(p).exists():
        issues.append({
            "item": str(p),
            "issue_type": "hard_missing_v74_input",
            "issue_detail": "Required v74 input missing.",
            "severity": "hard",
        })

if issues:
    issues_df = pd.DataFrame(issues)
    safe_to_csv(issues_df, OUT_ISSUES)
    decision = pd.DataFrame([{
        "v75_decision": "independent_final_delivery_audit_blocked",
        "hard_issue_count": int((issues_df["severity"] == "hard").sum()),
        "final_delivery_ready": False,
        "generated_at": datetime.now().isoformat(timespec="seconds"),
    }])
    safe_to_csv(decision, OUT_DECISION)
    print(decision.to_string(index=False))
    raise SystemExit(1)


v74_decision = read_csv_clean(V74_DECISION)
zip_index = read_csv_clean(ZIP_INDEX)
metrics = read_csv_clean(KEY_METRICS)
claims = read_csv_clean(CLAIMS)
qa74 = read_csv_clean(QA74)

v74_ready = bool_true(v74_decision.iloc[0].get("ready_for_v75_independent_final_delivery_audit", ""))
if not v74_ready:
    issues.append({
        "item": str(V74_DECISION),
        "issue_type": "hard_v74_not_ready",
        "issue_detail": "v74 is not marked ready for v75.",
        "severity": "hard",
    })

expected_v74_sha = clean(v74_decision.iloc[0].get("zip_sha256", ""))
actual_v74_sha = sha256_file(V74_ZIP)
sha_file_text = V74_SHA.read_text(errors="ignore").strip()

package_members = []
package_audit_rows = []

v74_zip_valid = zipfile.is_zipfile(V74_ZIP)

if v74_zip_valid:
    with zipfile.ZipFile(V74_ZIP, "r") as z:
        package_members = sorted(z.namelist())

expected_members = [
    "Week8_Final_Delivery_Package/README_Week8_Final_Delivery_Package.md",
    "Week8_Final_Delivery_Package/week8_v74_delivery_index.csv",
    "Week8_Final_Delivery_Package/week8_v74_zip_index.csv",
    "Week8_Final_Delivery_Package/week8_v74_key_metrics_summary.csv",
    "Week8_Final_Delivery_Package/week8_v74_final_claim_boundaries.csv",
    "Week8_Final_Delivery_Package/week8_v74_stage_status_summary.csv",
    "Week8_Final_Delivery_Package/week8_v74_quality_checks.csv",
    "Week8_Final_Delivery_Package/week8_v74_manifest.json",
]

for member in expected_members:
    package_audit_rows.append({
        "package_zip": str(V74_ZIP),
        "member": member,
        "exists_in_zip": member in package_members,
    })

package_audit = pd.DataFrame(package_audit_rows)
safe_to_csv(package_audit, OUT_PACKAGE_AUDIT)

zip_audit_rows = []

for _, r in zip_index.iterrows():
    stage = clean(r.get("stage", ""))
    path = Path(clean(r.get("zip_path", "")))
    expected_sha = clean(r.get("zip_sha256", ""))

    exists = path.exists()
    is_zip = zipfile.is_zipfile(path) if exists else False
    actual_sha = sha256_file(path) if exists else ""
    sha_match = actual_sha == expected_sha if expected_sha else False

    zip_audit_rows.append({
        "stage": stage,
        "zip_path": str(path),
        "exists": exists,
        "is_valid_zip": is_zip,
        "expected_sha256": expected_sha,
        "actual_sha256": actual_sha,
        "sha256_match": sha_match,
        "size_mb": round(path.stat().st_size / (1024 * 1024), 3) if exists else "",
    })

zip_audit = pd.DataFrame(zip_audit_rows)
safe_to_csv(zip_audit, OUT_ZIP_AUDIT)

metric = metrics.iloc[0].to_dict()

metric_checks = [
    {
        "metric": "manual_gt_reviewed_objects",
        "expected": "432",
        "actual": clean(metric.get("manual_gt_reviewed_objects", "")),
        "passed": clean(metric.get("manual_gt_reviewed_objects", "")) == "432",
    },
    {
        "metric": "strict_gold_objects",
        "expected": "372",
        "actual": clean(metric.get("strict_gold_objects", "")),
        "passed": clean(metric.get("strict_gold_objects", "")) == "372",
    },
    {
        "metric": "frozen_embedding_rows",
        "expected": "744",
        "actual": clean(metric.get("frozen_embedding_rows", "")),
        "passed": clean(metric.get("frozen_embedding_rows", "")) == "744",
    },
    {
        "metric": "final_champion",
        "expected": "frozen_yolov8_detector_embedding_baseline",
        "actual": clean(metric.get("final_champion", "")),
        "passed": clean(metric.get("final_champion", "")) == "frozen_yolov8_detector_embedding_baseline",
    },
    {
        "metric": "current_macro_f1_positive",
        "expected": ">0",
        "actual": clean(metric.get("current_macro_f1", "")),
        "passed": fnum(metric.get("current_macro_f1", "")) > 0,
    },
    {
        "metric": "group_macro_f1_positive",
        "expected": ">0",
        "actual": clean(metric.get("group_macro_f1", "")),
        "passed": fnum(metric.get("group_macro_f1", "")) > 0,
    },
]

metric_audit = pd.DataFrame(metric_checks)
safe_to_csv(metric_audit, OUT_METRIC_AUDIT)

required_claim_topics = {
    "source_of_truth",
    "dataset_scope",
    "classification",
    "model_training",
    "split_policy",
    "rare_classes",
}

claim_topics = set(claims["topic"].tolist()) if "topic" in claims.columns else set()
claim_audit_rows = []

for topic in sorted(required_claim_topics):
    claim_audit_rows.append({
        "required_topic": topic,
        "exists": topic in claim_topics,
    })

claim_audit = pd.DataFrame(claim_audit_rows)
safe_to_csv(claim_audit, OUT_CLAIM_AUDIT)

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

add_qa("v74_ready", True, v74_ready, v74_ready, "hard", "v74 must be ready for independent final audit.")
add_qa("v74_zip_exists", True, V74_ZIP.exists(), V74_ZIP.exists(), "hard", "v74 final zip must exist.")
add_qa("v74_zip_valid", True, v74_zip_valid, v74_zip_valid, "hard", "v74 final zip must be a valid zip.")
add_qa("v74_zip_sha_matches_decision", expected_v74_sha, actual_v74_sha, actual_v74_sha == expected_v74_sha, "hard", "v74 zip SHA must match v74 decision.")
add_qa("v74_sha_file_contains_actual", actual_v74_sha, sha_file_text, actual_v74_sha in sha_file_text, "hard", "v74 sha256 file should contain actual v74 zip SHA.")
add_qa("v74_required_members_present", len(expected_members), int(package_audit["exists_in_zip"].sum()), bool(package_audit["exists_in_zip"].all()), "hard", "Required files should exist inside v74 zip.")
add_qa("referenced_zip_rows", 15, len(zip_audit), len(zip_audit) == 15, "hard", "v74 zip index should contain 15 referenced zip artifacts.")
add_qa("referenced_zips_exist", 15, int(zip_audit["exists"].sum()), bool(zip_audit["exists"].all()), "hard", "All referenced zips should exist.")
add_qa("referenced_zips_valid", 15, int(zip_audit["is_valid_zip"].sum()), bool(zip_audit["is_valid_zip"].all()), "hard", "All referenced zips should be valid zip files.")
add_qa("referenced_zip_sha_match", 15, int(zip_audit["sha256_match"].sum()), bool(zip_audit["sha256_match"].all()), "hard", "All referenced zip SHA values should match index.")
add_qa("v74_quality_checks_hard_pass", True, bool(qa74[qa74["severity"] == "hard"]["passed"].astype(str).str.lower().eq("true").all()), bool(qa74[qa74["severity"] == "hard"]["passed"].astype(str).str.lower().eq("true").all()), "hard", "All v74 hard QA checks should pass.")
add_qa("metric_checks_pass", len(metric_audit), int(metric_audit["passed"].sum()), bool(metric_audit["passed"].all()), "hard", "Key final metrics should match expected values.")
add_qa("claim_topics_present", len(required_claim_topics), int(claim_audit["exists"].sum()), bool(claim_audit["exists"].all()), "hard", "Required claim boundary topics should exist.")

qa = pd.DataFrame(qa_rows)
safe_to_csv(qa, OUT_QA)

hard_quality_failures = int(((qa["severity"] == "hard") & (~qa["passed"])).sum())

if hard_quality_failures:
    issues.append({
        "item": "v75_quality_checks",
        "issue_type": "hard_independent_final_audit_failed",
        "issue_detail": f"{hard_quality_failures} hard checks failed.",
        "severity": "hard",
    })

issues.append({
    "item": "audit_scope",
    "issue_type": "info_independent_audit_only",
    "issue_detail": "v75 independently audits final delivery integrity. It does not modify GT, train models, or duplicate raw videos.",
    "severity": "info",
})

issues_df = pd.DataFrame(issues, columns=["item", "issue_type", "issue_detail", "severity"])
safe_to_csv(issues_df, OUT_ISSUES)

hard_issue_count = int((issues_df["severity"] == "hard").sum()) if len(issues_df) else 0
warning_count = int((issues_df["severity"] == "warning").sum()) if len(issues_df) else 0
info_count = int((issues_df["severity"] == "info").sum()) if len(issues_df) else 0

manifest = {
    "version": "week8_v75_independent_final_delivery_audit",
    "created_at": datetime.now().isoformat(timespec="seconds"),
    "v74_zip": str(V74_ZIP),
    "v74_zip_sha256": actual_v74_sha,
    "referenced_zip_rows": int(len(zip_audit)),
    "referenced_zip_sha_matches": int(zip_audit["sha256_match"].sum()),
    "hard_quality_failures": hard_quality_failures,
    "final_delivery_ready": bool(hard_issue_count == 0),
    "claim_boundary": "independent final delivery audit only",
}

OUT_MANIFEST.write_text(json.dumps(manifest, indent=2, ensure_ascii=False))

readme_text = f"""# Week8 Independent Final Delivery Audit

## Result

Final delivery ready: {bool(hard_issue_count == 0)}

## Audited items

- v74 final package zip exists and is valid
- v74 final package SHA matches decision summary
- required files exist inside v74 package zip
- 15 referenced zip artifacts exist
- all referenced artifact SHA256 values match
- key metrics are consistent
- claim boundary topics are present

## Final champion

Frozen YOLOv8 detector embedding baseline.

Current split macro-F1: {clean(metric.get("current_macro_f1", ""))}
Group-aware split macro-F1: {clean(metric.get("group_macro_f1", ""))}

## Scope

This audit does not train a model, modify GT, or duplicate raw videos.
"""

OUT_README.write_text(readme_text)
OUT_REPORT.write_text(readme_text)

if OUT_ZIP.exists():
    OUT_ZIP.unlink()

with zipfile.ZipFile(OUT_ZIP, "w", compression=zipfile.ZIP_DEFLATED) as z:
    for p in sorted(PKG.rglob("*")):
        if p.is_file():
            z.write(p, p.relative_to(OUT))

audit_zip_sha = sha256_file(OUT_ZIP)
OUT_SHA.write_text(f"{audit_zip_sha}  {OUT_ZIP.name}\n")

decision = pd.DataFrame([{
    "v75_decision": "independent_final_delivery_audit_passed" if hard_issue_count == 0 else "independent_final_delivery_audit_failed",
    "v74_zip_path": str(V74_ZIP),
    "v74_zip_sha256": actual_v74_sha,
    "v74_zip_valid": bool(v74_zip_valid),
    "v74_sha_matches_decision": bool(actual_v74_sha == expected_v74_sha),
    "referenced_zip_rows": int(len(zip_audit)),
    "referenced_zips_exist": int(zip_audit["exists"].sum()),
    "referenced_zips_valid": int(zip_audit["is_valid_zip"].sum()),
    "referenced_zip_sha_matches": int(zip_audit["sha256_match"].sum()),
    "final_champion": clean(metric.get("final_champion", "")),
    "strict_gold_objects": clean(metric.get("strict_gold_objects", "")),
    "frozen_embedding_rows": clean(metric.get("frozen_embedding_rows", "")),
    "current_macro_f1": clean(metric.get("current_macro_f1", "")),
    "group_macro_f1": clean(metric.get("group_macro_f1", "")),
    "audit_zip_path": str(OUT_ZIP),
    "audit_zip_sha256": audit_zip_sha,
    "hard_quality_failures": hard_quality_failures,
    "hard_issue_count": hard_issue_count,
    "warning_count": warning_count,
    "info_count": info_count,
    "issue_count": int(len(issues_df)),
    "final_delivery_ready": bool(hard_issue_count == 0),
    "claim_scope": "independent_final_delivery_audit_only",
    "generated_at": datetime.now().isoformat(timespec="seconds"),
}])

safe_to_csv(decision, OUT_DECISION)

OUT_NOTE.write_text(
    "# Week 8 v75 Independent Final Delivery Audit\n\n"
    f"- v75 decision: {decision.iloc[0]['v75_decision']}\n"
    f"- v74 zip valid: {bool(v74_zip_valid)}\n"
    f"- v74 SHA matches decision: {bool(actual_v74_sha == expected_v74_sha)}\n"
    f"- Referenced zip rows: {len(zip_audit)}\n"
    f"- Referenced zips exist: {int(zip_audit['exists'].sum())}\n"
    f"- Referenced zips valid: {int(zip_audit['is_valid_zip'].sum())}\n"
    f"- Referenced SHA matches: {int(zip_audit['sha256_match'].sum())}\n"
    f"- Final champion: {clean(metric.get('final_champion', ''))}\n"
    f"- Strict-gold objects: {clean(metric.get('strict_gold_objects', ''))}\n"
    f"- Frozen embedding rows: {clean(metric.get('frozen_embedding_rows', ''))}\n"
    f"- Hard issues: {hard_issue_count}\n"
    f"- Final delivery ready: {bool(hard_issue_count == 0)}\n\n"
    "This is the independent final delivery audit. No model is trained and no GT is modified.\n"
)

progress_row = pd.DataFrame([{
    "date": datetime.now().date().isoformat(),
    "stage": "v75",
    "task_name": "Independent final delivery audit",
    "status": "PASS" if hard_issue_count == 0 else "NEEDS_FIX",
    "input_summary": str(V74),
    "output_summary": str(PKG),
    "hard_issues": hard_issue_count,
    "warnings": warning_count,
    "next_action": "Week8 final delivery is ready." if hard_issue_count == 0 else "Fix v75 audit failures.",
}])

if OUT_PROGRESS.exists():
    old = pd.read_csv(OUT_PROGRESS)
    progress = pd.concat([old, progress_row], ignore_index=True)
else:
    progress = progress_row

safe_to_csv(progress, OUT_PROGRESS)

print("=== v75 decision ===")
print(decision.to_string(index=False))
print("\n=== QA ===")
print(qa.to_string(index=False))
print("\n=== issues ===")
print(issues_df.to_string(index=False) if len(issues_df) else "No issues found.")
