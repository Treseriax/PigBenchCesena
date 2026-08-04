from pathlib import Path
from datetime import datetime
import csv
import json
import hashlib
import zipfile
import tempfile
import shutil
import pandas as pd


ROOT = Path.home() / "PigBench"
W8 = ROOT / "Week8_Ground_Truth_Validation_Behaviour_Dataset_Preparation"

V67 = W8 / "outputs" / "v67_week8_report_ready_package"
ZIP_PATH = V67 / "Week8_Report_Ready_GT_v2_Package.zip"
SHA_PATH = V67 / "Week8_Report_Ready_GT_v2_Package.sha256"

OUT = W8 / "outputs" / "v67b_independent_package_audit"
NOTES = W8 / "notes"
REPORTS = W8 / "reports"
PROGRESS = W8 / "progress"

for p in [OUT, NOTES, REPORTS, PROGRESS]:
    p.mkdir(parents=True, exist_ok=True)

OUT_QA = OUT / "week8_v67b_independent_package_quality_checks.csv"
OUT_FILE_AUDIT = OUT / "week8_v67b_package_file_hash_audit.csv"
OUT_COUNTS = OUT / "week8_v67b_package_count_summary.csv"
OUT_DECISION = OUT / "week8_v67b_decision_summary.csv"
OUT_ISSUES = OUT / "week8_v67b_issues.csv"
OUT_NOTE = NOTES / "week8_v67b_independent_package_audit_notes.md"
OUT_REPORT = REPORTS / "week8_v67b_independent_package_audit_report.md"
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


def parse_sha_file(path):
    txt = path.read_text().strip()
    if not txt:
        return ""
    return txt.split()[0].strip()


issues = []
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


# Basic external package checks.
zip_exists = ZIP_PATH.exists()
sha_exists = SHA_PATH.exists()

add_qa("zip_exists", True, zip_exists, zip_exists, "hard", "v67 package zip must exist.")
add_qa("sha_file_exists", True, sha_exists, sha_exists, "hard", "v67 package sha256 file must exist.")

if not zip_exists or not sha_exists:
    issues.append({
        "item": "v67_package_files",
        "issue_type": "hard_missing_zip_or_sha",
        "issue_detail": f"ZIP exists={zip_exists}, SHA exists={sha_exists}",
        "severity": "hard",
    })

    qa_df = pd.DataFrame(qa_rows)
    safe_to_csv(qa_df, OUT_QA)
    issues_df = pd.DataFrame(issues)
    safe_to_csv(issues_df, OUT_ISSUES)

    decision = pd.DataFrame([{
        "v67b_decision": "independent_package_audit_blocked",
        "hard_issue_count": int((issues_df["severity"] == "hard").sum()),
        "ready_for_v67c_dataset_card": False,
        "generated_at": datetime.now().isoformat(timespec="seconds"),
    }])
    safe_to_csv(decision, OUT_DECISION)
    print(decision.to_string(index=False))
    raise SystemExit(1)


actual_zip_sha = sha256_file(ZIP_PATH)
expected_zip_sha = parse_sha_file(SHA_PATH)
sha_match = actual_zip_sha == expected_zip_sha

add_qa("zip_sha256_matches", expected_zip_sha, actual_zip_sha, sha_match, "hard", "ZIP SHA256 must match .sha256 file.")

if not sha_match:
    issues.append({
        "item": str(ZIP_PATH),
        "issue_type": "hard_zip_sha256_mismatch",
        "issue_detail": "ZIP SHA256 does not match .sha256 file.",
        "severity": "hard",
    })


zip_valid = False
zip_roots = []

try:
    with zipfile.ZipFile(ZIP_PATH, "r") as z:
        bad = z.testzip()
        names = z.namelist()
        zip_valid = bad is None and len(names) > 0
        zip_roots = sorted(set(n.split("/")[0] for n in names if "/" in n))
        add_qa("zip_testzip_valid", True, zip_valid, zip_valid, "hard", "ZIP must be readable and testzip must pass.")
        add_qa("zip_single_package_root", 1, len(zip_roots), len(zip_roots) == 1, "hard", "ZIP should contain one package root directory.")
except Exception as e:
    add_qa("zip_testzip_valid", True, False, False, "hard", f"ZIP read failed: {e}")
    issues.append({
        "item": str(ZIP_PATH),
        "issue_type": "hard_zip_read_error",
        "issue_detail": str(e),
        "severity": "hard",
    })


if not zip_valid:
    qa_df = pd.DataFrame(qa_rows)
    safe_to_csv(qa_df, OUT_QA)
    issues_df = pd.DataFrame(issues)
    safe_to_csv(issues_df, OUT_ISSUES)

    decision = pd.DataFrame([{
        "v67b_decision": "independent_package_audit_blocked_zip_invalid",
        "zip_sha256": actual_zip_sha,
        "hard_issue_count": int((issues_df["severity"] == "hard").sum()),
        "ready_for_v67c_dataset_card": False,
        "generated_at": datetime.now().isoformat(timespec="seconds"),
    }])
    safe_to_csv(decision, OUT_DECISION)
    print(decision.to_string(index=False))
    raise SystemExit(1)


tmpdir = Path(tempfile.mkdtemp(prefix="week8_v67b_audit_"))
try:
    with zipfile.ZipFile(ZIP_PATH, "r") as z:
        z.extractall(tmpdir)

    root_dirs = [p for p in tmpdir.iterdir() if p.is_dir()]
    if len(root_dirs) != 1:
        package_root = tmpdir / "Week8_Report_Ready_GT_v2_Package"
    else:
        package_root = root_dirs[0]

    add_qa("package_root_exists_after_extract", True, package_root.exists(), package_root.exists(), "hard", "Extracted package root must exist.")

    required_files = [
        "README_Week8_Report_Ready_GT_v2_Package.md",
        "Week8_Final_GT_v2_Report.md",
        "week8_v67_manifest.json",
        "week8_v67_file_hashes.csv",

        "data/week8_final_gt_v2_all_reviewed_objects.csv",
        "data/week8_final_gt_v2_strict_gold_objects_for_classification.csv",
        "data/week8_final_gt_v2_caution_objects_for_analysis.csv",
        "data/week8_final_gt_v2_nonusable_fix_or_excluded_objects.csv",
        "data/week8_final_gt_v2_scanframe_quality_summary.csv",
        "data/week8_final_gt_v2_manifest.json",
        "data/week8_v65a_strict_gold_behaviour_distribution.csv",
        "data/week8_v65a_classification_limitations.csv",
        "data/week8_v66a_final_gt_v2_object_table.csv",
        "data/week8_v66a_scanframe_propagation_summary.csv",
        "data/week8_v66a_behaviour_category_distribution.csv",

        "audit/week8_v64b_decision_summary.csv",
        "audit/week8_v64b_issues.csv",
        "audit/week8_v64c_decision_summary.csv",
        "audit/week8_v64c_issues.csv",
        "audit/week8_v65a_decision_summary.csv",
        "audit/week8_v65b_decision_summary.csv",
        "audit/week8_v65b_issues.csv",
        "audit/week8_v66a_decision_summary.csv",
        "audit/week8_v66a_issues.csv",
        "audit/week8_v66b_decision_summary.csv",
        "audit/week8_v66b_issues.csv",
        "audit/week8_v66c_decision_summary.csv",
        "audit/week8_v66c_issues.csv",
        "audit/week8_v67_report_ready_package_quality_checks.csv",

        "visualizer/week8_v66b_final_gt_v2_visualizer_data.json",
        "visualizer/week8_visualizer_server_v66b.py",
        "visualizer/static_v66b/index.html",
        "visualizer/static_v66b/app.js",
        "visualizer/static_v66b/style.css",

        "tracking_helper/week8_v66c_scanframe_tracking_helper_summary.csv",
        "tracking_helper/week8_v66c_final_gt_v2_object_tracking_helper_summary.csv",
        "tracking_helper/week8_v66c_tracking_helper_file_inventory.csv",
    ]

    missing_required = []
    for rel in required_files:
        exists = (package_root / rel).exists()
        if not exists:
            missing_required.append(rel)

    add_qa("required_files_present", 0, len(missing_required), len(missing_required) == 0, "hard", "All required package files should be present.")

    for rel in missing_required:
        issues.append({
            "item": rel,
            "issue_type": "hard_missing_required_package_file",
            "issue_detail": "Required file is missing inside extracted ZIP package.",
            "severity": "hard",
        })

    # File hash manifest audit.
    hash_path = package_root / "week8_v67_file_hashes.csv"
    file_audit_rows = []

    if hash_path.exists():
        hash_df = read_csv_clean(hash_path)

        for _, r in hash_df.iterrows():
            rel = r["relative_path"]
            expected = r["sha256"]
            target = package_root / rel

            if not target.exists():
                file_audit_rows.append({
                    "relative_path": rel,
                    "expected_sha256": expected,
                    "actual_sha256": "",
                    "exists": False,
                    "sha256_match": False,
                    "audit_status": "missing_file",
                })
                continue

            actual = sha256_file(target)
            file_audit_rows.append({
                "relative_path": rel,
                "expected_sha256": expected,
                "actual_sha256": actual,
                "exists": True,
                "sha256_match": actual == expected,
                "audit_status": "ok" if actual == expected else "sha256_mismatch",
            })

        file_audit = pd.DataFrame(file_audit_rows)
        safe_to_csv(file_audit, OUT_FILE_AUDIT)

        file_hash_failures = int((file_audit["sha256_match"] != True).sum()) if len(file_audit) else 0
        add_qa("internal_file_hash_manifest_valid", 0, file_hash_failures, file_hash_failures == 0, "hard", "All files listed in internal file hash manifest must match SHA256.")
    else:
        file_audit = pd.DataFrame()
        safe_to_csv(file_audit, OUT_FILE_AUDIT)
        add_qa("internal_file_hash_manifest_valid", True, False, False, "hard", "Internal file hash manifest is missing.")

    # Load core data files.
    all_gt = read_csv_clean(package_root / "data/week8_final_gt_v2_all_reviewed_objects.csv")
    strict = read_csv_clean(package_root / "data/week8_final_gt_v2_strict_gold_objects_for_classification.csv")
    caution = read_csv_clean(package_root / "data/week8_final_gt_v2_caution_objects_for_analysis.csv")
    nonusable = read_csv_clean(package_root / "data/week8_final_gt_v2_nonusable_fix_or_excluded_objects.csv")
    scanframe = read_csv_clean(package_root / "data/week8_final_gt_v2_scanframe_quality_summary.csv")
    v66a_objects = read_csv_clean(package_root / "data/week8_v66a_final_gt_v2_object_table.csv")
    v66a_scan = read_csv_clean(package_root / "data/week8_v66a_scanframe_propagation_summary.csv")
    v66c_scan = read_csv_clean(package_root / "tracking_helper/week8_v66c_scanframe_tracking_helper_summary.csv")

    total_count = len(all_gt)
    strict_count = len(strict)
    caution_count = len(caution)
    nonusable_count = len(nonusable)
    scan_count = all_gt["scan_frame_id"].nunique()

    add_qa("all_gt_rows", 432, total_count, total_count == 432, "hard", "All reviewed GT should contain 432 rows.")
    add_qa("strict_rows", 372, strict_count, strict_count == 372, "hard", "Strict gold rows should be 372 after bbox adjustment.")
    add_qa("caution_rows", 3, caution_count, caution_count == 3, "hard", "Caution rows should be 3.")
    add_qa("nonusable_rows", 57, nonusable_count, nonusable_count == 57, "hard", "Nonusable/fix/excluded rows should be 57.")
    add_qa("subset_partition", total_count, strict_count + caution_count + nonusable_count, total_count == strict_count + caution_count + nonusable_count, "hard", "Strict/caution/nonusable should partition all rows.")
    add_qa("scanframe_count", 72, scan_count, scan_count == 72, "hard", "There should be 72 scanframes.")

    if "canonical_gt_object_id" in all_gt.columns:
        all_ids = set(all_gt["canonical_gt_object_id"])
        strict_ids = set(strict["canonical_gt_object_id"])
        caution_ids = set(caution["canonical_gt_object_id"])
        nonusable_ids = set(nonusable["canonical_gt_object_id"])

        overlap_count = len(strict_ids & caution_ids) + len(strict_ids & nonusable_ids) + len(caution_ids & nonusable_ids)
        union_count = len(strict_ids | caution_ids | nonusable_ids)

        add_qa("subset_id_overlap_zero", 0, overlap_count, overlap_count == 0, "hard", "Strict/caution/nonusable object IDs must not overlap.")
        add_qa("subset_id_union_equals_all", len(all_ids), union_count, union_count == len(all_ids), "hard", "Union of subset IDs should equal all GT IDs.")
        add_qa("all_gt_ids_unique", len(all_gt), len(all_ids), len(all_gt) == len(all_ids), "hard", "canonical_gt_object_id should be unique in all GT.")
    else:
        add_qa("canonical_gt_object_id_column_present", True, False, False, "hard", "canonical_gt_object_id column missing.")

    if "canonical_colour_label_norm" in all_gt.columns:
        per_scan_colour_count = all_gt.groupby("scan_frame_id")["canonical_colour_label_norm"].nunique()
        bad_scanframes = int((per_scan_colour_count != 6).sum())
        add_qa("six_canonical_colours_per_scanframe", 0, bad_scanframes, bad_scanframes == 0, "hard", "Every scanframe should have 6 canonical colour identities.")
    else:
        add_qa("canonical_colour_column_present", True, False, False, "hard", "canonical_colour_label_norm column missing.")

    strict_rule = (
        (strict["manual_gt_v2_status"] == "gold_usable")
        & (strict["manual_classification_use"] == "use_for_classification")
        & (strict["manual_bbox_status"] == "bbox_ok")
        & (strict["manual_identity_status"] == "identity_confirmed")
        & (strict["manual_assigned_candidate_box_id"].astype(str).str.strip() != "")
    )

    strict_rule_pass = int(strict_rule.sum())
    add_qa("strict_rows_follow_strict_rule", strict_count, strict_rule_pass, strict_rule_pass == strict_count, "hard", "Strict rows must follow strict classification GT rule.")

    caution_train_leak = int((caution["manual_classification_use"] == "use_for_classification").sum()) if "manual_classification_use" in caution.columns else 0
    nonusable_train_leak = int((nonusable["manual_classification_use"] == "use_for_classification").sum()) if "manual_classification_use" in nonusable.columns else 0

    add_qa("caution_not_strict_train", 0, caution_train_leak, caution_train_leak == 0, "hard", "Caution rows must not be strict train/eval rows.")
    add_qa("nonusable_not_strict_train", 0, nonusable_train_leak, nonusable_train_leak == 0, "hard", "Nonusable rows must not be strict train/eval rows.")

    if "frame_object_rows" in v66a_scan.columns:
        frame_rows_sum = pd.to_numeric(v66a_scan["frame_object_rows"], errors="coerce").fillna(0).sum()
        add_qa("v66a_frame_object_rows_sum", 107862, int(frame_rows_sum), int(frame_rows_sum) == 107862, "hard", "Frame-object propagation rows should sum to 107862.")
    else:
        add_qa("v66a_frame_object_rows_column", True, False, False, "hard", "v66a frame_object_rows column missing.")

    if "strict_gold_frame_object_rows" in v66a_scan.columns:
        strict_frame_sum = pd.to_numeric(v66a_scan["strict_gold_frame_object_rows"], errors="coerce").fillna(0).sum()
        add_qa("v66a_strict_frame_object_rows_sum", 92879, int(strict_frame_sum), int(strict_frame_sum) == 92879, "hard", "Strict frame-object propagation rows should sum to 92879.")
    else:
        add_qa("v66a_strict_frame_object_rows_column", True, False, False, "hard", "v66a strict_gold_frame_object_rows column missing.")

    if "tracking_helper_total_rows" in v66c_scan.columns:
        helper_scanframes = int((pd.to_numeric(v66c_scan["tracking_helper_total_rows"], errors="coerce").fillna(0) > 0).sum())
        add_qa("tracking_helper_attached_72_scanframes", 72, helper_scanframes, helper_scanframes == 72, "hard", "Tracking helper should be attached to all 72 scanframes.")
    else:
        add_qa("tracking_helper_total_rows_column", True, False, False, "hard", "tracking_helper_total_rows column missing.")

    # Manifest consistency.
    manifest_path = package_root / "week8_v67_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    nums = manifest.get("key_numbers", {})

    manifest_checks = [
        ("manifest_reviewed_objects", total_count, nums.get("reviewed_objects")),
        ("manifest_scanframes", scan_count, nums.get("scanframes")),
        ("manifest_strict_gold_objects", strict_count, nums.get("strict_gold_objects")),
        ("manifest_caution_objects", caution_count, nums.get("caution_objects")),
        ("manifest_nonusable_objects", nonusable_count, nums.get("nonusable_fix_or_excluded_objects")),
        ("manifest_tracking_helper_scanframes", 72, nums.get("tracking_helper_scanframes")),
    ]

    for name, expected, actual in manifest_checks:
        add_qa(name, expected, actual, int(actual) == int(expected), "hard", "Manifest key number should match data.")

    # Claim boundary text audit.
    readme_text = (package_root / "README_Week8_Report_Ready_GT_v2_Package.md").read_text(encoding="utf-8")
    report_text = (package_root / "Week8_Final_GT_v2_Report.md").read_text(encoding="utf-8")
    combined_text = (readme_text + "\n" + report_text).lower()

    claim_phrases = [
        "manual gt v2 is the source of truth",
        "tracking is diagnostic helper only",
        "baseline/proof-of-concept",
        "not support a production-grade behaviour classifier",
        "72 annotated scanframe clips",
        "not the full raw unibo archive",
    ]

    for phrase in claim_phrases:
        add_qa(
            "claim_boundary_phrase_present",
            phrase,
            phrase in combined_text,
            phrase in combined_text,
            "hard",
            f"Claim boundary phrase should be present: {phrase}",
        )

    # Hard issues in copied issue CSV files.
    hard_issue_rows = []
    for issue_file in (package_root / "audit").glob("*issues.csv"):
        try:
            df_issue = read_csv_clean(issue_file)
            if "severity" in df_issue.columns and len(df_issue):
                h = df_issue[df_issue["severity"] == "hard"].copy()
                for _, r in h.iterrows():
                    hard_issue_rows.append({
                        "issue_file": str(issue_file.relative_to(package_root)),
                        "item": r.get("item", ""),
                        "issue_type": r.get("issue_type", ""),
                        "issue_detail": r.get("issue_detail", ""),
                        "severity": r.get("severity", ""),
                    })
        except Exception as e:
            hard_issue_rows.append({
                "issue_file": str(issue_file.relative_to(package_root)),
                "item": "",
                "issue_type": "issue_file_read_error",
                "issue_detail": str(e),
                "severity": "hard",
            })

    add_qa("copied_issue_files_have_no_hard_issues", 0, len(hard_issue_rows), len(hard_issue_rows) == 0, "hard", "Copied package issue files should not contain hard issues.")

    for r in hard_issue_rows:
        issues.append({
            "item": r["issue_file"],
            "issue_type": "hard_issue_inside_packaged_issue_file",
            "issue_detail": r["issue_detail"],
            "severity": "hard",
        })

    count_summary = pd.DataFrame([{
        "total_reviewed_objects": total_count,
        "strict_gold_objects": strict_count,
        "caution_objects": caution_count,
        "nonusable_fix_or_excluded_objects": nonusable_count,
        "scanframes": scan_count,
        "frame_object_rows": int(pd.to_numeric(v66a_scan["frame_object_rows"], errors="coerce").fillna(0).sum()) if "frame_object_rows" in v66a_scan.columns else "",
        "strict_gold_frame_object_rows": int(pd.to_numeric(v66a_scan["strict_gold_frame_object_rows"], errors="coerce").fillna(0).sum()) if "strict_gold_frame_object_rows" in v66a_scan.columns else "",
        "tracking_helper_scanframes": int((pd.to_numeric(v66c_scan["tracking_helper_total_rows"], errors="coerce").fillna(0) > 0).sum()) if "tracking_helper_total_rows" in v66c_scan.columns else "",
        "zip_sha256": actual_zip_sha,
    }])
    safe_to_csv(count_summary, OUT_COUNTS)

finally:
    shutil.rmtree(tmpdir, ignore_errors=True)


qa_df = pd.DataFrame(qa_rows)
safe_to_csv(qa_df, OUT_QA)

hard_quality_failures = int(((qa_df["severity"] == "hard") & (~qa_df["passed"])).sum())
warning_quality_failures = int(((qa_df["severity"] == "warning") & (~qa_df["passed"])).sum())

if hard_quality_failures:
    issues.append({
        "item": "v67b_quality_checks",
        "issue_type": "hard_independent_package_audit_failed",
        "issue_detail": f"{hard_quality_failures} hard quality checks failed.",
        "severity": "hard",
    })

issues_df = pd.DataFrame(issues, columns=["item", "issue_type", "issue_detail", "severity"])
safe_to_csv(issues_df, OUT_ISSUES)

hard_issue_count = int((issues_df["severity"] == "hard").sum()) if len(issues_df) else 0
warning_count = int((issues_df["severity"] == "warning").sum()) if len(issues_df) else 0
info_count = int((issues_df["severity"] == "info").sum()) if len(issues_df) else 0

decision = pd.DataFrame([{
    "v67b_decision": "independent_package_audit_passed" if hard_issue_count == 0 else "independent_package_audit_failed",
    "zip_path": str(ZIP_PATH),
    "zip_sha256": actual_zip_sha,
    "expected_sha256": expected_zip_sha,
    "sha256_match": bool(actual_zip_sha == expected_zip_sha),
    "total_quality_checks": int(len(qa_df)),
    "hard_quality_failures": hard_quality_failures,
    "warning_quality_failures": warning_quality_failures,
    "hard_issue_count": hard_issue_count,
    "warning_count": warning_count,
    "info_count": info_count,
    "issue_count": int(len(issues_df)),
    "ready_for_v67c_dataset_card": bool(hard_issue_count == 0),
    "ready_for_v68_dataset_materialization": bool(hard_issue_count == 0),
    "generated_at": datetime.now().isoformat(timespec="seconds"),
}])

safe_to_csv(decision, OUT_DECISION)

OUT_NOTE.write_text(
    "# Week 8 v67b Independent Package Audit\n\n"
    f"- v67b decision: {decision.iloc[0]['v67b_decision']}\n"
    f"- ZIP: {ZIP_PATH}\n"
    f"- SHA256: {actual_zip_sha}\n"
    f"- SHA256 match: {bool(actual_zip_sha == expected_zip_sha)}\n"
    f"- Total quality checks: {len(qa_df)}\n"
    f"- Hard quality failures: {hard_quality_failures}\n"
    f"- Hard issues: {hard_issue_count}\n"
    f"- Ready for v67c dataset card: {bool(hard_issue_count == 0)}\n"
    f"- Ready for v68 dataset materialization: {bool(hard_issue_count == 0)}\n\n"
    "This audit extracts the v67 ZIP independently and verifies package contents, counts, hashes, source-of-truth rules, and claim boundaries.\n"
)

OUT_REPORT.write_text(
    "# Week 8 v67b Independent Package Audit Report\n\n"
    f"Decision: {decision.iloc[0]['v67b_decision']}\n\n"
    f"ZIP: {ZIP_PATH}\n\n"
    f"SHA256: {actual_zip_sha}\n\n"
    f"Quality checks: {OUT_QA}\n\n"
    f"Issues: {OUT_ISSUES}\n"
)

progress_row = pd.DataFrame([{
    "date": datetime.now().date().isoformat(),
    "stage": "v67b",
    "task_name": "Independent final package audit",
    "status": "PASS" if hard_issue_count == 0 else "NEEDS_FIX",
    "input_summary": str(ZIP_PATH),
    "output_summary": str(OUT),
    "hard_issues": hard_issue_count,
    "warnings": warning_count,
    "next_action": "Create v67c dataset card and polished GT documentation." if hard_issue_count == 0 else "Fix v67 package audit issues before proceeding.",
}])

if OUT_PROGRESS.exists():
    old = pd.read_csv(OUT_PROGRESS)
    progress = pd.concat([old, progress_row], ignore_index=True)
else:
    progress = progress_row

safe_to_csv(progress, OUT_PROGRESS)

print("Saved:")
print(OUT_QA)
print(OUT_FILE_AUDIT)
print(OUT_COUNTS)
print(OUT_DECISION)
print(OUT_ISSUES)
print(OUT_NOTE)

print()
print("=== v67b decision ===")
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
