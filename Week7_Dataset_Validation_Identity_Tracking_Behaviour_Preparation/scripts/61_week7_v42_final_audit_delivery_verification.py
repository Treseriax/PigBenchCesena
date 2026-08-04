from pathlib import Path
from datetime import datetime
import csv
import hashlib
import zipfile
import io
import shutil
import pandas as pd


ROOT = Path.home() / "PigBench"
W7 = ROOT / "Week7_Dataset_Validation_Identity_Tracking_Behaviour_Preparation"

V41 = W7 / "outputs" / "week7_final_report_ready_summary_visual_package_v41"

FIXED_ZIP = V41 / "Week7_Final_Report_Ready_Summary_Visual_Package_v41b_FIXED.zip"
V41B_DECISION = V41 / "week7_v41b_key_number_fix_decision.csv"
V41B_ISSUES = V41 / "week7_v41b_key_number_fix_issues.csv"

OUT_ROOT = W7 / "outputs" / "week7_final_audit_delivery_verification_v42"
DELIVERY_DIR = OUT_ROOT / "Week7_Final_Delivery_Verification_v42"

OUT_ROOT.mkdir(parents=True, exist_ok=True)
DELIVERY_DIR.mkdir(parents=True, exist_ok=True)

OUT_MEMBER_MANIFEST = OUT_ROOT / "week7_v42_zip_member_manifest.csv"
OUT_REQUIRED_AUDIT = OUT_ROOT / "week7_v42_required_artifact_audit.csv"
OUT_CONTENT_AUDIT = OUT_ROOT / "week7_v42_content_claim_scope_audit.csv"
OUT_DECISION = OUT_ROOT / "week7_v42_final_audit_delivery_decision.csv"
OUT_ISSUES = OUT_ROOT / "week7_v42_final_audit_delivery_issues.csv"
OUT_README = OUT_ROOT / "Week7_Final_Delivery_Readme_v42.md"
OUT_DELIVERY_ZIP = OUT_ROOT / "Week7_Final_Delivery_Verification_v42.zip"
OUT_NOTE = W7 / "notes" / "week7_v42_final_audit_delivery_verification_notes.md"


def safe_to_csv(df, path):
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(
        path,
        index=False,
        quoting=csv.QUOTE_ALL,
        escapechar="\\",
        lineterminator="\n",
    )


def sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def clean(v):
    if pd.isna(v):
        return ""
    return str(v).strip()


def read_df(path):
    path = Path(path)
    if not path.exists():
        return pd.DataFrame()
    try:
        return pd.read_csv(path)
    except Exception:
        return pd.DataFrame()


def read_csv_from_zip(zf, member):
    with zf.open(member) as f:
        return pd.read_csv(f)


def read_text_from_zip(zf, member):
    with zf.open(member) as f:
        return f.read().decode("utf-8", errors="replace")


issues = []
required_rows = []
content_rows = []
member_rows = []

if not FIXED_ZIP.exists():
    issues.append({
        "item": str(FIXED_ZIP),
        "issue_type": "hard_missing_fixed_zip",
        "issue_detail": "v41b fixed zip is missing.",
    })

if not V41B_DECISION.exists():
    issues.append({
        "item": str(V41B_DECISION),
        "issue_type": "hard_missing_v41b_decision",
        "issue_detail": "v41b decision summary is missing.",
    })

v41b_decision = read_df(V41B_DECISION)
v41b_issues = read_df(V41B_ISSUES)

expected_sha = ""
expected_corrected_value = ""

if len(v41b_decision):
    expected_sha = clean(v41b_decision.iloc[0].get("fixed_zip_sha256", ""))
    expected_corrected_value = clean(v41b_decision.iloc[0].get("corrected_value", ""))
else:
    issues.append({
        "item": str(V41B_DECISION),
        "issue_type": "hard_empty_v41b_decision",
        "issue_detail": "v41b decision summary could not be read.",
    })

zip_valid = False
zip_test_pass = False
zip_test_bad_file = ""
computed_sha = ""
zip_size_bytes = 0
zip_member_count = 0

if FIXED_ZIP.exists():
    computed_sha = sha256_file(FIXED_ZIP)
    zip_size_bytes = FIXED_ZIP.stat().st_size

    try:
        with zipfile.ZipFile(FIXED_ZIP, "r") as zf:
            zip_valid = True
            bad = zf.testzip()

            if bad is None:
                zip_test_pass = True
            else:
                zip_test_bad_file = bad

            infos = zf.infolist()
            zip_member_count = len(infos)

            for info in infos:
                member_rows.append({
                    "member_name": info.filename,
                    "file_size": info.file_size,
                    "compress_size": info.compress_size,
                    "is_dir": info.is_dir(),
                })

    except Exception as e:
        issues.append({
            "item": str(FIXED_ZIP),
            "issue_type": "hard_zip_open_failed",
            "issue_detail": str(e),
        })

if expected_sha and computed_sha and expected_sha != computed_sha:
    issues.append({
        "item": str(FIXED_ZIP),
        "issue_type": "hard_fixed_zip_sha_mismatch",
        "issue_detail": f"expected={expected_sha}, computed={computed_sha}",
    })

if not zip_test_pass:
    issues.append({
        "item": str(FIXED_ZIP),
        "issue_type": "hard_zip_test_failed",
        "issue_detail": zip_test_bad_file or "zip test did not pass",
    })

member_manifest = pd.DataFrame(member_rows)
safe_to_csv(member_manifest, OUT_MEMBER_MANIFEST)

# ---------------------------------------------------------------------
# Required members and content checks
# ---------------------------------------------------------------------

root_name = "Week7_Final_Report_Ready_Summary_Visual_Package_v41"

required_members = [
    f"{root_name}/01_final_reports/Week7_Final_Report_Ready_Summary_v41.md",
    f"{root_name}/01_final_reports/Week7_Final_Report_Ready_Summary_v41_TR.md",
    f"{root_name}/02_tables/week7_v41_key_numbers.csv",
    f"{root_name}/02_tables/week7_v41_claim_scope_checklist.csv",
    f"{root_name}/02_tables/week7_v41_final_limitations.csv",
    f"{root_name}/02_tables/week7_v41_final_recommendations.csv",
    f"{root_name}/03_v40_evidence/Week7_Behaviour_Temporal_Evidence_Report_v40.md",
    f"{root_name}/03_v40_evidence/Week7_Behaviour_Temporal_Evidence_Package_v40.zip",
    f"{root_name}/04_metrics/week7_v39_clip_multilabel_metrics.csv",
    f"{root_name}/04_metrics/week7_v37_crop_baseline_metrics.csv",
    f"{root_name}/05_figures/week7_v41_crop_vs_clip_metric_comparison.png",
    f"{root_name}/05_figures/week7_v41_clip_label_counts_by_split.png",
    f"{root_name}/05_figures/week7_v41_pipeline_summary.png",
    f"{root_name}/06_manifest/week7_v41_package_manifest.csv",
    f"{root_name}/06_manifest/week7_v41_issues.csv",
    f"{root_name}/06_manifest/week7_v41_decision_summary.csv",
]

member_set = set(member_manifest["member_name"].tolist()) if len(member_manifest) else set()

for member in required_members:
    exists = member in member_set

    required_rows.append({
        "required_member": member,
        "exists": bool(exists),
    })

    if not exists:
        issues.append({
            "item": member,
            "issue_type": "hard_missing_required_zip_member",
            "issue_detail": "Required file missing inside v41b fixed zip.",
        })

# Content checks
if zip_valid:
    with zipfile.ZipFile(FIXED_ZIP, "r") as zf:
        # Key numbers audit
        key_member = f"{root_name}/02_tables/week7_v41_key_numbers.csv"

        if key_member in member_set:
            key_df = read_csv_from_zip(zf, key_member)

            model_ready_row = key_df[
                (key_df["category"].astype(str) == "dataset")
                & (key_df["metric"].astype(str) == "model_ready_clip_rows")
            ]

            if len(model_ready_row) == 1:
                value = clean(model_ready_row.iloc[0]["value"])
                ok = value == "70"

                content_rows.append({
                    "check_name": "key_numbers_model_ready_clip_rows_is_70",
                    "passed": bool(ok),
                    "observed_value": value,
                    "expected_value": "70",
                })

                if not ok:
                    issues.append({
                        "item": key_member,
                        "issue_type": "hard_wrong_model_ready_clip_rows",
                        "issue_detail": f"Expected 70, observed {value}",
                    })
            else:
                content_rows.append({
                    "check_name": "key_numbers_model_ready_clip_rows_row_exists_once",
                    "passed": False,
                    "observed_value": str(len(model_ready_row)),
                    "expected_value": "1",
                })
                issues.append({
                    "item": key_member,
                    "issue_type": "hard_model_ready_clip_rows_row_not_unique",
                    "issue_detail": f"Observed {len(model_ready_row)} rows.",
                })

            final_claim_row = key_df[
                (key_df["category"].astype(str) == "claim_scope")
                & (key_df["metric"].astype(str) == "final_classifier_claim")
            ]

            if len(final_claim_row) == 1:
                value = clean(final_claim_row.iloc[0]["value"])
                ok = value.lower() == "false"

                content_rows.append({
                    "check_name": "key_numbers_final_classifier_claim_false",
                    "passed": bool(ok),
                    "observed_value": value,
                    "expected_value": "False",
                })

                if not ok:
                    issues.append({
                        "item": key_member,
                        "issue_type": "hard_final_classifier_claim_not_false",
                        "issue_detail": f"Observed {value}",
                    })

        # Claim scope checklist audit
        claim_member = f"{root_name}/02_tables/week7_v41_claim_scope_checklist.csv"

        if claim_member in member_set:
            claim_df = read_csv_from_zip(zf, claim_member)
            cannot_final = claim_df[
                (claim_df["claim_type"].astype(str) == "cannot_claim")
                & (claim_df["claim"].astype(str).str.contains("final behaviour classifier", case=False, na=False))
            ]

            ok = len(cannot_final) >= 1

            content_rows.append({
                "check_name": "claim_scope_contains_cannot_claim_final_classifier",
                "passed": bool(ok),
                "observed_value": str(len(cannot_final)),
                "expected_value": ">=1",
            })

            if not ok:
                issues.append({
                    "item": claim_member,
                    "issue_type": "hard_missing_cannot_claim_final_classifier",
                    "issue_detail": "Claim scope checklist must explicitly forbid final classifier claim.",
                })

        # English report audit
        en_member = f"{root_name}/01_final_reports/Week7_Final_Report_Ready_Summary_v41.md"

        if en_member in member_set:
            en_text = read_text_from_zip(zf, en_member)
            checks = [
                (
                    "english_report_has_correct_model_ready_clip_rows_70",
                    "| dataset | model_ready_clip_rows | 70 |",
                    True,
                ),
                (
                    "english_report_does_not_have_old_wrong_model_ready_value",
                    "| dataset | model_ready_clip_rows | 0.4819 |",
                    False,
                ),
                (
                    "english_report_says_not_final_classifier",
                    "not a final classifier",
                    True,
                ),
                (
                    "english_report_says_sanity_check",
                    "sanity-check",
                    True,
                ),
            ]

            for check_name, needle, should_exist in checks:
                exists = needle in en_text
                passed = exists if should_exist else not exists

                content_rows.append({
                    "check_name": check_name,
                    "passed": bool(passed),
                    "observed_value": str(exists),
                    "expected_value": str(should_exist),
                })

                if not passed:
                    issues.append({
                        "item": en_member,
                        "issue_type": "hard_report_content_check_failed",
                        "issue_detail": f"{check_name}: needle={needle}",
                    })

        # Turkish report audit
        tr_member = f"{root_name}/01_final_reports/Week7_Final_Report_Ready_Summary_v41_TR.md"

        if tr_member in member_set:
            tr_text = read_text_from_zip(zf, tr_member)

            checks = [
                (
                    "turkish_report_has_correct_model_ready_clip_rows_70",
                    "| dataset | model_ready_clip_rows | 70 |",
                    True,
                ),
                (
                    "turkish_report_does_not_have_old_wrong_model_ready_value",
                    "| dataset | model_ready_clip_rows | 0.4819 |",
                    False,
                ),
                (
                    "turkish_report_says_final_classifier_degil",
                    "Final classifier değildir",
                    True,
                ),
            ]

            for check_name, needle, should_exist in checks:
                exists = needle in tr_text
                passed = exists if should_exist else not exists

                content_rows.append({
                    "check_name": check_name,
                    "passed": bool(passed),
                    "observed_value": str(exists),
                    "expected_value": str(should_exist),
                })

                if not passed:
                    issues.append({
                        "item": tr_member,
                        "issue_type": "hard_report_content_check_failed",
                        "issue_detail": f"{check_name}: needle={needle}",
                    })

        # v40 nested zip audit
        nested_member = f"{root_name}/03_v40_evidence/Week7_Behaviour_Temporal_Evidence_Package_v40.zip"

        if nested_member in member_set:
            try:
                nested_bytes = zf.read(nested_member)
                with zipfile.ZipFile(io.BytesIO(nested_bytes), "r") as nested:
                    bad = nested.testzip()
                    ok = bad is None
                    nested_count = len(nested.infolist())

                content_rows.append({
                    "check_name": "nested_v40_zip_valid",
                    "passed": bool(ok),
                    "observed_value": f"members={nested_count}, bad={bad}",
                    "expected_value": "valid zip",
                })

                if not ok:
                    issues.append({
                        "item": nested_member,
                        "issue_type": "hard_nested_v40_zip_invalid",
                        "issue_detail": str(bad),
                    })

            except Exception as e:
                content_rows.append({
                    "check_name": "nested_v40_zip_valid",
                    "passed": False,
                    "observed_value": str(e),
                    "expected_value": "valid zip",
                })
                issues.append({
                    "item": nested_member,
                    "issue_type": "hard_nested_v40_zip_open_failed",
                    "issue_detail": str(e),
                })

required_audit = pd.DataFrame(required_rows)
safe_to_csv(required_audit, OUT_REQUIRED_AUDIT)

content_audit = pd.DataFrame(content_rows)
safe_to_csv(content_audit, OUT_CONTENT_AUDIT)

# ---------------------------------------------------------------------
# Final delivery directory
# ---------------------------------------------------------------------

if FIXED_ZIP.exists():
    shutil.copy2(FIXED_ZIP, DELIVERY_DIR / FIXED_ZIP.name)

for p in [
    OUT_MEMBER_MANIFEST,
    OUT_REQUIRED_AUDIT,
    OUT_CONTENT_AUDIT,
    V41B_DECISION,
    V41B_ISSUES,
]:
    if p.exists():
        shutil.copy2(p, DELIVERY_DIR / p.name)

# Issues
hard_issues = [x for x in issues if str(x["issue_type"]).startswith("hard_")]
warnings = [x for x in issues if str(x["issue_type"]).startswith("warning_")]

issues_df = pd.DataFrame(issues, columns=["item", "issue_type", "issue_detail"])
safe_to_csv(issues_df, OUT_ISSUES)
shutil.copy2(OUT_ISSUES, DELIVERY_DIR / OUT_ISSUES.name)

# Decision
required_missing = int((required_audit["exists"] == False).sum()) if len(required_audit) else len(required_members)
content_failed = int((content_audit["passed"] == False).sum()) if len(content_audit) else 0

final_delivery_ready = (
    len(hard_issues) == 0
    and zip_valid
    and zip_test_pass
    and required_missing == 0
    and content_failed == 0
    and expected_sha == computed_sha
)

decision = pd.DataFrame([{
    "v42_decision": "final_audit_delivery_verification_passed" if final_delivery_ready else "final_audit_delivery_verification_issues_found",
    "fixed_zip": str(FIXED_ZIP),
    "zip_size_bytes": int(zip_size_bytes),
    "computed_zip_sha256": computed_sha,
    "expected_zip_sha256_from_v41b": expected_sha,
    "zip_sha256_match": bool(expected_sha == computed_sha),
    "zip_valid": bool(zip_valid),
    "zip_test_pass": bool(zip_test_pass),
    "zip_member_count": int(zip_member_count),
    "required_member_count": int(len(required_members)),
    "required_missing_count": int(required_missing),
    "content_check_count": int(len(content_audit)),
    "content_failed_count": int(content_failed),
    "hard_issue_count": int(len(hard_issues)),
    "warning_count": int(len(warnings)),
    "issue_count": int(len(issues_df)),
    "final_classifier_claim": False,
    "claim_scope_verified": bool(content_failed == 0),
    "final_delivery_ready": bool(final_delivery_ready),
    "recommended_delivery_zip": str(FIXED_ZIP),
    "generated_at": datetime.now().isoformat(timespec="seconds"),
}])

safe_to_csv(decision, OUT_DECISION)
shutil.copy2(OUT_DECISION, DELIVERY_DIR / OUT_DECISION.name)

# README
readme = f"""# Week 7 Final Delivery Verification v42

## Result

- v42 decision: `{decision.iloc[0]["v42_decision"]}`
- Final delivery ready: `{final_delivery_ready}`
- Hard issue count: `{len(hard_issues)}`
- Warning count: `{len(warnings)}`
- Required missing count: `{required_missing}`
- Content failed count: `{content_failed}`

## Recommended delivery package

`{FIXED_ZIP}`

## SHA256

`{computed_sha}`

## Verified facts

- Fixed zip is valid: `{zip_valid}`
- Zip test passed: `{zip_test_pass}`
- SHA256 matches v41b decision: `{expected_sha == computed_sha}`
- Required members present: `{required_missing == 0}`
- Content checks passed: `{content_failed == 0}`
- Correct key number verified: `model_ready_clip_rows = 70`
- Old wrong key number absent: `model_ready_clip_rows = 0.4819`
- Claim scope verified: not final classifier / sanity-check baseline only.

## Delivery note

Use the v41b fixed zip as the final report-ready package. The v42 audit package is supporting proof that the final zip is valid and claim-safe.
"""

OUT_README.write_text(readme)
shutil.copy2(OUT_README, DELIVERY_DIR / OUT_README.name)

# Delivery verification zip
if OUT_DELIVERY_ZIP.exists():
    OUT_DELIVERY_ZIP.unlink()

with zipfile.ZipFile(OUT_DELIVERY_ZIP, "w", compression=zipfile.ZIP_DEFLATED) as zf:
    for p in DELIVERY_DIR.rglob("*"):
        if p.is_file():
            zf.write(p, p.relative_to(DELIVERY_DIR.parent))

delivery_zip_size = OUT_DELIVERY_ZIP.stat().st_size
delivery_zip_sha = sha256_file(OUT_DELIVERY_ZIP)

# Note
OUT_NOTE.write_text(
    "# Week 7 v42 Final Audit / Delivery Verification\n\n"
    "## Summary\n\n"
    f"- v42 decision: `{decision.iloc[0]['v42_decision']}`\n"
    f"- Final delivery ready: `{final_delivery_ready}`\n"
    f"- Recommended delivery zip: `{FIXED_ZIP}`\n"
    f"- Recommended delivery SHA256: `{computed_sha}`\n"
    f"- Zip valid: `{zip_valid}`\n"
    f"- Zip test pass: `{zip_test_pass}`\n"
    f"- Required missing count: `{required_missing}`\n"
    f"- Content failed count: `{content_failed}`\n"
    f"- Hard issue count: `{len(hard_issues)}`\n"
    f"- Warning count: `{len(warnings)}`\n"
    f"- v42 verification package: `{OUT_DELIVERY_ZIP}`\n"
    f"- v42 verification package SHA256: `{delivery_zip_sha}`\n\n"
    "## Claim scope\n\n"
    "The final package is report-ready evidence only. It is not a final classifier claim.\n"
)

print("Saved:")
print(OUT_MEMBER_MANIFEST)
print(OUT_REQUIRED_AUDIT)
print(OUT_CONTENT_AUDIT)
print(OUT_DECISION)
print(OUT_ISSUES)
print(OUT_README)
print(OUT_DELIVERY_ZIP)
print(OUT_NOTE)

print()
print("=== v42 decision ===")
print(decision.to_string(index=False))

print()
print("=== v42 issues ===")
if len(issues_df):
    print(issues_df.to_string(index=False))
else:
    print("No issues found.")
