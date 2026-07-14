from pathlib import Path
from datetime import datetime
import csv
import hashlib
import zipfile
import pandas as pd


ROOT = Path.home() / "PigBench"
W7 = ROOT / "Week7_Dataset_Validation_Identity_Tracking_Behaviour_Preparation"

V30_ROOT = W7 / "outputs" / "week7_complete_final_package_v30"
PKG = V30_ROOT / "Week7_Complete_Final_Package_v30"
ZIP_PATH = V30_ROOT / "Week7_Complete_Final_Package_v30.zip"
V30_DECISION = V30_ROOT / "week7_complete_final_package_v30_decision_summary.csv"

OUT_ROOT = W7 / "outputs" / "week7_complete_final_audit_v31"
OUT_ROOT.mkdir(parents=True, exist_ok=True)

OUT_DECISION = OUT_ROOT / "week7_complete_final_audit_v31_decision_summary.csv"
OUT_FILE_INVENTORY = OUT_ROOT / "week7_complete_final_audit_v31_package_file_inventory.csv"
OUT_ZIP_INVENTORY = OUT_ROOT / "week7_complete_final_audit_v31_zip_inventory.csv"
OUT_SUBPACKAGE_AUDIT = OUT_ROOT / "week7_complete_final_audit_v31_subpackage_zip_audit.csv"
OUT_REPORT_AUDIT = OUT_ROOT / "week7_complete_final_audit_v31_report_content_audit.csv"
OUT_MANIFEST_AUDIT = OUT_ROOT / "week7_complete_final_audit_v31_manifest_audit.csv"
OUT_ISSUES = OUT_ROOT / "week7_complete_final_audit_v31_issues.csv"
OUT_REPORT = OUT_ROOT / "week7_complete_final_audit_v31_report.md"
OUT_NOTE = W7 / "notes" / "week7_complete_final_audit_v31_notes.md"


def safe_to_csv(df, path):
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


def read_first(path):
    path = Path(path)
    if not path.exists():
        return {}
    try:
        df = pd.read_csv(path)
        if len(df):
            return df.iloc[0].to_dict()
    except Exception:
        return {}
    return {}


def clean(v):
    if pd.isna(v):
        return ""
    return str(v).strip()


issues = []

# ---------------------------------------------------------------------
# Basic existence checks
# ---------------------------------------------------------------------

for key, path in [
    ("v30_package_dir", PKG),
    ("v30_zip", ZIP_PATH),
    ("v30_decision", V30_DECISION),
]:
    if not path.exists():
        issues.append({
            "issue_type": "missing_required_path",
            "item": key,
            "issue_detail": str(path),
        })

v30 = read_first(V30_DECISION)

expected_zip_sha = clean(v30.get("zip_sha256", ""))
expected_issue_count = clean(v30.get("issue_count", ""))
expected_status = clean(v30.get("final_identity_status", ""))
expected_v29c_overwrites = clean(v30.get("v29c_overwrites_v29b", ""))

# ---------------------------------------------------------------------
# Package file inventory
# ---------------------------------------------------------------------

file_rows = []

if PKG.exists():
    for p in sorted(PKG.rglob("*")):
        if not p.is_file():
            continue

        rel = p.relative_to(PKG)
        try:
            size = p.stat().st_size
            sha = sha256_file(p)
        except Exception as e:
            size = ""
            sha = ""
            issues.append({
                "issue_type": "file_inventory_error",
                "item": str(rel),
                "issue_detail": str(e),
            })

        file_rows.append({
            "relative_path": str(rel),
            "size_bytes": size,
            "sha256": sha,
        })

file_inventory = pd.DataFrame(file_rows)
safe_to_csv(file_inventory, OUT_FILE_INVENTORY)

# ---------------------------------------------------------------------
# Zip validation
# ---------------------------------------------------------------------

zip_rows = []
zip_exists = ZIP_PATH.exists()
zip_is_valid = False
zip_test_pass = False
zip_sha = ""
zip_size = ""

if zip_exists:
    try:
        zip_size = ZIP_PATH.stat().st_size
        zip_sha = sha256_file(ZIP_PATH)
        zip_is_valid = zipfile.is_zipfile(ZIP_PATH)

        if zip_is_valid:
            with zipfile.ZipFile(ZIP_PATH, "r") as zf:
                bad = zf.testzip()
                zip_test_pass = bad is None

                if bad is not None:
                    issues.append({
                        "issue_type": "zip_test_failed",
                        "item": "v30_zip",
                        "issue_detail": str(bad),
                    })

                for info in zf.infolist():
                    if info.is_dir():
                        continue

                    zip_rows.append({
                        "zip_member": info.filename,
                        "compressed_size": info.compress_size,
                        "uncompressed_size": info.file_size,
                    })
        else:
            issues.append({
                "issue_type": "invalid_zip",
                "item": "v30_zip",
                "issue_detail": str(ZIP_PATH),
            })
    except Exception as e:
        issues.append({
            "issue_type": "zip_validation_error",
            "item": "v30_zip",
            "issue_detail": str(e),
        })

if expected_zip_sha and zip_sha and expected_zip_sha != zip_sha:
    issues.append({
        "issue_type": "sha256_mismatch",
        "item": "v30_zip",
        "issue_detail": f"expected={expected_zip_sha}; computed={zip_sha}",
    })

zip_inventory = pd.DataFrame(zip_rows)
safe_to_csv(zip_inventory, OUT_ZIP_INVENTORY)

# ---------------------------------------------------------------------
# Manifest audit
# ---------------------------------------------------------------------

manifest_path = PKG / "manifests" / "week7_complete_final_package_v30_manifest.csv"
manifest_rows = []

manifest_exists = manifest_path.exists()
manifest_total_rows = 0
manifest_missing_paths = 0

if manifest_exists:
    try:
        manifest = pd.read_csv(manifest_path)
        manifest_total_rows = len(manifest)

        for _, r in manifest.iterrows():
            package_path = Path(clean(r.get("package_path", "")))
            exists = package_path.exists()

            if not exists:
                manifest_missing_paths += 1

            manifest_rows.append({
                "item_key": clean(r.get("item_key", "")),
                "package_section": clean(r.get("package_section", "")),
                "package_path": str(package_path),
                "exists": bool(exists),
                "size_bytes_manifest": clean(r.get("size_bytes", "")),
            })

        if manifest_missing_paths > 0:
            issues.append({
                "issue_type": "manifest_paths_missing",
                "item": "v30_manifest",
                "issue_detail": f"missing_paths={manifest_missing_paths}",
            })

    except Exception as e:
        issues.append({
            "issue_type": "manifest_read_error",
            "item": "v30_manifest",
            "issue_detail": str(e),
        })
else:
    issues.append({
        "issue_type": "missing_manifest",
        "item": "v30_manifest",
        "issue_detail": str(manifest_path),
    })

manifest_audit = pd.DataFrame(manifest_rows)
safe_to_csv(manifest_audit, OUT_MANIFEST_AUDIT)

# ---------------------------------------------------------------------
# Subpackage zip audit
# ---------------------------------------------------------------------

subpackage_rows = []

subpkg_dir = PKG / "subpackages"
required_subpackages = [
    "Week7_Final_Audit_Package_v24.zip",
    "Week7_Final_Identity_Linking_Report_Package_v29e.zip",
]

for name in required_subpackages:
    p = subpkg_dir / name

    row = {
        "subpackage_zip": name,
        "path": str(p),
        "exists": p.exists(),
        "is_zip": False,
        "testzip_pass": False,
        "size_bytes": "",
        "sha256": "",
    }

    if not p.exists():
        issues.append({
            "issue_type": "missing_required_subpackage",
            "item": name,
            "issue_detail": str(p),
        })
    else:
        try:
            row["size_bytes"] = p.stat().st_size
            row["sha256"] = sha256_file(p)
            row["is_zip"] = zipfile.is_zipfile(p)

            if row["is_zip"]:
                with zipfile.ZipFile(p, "r") as zf:
                    bad = zf.testzip()
                    row["testzip_pass"] = bad is None

                    if bad is not None:
                        issues.append({
                            "issue_type": "subpackage_zip_test_failed",
                            "item": name,
                            "issue_detail": str(bad),
                        })
            else:
                issues.append({
                    "issue_type": "subpackage_invalid_zip",
                    "item": name,
                    "issue_detail": str(p),
                })

        except Exception as e:
            issues.append({
                "issue_type": "subpackage_audit_error",
                "item": name,
                "issue_detail": str(e),
            })

    subpackage_rows.append(row)

subpackage_audit = pd.DataFrame(subpackage_rows)
safe_to_csv(subpackage_audit, OUT_SUBPACKAGE_AUDIT)

# ---------------------------------------------------------------------
# Report content audit
# ---------------------------------------------------------------------

report_checks = [
    {
        "check_name": "main_report_exists",
        "path": PKG / "Week7_Complete_Final_Report_v30.md",
        "required_text": "",
    },
    {
        "check_name": "states_v29c_does_not_overwrite_v29b",
        "path": PKG / "Week7_Complete_Final_Report_v30.md",
        "required_text": "v29c does not overwrite v29b",
    },
    {
        "check_name": "states_primary_identity_anchor",
        "path": PKG / "Week7_Complete_Final_Report_v30.md",
        "required_text": "v29b center-frame GT-overlap identity is the primary identity anchor",
    },
    {
        "check_name": "states_secondary_evidence_only",
        "path": PKG / "Week7_Complete_Final_Report_v30.md",
        "required_text": "v29c full-tracklet HSV colour evidence is secondary evidence only",
    },
    {
        "check_name": "states_not_final_production_tracking",
        "path": PKG / "Week7_Complete_Final_Report_v30.md",
        "required_text": "not final production tracking",
    },
    {
        "check_name": "states_review_queue",
        "path": PKG / "Week7_Complete_Final_Report_v30.md",
        "required_text": "review queue",
    },
]

report_rows = []

for chk in report_checks:
    p = Path(chk["path"])
    required = chk["required_text"]

    row = {
        "check_name": chk["check_name"],
        "path": str(p),
        "exists": p.exists(),
        "required_text": required,
        "check_pass": False,
    }

    if not p.exists():
        issues.append({
            "issue_type": "missing_report_file",
            "item": chk["check_name"],
            "issue_detail": str(p),
        })
    else:
        try:
            text = p.read_text(errors="ignore")
            if required == "":
                row["check_pass"] = True
            else:
                row["check_pass"] = required.lower() in text.lower()

            if not row["check_pass"]:
                issues.append({
                    "issue_type": "report_content_check_failed",
                    "item": chk["check_name"],
                    "issue_detail": f"missing text: {required}",
                })

        except Exception as e:
            issues.append({
                "issue_type": "report_read_error",
                "item": chk["check_name"],
                "issue_detail": str(e),
            })

    report_rows.append(row)

report_audit = pd.DataFrame(report_rows)
safe_to_csv(report_audit, OUT_REPORT_AUDIT)

# ---------------------------------------------------------------------
# Final decision
# ---------------------------------------------------------------------

issues_df = pd.DataFrame(issues, columns=["issue_type", "item", "issue_detail"])
safe_to_csv(issues_df, OUT_ISSUES)

package_file_count = len(file_inventory)
package_total_bytes = int(file_inventory["size_bytes"].replace("", 0).astype(float).sum()) if len(file_inventory) else 0
zip_member_count = len(zip_inventory)

sha_match = bool(expected_zip_sha and zip_sha and expected_zip_sha == zip_sha)

audit_pass = (
    len(issues_df) == 0
    and zip_exists
    and zip_is_valid
    and zip_test_pass
    and sha_match
    and manifest_exists
    and manifest_missing_paths == 0
)

decision = pd.DataFrame([{
    "v31_decision": "week7_complete_final_audit_passed" if audit_pass else "week7_complete_final_audit_issues_found",
    "audit_pass": bool(audit_pass),
    "package_dir": str(PKG),
    "zip_path": str(ZIP_PATH),
    "package_file_count": int(package_file_count),
    "package_total_bytes": int(package_total_bytes),
    "zip_size_bytes": zip_size,
    "expected_zip_sha256": expected_zip_sha,
    "computed_zip_sha256": zip_sha,
    "zip_sha256_match": bool(sha_match),
    "zip_is_valid": bool(zip_is_valid),
    "zip_test_pass": bool(zip_test_pass),
    "zip_member_count": int(zip_member_count),
    "manifest_exists": bool(manifest_exists),
    "manifest_total_rows": int(manifest_total_rows),
    "manifest_missing_paths": int(manifest_missing_paths),
    "required_subpackages_checked": int(len(required_subpackages)),
    "subpackage_zip_pass_count": int(subpackage_audit["testzip_pass"].sum()) if len(subpackage_audit) else 0,
    "report_checks_passed": int(report_audit["check_pass"].sum()) if len(report_audit) else 0,
    "report_checks_total": int(len(report_audit)),
    "v30_issue_count_reported": expected_issue_count,
    "final_identity_status": expected_status,
    "v29c_overwrites_v29b": expected_v29c_overwrites,
    "issue_count": int(len(issues_df)),
    "ready_for_v32_report_ready_text": bool(audit_pass),
    "generated_at": datetime.now().isoformat(timespec="seconds"),
}])

safe_to_csv(decision, OUT_DECISION)

# ---------------------------------------------------------------------
# Markdown report and note
# ---------------------------------------------------------------------

md = f"""# Week 7 Complete Final Audit v31

## Purpose

This audit verifies the Week 7 Complete Final Package v30.

It checks:

1. package folder existence,
2. v30 zip existence,
3. SHA256 consistency,
4. zip readability,
5. manifest path consistency,
6. required subpackage zip integrity,
7. main report content checks.

## Result

- Audit decision: `{decision.iloc[0]['v31_decision']}`
- Audit pass: `{bool(audit_pass)}`
- Issue count: `{len(issues_df)}`
- Ready for v32 report-ready text: `{bool(audit_pass)}`

## Package

- Package dir: `{PKG}`
- Zip path: `{ZIP_PATH}`
- Package file count: `{package_file_count}`
- Package total bytes: `{package_total_bytes}`
- Zip size bytes: `{zip_size}`

## Checksum

- Expected SHA256: `{expected_zip_sha}`
- Computed SHA256: `{zip_sha}`
- SHA256 match: `{bool(sha_match)}`

## Zip audit

- Zip is valid: `{bool(zip_is_valid)}`
- Zip test pass: `{bool(zip_test_pass)}`
- Zip member count: `{zip_member_count}`

## Manifest audit

- Manifest exists: `{bool(manifest_exists)}`
- Manifest rows: `{manifest_total_rows}`
- Missing manifest paths: `{manifest_missing_paths}`

## Subpackage audit

Required subpackages:

- `Week7_Final_Audit_Package_v24.zip`
- `Week7_Final_Identity_Linking_Report_Package_v29e.zip`

Subpackage zip pass count: `{int(subpackage_audit['testzip_pass'].sum()) if len(subpackage_audit) else 0}` / `{len(required_subpackages)}`

## Report content audit

Report checks passed: `{int(report_audit['check_pass'].sum()) if len(report_audit) else 0}` / `{len(report_audit)}`

## Final interpretation

The v30 package is considered audit-ready only if all checks pass. If this audit passes, the next professional step is v32: report-ready methodology and results text.
"""

OUT_REPORT.write_text(md)

OUT_NOTE.write_text(
    "# Week 7 Complete Final Audit v31\n\n"
    "## Summary\n\n"
    f"- Audit decision: `{decision.iloc[0]['v31_decision']}`\n"
    f"- Audit pass: `{bool(audit_pass)}`\n"
    f"- Package file count: `{package_file_count}`\n"
    f"- Zip member count: `{zip_member_count}`\n"
    f"- Zip SHA256 match: `{bool(sha_match)}`\n"
    f"- Zip test pass: `{bool(zip_test_pass)}`\n"
    f"- Manifest exists: `{bool(manifest_exists)}`\n"
    f"- Manifest missing paths: `{manifest_missing_paths}`\n"
    f"- Subpackage zip pass count: `{int(subpackage_audit['testzip_pass'].sum()) if len(subpackage_audit) else 0}` / `{len(required_subpackages)}`\n"
    f"- Report checks passed: `{int(report_audit['check_pass'].sum()) if len(report_audit) else 0}` / `{len(report_audit)}`\n"
    f"- Issue count: `{len(issues_df)}`\n"
    f"- Ready for v32 report-ready text: `{bool(audit_pass)}`\n\n"
    "## Outputs\n\n"
    f"- Decision summary: `{OUT_DECISION}`\n"
    f"- File inventory: `{OUT_FILE_INVENTORY}`\n"
    f"- Zip inventory: `{OUT_ZIP_INVENTORY}`\n"
    f"- Subpackage audit: `{OUT_SUBPACKAGE_AUDIT}`\n"
    f"- Report audit: `{OUT_REPORT_AUDIT}`\n"
    f"- Manifest audit: `{OUT_MANIFEST_AUDIT}`\n"
    f"- Issues: `{OUT_ISSUES}`\n"
)

print("Saved:")
print(OUT_DECISION)
print(OUT_FILE_INVENTORY)
print(OUT_ZIP_INVENTORY)
print(OUT_SUBPACKAGE_AUDIT)
print(OUT_REPORT_AUDIT)
print(OUT_MANIFEST_AUDIT)
print(OUT_ISSUES)
print(OUT_REPORT)
print(OUT_NOTE)

print()
print("=== v31 decision ===")
print(decision.to_string(index=False))

print()
print("=== v31 issues ===")
if len(issues_df):
    print(issues_df.to_string(index=False))
else:
    print("No issues found.")
