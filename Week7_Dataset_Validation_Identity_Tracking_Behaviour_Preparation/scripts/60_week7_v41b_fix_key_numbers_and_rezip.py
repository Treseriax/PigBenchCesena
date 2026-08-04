from pathlib import Path
from datetime import datetime
import csv
import hashlib
import zipfile
import pandas as pd


ROOT = Path.home() / "PigBench"
W7 = ROOT / "Week7_Dataset_Validation_Identity_Tracking_Behaviour_Preparation"

V39_DECISION = W7 / "outputs" / "week7_clip_multilabel_temporal_baseline_dryrun_v39" / "week7_v39_clip_multilabel_temporal_baseline_decision_summary.csv"

V41 = W7 / "outputs" / "week7_final_report_ready_summary_visual_package_v41"
PKG = V41 / "Week7_Final_Report_Ready_Summary_Visual_Package_v41"

KEY_OUT = V41 / "tables" / "week7_v41_key_numbers.csv"
KEY_PKG = PKG / "02_tables" / "week7_v41_key_numbers.csv"

REPORT_EN_OUT = V41 / "Week7_Final_Report_Ready_Summary_v41.md"
REPORT_TR_OUT = V41 / "Week7_Final_Report_Ready_Summary_v41_TR.md"
REPORT_EN_PKG = PKG / "01_final_reports" / "Week7_Final_Report_Ready_Summary_v41.md"
REPORT_TR_PKG = PKG / "01_final_reports" / "Week7_Final_Report_Ready_Summary_v41_TR.md"

MANIFEST_OUT = V41 / "week7_v41_package_manifest.csv"
MANIFEST_PKG = PKG / "06_manifest" / "week7_v41_package_manifest.csv"

OUT_DECISION = V41 / "week7_v41b_key_number_fix_decision.csv"
OUT_ISSUES = V41 / "week7_v41b_key_number_fix_issues.csv"
OUT_NOTE = W7 / "notes" / "week7_v41b_key_number_fix_notes.md"

FIXED_ZIP = V41 / "Week7_Final_Report_Ready_Summary_Visual_Package_v41b_FIXED.zip"


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


issues = []

if not V39_DECISION.exists():
    issues.append({
        "item": str(V39_DECISION),
        "issue_type": "hard_missing_v39_decision",
        "issue_detail": "Cannot read correct model_ready_clip_rows.",
    })
    model_ready_clip_rows = ""
else:
    v39 = pd.read_csv(V39_DECISION)
    model_ready_clip_rows = str(v39.iloc[0]["model_ready_clip_rows"])


new_note = "Train/val/test clip rows used for v39 clip-level multi-label baseline."


def fix_key_numbers_csv(path):
    if not path.exists():
        issues.append({
            "item": str(path),
            "issue_type": "hard_missing_key_numbers_csv",
            "issue_detail": "Cannot patch key numbers table.",
        })
        return

    df = pd.read_csv(path)

    mask = (df["category"] == "dataset") & (df["metric"] == "model_ready_clip_rows")

    if mask.sum() != 1:
        issues.append({
            "item": str(path),
            "issue_type": "hard_key_number_row_not_unique",
            "issue_detail": f"Expected exactly one model_ready_clip_rows row, found {int(mask.sum())}.",
        })
        return

    old_value = str(df.loc[mask, "value"].iloc[0])
    df.loc[mask, "value"] = model_ready_clip_rows
    df.loc[mask, "note"] = new_note
    safe_to_csv(df, path)

    print(f"Patched key numbers: {path}")
    print(f"  old value: {old_value}")
    print(f"  new value: {model_ready_clip_rows}")


def fix_report_md(path):
    if not path.exists():
        issues.append({
            "item": str(path),
            "issue_type": "hard_missing_report_md",
            "issue_detail": "Cannot patch report table.",
        })
        return

    text = path.read_text()
    lines = text.splitlines()
    changed = False

    new_line = f"| dataset | model_ready_clip_rows | {model_ready_clip_rows} | {new_note} |"

    fixed_lines = []
    for line in lines:
        if line.startswith("| dataset | model_ready_clip_rows |"):
            fixed_lines.append(new_line)
            changed = True
        else:
            fixed_lines.append(line)

    if not changed:
        issues.append({
            "item": str(path),
            "issue_type": "hard_report_key_number_line_not_found",
            "issue_detail": "Could not find model_ready_clip_rows markdown table row.",
        })
        return

    path.write_text("\n".join(fixed_lines) + "\n")
    print(f"Patched report: {path}")


if model_ready_clip_rows:
    for p in [KEY_OUT, KEY_PKG]:
        fix_key_numbers_csv(p)

    for p in [REPORT_EN_OUT, REPORT_TR_OUT, REPORT_EN_PKG, REPORT_TR_PKG]:
        fix_report_md(p)


# Rebuild package manifest after patch.
manifest_rows = []
if PKG.exists():
    for p in sorted(PKG.rglob("*")):
        if not p.is_file():
            continue

        rel = p.relative_to(PKG)

        if rel.as_posix() == "06_manifest/week7_v41_package_manifest.csv":
            sha = ""
            note = "self hash omitted"
        else:
            sha = sha256_file(p)
            note = ""

        manifest_rows.append({
            "package_relative_path": str(rel),
            "exists": True,
            "size_bytes": p.stat().st_size,
            "sha256": sha,
            "note": note,
        })

    manifest = pd.DataFrame(manifest_rows)
    safe_to_csv(manifest, MANIFEST_OUT)
    safe_to_csv(manifest, MANIFEST_PKG)
else:
    issues.append({
        "item": str(PKG),
        "issue_type": "hard_missing_package_dir",
        "issue_detail": "Package directory missing.",
    })


# Rezip fixed package.
if FIXED_ZIP.exists():
    FIXED_ZIP.unlink()

if PKG.exists():
    with zipfile.ZipFile(FIXED_ZIP, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        for p in PKG.rglob("*"):
            if p.is_file():
                zf.write(p, p.relative_to(PKG.parent))

zip_size = FIXED_ZIP.stat().st_size if FIXED_ZIP.exists() else 0
zip_sha = sha256_file(FIXED_ZIP) if FIXED_ZIP.exists() else ""

hard_issues = [x for x in issues if str(x["issue_type"]).startswith("hard_")]
ready = len(hard_issues) == 0 and FIXED_ZIP.exists()

issues_df = pd.DataFrame(issues, columns=["item", "issue_type", "issue_detail"])
safe_to_csv(issues_df, OUT_ISSUES)

decision = pd.DataFrame([{
    "v41b_decision": "key_number_bug_fixed_and_package_rezipped" if ready else "key_number_fix_issues_found",
    "corrected_metric": "model_ready_clip_rows",
    "corrected_value": model_ready_clip_rows,
    "fixed_zip": str(FIXED_ZIP),
    "fixed_zip_size_bytes": int(zip_size),
    "fixed_zip_sha256": zip_sha,
    "hard_issue_count": int(len(hard_issues)),
    "issue_count": int(len(issues_df)),
    "report_ready_fixed_package": bool(ready),
    "generated_at": datetime.now().isoformat(timespec="seconds"),
}])

safe_to_csv(decision, OUT_DECISION)

OUT_NOTE.write_text(
    "# Week 7 v41b Key Number Fix\n\n"
    "## Summary\n\n"
    f"- Corrected metric: `model_ready_clip_rows`\n"
    f"- Corrected value: `{model_ready_clip_rows}`\n"
    f"- Fixed zip: `{FIXED_ZIP}`\n"
    f"- Fixed zip SHA256: `{zip_sha}`\n"
    f"- Hard issue count: `{len(hard_issues)}`\n"
    f"- Report-ready fixed package: `{ready}`\n\n"
    "## Reason\n\n"
    "The original v41 key numbers table accidentally wrote clip_temporal_test_micro_f1 into the model_ready_clip_rows row.\n"
    "The correct value is taken from v39 decision summary.\n"
)

print()
print("=== v41b decision ===")
print(decision.to_string(index=False))

print()
print("=== v41b issues ===")
if len(issues_df):
    print(issues_df.to_string(index=False))
else:
    print("No issues found.")
