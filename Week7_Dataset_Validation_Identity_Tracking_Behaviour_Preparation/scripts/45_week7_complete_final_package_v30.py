from pathlib import Path
from datetime import datetime
import csv
import shutil
import hashlib
import zipfile
import pandas as pd


ROOT = Path.home() / "PigBench"
W7 = ROOT / "Week7_Dataset_Validation_Identity_Tracking_Behaviour_Preparation"

OUT_ROOT = W7 / "outputs" / "week7_complete_final_package_v30"
PKG = OUT_ROOT / "Week7_Complete_Final_Package_v30"

REPORTS = PKG / "reports"
TABLES = PKG / "tables"
SUBPACKAGES = PKG / "subpackages"
NOTES = PKG / "notes"
SCRIPTS = PKG / "scripts"
VISUALS = PKG / "selected_visual_previews"
MANIFESTS = PKG / "manifests"

for p in [OUT_ROOT, PKG, REPORTS, TABLES, SUBPACKAGES, NOTES, SCRIPTS, VISUALS, MANIFESTS]:
    p.mkdir(parents=True, exist_ok=True)

ZIP_PATH = OUT_ROOT / "Week7_Complete_Final_Package_v30.zip"

OUT_DECISION = OUT_ROOT / "week7_complete_final_package_v30_decision_summary.csv"
OUT_MANIFEST = OUT_ROOT / "week7_complete_final_package_v30_manifest.csv"
OUT_ISSUES = OUT_ROOT / "week7_complete_final_package_v30_issues.csv"
OUT_REPORT = PKG / "Week7_Complete_Final_Report_v30.md"
OUT_README = PKG / "README_Week7_Complete_Final_Package_v30.md"
OUT_NOTE = W7 / "notes" / "week7_complete_final_package_v30_notes.md"


def safe_to_csv(df, path):
    df.to_csv(
        path,
        index=False,
        quoting=csv.QUOTE_ALL,
        escapechar="\\",
        lineterminator="\n",
    )


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


def copy_file(src, dst_dir, new_name=None):
    src = Path(src)
    if not src.exists() or not src.is_file():
        return None

    dst_dir.mkdir(parents=True, exist_ok=True)
    dst = dst_dir / (new_name or src.name)

    if dst.exists():
        stem = dst.stem
        suffix = dst.suffix
        i = 2
        while True:
            candidate = dst_dir / f"{stem}__copy{i}{suffix}"
            if not candidate.exists():
                dst = candidate
                break
            i += 1

    shutil.copy2(src, dst)
    return dst


def sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def add_manifest(manifest_rows, section, key, src, dst):
    if dst is None:
        return
    manifest_rows.append({
        "package_section": section,
        "item_key": key,
        "source_path": str(src),
        "package_path": str(dst),
        "size_bytes": dst.stat().st_size,
    })


def md_table_from_rows(rows, cols):
    if not rows:
        return "_No rows._"

    out = []
    out.append("| " + " | ".join(cols) + " |")
    out.append("| " + " | ".join(["---"] * len(cols)) + " |")

    for r in rows:
        vals = []
        for c in cols:
            val = str(r.get(c, ""))
            val = val.replace("\n", " ").replace("|", "/")
            vals.append(val)
        out.append("| " + " | ".join(vals) + " |")

    return "\n".join(out)


issues = []
manifest_rows = []

# ---------------------------------------------------------------------
# Required subpackages
# ---------------------------------------------------------------------

expected_subpackages = {
    "v24_week7_audit_package": W7 / "outputs" / "final_audit_package_v24" / "Week7_Final_Audit_Package_v24.zip",
    "v29e_identity_linking_package": W7 / "outputs" / "final_identity_linking_report_package_v29e" / "Week7_Final_Identity_Linking_Report_Package_v29e.zip",
}

for key, path in expected_subpackages.items():
    if not path.exists():
        issues.append({
            "item": key,
            "issue_type": "missing_subpackage",
            "issue_detail": str(path),
        })
    else:
        dst = copy_file(path, SUBPACKAGES, path.name)
        add_manifest(manifest_rows, "subpackages", key, path, dst)

# ---------------------------------------------------------------------
# Key exact files
# ---------------------------------------------------------------------

key_files = {
    "v29e_decision": W7 / "outputs" / "final_identity_linking_report_package_v29e" / "week7_final_identity_linking_report_package_v29e_decision_summary.csv",
    "v29e_report": W7 / "outputs" / "final_identity_linking_report_package_v29e" / "Week7_Final_Identity_Linking_Report_Package_v29e" / "Week7_Final_Identity_Linking_Report_v29e.md",

    "v29d_decision_fixed": W7 / "outputs" / "conservative_identity_arbitration_v29d" / "week7_conservative_identity_arbitration_v29d_decision_summary_fixed.csv",
    "v29d_review_queue": W7 / "outputs" / "conservative_identity_arbitration_v29d" / "week7_conservative_identity_arbitration_v29d_review_queue.csv",
    "v29d_merge_candidates": W7 / "outputs" / "conservative_identity_arbitration_v29d" / "week7_conservative_identity_arbitration_v29d_conservative_merge_candidates.csv",

    "v29c_decision": W7 / "outputs" / "full_tracklet_colour_evidence_v29c" / "week7_full_tracklet_colour_evidence_v29c_decision_summary.csv",
    "v29b_decision": W7 / "outputs" / "colour_constrained_tracklet_linking_v29b" / "week7_colour_constrained_tracklet_linking_v29b_decision_summary.csv",
    "v29a_recommendation": W7 / "outputs" / "tracker_strategy_decision_v29a" / "week7_tracker_strategy_decision_v29a_recommendation.csv",
}

for key, path in key_files.items():
    if not path.exists():
        issues.append({
            "item": key,
            "issue_type": "missing_key_file",
            "issue_detail": str(path),
        })
        continue

    if path.suffix.lower() == ".md":
        dst = copy_file(path, REPORTS, f"{key}__{path.name}")
        add_manifest(manifest_rows, "reports", key, path, dst)
    else:
        dst = copy_file(path, TABLES, f"{key}__{path.name}")
        add_manifest(manifest_rows, "tables", key, path, dst)

# ---------------------------------------------------------------------
# Discover and copy all important CSV/MD summaries from outputs
# ---------------------------------------------------------------------

important_patterns = [
    "*decision_summary*.csv",
    "*summary*.csv",
    "*issues*.csv",
    "*manifest*.csv",
    "*recommendation*.csv",
    "*strategy_options*.csv",
    "*feasibility*.csv",
    "*report*.md",
    "*limitations*.md",
    "README*.md",
]

seen_sources = set()

for pattern in important_patterns:
    for src in sorted((W7 / "outputs").rglob(pattern)):
        if not src.is_file():
            continue

        if src in seen_sources:
            continue

        if "week7_complete_final_package_v30" in str(src):
            continue

        seen_sources.add(src)

        rel_parent = src.parent.relative_to(W7 / "outputs")
        safe_parent = "__".join(rel_parent.parts)

        if src.suffix.lower() == ".md":
            dst_name = f"{safe_parent}__{src.name}"
            dst = copy_file(src, REPORTS, dst_name)
            add_manifest(manifest_rows, "reports_discovered", "discovered_md", src, dst)
        else:
            dst_name = f"{safe_parent}__{src.name}"
            dst = copy_file(src, TABLES, dst_name)
            add_manifest(manifest_rows, "tables_discovered", "discovered_csv", src, dst)

# ---------------------------------------------------------------------
# Copy notes
# ---------------------------------------------------------------------

if (W7 / "notes").exists():
    for src in sorted((W7 / "notes").glob("week7_*.md")):
        dst = copy_file(src, NOTES, src.name)
        add_manifest(manifest_rows, "notes", src.stem, src, dst)

# ---------------------------------------------------------------------
# Copy scripts for reproducibility
# ---------------------------------------------------------------------

if (W7 / "scripts").exists():
    for src in sorted((W7 / "scripts").glob("*.py")):
        dst = copy_file(src, SCRIPTS, src.name)
        add_manifest(manifest_rows, "scripts", src.stem, src, dst)

# ---------------------------------------------------------------------
# Copy selected visual previews
# ---------------------------------------------------------------------

visual_dirs = [
    W7 / "outputs" / "colour_constrained_tracklet_linking_v29b" / "contact_sheets",
    W7 / "outputs" / "full_tracklet_colour_evidence_v29c" / "contact_sheets",
    W7 / "outputs" / "dense_polygon_filtered_tracking_v28d" / "contact_sheets",
    W7 / "outputs" / "detector_tracker_dryrun_v27b" / "contact_sheets",
    W7 / "outputs" / "clip_extraction_temporal_qa_v26" / "contact_sheets",
]

for vdir in visual_dirs:
    if not vdir.exists():
        continue

    rel = vdir.relative_to(W7 / "outputs")
    dst_dir = VISUALS / "__".join(rel.parts)

    # Keep package size controlled: copy up to 60 jpg/png files per visual dir.
    files = sorted(list(vdir.glob("*.jpg")) + list(vdir.glob("*.png")))[:60]

    for src in files:
        dst = copy_file(src, dst_dir, src.name)
        add_manifest(manifest_rows, "selected_visual_previews", "selected_visual", src, dst)

# ---------------------------------------------------------------------
# Read important metrics
# ---------------------------------------------------------------------

v29e_decision_path = key_files["v29e_decision"]
v29e = read_first(v29e_decision_path)

total_tracklets = int(v29e.get("total_tracklets", 0) or 0)
accepted_tracklets = int(v29e.get("conservative_accepted_tracklets", 0) or 0)
acceptance_rate = float(v29e.get("conservative_acceptance_rate", 0) or 0)
review_queue = int(v29e.get("review_queue_tracklets", 0) or 0)
merge_candidates = int(v29e.get("conservative_merge_candidates", 0) or 0)
identity_issue_count = int(v29e.get("issue_count", 0) or 0)
zip_v29e = str(v29e.get("zip_path", ""))
zip_v29e_sha = str(v29e.get("zip_sha256", ""))

subpackage_count = len([r for r in manifest_rows if r["package_section"] == "subpackages"])
copied_file_count_before = len(manifest_rows)

# ---------------------------------------------------------------------
# Final report
# ---------------------------------------------------------------------

stage_rows = [
    {
        "stage": "v24",
        "name": "Week 7 audit package",
        "status": "included_as_subpackage_if_available",
        "main_output": "Week7_Final_Audit_Package_v24.zip",
    },
    {
        "stage": "v29a",
        "name": "Tracker strategy decision",
        "status": "copied",
        "main_output": "simple IoU not final; colour-constrained linking recommended",
    },
    {
        "stage": "v29b",
        "name": "Colour-constrained tracklet identity linking",
        "status": "copied",
        "main_output": "center-frame GT-overlap identity anchor",
    },
    {
        "stage": "v29c",
        "name": "Full-tracklet HSV colour evidence",
        "status": "copied",
        "main_output": "secondary evidence only; conflicts observed",
    },
    {
        "stage": "v29d",
        "name": "Conservative identity arbitration",
        "status": "copied",
        "main_output": "v29c does not overwrite v29b",
    },
    {
        "stage": "v29e",
        "name": "Final identity-linking report package",
        "status": "included_as_subpackage",
        "main_output": "Week7_Final_Identity_Linking_Report_Package_v29e.zip",
    },
]

stage_table = md_table_from_rows(stage_rows, ["stage", "name", "status", "main_output"])

report = f"""# Week 7 Complete Final Report v30

## Executive summary

This package is the complete Week 7 final package for dataset validation, identity tracking preparation and behaviour-readiness documentation.

It includes the earlier Week 7 audit package, the final identity-linking package, key summary tables, reports, notes, scripts and selected visual previews.

## Main conclusion

The Week 7 identity-linking pipeline is now documented as a conservative, review-aware pipeline.

Final identity rule:

- v29b center-frame GT-overlap identity is the primary identity anchor.
- v29c full-tracklet HSV colour evidence is secondary evidence only.
- v29c does not overwrite v29b.
- Conflicts, recovered candidates, low-confidence candidates and unknowns remain in the review queue.

## Key identity-linking metrics

- Total tracklets evaluated: `{total_tracklets}`
- Conservative accepted tracklets: `{accepted_tracklets}`
- Conservative acceptance rate: `{acceptance_rate:.2%}`
- Review queue tracklets: `{review_queue}`
- Conservative merge candidates: `{merge_candidates}`
- Identity package issue count: `{identity_issue_count}`
- v29e package SHA256: `{zip_v29e_sha}`

## Included stages

{stage_table}

## Final package structure

- `reports/`: final reports and discovered markdown reports
- `tables/`: key CSV summaries, issues, manifests and decision tables
- `subpackages/`: v24 audit zip and v29e identity-linking zip
- `notes/`: Week 7 generated notes
- `scripts/`: Week 7 scripts for reproducibility
- `selected_visual_previews/`: selected contact sheets and visual QA outputs
- `manifests/`: package manifest

## What this package can be used for

1. Supervisor review of Week 7 progress.
2. Reproducibility audit of Week 7 outputs.
3. Report writing for methodology/results.
4. Next-phase planning for behaviour representation and improved tracking.
5. Evidence that identity-linking was not overclaimed as production MOT.

## Important limitations

1. The identity tracking output is conservative identity candidates, not final production tracking.
2. Simple IoU tracking remains fragmented in crowded/occluded scenes.
3. HSV full-tracklet evidence is useful diagnostically but too conflict-prone to overwrite GT-overlap identity anchors.
4. The review queue must be inspected before using recovered candidates.
5. Unknown identities are intentionally preserved rather than forced into a colour class.

## Recommended next phase

The next professional phase should be one of the following:

1. final audit/checksum of the v30 package;
2. report-ready methodology and results text;
3. behaviour / clip-level representation preparation;
4. stronger tracker feasibility with ByteTrack/BoT-SORT if dependencies become stable;
5. hybrid improved tracker plus marker-colour constrained linking.
"""

OUT_REPORT.write_text(report)

readme = f"""# Week 7 Complete Final Package v30

## Purpose

This package collects Week 7 final outputs into one deliverable folder and zip.

## Main subpackages

- `Week7_Final_Audit_Package_v24.zip`
- `Week7_Final_Identity_Linking_Report_Package_v29e.zip`

## Main decision

v29c does not overwrite v29b. v29b remains the primary identity anchor and v29c remains secondary support evidence.

## Zip

`{ZIP_PATH}`
"""

OUT_README.write_text(readme)

# ---------------------------------------------------------------------
# Issues, decision, manifest
# ---------------------------------------------------------------------

issues_df = pd.DataFrame(issues, columns=["item", "issue_type", "issue_detail"])
safe_to_csv(issues_df, OUT_ISSUES)

decision = pd.DataFrame([{
    "v30_decision": "week7_complete_final_package_created",
    "total_tracklets_from_v29e": total_tracklets,
    "conservative_accepted_tracklets_from_v29e": accepted_tracklets,
    "conservative_acceptance_rate_from_v29e": acceptance_rate,
    "review_queue_tracklets_from_v29e": review_queue,
    "conservative_merge_candidates_from_v29e": merge_candidates,
    "v29c_overwrites_v29b": False,
    "final_identity_status": "conservative_identity_candidates_not_final_production_tracking",
    "subpackages_included": subpackage_count,
    "copied_file_count_before_zip": copied_file_count_before,
    "package_dir": str(PKG),
    "zip_path": str(ZIP_PATH),
    "issue_count": len(issues_df),
    "generated_at": datetime.now().isoformat(timespec="seconds"),
}])

safe_to_csv(decision, OUT_DECISION)

# Copy v30 generated files into package.
generated_files = [
    (OUT_REPORT, REPORTS, "v30_final_report"),
    (OUT_README, REPORTS, "v30_readme"),
    (OUT_DECISION, TABLES, "v30_decision"),
    (OUT_ISSUES, TABLES, "v30_issues"),
]

for src, dst_dir, key in generated_files:
    dst = copy_file(src, dst_dir, src.name)
    add_manifest(manifest_rows, "v30_generated", key, src, dst)

manifest = pd.DataFrame(manifest_rows)
safe_to_csv(manifest, OUT_MANIFEST)

dst_manifest = copy_file(OUT_MANIFEST, MANIFESTS, OUT_MANIFEST.name)
add_manifest(manifest_rows, "manifests", "v30_manifest", OUT_MANIFEST, dst_manifest)

manifest = pd.DataFrame(manifest_rows)
safe_to_csv(manifest, OUT_MANIFEST)
if dst_manifest:
    safe_to_csv(manifest, dst_manifest)

# ---------------------------------------------------------------------
# Zip
# ---------------------------------------------------------------------

if ZIP_PATH.exists():
    ZIP_PATH.unlink()

with zipfile.ZipFile(ZIP_PATH, "w", compression=zipfile.ZIP_DEFLATED) as zf:
    for p in sorted(PKG.rglob("*")):
        if p.is_file():
            zf.write(p, p.relative_to(PKG.parent))

zip_size = ZIP_PATH.stat().st_size
zip_sha = sha256_file(ZIP_PATH)

decision.loc[0, "zip_size_bytes"] = zip_size
decision.loc[0, "zip_sha256"] = zip_sha
safe_to_csv(decision, OUT_DECISION)
copy_file(OUT_DECISION, TABLES, OUT_DECISION.name)

OUT_NOTE.write_text(
    "# Week 7 Complete Final Package v30\n\n"
    "## Summary\n\n"
    f"- Total tracklets from v29e: `{total_tracklets}`\n"
    f"- Conservative accepted tracklets from v29e: `{accepted_tracklets}`\n"
    f"- Conservative acceptance rate from v29e: `{acceptance_rate:.2%}`\n"
    f"- Review queue tracklets from v29e: `{review_queue}`\n"
    f"- Conservative merge candidates from v29e: `{merge_candidates}`\n"
    f"- v29c overwrites v29b: `False`\n"
    f"- Subpackages included: `{subpackage_count}`\n"
    f"- Package dir: `{PKG}`\n"
    f"- Zip path: `{ZIP_PATH}`\n"
    f"- Zip size bytes: `{zip_size}`\n"
    f"- Zip SHA256: `{zip_sha}`\n"
    f"- Issue count: `{len(issues_df)}`\n\n"
    "## Final decision\n\n"
    "Week 7 complete final package has been created. The package is report-ready and audit-ready.\n"
)

print("Saved package:")
print(PKG)
print()
print("Saved zip:")
print(ZIP_PATH)
print()
print("Zip SHA256:")
print(zip_sha)
print()
print("=== v30 decision ===")
print(decision.to_string(index=False))
print()
print("=== package file count ===")
print(sum(1 for x in PKG.rglob('*') if x.is_file()))
print()
print("=== issues ===")
if len(issues_df):
    print(issues_df.to_string(index=False))
else:
    print("No issues found.")
