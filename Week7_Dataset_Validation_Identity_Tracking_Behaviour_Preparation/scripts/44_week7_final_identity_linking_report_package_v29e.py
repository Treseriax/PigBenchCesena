from pathlib import Path
from datetime import datetime
import csv
import shutil
import hashlib
import zipfile
import pandas as pd


ROOT = Path.home() / "PigBench"
W7 = ROOT / "Week7_Dataset_Validation_Identity_Tracking_Behaviour_Preparation"

OUT_ROOT = W7 / "outputs" / "final_identity_linking_report_package_v29e"
PKG = OUT_ROOT / "Week7_Final_Identity_Linking_Report_Package_v29e"

REPORTS = PKG / "reports"
TABLES = PKG / "tables"
VISUALS = PKG / "visual_review_contact_sheets"
NOTES = PKG / "notes"

for p in [OUT_ROOT, PKG, REPORTS, TABLES, VISUALS, NOTES]:
    p.mkdir(parents=True, exist_ok=True)

ZIP_PATH = OUT_ROOT / "Week7_Final_Identity_Linking_Report_Package_v29e.zip"

OUT_DECISION = OUT_ROOT / "week7_final_identity_linking_report_package_v29e_decision_summary.csv"
OUT_MANIFEST = OUT_ROOT / "week7_final_identity_linking_report_package_v29e_manifest.csv"
OUT_ISSUES = OUT_ROOT / "week7_final_identity_linking_report_package_v29e_issues.csv"
OUT_NOTE = W7 / "notes" / "week7_final_identity_linking_report_package_v29e_notes.md"

OUT_REPORT = PKG / "Week7_Final_Identity_Linking_Report_v29e.md"
OUT_README = PKG / "README_Week7_Final_Identity_Linking_Report_Package_v29e.md"

INPUTS = {
    "v29a_recommendation": W7 / "outputs" / "tracker_strategy_decision_v29a" / "week7_tracker_strategy_decision_v29a_recommendation.csv",
    "v29a_report": W7 / "outputs" / "tracker_strategy_decision_v29a" / "week7_tracker_strategy_decision_v29a_report.md",

    "v29b_decision": W7 / "outputs" / "colour_constrained_tracklet_linking_v29b" / "week7_colour_constrained_tracklet_linking_v29b_decision_summary.csv",
    "v29b_clip_summary": W7 / "outputs" / "colour_constrained_tracklet_linking_v29b" / "week7_colour_constrained_tracklet_linking_v29b_clip_summary.csv",
    "v29b_tracklet_candidates": W7 / "outputs" / "colour_constrained_tracklet_linking_v29b" / "week7_colour_constrained_tracklet_linking_v29b_tracklet_identity_candidates.csv",
    "v29b_linked_tracks": W7 / "outputs" / "colour_constrained_tracklet_linking_v29b" / "week7_colour_constrained_tracklet_linking_v29b_linked_track_detections.csv",

    "v29c_decision": W7 / "outputs" / "full_tracklet_colour_evidence_v29c" / "week7_full_tracklet_colour_evidence_v29c_decision_summary.csv",
    "v29c_clip_summary": W7 / "outputs" / "full_tracklet_colour_evidence_v29c" / "week7_full_tracklet_colour_evidence_v29c_clip_summary.csv",
    "v29c_tracklet_evidence": W7 / "outputs" / "full_tracklet_colour_evidence_v29c" / "week7_full_tracklet_colour_evidence_v29c_tracklet_colour_evidence.csv",
    "v29c_compare_to_v29b": W7 / "outputs" / "full_tracklet_colour_evidence_v29c" / "week7_full_tracklet_colour_evidence_v29c_compare_to_v29b.csv",
    "v29c_merge_candidates": W7 / "outputs" / "full_tracklet_colour_evidence_v29c" / "week7_full_tracklet_colour_evidence_v29c_same_colour_merge_candidates.csv",

    "v29d_decision_fixed": W7 / "outputs" / "conservative_identity_arbitration_v29d" / "week7_conservative_identity_arbitration_v29d_decision_summary_fixed.csv",
    "v29d_report_fixed": W7 / "outputs" / "conservative_identity_arbitration_v29d" / "week7_conservative_identity_arbitration_v29d_report_fixed.md",
    "v29d_tracklet_arbitration": W7 / "outputs" / "conservative_identity_arbitration_v29d" / "week7_conservative_identity_arbitration_v29d_tracklet_arbitration.csv",
    "v29d_clip_summary": W7 / "outputs" / "conservative_identity_arbitration_v29d" / "week7_conservative_identity_arbitration_v29d_clip_summary.csv",
    "v29d_merge_candidates": W7 / "outputs" / "conservative_identity_arbitration_v29d" / "week7_conservative_identity_arbitration_v29d_conservative_merge_candidates.csv",
    "v29d_review_queue": W7 / "outputs" / "conservative_identity_arbitration_v29d" / "week7_conservative_identity_arbitration_v29d_review_queue.csv",
    "v29d_issues": W7 / "outputs" / "conservative_identity_arbitration_v29d" / "week7_conservative_identity_arbitration_v29d_issues.csv",
}

V29B_CONTACTS = W7 / "outputs" / "colour_constrained_tracklet_linking_v29b" / "contact_sheets"
V29C_CONTACTS = W7 / "outputs" / "full_tracklet_colour_evidence_v29c" / "contact_sheets"


def safe_to_csv(df, path):
    df.to_csv(path, index=False, quoting=csv.QUOTE_ALL, escapechar="\\", lineterminator="\n")


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


def read_df(path):
    path = Path(path)
    if not path.exists():
        return pd.DataFrame()
    try:
        return pd.read_csv(path)
    except Exception:
        return pd.DataFrame()


def copy_file(src, dst_dir, new_name=None):
    src = Path(src)
    if not src.exists():
        return None
    dst_dir.mkdir(parents=True, exist_ok=True)
    dst = dst_dir / (new_name or src.name)
    shutil.copy2(src, dst)
    return dst


def sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def md_table(df, columns):
    if df is None or len(df) == 0:
        return "_No rows._"
    cols = [c for c in columns if c in df.columns]
    if not cols:
        return "_No matching columns._"

    rows = []
    rows.append("| " + " | ".join(cols) + " |")
    rows.append("| " + " | ".join(["---"] * len(cols)) + " |")

    for _, r in df[cols].iterrows():
        vals = []
        for c in cols:
            val = str(r.get(c, ""))
            val = val.replace("\n", " ").replace("|", "/")
            vals.append(val)
        rows.append("| " + " | ".join(vals) + " |")

    return "\n".join(rows)


issues = []
manifest_rows = []

for key, path in INPUTS.items():
    if not path.exists():
        issues.append({"item": key, "issue_type": "missing_input", "issue_detail": str(path)})

v29d = read_first(INPUTS["v29d_decision_fixed"])
v29d_clip = read_df(INPUTS["v29d_clip_summary"])
v29d_review = read_df(INPUTS["v29d_review_queue"])
v29d_merge = read_df(INPUTS["v29d_merge_candidates"])

total_tracklets = int(v29d.get("total_tracklets", 0) or 0)
accepted = int(v29d.get("conservative_accepted_tracklets", 0) or 0)
confirmed = int(v29d.get("accepted_confirmed_by_v29c", 0) or 0)
v29b_only = int(v29d.get("accepted_from_v29b_only", 0) or 0)
conflict = int(v29d.get("accepted_keep_v29b_conflict_with_v29c", 0) or 0)
recovered = int(v29d.get("recovered_candidates_from_v29c", 0) or 0)
low = int(v29d.get("low_confidence_v29c_candidates", 0) or 0)
unknown = int(v29d.get("unknown_no_reliable_identity", 0) or 0)
review_count = int(v29d.get("review_queue_tracklets", 0) or 0)
merge_count = int(v29d.get("conservative_merge_candidates", 0) or 0)
clips_evaluated = int(v29d.get("clips_evaluated", 0) or 0)
acceptance_rate = accepted / total_tracklets if total_tracklets else 0.0

for key, path in INPUTS.items():
    if not path.exists():
        continue

    if path.suffix.lower() == ".md":
        copied = copy_file(path, REPORTS, f"{key}__{path.name}")
        section = "reports"
    else:
        copied = copy_file(path, TABLES, f"{key}__{path.name}")
        section = "tables"

    if copied:
        manifest_rows.append({
            "package_section": section,
            "item_key": key,
            "source_path": str(path),
            "package_path": str(copied),
            "size_bytes": copied.stat().st_size,
        })

note_files = sorted((W7 / "notes").glob("week7_*v29*_notes.md")) + sorted((W7 / "notes").glob("week7_*v29d_summary_fix_notes.md"))

for nf in note_files:
    copied = copy_file(nf, NOTES)
    if copied:
        manifest_rows.append({
            "package_section": "notes",
            "item_key": nf.stem,
            "source_path": str(nf),
            "package_path": str(copied),
            "size_bytes": copied.stat().st_size,
        })

if V29B_CONTACTS.exists():
    dst = VISUALS / "v29b_identity_linking_contact_sheets"
    for p in sorted(V29B_CONTACTS.glob("*.jpg")):
        copied = copy_file(p, dst)
        if copied:
            manifest_rows.append({
                "package_section": "visuals",
                "item_key": "v29b_identity_linking_contact_sheet",
                "source_path": str(p),
                "package_path": str(copied),
                "size_bytes": copied.stat().st_size,
            })

selected_v29c_names = set()

if len(v29d_review):
    for _, r in v29d_review.iterrows():
        scan = str(r.get("scan_frame_id", ""))
        tid = r.get("dense_track_id", "")
        try:
            tid_int = int(float(tid))
        except Exception:
            continue
        selected_v29c_names.add(f"{scan}_dense_polygon__T{tid_int:03d}_colour_evidence_sheet.jpg")

if len(v29d_merge):
    for _, r in v29d_merge.iterrows():
        scan = str(r.get("scan_frame_id", ""))
        tids = str(r.get("candidate_track_ids", "")).replace("|", " ").split()
        for t in tids:
            try:
                tid_int = int(float(t))
            except Exception:
                continue
            selected_v29c_names.add(f"{scan}_dense_polygon__T{tid_int:03d}_colour_evidence_sheet.jpg")

if V29C_CONTACTS.exists():
    dst = VISUALS / "v29c_selected_review_colour_evidence_contact_sheets"
    for name in sorted(selected_v29c_names):
        p = V29C_CONTACTS / name
        if p.exists():
            copied = copy_file(p, dst)
            if copied:
                manifest_rows.append({
                    "package_section": "visuals",
                    "item_key": "v29c_selected_review_colour_evidence_contact_sheet",
                    "source_path": str(p),
                    "package_path": str(copied),
                    "size_bytes": copied.stat().st_size,
                })

clip_table = md_table(
    v29d_clip,
    [
        "scan_frame_id",
        "raw_tracklets",
        "conservative_accepted_tracklets",
        "accepted_confirmed_by_v29c",
        "accepted_keep_v29b_conflict_with_v29c",
        "recovered_candidates_from_v29c",
        "low_confidence_v29c_candidates",
        "unknown_no_reliable_identity",
        "conservative_colours",
        "clip_arbitration_status",
    ],
)

merge_table = md_table(
    v29d_merge,
    [
        "scan_frame_id",
        "accepted_visual_colour",
        "accepted_behaviour_pig_id",
        "candidate_track_ids",
        "candidate_tracklet_count",
        "review_required",
        "merge_status",
    ],
)

report = f"""# Week 7 Final Identity-Linking Report v29e

## Executive summary

This package summarizes the Week 7 identity-linking work from tracker strategy decision to conservative identity arbitration.

The final conclusion is:

**Simple IoU track IDs are not final pig identities.**

The most reliable strategy is a conservative identity layer:

1. use **v29b center-frame GT-overlap identity linking** as the primary identity anchor;
2. use **v29c full-tracklet HSV colour evidence** only as secondary support;
3. never allow v29c to overwrite v29b automatically;
4. keep conflicts, recovered candidates, low-confidence candidates and unknowns in a review queue.

## Key result

- Total tracklets evaluated: `{total_tracklets}`
- Conservative accepted tracklets: `{accepted}`
- Conservative acceptance rate: `{acceptance_rate:.2%}`
- Accepted and confirmed by v29c: `{confirmed}`
- Accepted from v29b only: `{v29b_only}`
- Accepted from v29b despite v29c conflict: `{conflict}`
- Recovered candidates from v29c: `{recovered}`
- Low-confidence v29c candidates: `{low}`
- Unknown / no reliable identity: `{unknown}`
- Review queue tracklets: `{review_count}`
- Conservative merge candidates: `{merge_count}`

## Why this conservative design was necessary

Earlier dry-runs showed that detector and polygon ROI filtering are usable, but simple IoU tracking remains fragmented in crowded or occluded scenes.

v29b demonstrated that tracklets can be linked to visual marker colours through center-frame GT overlap. v29c tested full-tracklet HSV colour evidence, but it generated many conflicts with v29b. Therefore, v29c is useful as supporting evidence but not reliable enough to overwrite v29b.

## Final identity rule

| Evidence case | Decision |
|---|---|
| v29b valid and v29c agrees | accept identity, high confidence |
| v29b valid and v29c missing | accept v29b identity |
| v29b valid and v29c conflicts | keep v29b, flag for review |
| v29b missing and v29c high/medium | candidate only, review required |
| v29c low only | do not use as final identity |
| no reliable evidence | unknown |

## Clip-level summary

{clip_table}

## Conservative merge candidates

These are not automatically merged final identities. They are candidates where multiple accepted tracklets share the same conservative colour / behaviour identity in the same clip.

{merge_table}

## Final pipeline

YOLOv8-s detector
→ manual target-pen polygon ROI
→ dense simple-IoU tracklets as baseline fragments
→ v29b GT-overlap colour/behaviour identity anchor
→ v29c HSV full-tracklet support evidence
→ v29d conservative arbitration
→ v29e final report/package

## Limitations

1. This is not production-grade final MOT tracking.
2. Simple IoU track IDs remain fragmented in crowded scenes.
3. v29c HSV evidence can be affected by lighting, blur, occlusion and background colours.
4. v29b is only anchored around scanpoint GT overlap, so some tracklets remain candidates or unknown.
5. Recovered candidates and conflicts require manual/visual review before final dataset release.
6. Unknown identities are intentionally preserved rather than forced.

## Recommended next engineering steps

1. Review v29d review queue visually.
2. Accept only high-confidence conservative identities for downstream behaviour representation.
3. Use recovered candidates only after visual confirmation.
4. For future tracking, evaluate ByteTrack/BoT-SORT if dependency setup becomes stable.
5. For a stronger identity system, combine improved tracker association with marker-colour constrained linking.

## Package contents

- `reports/`: strategy and arbitration reports
- `tables/`: all key CSV tables
- `visual_review_contact_sheets/`: v29b identity sheets and selected v29c review/merge sheets
- `notes/`: generated notes from v29 stages
"""

OUT_REPORT.write_text(report)

readme = f"""# Week 7 Final Identity-Linking Report Package v29e

## Purpose

This package contains the final Week 7 identity-linking report and supporting CSV tables, notes and selected visual review sheets.

## Main decision

v29c does **not** overwrite v29b.

- Primary identity source: `v29b_center_gt_overlap`
- Secondary identity source: `v29c_full_tracklet_hsv_colour_evidence`

## Key files

- `Week7_Final_Identity_Linking_Report_v29e.md`
- `tables/`
- `reports/`
- `visual_review_contact_sheets/`
- `notes/`

## Zip path

`{ZIP_PATH}`
"""

OUT_README.write_text(readme)

issues_df = pd.DataFrame(issues, columns=["item", "issue_type", "issue_detail"])
safe_to_csv(issues_df, OUT_ISSUES)

decision = pd.DataFrame([{
    "v29e_decision": "final_identity_linking_report_package_created",
    "total_tracklets": total_tracklets,
    "conservative_accepted_tracklets": accepted,
    "conservative_acceptance_rate": round(acceptance_rate, 4),
    "accepted_confirmed_by_v29c": confirmed,
    "accepted_from_v29b_only": v29b_only,
    "accepted_keep_v29b_conflict_with_v29c": conflict,
    "recovered_candidates_from_v29c": recovered,
    "low_confidence_v29c_candidates": low,
    "unknown_no_reliable_identity": unknown,
    "review_queue_tracklets": review_count,
    "conservative_merge_candidates": merge_count,
    "clips_evaluated": clips_evaluated,
    "primary_identity_source": "v29b_center_gt_overlap",
    "secondary_identity_source": "v29c_full_tracklet_hsv_colour_evidence",
    "v29c_overwrites_v29b": False,
    "final_identity_status": "conservative_identity_candidates_not_final_production_tracking",
    "package_dir": str(PKG),
    "zip_path": str(ZIP_PATH),
    "issue_count": len(issues_df),
    "generated_at": datetime.now().isoformat(timespec="seconds"),
}])

safe_to_csv(decision, OUT_DECISION)

for key, src, dst_dir in [
    ("v29e_main_report", OUT_REPORT, REPORTS),
    ("v29e_readme", OUT_README, REPORTS),
    ("v29e_decision", OUT_DECISION, TABLES),
    ("v29e_issues", OUT_ISSUES, TABLES),
]:
    copied = copy_file(src, dst_dir, src.name)
    if copied:
        manifest_rows.append({
            "package_section": "reports" if dst_dir == REPORTS else "tables",
            "item_key": key,
            "source_path": str(src),
            "package_path": str(copied),
            "size_bytes": copied.stat().st_size,
        })

manifest = pd.DataFrame(manifest_rows)
safe_to_csv(manifest, OUT_MANIFEST)

copied_manifest = copy_file(OUT_MANIFEST, TABLES, OUT_MANIFEST.name)
if copied_manifest:
    manifest_rows.append({
        "package_section": "tables",
        "item_key": "v29e_manifest",
        "source_path": str(OUT_MANIFEST),
        "package_path": str(copied_manifest),
        "size_bytes": copied_manifest.stat().st_size,
    })

manifest = pd.DataFrame(manifest_rows)
safe_to_csv(manifest, OUT_MANIFEST)
safe_to_csv(manifest, copied_manifest)

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
    "# Week 7 Final Identity-Linking Report Package v29e\n\n"
    "## Summary\n\n"
    f"- Total tracklets: `{total_tracklets}`\n"
    f"- Conservative accepted tracklets: `{accepted}`\n"
    f"- Acceptance rate: `{acceptance_rate:.2%}`\n"
    f"- Accepted confirmed by v29c: `{confirmed}`\n"
    f"- Accepted from v29b only: `{v29b_only}`\n"
    f"- Accepted keep v29b despite v29c conflict: `{conflict}`\n"
    f"- Recovered candidates from v29c: `{recovered}`\n"
    f"- Low-confidence v29c candidates: `{low}`\n"
    f"- Unknown identities: `{unknown}`\n"
    f"- Review queue tracklets: `{review_count}`\n"
    f"- Conservative merge candidates: `{merge_count}`\n"
    f"- Issue count: `{len(issues_df)}`\n"
    f"- Package dir: `{PKG}`\n"
    f"- Zip path: `{ZIP_PATH}`\n"
    f"- Zip size bytes: `{zip_size}`\n"
    f"- Zip SHA256: `{zip_sha}`\n\n"
    "## Final decision\n\n"
    "`v29c_overwrites_v29b = False`; v29b remains the primary identity anchor and v29c remains secondary support evidence.\n"
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
print("=== v29e decision ===")
print(decision.to_string(index=False))
print()
print("=== package file count ===")
print(sum(1 for x in PKG.rglob("*") if x.is_file()))
print()
print("=== issues ===")
if len(issues_df):
    print(issues_df.to_string(index=False))
else:
    print("No issues found.")
