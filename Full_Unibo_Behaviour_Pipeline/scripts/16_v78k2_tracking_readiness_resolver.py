from pathlib import Path
from datetime import datetime
import csv
import json
import hashlib
import zipfile
import pandas as pd

ROOT = Path.home() / "PigBench"
FULL = ROOT / "Full_Unibo_Behaviour_Pipeline"

K1_PKG = FULL / "outputs" / "v78k1_date_specific_excel_pen_alias_resolver" / "Full_Unibo_Date_Specific_Excel_Pen_Alias_Resolver"
K1_TARGETS = K1_PKG / "v78k1_v78j_targets_resolved_with_aliases.csv"
K1_ALIAS = K1_PKG / "v78k1_big_small_alias_table.csv"
K1_CANONICAL = K1_PKG / "v78k1_date_specific_canonical_tlc_room_pen_pattern.csv"

V78I_PKG = FULL / "outputs" / "v78i_visual_stream_grouping_audit" / "Full_Unibo_Visual_Stream_Grouping_Audit"
V78I_OBS = V78I_PKG / "v78i_human_visual_observations.csv"

OUT = FULL / "outputs" / "v78k2_tracking_readiness_resolver"
PKG = OUT / "Full_Unibo_Tracking_Readiness_Resolver"
NOTES = FULL / "notes"
REPORTS = FULL / "reports"
PROGRESS = FULL / "progress"

for p in [OUT, PKG, NOTES, REPORTS, PROGRESS]:
    p.mkdir(parents=True, exist_ok=True)

OUT_READINESS = PKG / "v78k2_target_tracking_readiness_table.csv"
OUT_COUNTS = PKG / "v78k2_readiness_counts.csv"
OUT_BLOCKERS = PKG / "v78k2_blocker_summary.csv"
OUT_SAFE_SUBSET = PKG / "v78k2_safe_tracking_subset.csv"
OUT_DIAGNOSTIC_SUBSET = PKG / "v78k2_diagnostic_tracking_subset.csv"
OUT_DECISION = OUT / "v78k2_decision_summary.csv"
OUT_ISSUES = OUT / "v78k2_issues.csv"
OUT_README = PKG / "README_v78k2_Tracking_Readiness_Resolver.md"
OUT_MANIFEST = PKG / "v78k2_manifest.json"
OUT_ZIP = OUT / "Full_Unibo_Tracking_Readiness_Resolver.zip"
OUT_SHA = OUT / "Full_Unibo_Tracking_Readiness_Resolver.sha256"
OUT_NOTE = NOTES / "v78k2_tracking_readiness_resolver_notes.md"
OUT_REPORT = REPORTS / "v78k2_tracking_readiness_resolver_report.md"
OUT_PROGRESS = PROGRESS / "full_unibo_pipeline_progress_log.csv"

def clean(x):
    if pd.isna(x):
        return ""
    s = str(x).strip()
    if s.lower() in {"nan", "none", "null"}:
        return ""
    return s

def read_csv(path):
    if not path.exists():
        return pd.DataFrame()
    df = pd.read_csv(path).fillna("")
    for c in df.columns:
        if df[c].dtype == object:
            df[c] = df[c].map(clean)
    return df

def to_csv(df, path):
    df.to_csv(path, index=False, quoting=csv.QUOTE_ALL, escapechar="\\", lineterminator="\n")

def sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()

issues = []

targets = read_csv(K1_TARGETS)
alias = read_csv(K1_ALIAS)
canonical = read_csv(K1_CANONICAL)
obs = read_csv(V78I_OBS)

if targets.empty:
    issues.append({
        "item": str(K1_TARGETS),
        "issue_type": "hard_missing_or_empty_k1_targets",
        "issue_detail": "v78k1 resolved target table is missing or empty.",
        "severity": "hard",
    })

if obs.empty:
    issues.append({
        "item": str(V78I_OBS),
        "issue_type": "warning_missing_v78i_observations",
        "issue_detail": "v78i human visual observations missing; using conservative readiness defaults.",
        "severity": "warning",
    })

rows = []

# Conservative known mapping from visual evidence:
# TLC1/c0002 anchor is manually confirmed. c0000 is duplicate/same-view and c0100 top-view group.
# No reliable final TLC2-TLC6 -> c-code mapping.
for _, r in targets.iterrows():
    target_id = clean(r.get("target_id"))
    date = clean(r.get("date"))
    tlc = clean(r.get("camera"))
    original_pen = clean(r.get("target_pen_original") or r.get("pen"))
    resolved_pen = clean(r.get("resolved_room_pen") or r.get("pen"))
    excel_found = str(clean(r.get("excel_pattern_found"))).lower() == "true"
    match_type = clean(r.get("match_type"))
    unresolved_windows = clean(r.get("unresolved_windows"))
    candidate_codes = clean(r.get("candidate_video_camera_codes"))

    camera_pen_status = "READY_CAMERA_PEN_ASSOCIATION" if excel_found else "BLOCKED_CAMERA_PEN_UNRESOLVED"

    if date == "2021-07-22" and tlc == "TLC1":
        video_mapping_status = "PARTIAL_READY_TLC1_VISUAL_ANCHOR"
        primary_code = "c0002"
        secondary_code = "c0000"
        top_code = "c0100"
        visual_group = "GROUP_A"
        video_evidence = "TLC1/c0002 manually confirmed visual anchor; c0000 duplicate/same-view and c0100 top-view group from v78i."
    else:
        video_mapping_status = "BLOCKED_NEEDS_TLC_TO_ENCODED_VIDEO_MAPPING"
        primary_code = ""
        secondary_code = ""
        top_code = ""
        visual_group = ""
        video_evidence = "No trusted TLC-to-encoded-c-code mapping for this TLC camera yet."

    spatial_roi_status = "BLOCKED_NEEDS_FRAME_LEVEL_PEN_ROI"
    spatial_roi_evidence = "Excel resolves Camera/Room/Pen association but does not provide frame-level ROI coordinates."

    if not excel_found:
        readiness = "NOT_READY_CAMERA_PEN_UNRESOLVED"
        blocker = "Excel/date-specific camera-pen association unresolved."
        next_action = "Fix Excel target resolution before any tracking."
    elif video_mapping_status.startswith("BLOCKED"):
        readiness = "NOT_READY_NEEDS_VIDEO_MAPPING_AND_SPATIAL_ROI"
        blocker = "Camera/pen association is ready, but encoded video mapping and pen-level spatial ROI are not ready."
        next_action = "Resolve TLC-to-c-code video mapping, then define/validate pen ROI if pen-level filtering is required."
    elif spatial_roi_status.startswith("BLOCKED"):
        readiness = "DIAGNOSTIC_ONLY_CAMERA_LEVEL_NOT_PEN_LEVEL"
        blocker = "Video mapping is partially ready only for TLC1, but pen-level spatial ROI is not available."
        next_action = "Use only for diagnostic camera-level inspection, not final pen-level tracking labels."
    else:
        readiness = "READY_FOR_TRACKING"
        blocker = ""
        next_action = "Proceed to tracking preparation."

    rows.append({
        "target_id": target_id,
        "date": date,
        "tlc_camera": tlc,
        "target_pen_original": original_pen,
        "resolved_room_pen": resolved_pen,
        "match_type": match_type,
        "unresolved_windows": unresolved_windows,
        "candidate_video_camera_codes": candidate_codes,
        "camera_pen_status": camera_pen_status,
        "video_mapping_status": video_mapping_status,
        "visual_group": visual_group,
        "primary_video_camera_code": primary_code,
        "secondary_video_camera_code": secondary_code,
        "top_view_camera_code": top_code,
        "video_mapping_evidence": video_evidence,
        "spatial_roi_status": spatial_roi_status,
        "spatial_roi_evidence": spatial_roi_evidence,
        "tracking_readiness_status": readiness,
        "blocking_reason": blocker,
        "recommended_next_action": next_action,
    })

readiness = pd.DataFrame(rows)
to_csv(readiness, OUT_READINESS)

if readiness.empty:
    counts = pd.DataFrame()
    blockers = pd.DataFrame()
    safe_subset = pd.DataFrame()
    diagnostic_subset = pd.DataFrame()
else:
    counts = (
        readiness.groupby("tracking_readiness_status")
        .size()
        .reset_index(name="target_count")
        .sort_values("tracking_readiness_status")
    )
    to_csv(counts, OUT_COUNTS)

    blockers = (
        readiness.groupby(["blocking_reason", "recommended_next_action"])
        .size()
        .reset_index(name="target_count")
        .sort_values("target_count", ascending=False)
    )
    to_csv(blockers, OUT_BLOCKERS)

    safe_subset = readiness[readiness["tracking_readiness_status"] == "READY_FOR_TRACKING"].copy()
    diagnostic_subset = readiness[readiness["tracking_readiness_status"] == "DIAGNOSTIC_ONLY_CAMERA_LEVEL_NOT_PEN_LEVEL"].copy()

    to_csv(safe_subset, OUT_SAFE_SUBSET)
    to_csv(diagnostic_subset, OUT_DIAGNOSTIC_SUBSET)

hard_count = int(sum(1 for x in issues if x["severity"] == "hard"))
warning_count = int(sum(1 for x in issues if x["severity"] == "warning"))

if not readiness.empty:
    all_camera_pen_ready = bool((readiness["camera_pen_status"] == "READY_CAMERA_PEN_ASSOCIATION").all())
    ready_for_tracking_count = int((readiness["tracking_readiness_status"] == "READY_FOR_TRACKING").sum())
    diagnostic_count = int((readiness["tracking_readiness_status"] == "DIAGNOSTIC_ONLY_CAMERA_LEVEL_NOT_PEN_LEVEL").sum())
    blocked_count = int(len(readiness) - ready_for_tracking_count - diagnostic_count)
else:
    all_camera_pen_ready = False
    ready_for_tracking_count = 0
    diagnostic_count = 0
    blocked_count = 0

if not all_camera_pen_ready:
    issues.append({
        "item": "camera_pen_association",
        "issue_type": "hard_not_all_camera_pen_associations_ready",
        "issue_detail": "At least one target does not have ready camera/pen association.",
        "severity": "hard",
    })

issues.append({
    "item": "tracking_scope",
    "issue_type": "info_readiness_only",
    "issue_detail": "v78k2 only classifies readiness. It does not run detector/tracker and does not create final behaviour tracking labels.",
    "severity": "info",
})

issues.append({
    "item": "full_tracking",
    "issue_type": "info_full_tracking_not_ready",
    "issue_detail": "Full pen-level tracking is not ready because TLC-to-c-code mapping and/or frame-level pen ROI are unresolved for targets.",
    "severity": "info",
})

issues_df = pd.DataFrame(issues, columns=["item", "issue_type", "issue_detail", "severity"])
to_csv(issues_df, OUT_ISSUES)

hard_count = int((issues_df["severity"] == "hard").sum()) if len(issues_df) else 0
warning_count = int((issues_df["severity"] == "warning").sum()) if len(issues_df) else 0
info_count = int((issues_df["severity"] == "info").sum()) if len(issues_df) else 0

decision_label = "tracking_readiness_classified_not_ready_for_full_tracking"
if hard_count > 0:
    decision_label = "tracking_readiness_classification_has_blocking_input_issues"
elif ready_for_tracking_count > 0:
    decision_label = "tracking_readiness_classified_with_safe_subset"

readme = """# v78k2 Tracking Readiness Resolver

This package classifies each target by readiness for tracking.

It separates:

1. Camera/pen association readiness
2. TLC-to-encoded-video mapping readiness
3. Frame-level pen ROI readiness
4. Final tracking readiness

Important claim boundary:
This stage does not run tracking, does not create final behaviour labels, and does not claim full tracking readiness.
"""
OUT_README.write_text(readme, encoding="utf-8")
OUT_REPORT.write_text(readme, encoding="utf-8")

manifest = {
    "version": "v78k2_tracking_readiness_resolver",
    "generated_at": datetime.now().isoformat(timespec="seconds"),
    "target_rows": int(len(readiness)),
    "ready_for_tracking_count": ready_for_tracking_count,
    "diagnostic_only_count": diagnostic_count,
    "blocked_count": blocked_count,
    "all_camera_pen_ready": all_camera_pen_ready,
    "hard_issue_count": hard_count,
    "warning_count": warning_count,
    "claim_boundary": "readiness classification only; no tracking run",
}
OUT_MANIFEST.write_text(json.dumps(manifest, indent=2), encoding="utf-8")

if OUT_ZIP.exists():
    OUT_ZIP.unlink()

with zipfile.ZipFile(OUT_ZIP, "w", compression=zipfile.ZIP_DEFLATED) as z:
    for p in sorted(PKG.rglob("*")):
        if p.is_file():
            z.write(p, p.relative_to(OUT))

zip_hash = sha256_file(OUT_ZIP)
OUT_SHA.write_text(f"{zip_hash}  {OUT_ZIP.name}\n")

decision = pd.DataFrame([{
    "v78k2_decision": decision_label,
    "target_rows": int(len(readiness)),
    "all_camera_pen_associations_ready": all_camera_pen_ready,
    "ready_for_tracking_count": ready_for_tracking_count,
    "diagnostic_only_camera_level_count": diagnostic_count,
    "blocked_count": blocked_count,
    "safe_tracking_subset_rows": int(len(safe_subset)) if 'safe_subset' in locals() else 0,
    "diagnostic_subset_rows": int(len(diagnostic_subset)) if 'diagnostic_subset' in locals() else 0,
    "hard_issue_count": hard_count,
    "warning_count": warning_count,
    "info_count": info_count,
    "ready_for_v79_full_tracking_preparation": bool(ready_for_tracking_count > 0 and hard_count == 0),
    "ready_for_full_pen_level_tracking": False,
    "zip_path": str(OUT_ZIP),
    "zip_sha256": zip_hash,
    "claim_scope": "tracking_readiness_classification_only",
    "generated_at": datetime.now().isoformat(timespec="seconds"),
}])
to_csv(decision, OUT_DECISION)

OUT_NOTE.write_text(
    "# v78k2 Tracking Readiness Resolver\n\n"
    f"- Decision: {decision.iloc[0]['v78k2_decision']}\n"
    f"- Target rows: {len(readiness)}\n"
    f"- All camera/pen associations ready: {all_camera_pen_ready}\n"
    f"- Ready for tracking count: {ready_for_tracking_count}\n"
    f"- Diagnostic-only camera-level count: {diagnostic_count}\n"
    f"- Blocked count: {blocked_count}\n"
    f"- Ready for v79 full tracking preparation: {bool(ready_for_tracking_count > 0 and hard_count == 0)}\n"
    f"- Ready for full pen-level tracking: False\n\n"
    "Camera/pen association is solved by Excel pattern, but full pen-level tracking remains blocked by video mapping and/or spatial ROI constraints.\n",
    encoding="utf-8"
)

progress_row = pd.DataFrame([{
    "date": datetime.now().date().isoformat(),
    "stage": "v78k2",
    "task_name": "Tracking readiness resolver",
    "status": "CLASSIFIED_NOT_FULL_READY" if ready_for_tracking_count == 0 else "CLASSIFIED_WITH_SAFE_SUBSET",
    "input_summary": "v78k1 resolved targets + v78i visual observations",
    "output_summary": str(PKG),
    "hard_issues": hard_count,
    "warnings": warning_count,
    "next_action": "Resolve TLC-to-c-code video mapping and frame-level pen ROI before full pen-level tracking.",
}])

if OUT_PROGRESS.exists():
    old = pd.read_csv(OUT_PROGRESS)
    progress = pd.concat([old, progress_row], ignore_index=True)
else:
    progress = progress_row
to_csv(progress, OUT_PROGRESS)

print("=== v78k2 decision ===")
print(decision.to_string(index=False))

print("\n=== readiness counts ===")
print(counts.to_string(index=False) if not counts.empty else "none")

print("\n=== readiness table ===")
print(readiness.to_string(index=False) if not readiness.empty else "none")

print("\n=== blocker summary ===")
print(blockers.to_string(index=False) if not blockers.empty else "none")

print("\n=== issues ===")
print(issues_df.to_string(index=False))
