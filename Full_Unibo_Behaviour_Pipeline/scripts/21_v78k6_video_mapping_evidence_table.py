from pathlib import Path
from datetime import datetime
import re
import csv
import json
import hashlib
import zipfile
import pandas as pd

ROOT = Path.home() / "PigBench"
FULL = ROOT / "Full_Unibo_Behaviour_Pipeline"
RAW_UNIBO = Path("/work/pig/datasets/Unibo")

K5B_PKG = FULL / "outputs" / "v78k5b_corrected_full_frozen_annotation_truth_table" / "Full_Unibo_Corrected_Full_Frozen_Annotation_Truth_Table"
FROZEN = K5B_PKG / "v78k5b_corrected_full_frozen_annotation_truth_table.csv"

K3_PKG = FULL / "outputs" / "v78k3_tlc_to_c_code_mapping_resolver" / "Full_Unibo_TLC_to_CCode_Mapping_Resolver"
K3_MATRIX = K3_PKG / "v78k3_tlc_to_c_code_mapping_matrix.csv"

OUT = FULL / "outputs" / "v78k6_video_mapping_evidence_table"
PKG = OUT / "Full_Unibo_Video_Mapping_Evidence_Table"
NOTES = FULL / "notes"
REPORTS = FULL / "reports"
PROGRESS = FULL / "progress"

for p in [OUT, PKG, NOTES, REPORTS, PROGRESS]:
    p.mkdir(parents=True, exist_ok=True)

OUT_VIDEO_INV = PKG / "v78k6_video_inventory.csv"
OUT_HOURLY_AVAIL = PKG / "v78k6_hourly_video_availability.csv"
OUT_ANN_HOUR = PKG / "v78k6_annotation_hour_target_summary.csv"
OUT_CANDIDATE_MATRIX = PKG / "v78k6_annotation_video_candidate_matrix.csv"
OUT_TLC_EVIDENCE = PKG / "v78k6_tlc_to_c_code_evidence_matrix.csv"
OUT_GAPS = PKG / "v78k6_video_mapping_gaps.csv"
OUT_DECISION = OUT / "v78k6_decision_summary.csv"
OUT_ISSUES = OUT / "v78k6_issues.csv"
OUT_README = PKG / "README_v78k6_Video_Mapping_Evidence_Table.md"
OUT_MANIFEST = PKG / "v78k6_manifest.json"
OUT_ZIP = OUT / "Full_Unibo_Video_Mapping_Evidence_Table.zip"
OUT_SHA = OUT / "Full_Unibo_Video_Mapping_Evidence_Table.sha256"
OUT_NOTE = NOTES / "v78k6_video_mapping_evidence_table_notes.md"
OUT_REPORT = REPORTS / "v78k6_video_mapping_evidence_table_report.md"
OUT_PROGRESS = PROGRESS / "full_unibo_pipeline_progress_log.csv"

VIDEO_EXTS = {".mp4", ".avi", ".mov", ".mkv", ".m4v"}
ENCODED_RE = re.compile(r"(c\d{4})(\d{2})(\d{2})(\d{2})(\d{2})(\d{2})(\d{2})", re.I)

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

def parse_time_to_hour(x):
    s = clean(x)
    if not s:
        return ""
    # Handles 07:00:00, 07:00, 2021-07-22 07:00:00, 7.00 etc.
    m = re.search(r"\b([01]?\d|2[0-3])[:.](\d{2})", s)
    if m:
        return f"{int(m.group(1)):02d}:00"
    m2 = re.search(r"\b([01]?\d|2[0-3])00\b", s)
    if m2:
        return f"{int(m2.group(1)):02d}:00"
    return ""

def parse_date_from_name(name):
    # 2021-07-22 / 2021_07_22
    m = re.search(r"(20\d{2})[-_](\d{1,2})[-_](\d{1,2})", name)
    if m:
        y, mo, d = m.groups()
        return f"{int(y):04d}-{int(mo):02d}-{int(d):02d}"

    # 22_7_2021 / 22-07-2021
    m = re.search(r"\b(\d{1,2})[-_](\d{1,2})[-_](20\d{2})\b", name)
    if m:
        d, mo, y = m.groups()
        return f"{int(y):04d}-{int(mo):02d}-{int(d):02d}"

    return ""

def parse_video_path(path):
    name = path.name
    lower = name.lower()

    video_type = "unknown"
    camera_code = ""
    tlc_camera = ""
    room_pen = ""
    date = ""
    start_time = ""
    start_hour = ""

    # Encoded filename example: c0002210722070000.mp4
    m = ENCODED_RE.search(name)
    if m:
        video_type = "encoded_c_code"
        camera_code = m.group(1).lower()
        yy, mo, dd, hh, mm, ss = m.group(2), m.group(3), m.group(4), m.group(5), m.group(6), m.group(7)
        date = f"20{int(yy):02d}-{int(mo):02d}-{int(dd):02d}"
        start_time = f"{int(hh):02d}:{int(mm):02d}:{int(ss):02d}"
        start_hour = f"{int(hh):02d}:00"

    # Friendly TLC naming.
    mt = re.search(r"tlc[\s_ -]*([1-6])", name, re.I)
    if mt:
        tlc_camera = f"TLC{mt.group(1)}"
        if video_type == "unknown":
            video_type = "friendly_tlc"

    mp = re.search(r"\b([BCM])[\s_ -]*([0-9]+)\b", name, re.I)
    if mp:
        room_pen = f"{mp.group(1).upper()}{int(mp.group(2))}"

    if not date:
        date = parse_date_from_name(name)

    if not start_hour:
        # hour in filename, e.g., 07-00 or 070000
        mh = re.search(r"\b([01]?\d|2[0-3])[:._-]?([0-5]\d)[:._-]?([0-5]\d)?\b", name)
        if mh:
            hh = int(mh.group(1))
            mm = int(mh.group(2))
            ss = int(mh.group(3) or 0)
            start_time = f"{hh:02d}:{mm:02d}:{ss:02d}"
            start_hour = f"{hh:02d}:00"

    return {
        "video_path": str(path),
        "video_filename": name,
        "video_parent": str(path.parent),
        "video_type": video_type,
        "date": date,
        "start_time": start_time,
        "start_hour": start_hour,
        "camera_code": camera_code,
        "tlc_camera_from_filename": tlc_camera,
        "room_pen_from_filename": room_pen,
        "file_size_bytes": path.stat().st_size if path.exists() else 0,
    }

issues = []

frozen = read_csv(FROZEN)
k3 = read_csv(K3_MATRIX)

if frozen.empty:
    issues.append({
        "item": str(FROZEN),
        "issue_type": "hard_missing_frozen_annotation_truth",
        "issue_detail": "v78k5b frozen annotation truth table is missing or empty.",
        "severity": "hard",
    })

if not RAW_UNIBO.exists():
    issues.append({
        "item": str(RAW_UNIBO),
        "issue_type": "hard_missing_raw_unibo_directory",
        "issue_detail": "Raw Unibo directory does not exist.",
        "severity": "hard",
    })

# 1) Video inventory
video_rows = []
if RAW_UNIBO.exists():
    for p in sorted(RAW_UNIBO.rglob("*")):
        if p.is_file() and p.suffix.lower() in VIDEO_EXTS:
            video_rows.append(parse_video_path(p))

video_inv = pd.DataFrame(video_rows)
to_csv(video_inv, OUT_VIDEO_INV)

if video_inv.empty:
    issues.append({
        "item": "video_inventory",
        "issue_type": "hard_no_videos_found",
        "issue_detail": f"No video files found under {RAW_UNIBO}.",
        "severity": "hard",
    })

# 2) Annotation target/hour summary from frozen truth.
ann = frozen.copy()

required_ann_cols = ["date", "tlc_camera", "resolved_room_pen", "window_start", "window_end", "behaviour_label", "identity_colour"]
for c in required_ann_cols:
    if c not in ann.columns:
        issues.append({
            "item": c,
            "issue_type": "hard_missing_required_frozen_column",
            "issue_detail": f"Missing required frozen annotation column: {c}",
            "severity": "hard",
        })

if not ann.empty and all(c in ann.columns for c in ["date", "tlc_camera", "resolved_room_pen"]):
    ann["annotation_hour"] = ann["window_start"].map(parse_time_to_hour) if "window_start" in ann.columns else ""
    blank_hour = int((ann["annotation_hour"] == "").sum())
    if blank_hour:
        issues.append({
            "item": "annotation_hour",
            "issue_type": "warning_some_annotation_hours_blank",
            "issue_detail": f"{blank_hour} annotation rows have blank/unparsed annotation hour.",
            "severity": "warning",
        })

    ann_hour = (
        ann.groupby(["date", "annotation_hour", "tlc_camera", "resolved_room_pen"], dropna=False)
        .agg(
            annotation_rows=("annotation_window_id", "count"),
            behaviour_labels=("behaviour_label", lambda x: ";".join(sorted(set([clean(v) for v in x if clean(v)])))),
            identity_colours=("identity_colour", lambda x: ";".join(sorted(set([clean(v) for v in x if clean(v)])))),
            window_start_min=("window_start", "min"),
            window_end_max=("window_end", "max"),
        )
        .reset_index()
        .sort_values(["date", "annotation_hour", "tlc_camera", "resolved_room_pen"])
    )
else:
    ann_hour = pd.DataFrame()

to_csv(ann_hour, OUT_ANN_HOUR)

# 3) Hourly video availability.
if not video_inv.empty:
    hourly = (
        video_inv.groupby(["date", "start_hour"], dropna=False)
        .agg(
            video_count=("video_path", "count"),
            encoded_video_count=("video_type", lambda x: int((x == "encoded_c_code").sum())),
            friendly_video_count=("video_type", lambda x: int((x == "friendly_tlc").sum())),
            encoded_camera_codes=("camera_code", lambda x: ";".join(sorted(set([clean(v) for v in x if clean(v)])))),
            friendly_tlc_cameras=("tlc_camera_from_filename", lambda x: ";".join(sorted(set([clean(v) for v in x if clean(v)])))),
            video_filenames=("video_filename", lambda x: ";".join(list(x)[:30])),
        )
        .reset_index()
        .sort_values(["date", "start_hour"])
    )
else:
    hourly = pd.DataFrame()

to_csv(hourly, OUT_HOURLY_AVAIL)

# 4) Candidate matrix: annotation target-hour -> candidate videos by date/hour.
candidate_rows = []

if not ann_hour.empty:
    for _, r in ann_hour.iterrows():
        date = clean(r["date"])
        hour = clean(r["annotation_hour"])
        tlc = clean(r["tlc_camera"])
        pen = clean(r["resolved_room_pen"])

        vids = video_inv[(video_inv["date"] == date) & (video_inv["start_hour"] == hour)].copy() if not video_inv.empty else pd.DataFrame()

        encoded = vids[vids["video_type"] == "encoded_c_code"].copy() if not vids.empty else pd.DataFrame()
        friendly = vids[vids["video_type"] == "friendly_tlc"].copy() if not vids.empty else pd.DataFrame()
        friendly_same_tlc = friendly[friendly["tlc_camera_from_filename"] == tlc].copy() if not friendly.empty else pd.DataFrame()

        encoded_codes = ";".join(sorted(set(encoded["camera_code"].tolist()))) if not encoded.empty else ""
        friendly_tlcs = ";".join(sorted(set(friendly["tlc_camera_from_filename"].tolist()))) if not friendly.empty else ""

        candidate_rows.append({
            "date": date,
            "annotation_hour": hour,
            "tlc_camera": tlc,
            "resolved_room_pen": pen,
            "annotation_rows": int(r["annotation_rows"]),
            "behaviour_labels": clean(r["behaviour_labels"]),
            "identity_colours": clean(r["identity_colours"]),
            "encoded_candidate_count": int(len(encoded)),
            "encoded_candidate_codes": encoded_codes,
            "encoded_candidate_filenames": ";".join(encoded["video_filename"].tolist()[:20]) if not encoded.empty else "",
            "friendly_candidate_count": int(len(friendly)),
            "friendly_tlc_cameras_available": friendly_tlcs,
            "friendly_same_tlc_count": int(len(friendly_same_tlc)),
            "friendly_same_tlc_filenames": ";".join(friendly_same_tlc["video_filename"].tolist()[:20]) if not friendly_same_tlc.empty else "",
            "candidate_status": (
                "HAS_FRIENDLY_SAME_TLC_AND_ENCODED_CANDIDATES" if len(friendly_same_tlc) and len(encoded) else
                "HAS_ENCODED_CANDIDATES_ONLY" if len(encoded) else
                "HAS_FRIENDLY_SAME_TLC_ONLY" if len(friendly_same_tlc) else
                "NO_DATE_HOUR_VIDEO_CANDIDATES"
            ),
        })

candidate_matrix = pd.DataFrame(candidate_rows)
to_csv(candidate_matrix, OUT_CANDIDATE_MATRIX)

# 5) TLC-level evidence matrix.
tlc_rows = []
if not candidate_matrix.empty:
    for (date, tlc), g in candidate_matrix.groupby(["date", "tlc_camera"]):
        encoded_codes = sorted(set(";".join(g["encoded_candidate_codes"].tolist()).split(";")) - {""})
        same_tlc_hours = sorted(set(g[g["friendly_same_tlc_count"] > 0]["annotation_hour"].tolist()))
        encoded_hours = sorted(set(g[g["encoded_candidate_count"] > 0]["annotation_hour"].tolist()))

        # Conservative default.
        locked = False
        selected_group = ""
        primary_code = ""
        evidence_status = "NO_TRUSTED_TLC_TO_CCODE_MAPPING"

        if date == "2021-07-22" and tlc == "TLC1":
            locked = True
            selected_group = "GROUP_A"
            primary_code = "c0002"
            evidence_status = "LOCKED_FROM_PRIOR_TLC1_VISUAL_ANCHOR"

        tlc_rows.append({
            "date": date,
            "tlc_camera": tlc,
            "room_pens": ";".join(sorted(set(g["resolved_room_pen"].tolist()))),
            "annotation_rows": int(g["annotation_rows"].sum()),
            "annotation_hours": ";".join(sorted(set(g["annotation_hour"].tolist()))),
            "encoded_candidate_codes_observed": ";".join(encoded_codes),
            "encoded_candidate_hours": ";".join(encoded_hours),
            "friendly_same_tlc_hours": ";".join(same_tlc_hours),
            "has_friendly_same_tlc_evidence": bool(len(same_tlc_hours)),
            "has_encoded_candidates": bool(len(encoded_codes)),
            "locked_mapping": locked,
            "selected_visual_group_id": selected_group,
            "primary_side_code": primary_code,
            "evidence_status": evidence_status,
            "recommended_action": "Keep TLC1 locked; do not generalize." if locked else "Review candidate videos visually or find external TLC-to-c-code evidence. Do not guess.",
        })

tlc_evidence = pd.DataFrame(tlc_rows)
to_csv(tlc_evidence, OUT_TLC_EVIDENCE)

# 6) Gaps
gap_rows = []
if not candidate_matrix.empty:
    gaps = candidate_matrix[candidate_matrix["candidate_status"] == "NO_DATE_HOUR_VIDEO_CANDIDATES"].copy()
    for _, r in gaps.iterrows():
        gap_rows.append({
            "date": clean(r["date"]),
            "annotation_hour": clean(r["annotation_hour"]),
            "tlc_camera": clean(r["tlc_camera"]),
            "resolved_room_pen": clean(r["resolved_room_pen"]),
            "annotation_rows": int(r["annotation_rows"]),
            "gap_type": "no_video_candidates_for_annotation_date_hour",
        })

    # Rows with encoded videos but no direct same-TLC friendly reference.
    weak = candidate_matrix[
        (candidate_matrix["encoded_candidate_count"] > 0) &
        (candidate_matrix["friendly_same_tlc_count"] == 0)
    ].copy()
    for _, r in weak.iterrows():
        gap_rows.append({
            "date": clean(r["date"]),
            "annotation_hour": clean(r["annotation_hour"]),
            "tlc_camera": clean(r["tlc_camera"]),
            "resolved_room_pen": clean(r["resolved_room_pen"]),
            "annotation_rows": int(r["annotation_rows"]),
            "gap_type": "encoded_candidates_exist_but_no_same_tlc_friendly_reference",
        })

gaps_df = pd.DataFrame(gap_rows)
to_csv(gaps_df, OUT_GAPS)

# QA / issues.
if not frozen.empty and len(frozen) != 15733:
    issues.append({
        "item": "frozen_input_row_count",
        "issue_type": "hard_frozen_input_not_full",
        "issue_detail": f"Expected v78k5b frozen rows 15733, got {len(frozen)}.",
        "severity": "hard",
    })

if not candidate_matrix.empty:
    no_candidate_rows = int(candidate_matrix[candidate_matrix["candidate_status"] == "NO_DATE_HOUR_VIDEO_CANDIDATES"]["annotation_rows"].sum())
    if no_candidate_rows:
        issues.append({
            "item": "video_candidates",
            "issue_type": "warning_some_annotation_rows_have_no_date_hour_video_candidates",
            "issue_detail": f"{no_candidate_rows} annotation rows have no date/hour video candidates.",
            "severity": "warning",
        })

if not tlc_evidence.empty:
    unresolved_tlc = int((tlc_evidence["evidence_status"] == "NO_TRUSTED_TLC_TO_CCODE_MAPPING").sum())
    if unresolved_tlc:
        issues.append({
            "item": "tlc_to_c_code_mapping",
            "issue_type": "info_unresolved_tlc_to_c_code_mappings_remain",
            "issue_detail": f"{unresolved_tlc} date/TLC rows remain without trusted TLC-to-c-code mapping.",
            "severity": "info",
        })

issues.append({
    "item": "scope",
    "issue_type": "info_video_mapping_evidence_only",
    "issue_detail": "v78k6 builds video mapping evidence tables. It does not assign unresolved TLC-to-c-code mappings, does not define ROI, and does not run tracking.",
    "severity": "info",
})

issues_df = pd.DataFrame(issues, columns=["item", "issue_type", "issue_detail", "severity"])
to_csv(issues_df, OUT_ISSUES)

hard_count = int((issues_df["severity"] == "hard").sum()) if len(issues_df) else 0
warning_count = int((issues_df["severity"] == "warning").sum()) if len(issues_df) else 0
info_count = int((issues_df["severity"] == "info").sum()) if len(issues_df) else 0

readme = """# v78k6 Video Mapping Evidence Table

This package links frozen annotation truth to available raw video candidates by date/hour.

It does not make unsafe c-code assignments.

Known locked mapping:
- 2021-07-22 TLC1 -> GROUP_A / c0002 primary from prior visual anchor.

All other TLC-to-c-code mappings require direct evidence or manual review.
"""
OUT_README.write_text(readme, encoding="utf-8")
OUT_REPORT.write_text(readme, encoding="utf-8")

manifest = {
    "version": "v78k6_video_mapping_evidence_table",
    "generated_at": datetime.now().isoformat(timespec="seconds"),
    "frozen_rows": int(len(frozen)),
    "video_inventory_rows": int(len(video_inv)),
    "annotation_hour_rows": int(len(ann_hour)),
    "candidate_matrix_rows": int(len(candidate_matrix)),
    "tlc_evidence_rows": int(len(tlc_evidence)),
    "gap_rows": int(len(gaps_df)),
    "hard_issue_count": hard_count,
    "warning_count": warning_count,
    "claim_boundary": "video mapping evidence only; no unsafe mapping assignment",
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

locked_count = int((tlc_evidence["locked_mapping"] == True).sum()) if not tlc_evidence.empty else 0
unresolved_count = int((tlc_evidence["evidence_status"] == "NO_TRUSTED_TLC_TO_CCODE_MAPPING").sum()) if not tlc_evidence.empty else 0

decision = pd.DataFrame([{
    "v78k6_decision": "video_mapping_evidence_table_created",
    "frozen_annotation_rows": int(len(frozen)),
    "video_inventory_rows": int(len(video_inv)),
    "annotation_hour_target_rows": int(len(ann_hour)),
    "candidate_matrix_rows": int(len(candidate_matrix)),
    "tlc_evidence_rows": int(len(tlc_evidence)),
    "locked_tlc_mapping_rows": locked_count,
    "unresolved_tlc_mapping_rows": unresolved_count,
    "gap_rows": int(len(gaps_df)),
    "zip_path": str(OUT_ZIP),
    "zip_sha256": zip_hash,
    "hard_issue_count": hard_count,
    "warning_count": warning_count,
    "info_count": info_count,
    "ready_for_manual_video_mapping_review": bool(hard_count == 0 and len(candidate_matrix) > 0),
    "ready_for_tracking": False,
    "claim_scope": "video_mapping_evidence_only",
    "generated_at": datetime.now().isoformat(timespec="seconds"),
}])
to_csv(decision, OUT_DECISION)

OUT_NOTE.write_text(
    "# v78k6 Video Mapping Evidence Table\n\n"
    f"- Decision: {decision.iloc[0]['v78k6_decision']}\n"
    f"- Frozen annotation rows: {len(frozen)}\n"
    f"- Video inventory rows: {len(video_inv)}\n"
    f"- Annotation hour target rows: {len(ann_hour)}\n"
    f"- Candidate matrix rows: {len(candidate_matrix)}\n"
    f"- TLC evidence rows: {len(tlc_evidence)}\n"
    f"- Locked TLC mapping rows: {locked_count}\n"
    f"- Unresolved TLC mapping rows: {unresolved_count}\n"
    f"- Hard issues: {hard_count}\n"
    f"- Ready for manual video mapping review: {bool(hard_count == 0 and len(candidate_matrix) > 0)}\n"
    f"- Ready for tracking: False\n\n"
    "This stage creates evidence tables only. It does not assign unresolved TLC-to-c-code mappings.\n",
    encoding="utf-8"
)

progress_row = pd.DataFrame([{
    "date": datetime.now().date().isoformat(),
    "stage": "v78k6",
    "task_name": "Video mapping evidence table",
    "status": "PASS" if hard_count == 0 else "NEEDS_FIX",
    "input_summary": "v78k5b frozen annotation truth + raw Unibo videos",
    "output_summary": str(PKG),
    "hard_issues": hard_count,
    "warnings": warning_count,
    "next_action": "Use evidence matrix to resolve TLC-to-c-code mapping or document unresolved mappings.",
}])

if OUT_PROGRESS.exists():
    old = pd.read_csv(OUT_PROGRESS)
    progress = pd.concat([old, progress_row], ignore_index=True)
else:
    progress = progress_row
to_csv(progress, OUT_PROGRESS)

print("=== v78k6 decision ===")
print(decision.to_string(index=False))

print("\n=== video inventory summary ===")
if not video_inv.empty:
    print(video_inv.groupby(["video_type"]).size().reset_index(name="count").to_string(index=False))
else:
    print("none")

print("\n=== hourly availability sample ===")
print(hourly.head(80).to_string(index=False) if not hourly.empty else "none")

print("\n=== TLC evidence matrix ===")
print(tlc_evidence.to_string(index=False) if not tlc_evidence.empty else "none")

print("\n=== candidate matrix sample ===")
print(candidate_matrix.head(80).to_string(index=False) if not candidate_matrix.empty else "none")

print("\n=== gaps sample ===")
print(gaps_df.head(80).to_string(index=False) if not gaps_df.empty else "none")

print("\n=== issues ===")
print(issues_df.to_string(index=False))
