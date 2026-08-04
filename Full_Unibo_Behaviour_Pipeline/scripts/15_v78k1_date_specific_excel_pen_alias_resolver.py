from pathlib import Path
from datetime import datetime
import re
import csv
import json
import hashlib
import zipfile
import pandas as pd
import openpyxl

ROOT = Path.home() / "PigBench"
FULL = ROOT / "Full_Unibo_Behaviour_Pipeline"
EXCEL_DIR = Path("/work/pig/datasets/Unibo/excel")

V78J_PKG = FULL / "outputs" / "v78j_time_aware_visual_stream_resolver" / "Full_Unibo_Time_Aware_Visual_Stream_Resolver"
V78J_TARGETS = V78J_PKG / "v78j_excel_visual_mapping_targets.csv"

OUT = FULL / "outputs" / "v78k1_date_specific_excel_pen_alias_resolver"
PKG = OUT / "Full_Unibo_Date_Specific_Excel_Pen_Alias_Resolver"
NOTES = FULL / "notes"
REPORTS = FULL / "reports"
PROGRESS = FULL / "progress"

for p in [OUT, PKG, NOTES, REPORTS, PROGRESS]:
    p.mkdir(parents=True, exist_ok=True)

OUT_RAW = PKG / "v78k1_raw_date_specific_excel_camera_room_pen_rows.csv"
OUT_ALIAS = PKG / "v78k1_big_small_alias_table.csv"
OUT_CANONICAL = PKG / "v78k1_date_specific_canonical_tlc_room_pen_pattern.csv"
OUT_TARGETS = PKG / "v78k1_v78j_targets_resolved_with_aliases.csv"
OUT_ISSUES = OUT / "v78k1_issues.csv"
OUT_DECISION = OUT / "v78k1_decision_summary.csv"
OUT_README = PKG / "README_v78k1_Date_Specific_Excel_Pen_Alias_Resolver.md"
OUT_MANIFEST = PKG / "v78k1_manifest.json"
OUT_ZIP = OUT / "Full_Unibo_Date_Specific_Excel_Pen_Alias_Resolver.zip"
OUT_SHA = OUT / "Full_Unibo_Date_Specific_Excel_Pen_Alias_Resolver.sha256"
OUT_NOTE = NOTES / "v78k1_date_specific_excel_pen_alias_resolver_notes.md"
OUT_REPORT = REPORTS / "v78k1_date_specific_excel_pen_alias_resolver_report.md"

def clean(x):
    if x is None:
        return ""
    s = str(x).strip()
    if s.lower() in {"nan", "none", "null"}:
        return ""
    return s

def to_csv(df, path):
    df.to_csv(path, index=False, quoting=csv.QUOTE_ALL, escapechar="\\", lineterminator="\n")

def sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()

def parse_date_from_filename(name):
    # Giorno 1 - 22_7_2021 Tutti.xlsx
    m = re.search(r"(\d{1,2})_(\d{1,2})_(\d{4})", name)
    if not m:
        return ""
    d, mo, y = m.groups()
    return f"{int(y):04d}-{int(mo):02d}-{int(d):02d}"

def parse_sheet_title(title):
    title_clean = clean(title)
    low = title_clean.lower()

    tlc = ""
    m = re.search(r"tlc\s*([1-6])", title_clean, re.I)
    if m:
        tlc = f"TLC{m.group(1)}"

    size_alias = ""
    if "big" in low:
        size_alias = "big"
    elif "small" in low:
        size_alias = "small"

    explicit_pen = ""
    m2 = re.search(r"\b([BCM]\s*[0-9]+)\b", title_clean, re.I)
    if m2:
        explicit_pen = m2.group(1).replace(" ", "").upper()

    return tlc, size_alias, explicit_pen

def parse_row1(ws):
    vals = [clean(c.value) for c in ws[1]]
    vals = [v for v in vals if v]
    text = " ; ".join(vals)

    camera_num = ""
    room = ""
    pen_num = ""

    for i, v in enumerate(vals):
        low = v.lower()
        if low == "camera" and i + 1 < len(vals):
            camera_num = clean(vals[i + 1])
        elif low == "room" and i + 1 < len(vals):
            room = clean(vals[i + 1]).upper()
        elif low == "pen" and i + 1 < len(vals):
            pen_num = clean(vals[i + 1])

    tlc = f"TLC{camera_num}" if camera_num else ""
    room_pen = f"{room}{pen_num}" if room and pen_num else ""

    return text, camera_num, tlc, room, pen_num, room_pen

rows = []
issues = []

excel_files = sorted(list(EXCEL_DIR.glob("*.xlsx")) + list(EXCEL_DIR.glob("*.xlsm")))

for path in excel_files:
    date = parse_date_from_filename(path.name)
    try:
        wb = openpyxl.load_workbook(path, data_only=False, read_only=False)
        for ws in wb.worksheets:
            title_tlc, size_alias, explicit_pen = parse_sheet_title(ws.title)
            row1_text, camera_num, row_tlc, room, pen_num, row_room_pen = parse_row1(ws)

            tlc = row_tlc or title_tlc
            if not tlc or not row_room_pen:
                continue

            rows.append({
                "date": date,
                "excel_file_name": path.name,
                "excel_file": str(path),
                "sheet_name": ws.title,
                "row1_text": row1_text,
                "camera_num": camera_num,
                "tlc_camera": tlc,
                "room": room,
                "pen_num": pen_num,
                "row_room_pen": row_room_pen,
                "sheet_size_alias": size_alias,
                "sheet_explicit_pen": explicit_pen,
                "sheet_label_type": "alias_big_small" if size_alias else ("explicit_pen" if explicit_pen else "unknown"),
                "alias_or_pen_key": size_alias if size_alias else explicit_pen,
            })
        wb.close()
    except Exception as e:
        issues.append({
            "item": str(path),
            "issue_type": "excel_read_error",
            "issue_detail": str(e),
            "severity": "warning",
        })

raw = pd.DataFrame(rows)
to_csv(raw, OUT_RAW)

if raw.empty:
    issues.append({
        "item": "raw_excel_rows",
        "issue_type": "hard_no_excel_camera_pen_rows",
        "issue_detail": "No usable Camera/Room/Pen rows extracted.",
        "severity": "hard",
    })
    alias = pd.DataFrame()
    canonical = pd.DataFrame()
    resolved = pd.DataFrame()
else:
    alias = raw[raw["sheet_size_alias"].isin(["big", "small"])].copy()
    alias = alias[[
        "date", "tlc_camera", "sheet_size_alias", "row_room_pen",
        "excel_file_name", "sheet_name", "row1_text"
    ]].drop_duplicates().sort_values(["date", "tlc_camera", "sheet_size_alias", "row_room_pen"])
    alias = alias.rename(columns={
        "sheet_size_alias": "alias",
        "row_room_pen": "resolved_room_pen",
    })
    to_csv(alias, OUT_ALIAS)

    canonical_rows = []
    for (date, tlc), g in raw.groupby(["date", "tlc_camera"]):
        pens = sorted(set([p for p in g["row_room_pen"].tolist() if p]))
        aliases = []
        for _, a in g[g["sheet_size_alias"].isin(["big", "small"])].iterrows():
            aliases.append(f"{a['sheet_size_alias']}={a['row_room_pen']}")
        aliases = sorted(set(aliases))

        canonical_rows.append({
            "date": date,
            "tlc_camera": tlc,
            "room_pens_observed": ";".join(pens),
            "aliases_observed": ";".join(aliases),
            "supporting_sheet_count": len(g),
            "supporting_excel_files": ";".join(sorted(set(g["excel_file_name"]))),
            "supporting_sheet_names": ";".join(sorted(set(g["sheet_name"]))),
            "canonical_status": "date_specific_pattern_resolved",
        })

    canonical = pd.DataFrame(canonical_rows).sort_values(["date", "tlc_camera"])
    to_csv(canonical, OUT_CANONICAL)

    if V78J_TARGETS.exists():
        targets = pd.read_csv(V78J_TARGETS).fillna("")
        for c in targets.columns:
            if targets[c].dtype == object:
                targets[c] = targets[c].map(clean)

        out_rows = []
        for _, t in targets.iterrows():
            date = clean(t.get("date", ""))
            tlc = clean(t.get("camera", ""))
            pen = clean(t.get("pen", ""))

            resolved_pen = ""
            match_type = ""
            support = pd.DataFrame()

            if pen.lower() in ["big", "small"]:
                support = alias[
                    (alias["date"] == date) &
                    (alias["tlc_camera"] == tlc) &
                    (alias["alias"] == pen.lower())
                ].copy()
                if not support.empty:
                    resolved_pen = support.iloc[0]["resolved_room_pen"]
                    match_type = f"alias_{pen.lower()}_resolved_to_{resolved_pen}"
            else:
                support = raw[
                    (raw["date"] == date) &
                    (raw["tlc_camera"] == tlc) &
                    (raw["row_room_pen"] == pen)
                ].copy()
                if not support.empty:
                    resolved_pen = pen
                    match_type = "explicit_room_pen_match"

            found = bool(resolved_pen)

            out_rows.append({
                **t.to_dict(),
                "date_specific_key": f"{date}|{tlc}|{pen}",
                "target_pen_original": pen,
                "resolved_room_pen": resolved_pen,
                "match_type": match_type,
                "excel_pattern_found": found,
                "excel_supporting_rows": len(support),
                "excel_supporting_files": ";".join(sorted(set(support["excel_file_name"].tolist()))) if not support.empty else "",
                "excel_supporting_sheets": ";".join(sorted(set(support["sheet_name"].tolist()))) if not support.empty else "",
                "camera_pen_resolution_status": "resolved_from_date_specific_excel_pattern" if found else "unresolved_no_date_specific_excel_match",
                "spatial_roi_status": "not_available_from_excel_pattern",
                "tracking_readiness_note": "Camera/pen association resolved from date-specific Excel pattern. Frame-level ROI coordinates still not provided by Excel.",
            })

        resolved = pd.DataFrame(out_rows)
        to_csv(resolved, OUT_TARGETS)
    else:
        issues.append({
            "item": str(V78J_TARGETS),
            "issue_type": "hard_missing_v78j_targets",
            "issue_detail": "v78j target file missing.",
            "severity": "hard",
        })
        resolved = pd.DataFrame()

# Validation: target-level must all resolve.
if not resolved.empty:
    unresolved_n = int((resolved["excel_pattern_found"] == False).sum())
    if unresolved_n:
        bad = resolved[resolved["excel_pattern_found"] == False]["date_specific_key"].tolist()
        issues.append({
            "item": "v78j_targets",
            "issue_type": "hard_unresolved_targets_after_alias_resolution",
            "issue_detail": f"{unresolved_n} unresolved targets: {bad}",
            "severity": "hard",
        })

# For 2021-07-22 specifically, expected pattern.
expected_20210722 = {
    "TLC1": {"B1", "B3"},
    "TLC2": {"B4", "B6"},
    "TLC3": {"C1", "C3"},
    "TLC4": {"C4", "C6"},
    "TLC5": {"M1", "M2"},
    "TLC6": {"M3", "M4"},
}

if not raw.empty:
    day = raw[raw["date"] == "2021-07-22"].copy()
    for tlc, exp in expected_20210722.items():
        obs = set(day[day["tlc_camera"] == tlc]["row_room_pen"].tolist())
        missing = exp - obs
        if missing:
            issues.append({
                "item": f"2021-07-22 {tlc}",
                "issue_type": "hard_missing_expected_day1_pattern",
                "issue_detail": f"Missing {sorted(missing)}; observed {sorted(obs)}",
                "severity": "hard",
            })

issues.append({
    "item": "spatial_roi",
    "issue_type": "info_excel_pattern_not_spatial_roi",
    "issue_detail": "Excel resolves date-specific TLC/camera-to-room/pen association and big/small aliases, but it does not provide frame-level ROI coordinates.",
    "severity": "info",
})

issues_df = pd.DataFrame(issues, columns=["item", "issue_type", "issue_detail", "severity"])
to_csv(issues_df, OUT_ISSUES)

hard_count = int((issues_df["severity"] == "hard").sum()) if len(issues_df) else 0
warning_count = int((issues_df["severity"] == "warning").sum()) if len(issues_df) else 0
info_count = int((issues_df["severity"] == "info").sum()) if len(issues_df) else 0

readme = """# v78k1 Date-Specific Excel Pen Alias Resolver

This stage fixes v78k0 by resolving `big` / `small` sheet aliases using the actual Excel row-1 Camera / Room / Pen metadata.

Important:
- `big` and `small` are not literal pen names.
- They are sheet aliases and must be resolved per date and TLC camera.
- This stage resolves camera/pen association, not frame-level spatial ROI.
"""
OUT_README.write_text(readme, encoding="utf-8")
OUT_REPORT.write_text(readme, encoding="utf-8")

manifest = {
    "version": "v78k1_date_specific_excel_pen_alias_resolver",
    "generated_at": datetime.now().isoformat(timespec="seconds"),
    "raw_rows": int(len(raw)),
    "alias_rows": int(len(alias)),
    "canonical_rows": int(len(canonical)),
    "resolved_target_rows": int(len(resolved)),
    "hard_issue_count": hard_count,
    "warning_count": warning_count,
    "claim_boundary": "date-specific Excel camera/pen association with alias resolution; no spatial ROI",
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
    "v78k1_decision": "date_specific_excel_pen_alias_pattern_resolved" if hard_count == 0 else "date_specific_excel_pen_alias_pattern_has_blocking_issues",
    "raw_rows": int(len(raw)),
    "alias_rows": int(len(alias)),
    "canonical_rows": int(len(canonical)),
    "resolved_target_rows": int(len(resolved)),
    "all_v78j_targets_resolved": bool((not resolved.empty) and int((resolved["excel_pattern_found"] == False).sum()) == 0),
    "zip_path": str(OUT_ZIP),
    "zip_sha256": zip_hash,
    "hard_issue_count": hard_count,
    "warning_count": warning_count,
    "info_count": info_count,
    "ready_for_camera_pen_association_use": bool(hard_count == 0),
    "ready_for_pen_level_spatial_roi_use": False,
    "ready_for_v79_full_tracking_preparation": False,
    "claim_scope": "date_specific_excel_camera_pen_alias_pattern_only",
    "generated_at": datetime.now().isoformat(timespec="seconds"),
}])
to_csv(decision, OUT_DECISION)

OUT_NOTE.write_text(
    "# v78k1 Date-Specific Excel Pen Alias Resolver\n\n"
    f"- Decision: {decision.iloc[0]['v78k1_decision']}\n"
    f"- Raw Excel rows: {len(raw)}\n"
    f"- Alias rows: {len(alias)}\n"
    f"- Canonical date-specific rows: {len(canonical)}\n"
    f"- Resolved v78j target rows: {len(resolved)}\n"
    f"- All v78j targets resolved: {bool((not resolved.empty) and int((resolved['excel_pattern_found'] == False).sum()) == 0)}\n"
    f"- Hard issues: {hard_count}\n"
    f"- Ready for camera/pen association use: {bool(hard_count == 0)}\n"
    f"- Ready for pen-level spatial ROI use: False\n\n"
    "This resolves big/small aliases per date and TLC camera. It does not provide frame-level ROI coordinates.\n",
    encoding="utf-8"
)

print("=== v78k1 decision ===")
print(decision.to_string(index=False))

print("\n=== alias table ===")
print(alias.to_string(index=False) if not alias.empty else "none")

print("\n=== 2021-07-22 canonical pattern ===")
print(canonical[canonical["date"] == "2021-07-22"].to_string(index=False) if not canonical.empty else "none")

print("\n=== resolved targets ===")
print(resolved.to_string(index=False) if not resolved.empty else "none")

print("\n=== issues ===")
print(issues_df.to_string(index=False))
