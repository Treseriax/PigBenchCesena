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

OUT = FULL / "outputs" / "v78k0_excel_camera_pen_pattern_resolver"
PKG = OUT / "Full_Unibo_Excel_Camera_Pen_Pattern_Resolver"
NOTES = FULL / "notes"
REPORTS = FULL / "reports"
PROGRESS = FULL / "progress"

for p in [OUT, PKG, NOTES, REPORTS, PROGRESS]:
    p.mkdir(parents=True, exist_ok=True)

OUT_RAW = PKG / "v78k0_raw_excel_camera_room_pen_rows.csv"
OUT_CANONICAL = PKG / "v78k0_canonical_tlc_room_pen_pattern.csv"
OUT_TARGETS = PKG / "v78k0_v78j_targets_resolved_from_excel_pattern.csv"
OUT_MATRIX = PKG / "v78k0_camera_pen_matrix.csv"
OUT_ISSUES = OUT / "v78k0_issues.csv"
OUT_DECISION = OUT / "v78k0_decision_summary.csv"
OUT_README = PKG / "README_v78k0_Excel_Camera_Pen_Pattern_Resolver.md"
OUT_MANIFEST = PKG / "v78k0_manifest.json"
OUT_ZIP = OUT / "Full_Unibo_Excel_Camera_Pen_Pattern_Resolver.zip"
OUT_SHA = OUT / "Full_Unibo_Excel_Camera_Pen_Pattern_Resolver.sha256"
OUT_NOTE = NOTES / "v78k0_excel_camera_pen_pattern_resolver_notes.md"
OUT_REPORT = REPORTS / "v78k0_excel_camera_pen_pattern_resolver_report.md"
OUT_PROGRESS = PROGRESS / "full_unibo_pipeline_progress_log.csv"


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


def parse_sheet_title(title):
    title = clean(title)
    m = re.search(r"TLC\s*([1-6])\s*([BCM])?\s*([0-9]+|big|small|B1|B3|B4|B6|C1|C3|C4|C6|M1|M2|M3|M4)?", title, re.I)
    out = {
        "sheet_tlc_from_title": "",
        "sheet_pen_token_from_title": "",
        "sheet_size_token": "",
    }
    if m:
        out["sheet_tlc_from_title"] = f"TLC{m.group(1)}"
        token = clean(m.group(2) or "") + clean(m.group(3) or "")
        token = token.strip()
        if token.lower() in {"big", "small"}:
            out["sheet_size_token"] = token.lower()
        else:
            out["sheet_pen_token_from_title"] = token.upper()
    if "big" in title.lower():
        out["sheet_size_token"] = "big"
    if "small" in title.lower():
        out["sheet_size_token"] = "small"
    return out


def parse_row1(ws):
    vals = [clean(c.value) for c in ws[1]]
    nonempty = [v for v in vals if v]
    text = " ; ".join(nonempty)

    # Expected form: Camera ; 2 ; Room ; B ; Pen ; 4 ; Tesi
    camera_num = ""
    room = ""
    pen_num = ""

    for i, v in enumerate(nonempty):
        low = v.lower()
        if low == "camera" and i + 1 < len(nonempty):
            camera_num = clean(nonempty[i + 1])
        if low == "room" and i + 1 < len(nonempty):
            room = clean(nonempty[i + 1]).upper()
        if low == "pen" and i + 1 < len(nonempty):
            pen_num = clean(nonempty[i + 1])

    return text, camera_num, room, pen_num


rows = []
issues = []

excel_files = sorted(list(EXCEL_DIR.glob("*.xlsx")) + list(EXCEL_DIR.glob("*.xlsm")))

for path in excel_files:
    try:
        wb = openpyxl.load_workbook(path, data_only=False, read_only=False)
        for ws in wb.worksheets:
            title_info = parse_sheet_title(ws.title)
            row1_text, camera_num, room, pen_num = parse_row1(ws)

            if not camera_num and not room and not pen_num:
                continue

            tlc = f"TLC{camera_num}" if camera_num else title_info["sheet_tlc_from_title"]
            room_pen = f"{room}{pen_num}" if room and pen_num else ""

            rows.append({
                "excel_file": str(path),
                "excel_file_name": path.name,
                "sheet_name": ws.title,
                "row1_text": row1_text,
                "camera_num": camera_num,
                "tlc_camera": tlc,
                "room": room,
                "pen_num": pen_num,
                "room_pen": room_pen,
                "sheet_tlc_from_title": title_info["sheet_tlc_from_title"],
                "sheet_pen_token_from_title": title_info["sheet_pen_token_from_title"],
                "sheet_size_token": title_info["sheet_size_token"],
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
        "item": "excel_camera_pen_rows",
        "issue_type": "hard_no_camera_pen_rows",
        "issue_detail": "No Camera/Room/Pen rows were extracted from Excel.",
        "severity": "hard",
    })
    canonical = pd.DataFrame()
    targets_resolved = pd.DataFrame()
    matrix = pd.DataFrame()
else:
    # Canonical: each TLC camera has all observed room_pen values.
    canonical_rows = []
    for tlc, g in raw.groupby("tlc_camera"):
        pens = sorted(set([p for p in g["room_pen"].tolist() if p]))
        rooms = sorted(set([r for r in g["room"].tolist() if r]))
        sheet_names = sorted(set(g["sheet_name"].tolist()))
        size_tokens = sorted(set([x for x in g["sheet_size_token"].tolist() if x]))
        canonical_rows.append({
            "tlc_camera": tlc,
            "camera_num": tlc.replace("TLC", ""),
            "rooms_observed": ";".join(rooms),
            "room_pens_observed": ";".join(pens),
            "num_room_pens": len(pens),
            "sheet_size_tokens": ";".join(size_tokens),
            "supporting_sheet_count": len(g),
            "supporting_excel_files": ";".join(sorted(set(g["excel_file_name"]))),
            "supporting_sheet_names_sample": ";".join(sheet_names[:20]),
            "canonical_status": "resolved_from_excel_pattern" if pens else "unresolved",
        })

    canonical = pd.DataFrame(canonical_rows).sort_values("tlc_camera")
    to_csv(canonical, OUT_CANONICAL)

    # Matrix format: Camera x pens.
    matrix_rows = []
    for _, r in canonical.iterrows():
        pens = r["room_pens_observed"].split(";") if clean(r["room_pens_observed"]) else []
        matrix_rows.append({
            "tlc_camera": r["tlc_camera"],
            "camera_num": r["camera_num"],
            "pen_slot_1": pens[0] if len(pens) > 0 else "",
            "pen_slot_2": pens[1] if len(pens) > 1 else "",
            "all_pens": ";".join(pens),
            "interpretation": f"{r['tlc_camera']} is associated in Excel with pen(s): {'; '.join(pens)}",
        })
    matrix = pd.DataFrame(matrix_rows)
    to_csv(matrix, OUT_MATRIX)

    # Resolve v78j targets using Excel pattern.
    if V78J_TARGETS.exists():
        t = pd.read_csv(V78J_TARGETS).fillna("")
        for c in t.columns:
            if t[c].dtype == object:
                t[c] = t[c].map(clean)

        out_rows = []
        known_pairs = set((raw["tlc_camera"] + "|" + raw["room_pen"]).tolist())

        for _, r in t.iterrows():
            camera = clean(r.get("camera", ""))
            pen = clean(r.get("pen", ""))
            key = f"{camera}|{pen}"
            support = raw[(raw["tlc_camera"] == camera) & (raw["room_pen"] == pen)]

            out_rows.append({
                **r.to_dict(),
                "excel_pattern_key": key,
                "excel_pattern_found": key in known_pairs,
                "excel_supporting_rows": len(support),
                "excel_supporting_files": ";".join(sorted(set(support["excel_file_name"].tolist()))),
                "excel_supporting_sheets": ";".join(sorted(set(support["sheet_name"].tolist()))),
                "camera_pen_resolution_status": "resolved_from_excel_camera_room_pen_pattern" if key in known_pairs else "unresolved_no_excel_pattern_match",
                "spatial_roi_status": "not_available_from_excel_pattern",
                "tracking_readiness_note": "Camera/pen association is resolved, but frame-level ROI is not derived from Excel. Do not crop by pen unless additional ROI evidence is available.",
            })

        targets_resolved = pd.DataFrame(out_rows)
        to_csv(targets_resolved, OUT_TARGETS)
    else:
        issues.append({
            "item": str(V78J_TARGETS),
            "issue_type": "warning_missing_v78j_targets",
            "issue_detail": "v78j target table not found; only canonical Excel pattern was generated.",
            "severity": "warning",
        })
        targets_resolved = pd.DataFrame()

# QA
expected = {
    "TLC1": {"B1", "B3"},
    "TLC2": {"B4", "B6"},
    "TLC3": {"C1", "C3"},
    "TLC4": {"C4", "C6"},
    "TLC5": {"M1", "M2"},
    "TLC6": {"M3", "M4"},
}

if not raw.empty:
    for tlc, exp in expected.items():
        obs = set(raw[raw["tlc_camera"] == tlc]["room_pen"].dropna().tolist())
        missing = exp - obs
        extra = obs - exp
        if missing:
            issues.append({
                "item": tlc,
                "issue_type": "hard_expected_pen_pattern_missing",
                "issue_detail": f"Missing expected pens {sorted(missing)}; observed {sorted(obs)}",
                "severity": "hard",
            })
        if extra:
            issues.append({
                "item": tlc,
                "issue_type": "warning_extra_pen_pattern_observed",
                "issue_detail": f"Extra observed pens {sorted(extra)} beyond expected {sorted(exp)}",
                "severity": "warning",
            })

    if len(raw["tlc_camera"].dropna().unique()) < 6:
        issues.append({
            "item": "tlc_camera_count",
            "issue_type": "hard_less_than_6_tlc_cameras",
            "issue_detail": f"Observed TLC cameras: {sorted(raw['tlc_camera'].dropna().unique())}",
            "severity": "hard",
        })

if targets_resolved is not None and not targets_resolved.empty:
    unresolved_target_count = int((targets_resolved["excel_pattern_found"] == False).sum())
    if unresolved_target_count:
        issues.append({
            "item": "v78j_targets",
            "issue_type": "hard_some_targets_not_resolved_by_excel_pattern",
            "issue_detail": f"{unresolved_target_count} v78j target rows did not match Excel pattern.",
            "severity": "hard",
        })

issues.append({
    "item": "spatial_roi",
    "issue_type": "info_excel_pattern_not_spatial_roi",
    "issue_detail": "Excel resolves which TLC/camera is associated with which room/pen, but it does not provide frame-level ROI coordinates.",
    "severity": "info",
})

issues_df = pd.DataFrame(issues, columns=["item", "issue_type", "issue_detail", "severity"])
to_csv(issues_df, OUT_ISSUES)

hard_count = int((issues_df["severity"] == "hard").sum()) if len(issues_df) else 0
warning_count = int((issues_df["severity"] == "warning").sum()) if len(issues_df) else 0
info_count = int((issues_df["severity"] == "info").sum()) if len(issues_df) else 0

readme = f"""# v78k0 Excel Camera/Pen Pattern Resolver

## Purpose

This stage extracts the Camera / Room / Pen pattern directly from the Excel annotation sheets.

It resolves which TLC/camera is associated with which pen labels, without relying on visual guessing.

## Canonical pattern expected

- TLC1 -> B1 / B3
- TLC2 -> B6 / B4
- TLC3 -> C1 / C3
- TLC4 -> C6 / C4
- TLC5 -> M1 / M2
- TLC6 -> M4 / M3

## Important boundary

This stage resolves camera/pen association from Excel, but it does **not** provide frame-level ROI coordinates.

If a video frame contains multiple pens, pen-specific spatial filtering still requires ROI or layout evidence.
"""

OUT_README.write_text(readme, encoding="utf-8")
OUT_REPORT.write_text(readme, encoding="utf-8")

manifest = {
    "version": "v78k0_excel_camera_pen_pattern_resolver",
    "generated_at": datetime.now().isoformat(timespec="seconds"),
    "raw_excel_rows": int(len(raw)),
    "canonical_rows": int(len(canonical)),
    "targets_resolved_rows": int(len(targets_resolved)) if targets_resolved is not None else 0,
    "hard_issue_count": hard_count,
    "warning_count": warning_count,
    "claim_boundary": "Excel-derived TLC/camera-to-room/pen association only; no spatial ROI",
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
    "v78k0_decision": "excel_camera_pen_pattern_resolved" if hard_count == 0 else "excel_camera_pen_pattern_has_blocking_issues",
    "raw_excel_rows": int(len(raw)),
    "canonical_rows": int(len(canonical)),
    "targets_resolved_rows": int(len(targets_resolved)) if targets_resolved is not None else 0,
    "zip_path": str(OUT_ZIP),
    "zip_sha256": zip_hash,
    "hard_issue_count": hard_count,
    "warning_count": warning_count,
    "info_count": info_count,
    "ready_for_camera_pen_association_use": bool(hard_count == 0),
    "ready_for_pen_level_spatial_roi_use": False,
    "ready_for_v79_full_tracking_preparation": False,
    "claim_scope": "excel_camera_pen_pattern_only",
    "generated_at": datetime.now().isoformat(timespec="seconds"),
}])
to_csv(decision, OUT_DECISION)

OUT_NOTE.write_text(
    "# v78k0 Excel Camera/Pen Pattern Resolver\n\n"
    f"- Decision: {decision.iloc[0]['v78k0_decision']}\n"
    f"- Raw Excel Camera/Room/Pen rows: {len(raw)}\n"
    f"- Canonical TLC rows: {len(canonical)}\n"
    f"- v78j targets resolved: {len(targets_resolved) if targets_resolved is not None else 0}\n"
    f"- Hard issues: {hard_count}\n"
    f"- Ready for camera/pen association use: {bool(hard_count == 0)}\n"
    f"- Ready for pen-level spatial ROI use: False\n\n"
    "Excel pattern resolves which TLC/camera is associated with which room/pen. It does not define frame-level ROI coordinates.\n",
    encoding="utf-8"
)

progress_row = pd.DataFrame([{
    "date": datetime.now().date().isoformat(),
    "stage": "v78k0",
    "task_name": "Excel camera/pen pattern resolver",
    "status": "PASS" if hard_count == 0 else "NEEDS_FIX",
    "input_summary": "Unibo Excel sheets + v78j targets",
    "output_summary": str(PKG),
    "hard_issues": hard_count,
    "warnings": warning_count,
    "next_action": "Use Excel pattern for camera/pen association; separately handle missing spatial ROI if required.",
}])

if OUT_PROGRESS.exists():
    old = pd.read_csv(OUT_PROGRESS)
    progress = pd.concat([old, progress_row], ignore_index=True)
else:
    progress = progress_row
to_csv(progress, OUT_PROGRESS)

print("=== v78k0 decision ===")
print(decision.to_string(index=False))

print("\n=== canonical pattern ===")
print(canonical.to_string(index=False) if not canonical.empty else "none")

print("\n=== v78j targets resolved ===")
print(targets_resolved.to_string(index=False) if targets_resolved is not None and not targets_resolved.empty else "none")

print("\n=== issues ===")
print(issues_df.to_string(index=False))
