from pathlib import Path
from datetime import datetime
import csv
import json
import hashlib
import zipfile
import pandas as pd

ROOT = Path.home() / "PigBench"
FULL = ROOT / "Full_Unibo_Behaviour_Pipeline"

K4_PKG = FULL / "outputs" / "v78k4_annotation_source_of_truth_rebuild" / "Full_Unibo_Annotation_Source_of_Truth_Rebuild"
JOINED = K4_PKG / "v78k4_decoded_annotations_joined_with_sheet_truth.csv"
SHEET_TRUTH = K4_PKG / "v78k4_excel_sheet_source_of_truth_inventory.csv"
RULES = K4_PKG / "v78k4_annotation_interpretation_rules.csv"

OUT = FULL / "outputs" / "v78k5_frozen_annotation_truth_table"
PKG = OUT / "Full_Unibo_Frozen_Annotation_Truth_Table"
NOTES = FULL / "notes"
REPORTS = FULL / "reports"
PROGRESS = FULL / "progress"

for p in [OUT, PKG, NOTES, REPORTS, PROGRESS]:
    p.mkdir(parents=True, exist_ok=True)

OUT_FROZEN = PKG / "v78k5_frozen_annotation_truth_table.csv"
OUT_COLUMNS = PKG / "v78k5_input_column_inventory.csv"
OUT_SUMMARY = PKG / "v78k5_frozen_annotation_truth_summary.csv"
OUT_CLASS_COUNTS = PKG / "v78k5_behaviour_class_counts.csv"
OUT_TARGET_COUNTS = PKG / "v78k5_target_counts.csv"
OUT_RULES_COPY = PKG / "v78k5_annotation_truth_rules_used.csv"
OUT_ISSUES = OUT / "v78k5_issues.csv"
OUT_DECISION = OUT / "v78k5_decision_summary.csv"
OUT_README = PKG / "README_v78k5_Frozen_Annotation_Truth_Table.md"
OUT_MANIFEST = PKG / "v78k5_manifest.json"
OUT_ZIP = OUT / "Full_Unibo_Frozen_Annotation_Truth_Table.zip"
OUT_SHA = OUT / "Full_Unibo_Frozen_Annotation_Truth_Table.sha256"
OUT_NOTE = NOTES / "v78k5_frozen_annotation_truth_table_notes.md"
OUT_REPORT = REPORTS / "v78k5_frozen_annotation_truth_table_report.md"
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

def first_existing_col(df, candidates):
    lower_map = {c.lower(): c for c in df.columns}
    for cand in candidates:
        if cand.lower() in lower_map:
            return lower_map[cand.lower()]
    for c in df.columns:
        cl = c.lower()
        for cand in candidates:
            if cand.lower() in cl:
                return c
    return ""

def make_id(*parts):
    raw = "|".join(clean(p) for p in parts)
    return hashlib.sha1(raw.encode("utf-8")).hexdigest()[:16]

issues = []

joined = read_csv(JOINED)
sheet_truth = read_csv(SHEET_TRUTH)
rules = read_csv(RULES)

if joined.empty:
    issues.append({
        "item": str(JOINED),
        "issue_type": "hard_missing_or_empty_joined_annotation_truth",
        "issue_detail": "v78k4 joined annotation truth table is missing or empty.",
        "severity": "hard",
    })

if sheet_truth.empty:
    issues.append({
        "item": str(SHEET_TRUTH),
        "issue_type": "warning_missing_sheet_truth_inventory",
        "issue_detail": "Sheet truth inventory missing or empty.",
        "severity": "warning",
    })

if not rules.empty:
    to_csv(rules, OUT_RULES_COPY)

# Input column inventory
col_inventory = pd.DataFrame([{
    "column_name": c,
    "non_empty_count": int((joined[c].astype(str).map(clean) != "").sum()) if not joined.empty else 0,
    "sample_values": "; ".join(joined[c].astype(str).map(clean).drop_duplicates().head(8).tolist()) if not joined.empty else "",
} for c in joined.columns])
to_csv(col_inventory, OUT_COLUMNS)

# Dynamic column detection from v77c/v78k4 join output.
date_col = first_existing_col(joined, ["date", "annotation_date", "day"])
excel_file_col = first_existing_col(joined, ["excel_file_name", "excel_file", "source_excel_file"])
sheet_col = first_existing_col(joined, ["sheet_name", "sheet", "worksheet", "excel_sheet"])
tlc_col = first_existing_col(joined, ["tlc_camera", "camera", "source_tlc"])
room_pen_col = first_existing_col(joined, ["resolved_room_pen", "room_pen", "pen", "target_pen"])
room_col = first_existing_col(joined, ["room"])
pen_num_col = first_existing_col(joined, ["pen_num"])
behaviour_col = first_existing_col(joined, ["behaviour", "behaviour_label", "behavior", "behavior_label", "class", "label"])
colour_col = first_existing_col(joined, ["colour", "color", "identity_colour", "identity_color", "pig_colour", "pig_color"])
start_col = first_existing_col(joined, ["window_start", "start_time", "start_seconds", "start_sec", "t_start", "start"])
end_col = first_existing_col(joined, ["window_end", "end_time", "end_seconds", "end_sec", "t_end", "end"])
duration_col = first_existing_col(joined, ["window_duration", "duration", "duration_seconds"])

required_truth_cols = {
    "date": date_col,
    "sheet_name": sheet_col,
    "tlc_camera": tlc_col,
    "resolved_room_pen": room_pen_col,
}

for logical, actual in required_truth_cols.items():
    if not actual:
        issues.append({
            "item": logical,
            "issue_type": "hard_missing_required_truth_column",
            "issue_detail": f"Could not detect required column for {logical}.",
            "severity": "hard",
        })

if not behaviour_col:
    issues.append({
        "item": "behaviour_label",
        "issue_type": "warning_behaviour_column_not_detected",
        "issue_detail": "Could not confidently detect behaviour label column. Frozen table will include blank behaviour_label.",
        "severity": "warning",
    })

frozen_rows = []

if not joined.empty and all(required_truth_cols.values()):
    for i, r in joined.iterrows():
        date = clean(r.get(date_col, ""))
        excel_file = clean(r.get(excel_file_col, "")) if excel_file_col else ""
        sheet = clean(r.get(sheet_col, ""))
        tlc = clean(r.get(tlc_col, ""))
        resolved_room_pen = clean(r.get(room_pen_col, ""))

        room = clean(r.get(room_col, "")) if room_col else ""
        pen_num = clean(r.get(pen_num_col, "")) if pen_num_col else ""

        if not room and resolved_room_pen:
            room = resolved_room_pen[0]
        if not pen_num and len(resolved_room_pen) > 1:
            pen_num = resolved_room_pen[1:]

        behaviour = clean(r.get(behaviour_col, "")) if behaviour_col else ""
        colour = clean(r.get(colour_col, "")) if colour_col else ""

        start = clean(r.get(start_col, "")) if start_col else ""
        end = clean(r.get(end_col, "")) if end_col else ""
        duration = clean(r.get(duration_col, "")) if duration_col else ""

        source_truth_statement = clean(r.get("source_truth_statement", ""))
        annotation_unit_type = clean(r.get("annotation_unit_type", ""))

        annotation_window_id = make_id(date, excel_file, sheet, tlc, resolved_room_pen, behaviour, colour, start, end, str(i))

        frozen_rows.append({
            "annotation_window_id": annotation_window_id,
            "source_row_index": i,
            "date": date,
            "source_excel_file": excel_file,
            "sheet_name": sheet,
            "tlc_camera": tlc,
            "room": room,
            "pen_num": pen_num,
            "resolved_room_pen": resolved_room_pen,
            "identity_colour": colour,
            "behaviour_label": behaviour,
            "window_start": start,
            "window_end": end,
            "window_duration": duration,
            "annotation_unit_type": annotation_unit_type,
            "source_truth_statement": source_truth_statement,
            "annotation_source_status": "frozen_from_v78k4_sheet_truth",
            "video_mapping_status": "pending_not_part_of_annotation_truth",
            "spatial_roi_status": "not_required_for_annotation_truth",
            "tracking_status": "not_started",
            "claim_boundary": "annotation_truth_only_no_video_mapping_no_roi_no_tracking",
        })

frozen = pd.DataFrame(frozen_rows)
to_csv(frozen, OUT_FROZEN)

if frozen.empty:
    issues.append({
        "item": "frozen_table",
        "issue_type": "hard_empty_frozen_annotation_truth_table",
        "issue_detail": "Frozen annotation truth table is empty.",
        "severity": "hard",
    })

# QA checks
if not frozen.empty:
    missing_truth = frozen[
        (frozen["date"] == "") |
        (frozen["sheet_name"] == "") |
        (frozen["tlc_camera"] == "") |
        (frozen["resolved_room_pen"] == "")
    ]

    if len(missing_truth):
        issues.append({
            "item": "frozen_required_fields",
            "issue_type": "hard_missing_required_truth_values",
            "issue_detail": f"{len(missing_truth)} rows have missing date/sheet/tlc/resolved_room_pen.",
            "severity": "hard",
        })

    duplicate_ids = int(frozen["annotation_window_id"].duplicated().sum())
    if duplicate_ids:
        issues.append({
            "item": "annotation_window_id",
            "issue_type": "hard_duplicate_annotation_window_ids",
            "issue_detail": f"{duplicate_ids} duplicate annotation_window_id values.",
            "severity": "hard",
        })

    if behaviour_col and int((frozen["behaviour_label"] == "").sum()):
        issues.append({
            "item": "behaviour_label",
            "issue_type": "warning_some_blank_behaviour_labels",
            "issue_detail": f"{int((frozen['behaviour_label'] == '').sum())} rows have blank behaviour_label.",
            "severity": "warning",
        })

    # Expected 2021-07-22 target coverage.
    expected_day1 = {
        "TLC1": {"B1", "B3"},
        "TLC2": {"B4", "B6"},
        "TLC3": {"C1", "C3"},
        "TLC4": {"C4", "C6"},
        "TLC5": {"M1", "M2"},
        "TLC6": {"M3", "M4"},
    }
    d1 = frozen[frozen["date"] == "2021-07-22"].copy()
    for tlc, expected in expected_day1.items():
        observed = set(d1[d1["tlc_camera"] == tlc]["resolved_room_pen"].dropna().tolist())
        missing = expected - observed
        if missing:
            issues.append({
                "item": f"2021-07-22 {tlc}",
                "issue_type": "hard_missing_expected_frozen_day1_target",
                "issue_detail": f"Missing expected target(s) {sorted(missing)}; observed={sorted(observed)}",
                "severity": "hard",
            })

# Summary outputs
if not frozen.empty:
    summary = pd.DataFrame([{
        "frozen_rows": len(frozen),
        "unique_dates": frozen["date"].nunique(),
        "unique_sheets": frozen["sheet_name"].nunique(),
        "unique_tlc_cameras": frozen["tlc_camera"].nunique(),
        "unique_room_pens": frozen["resolved_room_pen"].nunique(),
        "unique_behaviour_labels": frozen["behaviour_label"].nunique() if "behaviour_label" in frozen.columns else 0,
        "rows_with_behaviour_label": int((frozen["behaviour_label"] != "").sum()),
        "rows_with_identity_colour": int((frozen["identity_colour"] != "").sum()),
        "video_mapping_status": "pending_not_part_of_annotation_truth",
        "spatial_roi_status": "not_required_for_annotation_truth",
    }])
else:
    summary = pd.DataFrame([{
        "frozen_rows": 0,
        "unique_dates": 0,
        "unique_sheets": 0,
        "unique_tlc_cameras": 0,
        "unique_room_pens": 0,
        "unique_behaviour_labels": 0,
        "rows_with_behaviour_label": 0,
        "rows_with_identity_colour": 0,
        "video_mapping_status": "not_created",
        "spatial_roi_status": "not_created",
    }])

to_csv(summary, OUT_SUMMARY)

if not frozen.empty and "behaviour_label" in frozen.columns:
    class_counts = (
        frozen.groupby("behaviour_label")
        .size()
        .reset_index(name="row_count")
        .sort_values("row_count", ascending=False)
    )
else:
    class_counts = pd.DataFrame(columns=["behaviour_label", "row_count"])
to_csv(class_counts, OUT_CLASS_COUNTS)

if not frozen.empty:
    target_counts = (
        frozen.groupby(["date", "tlc_camera", "resolved_room_pen"])
        .size()
        .reset_index(name="row_count")
        .sort_values(["date", "tlc_camera", "resolved_room_pen"])
    )
else:
    target_counts = pd.DataFrame(columns=["date", "tlc_camera", "resolved_room_pen", "row_count"])
to_csv(target_counts, OUT_TARGET_COUNTS)

issues.append({
    "item": "scope",
    "issue_type": "info_frozen_annotation_truth_only",
    "issue_detail": "v78k5 freezes annotation source truth only. It does not solve encoded c-code mapping, spatial ROI, or tracking.",
    "severity": "info",
})

issues_df = pd.DataFrame(issues, columns=["item", "issue_type", "issue_detail", "severity"])
to_csv(issues_df, OUT_ISSUES)

hard_count = int((issues_df["severity"] == "hard").sum()) if len(issues_df) else 0
warning_count = int((issues_df["severity"] == "warning").sum()) if len(issues_df) else 0
info_count = int((issues_df["severity"] == "info").sum()) if len(issues_df) else 0

readme = """# v78k5 Frozen Annotation Truth Table

This package freezes the annotation source-of-truth table derived from v78k4.

Core rule:
- Excel sheet metadata defines the annotation target.
- B1/B3/etc. are sheet-level target labels, not visually inferred frame regions.
- Video mapping, ROI, and tracking are intentionally outside this stage.
"""
OUT_README.write_text(readme, encoding="utf-8")
OUT_REPORT.write_text(readme, encoding="utf-8")

manifest = {
    "version": "v78k5_frozen_annotation_truth_table",
    "generated_at": datetime.now().isoformat(timespec="seconds"),
    "input_joined": str(JOINED),
    "frozen_rows": int(len(frozen)),
    "summary": summary.iloc[0].to_dict() if len(summary) else {},
    "detected_columns": {
        "date_col": date_col,
        "excel_file_col": excel_file_col,
        "sheet_col": sheet_col,
        "tlc_col": tlc_col,
        "room_pen_col": room_pen_col,
        "behaviour_col": behaviour_col,
        "colour_col": colour_col,
        "start_col": start_col,
        "end_col": end_col,
        "duration_col": duration_col,
    },
    "hard_issue_count": hard_count,
    "warning_count": warning_count,
    "claim_boundary": "frozen annotation truth only; no video mapping, ROI, or tracking",
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
    "v78k5_decision": "frozen_annotation_truth_table_created" if hard_count == 0 else "frozen_annotation_truth_table_has_blocking_issues",
    "input_joined_rows": int(len(joined)),
    "frozen_rows": int(len(frozen)),
    "unique_dates": int(summary.iloc[0]["unique_dates"]),
    "unique_sheets": int(summary.iloc[0]["unique_sheets"]),
    "unique_tlc_cameras": int(summary.iloc[0]["unique_tlc_cameras"]),
    "unique_room_pens": int(summary.iloc[0]["unique_room_pens"]),
    "unique_behaviour_labels": int(summary.iloc[0]["unique_behaviour_labels"]),
    "rows_with_behaviour_label": int(summary.iloc[0]["rows_with_behaviour_label"]),
    "rows_with_identity_colour": int(summary.iloc[0]["rows_with_identity_colour"]),
    "zip_path": str(OUT_ZIP),
    "zip_sha256": zip_hash,
    "hard_issue_count": hard_count,
    "warning_count": warning_count,
    "info_count": info_count,
    "ready_for_annotation_truth_use": bool(hard_count == 0 and len(frozen) > 0),
    "ready_for_video_mapping": False,
    "ready_for_tracking": False,
    "claim_scope": "frozen_annotation_truth_only",
    "generated_at": datetime.now().isoformat(timespec="seconds"),
}])
to_csv(decision, OUT_DECISION)

OUT_NOTE.write_text(
    "# v78k5 Frozen Annotation Truth Table\n\n"
    f"- Decision: {decision.iloc[0]['v78k5_decision']}\n"
    f"- Input joined rows: {len(joined)}\n"
    f"- Frozen rows: {len(frozen)}\n"
    f"- Unique dates: {summary.iloc[0]['unique_dates']}\n"
    f"- Unique sheets: {summary.iloc[0]['unique_sheets']}\n"
    f"- Unique TLC cameras: {summary.iloc[0]['unique_tlc_cameras']}\n"
    f"- Unique room/pens: {summary.iloc[0]['unique_room_pens']}\n"
    f"- Unique behaviour labels: {summary.iloc[0]['unique_behaviour_labels']}\n"
    f"- Hard issues: {hard_count}\n"
    f"- Ready for annotation truth use: {bool(hard_count == 0 and len(frozen) > 0)}\n\n"
    "This is the frozen source-of-truth annotation table. Video mapping, ROI and tracking remain separate downstream tasks.\n",
    encoding="utf-8"
)

progress_row = pd.DataFrame([{
    "date": datetime.now().date().isoformat(),
    "stage": "v78k5",
    "task_name": "Frozen annotation truth table",
    "status": "PASS" if hard_count == 0 else "NEEDS_FIX",
    "input_summary": "v78k4 joined annotation truth",
    "output_summary": str(PKG),
    "hard_issues": hard_count,
    "warnings": warning_count,
    "next_action": "Use frozen annotation truth table as source-of-truth; handle video mapping separately.",
}])

if OUT_PROGRESS.exists():
    old = pd.read_csv(OUT_PROGRESS)
    progress = pd.concat([old, progress_row], ignore_index=True)
else:
    progress = progress_row
to_csv(progress, OUT_PROGRESS)

print("=== v78k5 decision ===")
print(decision.to_string(index=False))

print("\n=== summary ===")
print(summary.to_string(index=False))

print("\n=== target counts sample ===")
print(target_counts.head(40).to_string(index=False))

print("\n=== class counts ===")
print(class_counts.to_string(index=False))

print("\n=== issues ===")
print(issues_df.to_string(index=False))
