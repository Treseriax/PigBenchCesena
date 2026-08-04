from pathlib import Path
from datetime import datetime
import csv
import json
import hashlib
import zipfile
import pandas as pd

ROOT = Path.home() / "PigBench"
FULL = ROOT / "Full_Unibo_Behaviour_Pipeline"

V77C_FULL = FULL / "outputs" / "v77c_layout_aware_excel_decoder" / "Full_Unibo_Layout_Aware_Excel_Decoder" / "v77c_layout_aware_annotation_windows.csv"

K4_PKG = FULL / "outputs" / "v78k4_annotation_source_of_truth_rebuild" / "Full_Unibo_Annotation_Source_of_Truth_Rebuild"
SHEET_TRUTH = K4_PKG / "v78k4_excel_sheet_source_of_truth_inventory.csv"
RULES = K4_PKG / "v78k4_annotation_interpretation_rules.csv"

OUT = FULL / "outputs" / "v78k5b_corrected_full_frozen_annotation_truth_table"
PKG = OUT / "Full_Unibo_Corrected_Full_Frozen_Annotation_Truth_Table"
NOTES = FULL / "notes"
REPORTS = FULL / "reports"
PROGRESS = FULL / "progress"

for p in [OUT, PKG, NOTES, REPORTS, PROGRESS]:
    p.mkdir(parents=True, exist_ok=True)

OUT_FROZEN = PKG / "v78k5b_corrected_full_frozen_annotation_truth_table.csv"
OUT_COLUMNS = PKG / "v78k5b_input_column_inventory.csv"
OUT_SUMMARY = PKG / "v78k5b_frozen_annotation_truth_summary.csv"
OUT_CLASS_COUNTS = PKG / "v78k5b_behaviour_class_counts.csv"
OUT_TARGET_COUNTS = PKG / "v78k5b_target_counts.csv"
OUT_JOIN_QA = PKG / "v78k5b_join_quality_report.csv"
OUT_RULES_COPY = PKG / "v78k5b_annotation_truth_rules_used.csv"
OUT_ISSUES = OUT / "v78k5b_issues.csv"
OUT_DECISION = OUT / "v78k5b_decision_summary.csv"
OUT_README = PKG / "README_v78k5b_Corrected_Full_Frozen_Annotation_Truth_Table.md"
OUT_MANIFEST = PKG / "v78k5b_manifest.json"
OUT_ZIP = OUT / "Full_Unibo_Corrected_Full_Frozen_Annotation_Truth_Table.zip"
OUT_SHA = OUT / "Full_Unibo_Corrected_Full_Frozen_Annotation_Truth_Table.sha256"
OUT_NOTE = NOTES / "v78k5b_corrected_full_frozen_annotation_truth_table_notes.md"
OUT_REPORT = REPORTS / "v78k5b_corrected_full_frozen_annotation_truth_table_report.md"
OUT_PROGRESS = PROGRESS / "full_unibo_pipeline_progress_log.csv"

EXPECTED_FULL_ROWS = 15733

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

full = read_csv(V77C_FULL)
truth = read_csv(SHEET_TRUTH)
rules = read_csv(RULES)

if full.empty:
    issues.append({
        "item": str(V77C_FULL),
        "issue_type": "hard_missing_v77c_full_annotation_windows",
        "issue_detail": "Full v77c annotation windows file is missing or empty.",
        "severity": "hard",
    })

if truth.empty:
    issues.append({
        "item": str(SHEET_TRUTH),
        "issue_type": "hard_missing_sheet_truth_inventory",
        "issue_detail": "v78k4 sheet truth inventory is missing or empty.",
        "severity": "hard",
    })

if not rules.empty:
    to_csv(rules, OUT_RULES_COPY)

col_inventory = pd.DataFrame([{
    "column_name": c,
    "non_empty_count": int((full[c].astype(str).map(clean) != "").sum()) if not full.empty else 0,
    "sample_values": "; ".join(full[c].astype(str).map(clean).drop_duplicates().head(8).tolist()) if not full.empty else "",
} for c in full.columns])
to_csv(col_inventory, OUT_COLUMNS)

date_col = first_existing_col(full, ["date", "annotation_date", "day"])
excel_file_col = first_existing_col(full, ["excel_file_name", "excel_file", "source_excel_file"])
sheet_col = first_existing_col(full, ["sheet_name", "sheet", "worksheet", "excel_sheet"])
behaviour_col = first_existing_col(full, ["behaviour", "behaviour_label", "behavior", "behavior_label", "class", "label"])
colour_col = first_existing_col(full, ["colour", "color", "identity_colour", "identity_color", "pig_colour", "pig_color"])
start_col = first_existing_col(full, ["window_start", "start_time", "start_seconds", "start_sec", "t_start", "start"])
end_col = first_existing_col(full, ["window_end", "end_time", "end_seconds", "end_sec", "t_end", "end"])
duration_col = first_existing_col(full, ["window_duration", "duration", "duration_seconds"])

for logical, actual in {
    "date": date_col,
    "sheet_name": sheet_col,
}.items():
    if not actual:
        issues.append({
            "item": logical,
            "issue_type": "hard_missing_required_v77c_column",
            "issue_detail": f"Could not detect required v77c column: {logical}",
            "severity": "hard",
        })

# Build truth key. Keep only rows with usable date + sheet.
if not truth.empty:
    truth2 = truth.copy()

    required_truth = ["date", "sheet_name", "tlc_camera", "resolved_room_pen"]
    for c in required_truth:
        if c not in truth2.columns:
            issues.append({
                "item": c,
                "issue_type": "hard_missing_required_sheet_truth_column",
                "issue_detail": f"Missing required sheet truth column: {c}",
                "severity": "hard",
            })

    if all(c in truth2.columns for c in required_truth):
        truth2 = truth2[[
            "date", "excel_file_name", "excel_file", "sheet_name",
            "tlc_camera", "room", "pen_num", "resolved_room_pen",
            "sheet_alias", "sheet_explicit_room_pen",
            "annotation_unit_type", "source_truth_statement"
        ]].drop_duplicates()

        # Some Excel has both TLC 1 B1 and TLC 1 big resolving B1.
        # This is okay; join by exact date + sheet_name, not by pen.
        duplicate_truth_keys = int(truth2[["date", "sheet_name"]].duplicated().sum())
        if duplicate_truth_keys:
            issues.append({
                "item": "sheet_truth_keys",
                "issue_type": "hard_duplicate_date_sheet_truth_keys",
                "issue_detail": f"{duplicate_truth_keys} duplicate date+sheet_name keys in sheet truth inventory.",
                "severity": "hard",
            })
else:
    truth2 = pd.DataFrame()

joined = pd.DataFrame()

if not full.empty and not truth2.empty and date_col and sheet_col:
    joined = full.merge(
        truth2,
        left_on=[date_col, sheet_col],
        right_on=["date", "sheet_name"],
        how="left",
        suffixes=("", "_sheet_truth")
    )

    unmatched = joined[joined["tlc_camera"].astype(str).map(clean) == ""].copy()
    if len(unmatched):
        sample = unmatched[[date_col, sheet_col]].drop_duplicates().head(20).to_dict("records")
        issues.append({
            "item": "v77c_to_sheet_truth_join",
            "issue_type": "hard_unmatched_v77c_rows_to_sheet_truth",
            "issue_detail": f"{len(unmatched)} v77c rows did not join to sheet truth. Sample={sample}",
            "severity": "hard",
        })

    if len(joined) != len(full):
        issues.append({
            "item": "join_row_count",
            "issue_type": "hard_join_changed_row_count",
            "issue_detail": f"Full rows={len(full)}, joined rows={len(joined)}.",
            "severity": "hard",
        })

join_qa = pd.DataFrame([{
    "input_v77c_rows": int(len(full)),
    "joined_rows": int(len(joined)),
    "unmatched_rows": int(len(joined[joined['tlc_camera'].astype(str).map(clean) == ''])) if not joined.empty and "tlc_camera" in joined.columns else -1,
    "expected_full_rows": EXPECTED_FULL_ROWS,
    "input_matches_expected_full_rows": bool(len(full) == EXPECTED_FULL_ROWS),
    "join_key": f"{date_col}+{sheet_col}",
}])
to_csv(join_qa, OUT_JOIN_QA)

if len(full) != EXPECTED_FULL_ROWS:
    issues.append({
        "item": "v77c_full_row_count",
        "issue_type": "warning_unexpected_v77c_full_row_count",
        "issue_detail": f"Expected {EXPECTED_FULL_ROWS} full v77c rows, found {len(full)}.",
        "severity": "warning",
    })

frozen_rows = []

if not joined.empty:
    for i, r in joined.iterrows():
        date = clean(r.get("date", "")) or clean(r.get(date_col, ""))
        source_excel = clean(r.get("excel_file_name", "")) or clean(r.get(excel_file_col, "")) if excel_file_col else clean(r.get("excel_file_name", ""))
        sheet = clean(r.get("sheet_name", "")) or clean(r.get(sheet_col, ""))
        tlc = clean(r.get("tlc_camera", ""))
        room = clean(r.get("room", ""))
        pen_num = clean(r.get("pen_num", ""))
        resolved_room_pen = clean(r.get("resolved_room_pen", ""))

        behaviour = clean(r.get(behaviour_col, "")) if behaviour_col else ""
        colour = clean(r.get(colour_col, "")) if colour_col else ""

        start = clean(r.get(start_col, "")) if start_col else ""
        end = clean(r.get(end_col, "")) if end_col else ""
        duration = clean(r.get(duration_col, "")) if duration_col else ""

        source_truth_statement = clean(r.get("source_truth_statement", ""))
        annotation_unit_type = clean(r.get("annotation_unit_type", ""))

        annotation_window_id = make_id(date, source_excel, sheet, tlc, resolved_room_pen, behaviour, colour, start, end, str(i))

        frozen_rows.append({
            "annotation_window_id": annotation_window_id,
            "source_row_index": i,
            "date": date,
            "source_excel_file": source_excel,
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
            "annotation_source_status": "frozen_from_v77c_full_plus_v78k4_sheet_truth",
            "video_mapping_status": "pending_not_part_of_annotation_truth",
            "spatial_roi_status": "not_required_for_annotation_truth",
            "tracking_status": "not_started",
            "claim_boundary": "corrected_full_annotation_truth_only_no_video_mapping_no_roi_no_tracking",
        })

frozen = pd.DataFrame(frozen_rows)
to_csv(frozen, OUT_FROZEN)

if frozen.empty:
    issues.append({
        "item": "frozen_table",
        "issue_type": "hard_empty_corrected_frozen_annotation_truth_table",
        "issue_detail": "Corrected frozen annotation truth table is empty.",
        "severity": "hard",
    })

if not frozen.empty:
    if len(frozen) != EXPECTED_FULL_ROWS:
        issues.append({
            "item": "corrected_frozen_row_count",
            "issue_type": "hard_corrected_frozen_row_count_not_full",
            "issue_detail": f"Expected {EXPECTED_FULL_ROWS} rows, got {len(frozen)}.",
            "severity": "hard",
        })

    missing_required = frozen[
        (frozen["date"] == "") |
        (frozen["sheet_name"] == "") |
        (frozen["tlc_camera"] == "") |
        (frozen["resolved_room_pen"] == "")
    ]
    if len(missing_required):
        issues.append({
            "item": "required_truth_fields",
            "issue_type": "hard_missing_required_truth_values",
            "issue_detail": f"{len(missing_required)} rows missing date/sheet/tlc/resolved_room_pen.",
            "severity": "hard",
        })

    dup_ids = int(frozen["annotation_window_id"].duplicated().sum())
    if dup_ids:
        issues.append({
            "item": "annotation_window_id",
            "issue_type": "hard_duplicate_annotation_window_ids",
            "issue_detail": f"{dup_ids} duplicate annotation_window_id values.",
            "severity": "hard",
        })

# Outputs
if not frozen.empty:
    summary = pd.DataFrame([{
        "frozen_rows": len(frozen),
        "unique_dates": frozen["date"].nunique(),
        "unique_date_sheet_pairs": frozen[["date", "sheet_name"]].drop_duplicates().shape[0],
        "unique_sheet_names": frozen["sheet_name"].nunique(),
        "unique_tlc_cameras": frozen["tlc_camera"].nunique(),
        "unique_room_pens": frozen["resolved_room_pen"].nunique(),
        "unique_behaviour_labels": frozen["behaviour_label"].nunique(),
        "rows_with_behaviour_label": int((frozen["behaviour_label"] != "").sum()),
        "rows_with_identity_colour": int((frozen["identity_colour"] != "").sum()),
        "video_mapping_status": "pending_not_part_of_annotation_truth",
        "spatial_roi_status": "not_required_for_annotation_truth",
    }])
else:
    summary = pd.DataFrame([{
        "frozen_rows": 0,
        "unique_dates": 0,
        "unique_date_sheet_pairs": 0,
        "unique_sheet_names": 0,
        "unique_tlc_cameras": 0,
        "unique_room_pens": 0,
        "unique_behaviour_labels": 0,
        "rows_with_behaviour_label": 0,
        "rows_with_identity_colour": 0,
        "video_mapping_status": "not_created",
        "spatial_roi_status": "not_created",
    }])
to_csv(summary, OUT_SUMMARY)

if not frozen.empty:
    class_counts = frozen.groupby("behaviour_label").size().reset_index(name="row_count").sort_values("row_count", ascending=False)
    target_counts = frozen.groupby(["date", "tlc_camera", "resolved_room_pen"]).size().reset_index(name="row_count").sort_values(["date", "tlc_camera", "resolved_room_pen"])
else:
    class_counts = pd.DataFrame(columns=["behaviour_label", "row_count"])
    target_counts = pd.DataFrame(columns=["date", "tlc_camera", "resolved_room_pen", "row_count"])

to_csv(class_counts, OUT_CLASS_COUNTS)
to_csv(target_counts, OUT_TARGET_COUNTS)

issues.append({
    "item": "scope",
    "issue_type": "info_corrected_full_frozen_annotation_truth_only",
    "issue_detail": "v78k5b explicitly uses full v77c annotation windows, not v78b unresolved rows. It does not solve video mapping, ROI, or tracking.",
    "severity": "info",
})

issues_df = pd.DataFrame(issues, columns=["item", "issue_type", "issue_detail", "severity"])
to_csv(issues_df, OUT_ISSUES)

hard_count = int((issues_df["severity"] == "hard").sum()) if len(issues_df) else 0
warning_count = int((issues_df["severity"] == "warning").sum()) if len(issues_df) else 0
info_count = int((issues_df["severity"] == "info").sum()) if len(issues_df) else 0

readme = """# v78k5b Corrected Full Frozen Annotation Truth Table

This supersedes v78k5.

Reason:
- v78k5 froze 14,828 rows because it used a v78b unresolved annotation table.
- v78k5b explicitly freezes the full v77c annotation table.
- Expected full row count is 15,733.

Boundary:
- Annotation truth only.
- No encoded c-code mapping.
- No spatial ROI.
- No tracking.
"""
OUT_README.write_text(readme, encoding="utf-8")
OUT_REPORT.write_text(readme, encoding="utf-8")

manifest = {
    "version": "v78k5b_corrected_full_frozen_annotation_truth_table",
    "generated_at": datetime.now().isoformat(timespec="seconds"),
    "input_v77c_full": str(V77C_FULL),
    "expected_full_rows": EXPECTED_FULL_ROWS,
    "input_v77c_rows": int(len(full)),
    "joined_rows": int(len(joined)),
    "frozen_rows": int(len(frozen)),
    "summary": summary.iloc[0].to_dict() if len(summary) else {},
    "detected_columns": {
        "date_col": date_col,
        "excel_file_col": excel_file_col,
        "sheet_col": sheet_col,
        "behaviour_col": behaviour_col,
        "colour_col": colour_col,
        "start_col": start_col,
        "end_col": end_col,
        "duration_col": duration_col,
    },
    "hard_issue_count": hard_count,
    "warning_count": warning_count,
    "claim_boundary": "corrected full frozen annotation truth only",
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
    "v78k5b_decision": "corrected_full_frozen_annotation_truth_table_created" if hard_count == 0 else "corrected_full_frozen_annotation_truth_table_has_blocking_issues",
    "supersedes": "v78k5_frozen_annotation_truth_table",
    "input_v77c_rows": int(len(full)),
    "joined_rows": int(len(joined)),
    "frozen_rows": int(len(frozen)),
    "expected_full_rows": EXPECTED_FULL_ROWS,
    "matches_expected_full_rows": bool(len(frozen) == EXPECTED_FULL_ROWS),
    "unique_dates": int(summary.iloc[0]["unique_dates"]),
    "unique_date_sheet_pairs": int(summary.iloc[0]["unique_date_sheet_pairs"]),
    "unique_sheet_names": int(summary.iloc[0]["unique_sheet_names"]),
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
    "ready_for_annotation_truth_use": bool(hard_count == 0 and len(frozen) == EXPECTED_FULL_ROWS),
    "ready_for_video_mapping": False,
    "ready_for_tracking": False,
    "claim_scope": "corrected_full_frozen_annotation_truth_only",
    "generated_at": datetime.now().isoformat(timespec="seconds"),
}])
to_csv(decision, OUT_DECISION)

OUT_NOTE.write_text(
    "# v78k5b Corrected Full Frozen Annotation Truth Table\n\n"
    f"- Decision: {decision.iloc[0]['v78k5b_decision']}\n"
    "- Supersedes: v78k5_frozen_annotation_truth_table\n"
    f"- Input v77c rows: {len(full)}\n"
    f"- Joined rows: {len(joined)}\n"
    f"- Frozen rows: {len(frozen)}\n"
    f"- Expected full rows: {EXPECTED_FULL_ROWS}\n"
    f"- Matches expected full rows: {bool(len(frozen) == EXPECTED_FULL_ROWS)}\n"
    f"- Hard issues: {hard_count}\n"
    f"- Ready for annotation truth use: {bool(hard_count == 0 and len(frozen) == EXPECTED_FULL_ROWS)}\n\n"
    "This corrected table explicitly uses the full v77c annotation windows. Video mapping, ROI and tracking remain separate downstream tasks.\n",
    encoding="utf-8"
)

progress_row = pd.DataFrame([{
    "date": datetime.now().date().isoformat(),
    "stage": "v78k5b",
    "task_name": "Corrected full frozen annotation truth table",
    "status": "PASS" if hard_count == 0 else "NEEDS_FIX",
    "input_summary": "full v77c annotation windows + v78k4 sheet truth",
    "output_summary": str(PKG),
    "hard_issues": hard_count,
    "warnings": warning_count,
    "next_action": "Use v78k5b as annotation source-of-truth; v78k5 is superseded.",
}])

if OUT_PROGRESS.exists():
    old = pd.read_csv(OUT_PROGRESS)
    progress = pd.concat([old, progress_row], ignore_index=True)
else:
    progress = progress_row
to_csv(progress, OUT_PROGRESS)

print("=== v78k5b decision ===")
print(decision.to_string(index=False))

print("\n=== join QA ===")
print(join_qa.to_string(index=False))

print("\n=== summary ===")
print(summary.to_string(index=False))

print("\n=== target counts sample ===")
print(target_counts.head(60).to_string(index=False))

print("\n=== class counts ===")
print(class_counts.to_string(index=False))

print("\n=== issues ===")
print(issues_df.to_string(index=False))
