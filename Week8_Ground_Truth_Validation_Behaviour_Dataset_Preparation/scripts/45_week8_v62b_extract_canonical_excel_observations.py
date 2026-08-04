from pathlib import Path
from datetime import datetime
import re
import csv
import pandas as pd


ROOT = Path.home() / "PigBench"
W8 = ROOT / "Week8_Ground_Truth_Validation_Behaviour_Dataset_Preparation"

V45_CLIP_OBJECTS = W8 / "outputs" / "propagated_ground_truth" / "v45_label_propagation" / "week8_v45_clip_object_propagated_annotations.csv"
V62A2_VIDEO_LOCK = W8 / "outputs" / "v62a2_canonical_video_time_sheet_lock" / "week8_v62a2_video_time_sheet_lock.csv"
V62A2_BLOCKS = W8 / "outputs" / "v62a2_canonical_video_time_sheet_lock" / "week8_v62a2_relevant_excel_blocks.csv"

OUT = W8 / "outputs" / "v62b_canonical_excel_observation_extraction"
NOTES = W8 / "notes"
REPORTS = W8 / "reports"
PROGRESS = W8 / "progress"

for p in [OUT, NOTES, REPORTS, PROGRESS]:
    p.mkdir(parents=True, exist_ok=True)

OUT_OBSERVATIONS = OUT / "week8_v62b_canonical_colour_behaviour_observations.csv"
OUT_COLOUR_ROWS = OUT / "week8_v62b_canonical_colour_row_inventory.csv"
OUT_SCANFRAME_SUMMARY = OUT / "week8_v62b_scanframe_observation_summary.csv"
OUT_COMPARISON_PREVIOUS = OUT / "week8_v62b_previous_vs_canonical_colour_summary.csv"
OUT_DECISION = OUT / "week8_v62b_decision_summary.csv"
OUT_ISSUES = OUT / "week8_v62b_issues.csv"
OUT_NOTE = NOTES / "week8_v62b_canonical_excel_observation_extraction_notes.md"
OUT_REPORT = REPORTS / "week8_v62b_canonical_excel_observation_extraction_report.md"
OUT_PROGRESS = PROGRESS / "week8_experiment_progress_log.csv"


VALID_BEHAVIOUR_CODES = {
    "PI", "SI", "LAI", "STI", "NU", "BE", "DE", "AN", "IN", "IA", "MC", "ARR", "BOX"
}


def safe_to_csv(df, path):
    df.to_csv(path, index=False, quoting=csv.QUOTE_ALL, escapechar="\\", lineterminator="\n")


def clean_str(x):
    if pd.isna(x):
        return ""
    s = str(x).strip()
    if s.lower() in ["nan", "none", "null"]:
        return ""
    return s


def scan_num(s):
    m = re.search(r"(\d+)$", clean_str(s))
    return int(m.group(1)) if m else 10**9


def parse_int_like(x):
    s = clean_str(x)
    if not s:
        return None
    s = s.replace(",", ".")
    try:
        return int(float(s))
    except Exception:
        return None


def normalize_behaviour(x):
    s = clean_str(x).upper()
    s = s.replace(" ", "")
    if not s:
        return ""
    if s in VALID_BEHAVIOUR_CODES:
        return s
    return s


def normalize_colour_label(label):
    s = clean_str(label).lower()
    s = s.replace("_", " ").replace("-", " ")
    s = re.sub(r"\s+", " ", s).strip()

    if not s:
        return ""

    if "verde" in s or "green" in s:
        return "green"
    if "blu" in s or "blue" in s:
        return "blue"
    if "viola" in s or "purple" in s:
        return "purple"

    if ("rosso" in s or "red" in s) and ("testa" in s or "neck" in s or "head" in s):
        return "red_neck"
    if ("rosso" in s or "red" in s) and ("coda" in s or "tail" in s):
        return "red_tail"

    if "rosa" in s or "pink" in s:
        return "pink"
    if "ciano" in s or "cyan" in s or "azzurro" in s:
        return "cyan"
    if "giallo" in s or "yellow" in s:
        return "yellow"
    if "arancione" in s or "orange" in s:
        return "orange"
    if "senza" in s or "no color" in s or "no colour" in s or "nessun" in s:
        return "no_colour"

    return s.replace(" ", "_")


def looks_like_colour_label(label):
    s = clean_str(label).lower()
    tokens = [
        "verde", "green",
        "blu", "blue",
        "viola", "purple",
        "rosso", "red",
        "coda", "tail",
        "testa", "neck",
        "rosa", "pink",
        "ciano", "cyan", "azzurro",
        "giallo", "yellow",
        "arancione", "orange",
        "senza", "no color", "no colour", "nessun",
    ]
    return any(t in s for t in tokens)


def find_colour_label_in_row(df, row_idx):
    # Canonical colour labels are expected near the left side of the sheet.
    # We deliberately avoid scanning far-right summary tables.
    max_col = min(df.shape[1], 8)
    for c in range(max_col):
        val = clean_str(df.iat[row_idx, c])
        if looks_like_colour_label(val):
            return val, c
    return "", ""


def extract_colour_rows(df, time_block_row):
    data_start = int(time_block_row) + 3
    rows = []

    for r in range(data_start, min(df.shape[0], data_start + 12)):
        label, label_col = find_colour_label_in_row(df, r)
        if not label:
            continue

        rows.append({
            "excel_row_index_0based": r,
            "colour_label_col_index_0based": label_col,
            "canonical_colour_label_raw": label,
            "canonical_colour_label_norm": normalize_colour_label(label),
        })

    return rows


def get_context(df, row_idx, col_idx, radius_rows=1, radius_cols=2):
    parts = []
    for rr in range(max(0, row_idx - radius_rows), min(df.shape[0], row_idx + radius_rows + 1)):
        vals = []
        for cc in range(max(0, col_idx - radius_cols), min(df.shape[1], col_idx + radius_cols + 1)):
            vals.append(clean_str(df.iat[rr, cc]))
        parts.append(f"r{rr}: " + " | ".join(vals))
    return " || ".join(parts)


issues = []

for p in [V45_CLIP_OBJECTS, V62A2_VIDEO_LOCK, V62A2_BLOCKS]:
    if not p.exists():
        issues.append({
            "item": str(p),
            "issue_type": "hard_missing_required_input",
            "issue_detail": "Required v62b input is missing.",
            "severity": "hard",
        })

if issues:
    issues_df = pd.DataFrame(issues)
    safe_to_csv(issues_df, OUT_ISSUES)
    decision = pd.DataFrame([{
        "v62b_decision": "canonical_excel_observation_extraction_blocked",
        "hard_issue_count": int((issues_df["severity"] == "hard").sum()),
        "ready_for_v63_canonical_gt_v2_schema": False,
        "generated_at": datetime.now().isoformat(timespec="seconds"),
    }])
    safe_to_csv(decision, OUT_DECISION)
    print(decision.to_string(index=False))
    raise SystemExit(1)


clip_objects = pd.read_csv(V45_CLIP_OBJECTS)
video_lock = pd.read_csv(V62A2_VIDEO_LOCK)
blocks = pd.read_csv(V62A2_BLOCKS)

# Map each video to its six scanframes in chronological/annotation order.
scanframe_map = {}
for video_id, g in clip_objects.groupby("video_id"):
    scanframes = sorted(g["scan_frame_id"].astype(str).unique().tolist(), key=scan_num)
    scanframe_map[str(video_id)] = scanframes

observations = []
colour_row_inventory = []

# Cache Excel sheets.
sheet_cache = {}

for _, b in blocks.iterrows():
    video_id = clean_str(b["video_id"])
    excel_path = Path(clean_str(b["canonical_excel_path"]))
    sheet = clean_str(b["canonical_sheet"])
    matched = str(b["matched_time_block"]).lower() == "true"

    if not matched:
        issues.append({
            "item": video_id,
            "issue_type": "hard_unmatched_time_block",
            "issue_detail": "Video does not have matched Excel time block.",
            "severity": "hard",
        })
        continue

    if not excel_path.exists():
        issues.append({
            "item": str(excel_path),
            "issue_type": "hard_missing_excel_file",
            "issue_detail": "Canonical Excel file missing.",
            "severity": "hard",
        })
        continue

    key = (str(excel_path), sheet)
    if key not in sheet_cache:
        try:
            sheet_cache[key] = pd.read_excel(excel_path, sheet_name=sheet, header=None, dtype=object)
        except Exception as e:
            issues.append({
                "item": f"{excel_path}::{sheet}",
                "issue_type": "hard_excel_read_error",
                "issue_detail": str(e)[:500],
                "severity": "hard",
            })
            continue

    df = sheet_cache[key]

    try:
        time_row = int(b["time_block_row"])
        time_col = int(b["time_block_col"])
    except Exception:
        issues.append({
            "item": video_id,
            "issue_type": "hard_invalid_time_block_coordinates",
            "issue_detail": "time_block_row/time_block_col could not be parsed.",
            "severity": "hard",
        })
        continue

    period_row = time_row + 1
    colour_rows = extract_colour_rows(df, time_row)

    for cr in colour_rows:
        colour_row_inventory.append({
            "video_id": video_id,
            "start_hour": clean_str(b["start_hour"]),
            "canonical_excel_path": str(excel_path),
            "canonical_sheet": sheet,
            "time_block_row": time_row,
            "time_block_col": time_col,
            **cr,
        })

    scanframes = scanframe_map.get(video_id, [])
    if len(scanframes) != 6:
        issues.append({
            "item": video_id,
            "issue_type": "warning_unexpected_scanframe_count",
            "issue_detail": f"Expected 6 scanframes for hourly block, found {len(scanframes)}.",
            "severity": "warning",
        })

    offsets = []
    for k in range(6):
        c = time_col + k
        offset_val = parse_int_like(df.iat[period_row, c]) if c < df.shape[1] else None
        offsets.append(offset_val)

    expected_offsets = [0, 10, 20, 30, 40, 50]
    if offsets != expected_offsets:
        issues.append({
            "item": video_id,
            "issue_type": "warning_observation_offsets_not_standard",
            "issue_detail": f"Expected offsets {expected_offsets}, found {offsets}.",
            "severity": "warning",
        })

    if len(colour_rows) < 6:
        issues.append({
            "item": video_id,
            "issue_type": "warning_less_than_six_colour_rows_detected",
            "issue_detail": f"Detected {len(colour_rows)} colour rows for this time block.",
            "severity": "warning",
        })

    for k in range(min(6, len(scanframes))):
        scan_frame_id = scanframes[k]
        obs_offset_min = offsets[k] if k < len(offsets) else expected_offsets[k]
        behaviour_col = time_col + k

        for cr in colour_rows:
            r = cr["excel_row_index_0based"]
            raw_beh = clean_str(df.iat[r, behaviour_col]) if behaviour_col < df.shape[1] else ""
            beh = normalize_behaviour(raw_beh)

            observations.append({
                "dataset_version": "canonical_excel_v62b",
                "scan_frame_id": scan_frame_id,
                "video_id": video_id,
                "source_hour_start": clean_str(b["start_hour"]),
                "observation_offset_min": obs_offset_min,
                "canonical_colour_label_raw": cr["canonical_colour_label_raw"],
                "canonical_colour_label_norm": cr["canonical_colour_label_norm"],
                "behaviour_code": beh,
                "behaviour_raw": raw_beh,
                "has_behaviour_code": bool(beh),
                "behaviour_code_is_known": bool(beh in VALID_BEHAVIOUR_CODES) if beh else False,
                "canonical_excel_path": str(excel_path),
                "canonical_sheet": sheet,
                "excel_time_block_row_0based": time_row,
                "excel_time_block_col_0based": time_col,
                "excel_period_row_0based": period_row,
                "excel_behaviour_row_0based": r,
                "excel_behaviour_col_0based": behaviour_col,
                "excel_cell_context": get_context(df, r, behaviour_col),
                "source_status": "extracted_from_original_excel",
            })


obs_df = pd.DataFrame(observations)
colour_rows_df = pd.DataFrame(colour_row_inventory)

safe_to_csv(obs_df, OUT_OBSERVATIONS)
safe_to_csv(colour_rows_df, OUT_COLOUR_ROWS)

if len(obs_df):
    scan_summary = (
        obs_df.groupby("scan_frame_id")
        .agg(
            video_id=("video_id", "first"),
            canonical_colour_count=("canonical_colour_label_norm", "nunique"),
            canonical_observation_rows=("canonical_colour_label_norm", "count"),
            behaviour_code_count=("behaviour_code", lambda x: int((x.astype(str) != "").sum())),
            missing_behaviour_count=("behaviour_code", lambda x: int((x.astype(str) == "").sum())),
            colour_labels=("canonical_colour_label_norm", lambda x: ";".join(sorted(set(x.astype(str))))),
            behaviour_codes=("behaviour_code", lambda x: ";".join([v for v in x.astype(str).tolist() if v])),
        )
        .reset_index()
        .sort_values("scan_frame_id", key=lambda s: s.map(scan_num))
    )
else:
    scan_summary = pd.DataFrame(columns=[
        "scan_frame_id", "video_id", "canonical_colour_count",
        "canonical_observation_rows", "behaviour_code_count",
        "missing_behaviour_count",
        "colour_labels", "behaviour_codes",
    ])

safe_to_csv(scan_summary, OUT_SCANFRAME_SUMMARY)

# Previous-vs-canonical only at colour vocabulary level for now.
prev_colour_col = "visual_marker_colour" if "visual_marker_colour" in clip_objects.columns else None
if prev_colour_col and len(obs_df):
    prev_colours = sorted(set(clip_objects[prev_colour_col].astype(str).map(clean_str)))
    canon_colours = sorted(set(obs_df["canonical_colour_label_norm"].astype(str).map(clean_str)))
    comparison = pd.DataFrame([
        {
            "summary_type": "previous_v45_visual_marker_colours",
            "values": ";".join(prev_colours),
            "count": len(prev_colours),
        },
        {
            "summary_type": "canonical_excel_colour_labels",
            "values": ";".join(canon_colours),
            "count": len(canon_colours),
        },
    ])
else:
    comparison = pd.DataFrame(columns=["summary_type", "values", "count"])

safe_to_csv(comparison, OUT_COMPARISON_PREVIOUS)

# Additional QA issues.
if len(obs_df):
    unknown_beh = obs_df[
        (obs_df["behaviour_code"].astype(str) != "")
        & (obs_df["behaviour_code_is_known"] == False)
    ]
    if len(unknown_beh):
        issues.append({
            "item": "canonical_observations",
            "issue_type": "warning_unknown_behaviour_codes_found",
            "issue_detail": f"{len(unknown_beh)} extracted observations have unknown behaviour codes.",
            "severity": "warning",
        })

    incomplete_scanframes = scan_summary[scan_summary["canonical_colour_count"] < 6]
    if len(incomplete_scanframes):
        issues.append({
            "item": "scanframe_summary",
            "issue_type": "warning_some_scanframes_have_less_than_six_colours",
            "issue_detail": f"{len(incomplete_scanframes)} scanframes have less than six canonical colour labels.",
            "severity": "warning",
        })

    expected_min_rows = 72 * 5
    if len(obs_df) < expected_min_rows:
        issues.append({
            "item": "canonical_observations",
            "issue_type": "hard_too_few_canonical_observations",
            "issue_detail": f"Only {len(obs_df)} observations extracted; expected at least {expected_min_rows}.",
            "severity": "hard",
        })
else:
    issues.append({
        "item": "canonical_observations",
        "issue_type": "hard_no_observations_extracted",
        "issue_detail": "No canonical observations were extracted.",
        "severity": "hard",
    })


issues_df = pd.DataFrame(issues, columns=["item", "issue_type", "issue_detail", "severity"])
safe_to_csv(issues_df, OUT_ISSUES)

hard_issue_count = int((issues_df["severity"] == "hard").sum()) if len(issues_df) else 0
warning_count = int((issues_df["severity"] == "warning").sum()) if len(issues_df) else 0

scanframe_count = int(obs_df["scan_frame_id"].nunique()) if len(obs_df) else 0
colour_label_count = int(obs_df["canonical_colour_label_norm"].nunique()) if len(obs_df) else 0
observation_count = int(len(obs_df))
all_scanframes_have_six = bool(len(scan_summary) and (scan_summary["canonical_colour_count"] >= 6).all())

decision = pd.DataFrame([{
    "v62b_decision": "canonical_excel_observation_extraction_completed" if hard_issue_count == 0 else "canonical_excel_observation_extraction_blocked",
    "canonical_observation_rows": observation_count,
    "scanframe_count": scanframe_count,
    "canonical_colour_label_count": colour_label_count,
    "all_scanframes_have_at_least_six_colours": all_scanframes_have_six,
    "canonical_colour_labels": ";".join(sorted(obs_df["canonical_colour_label_norm"].unique())) if len(obs_df) else "",
    "hard_issue_count": hard_issue_count,
    "warning_count": warning_count,
    "issue_count": int(len(issues_df)),
    "ready_for_v63_canonical_gt_v2_schema": bool(hard_issue_count == 0),
    "generated_at": datetime.now().isoformat(timespec="seconds"),
}])

safe_to_csv(decision, OUT_DECISION)

OUT_NOTE.write_text(
    "# Week 8 v62b Canonical Excel Observation Extraction\n\n"
    f"- v62b decision: {decision.iloc[0]['v62b_decision']}\n"
    f"- Canonical observation rows: {observation_count}\n"
    f"- Scanframes covered: {scanframe_count}\n"
    f"- Canonical colour label count: {colour_label_count}\n"
    f"- Canonical colour labels: {decision.iloc[0]['canonical_colour_labels']}\n"
    f"- All scanframes have at least six colours: {all_scanframes_have_six}\n"
    f"- Hard issues: {hard_issue_count}\n"
    f"- Warnings: {warning_count}\n"
    f"- Ready for v63 canonical GT v2 schema: {bool(hard_issue_count == 0)}\n\n"
    "Important: this extraction comes from the original annotation Excel file. It replaces previous guessed visual colour mappings as the canonical colour/behaviour source.\n"
)

OUT_REPORT.write_text(
    "# Week 8 v62b Canonical Excel Observation Extraction Report\n\n"
    f"Decision: {decision.iloc[0]['v62b_decision']}\n\n"
    f"Canonical observations: `{OUT_OBSERVATIONS}`\n\n"
    f"Colour row inventory: `{OUT_COLOUR_ROWS}`\n\n"
    f"Scanframe summary: `{OUT_SCANFRAME_SUMMARY}`\n\n"
    f"Previous-vs-canonical colour summary: `{OUT_COMPARISON_PREVIOUS}`\n"
)

progress_row = pd.DataFrame([{
    "date": datetime.now().date().isoformat(),
    "stage": "v62b",
    "task_name": "Canonical Excel observation extraction",
    "status": "PASS" if hard_issue_count == 0 else "BLOCKED",
    "input_summary": str(V62A2_BLOCKS),
    "output_summary": str(OUT),
    "hard_issues": hard_issue_count,
    "warnings": warning_count,
    "next_action": "v63 canonical GT v2 schema and visual validation interface update.",
}])

if OUT_PROGRESS.exists():
    old = pd.read_csv(OUT_PROGRESS)
    progress = pd.concat([old, progress_row], ignore_index=True)
else:
    progress = progress_row

safe_to_csv(progress, OUT_PROGRESS)

print("Saved:")
print(OUT_OBSERVATIONS)
print(OUT_COLOUR_ROWS)
print(OUT_SCANFRAME_SUMMARY)
print(OUT_COMPARISON_PREVIOUS)
print(OUT_DECISION)
print(OUT_ISSUES)
print(OUT_NOTE)

print()
print("=== v62b decision ===")
print(decision.to_string(index=False))

print()
print("=== canonical colour rows ===")
print(colour_rows_df.drop_duplicates([
    "canonical_excel_path",
    "canonical_sheet",
    "excel_row_index_0based",
    "canonical_colour_label_norm",
])[[
    "canonical_sheet",
    "excel_row_index_0based",
    "canonical_colour_label_raw",
    "canonical_colour_label_norm",
]].head(30).to_string(index=False))

print()
print("=== scanframe summary head ===")
print(scan_summary.head(20).to_string(index=False))

print()
print("=== previous vs canonical colour summary ===")
print(comparison.to_string(index=False))

print()
print("=== issues ===")
if len(issues_df):
    print(issues_df.to_string(index=False))
else:
    print("No issues found.")
