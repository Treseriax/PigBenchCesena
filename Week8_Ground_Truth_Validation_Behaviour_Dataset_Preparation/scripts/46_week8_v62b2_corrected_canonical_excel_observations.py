from pathlib import Path
from datetime import datetime
import re
import csv
import pandas as pd


ROOT = Path.home() / "PigBench"
W8 = ROOT / "Week8_Ground_Truth_Validation_Behaviour_Dataset_Preparation"

V45_CLIP_OBJECTS = W8 / "outputs" / "propagated_ground_truth" / "v45_label_propagation" / "week8_v45_clip_object_propagated_annotations.csv"
V62A2_BLOCKS = W8 / "outputs" / "v62a2_canonical_video_time_sheet_lock" / "week8_v62a2_relevant_excel_blocks.csv"
OLD_V62B_OBS = W8 / "outputs" / "v62b_canonical_excel_observation_extraction" / "week8_v62b_canonical_colour_behaviour_observations.csv"

OUT = W8 / "outputs" / "v62b2_corrected_canonical_excel_observation_extraction"
NOTES = W8 / "notes"
REPORTS = W8 / "reports"
PROGRESS = W8 / "progress"

for p in [OUT, NOTES, REPORTS, PROGRESS]:
    p.mkdir(parents=True, exist_ok=True)

OUT_OBSERVATIONS = OUT / "week8_v62b2_corrected_canonical_colour_behaviour_observations.csv"
OUT_COLOUR_ROWS = OUT / "week8_v62b2_corrected_canonical_colour_row_inventory.csv"
OUT_SCANFRAME_SUMMARY = OUT / "week8_v62b2_scanframe_observation_summary.csv"
OUT_QA = OUT / "week8_v62b2_extraction_qa.csv"
OUT_PREVIOUS_COMPARISON = OUT / "week8_v62b2_previous_v62b_comparison.csv"
OUT_DECISION = OUT / "week8_v62b2_decision_summary.csv"
OUT_ISSUES = OUT / "week8_v62b2_issues.csv"
OUT_NOTE = NOTES / "week8_v62b2_corrected_canonical_excel_observation_extraction_notes.md"
OUT_REPORT = REPORTS / "week8_v62b2_corrected_canonical_excel_observation_extraction_report.md"
OUT_PROGRESS = PROGRESS / "week8_experiment_progress_log.csv"


VALID_BEHAVIOUR_CODES = {
    "PI", "SI", "LAI", "STI", "NU", "BE", "DE", "AN", "IN", "IA", "MC", "ARR", "BOX"
}

EXPECTED_COLOURS = ["blue", "green", "no_colour", "purple", "red_neck", "red_tail"]
EXPECTED_ROW_COUNT = 72 * 6


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
    s = clean_str(x).upper().replace(" ", "")
    if not s:
        return ""
    return s


def normalize_colour_label(label):
    s = clean_str(label).lower()
    s = s.replace("_", " ").replace("-", " ")
    s = re.sub(r"\s+", " ", s).strip()

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
    if "#" in s or "senza" in s or "no color" in s or "no colour" in s or "nessun" in s:
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
        "ciano", "cyan",
        "azzurro",
        "giallo", "yellow",
        "arancione", "orange",
        "#", "senza", "no color", "no colour", "nessun",
    ]
    return any(t in s for t in tokens)


def find_colour_label_in_row(df, row_idx):
    # Canonical colour labels sit at the left of the Excel behaviour grid.
    for c in range(min(df.shape[1], 4)):
        val = clean_str(df.iat[row_idx, c])
        if looks_like_colour_label(val):
            return val, c
    return "", ""


def extract_strict_six_colour_rows(df, time_block_row):
    data_start = int(time_block_row) + 3
    rows = []
    seen = set()

    # Strictly read from the colour block immediately under the time header.
    # Stop after six unique canonical colour labels to prevent leaking into the next time block.
    for r in range(data_start, min(df.shape[0], data_start + 10)):
        label, label_col = find_colour_label_in_row(df, r)
        if not label:
            # Once at least one colour was found, a blank/non-colour row means the local colour block ended.
            if rows:
                break
            continue

        norm = normalize_colour_label(label)
        if norm in seen:
            continue

        rows.append({
            "excel_row_index_0based": r,
            "colour_label_col_index_0based": label_col,
            "canonical_colour_label_raw": label,
            "canonical_colour_label_norm": norm,
        })
        seen.add(norm)

        if len(rows) == 6:
            break

    return rows


def get_context(df, row_idx, col_idx):
    parts = []
    for rr in range(max(0, row_idx - 1), min(df.shape[0], row_idx + 2)):
        vals = []
        for cc in range(max(0, col_idx - 2), min(df.shape[1], col_idx + 3)):
            vals.append(clean_str(df.iat[rr, cc]))
        parts.append(f"r{rr}: " + " | ".join(vals))
    return " || ".join(parts)


issues = []

for p in [V45_CLIP_OBJECTS, V62A2_BLOCKS]:
    if not p.exists():
        issues.append({
            "item": str(p),
            "issue_type": "hard_missing_required_input",
            "issue_detail": "Required input is missing.",
            "severity": "hard",
        })

if issues:
    issues_df = pd.DataFrame(issues)
    safe_to_csv(issues_df, OUT_ISSUES)
    decision = pd.DataFrame([{
        "v62b2_decision": "corrected_canonical_extraction_blocked",
        "hard_issue_count": int((issues_df["severity"] == "hard").sum()),
        "ready_for_v63_canonical_gt_v2_schema": False,
        "generated_at": datetime.now().isoformat(timespec="seconds"),
    }])
    safe_to_csv(decision, OUT_DECISION)
    print(decision.to_string(index=False))
    raise SystemExit(1)


clip_objects = pd.read_csv(V45_CLIP_OBJECTS)
blocks = pd.read_csv(V62A2_BLOCKS)

scanframe_map = {}
for video_id, g in clip_objects.groupby("video_id"):
    scanframes = sorted(g["scan_frame_id"].astype(str).unique().tolist(), key=scan_num)
    scanframe_map[str(video_id)] = scanframes

sheet_cache = {}
observations = []
colour_row_inventory = []
qa_rows = []

for _, b in blocks.iterrows():
    video_id = clean_str(b["video_id"])
    excel_path = Path(clean_str(b["canonical_excel_path"]))
    sheet = clean_str(b["canonical_sheet"])

    if not excel_path.exists():
        issues.append({
            "item": str(excel_path),
            "issue_type": "hard_missing_excel_file",
            "issue_detail": "Canonical Excel file is missing.",
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

    time_row = int(b["time_block_row"])
    time_col = int(b["time_block_col"])
    period_row = time_row + 1

    colour_rows = extract_strict_six_colour_rows(df, time_row)
    found_colours = sorted([r["canonical_colour_label_norm"] for r in colour_rows])

    scanframes = scanframe_map.get(video_id, [])

    qa_status = "pass"
    qa_detail = ""

    if len(scanframes) != 6:
        qa_status = "warning"
        qa_detail += f"Expected 6 scanframes, found {len(scanframes)}. "

    if len(colour_rows) != 6:
        qa_status = "hard"
        qa_detail += f"Expected 6 colour rows, found {len(colour_rows)}. "

    if sorted(EXPECTED_COLOURS) != found_colours:
        qa_status = "hard"
        qa_detail += f"Expected colours {sorted(EXPECTED_COLOURS)}, found {found_colours}. "

    offsets = []
    for k in range(6):
        c = time_col + k
        offsets.append(parse_int_like(df.iat[period_row, c]) if c < df.shape[1] else None)

    if offsets != [0, 10, 20, 30, 40, 50]:
        qa_status = "warning" if qa_status == "pass" else qa_status
        qa_detail += f"Unexpected offsets {offsets}. "

    qa_rows.append({
        "video_id": video_id,
        "start_hour": clean_str(b["start_hour"]),
        "time_block_row": time_row,
        "time_block_col": time_col,
        "scanframe_count": len(scanframes),
        "strict_colour_row_count": len(colour_rows),
        "strict_colour_labels": ";".join(found_colours),
        "offsets": ";".join([str(x) for x in offsets]),
        "qa_status": qa_status,
        "qa_detail": qa_detail.strip(),
    })

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

    for k in range(min(6, len(scanframes))):
        scan_frame_id = scanframes[k]
        behaviour_col = time_col + k
        offset_min = offsets[k] if k < len(offsets) else None

        for cr in colour_rows:
            r = cr["excel_row_index_0based"]
            raw_beh = clean_str(df.iat[r, behaviour_col]) if behaviour_col < df.shape[1] else ""
            beh = normalize_behaviour(raw_beh)

            observations.append({
                "dataset_version": "canonical_excel_v62b2_corrected",
                "scan_frame_id": scan_frame_id,
                "video_id": video_id,
                "source_hour_start": clean_str(b["start_hour"]),
                "observation_offset_min": offset_min,
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
                "source_status": "strict_extracted_from_original_excel",
            })


obs_df = pd.DataFrame(observations)
colour_rows_df = pd.DataFrame(colour_row_inventory)
qa_df = pd.DataFrame(qa_rows)

safe_to_csv(obs_df, OUT_OBSERVATIONS)
safe_to_csv(colour_rows_df, OUT_COLOUR_ROWS)
safe_to_csv(qa_df, OUT_QA)

if len(obs_df):
    scan_summary = (
        obs_df.groupby("scan_frame_id")
        .agg(
            video_id=("video_id", "first"),
            canonical_observation_rows=("canonical_colour_label_norm", "count"),
            canonical_colour_count=("canonical_colour_label_norm", "nunique"),
            behaviour_code_count=("behaviour_code", lambda x: int((x.astype(str) != "").sum())),
            missing_behaviour_count=("behaviour_code", lambda x: int((x.astype(str) == "").sum())),
            colour_labels=("canonical_colour_label_norm", lambda x: ";".join(sorted(set(x.astype(str))))),
            behaviour_codes=("behaviour_code", lambda x: ";".join([v for v in x.astype(str).tolist() if v])),
        )
        .reset_index()
        .sort_values("scan_frame_id", key=lambda s: s.map(scan_num))
    )
else:
    scan_summary = pd.DataFrame()

safe_to_csv(scan_summary, OUT_SCANFRAME_SUMMARY)

old_rows = ""
row_delta = ""
if OLD_V62B_OBS.exists():
    try:
        old = pd.read_csv(OLD_V62B_OBS)
        old_rows = len(old)
        row_delta = len(obs_df) - len(old)
    except Exception:
        old_rows = "read_error"
        row_delta = ""

comparison = pd.DataFrame([{
    "old_v62b_rows": old_rows,
    "corrected_v62b2_rows": len(obs_df),
    "row_delta_corrected_minus_old": row_delta,
    "expected_rows": EXPECTED_ROW_COUNT,
    "correction_reason": "v62b sometimes leaked the next time-block first colour row; v62b2 restricts extraction to exactly six canonical colour rows per hour block.",
}])
safe_to_csv(comparison, OUT_PREVIOUS_COMPARISON)

# Final issues.
if len(obs_df) != EXPECTED_ROW_COUNT:
    issues.append({
        "item": "corrected_observations",
        "issue_type": "hard_unexpected_total_row_count",
        "issue_detail": f"Expected {EXPECTED_ROW_COUNT} rows, found {len(obs_df)}.",
        "severity": "hard",
    })

if len(scan_summary) != 72:
    issues.append({
        "item": "scanframe_summary",
        "issue_type": "hard_unexpected_scanframe_count",
        "issue_detail": f"Expected 72 scanframes, found {len(scan_summary)}.",
        "severity": "hard",
    })

if len(scan_summary) and not (scan_summary["canonical_observation_rows"] == 6).all():
    bad = scan_summary[scan_summary["canonical_observation_rows"] != 6]
    issues.append({
        "item": "scanframe_summary",
        "issue_type": "hard_not_all_scanframes_have_exactly_six_rows",
        "issue_detail": f"{len(bad)} scanframes do not have exactly six canonical observations.",
        "severity": "hard",
    })

if len(obs_df):
    extracted_colours = sorted(obs_df["canonical_colour_label_norm"].unique().tolist())
    if extracted_colours != sorted(EXPECTED_COLOURS):
        issues.append({
            "item": "canonical_colour_labels",
            "issue_type": "hard_unexpected_colour_vocabulary",
            "issue_detail": f"Expected {sorted(EXPECTED_COLOURS)}, found {extracted_colours}.",
            "severity": "hard",
        })

    unknown_beh = obs_df[
        (obs_df["behaviour_code"].astype(str) != "")
        & (~obs_df["behaviour_code"].isin(VALID_BEHAVIOUR_CODES))
    ]
    if len(unknown_beh):
        issues.append({
            "item": "behaviour_codes",
            "issue_type": "warning_unknown_behaviour_codes",
            "issue_detail": f"{len(unknown_beh)} observations contain unknown behaviour codes.",
            "severity": "warning",
        })

qa_hard = qa_df[qa_df["qa_status"] == "hard"] if len(qa_df) else pd.DataFrame()
if len(qa_hard):
    issues.append({
        "item": "block_qa",
        "issue_type": "hard_block_level_qa_failed",
        "issue_detail": f"{len(qa_hard)} time blocks failed strict extraction QA.",
        "severity": "hard",
    })

issues_df = pd.DataFrame(issues, columns=["item", "issue_type", "issue_detail", "severity"])
safe_to_csv(issues_df, OUT_ISSUES)

hard_issue_count = int((issues_df["severity"] == "hard").sum()) if len(issues_df) else 0
warning_count = int((issues_df["severity"] == "warning").sum()) if len(issues_df) else 0

decision = pd.DataFrame([{
    "v62b2_decision": "corrected_canonical_excel_observation_extraction_completed" if hard_issue_count == 0 else "corrected_canonical_excel_observation_extraction_blocked",
    "corrected_canonical_observation_rows": int(len(obs_df)),
    "expected_canonical_observation_rows": EXPECTED_ROW_COUNT,
    "scanframe_count": int(len(scan_summary)),
    "all_scanframes_exactly_six_observations": bool(len(scan_summary) == 72 and (scan_summary["canonical_observation_rows"] == 6).all()),
    "canonical_colour_labels": ";".join(sorted(obs_df["canonical_colour_label_norm"].unique())) if len(obs_df) else "",
    "old_v62b_rows": old_rows,
    "row_delta_corrected_minus_old": row_delta,
    "hard_issue_count": hard_issue_count,
    "warning_count": warning_count,
    "issue_count": int(len(issues_df)),
    "ready_for_v63_canonical_gt_v2_schema": bool(hard_issue_count == 0 and len(obs_df) == EXPECTED_ROW_COUNT),
    "generated_at": datetime.now().isoformat(timespec="seconds"),
}])
safe_to_csv(decision, OUT_DECISION)

OUT_NOTE.write_text(
    "# Week 8 v62b2 Corrected Canonical Excel Observation Extraction\n\n"
    f"- v62b2 decision: {decision.iloc[0]['v62b2_decision']}\n"
    f"- Corrected canonical observation rows: {len(obs_df)}\n"
    f"- Expected rows: {EXPECTED_ROW_COUNT}\n"
    f"- Scanframes: {len(scan_summary)}\n"
    f"- All scanframes exactly six observations: {decision.iloc[0]['all_scanframes_exactly_six_observations']}\n"
    f"- Canonical colour labels: {decision.iloc[0]['canonical_colour_labels']}\n"
    f"- Old v62b rows: {old_rows}\n"
    f"- Row delta corrected-minus-old: {row_delta}\n"
    f"- Hard issues: {hard_issue_count}\n"
    f"- Warnings: {warning_count}\n"
    f"- Ready for v63 canonical GT v2 schema: {decision.iloc[0]['ready_for_v63_canonical_gt_v2_schema']}\n\n"
    "Correction: v62b2 restricts extraction to exactly six canonical colour rows per time block, preventing leakage from the next Excel time block.\n"
)

OUT_REPORT.write_text(
    "# Week 8 v62b2 Corrected Canonical Excel Observation Extraction Report\n\n"
    f"Decision: {decision.iloc[0]['v62b2_decision']}\n\n"
    f"Corrected observations: `{OUT_OBSERVATIONS}`\n\n"
    f"Scanframe summary: `{OUT_SCANFRAME_SUMMARY}`\n\n"
    f"Extraction QA: `{OUT_QA}`\n\n"
    f"Old-vs-corrected comparison: `{OUT_PREVIOUS_COMPARISON}`\n"
)

progress_row = pd.DataFrame([{
    "date": datetime.now().date().isoformat(),
    "stage": "v62b2",
    "task_name": "Corrected canonical Excel observation extraction",
    "status": "PASS" if hard_issue_count == 0 else "BLOCKED",
    "input_summary": str(V62A2_BLOCKS),
    "output_summary": str(OUT),
    "hard_issues": hard_issue_count,
    "warnings": warning_count,
    "next_action": "v63 canonical GT v2 schema and annotation-assisted visual correction interface.",
}])

if OUT_PROGRESS.exists():
    old_progress = pd.read_csv(OUT_PROGRESS)
    progress = pd.concat([old_progress, progress_row], ignore_index=True)
else:
    progress = progress_row

safe_to_csv(progress, OUT_PROGRESS)

print("Saved:")
print(OUT_OBSERVATIONS)
print(OUT_COLOUR_ROWS)
print(OUT_SCANFRAME_SUMMARY)
print(OUT_QA)
print(OUT_PREVIOUS_COMPARISON)
print(OUT_DECISION)
print(OUT_ISSUES)
print(OUT_NOTE)

print()
print("=== v62b2 decision ===")
print(decision.to_string(index=False))

print()
print("=== v62b2 QA ===")
print(qa_df.to_string(index=False))

print()
print("=== v62b2 scanframe summary head ===")
print(scan_summary.head(20).to_string(index=False))

print()
print("=== old vs corrected ===")
print(comparison.to_string(index=False))

print()
print("=== issues ===")
if len(issues_df):
    print(issues_df.to_string(index=False))
else:
    print("No issues found.")
