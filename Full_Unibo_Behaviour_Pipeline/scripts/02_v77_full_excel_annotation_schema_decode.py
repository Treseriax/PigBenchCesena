from pathlib import Path
from datetime import datetime
from collections import Counter
import re
import csv
import json
import hashlib
import zipfile
import pandas as pd
import openpyxl


ROOT = Path.home() / "PigBench"
UNIBO = Path("/work/pig/datasets/Unibo")
FULL = ROOT / "Full_Unibo_Behaviour_Pipeline"

V76 = FULL / "outputs" / "v76_full_annotation_video_mapping_audit"
V76_PKG = V76 / "Full_Unibo_Annotation_Video_Mapping_Audit"
V76_DECISION = V76 / "v76_decision_summary.csv"
V76_SHEETS = V76_PKG / "v76_all_excel_sheets_inventory.csv"
V76_VIDEOS = V76_PKG / "v76_all_videos_inventory.csv"

OUT = FULL / "outputs" / "v77_full_excel_annotation_schema_decode"
PKG = OUT / "Full_Unibo_Excel_Annotation_Schema_Decode"
NOTES = FULL / "notes"
REPORTS = FULL / "reports"
PROGRESS = FULL / "progress"

for p in [OUT, PKG, NOTES, REPORTS, PROGRESS]:
    p.mkdir(parents=True, exist_ok=True)

OUT_SHEET_SCHEMA = PKG / "v77_sheet_schema_summary.csv"
OUT_TIME_SLOTS = PKG / "v77_extracted_time_slots.csv"
OUT_OBS_PERIODS = PKG / "v77_extracted_observation_periods.csv"
OUT_IDENTITY_ROWS = PKG / "v77_candidate_identity_rows.csv"
OUT_NORMALIZED = PKG / "v77_normalized_annotation_candidates.csv"
OUT_BEHAV_DIST = PKG / "v77_behaviour_distribution.csv"
OUT_VIDEO_MATCH = PKG / "v77_time_slot_video_candidates.csv"
OUT_NEEDS_REVIEW = PKG / "v77_annotation_rows_needing_review.csv"
OUT_QA = PKG / "v77_quality_checks.csv"
OUT_README = PKG / "README_v77_Full_Excel_Annotation_Schema_Decode.md"
OUT_MANIFEST = PKG / "v77_manifest.json"

OUT_DECISION = OUT / "v77_decision_summary.csv"
OUT_ISSUES = OUT / "v77_issues.csv"
OUT_ZIP = OUT / "Full_Unibo_Excel_Annotation_Schema_Decode.zip"
OUT_SHA = OUT / "Full_Unibo_Excel_Annotation_Schema_Decode.sha256"
OUT_NOTE = NOTES / "v77_full_excel_annotation_schema_decode_notes.md"
OUT_REPORT = REPORTS / "v77_full_excel_annotation_schema_decode_report.md"
OUT_PROGRESS = PROGRESS / "full_unibo_pipeline_progress_log.csv"

BEHAVIOURS = {"PI", "SI", "LAI", "STI", "NU", "BE", "DE", "AN", "IN", "IA", "BOX"}
NON_BEHAVIOURS = {"MC", "ARR", "SOMMA", "SUM", "TOTAL", "TOTALE"}

COLOUR_HINTS = {
    "blue": "blue",
    "blu": "blue",
    "green": "green",
    "verde": "green",
    "purple": "purple",
    "viola": "purple",
    "red_neck": "red_neck",
    "red tail": "red_tail",
    "red_tail": "red_tail",
    "neck": "red_neck",
    "tail": "red_tail",
    "red": "red",
    "rosso": "red",
    "no colour": "no_colour",
    "no color": "no_colour",
    "no_colour": "no_colour",
    "no_color": "no_colour",
    "pink": "pink",
    "cyan": "cyan",
}


def safe_to_csv(df, path):
    df.to_csv(path, index=False, quoting=csv.QUOTE_ALL, escapechar="\\", lineterminator="\n")


def clean(x):
    if pd.isna(x):
        return ""
    s = str(x).strip()
    if s.lower() in ["nan", "none", "null"]:
        return ""
    return s


def read_csv(path):
    df = pd.read_csv(path).fillna("")
    for c in df.columns:
        if df[c].dtype == object:
            df[c] = df[c].map(clean)
    return df


def bool_true(x):
    return str(x).strip().lower() == "true"


def sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def parse_day_date(filename):
    day = ""
    date = ""

    m = re.search(r"Giorno\s*([0-9]+)", filename, flags=re.IGNORECASE)
    if m:
        day = m.group(1)

    m = re.search(r"([0-9]{1,2})_([0-9]{1,2})_([0-9]{4})", filename)
    if m:
        date = f"{int(m.group(3)):04d}-{int(m.group(2)):02d}-{int(m.group(1)):02d}"

    return day, date


def norm_value(v):
    if v is None:
        return ""
    if isinstance(v, float) and v.is_integer():
        return str(int(v))
    return str(v).strip()


def time_to_min(text):
    s = str(text).strip().replace(",", ":").replace(".", ":")
    s = re.sub(r"[^0-9:]", "", s)

    if ":" in s:
        parts = s.split(":")
        h = int(parts[0])
        m = int(parts[1]) if len(parts) > 1 and parts[1] else 0
        return h * 60 + m

    digits = re.sub(r"[^0-9]", "", s)
    if len(digits) <= 2:
        return int(digits) * 60
    if len(digits) == 3:
        return int(digits[0]) * 60 + int(digits[1:])
    return int(digits[:2]) * 60 + int(digits[2:4])


def min_to_hhmm(m):
    if m == "" or m is None:
        return ""
    m = int(m)
    return f"{m // 60:02d}:{m % 60:02d}"


def parse_time_range(v):
    s = str(v).strip().replace(",", ":").replace(".", ":")
    m = re.search(r"([0-2]?[0-9](?::?[0-5][0-9])?)\s*[-–]\s*([0-2]?[0-9](?::?[0-5][0-9])?)", s)

    if not m:
        return "", ""

    try:
        a = time_to_min(m.group(1))
        b = time_to_min(m.group(2))
    except Exception:
        return "", ""

    if b <= a:
        return "", ""
    if a < 300 or b > 1440:
        return "", ""

    return a, b


def parse_sheet_identity(sheet_name):
    camera = ""
    room = ""
    pen = ""

    m = re.search(r"TLC\s*([0-9]+)", sheet_name, flags=re.IGNORECASE)
    if m:
        camera = f"TLC{m.group(1)}"

    m = re.search(r"\b([BCM])\s*([0-9]+)\b", sheet_name, flags=re.IGNORECASE)
    if m:
        room = m.group(1).upper()
        pen = f"{room}{m.group(2)}"

    if "big" in sheet_name.lower():
        pen = pen or "big"
    if "small" in sheet_name.lower():
        pen = pen or "small"

    return camera, room, pen


def detect_colour(row_values):
    text = " ".join([str(x).lower() for x in row_values if str(x).strip()])
    found = []

    for hint, canonical in COLOUR_HINTS.items():
        if hint in text:
            found.append(canonical)

    if not found:
        return ""

    return Counter(found).most_common(1)[0][0]


def load_matrix(excel_path, sheet_name):
    wb = openpyxl.load_workbook(excel_path, read_only=True, data_only=True)
    ws = wb[sheet_name]

    matrix = []
    for row in ws.iter_rows(values_only=True):
        matrix.append([norm_value(v) for v in row])

    wb.close()
    return matrix


def row_nonempty(row):
    return [norm_value(v) for v in row if norm_value(v)]


def find_time_slots(matrix):
    rows = []
    seen = set()

    for r, row in enumerate(matrix, start=1):
        for c, v in enumerate(row, start=1):
            a, b = parse_time_range(v)
            if a == "":
                continue

            key = (r, c, a, b)
            if key in seen:
                continue
            seen.add(key)

            rows.append({
                "header_row": r,
                "header_col": c,
                "start_minute": a,
                "end_minute": b,
                "start_hhmm": min_to_hhmm(a),
                "end_hhmm": min_to_hhmm(b),
                "raw_value": v,
            })

    return rows


def find_observation_periods(matrix):
    rows = []

    for r, row in enumerate(matrix[:15], start=1):
        row_text = " ".join([str(x).lower() for x in row])
        has_period_hint = "periodo" in row_text or "osservazione" in row_text

        nums = []
        for c, v in enumerate(row, start=1):
            s = norm_value(v)
            if re.fullmatch(r"[0-9]+", s) and int(s) in {0, 10, 20, 30, 40, 50}:
                nums.append((c, int(s), s))

        if has_period_hint or len(nums) >= 6:
            for c, n, raw in nums:
                rows.append({
                    "period_header_row": r,
                    "period_col": c,
                    "period_minute_offset": n,
                    "raw_value": raw,
                })

    return rows


def nearest_time_slot(col, time_slots):
    if not time_slots:
        return None

    before = [x for x in time_slots if int(x["header_col"]) <= int(col)]
    if before:
        return sorted(before, key=lambda x: int(x["header_col"]))[-1]

    return sorted(time_slots, key=lambda x: abs(int(x["header_col"]) - int(col)))[0]


def nearest_period(col, periods):
    if not periods:
        return ""

    before = [x for x in periods if int(x["period_col"]) <= int(col)]
    if before:
        return sorted(before, key=lambda x: int(x["period_col"]))[-1]["period_minute_offset"]

    return sorted(periods, key=lambda x: abs(int(x["period_col"]) - int(col)))[0]["period_minute_offset"]


def parse_sheet(excel_path, sheet_name):
    matrix = load_matrix(excel_path, sheet_name)

    day, date = parse_day_date(Path(excel_path).name)
    camera, room, pen = parse_sheet_identity(sheet_name)

    time_slots = find_time_slots(matrix)
    periods = find_observation_periods(matrix)

    schema = {
        "excel_path": str(excel_path),
        "excel_filename": Path(excel_path).name,
        "sheet_name": sheet_name,
        "day_number": day,
        "date": date,
        "camera": camera,
        "room": room,
        "pen": pen,
        "row_count": len(matrix),
        "col_count": max([len(r) for r in matrix]) if matrix else 0,
        "time_slot_count": len(time_slots),
        "observation_period_count": len(periods),
    }

    identity_rows = []
    norm_rows = []

    for r_idx, row in enumerate(matrix, start=1):
        vals = row_nonempty(row)
        if not vals:
            continue

        colour = detect_colour(vals)
        behav_count = sum(1 for x in vals if str(x).upper() in BEHAVIOURS)

        if colour or behav_count >= 3:
            identity_rows.append({
                "excel_path": str(excel_path),
                "excel_filename": Path(excel_path).name,
                "sheet_name": sheet_name,
                "row": r_idx,
                "detected_colour_identity": colour,
                "behaviour_token_count_in_row": behav_count,
                "row_values_preview": " ; ".join(vals[:80]),
            })

        # Skip first row to avoid treating behaviour headers as annotations.
        if r_idx <= 1:
            continue

        for c_idx, v in enumerate(row, start=1):
            code = str(v).upper().strip()

            if code not in BEHAVIOURS:
                continue

            ts = nearest_time_slot(c_idx, time_slots)
            period = nearest_period(c_idx, periods)

            confidence_reasons = []
            if date:
                confidence_reasons.append("date")
            if camera:
                confidence_reasons.append("camera")
            if ts:
                confidence_reasons.append("time_slot")
            if period != "":
                confidence_reasons.append("period")
            if colour:
                confidence_reasons.append("colour_hint")

            confidence = "medium" if date and ts and period != "" else "low"

            norm_rows.append({
                "annotation_candidate_id": f"{Path(excel_path).stem}__{sheet_name}__r{r_idx}_c{c_idx}",
                "excel_path": str(excel_path),
                "excel_filename": Path(excel_path).name,
                "sheet_name": sheet_name,
                "day_number": day,
                "date": date,
                "camera": camera,
                "room": room,
                "pen": pen,
                "row": r_idx,
                "col": c_idx,
                "detected_colour_identity": colour,
                "behaviour_code": code,
                "time_slot_start_minute": ts["start_minute"] if ts else "",
                "time_slot_end_minute": ts["end_minute"] if ts else "",
                "time_slot_start_hhmm": ts["start_hhmm"] if ts else "",
                "time_slot_end_hhmm": ts["end_hhmm"] if ts else "",
                "observation_period_offset_min": period,
                "observation_window_start_sec_within_slot": int(period) * 60 if period != "" else "",
                "observation_window_end_sec_within_slot": int(period) * 60 + 600 if period != "" else "",
                "normalization_confidence": confidence,
                "confidence_reasons": ";".join(confidence_reasons),
                "raw_cell_value": v,
                "row_values_preview": " ; ".join(vals[:80]),
            })

    return schema, time_slots, periods, identity_rows, norm_rows


def match_video_candidates(norm_df, videos):
    rows = []
    if len(norm_df) == 0:
        return pd.DataFrame(rows)

    slot_cols = [
        "excel_path", "excel_filename", "sheet_name", "date", "camera", "pen",
        "time_slot_start_minute", "time_slot_end_minute",
        "time_slot_start_hhmm", "time_slot_end_hhmm",
    ]

    slots = norm_df[slot_cols].drop_duplicates()

    for _, s in slots.iterrows():
        if clean(s["time_slot_start_minute"]) == "":
            continue

        for _, v in videos.iterrows():
            score = 0
            reasons = []

            if clean(s["date"]) and clean(v.get("parsed_date", "")) and clean(s["date"]) == clean(v.get("parsed_date", "")):
                score += 50
                reasons.append("date_match")

            if clean(s["camera"]) and clean(v.get("parsed_camera", "")) and clean(s["camera"]) == clean(v.get("parsed_camera", "")):
                score += 20
                reasons.append("camera_match")

            if clean(s["pen"]) and clean(v.get("parsed_pen", "")) and clean(s["pen"]) == clean(v.get("parsed_pen", "")):
                score += 20
                reasons.append("pen_match")

            if clean(s["time_slot_start_minute"]) and clean(v.get("parsed_start_minute", "")):
                try:
                    if int(float(s["time_slot_start_minute"])) == int(float(v["parsed_start_minute"])):
                        score += 30
                        reasons.append("start_time_match")
                except Exception:
                    pass

            if score == 0:
                continue

            quality = "high" if score >= 90 else ("medium" if score >= 70 else ("low" if score >= 50 else "weak"))

            rows.append({
                "excel_path": s["excel_path"],
                "excel_filename": s["excel_filename"],
                "sheet_name": s["sheet_name"],
                "annotation_date": s["date"],
                "annotation_camera": s["camera"],
                "annotation_pen": s["pen"],
                "annotation_time_start_hhmm": s["time_slot_start_hhmm"],
                "annotation_time_end_hhmm": s["time_slot_end_hhmm"],
                "video_path": clean(v.get("video_path", "")),
                "video_filename": clean(v.get("video_filename", "")),
                "video_date": clean(v.get("parsed_date", "")),
                "video_name_type": clean(v.get("video_name_type", "")),
                "video_camera": clean(v.get("parsed_camera", "")),
                "video_pen": clean(v.get("parsed_pen", "")),
                "video_camera_code": clean(v.get("parsed_camera_code", "")),
                "video_start_hhmm": clean(v.get("parsed_start_hhmm", "")),
                "video_end_hhmm": clean(v.get("parsed_end_hhmm_estimate", "")),
                "match_score": score,
                "match_quality": quality,
                "match_reasons": ";".join(reasons),
            })

    df = pd.DataFrame(rows)
    if len(df):
        df = df.sort_values(["excel_filename", "sheet_name", "annotation_time_start_hhmm", "match_score"], ascending=[True, True, True, False])
    return df


issues = []

for p in [V76_DECISION, V76_SHEETS, V76_VIDEOS]:
    if not p.exists():
        issues.append({
            "item": str(p),
            "issue_type": "hard_missing_v76_input",
            "issue_detail": "v77 requires v76 outputs.",
            "severity": "hard",
        })

if issues:
    issues_df = pd.DataFrame(issues)
    safe_to_csv(issues_df, OUT_ISSUES)
    raise SystemExit("Missing v76 inputs.")

v76_decision = read_csv(V76_DECISION)
sheets = read_csv(V76_SHEETS)
videos = read_csv(V76_VIDEOS)

if not bool_true(v76_decision.iloc[0].get("ready_for_v77_mapping_resolution", "")):
    issues.append({
        "item": "v76_decision",
        "issue_type": "hard_v76_not_ready",
        "issue_detail": "v76 is not marked ready for v77.",
        "severity": "hard",
    })

schema_rows = []
time_rows = []
period_rows = []
identity_rows = []
norm_rows = []

for _, s in sheets.iterrows():
    excel_path = Path(clean(s["excel_path"]))
    sheet_name = clean(s["sheet_name"])

    try:
        schema, ts, periods, ids, norms = parse_sheet(excel_path, sheet_name)
        schema_rows.append(schema)

        for r in ts:
            time_rows.append({
                "excel_path": str(excel_path),
                "excel_filename": excel_path.name,
                "sheet_name": sheet_name,
                **r,
            })

        for r in periods:
            period_rows.append({
                "excel_path": str(excel_path),
                "excel_filename": excel_path.name,
                "sheet_name": sheet_name,
                **r,
            })

        identity_rows.extend(ids)
        norm_rows.extend(norms)

    except Exception as e:
        issues.append({
            "item": f"{excel_path}::{sheet_name}",
            "issue_type": "warning_sheet_parse_failed",
            "issue_detail": str(e)[:1000],
            "severity": "warning",
        })

schema_df = pd.DataFrame(schema_rows)
time_df = pd.DataFrame(time_rows)
period_df = pd.DataFrame(period_rows)
identity_df = pd.DataFrame(identity_rows)
norm_df = pd.DataFrame(norm_rows)

safe_to_csv(schema_df, OUT_SHEET_SCHEMA)
safe_to_csv(time_df, OUT_TIME_SLOTS)
safe_to_csv(period_df, OUT_OBS_PERIODS)
safe_to_csv(identity_df, OUT_IDENTITY_ROWS)
safe_to_csv(norm_df, OUT_NORMALIZED)

if len(norm_df):
    behav_dist = norm_df.groupby("behaviour_code").size().reset_index(name="candidate_count").sort_values("candidate_count", ascending=False)
else:
    behav_dist = pd.DataFrame(columns=["behaviour_code", "candidate_count"])

safe_to_csv(behav_dist, OUT_BEHAV_DIST)

video_match_df = match_video_candidates(norm_df, videos)
safe_to_csv(video_match_df, OUT_VIDEO_MATCH)

if len(norm_df):
    needs_review = norm_df[
        (norm_df["normalization_confidence"] != "medium")
        | (norm_df["detected_colour_identity"] == "")
        | (norm_df["time_slot_start_hhmm"] == "")
        | (norm_df["observation_period_offset_min"] == "")
    ].copy()
else:
    needs_review = pd.DataFrame()

safe_to_csv(needs_review, OUT_NEEDS_REVIEW)

qa_rows = []

def add_qa(name, expected, actual, passed, severity, detail):
    qa_rows.append({
        "check_name": name,
        "expected": str(expected),
        "actual": str(actual),
        "passed": bool(passed),
        "severity": severity,
        "detail": detail,
    })

add_qa("v76_ready", True, bool_true(v76_decision.iloc[0].get("ready_for_v77_mapping_resolution", "")), bool_true(v76_decision.iloc[0].get("ready_for_v77_mapping_resolution", "")), "hard", "v76 should be ready.")
add_qa("sheets_processed", len(sheets), len(schema_df), len(schema_df) == len(sheets), "hard", "All v76 sheets should be processed.")
add_qa("time_slots_found", ">0", len(time_df), len(time_df) > 0, "hard", "Time slots should be extracted.")
add_qa("observation_periods_found", ">0", len(period_df), len(period_df) > 0, "hard", "Observation period columns should be extracted.")
add_qa("identity_rows_found", ">0", len(identity_df), len(identity_df) > 0, "warning", "Identity/colour candidate rows should be found.")
add_qa("normalized_annotation_candidates", ">0", len(norm_df), len(norm_df) > 0, "hard", "Behaviour annotation candidates should be normalized.")
add_qa("behaviour_classes_found", ">=10", norm_df["behaviour_code"].nunique() if len(norm_df) else 0, (norm_df["behaviour_code"].nunique() if len(norm_df) else 0) >= 10, "hard", "Most/all behaviour classes should be found.")
add_qa("video_time_candidates_found", ">0", len(video_match_df), len(video_match_df) > 0, "warning", "Video-time candidate matches should be generated.")
add_qa("needs_review_table_created", ">=0", len(needs_review), len(needs_review) >= 0, "hard", "Needs-review table should be created.")

qa = pd.DataFrame(qa_rows)
safe_to_csv(qa, OUT_QA)

hard_quality_failures = int(((qa["severity"] == "hard") & (~qa["passed"])).sum())
warning_quality_failures = int(((qa["severity"] == "warning") & (~qa["passed"])).sum())

if hard_quality_failures:
    issues.append({
        "item": "v77_quality_checks",
        "issue_type": "hard_schema_decode_quality_failed",
        "issue_detail": f"{hard_quality_failures} hard QA checks failed.",
        "severity": "hard",
    })

if warning_quality_failures:
    issues.append({
        "item": "v77_quality_checks",
        "issue_type": "warning_schema_decode_has_warnings",
        "issue_detail": f"{warning_quality_failures} warning QA checks failed.",
        "severity": "warning",
    })

issues.append({
    "item": "scope",
    "issue_type": "info_schema_decode_only",
    "issue_detail": "v77 decodes Excel schema and creates normalized candidates. It does not finalize mapping, tracking, or GT.",
    "severity": "info",
})

issues_df = pd.DataFrame(issues, columns=["item", "issue_type", "issue_detail", "severity"])
safe_to_csv(issues_df, OUT_ISSUES)

hard_issue_count = int((issues_df["severity"] == "hard").sum())
warning_count = int((issues_df["severity"] == "warning").sum())
info_count = int((issues_df["severity"] == "info").sum())

medium_rows = int((norm_df["normalization_confidence"] == "medium").sum()) if len(norm_df) else 0
review_rows = int(len(needs_review))

manifest = {
    "version": "v77_full_excel_annotation_schema_decode",
    "created_at": datetime.now().isoformat(timespec="seconds"),
    "sheet_count": int(len(schema_df)),
    "time_slot_rows": int(len(time_df)),
    "observation_period_rows": int(len(period_df)),
    "identity_candidate_rows": int(len(identity_df)),
    "normalized_annotation_candidate_rows": int(len(norm_df)),
    "behaviour_class_count": int(norm_df["behaviour_code"].nunique()) if len(norm_df) else 0,
    "medium_confidence_rows": medium_rows,
    "rows_needing_review": review_rows,
    "video_time_candidate_matches": int(len(video_match_df)),
    "hard_issue_count": hard_issue_count,
    "claim_boundary": "Excel schema decode only",
}

OUT_MANIFEST.write_text(json.dumps(manifest, indent=2, ensure_ascii=False))

readme = f"""# v77 Full Excel Annotation Schema Decode

This stage decodes the internal structure of all Excel annotation sheets.

It extracts time slots, observation periods, candidate identity/colour rows, behaviour-coded cells, normalized annotation candidates and time-slot/video candidate matches.

This is not final moving GT yet.

Main counts:
- Sheets processed: {len(schema_df)}
- Time slot rows: {len(time_df)}
- Observation period rows: {len(period_df)}
- Identity candidate rows: {len(identity_df)}
- Normalized annotation candidate rows: {len(norm_df)}
- Behaviour classes: {norm_df['behaviour_code'].nunique() if len(norm_df) else 0}
- Rows needing review: {review_rows}
- Video-time candidate matches: {len(video_match_df)}
- Hard issues: {hard_issue_count}
"""

OUT_README.write_text(readme)
OUT_REPORT.write_text(readme)

if OUT_ZIP.exists():
    OUT_ZIP.unlink()

with zipfile.ZipFile(OUT_ZIP, "w", compression=zipfile.ZIP_DEFLATED) as z:
    for p in sorted(PKG.rglob("*")):
        if p.is_file():
            z.write(p, p.relative_to(OUT))

zip_hash = sha256_file(OUT_ZIP)
OUT_SHA.write_text(f"{zip_hash}  {OUT_ZIP.name}\n")

decision = pd.DataFrame([{
    "v77_decision": "full_excel_annotation_schema_decode_completed" if hard_issue_count == 0 else "full_excel_annotation_schema_decode_has_blocking_issues",
    "sheet_count": int(len(schema_df)),
    "time_slot_rows": int(len(time_df)),
    "observation_period_rows": int(len(period_df)),
    "identity_candidate_rows": int(len(identity_df)),
    "normalized_annotation_candidate_rows": int(len(norm_df)),
    "behaviour_class_count": int(norm_df["behaviour_code"].nunique()) if len(norm_df) else 0,
    "medium_confidence_normalized_rows": medium_rows,
    "rows_needing_review": review_rows,
    "video_time_candidate_matches": int(len(video_match_df)),
    "zip_path": str(OUT_ZIP),
    "zip_sha256": zip_hash,
    "hard_quality_failures": hard_quality_failures,
    "warning_quality_failures": warning_quality_failures,
    "hard_issue_count": hard_issue_count,
    "warning_count": warning_count,
    "info_count": info_count,
    "issue_count": int(len(issues_df)),
    "ready_for_v78_final_annotation_window_table": bool(hard_issue_count == 0),
    "claim_scope": "full_excel_schema_decode_and_normalized_annotation_candidates_only",
    "generated_at": datetime.now().isoformat(timespec="seconds"),
}])

safe_to_csv(decision, OUT_DECISION)

OUT_NOTE.write_text(
    "# v77 Full Excel Annotation Schema Decode\n\n"
    f"- v77 decision: {decision.iloc[0]['v77_decision']}\n"
    f"- Sheets processed: {len(schema_df)}\n"
    f"- Time slot rows: {len(time_df)}\n"
    f"- Observation period rows: {len(period_df)}\n"
    f"- Identity candidate rows: {len(identity_df)}\n"
    f"- Normalized annotation candidate rows: {len(norm_df)}\n"
    f"- Behaviour classes: {norm_df['behaviour_code'].nunique() if len(norm_df) else 0}\n"
    f"- Medium-confidence normalized rows: {medium_rows}\n"
    f"- Rows needing review: {review_rows}\n"
    f"- Video-time candidate matches: {len(video_match_df)}\n"
    f"- Hard issues: {hard_issue_count}\n"
    f"- Ready for v78 final annotation window table: {bool(hard_issue_count == 0)}\n\n"
    "This stage decodes Excel annotation structure and creates normalized candidates. It does not create final moving GT yet.\n"
)

progress_row = pd.DataFrame([{
    "date": datetime.now().date().isoformat(),
    "stage": "v77",
    "task_name": "Full Excel annotation schema decode",
    "status": "PASS_WITH_WARNINGS" if hard_issue_count == 0 and warning_count > 0 else ("PASS" if hard_issue_count == 0 else "NEEDS_FIX"),
    "input_summary": str(V76),
    "output_summary": str(PKG),
    "hard_issues": hard_issue_count,
    "warnings": warning_count,
    "next_action": "Create finalized annotation-window table and resolve video mapping in v78." if hard_issue_count == 0 else "Fix v77 hard issues.",
}])

if OUT_PROGRESS.exists():
    old = pd.read_csv(OUT_PROGRESS)
    progress = pd.concat([old, progress_row], ignore_index=True)
else:
    progress = progress_row

safe_to_csv(progress, OUT_PROGRESS)

print("=== v77 decision ===")
print(decision.to_string(index=False))

print("\n=== behaviour distribution ===")
print(behav_dist.to_string(index=False))

print("\n=== schema sample ===")
print(schema_df.head(20).to_string(index=False))

print("\n=== normalized sample ===")
print(norm_df.head(20).to_string(index=False) if len(norm_df) else "No normalized rows.")

print("\n=== QA ===")
print(qa.to_string(index=False))

print("\n=== issues ===")
print(issues_df.to_string(index=False) if len(issues_df) else "No issues found.")
