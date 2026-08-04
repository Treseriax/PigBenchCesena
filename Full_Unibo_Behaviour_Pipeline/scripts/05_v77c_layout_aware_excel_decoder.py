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
FULL = ROOT / "Full_Unibo_Behaviour_Pipeline"

V76 = FULL / "outputs" / "v76_full_annotation_video_mapping_audit"
V76_PKG = V76 / "Full_Unibo_Annotation_Video_Mapping_Audit"
V76_DECISION = V76 / "v76_decision_summary.csv"
V76_EXCEL_FILES = V76_PKG / "v76_all_excel_files_inventory.csv"
V76_SHEETS = V76_PKG / "v76_all_excel_sheets_inventory.csv"

OUT = FULL / "outputs" / "v77c_layout_aware_excel_decoder"
PKG = OUT / "Full_Unibo_Layout_Aware_Excel_Decoder"
NOTES = FULL / "notes"
REPORTS = FULL / "reports"
PROGRESS = FULL / "progress"

for p in [OUT, PKG, NOTES, REPORTS, PROGRESS]:
    p.mkdir(parents=True, exist_ok=True)

OUT_ROWS = PKG / "v77c_layout_aware_annotation_windows.csv"
OUT_SHEET_SUMMARY = PKG / "v77c_sheet_decode_summary.csv"
OUT_EXCLUDED = PKG / "v77c_excluded_rows.csv"
OUT_BEHAV = PKG / "v77c_behaviour_distribution.csv"
OUT_COLOUR = PKG / "v77c_colour_identity_distribution.csv"
OUT_TIME = PKG / "v77c_time_slot_distribution.csv"
OUT_DUP_POLICY = PKG / "v77c_duplicate_excel_policy.csv"
OUT_QA = PKG / "v77c_quality_checks.csv"
OUT_README = PKG / "README_v77c_Layout_Aware_Excel_Decoder.md"
OUT_MANIFEST = PKG / "v77c_manifest.json"

OUT_DECISION = OUT / "v77c_decision_summary.csv"
OUT_ISSUES = OUT / "v77c_issues.csv"
OUT_ZIP = OUT / "Full_Unibo_Layout_Aware_Excel_Decoder.zip"
OUT_SHA = OUT / "Full_Unibo_Layout_Aware_Excel_Decoder.sha256"
OUT_NOTE = NOTES / "v77c_layout_aware_excel_decoder_notes.md"
OUT_REPORT = REPORTS / "v77c_layout_aware_excel_decoder_report.md"
OUT_PROGRESS = PROGRESS / "full_unibo_pipeline_progress_log.csv"

BEHAVIOURS = {"PI", "SI", "LAI", "STI", "NU", "BE", "DE", "AN", "IN", "IA", "BOX"}
PERIODS = [0, 10, 20, 30, 40, 50]

COLOUR_PATTERNS = [
    ("red_tail", [r"red\s*tail", r"tail", r"coda"]),
    ("red_neck", [r"red\s*neck", r"neck", r"collo"]),
    ("no_colour", [r"no\s*colour", r"no\s*color", r"no_colour", r"no_color", r"senza"]),
    ("blue", [r"blue", r"blu"]),
    ("green", [r"green", r"verde"]),
    ("purple", [r"purple", r"viola"]),
    ("pink", [r"pink", r"rosa"]),
    ("cyan", [r"cyan"]),
    ("red", [r"red", r"rosso"]),
]


def safe_to_csv(df, path):
    df.to_csv(path, index=False, quoting=csv.QUOTE_ALL, escapechar="\\", lineterminator="\n")


def clean(x):
    if pd.isna(x):
        return ""
    s = str(x).strip()
    if s.lower() in ["nan", "none", "null"]:
        return ""
    return s


def read_csv_clean(path):
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


def norm_value(v):
    if v is None:
        return ""
    if isinstance(v, float) and v.is_integer():
        return str(int(v))
    return str(v).strip()


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
    m = int(m)
    return f"{m // 60:02d}:{m % 60:02d}"


def sec_to_hhmmss(sec):
    sec = int(sec)
    h = sec // 3600
    m = (sec % 3600) // 60
    s = sec % 60
    return f"{h:02d}:{m:02d}:{s:02d}"


def parse_time_range(v):
    s = str(v).strip().replace(",", ":").replace(".", ":")
    m = re.search(r"([0-2]?[0-9](?::?[0-5][0-9])?)\s*[-–]\s*([0-2]?[0-9](?::?[0-5][0-9])?)", s)

    if not m:
        return None

    try:
        a = time_to_min(m.group(1))
        b = time_to_min(m.group(2))
    except Exception:
        return None

    if b <= a:
        return None
    if a < 300 or b > 1440:
        return None

    return a, b


def detect_colour_from_text(text):
    low = str(text).lower()

    for canonical, patterns in COLOUR_PATTERNS:
        for pat in patterns:
            if re.search(pat, low):
                return canonical

    return ""


def load_matrix(excel_path, sheet_name):
    wb = openpyxl.load_workbook(excel_path, read_only=True, data_only=True)
    ws = wb[sheet_name]

    matrix = []
    for row in ws.iter_rows(values_only=True):
        matrix.append([norm_value(v) for v in row])

    wb.close()
    return matrix


def nonempty_with_cols(row):
    out = []
    for idx, v in enumerate(row, start=1):
        s = norm_value(v)
        if s:
            out.append((idx, s))
    return out


def find_header_time_slots(matrix):
    candidates = []

    for r_idx, row in enumerate(matrix[:10], start=1):
        vals = nonempty_with_cols(row)
        ranges = []

        for c, v in vals:
            tr = parse_time_range(v)
            if tr is not None:
                ranges.append((c, tr[0], tr[1], v))

        # Prefer explicit Fascia oraria row.
        row_text = " ".join(v.lower() for _, v in vals)
        if len(ranges) >= 2:
            score = len(ranges)
            if "fascia" in row_text or "oraria" in row_text:
                score += 100
            candidates.append((score, r_idx, ranges))

    if not candidates:
        return [], ""

    candidates = sorted(candidates, key=lambda x: x[0], reverse=True)
    _, header_row, ranges = candidates[0]

    # Deduplicate by time range, preserve order.
    seen = set()
    slots = []

    for _, a, b, raw in ranges:
        key = (a, b)
        if key in seen:
            continue
        seen.add(key)
        slots.append({
            "slot_index": len(slots),
            "slot_start_minute": a,
            "slot_end_minute": b,
            "slot_start_hhmm": min_to_hhmm(a),
            "slot_end_hhmm": min_to_hhmm(b),
            "raw_value": raw,
            "header_row": header_row,
        })

    return slots, "header_time_slots"


def extract_behaviour_sequence_from_row(row):
    vals = nonempty_with_cols(row)
    if not vals:
        return "", [], "empty"

    first_text = vals[0][1]
    colour = detect_colour_from_text(first_text)

    # Sometimes colour may occur in second/nearby cell.
    if not colour:
        colour = detect_colour_from_text(" ".join(v for _, v in vals[:3]))

    if not colour:
        return "", [], "no_colour"

    seq = []
    started = False

    for c, v in vals[1:]:
        code = v.upper().strip()

        if code in BEHAVIOURS:
            seq.append((c, code))
            started = True
            continue

        if started:
            # stop at repeated colour label / summary numbers / totals
            break

    if not seq:
        return colour, [], "no_behaviour_sequence"

    return colour, seq, "ok"


def canonical_copy_policy(excel_files):
    rows = []

    if "sha256" not in excel_files.columns:
        for _, r in excel_files.iterrows():
            rows.append({
                "excel_path": r["excel_path"],
                "excel_filename": r["excel_filename"],
                "sha256": "",
                "canonical_excel_path_for_sha": r["excel_path"],
                "is_canonical_copy": True,
                "duplicate_group_size": 1,
            })
        return pd.DataFrame(rows)

    for sha, g in excel_files.groupby("sha256"):
        gg = g.copy()
        gg["prefer_score"] = gg["excel_path"].map(lambda p: 1 if "/excel/" in str(p) else 0)
        gg = gg.sort_values(["prefer_score", "excel_path"], ascending=[False, True])
        canonical = gg.iloc[0]["excel_path"]

        for _, r in gg.iterrows():
            rows.append({
                "excel_path": r["excel_path"],
                "excel_filename": r["excel_filename"],
                "sha256": sha,
                "canonical_excel_path_for_sha": canonical,
                "is_canonical_copy": clean(r["excel_path"]) == clean(canonical),
                "duplicate_group_size": len(gg),
            })

    return pd.DataFrame(rows)


issues = []

for p in [V76_DECISION, V76_EXCEL_FILES, V76_SHEETS]:
    if not p.exists():
        issues.append({
            "item": str(p),
            "issue_type": "hard_missing_v76_input",
            "issue_detail": "v77c requires v76 outputs.",
            "severity": "hard",
        })

if issues:
    issues_df = pd.DataFrame(issues)
    safe_to_csv(issues_df, OUT_ISSUES)
    raise SystemExit("Missing v76 inputs.")

v76_decision = read_csv_clean(V76_DECISION)
excel_files = read_csv_clean(V76_EXCEL_FILES)
sheets = read_csv_clean(V76_SHEETS)

if not bool_true(v76_decision.iloc[0].get("ready_for_v77_mapping_resolution", "")):
    issues.append({
        "item": "v76_decision",
        "issue_type": "hard_v76_not_ready",
        "issue_detail": "v76 must be ready before v77c.",
        "severity": "hard",
    })

dup_policy = canonical_copy_policy(excel_files)
safe_to_csv(dup_policy, OUT_DUP_POLICY)

canonical_by_path = {}
for _, r in dup_policy.iterrows():
    canonical_by_path[clean(r["excel_path"])] = clean(r["canonical_excel_path_for_sha"])

rows = []
excluded = []
sheet_summary = []

for _, s in sheets.iterrows():
    excel_path = Path(clean(s["excel_path"]))
    sheet_name = clean(s["sheet_name"])

    canonical_path = canonical_by_path.get(str(excel_path), str(excel_path))
    is_canonical = str(excel_path) == canonical_path

    day, date = parse_day_date(excel_path.name)
    camera, room, pen = parse_sheet_identity(sheet_name)

    if not is_canonical:
        excluded.append({
            "excel_path": str(excel_path),
            "sheet_name": sheet_name,
            "reason": "duplicate_excel_copy",
            "detail": f"canonical={canonical_path}",
        })
        continue

    try:
        matrix = load_matrix(excel_path, sheet_name)
    except Exception as e:
        issues.append({
            "item": f"{excel_path}::{sheet_name}",
            "issue_type": "warning_sheet_load_failed",
            "issue_detail": str(e)[:1000],
            "severity": "warning",
        })
        continue

    slots, slot_source = find_header_time_slots(matrix)

    decoded_rows_this_sheet = 0
    excluded_rows_this_sheet = 0
    identity_rows_this_sheet = 0
    behaviour_cells_this_sheet = 0
    behaviour_sequence_lengths = []

    if not slots:
        excluded.append({
            "excel_path": str(excel_path),
            "sheet_name": sheet_name,
            "reason": "no_time_slots_detected",
            "detail": "Could not find header time ranges.",
        })

    for r_idx, row in enumerate(matrix, start=1):
        colour, seq, status = extract_behaviour_sequence_from_row(row)

        if status == "no_colour":
            continue

        if status != "ok":
            excluded_rows_this_sheet += 1
            excluded.append({
                "excel_path": str(excel_path),
                "sheet_name": sheet_name,
                "row": r_idx,
                "reason": status,
                "detail": "Row looked colour-related but no behaviour sequence was decoded.",
            })
            continue

        identity_rows_this_sheet += 1
        behaviour_sequence_lengths.append(len(seq))

        if not slots:
            excluded_rows_this_sheet += 1
            excluded.append({
                "excel_path": str(excel_path),
                "sheet_name": sheet_name,
                "row": r_idx,
                "reason": "no_time_slots_for_sequence",
                "detail": f"colour={colour}, sequence_len={len(seq)}",
            })
            continue

        max_cells = min(len(seq), len(slots) * len(PERIODS))

        if len(seq) > max_cells:
            excluded.append({
                "excel_path": str(excel_path),
                "sheet_name": sheet_name,
                "row": r_idx,
                "reason": "truncated_extra_behaviour_cells",
                "detail": f"sequence_len={len(seq)}, usable={max_cells}, slots={len(slots)}",
            })

        for seq_idx, (col_idx, behaviour) in enumerate(seq[:max_cells]):
            slot_idx = seq_idx // len(PERIODS)
            period_idx = seq_idx % len(PERIODS)

            slot = slots[slot_idx]
            period_min = PERIODS[period_idx]

            absolute_start_sec = int(slot["slot_start_minute"]) * 60 + period_min * 60
            absolute_end_sec = absolute_start_sec + 10

            annotation_id = (
                f"{date}__{sheet_name.replace(' ', '_')}__"
                f"{slot['slot_start_hhmm'].replace(':','')}__"
                f"p{period_min:02d}__{colour}__r{r_idx}_c{col_idx}"
            )

            identity_status = "specific_identity"
            if colour == "red":
                identity_status = "ambiguous_red_identity_needs_resolution"

            rows.append({
                "annotation_window_id": annotation_id,
                "excel_path": str(excel_path),
                "excel_filename": excel_path.name,
                "sheet_name": sheet_name,
                "day_number": day,
                "date": date,
                "camera": camera,
                "room": room,
                "pen": pen,
                "row": r_idx,
                "col": col_idx,
                "sequence_index": seq_idx,
                "sequence_length": len(seq),
                "canonical_colour_identity": colour,
                "identity_resolution_status": identity_status,
                "behaviour_code": behaviour,
                "slot_index": slot_idx,
                "slot_start_minute": slot["slot_start_minute"],
                "slot_end_minute": slot["slot_end_minute"],
                "slot_start_hhmm": slot["slot_start_hhmm"],
                "slot_end_hhmm": slot["slot_end_hhmm"],
                "observation_period_offset_min": period_min,
                "window_duration_sec": 10,
                "absolute_start_sec_of_day": absolute_start_sec,
                "absolute_end_sec_of_day": absolute_end_sec,
                "absolute_start_hhmmss": sec_to_hhmmss(absolute_start_sec),
                "absolute_end_hhmmss": sec_to_hhmmss(absolute_end_sec),
                "decoder": "layout_aware_sequence_v77c",
                "time_slot_source": slot_source,
            })

            decoded_rows_this_sheet += 1
            behaviour_cells_this_sheet += 1

    sheet_summary.append({
        "excel_path": str(excel_path),
        "excel_filename": excel_path.name,
        "sheet_name": sheet_name,
        "date": date,
        "camera": camera,
        "room": room,
        "pen": pen,
        "time_slot_count": len(slots),
        "time_slots": ";".join([f"{x['slot_start_hhmm']}-{x['slot_end_hhmm']}" for x in slots]),
        "identity_rows_decoded": identity_rows_this_sheet,
        "decoded_annotation_windows": decoded_rows_this_sheet,
        "behaviour_cells_decoded": behaviour_cells_this_sheet,
        "excluded_rows": excluded_rows_this_sheet,
        "sequence_lengths": ";".join(map(str, sorted(set(behaviour_sequence_lengths)))),
    })

rows_df = pd.DataFrame(rows)
excluded_df = pd.DataFrame(excluded)
summary_df = pd.DataFrame(sheet_summary)

safe_to_csv(rows_df, OUT_ROWS)
safe_to_csv(excluded_df, OUT_EXCLUDED)
safe_to_csv(summary_df, OUT_SHEET_SUMMARY)

if len(rows_df):
    behav = rows_df.groupby("behaviour_code").size().reset_index(name="count").sort_values("count", ascending=False)
    colour = rows_df.groupby("canonical_colour_identity").size().reset_index(name="count").sort_values("count", ascending=False)
    time_dist = rows_df.groupby(["slot_start_hhmm", "slot_end_hhmm"]).size().reset_index(name="count").sort_values("slot_start_hhmm")
else:
    behav = pd.DataFrame(columns=["behaviour_code", "count"])
    colour = pd.DataFrame(columns=["canonical_colour_identity", "count"])
    time_dist = pd.DataFrame(columns=["slot_start_hhmm", "slot_end_hhmm", "count"])

safe_to_csv(behav, OUT_BEHAV)
safe_to_csv(colour, OUT_COLOUR)
safe_to_csv(time_dist, OUT_TIME)

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

decoded_count = len(rows_df)
sheet_count = len(summary_df)
behaviour_class_count = int(rows_df["behaviour_code"].nunique()) if len(rows_df) else 0
colour_count = int(rows_df["canonical_colour_identity"].nunique()) if len(rows_df) else 0
duration_ok = bool((rows_df["window_duration_sec"] == 10).all()) if len(rows_df) else False
has_early_slots = bool(rows_df["slot_start_hhmm"].isin(["07:00", "08:00", "09:00"]).any()) if len(rows_df) else False
has_no_header_rows = True

if len(rows_df):
    has_no_header_rows = not rows_df["row"].astype(int).le(4).any()

add_qa("decoded_rows_created", ">0", decoded_count, decoded_count > 0, "hard", "Layout-aware decoder should create rows.")
add_qa("canonical_sheets_processed", ">0", sheet_count, sheet_count > 0, "hard", "At least one canonical sheet should be processed.")
add_qa("behaviour_classes_preserved", ">=10", behaviour_class_count, behaviour_class_count >= 10, "hard", "Most/all behaviour classes should remain.")
add_qa("colour_identities_found", ">=5", colour_count, colour_count >= 5, "hard", "Multiple pig colour identities should be found.")
add_qa("all_windows_10_seconds", True, duration_ok, duration_ok, "hard", "Every decoded observation window should be 10 seconds.")
add_qa("early_time_slots_present", True, has_early_slots, has_early_slots, "hard", "Decoder should recover early slots such as 07:00/08:00/09:00.")
add_qa("no_header_rows_in_output", True, has_no_header_rows, has_no_header_rows, "hard", "Header rows should not become annotation rows.")

qa = pd.DataFrame(qa_rows)
safe_to_csv(qa, OUT_QA)

hard_quality_failures = int(((qa["severity"] == "hard") & (~qa["passed"])).sum())

if hard_quality_failures:
    issues.append({
        "item": "v77c_quality_checks",
        "issue_type": "hard_layout_aware_decode_failed",
        "issue_detail": f"{hard_quality_failures} hard QA checks failed.",
        "severity": "hard",
    })

ambiguous_red = int((rows_df["canonical_colour_identity"] == "red").sum()) if len(rows_df) else 0
if ambiguous_red > 0:
    issues.append({
        "item": "identity_resolution",
        "issue_type": "info_generic_red_identity_exists",
        "issue_detail": f"{ambiguous_red} rows have generic red identity; later red_neck/red_tail review may be needed.",
        "severity": "info",
    })

issues.append({
    "item": "scope",
    "issue_type": "info_layout_decode_only",
    "issue_detail": "v77c creates layout-aware annotation windows. It does not perform video mapping, tracking, or model training.",
    "severity": "info",
})

issues_df = pd.DataFrame(issues, columns=["item", "issue_type", "issue_detail", "severity"])
safe_to_csv(issues_df, OUT_ISSUES)

hard_issue_count = int((issues_df["severity"] == "hard").sum())
warning_count = int((issues_df["severity"] == "warning").sum()) if "warning" in set(issues_df["severity"]) else 0
info_count = int((issues_df["severity"] == "info").sum())

manifest = {
    "version": "v77c_layout_aware_excel_decoder",
    "created_at": datetime.now().isoformat(timespec="seconds"),
    "decoded_annotation_windows": int(decoded_count),
    "sheet_count": int(sheet_count),
    "behaviour_class_count": int(behaviour_class_count),
    "colour_identity_count": int(colour_count),
    "ambiguous_red_rows": int(ambiguous_red),
    "hard_issue_count": int(hard_issue_count),
    "claim_boundary": "layout-aware annotation windows only; no video mapping/tracking/model",
}

OUT_MANIFEST.write_text(json.dumps(manifest, indent=2, ensure_ascii=False))

readme = f"""# v77c Layout-Aware Excel Decoder

## Purpose

v77c replaces the earlier column-nearest decoding logic with a sequential layout-aware decoder.

For each pig colour row, it reads the behaviour sequence and maps sequence index to:
- hourly slot,
- observation period offset,
- 10-second window.

## Main counts

- Decoded annotation windows: {decoded_count}
- Canonical sheets processed: {sheet_count}
- Behaviour classes: {behaviour_class_count}
- Colour identities: {colour_count}
- Generic red rows: {ambiguous_red}
- Hard issues: {hard_issue_count}

## Boundary

This is not video-mapped GT yet.
v78b/v78c should map these corrected windows to videos.
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
    "v77c_decision": "layout_aware_excel_decoder_completed" if hard_issue_count == 0 else "layout_aware_excel_decoder_has_blocking_issues",
    "decoded_annotation_windows": int(decoded_count),
    "canonical_sheets_processed": int(sheet_count),
    "behaviour_class_count": int(behaviour_class_count),
    "colour_identity_count": int(colour_count),
    "generic_red_rows": int(ambiguous_red),
    "zip_path": str(OUT_ZIP),
    "zip_sha256": zip_hash,
    "hard_quality_failures": int(hard_quality_failures),
    "hard_issue_count": int(hard_issue_count),
    "warning_count": int(warning_count),
    "info_count": int(info_count),
    "issue_count": int(len(issues_df)),
    "ready_for_corrected_video_mapping": bool(hard_issue_count == 0),
    "claim_scope": "layout_aware_annotation_windows_only",
    "generated_at": datetime.now().isoformat(timespec="seconds"),
}])

safe_to_csv(decision, OUT_DECISION)

OUT_NOTE.write_text(
    "# v77c Layout-Aware Excel Decoder\n\n"
    f"- v77c decision: {decision.iloc[0]['v77c_decision']}\n"
    f"- Decoded annotation windows: {decoded_count}\n"
    f"- Canonical sheets processed: {sheet_count}\n"
    f"- Behaviour classes: {behaviour_class_count}\n"
    f"- Colour identities: {colour_count}\n"
    f"- Generic red rows: {ambiguous_red}\n"
    f"- Hard issues: {hard_issue_count}\n"
    f"- Ready for corrected video mapping: {bool(hard_issue_count == 0)}\n\n"
    "v77c is the corrected layout-aware annotation decoder. Earlier v77/v77b/v78 outputs should be treated as diagnostic if their timestamps conflict with v77c.\n"
)

progress_row = pd.DataFrame([{
    "date": datetime.now().date().isoformat(),
    "stage": "v77c",
    "task_name": "Layout-aware Excel annotation decoder",
    "status": "PASS" if hard_issue_count == 0 else "NEEDS_FIX",
    "input_summary": str(V76_SHEETS),
    "output_summary": str(PKG),
    "hard_issues": hard_issue_count,
    "warnings": warning_count,
    "next_action": "Run corrected video mapping using v77c output." if hard_issue_count == 0 else "Fix v77c hard issues.",
}])

if OUT_PROGRESS.exists():
    old = pd.read_csv(OUT_PROGRESS)
    progress = pd.concat([old, progress_row], ignore_index=True)
else:
    progress = progress_row

safe_to_csv(progress, OUT_PROGRESS)

print("=== v77c decision ===")
print(decision.to_string(index=False))

print("\n=== behaviour distribution ===")
print(behav.to_string(index=False))

print("\n=== colour distribution ===")
print(colour.to_string(index=False))

print("\n=== time slot distribution ===")
print(time_dist.to_string(index=False))

print("\n=== sheet summary sample ===")
print(summary_df.head(30).to_string(index=False))

print("\n=== decoded sample ===")
print(rows_df.head(30).to_string(index=False))

print("\n=== QA ===")
print(qa.to_string(index=False))

print("\n=== issues ===")
print(issues_df.to_string(index=False))
