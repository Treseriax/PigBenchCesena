from pathlib import Path
from datetime import datetime
import re
import csv
import pandas as pd


ROOT = Path.home() / "PigBench"
W8 = ROOT / "Week8_Ground_Truth_Validation_Behaviour_Dataset_Preparation"

V45_CLIP_OBJECTS = W8 / "outputs" / "propagated_ground_truth" / "v45_label_propagation" / "week8_v45_clip_object_propagated_annotations.csv"
EXCEL_DIR = Path("/work/pig/datasets/Unibo/excel")

OUT = W8 / "outputs" / "v62a2_canonical_video_time_sheet_lock"
NOTES = W8 / "notes"
REPORTS = W8 / "reports"
PROGRESS = W8 / "progress"

for p in [OUT, NOTES, REPORTS, PROGRESS]:
    p.mkdir(parents=True, exist_ok=True)

OUT_VIDEO_LOCK = OUT / "week8_v62a2_video_time_sheet_lock.csv"
OUT_EXCEL_TIME_BLOCKS = OUT / "week8_v62a2_excel_time_blocks_inventory.csv"
OUT_RELEVANT_EXCEL_BLOCKS = OUT / "week8_v62a2_relevant_excel_blocks.csv"
OUT_DECISION = OUT / "week8_v62a2_decision_summary.csv"
OUT_ISSUES = OUT / "week8_v62a2_issues.csv"
OUT_NOTE = NOTES / "week8_v62a2_canonical_video_time_sheet_lock_notes.md"
OUT_REPORT = REPORTS / "week8_v62a2_canonical_video_time_sheet_lock_report.md"
OUT_PROGRESS = PROGRESS / "week8_experiment_progress_log.csv"


def safe_to_csv(df, path):
    df.to_csv(path, index=False, quoting=csv.QUOTE_ALL, escapechar="\\", lineterminator="\n")


def clean_str(x):
    if pd.isna(x):
        return ""
    s = str(x).strip()
    if s.lower() in ["nan", "none", "null"]:
        return ""
    return s


def parse_video_id(video_id):
    s = clean_str(video_id)
    sl = s.lower()

    out = {
        "video_id": s,
        "camera": "",
        "room": "",
        "pen": "",
        "date_yyyymmdd": "",
        "start_hour": "",
        "end_hour": "",
        "canonical_day_file_hint": "",
        "canonical_sheet_hint": "",
        "parse_method": "",
    }

    # Pattern 1: TLC 1 -B1 0700-0800 / TLC1 B1 1000-1100
    m = re.search(r"tlc\s*([0-9]+).*?([bcm])\s*([0-9]+).*?(\d{3,4})[-_]?(\d{3,4})", sl)
    if m:
        cam = m.group(1)
        room = m.group(2).upper()
        pen = m.group(3)
        start = m.group(4).zfill(4)
        end = m.group(5).zfill(4)

        out.update({
            "camera": cam,
            "room": room,
            "pen": pen,
            "start_hour": start[:2],
            "end_hour": end[:2],
            "canonical_day_file_hint": "Giorno 1 - 22_7_2021 Tutti.xlsx",
            "canonical_sheet_hint": f"TLC {cam} {room}{pen}",
            "parse_method": "tlc_room_pen_hour_pattern",
        })
        return out

    # Pattern 2: c0001210722150000
    # Interpreted as c + camera(0001) + date(210722) + time(150000)
    m = re.match(r"c(\d{4})(\d{6})(\d{6})$", sl)
    if m:
        cam_raw = m.group(1)
        date_raw = m.group(2)
        time_raw = m.group(3)

        cam = str(int(cam_raw))
        yy = "20" + date_raw[:2]
        mm = date_raw[2:4]
        dd = date_raw[4:6]
        hour = time_raw[:2]

        out.update({
            "camera": cam,
            "room": "B",
            "pen": "1",
            "date_yyyymmdd": f"{yy}-{mm}-{dd}",
            "start_hour": hour,
            "end_hour": str(int(hour) + 1).zfill(2),
            "canonical_day_file_hint": "Giorno 1 - 22_7_2021 Tutti.xlsx",
            "canonical_sheet_hint": f"TLC {cam} B1",
            "parse_method": "c_camera_date_time_pattern_assume_same_pen_B1",
        })
        return out

    out["parse_method"] = "unparsed"
    return out


def normalize_time_value(v):
    s = clean_str(v).lower()
    s = s.replace(".", ",")
    s = s.replace(":", ",")

    m = re.match(r"^(\d{1,2}),?0{0,2}$", s)
    if m:
        return str(int(m.group(1))).zfill(2)

    m = re.match(r"^(\d{1,2}),00[-–](\d{1,2}),00$", s)
    if m:
        return str(int(m.group(1))).zfill(2)

    return ""


def scan_excel_time_blocks(path, sheet):
    rows = []

    try:
        df = pd.read_excel(path, sheet_name=sheet, header=None, dtype=object)
    except Exception as e:
        return [{
            "excel_path": str(path),
            "sheet": sheet,
            "status": "read_error",
            "error": str(e),
        }]

    for r in range(df.shape[0]):
        for c in range(df.shape[1]):
            val = clean_str(df.iat[r, c])
            if not val:
                continue

            val_l = val.lower()
            hour = ""

            # Direct time labels such as 07,00-08,00 or 15,00-16,00
            m = re.search(r"(\d{1,2})[,.:]00\s*[-–]\s*(\d{1,2})[,.:]00", val_l)
            if m:
                hour = str(int(m.group(1))).zfill(2)

            # Sometimes the row after "Periodo di osservazione" contains 0,10,20...
            if "fascia oraria" in val_l or hour:
                context = []
                for rr in range(max(0, r - 2), min(df.shape[0], r + 8)):
                    vals = []
                    for cc in range(max(0, c - 3), min(df.shape[1], c + 20)):
                        vals.append(clean_str(df.iat[rr, cc]))
                    context.append(f"r{rr}: " + " | ".join(vals))

                rows.append({
                    "excel_path": str(path),
                    "sheet": sheet,
                    "row_index_0based": r,
                    "col_index_0based": c,
                    "cell_value": val,
                    "detected_start_hour": hour,
                    "context": " || ".join(context),
                    "status": "ok",
                    "error": "",
                })

    return rows


issues = []

if not V45_CLIP_OBJECTS.exists():
    issues.append({
        "item": str(V45_CLIP_OBJECTS),
        "issue_type": "hard_missing_clip_objects",
        "issue_detail": "v45 clip objects file is missing.",
        "severity": "hard",
    })

if not EXCEL_DIR.exists():
    issues.append({
        "item": str(EXCEL_DIR),
        "issue_type": "hard_missing_excel_dir",
        "issue_detail": "Original Unibo Excel directory is missing.",
        "severity": "hard",
    })

if issues:
    issues_df = pd.DataFrame(issues)
    safe_to_csv(issues_df, OUT_ISSUES)
    decision = pd.DataFrame([{
        "v62a2_decision": "canonical_video_time_sheet_lock_blocked",
        "hard_issue_count": int((issues_df["severity"] == "hard").sum()),
        "ready_for_v62b_canonical_excel_extraction": False,
        "generated_at": datetime.now().isoformat(timespec="seconds"),
    }])
    safe_to_csv(decision, OUT_DECISION)
    print(decision.to_string(index=False))
    raise SystemExit(1)


clip_objects = pd.read_csv(V45_CLIP_OBJECTS)

video_summary = (
    clip_objects.groupby("video_id")
    .agg(
        scanframe_count=("scan_frame_id", "nunique"),
        object_count=("final_box_id", "count"),
        first_scanframe=("scan_frame_id", "min"),
        last_scanframe=("scan_frame_id", "max"),
    )
    .reset_index()
    .sort_values("first_scanframe")
)

parsed = pd.DataFrame([parse_video_id(v) for v in video_summary["video_id"]])
video_lock = video_summary.merge(parsed, on="video_id", how="left")

video_lock["canonical_excel_path"] = video_lock["canonical_day_file_hint"].apply(
    lambda x: str(EXCEL_DIR / clean_str(x)) if clean_str(x) else ""
)

video_lock["canonical_sheet"] = video_lock["canonical_sheet_hint"]

# Lock status.
def lock_status(row):
    p = Path(clean_str(row["canonical_excel_path"]))
    sheet = clean_str(row["canonical_sheet"])
    if clean_str(row["parse_method"]) == "unparsed":
        return "unparsed_video_id"
    if not p.exists():
        return "missing_excel_file"
    try:
        xl = pd.ExcelFile(p)
        if sheet not in xl.sheet_names:
            return "missing_sheet"
    except Exception:
        return "excel_read_error"
    return "candidate_locked"

video_lock["video_sheet_lock_status"] = video_lock.apply(lock_status, axis=1)

safe_to_csv(video_lock, OUT_VIDEO_LOCK)

# Scan all original Excel files for time block positions.
excel_rows = []
for path in sorted(EXCEL_DIR.glob("*.xlsx")):
    try:
        xl = pd.ExcelFile(path)
    except Exception as e:
        excel_rows.append({
            "excel_path": str(path),
            "sheet": "",
            "status": "excel_read_error",
            "error": str(e),
        })
        continue

    for sheet in xl.sheet_names:
        excel_rows.extend(scan_excel_time_blocks(path, sheet))

time_blocks = pd.DataFrame(excel_rows)
safe_to_csv(time_blocks, OUT_EXCEL_TIME_BLOCKS)

# Relevant blocks for locked videos.
rel_rows = []

for _, v in video_lock.iterrows():
    p = clean_str(v["canonical_excel_path"])
    sheet = clean_str(v["canonical_sheet"])
    hour = clean_str(v["start_hour"])

    tb = time_blocks[
        (time_blocks["excel_path"].astype(str) == p)
        & (time_blocks["sheet"].astype(str) == sheet)
    ].copy()

    if hour:
        tb_hour = tb[tb["detected_start_hour"].astype(str) == hour].copy()
    else:
        tb_hour = pd.DataFrame()

    if len(tb_hour):
        for _, r in tb_hour.iterrows():
            rel_rows.append({
                "video_id": v["video_id"],
                "scanframe_count": v["scanframe_count"],
                "first_scanframe": v["first_scanframe"],
                "last_scanframe": v["last_scanframe"],
                "start_hour": hour,
                "canonical_excel_path": p,
                "canonical_sheet": sheet,
                "matched_time_block": True,
                "time_block_row": r.get("row_index_0based", ""),
                "time_block_col": r.get("col_index_0based", ""),
                "time_block_cell": r.get("cell_value", ""),
                "context": r.get("context", ""),
            })
    else:
        rel_rows.append({
            "video_id": v["video_id"],
            "scanframe_count": v["scanframe_count"],
            "first_scanframe": v["first_scanframe"],
            "last_scanframe": v["last_scanframe"],
            "start_hour": hour,
            "canonical_excel_path": p,
            "canonical_sheet": sheet,
            "matched_time_block": False,
            "time_block_row": "",
            "time_block_col": "",
            "time_block_cell": "",
            "context": "",
        })

relevant_blocks = pd.DataFrame(rel_rows)
safe_to_csv(relevant_blocks, OUT_RELEVANT_EXCEL_BLOCKS)

# Issues.
if (video_lock["video_sheet_lock_status"] != "candidate_locked").any():
    bad = video_lock[video_lock["video_sheet_lock_status"] != "candidate_locked"]
    issues.append({
        "item": "video_sheet_lock",
        "issue_type": "warning_some_videos_not_locked",
        "issue_detail": f"{len(bad)} videos are not locked to an existing Excel file/sheet.",
        "severity": "warning",
    })

missing_time_blocks = relevant_blocks[relevant_blocks["matched_time_block"] == False]
if len(missing_time_blocks):
    issues.append({
        "item": "excel_time_blocks",
        "issue_type": "warning_some_video_hours_not_found_in_excel",
        "issue_detail": f"{len(missing_time_blocks)} video hours were not found as explicit Excel time blocks.",
        "severity": "warning",
    })

issues_df = pd.DataFrame(issues, columns=["item", "issue_type", "issue_detail", "severity"])
safe_to_csv(issues_df, OUT_ISSUES)

hard_issue_count = int((issues_df["severity"] == "hard").sum()) if len(issues_df) else 0
warning_count = int((issues_df["severity"] == "warning").sum()) if len(issues_df) else 0

locked_count = int((video_lock["video_sheet_lock_status"] == "candidate_locked").sum())
timeblock_match_count = int(relevant_blocks["matched_time_block"].sum())

decision = pd.DataFrame([{
    "v62a2_decision": "canonical_video_time_sheet_lock_completed" if hard_issue_count == 0 else "canonical_video_time_sheet_lock_blocked",
    "source_video_count": int(len(video_lock)),
    "candidate_locked_video_count": locked_count,
    "explicit_timeblock_match_count": timeblock_match_count,
    "missing_timeblock_count": int(len(missing_time_blocks)),
    "hard_issue_count": hard_issue_count,
    "warning_count": warning_count,
    "issue_count": int(len(issues_df)),
    "ready_for_v62b_canonical_excel_extraction": bool(hard_issue_count == 0 and locked_count == len(video_lock)),
    "generated_at": datetime.now().isoformat(timespec="seconds"),
}])

safe_to_csv(decision, OUT_DECISION)

OUT_NOTE.write_text(
    "# Week 8 v62a2 Canonical Video Time Sheet Lock\n\n"
    f"- v62a2 decision: {decision.iloc[0]['v62a2_decision']}\n"
    f"- Source videos: {len(video_lock)}\n"
    f"- Candidate locked videos: {locked_count}\n"
    f"- Explicit Excel time-block matches: {timeblock_match_count}\n"
    f"- Missing time-block matches: {len(missing_time_blocks)}\n"
    f"- Ready for v62b canonical Excel extraction: {bool(hard_issue_count == 0 and locked_count == len(video_lock))}\n\n"
    "Important: c0001 video IDs are interpreted as camera 1, date 2021-07-22, start time encoded in filename, same B1 pen context unless contradicted by later evidence.\n"
)

OUT_REPORT.write_text(
    "# Week 8 v62a2 Canonical Video Time Sheet Lock Report\n\n"
    f"Decision: {decision.iloc[0]['v62a2_decision']}\n\n"
    f"Video lock: `{OUT_VIDEO_LOCK}`\n\n"
    f"Excel time blocks: `{OUT_EXCEL_TIME_BLOCKS}`\n\n"
    f"Relevant blocks: `{OUT_RELEVANT_EXCEL_BLOCKS}`\n"
)

progress_row = pd.DataFrame([{
    "date": datetime.now().date().isoformat(),
    "stage": "v62a2",
    "task_name": "Canonical video time sheet lock",
    "status": "PASS" if hard_issue_count == 0 else "BLOCKED",
    "input_summary": str(V45_CLIP_OBJECTS),
    "output_summary": str(OUT),
    "hard_issues": hard_issue_count,
    "warnings": warning_count,
    "next_action": "v62b canonical Excel extraction from locked file/sheet/time blocks.",
}])

if OUT_PROGRESS.exists():
    old = pd.read_csv(OUT_PROGRESS)
    progress = pd.concat([old, progress_row], ignore_index=True)
else:
    progress = progress_row

safe_to_csv(progress, OUT_PROGRESS)

print("Saved:")
print(OUT_VIDEO_LOCK)
print(OUT_EXCEL_TIME_BLOCKS)
print(OUT_RELEVANT_EXCEL_BLOCKS)
print(OUT_DECISION)
print(OUT_ISSUES)
print(OUT_NOTE)

print()
print("=== v62a2 decision ===")
print(decision.to_string(index=False))

print()
print("=== video lock ===")
print(video_lock.to_string(index=False))

print()
print("=== relevant Excel blocks ===")
pd.set_option("display.max_colwidth", 200)
print(relevant_blocks[[
    "video_id",
    "start_hour",
    "canonical_excel_path",
    "canonical_sheet",
    "matched_time_block",
    "time_block_row",
    "time_block_col",
    "time_block_cell",
]].to_string(index=False))

print()
print("=== issues ===")
if len(issues_df):
    print(issues_df.to_string(index=False))
else:
    print("No issues found.")
