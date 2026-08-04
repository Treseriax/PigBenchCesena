from pathlib import Path
from datetime import datetime, timedelta
import re
import csv
import json
import hashlib
import zipfile
import pandas as pd

ROOT = Path.home() / "PigBench"
FULL = ROOT / "Full_Unibo_Behaviour_Pipeline"

FROZEN = FULL / "outputs" / "v78k5b_corrected_full_frozen_annotation_truth_table" / "Full_Unibo_Corrected_Full_Frozen_Annotation_Truth_Table" / "v78k5b_corrected_full_frozen_annotation_truth_table.csv"
V77C_FULL = FULL / "outputs" / "v77c_layout_aware_excel_decoder" / "Full_Unibo_Layout_Aware_Excel_Decoder" / "v77c_layout_aware_annotation_windows.csv"
VIDEO_FINAL = FULL / "outputs" / "v78k8_full_video_manual_overlay_mapping" / "Full_Unibo_Full_Video_Manual_Overlay_Mapping" / "v78k8_FULL_VIDEO_MAPPING_FINAL.csv"

OUT = FULL / "outputs" / "v78k9_annotation_video_join"
PKG = OUT / "Full_Unibo_Annotation_Video_Join"
NOTES = FULL / "notes"
REPORTS = FULL / "reports"
PROGRESS = FULL / "progress"

for p in [OUT, PKG, NOTES, REPORTS, PROGRESS]:
    p.mkdir(parents=True, exist_ok=True)

OUT_TIME_REPORT = PKG / "v78k9_annotation_time_column_selection_report.csv"
OUT_VIDEO_INTERVALS = PKG / "v78k9_video_intervals_from_final_mapping.csv"
OUT_JOIN = PKG / "v78k9_annotation_to_video_join_table.csv"
OUT_READY = PKG / "v78k9_ready_annotation_video_rows.csv"
OUT_UNMATCHED = PKG / "v78k9_unmatched_annotation_rows.csv"
OUT_COVERAGE_TARGET = PKG / "v78k9_coverage_by_date_tlc_pen.csv"
OUT_COVERAGE_VIDEO = PKG / "v78k9_coverage_by_video.csv"
OUT_DECISION = OUT / "v78k9_decision_summary.csv"
OUT_ISSUES = OUT / "v78k9_issues.csv"
OUT_README = PKG / "README_v78k9_Annotation_Video_Join.md"
OUT_MANIFEST = PKG / "v78k9_manifest.json"
OUT_ZIP = OUT / "Full_Unibo_Annotation_Video_Join.zip"
OUT_SHA = OUT / "Full_Unibo_Annotation_Video_Join.sha256"
OUT_NOTE = NOTES / "v78k9_annotation_video_join_notes.md"
OUT_REPORT = REPORTS / "v78k9_annotation_video_join_report.md"
OUT_PROGRESS = PROGRESS / "full_unibo_pipeline_progress_log.csv"

VIDEO_DURATION_MINUTES_DEFAULT = 60

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

def parse_date_anywhere(text):
    s = clean(text)

    m = re.search(r"(20\d{2})[-_](\d{1,2})[-_](\d{1,2})", s)
    if m:
        y, mo, d = m.groups()
        return f"{int(y):04d}-{int(mo):02d}-{int(d):02d}"

    m = re.search(r"\b(\d{1,2})[-_](\d{1,2})[-_](20\d{2})\b", s)
    if m:
        d, mo, y = m.groups()
        return f"{int(y):04d}-{int(mo):02d}-{int(d):02d}"

    return ""

def parse_hhmmss(text):
    s = clean(text)
    if not s:
        return ""

    # HH:MM[:SS] or HH.MM
    m = re.search(r"\b([01]?\d|2[0-3])[:.](\d{2})(?:[:.](\d{2}))?\b", s)
    if m:
        hh = int(m.group(1))
        mm = int(m.group(2))
        ss = int(m.group(3) or 0)
        return f"{hh:02d}:{mm:02d}:{ss:02d}"

    # 0700 or 070000, but avoid interpreting long dates/codes as time.
    if re.fullmatch(r"\d{4}", s):
        hh = int(s[:2])
        mm = int(s[2:])
        if 0 <= hh <= 23 and 0 <= mm <= 59:
            return f"{hh:02d}:{mm:02d}:00"

    if re.fullmatch(r"\d{6}", s):
        hh = int(s[:2])
        mm = int(s[2:4])
        ss = int(s[4:])
        if 0 <= hh <= 23 and 0 <= mm <= 59 and 0 <= ss <= 59:
            return f"{hh:02d}:{mm:02d}:{ss:02d}"

    return ""

def time_to_dt(date, time_s):
    date = clean(date)
    time_s = clean(time_s)
    if not date or not time_s:
        return None
    try:
        return datetime.fromisoformat(f"{date}T{time_s}")
    except Exception:
        return None

def is_time_like_col(c):
    cl = c.lower()
    keys = ["time", "start", "end", "hour", "slot", "window", "sec", "sequence"]
    return any(k in cl for k in keys)

def score_time_columns(df):
    rows = []
    for c in df.columns:
        if not is_time_like_col(c):
            continue

        vals = df[c].astype(str).map(clean)
        parsed = vals.map(parse_hhmmss)
        parse_count = int((parsed != "").sum())
        nonempty = int((vals != "").sum())

        prefer = 0
        cl = c.lower()
        if "start" in cl:
            prefer += 5
        if "window" in cl:
            prefer += 4
        if "slot" in cl:
            prefer += 3
        if "end" in cl:
            prefer -= 2
        if "duration" in cl:
            prefer -= 5

        rows.append({
            "column_source": "candidate",
            "column_name": c,
            "nonempty_count": nonempty,
            "parse_count": parse_count,
            "parse_ratio": parse_count / len(df) if len(df) else 0,
            "preference_score": prefer,
            "sample_values": ";".join(vals.drop_duplicates().head(20).tolist()),
        })

    rep = pd.DataFrame(rows)
    if rep.empty:
        return rep, ""

    rep["rank_score"] = rep["parse_count"] * 100 + rep["preference_score"]
    rep = rep.sort_values("rank_score", ascending=False)

    best = clean(rep.iloc[0]["column_name"]) if int(rep.iloc[0]["parse_count"]) > 0 else ""
    return rep, best

def get_value_safe(row, col):
    if col and col in row.index:
        return clean(row[col])
    return ""

issues = []

frozen = read_csv(FROZEN)
v77c = read_csv(V77C_FULL)
videos = read_csv(VIDEO_FINAL)

if frozen.empty:
    issues.append({
        "item": str(FROZEN),
        "issue_type": "hard_missing_frozen_annotation_truth",
        "issue_detail": "v78k5b frozen annotation truth table missing or empty.",
        "severity": "hard",
    })

if videos.empty:
    issues.append({
        "item": str(VIDEO_FINAL),
        "issue_type": "hard_missing_final_video_mapping",
        "issue_detail": "v78k8 final video mapping table missing or empty.",
        "severity": "hard",
    })

# Validate video mapping final.
required_video_cols = [
    "video_id", "video_filename", "video_path", "video_type",
    "filename_date", "filename_start_time",
    "manual_tlc_camera", "manual_room_pen", "manual_review_status"
]
for c in required_video_cols:
    if not videos.empty and c not in videos.columns:
        issues.append({
            "item": c,
            "issue_type": "hard_missing_video_mapping_column",
            "issue_detail": f"Missing required v78k8 final column: {c}",
            "severity": "hard",
        })

if not videos.empty and "manual_review_status" in videos.columns:
    unreviewed = int((videos["manual_review_status"] == "UNREVIEWED").sum())
    unresolved = int((videos["manual_review_status"] != "RESOLVED").sum())
    if unreviewed:
        issues.append({
            "item": "video_mapping",
            "issue_type": "hard_unreviewed_videos_remain",
            "issue_detail": f"{unreviewed} videos remain UNREVIEWED.",
            "severity": "hard",
        })
    if unresolved:
        issues.append({
            "item": "video_mapping",
            "issue_type": "warning_some_videos_not_resolved",
            "issue_detail": f"{unresolved} videos are not RESOLVED.",
            "severity": "warning",
        })

# Merge v77c extra columns by source_row_index when possible for better time parsing.
ann = frozen.copy()
if not frozen.empty and not v77c.empty and "source_row_index" in frozen.columns:
    v77c_extra = v77c.copy()
    v77c_extra["source_row_index"] = range(len(v77c_extra))
    # Prefix raw v77c columns to avoid collisions.
    v77c_extra = v77c_extra.rename(columns={c: f"v77c__{c}" for c in v77c_extra.columns if c != "source_row_index"})
    ann = ann.merge(v77c_extra, on="source_row_index", how="left")

time_report, best_time_col = score_time_columns(ann)
to_csv(time_report, OUT_TIME_REPORT)

if not best_time_col:
    issues.append({
        "item": "annotation_time_column",
        "issue_type": "hard_no_parseable_annotation_time_column",
        "issue_detail": "No parseable annotation time column was found in frozen/v77c data.",
        "severity": "hard",
    })

ann_rows = []

if not ann.empty:
    for _, r in ann.iterrows():
        date = get_value_safe(r, "date")
        tlc = get_value_safe(r, "tlc_camera")
        pen = get_value_safe(r, "resolved_room_pen")
        start_raw = get_value_safe(r, best_time_col)
        start_time = parse_hhmmss(start_raw)

        ann_rows.append({
            "annotation_window_id": get_value_safe(r, "annotation_window_id"),
            "source_row_index": get_value_safe(r, "source_row_index"),
            "date": date,
            "tlc_camera": tlc,
            "resolved_room_pen": pen,
            "identity_colour": get_value_safe(r, "identity_colour"),
            "behaviour_label": get_value_safe(r, "behaviour_label"),
            "annotation_time_raw_column": best_time_col,
            "annotation_time_raw": start_raw,
            "annotation_start_time": start_time,
            "annotation_datetime": time_to_dt(date, start_time),
            "sheet_name": get_value_safe(r, "sheet_name"),
            "source_excel_file": get_value_safe(r, "source_excel_file"),
        })

ann2 = pd.DataFrame(ann_rows)

# Build video intervals.
video_rows = []

if not videos.empty:
    for _, r in videos.iterrows():
        status = clean(r.get("manual_review_status"))
        if status != "RESOLVED":
            continue

        date = clean(r.get("filename_date")) or parse_date_anywhere(clean(r.get("video_path")) + " " + clean(r.get("video_filename")))
        start_time = parse_hhmmss(clean(r.get("filename_start_time")))
        start_dt = time_to_dt(date, start_time)

        end_dt = start_dt + timedelta(minutes=VIDEO_DURATION_MINUTES_DEFAULT) if start_dt else None

        video_rows.append({
            "video_id": clean(r.get("video_id")),
            "video_filename": clean(r.get("video_filename")),
            "video_path": clean(r.get("video_path")),
            "video_type": clean(r.get("video_type")),
            "filename_c_code": clean(r.get("filename_c_code")),
            "video_date": date,
            "video_start_time": start_time,
            "video_end_time": end_dt.strftime("%H:%M:%S") if end_dt else "",
            "video_start_datetime": start_dt,
            "video_end_datetime": end_dt,
            "manual_tlc_camera": clean(r.get("manual_tlc_camera")),
            "manual_room_pen": clean(r.get("manual_room_pen")),
            "manual_overlay_text": clean(r.get("manual_overlay_text")),
            "manual_confidence": clean(r.get("manual_confidence")),
            "manual_note": clean(r.get("manual_note")),
        })

video_intervals = pd.DataFrame(video_rows)
to_csv(video_intervals.drop(columns=["video_start_datetime", "video_end_datetime"], errors="ignore"), OUT_VIDEO_INTERVALS)

if not video_intervals.empty:
    missing_video_date = int((video_intervals["video_date"] == "").sum())
    missing_video_time = int((video_intervals["video_start_time"] == "").sum())

    if missing_video_date:
        issues.append({
            "item": "video_date",
            "issue_type": "warning_some_video_dates_missing",
            "issue_detail": f"{missing_video_date} resolved videos have missing date.",
            "severity": "warning",
        })

    if missing_video_time:
        issues.append({
            "item": "video_start_time",
            "issue_type": "warning_some_video_start_times_missing",
            "issue_detail": f"{missing_video_time} resolved videos have missing start time.",
            "severity": "warning",
        })

# Join annotation rows to videos by date + TLC + pen + interval.
join_rows = []

if not ann2.empty and not video_intervals.empty:
    for _, a in ann2.iterrows():
        a_dt = a["annotation_datetime"]
        candidates = pd.DataFrame()

        if a_dt is not None:
            mask = (
                (video_intervals["video_date"] == clean(a["date"])) &
                (video_intervals["manual_tlc_camera"] == clean(a["tlc_camera"])) &
                (video_intervals["manual_room_pen"] == clean(a["resolved_room_pen"]))
            )

            subset = video_intervals[mask].copy()
            if not subset.empty:
                # interval containment
                cand_idx = []
                for vi, v in subset.iterrows():
                    if v["video_start_datetime"] is not None and v["video_end_datetime"] is not None:
                        if v["video_start_datetime"] <= a_dt < v["video_end_datetime"]:
                            cand_idx.append(vi)
                candidates = subset.loc[cand_idx].copy() if cand_idx else pd.DataFrame()

        if candidates.empty:
            join_rows.append({
                **{k: v for k, v in a.items() if k != "annotation_datetime"},
                "matched_video_count": 0,
                "matched_video_ids": "",
                "matched_video_filenames": "",
                "matched_c_codes": "",
                "matched_video_types": "",
                "video_mapping_status": "UNMATCHED_NO_VIDEO_FOR_DATE_TIME_TLC_PEN",
                "tracking_preparation_status": "NOT_READY_NO_MATCHED_VIDEO",
            })
        else:
            join_rows.append({
                **{k: v for k, v in a.items() if k != "annotation_datetime"},
                "matched_video_count": int(len(candidates)),
                "matched_video_ids": ";".join(candidates["video_id"].tolist()),
                "matched_video_filenames": ";".join(candidates["video_filename"].tolist()),
                "matched_c_codes": ";".join(sorted(set([clean(x) for x in candidates["filename_c_code"].tolist() if clean(x)]))),
                "matched_video_types": ";".join(sorted(set([clean(x) for x in candidates["video_type"].tolist() if clean(x)]))),
                "video_mapping_status": "MATCHED_TO_REVIEWED_VIDEO",
                "tracking_preparation_status": "READY_FOR_TRACKING_PREPARATION",
            })

join_df = pd.DataFrame(join_rows)
to_csv(join_df, OUT_JOIN)

ready = join_df[join_df["video_mapping_status"] == "MATCHED_TO_REVIEWED_VIDEO"].copy() if not join_df.empty else pd.DataFrame()
unmatched = join_df[join_df["video_mapping_status"] != "MATCHED_TO_REVIEWED_VIDEO"].copy() if not join_df.empty else pd.DataFrame()

to_csv(ready, OUT_READY)
to_csv(unmatched, OUT_UNMATCHED)

if not join_df.empty:
    coverage_target = (
        join_df.groupby(["date", "tlc_camera", "resolved_room_pen"], dropna=False)
        .agg(
            annotation_rows=("annotation_window_id", "count"),
            matched_rows=("video_mapping_status", lambda x: int((x == "MATCHED_TO_REVIEWED_VIDEO").sum())),
            unmatched_rows=("video_mapping_status", lambda x: int((x != "MATCHED_TO_REVIEWED_VIDEO").sum())),
            matched_c_codes=("matched_c_codes", lambda x: ";".join(sorted(set(";".join([clean(v) for v in x]).split(";")) - {""}))),
            matched_video_types=("matched_video_types", lambda x: ";".join(sorted(set(";".join([clean(v) for v in x]).split(";")) - {""}))),
        )
        .reset_index()
    )
    coverage_target["match_ratio"] = coverage_target["matched_rows"] / coverage_target["annotation_rows"]
else:
    coverage_target = pd.DataFrame()

to_csv(coverage_target, OUT_COVERAGE_TARGET)

if not ready.empty:
    rows = []
    for _, r in ready.iterrows():
        ids = [x for x in clean(r["matched_video_ids"]).split(";") if x]
        for vid in ids:
            rows.append({
                "video_id": vid,
                "annotation_window_id": clean(r["annotation_window_id"]),
                "date": clean(r["date"]),
                "tlc_camera": clean(r["tlc_camera"]),
                "resolved_room_pen": clean(r["resolved_room_pen"]),
                "behaviour_label": clean(r["behaviour_label"]),
            })
    tmp = pd.DataFrame(rows)
    coverage_video = (
        tmp.groupby(["video_id", "date", "tlc_camera", "resolved_room_pen"], dropna=False)
        .agg(
            annotation_rows=("annotation_window_id", "count"),
            behaviour_labels=("behaviour_label", lambda x: ";".join(sorted(set([clean(v) for v in x if clean(v)])))),
        )
        .reset_index()
    )
else:
    coverage_video = pd.DataFrame()

to_csv(coverage_video, OUT_COVERAGE_VIDEO)

# Issues / decision.
if not join_df.empty:
    total = len(join_df)
    matched = int((join_df["video_mapping_status"] == "MATCHED_TO_REVIEWED_VIDEO").sum())
    unmatched_count = total - matched
    match_ratio = matched / total if total else 0

    if matched == 0:
        issues.append({
            "item": "annotation_video_join",
            "issue_type": "hard_no_annotation_rows_matched_to_video",
            "issue_detail": "No annotation rows matched reviewed videos.",
            "severity": "hard",
        })
    elif unmatched_count:
        issues.append({
            "item": "annotation_video_join",
            "issue_type": "warning_some_annotation_rows_unmatched",
            "issue_detail": f"{unmatched_count} annotation rows did not match reviewed videos.",
            "severity": "warning",
        })
else:
    total = 0
    matched = 0
    unmatched_count = 0
    match_ratio = 0
    issues.append({
        "item": "annotation_video_join",
        "issue_type": "hard_empty_join_output",
        "issue_detail": "Annotation-video join output is empty.",
        "severity": "hard",
    })

issues.append({
    "item": "scope",
    "issue_type": "info_join_only",
    "issue_detail": "v78k9 joins annotation truth to final video mapping. It does not run detector/tracker.",
    "severity": "info",
})

issues_df = pd.DataFrame(issues, columns=["item", "issue_type", "issue_detail", "severity"])
to_csv(issues_df, OUT_ISSUES)

hard_count = int((issues_df["severity"] == "hard").sum()) if len(issues_df) else 0
warning_count = int((issues_df["severity"] == "warning").sum()) if len(issues_df) else 0
info_count = int((issues_df["severity"] == "info").sum()) if len(issues_df) else 0

readme = """# v78k9 Annotation-to-Video Join

Inputs:
- v78k5b corrected frozen annotation truth
- v78k8 final full-video manual overlay mapping

Join key:
- date
- TLC camera
- room/pen
- annotation time contained in video interval

Boundary:
- This stage does not run detector/tracker.
- It only prepares a clean annotation-to-video mapping table.
"""
OUT_README.write_text(readme, encoding="utf-8")
OUT_REPORT.write_text(readme, encoding="utf-8")

manifest = {
    "version": "v78k9_annotation_video_join",
    "generated_at": datetime.now().isoformat(timespec="seconds"),
    "frozen_rows": int(len(frozen)),
    "video_rows": int(len(videos)),
    "annotation_rows_joined": int(total),
    "matched_rows": int(matched),
    "unmatched_rows": int(unmatched_count),
    "match_ratio": float(match_ratio),
    "best_annotation_time_column": best_time_col,
    "hard_issue_count": hard_count,
    "warning_count": warning_count,
    "claim_boundary": "annotation-video join only; no tracking",
}
OUT_MANIFEST.write_text(json.dumps(manifest, indent=2, default=str), encoding="utf-8")

if OUT_ZIP.exists():
    OUT_ZIP.unlink()

with zipfile.ZipFile(OUT_ZIP, "w", compression=zipfile.ZIP_DEFLATED) as z:
    for p in sorted(PKG.rglob("*")):
        if p.is_file():
            z.write(p, p.relative_to(OUT))

zip_hash = sha256_file(OUT_ZIP)
OUT_SHA.write_text(f"{zip_hash}  {OUT_ZIP.name}\n")

decision = pd.DataFrame([{
    "v78k9_decision": "annotation_video_join_created" if hard_count == 0 else "annotation_video_join_has_blocking_issues",
    "frozen_annotation_rows": int(len(frozen)),
    "final_video_mapping_rows": int(len(videos)),
    "annotation_join_rows": int(total),
    "matched_annotation_rows": int(matched),
    "unmatched_annotation_rows": int(unmatched_count),
    "match_ratio": float(match_ratio),
    "best_annotation_time_column": best_time_col,
    "ready_rows_csv": str(OUT_READY),
    "unmatched_rows_csv": str(OUT_UNMATCHED),
    "zip_path": str(OUT_ZIP),
    "zip_sha256": zip_hash,
    "hard_issue_count": hard_count,
    "warning_count": warning_count,
    "info_count": info_count,
    "ready_for_tracking_preparation": bool(hard_count == 0 and matched > 0),
    "ready_for_full_tracking": False,
    "claim_scope": "annotation_video_join_only",
    "generated_at": datetime.now().isoformat(timespec="seconds"),
}])
to_csv(decision, OUT_DECISION)

OUT_NOTE.write_text(
    "# v78k9 Annotation-to-Video Join\n\n"
    f"- Decision: {decision.iloc[0]['v78k9_decision']}\n"
    f"- Frozen annotation rows: {len(frozen)}\n"
    f"- Final video mapping rows: {len(videos)}\n"
    f"- Join rows: {total}\n"
    f"- Matched rows: {matched}\n"
    f"- Unmatched rows: {unmatched_count}\n"
    f"- Match ratio: {match_ratio:.4f}\n"
    f"- Best annotation time column: {best_time_col}\n"
    f"- Hard issues: {hard_count}\n"
    f"- Ready for tracking preparation: {bool(hard_count == 0 and matched > 0)}\n\n"
    "This stage joins annotation truth to reviewed video identity. It does not run tracking.\n",
    encoding="utf-8"
)

progress_row = pd.DataFrame([{
    "date": datetime.now().date().isoformat(),
    "stage": "v78k9",
    "task_name": "Annotation-video join",
    "status": "PASS" if hard_count == 0 else "NEEDS_FIX",
    "input_summary": "v78k5b frozen annotation truth + v78k8 final video mapping",
    "output_summary": str(PKG),
    "hard_issues": hard_count,
    "warnings": warning_count,
    "next_action": "Use matched annotation-video rows for tracking preparation subset.",
}])

if OUT_PROGRESS.exists():
    old = pd.read_csv(OUT_PROGRESS)
    progress = pd.concat([old, progress_row], ignore_index=True)
else:
    progress = progress_row
to_csv(progress, OUT_PROGRESS)

print("=== v78k9 decision ===")
print(decision.to_string(index=False))

print("\n=== time column report top 20 ===")
print(time_report.head(20).to_string(index=False) if not time_report.empty else "none")

print("\n=== coverage by target sample ===")
print(coverage_target not time_report.empty else "none")

print("\n=== coverage by.head(80).to_string(index=False) if not coverage_target.empty else "none")

print("\n=== video intervals sample ===")
print(video_intervals.drop(columns=['video_start_datetime','video_end_datetime'], errors='ignore').head(80).to_string(index=False) if not video_intervals.empty else "none")

print("\n=== issues ===")
print(issues_df.to_string(index=False))
