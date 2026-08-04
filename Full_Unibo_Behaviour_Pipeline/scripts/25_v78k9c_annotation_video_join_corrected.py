from pathlib import Path
from datetime import datetime, timedelta
import re
import csv
import pandas as pd

ROOT = Path.home() / "PigBench"
FULL = ROOT / "Full_Unibo_Behaviour_Pipeline"

FROZEN = FULL / "outputs/v78k5b_corrected_full_frozen_annotation_truth_table/Full_Unibo_Corrected_Full_Frozen_Annotation_Truth_Table/v78k5b_corrected_full_frozen_annotation_truth_table.csv"
V77C = FULL / "outputs/v77c_layout_aware_excel_decoder/Full_Unibo_Layout_Aware_Excel_Decoder/v77c_layout_aware_annotation_windows.csv"
VIDEO_FINAL = FULL / "outputs/v78k8_full_video_manual_overlay_mapping/Full_Unibo_Full_Video_Manual_Overlay_Mapping/v78k8_FULL_VIDEO_MAPPING_FINAL.csv"

OUT = FULL / "outputs/v78k9c_annotation_video_join_Video_Manual_Overlay_Mapping/v78k8_FULL_VIDEO_MAPPING_FINAL.csv"

OUT = FULL / "outputs/v78k9c_annotation_video_join_corrected"
OUT.mkdir(parents=True, exist_ok=True)

OUT_JOIN = OUT / "v78k9c_annotation_to_video_join_table.csv"
OUT_READY = OUT / "v78k9c_ready_annotation_video_rows.csv"
OUT_UNMATCHED = OUT / "v78k9c_unmatched_annotation_rows.csv"
OUT_VIDEO_INTERVALS = OUT / "v78k9c_video_intervals.csv"
OUT_COVERAGE = OUT / "v78k9c_coverage_by_target.csv"
OUT_DECISION = OUT / "v78k9c_decision_summary.csv"
OUT_ISSUES = OUT / "v78k9c_issues.csv"
NOTE = FULL / "notes/v78k9c_annotation_video_join_corrected_notes.md"
NOTE.parent.mkdir(parents=True, exist_ok=True)

FORCED_ANNOTATION_TIME_COLUMN = "v77c__absolute_start_hhmmss"
FRIENDLY_MISSING_DATE_INFERRED_AS = "2021-07-22"

def clean(x):
    if pd.isna(x):
        return ""
    s = str(x).strip()
    if s.lower() in {"nan", "none", "null"}:
        return ""
    return s

def read_csv(path):
    df = pd.read_csv(path).fillna("")
    for c in df.columns:
        if df[c].dtype == object:
            df[c] = df[c].map(clean)
    return df

def write(df, path):
    df.to_csv(path, index=False, quoting=csv.QUOTE_ALL, escapechar="\\", lineterminator="\n")

def parse_time(x):
    s = clean(x)
    if not s:
        return ""

    m = re.search(r"\b([01]?\d|2[0-3])[:.](\d{2})(?:[:.](\d{2}))?\b", s)
    if m:
        return f"{int(m.group(1)):02d}:{int(m.group(2)):02d}:{int(m.group(3) or 0):02d}"

    if re.fullmatch(r"\d{3,4}", s):
        s = s.zfill(4)
        hh, mm = int(s[:2]), int(s[2:])
        if 0 <= hh <= 23 and 0 <= mm <= 59:
            return f"{hh:02d}:{mm:02d}:00"

    if re.fullmatch(r"\d{6}", s):
        hh, mm, ss = int(s[:2]), int(s[2:4]), int(s[4:])
        if 0 <= hh <= 23 and 0 <= mm <= 59 and 0 <= ss <= 59:
            return f"{hh:02d}:{mm:02d}:{ss:02d}"

    return ""

def parse_end_time_from_filename(name, start_time):
    s = clean(name)

    # examples: 0700-0800, 800-900, 1000-1100
    m = re.search(r"(\d{3,4})\s*[-_]\s*(\d{3,4})", s)
    if m:
        end = parse_time(m.group(2))
        if end:
            return end

    if start_time:
        try:
            dummy_date = "2021-01-01"
            start_dt = datetime.fromisoformat(f"{dummy_date}T{start_time}")
            end_dt = start_dt + timedelta(minutes=60)
            return end_dt.strftime("%H:%M:%S")
        except Exception:
            return ""
    return ""

def to_dt(date, time_s):
    date = clean(date)
    time_s = clean(time_s)
    if not date or not time_s:
        return None
    try:
        return datetime.fromisoformat(f"{date}T{time_s}")
    except Exception:
        return None

issues = []

frozen = read_csv(FROZEN)
v77c = read_csv(V77C)
videos = read_csv(VIDEO_FINAL)

ann = frozen.copy()

# Merge raw v77c columns to get exact absolute_start_hhmmss.
raw = v77c.copy()
raw["source_row_index"] = range(len(raw))
raw = raw.rename(columns={c: "v77c__" + c for c in raw.columns if c != "source_row_index"})
ann = ann.merge(raw, on="source_row_index", how="left")

if FORCED_ANNOTATION_TIME_COLUMN not in ann.columns:
    issues.append({
        "item": "annotation_time_column",
        "issue_type": "hard_missing_absolute_start_hhmmss",
        "severity": "hard",
        "detail": f"Missing required column {FORCED_ANNOTATION_TIME_COLUMN}"
    })

# Build video intervals from final manual overlay mapping.
video_rows = []

for _, r in videos.iterrows():
    if clean(r.get("manual_review_status")) != "RESOLVED":
        continue

    video_type = clean(r.get("video_type"))
    date = clean(r.get("filename_date"))

    date_source = "filename_date"
    if not date and video_type == "friendly_tlc":
        date = FRIENDLY_MISSING_DATE_INFERRED_AS
        date_source = "inferred_for_friendly_tlc_missing_date"

    start_time = parse_time(clean(r.get("filename_start_time")))
    end_time = parse_end_time_from_filename(clean(r.get("video_filename")), start_time)

    start_dt = to_dt(date, start_time)
    end_dt = to_dt(date, end_time)

    if start_dt and end_dt and end_dt <= start_dt:
        end_dt = start_dt + timedelta(minutes=60)
        end_time = end_dt.strftime("%H:%M:%S")

    video_rows.append({
        "video_id": clean(r.get("video_id")),
        "video_filename": clean(r.get("video_filename")),
        "video_path": clean(r.get("video_path")),
        "video_type": video_type,
        "filename_c_code": clean(r.get("filename_c_code")),
        "video_date": date,
        "video_date_source": date_source,
        "video_start_time": start_time,
        "video_end_time": end_time,
        "video_start_dt": start_dt,
        "video_end_dt": end_dt,
        "tlc_camera": clean(r.get("manual_tlc_camera")),
        "room_pen": clean(r.get("manual_room_pen")),
        "overlay_text": clean(r.get("manual_overlay_text")),
        "confidence": clean(r.get("manual_confidence")),
    })

video_intervals = pd.DataFrame(video_rows)
write(video_intervals.drop(columns=["video_start_dt", "video_end_dt"], errors="ignore"), OUT_VIDEO_INTERVALS)

friendly_inferred = int((video_intervals["video_date_source"] == "inferred_for_friendly_tlc_missing_date").sum()) if len(video_intervals) else 0
if friendly_inferred:
    issues.append({
        "item": "friendly_video_date",
        "issue_type": "info_friendly_dates_inferred",
        "severity": "info",
        "detail": f"{friendly_inferred} friendly TLC videos had missing date and were assigned {FRIENDLY_MISSING_DATE_INFERRED_AS}."
    })

join_rows = []

for _, a in ann.iterrows():
    ann_date = clean(a.get("date"))
    ann_tlc = clean(a.get("tlc_camera"))
    ann_pen = clean(a.get("resolved_room_pen"))
    ann_time_raw = clean(a.get(FORCED_ANNOTATION_TIME_COLUMN))
    ann_time = parse_time(ann_time_raw)
    ann_dt = to_dt(ann_date, ann_time)

    base = {
        "annotation_window_id": clean(a.get("annotation_window_id")),
        "source_row_index": clean(a.get("source_row_index")),
        "date": ann_date,
        "tlc_camera": ann_tlc,
        "resolved_room_pen": ann_pen,
        "identity_colour": clean(a.get("identity_colour")),
        "behaviour_label": clean(a.get("behaviour_label")),
        "sheet_name": clean(a.get("sheet_name")),
        "annotation_time_column": FORCED_ANNOTATION_TIME_COLUMN,
        "annotation_start_time": ann_time,
    }

    candidates = pd.DataFrame()

    if ann_dt is not None and len(video_intervals):
        sub = video_intervals[
            (video_intervals["video_date"] == ann_date) &
            (video_intervals["tlc_camera"] == ann_tlc) &
            (video_intervals["room_pen"] == ann_pen)
        ].copy()

        if len(sub):
            hit_indices = []
            for idx, v in sub.iterrows():
                if v["video_start_dt"] is not None and v["video_end_dt"] is not None:
                    if v["video_start_dt"] <= ann_dt < v["video_end_dt"]:
                        hit_indices.append(idx)
            candidates = sub.loc[hit_indices].copy() if hit_indices else pd.DataFrame()

    if candidates.empty:
        join_rows.append({
            **base,
            "matched_video_id": "",
            "matched_video_filename": "",
            "matched_video_path": "",
            "matched_c_code": "",
            "matched_video_type": "",
            "matched_video_start_time": "",
            "matched_video_end_time": "",
            "candidate_video_count": 0,
            "all_candidate_video_filenames": "",
            "video_mapping_status": "UNMATCHED",
            "tracking_preparation_status": "NOT_READY_NO_MATCHED_VIDEO",
        })
    else:
        # Prefer encoded c-code videos when both encoded and friendly exist for the same window.
        candidates["video_type_rank"] = candidates["video_type"].map(lambda x: 0 if x == "encoded_c_code" else 1)
        candidates["delta_seconds"] = candidates["video_start_dt"].map(lambda x: abs((ann_dt - x).total_seconds()) if x else 10**9)
        candidates = candidates.sort_values(["video_type_rank", "delta_seconds", "video_start_time"])

        chosen = candidates.iloc[0]

        join_rows.append({
            **base,
            "matched_video_id": clean(chosen["video_id"]),
            "matched_video_filename": clean(chosen["video_filename"]),
            "matched_video_path": clean(chosen["video_path"]),
            "matched_c_code": clean(chosen["filename_c_code"]),
            "matched_video_type": clean(chosen["video_type"]),
            "matched_video_start_time": clean(chosen["video_start_time"]),
            "matched_video_end_time": clean(chosen["video_end_time"]),
            "candidate_video_count": int(len(candidates)),
            "all_candidate_video_filenames": ";".join(candidates["video_filename"].tolist()),
            "video_mapping_status": "MATCHED",
            "tracking_preparation_status": "READY_FOR_TRACKING_PREPARATION",
        })

join = pd.DataFrame(join_rows)
ready = join[join["video_mapping_status"] == "MATCHED"].copy()
unmatched = join[join["video_mapping_status"] != "MATCHED"].copy()

write(join, OUT_JOIN)
write(ready, OUT_READY)
write(unmatched, OUT_UNMATCHED)

coverage = (
    join.groupby(["date", "tlc_camera", "resolved_room_pen"], dropna=False)
    .agg(
        annotation_rows=("annotation_window_id", "count"),
        matched_rows=("video_mapping_status", lambda x: int((x == "MATCHED").sum())),
        unmatched_rows=("video_mapping_status", lambda x: int((x != "MATCHED").sum())),
        matched_c_codes=("matched_c_code", lambda x: ";".join(sorted(set([clean(v) for v in x if clean(v)])))),
        matched_video_types=("matched_video_type", lambda x: ";".join(sorted(set([clean(v) for v in x if clean(v)])))),
    )
    .reset_index()
)
coverage["match_ratio"] = coverage["matched_rows"] / coverage["annotation_rows"]
write(coverage, OUT_COVERAGE)

total = len(join)
matched = len(ready)
unmatched_count = len(unmatched)
match_ratio = matched / total if total else 0

if matched == 0:
    issues.append({
        "item": "join",
        "issue_type": "hard_no_matches",
        "severity": "hard",
        "detail": "No annotation rows matched videos."
    })
elif unmatched_count:
    issues.append({
        "item": "join",
        "issue_type": "warning_some_unmatched",
        "severity": "warning",
        "detail": f"{unmatched_count} annotation rows unmatched. This is expected for dates/pens with no reviewed video."
    })

issues.append({
    "item": "scope",
    "issue_type": "info_join_only",
    "severity": "info",
    "detail": "v78k9c joins annotation truth to reviewed video mapping. It does not run tracking."
})

issues_df = pd.DataFrame(issues)
write(issues_df, OUT_ISSUES)

hard_count = int((issues_df["severity"] == "hard").sum()) if "severity" in issues_df.columns else 0
warning_count = int((issues_df["severity"] == "warning").sum()) if "severity" in issues_df.columns else 0

decision = pd.DataFrame([{
    "v78k9c_decision": "corrected_annotation_video_join_created" if hard_count == 0 else "corrected_annotation_video_join_has_blocking_issues",
    "supersedes": "v78k9b_annotation_video_join_compact",
    "frozen_annotation_rows": len(frozen),
    "final_video_mapping_rows": len(videos),
    "annotation_join_rows": total,
    "matched_annotation_rows": matched,
    "unmatched_annotation_rows": unmatched_count,
    "match_ratio": round(match_ratio, 6),
    "annotation_time_column_used": FORCED_ANNOTATION_TIME_COLUMN,
    "friendly_missing_date_inferred_as": FRIENDLY_MISSING_DATE_INFERRED_AS,
    "friendly_video_dates_inferred_count": friendly_inferred,
    "ready_rows_csv": str(OUT_READY),
    "unmatched_rows_csv": str(OUT_UNMATCHED),
    "hard_issue_count": hard_count,
    "warning_count": warning_count,
    "ready_for_tracking_preparation": bool(hard_count == 0 and matched > 0),
    "ready_for_full_tracking": False,
    "claim_scope": "corrected_annotation_video_join_only",
    "generated_at": datetime.now().isoformat(timespec="seconds"),
}])
write(decision, OUT_DECISION)

NOTE.write_text(
    "# v78k9c Corrected Annotation-Video Join\n\n"
    f"- Decision: {decision.iloc[0]['v78k9c_decision']}\n"
    f"- Supersedes: v78k9b_annotation_video_join_compact\n"
    f"- Frozen annotation rows: {len(frozen)}\n"
    f"- Final video mapping rows: {len(videos)}\n"
    f"- Join rows: {total}\n"
    f"- Matched rows: {matched}\n"
    f"- Unmatched rows: {unmatched_count}\n"
    f"- Match ratio: {match_ratio:.6f}\n"
    f"- Annotation time column used: {FORCED_ANNOTATION_TIME_COLUMN}\n"
    f"- Friendly missing dates inferred as: {FRIENDLY_MISSING_DATE_INFERRED_AS}\n"
    f"- Friendly inferred count: {friendly_inferred}\n"
    f"- Hard issues: {hard_count}\n"
    f"- Ready for tracking preparation: {bool(hard_count == 0 and matched > 0)}\n\n"
    "This corrected join uses exact absolute annotation start time and manually reviewed video overlay mapping. It does not run tracking.\n",
    encoding="utf-8"
)

print("=== decision ===")
print(decision.to_string(index=False))

print("\n=== coverage ===")
print(coverage.to_string(index=False))

print("\n=== video intervals summary ===")
print(video_intervals.groupby(["video_date","tlc_camera","room_pen","video_type"]).size().reset_index(name="video_count").to_string(index=False))

print("\n=== issues ===")
print(issues_df.to_string(index=False))
