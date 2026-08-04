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

OUT = FULL / "outputs/v78k9b_annotation_video_join_compact"
OUT.mkdir(parents=True, exist_ok=True)

OUT_JOIN = OUT / "v78k9b_annotation_to_video_join_table.csv"
OUT_READY = OUT / "v78k9b_ready_annotation_video_rows.csv"
OUT_UNMATCHED = OUT / "v78k9b_unmatched_annotation_rows.csv"
OUT_TIME_REPORT = OUT / "v78k9b_time_column_report.csv"
OUT_VIDEO_INTERVALS = OUT / "v78k9b_video_intervals.csv"
OUT_COVERAGE = OUT / "v78k9b_coverage_by_target.csv"
OUT_DECISION = OUT / "v78k9b_decision_summary.csv"
OUT_ISSUES = OUT / "v78k9b_issues.csv"
NOTE = FULL / "notes/v78k9b_annotation_video_join_compact_notes.md"
NOTE.parent.mkdir(parents=True, exist_ok=True)

def clean(x):
    if pd.isna(x):
        return ""
    s = str(x).strip()
    if s.lower() in {"nan", "none", "null"}:
        return ""
    return s

def read_csv(p):
    df = pd.read_csv(p).fillna("")
    for c in df.columns:
        if df[c].dtype == object:
            df[c] = df[c].map(clean)
    return df

def write(df, p):
    df.to_csv(p, index=False, quoting=csv.QUOTE_ALL, escapechar="\\", lineterminator="\n")

def parse_time(x):
    s = clean(x)
    if not s:
        return ""

    m = re.search(r"\b([01]?\d|2[0-3])[:.](\d{2})(?:[:.](\d{2}))?\b", s)
    if m:
        return f"{int(m.group(1)):02d}:{int(m.group(2)):02d}:{int(m.group(3) or 0):02d}"

    if re.fullmatch(r"\d{4}", s):
        hh, mm = int(s[:2]), int(s[2:])
        if 0 <= hh <= 23 and 0 <= mm <= 59:
            return f"{hh:02d}:{mm:02d}:00"

    if re.fullmatch(r"\d{6}", s):
        hh, mm, ss = int(s[:2]), int(s[2:4]), int(s[4:])
        if 0 <= hh <= 23 and 0 <= mm <= 59 and 0 <= ss <= 59:
            return f"{hh:02d}:{mm:02d}:{ss:02d}"

    return ""

def dt(date, time_s):
    date = clean(date)
    time_s = clean(time_s)
    if not date or not time_s:
        return None
    try:
        return datetime.fromisoformat(f"{date}T{time_s}")
    except Exception:
        return None

def parse_date_from_path(s):
    s = clean(s)

    m = re.search(r"(20\d{2})[-_](\d{1,2})[-_](\d{1,2})", s)
    if m:
        y, mo, d = m.groups()
        return f"{int(y):04d}-{int(mo):02d}-{int(d):02d}"

    m = re.search(r"\b(\d{1,2})[-_](\d{1,2})[-_](20\d{2})\b", s)
    if m:
        d, mo, y = m.groups()
        return f"{int(y):04d}-{int(mo):02d}-{int(d):02d}"

    return ""

def choose_time_column(df):
    rows = []
    for c in df.columns:
        cl = c.lower()
        if not any(k in cl for k in ["time", "start", "slot", "hour"]):
            continue
        if any(k in cl for k in ["duration", "end"]):
            continue

        vals = df[c].astype(str).map(clean)
        parsed = vals.map(parse_time)
        parse_count = int((parsed != "").sum())
        nonempty = int((vals != "").sum())

        prefer = 0
        if "window" in cl: prefer += 5
        if "start" in cl: prefer += 5
        if "slot" in cl: prefer += 3
        if "time" in cl: prefer += 2

        rows.append({
            "column_name": c,
            "nonempty_count": nonempty,
            "parse_count": parse_count,
            "preference": prefer,
            "score": parse_count * 100 + prefer,
            "samples": "; ".join(vals.drop_duplicates().head(12).tolist())
        })

    rep = pd.DataFrame(rows)
    if rep.empty:
        return rep, ""

    rep = rep.sort_values("score", ascending=False)
    best = rep.iloc[0]["column_name"] if int(rep.iloc[0]["parse_count"]) > 0 else ""
    return rep, best

issues = []

frozen = read_csv(FROZEN)
v77c = read_csv(V77C)
videos = read_csv(VIDEO_FINAL)

# v77c raw columns merge, source_row_index üzerinden.
ann = frozen.copy()
if "source_row_index" in ann.columns and not v77c.empty:
    raw = v77c.copy()
    raw["source_row_index"] = range(len(raw))
    raw = raw.rename(columns={c: "v77c__" + c for c in raw.columns if c != "source_row_index"})
    ann = ann.merge(raw, on="source_row_index", how="left")

time_report, best_time_col = choose_time_column(ann)
write(time_report, OUT_TIME_REPORT)

if not best_time_col:
    issues.append({
        "item": "annotation_time",
        "issue_type": "hard_no_parseable_time_column",
        "severity": "hard",
        "detail": "No parseable annotation time column found. Check v78k9b_time_column_report.csv"
    })

# Video intervals from final manual mapping.
video_rows = []
for _, r in videos.iterrows():
    if clean(r.get("manual_review_status")) != "RESOLVED":
        continue

    date = clean(r.get("filename_date"))
    if not date:
        date = parse_date_from_path(clean(r.get("video_path")) + " " + clean(r.get("video_filename")))

    start = parse_time(clean(r.get("filename_start_time")))
    start_dt = dt(date, start)
    end_dt = start_dt + timedelta(minutes=60) if start_dt else None

    video_rows.append({
        "video_id": clean(r.get("video_id")),
        "video_filename": clean(r.get("video_filename")),
        "video_path": clean(r.get("video_path")),
        "video_type": clean(r.get("video_type")),
        "filename_c_code": clean(r.get("filename_c_code")),
        "video_date": date,
        "video_start_time": start,
        "video_end_time": end_dt.strftime("%H:%M:%S") if end_dt else "",
        "video_start_dt": start_dt,
        "video_end_dt": end_dt,
        "tlc_camera": clean(r.get("manual_tlc_camera")),
        "room_pen": clean(r.get("manual_room_pen")),
        "overlay_text": clean(r.get("manual_overlay_text")),
        "confidence": clean(r.get("manual_confidence")),
    })

video_intervals = pd.DataFrame(video_rows)
write(video_intervals.drop(columns=["video_start_dt", "video_end_dt"], errors="ignore"), OUT_VIDEO_INTERVALS)

join_rows = []

for _, a in ann.iterrows():
    ann_date = clean(a.get("date"))
    ann_tlc = clean(a.get("tlc_camera"))
    ann_pen = clean(a.get("resolved_room_pen"))
    ann_time_raw = clean(a.get(best_time_col)) if best_time_col else ""
    ann_time = parse_time(ann_time_raw)
    ann_dt = dt(ann_date, ann_time)

    base = {
        "annotation_window_id": clean(a.get("annotation_window_id")),
        "source_row_index": clean(a.get("source_row_index")),
        "date": ann_date,
        "tlc_camera": ann_tlc,
        "resolved_room_pen": ann_pen,
        "identity_colour": clean(a.get("identity_colour")),
        "behaviour_label": clean(a.get("behaviour_label")),
        "sheet_name": clean(a.get("sheet_name")),
        "annotation_time_column": best_time_col,
        "annotation_time_raw": ann_time_raw,
        "annotation_start_time": ann_time,
    }

    candidates = pd.DataFrame()
    if ann_dt is not None and not video_intervals.empty:
        sub = video_intervals[
            (video_intervals["video_date"] == ann_date) &
            (video_intervals["tlc_camera"] == ann_tlc) &
            (video_intervals["room_pen"] == ann_pen)
        ].copy()

        if not sub.empty:
            idxs = []
            for vi, v in sub.iterrows():
                if v["video_start_dt"] is not None and v["video_end_dt"] is not None:
                    if v["video_start_dt"] <= ann_dt < v["video_end_dt"]:
                        idxs.append(vi)
            candidates = sub.loc[idxs].copy() if idxs else pd.DataFrame()

    if candidates.empty:
        join_rows.append({
            **base,
            "matched_video_id": "",
            "matched_video_filename": "",
            "matched_video_path": "",
            "matched_c_code": "",
            "matched_video_type": "",
            "matched_video_start_time": "",
            "candidate_video_count": 0,
            "video_mapping_status": "UNMATCHED",
            "tracking_preparation_status": "NOT_READY_NO_MATCHED_VIDEO",
        })
    else:
        # Overlap varsa en yakın/son başlayan videoyu seç.
        candidates["delta_seconds"] = candidates["video_start_dt"].map(lambda x: abs((ann_dt - x).total_seconds()) if x else 10**9)
        candidates = candidates.sort_values(["delta_seconds", "video_start_time"])
        v = candidates.iloc[0]

        join_rows.append({
            **base,
            "matched_video_id": clean(v["video_id"]),
            "matched_video_filename": clean(v["video_filename"]),
            "matched_video_path": clean(v["video_path"]),
            "matched_c_code": clean(v["filename_c_code"]),
            "matched_video_type": clean(v["video_type"]),
            "matched_video_start_time": clean(v["video_start_time"]),
            "candidate_video_count": int(len(candidates)),
            "video_mapping_status": "MATCHED",
            "tracking_preparation_status": "READY_FOR_TRACKING_PREPARATION",
        })

join = pd.DataFrame(join_rows)
ready = join[join["video_mapping_status"] == "MATCHED"].copy()
unmatched = join[join["video_mapping_status"] != "MATCHED"].copy()

write(join, OUT_JOIN)
write(ready, OUT_READY)
write(unmatched, OUT_UNMATCHED)

if not join.empty:
    coverage = (
        join.groupby(["date", "tlc_camera", "resolved_room_pen"], dropna=False)
        .agg(
            annotation_rows=("annotation_window_id", "count"),
            matched_rows=("video_mapping_status", lambda x: int((x == "MATCHED").sum())),
            unmatched_rows=("video_mapping_status", lambda x: int((x != "MATCHED").sum())),
            matched_c_codes=("matched_c_code", lambda x: ";".join(sorted(set([clean(v) for v in x if clean(v)])))),
        )
        .reset_index()
    )
    coverage["match_ratio"] = coverage["matched_rows"] / coverage["annotation_rows"]
else:
    coverage = pd.DataFrame()

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
        "detail": f"{unmatched_count} annotation rows unmatched."
    })

issues.append({
    "item": "scope",
    "issue_type": "info_join_only",
    "severity": "info",
    "detail": "v78k9b joins annotation truth to reviewed video mapping. It does not run tracking."
})

issues_df = pd.DataFrame(issues)
write(issues_df, OUT_ISSUES)

hard_count = int((issues_df["severity"] == "hard").sum()) if "severity" in issues_df.columns else 0
warning_count = int((issues_df["severity"] == "warning").sum()) if "severity" in issues_df.columns else 0

decision = pd.DataFrame([{
    "v78k9b_decision": "annotation_video_join_created" if hard_count == 0 else "annotation_video_join_has_blocking_issues",
    "frozen_annotation_rows": len(frozen),
    "final_video_mapping_rows": len(videos),
    "annotation_join_rows": total,
    "matched_annotation_rows": matched,
    "unmatched_annotation_rows": unmatched_count,
    "match_ratio": round(match_ratio, 6),
    "best_annotation_time_column": best_time_col,
    "ready_rows_csv": str(OUT_READY),
    "unmatched_rows_csv": str(OUT_UNMATCHED),
    "hard_issue_count": hard_count,
    "warning_count": warning_count,
    "ready_for_tracking_preparation": bool(hard_count == 0 and matched > 0),
    "ready_for_full_tracking": False,
    "claim_scope": "annotation_video_join_only",
    "generated_at": datetime.now().isoformat(timespec="seconds"),
}])

write(decision, OUT_DECISION)

NOTE.write_text(
    "# v78k9b Annotation-Video Join Compact\n\n"
    f"- Decision: {decision.iloc[0]['v78k9b_decision']}\n"
    f"- Frozen annotation rows: {len(frozen)}\n"
    f"- Final video mapping rows: {len(videos)}\n"
    f"- Join rows: {total}\n"
    f"- Matched rows: {matched}\n"
    f"- Unmatched rows: {unmatched_count}\n"
    f"- Match ratio: {match_ratio:.6f}\n"
    f"- Best annotation time column: {best_time_col}\n"
    f"- Hard issues: {hard_count}\n"
    f"- Ready for tracking preparation: {bool(hard_count == 0 and matched > 0)}\n\n"
    "This joins v78k5b annotation truth with v78k8 final video mapping. It does not run tracking.\n",
    encoding="utf-8"
)

print("=== decision ===")
print(decision.to_string(index=False))

print("\n=== time report top 20 ===")
print(time_report.head(20).to_string(index=False) if not time_report.empty else "none")

print("\n=== coverage sample ===")
print(coverage.head(80).to_string(index=False) if not coverage.empty else "none")

print("\n=== issues ===")
print(issues_df.to_string(index=False))
