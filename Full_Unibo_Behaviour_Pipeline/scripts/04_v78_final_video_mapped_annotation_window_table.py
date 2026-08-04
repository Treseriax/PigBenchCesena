from pathlib import Path
from datetime import datetime
import csv
import json
import hashlib
import zipfile
import pandas as pd


ROOT = Path.home() / "PigBench"
FULL = ROOT / "Full_Unibo_Behaviour_Pipeline"

V76 = FULL / "outputs" / "v76_full_annotation_video_mapping_audit"
V76_PKG = V76 / "Full_Unibo_Annotation_Video_Mapping_Audit"
V76_VIDEOS = V76_PKG / "v76_all_videos_inventory.csv"

V77B = FULL / "outputs" / "v77b_cleaned_annotation_window_candidates"
V77B_PKG = V77B / "Full_Unibo_Cleaned_Annotation_Window_Candidates"
V77B_DECISION = V77B / "v77b_decision_summary.csv"
V77B_CLEAN = V77B_PKG / "v77b_cleaned_annotation_window_candidates.csv"

OUT = FULL / "outputs" / "v78_final_video_mapped_annotation_window_table"
PKG = OUT / "Full_Unibo_Final_Video_Mapped_Annotation_Window_Table"
NOTES = FULL / "notes"
REPORTS = FULL / "reports"
PROGRESS = FULL / "progress"

for p in [OUT, PKG, NOTES, REPORTS, PROGRESS]:
    p.mkdir(parents=True, exist_ok=True)

OUT_FINAL = PKG / "v78_final_video_mapped_annotation_windows.csv"
OUT_UNRESOLVED = PKG / "v78_unresolved_annotation_windows.csv"
OUT_VIDEO_SLOT_SUMMARY = PKG / "v78_video_slot_mapping_summary.csv"
OUT_MAPPING_CANDIDATES = PKG / "v78_mapping_candidates_long.csv"
OUT_BEHAV = PKG / "v78_mapped_behaviour_distribution.csv"
OUT_COLOUR = PKG / "v78_mapped_colour_identity_distribution.csv"
OUT_QA = PKG / "v78_quality_checks.csv"
OUT_README = PKG / "README_v78_Final_Video_Mapped_Annotation_Window_Table.md"
OUT_MANIFEST = PKG / "v78_manifest.json"

OUT_DECISION = OUT / "v78_decision_summary.csv"
OUT_ISSUES = OUT / "v78_issues.csv"
OUT_ZIP = OUT / "Full_Unibo_Final_Video_Mapped_Annotation_Window_Table.zip"
OUT_SHA = OUT / "Full_Unibo_Final_Video_Mapped_Annotation_Window_Table.sha256"
OUT_NOTE = NOTES / "v78_final_video_mapped_annotation_window_table_notes.md"
OUT_REPORT = REPORTS / "v78_final_video_mapped_annotation_window_table_report.md"
OUT_PROGRESS = PROGRESS / "full_unibo_pipeline_progress_log.csv"


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


def to_int(x, default=None):
    try:
        if clean(x) == "":
            return default
        return int(float(x))
    except Exception:
        return default


def sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def sec_to_hhmmss(sec):
    sec = int(sec)
    h = sec // 3600
    m = (sec % 3600) // 60
    s = sec % 60
    return f"{h:02d}:{m:02d}:{s:02d}"


def video_interval_seconds(v):
    start_min = to_int(v.get("parsed_start_minute", ""), None)
    end_min = to_int(v.get("parsed_end_minute_estimate", ""), None)

    if start_min is None or end_min is None:
        return None, None

    return start_min * 60, end_min * 60


def video_matches_time(v, absolute_start_sec, absolute_end_sec):
    vs, ve = video_interval_seconds(v)
    if vs is None or ve is None:
        return False
    return vs <= absolute_start_sec and absolute_end_sec <= ve


def score_candidate(annotation, video):
    score = 0
    reasons = []

    a_date = clean(annotation.get("date", ""))
    v_date = clean(video.get("parsed_date", ""))

    a_camera = clean(annotation.get("camera", ""))
    v_camera = clean(video.get("parsed_camera", ""))

    a_pen = clean(annotation.get("pen", ""))
    v_pen = clean(video.get("parsed_pen", ""))

    a_abs_start_min = to_int(annotation.get("absolute_start_minute_of_day", ""), None)
    a_abs_end_min = to_int(annotation.get("absolute_end_minute_of_day", ""), None)

    a_abs_start_sec = None
    a_abs_end_sec = None

    if a_abs_start_min is not None:
        extra_start = to_int(annotation.get("fixed_window_start_sec_within_slot", ""), 0)
        extra_end = to_int(annotation.get("fixed_window_end_sec_within_slot", ""), 10)

        slot_start_min = to_int(annotation.get("time_slot_start_minute", ""), None)
        if slot_start_min is not None:
            a_abs_start_sec = slot_start_min * 60 + extra_start
            a_abs_end_sec = slot_start_min * 60 + extra_end

    if a_date and v_date and a_date == v_date:
        score += 50
        reasons.append("date_match")

    if a_camera and v_camera and a_camera == v_camera:
        score += 25
        reasons.append("camera_match")

    if a_pen and v_pen and a_pen == v_pen:
        score += 25
        reasons.append("pen_match")

    if a_abs_start_sec is not None and a_abs_end_sec is not None:
        if video_matches_time(video, a_abs_start_sec, a_abs_end_sec):
            score += 35
            reasons.append("window_inside_video_time_interval")

    # Friendly TLC videos do not have date but have camera/pen/time in filename.
    if clean(video.get("video_name_type", "")) == "friendly_tlc_time_range":
        if a_camera and v_camera and a_camera == v_camera:
            score += 10
            reasons.append("friendly_camera_match")
        if a_pen and v_pen and a_pen == v_pen:
            score += 10
            reasons.append("friendly_pen_match")

    # Encoded videos have date/time but no decoded TLC/pen yet.
    if clean(video.get("video_name_type", "")) == "encoded_camera_datetime":
        if a_date and v_date and a_date == v_date:
            score += 5
            reasons.append("encoded_date_available")

    return score, ";".join(reasons), a_abs_start_sec, a_abs_end_sec


def quality_from_candidates(best_score, best_count, best_video_type, reasons):
    if best_count != 1:
        return "ambiguous"

    if best_score >= 100:
        return "high"

    if best_video_type == "friendly_tlc_time_range" and "camera_match" in reasons and "pen_match" in reasons and "window_inside_video_time_interval" in reasons:
        return "high_missing_date_in_filename"

    if best_score >= 85:
        return "medium"

    return "low"


issues = []

for p in [V77B_DECISION, V77B_CLEAN, V76_VIDEOS]:
    if not p.exists():
        issues.append({
            "item": str(p),
            "issue_type": "hard_missing_required_input",
            "issue_detail": "v78 requires v76 video inventory and v77b clean annotation windows.",
            "severity": "hard",
        })

if issues:
    issues_df = pd.DataFrame(issues)
    safe_to_csv(issues_df, OUT_ISSUES)
    decision = pd.DataFrame([{
        "v78_decision": "final_video_mapped_annotation_window_table_blocked",
        "hard_issue_count": int((issues_df["severity"] == "hard").sum()),
        "ready_for_v79_tracking_preparation": False,
        "generated_at": datetime.now().isoformat(timespec="seconds"),
    }])
    safe_to_csv(decision, OUT_DECISION)
    print(decision.to_string(index=False))
    raise SystemExit(1)


v77b_decision = read_csv_clean(V77B_DECISION)
ann = read_csv_clean(V77B_CLEAN)
videos = read_csv_clean(V76_VIDEOS)

if len(v77b_decision) == 0 or not bool_true(v77b_decision.iloc[0].get("ready_for_v78_final_annotation_window_table", "")):
    issues.append({
        "item": "v77b_decision",
        "issue_type": "hard_v77b_not_ready",
        "issue_detail": "v77b must be ready before v78.",
        "severity": "hard",
    })

candidate_rows = []
final_rows = []
unresolved_rows = []

video_records = [r.to_dict() for _, r in videos.iterrows()]

for _, a in ann.iterrows():
    a_dict = a.to_dict()

    local_candidates = []

    for v in video_records:
        score, reasons, abs_start_sec, abs_end_sec = score_candidate(a_dict, v)

        if score <= 0:
            continue

        row = {
            "clean_annotation_window_id": clean(a_dict.get("clean_annotation_window_id", "")),
            "date": clean(a_dict.get("date", "")),
            "sheet_name": clean(a_dict.get("sheet_name", "")),
            "camera": clean(a_dict.get("camera", "")),
            "pen": clean(a_dict.get("pen", "")),
            "canonical_colour_identity": clean(a_dict.get("canonical_colour_identity", "")),
            "behaviour_code": clean(a_dict.get("behaviour_code", "")),
            "absolute_start_hhmmss": clean(a_dict.get("absolute_start_hhmmss", "")),
            "absolute_end_hhmmss": clean(a_dict.get("absolute_end_hhmmss", "")),
            "candidate_video_path": clean(v.get("video_path", "")),
            "candidate_video_filename": clean(v.get("video_filename", "")),
            "candidate_video_type": clean(v.get("video_name_type", "")),
            "candidate_video_date": clean(v.get("parsed_date", "")),
            "candidate_video_camera": clean(v.get("parsed_camera", "")),
            "candidate_video_pen": clean(v.get("parsed_pen", "")),
            "candidate_video_camera_code": clean(v.get("parsed_camera_code", "")),
            "candidate_video_start_hhmm": clean(v.get("parsed_start_hhmm", "")),
            "candidate_video_end_hhmm": clean(v.get("parsed_end_hhmm_estimate", "")),
            "match_score": score,
            "match_reasons": reasons,
            "absolute_start_sec_of_day": abs_start_sec if abs_start_sec is not None else "",
            "absolute_end_sec_of_day": abs_end_sec if abs_end_sec is not None else "",
        }

        candidate_rows.append(row)
        local_candidates.append(row)

    if not local_candidates:
        u = dict(a_dict)
        u.update({
            "mapping_status": "unresolved_no_candidate",
            "mapping_issue": "No video candidate received a positive score.",
        })
        unresolved_rows.append(u)
        continue

    local_candidates = sorted(local_candidates, key=lambda x: x["match_score"], reverse=True)
    best_score = local_candidates[0]["match_score"]
    best = [c for c in local_candidates if c["match_score"] == best_score]

    best_quality = quality_from_candidates(
        best_score,
        len(best),
        best[0]["candidate_video_type"],
        best[0]["match_reasons"],
    )

    if best_quality in {"high", "high_missing_date_in_filename", "medium"}:
        chosen = best[0]
        f = dict(a_dict)
        f.update({
            "video_path": chosen["candidate_video_path"],
            "video_filename": chosen["candidate_video_filename"],
            "video_name_type": chosen["candidate_video_type"],
            "video_date": chosen["candidate_video_date"],
            "video_camera": chosen["candidate_video_camera"],
            "video_pen": chosen["candidate_video_pen"],
            "video_camera_code": chosen["candidate_video_camera_code"],
            "video_start_hhmm": chosen["candidate_video_start_hhmm"],
            "video_end_hhmm": chosen["candidate_video_end_hhmm"],
            "absolute_start_sec_of_day": chosen["absolute_start_sec_of_day"],
            "absolute_end_sec_of_day": chosen["absolute_end_sec_of_day"],
            "video_relative_start_sec": int(chosen["absolute_start_sec_of_day"]) - to_int(videos[videos["video_path"] == chosen["candidate_video_path"]].iloc[0].get("parsed_start_minute", ""), 0) * 60,
            "video_relative_end_sec": int(chosen["absolute_end_sec_of_day"]) - to_int(videos[videos["video_path"] == chosen["candidate_video_path"]].iloc[0].get("parsed_start_minute", ""), 0) * 60,
            "mapping_status": "mapped",
            "mapping_quality": best_quality,
            "mapping_score": best_score,
            "mapping_reasons": chosen["match_reasons"],
            "mapping_candidate_count_at_best_score": len(best),
        })
        final_rows.append(f)
    else:
        u = dict(a_dict)
        u.update({
            "mapping_status": "unresolved_ambiguous_or_low_confidence",
            "mapping_issue": f"best_score={best_score}; best_candidate_count={len(best)}",
            "best_candidate_video_filenames": ";".join([x["candidate_video_filename"] for x in best[:20]]),
            "best_candidate_video_paths": ";".join([x["candidate_video_path"] for x in best[:20]]),
            "best_match_score": best_score,
            "best_match_reasons": best[0]["match_reasons"],
        })
        unresolved_rows.append(u)

candidates_df = pd.DataFrame(candidate_rows)
final_df = pd.DataFrame(final_rows)
unresolved_df = pd.DataFrame(unresolved_rows)

safe_to_csv(candidates_df, OUT_MAPPING_CANDIDATES)
safe_to_csv(final_df, OUT_FINAL)
safe_to_csv(unresolved_df, OUT_UNRESOLVED)

if len(final_df):
    behav = final_df.groupby("behaviour_code").size().reset_index(name="mapped_count").sort_values("mapped_count", ascending=False)
    colour = final_df.groupby("canonical_colour_identity").size().reset_index(name="mapped_count").sort_values("mapped_count", ascending=False)
    slot_summary = (
        final_df.groupby(["date", "sheet_name", "video_filename", "mapping_quality"])
        .size()
        .reset_index(name="mapped_rows")
        .sort_values(["date", "sheet_name", "video_filename"])
    )
else:
    behav = pd.DataFrame(columns=["behaviour_code", "mapped_count"])
    colour = pd.DataFrame(columns=["canonical_colour_identity", "mapped_count"])
    slot_summary = pd.DataFrame(columns=["date", "sheet_name", "video_filename", "mapping_quality", "mapped_rows"])

safe_to_csv(behav, OUT_BEHAV)
safe_to_csv(colour, OUT_COLOUR)
safe_to_csv(slot_summary, OUT_VIDEO_SLOT_SUMMARY)

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

mapped_count = len(final_df)
unresolved_count = len(unresolved_df)
input_count = len(ann)
mapped_ratio = mapped_count / input_count if input_count else 0

all_relative_nonnegative = True
all_relative_duration_10 = True
all_have_video = True

if len(final_df):
    all_relative_nonnegative = bool((final_df["video_relative_start_sec"].astype(float) >= 0).all())
    all_relative_duration_10 = bool(((final_df["video_relative_end_sec"].astype(float) - final_df["video_relative_start_sec"].astype(float)) == 10).all())
    all_have_video = bool(final_df["video_path"].astype(str).str.len().gt(0).all())

behaviour_class_count = int(final_df["behaviour_code"].nunique()) if len(final_df) else 0
unique_video_count = int(final_df["video_path"].nunique()) if len(final_df) else 0
ambiguous_red_mapped = int((final_df["canonical_colour_identity"] == "red").sum()) if len(final_df) else 0

add_qa("v77b_input_rows", ">0", input_count, input_count > 0, "hard", "v77b clean rows must exist.")
add_qa("mapped_rows_created", ">0", mapped_count, mapped_count > 0, "hard", "Some annotation windows should be video-mapped.")
add_qa("mapping_candidates_created", ">0", len(candidates_df), len(candidates_df) > 0, "hard", "Mapping candidates should be created.")
add_qa("all_mapped_have_video_path", True, all_have_video, all_have_video, "hard", "Every mapped row must have a video path.")
add_qa("all_mapped_relative_start_nonnegative", True, all_relative_nonnegative, all_relative_nonnegative, "hard", "Video-relative start seconds must be non-negative.")
add_qa("all_mapped_duration_10_sec", True, all_relative_duration_10, all_relative_duration_10, "hard", "Mapped windows must remain 10 seconds.")
add_qa("behaviour_classes_in_mapped_rows", ">=1", behaviour_class_count, behaviour_class_count >= 1, "hard", "Mapped rows should contain behaviour labels.")
add_qa("unique_videos_mapped", ">=1", unique_video_count, unique_video_count >= 1, "hard", "At least one video should be mapped.")
add_qa("unresolved_rows_reported", ">=0", unresolved_count, unresolved_count >= 0, "info", "Unresolved rows are expected if video mapping is ambiguous.")
add_qa("mapped_ratio_reported", ">=0", round(mapped_ratio, 4), mapped_ratio >= 0, "info", "Mapped ratio is reported, not forced.")

qa = pd.DataFrame(qa_rows)
safe_to_csv(qa, OUT_QA)

hard_quality_failures = int(((qa["severity"] == "hard") & (~qa["passed"])).sum())

if hard_quality_failures:
    issues.append({
        "item": "v78_quality_checks",
        "issue_type": "hard_final_mapping_quality_failed",
        "issue_detail": f"{hard_quality_failures} hard QA checks failed.",
        "severity": "hard",
    })

if unresolved_count > 0:
    issues.append({
        "item": "unresolved_annotation_windows",
        "issue_type": "info_unresolved_or_ambiguous_mapping_exists",
        "issue_detail": f"{unresolved_count} annotation windows are not mapped automatically and require mapping review before tracking/model use.",
        "severity": "info",
    })

if ambiguous_red_mapped > 0:
    issues.append({
        "item": "generic_red_identity",
        "issue_type": "info_generic_red_identity_mapped",
        "issue_detail": f"{ambiguous_red_mapped} mapped windows contain generic red identity; later identity refinement may be needed.",
        "severity": "info",
    })

issues.append({
    "item": "scope",
    "issue_type": "info_video_mapping_only",
    "issue_detail": "v78 creates video-mapped annotation-window tables. It does not run detection, tracking, or VideoMAE.",
    "severity": "info",
})

issues_df = pd.DataFrame(issues, columns=["item", "issue_type", "issue_detail", "severity"])
safe_to_csv(issues_df, OUT_ISSUES)

hard_issue_count = int((issues_df["severity"] == "hard").sum())
warning_count = int((issues_df["severity"] == "warning").sum()) if "warning" in set(issues_df["severity"]) else 0
info_count = int((issues_df["severity"] == "info").sum())

manifest = {
    "version": "v78_final_video_mapped_annotation_window_table",
    "created_at": datetime.now().isoformat(timespec="seconds"),
    "input_clean_annotation_windows": int(input_count),
    "mapped_annotation_windows": int(mapped_count),
    "unresolved_annotation_windows": int(unresolved_count),
    "mapped_ratio": mapped_ratio,
    "unique_mapped_videos": int(unique_video_count),
    "behaviour_class_count": int(behaviour_class_count),
    "hard_issue_count": int(hard_issue_count),
    "claim_boundary": "video-mapped annotation-window table only; no tracking/model yet",
}

OUT_MANIFEST.write_text(json.dumps(manifest, indent=2, ensure_ascii=False))

readme = f"""# v78 Final Video-Mapped Annotation Window Table

## Purpose

v78 maps cleaned annotation windows to video files and video-relative timestamps.

It creates:
- final mapped annotation-window rows,
- unresolved/ambiguous rows,
- full mapping candidate table,
- mapped behaviour and colour distributions.

## Main counts

- Input clean windows: {input_count}
- Mapped windows: {mapped_count}
- Unresolved windows: {unresolved_count}
- Mapped ratio: {mapped_ratio:.4f}
- Unique mapped videos: {unique_video_count}
- Behaviour classes in mapped rows: {behaviour_class_count}
- Generic red mapped rows: {ambiguous_red_mapped}
- Hard issues: {hard_issue_count}

## Boundary

This is not moving GT yet.
Tracking starts only after the video mapping table is reviewed.
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
    "v78_decision": "final_video_mapped_annotation_window_table_completed" if hard_issue_count == 0 else "final_video_mapped_annotation_window_table_has_blocking_issues",
    "input_clean_annotation_windows": int(input_count),
    "mapped_annotation_windows": int(mapped_count),
    "unresolved_annotation_windows": int(unresolved_count),
    "mapped_ratio": round(mapped_ratio, 6),
    "unique_mapped_videos": int(unique_video_count),
    "behaviour_class_count": int(behaviour_class_count),
    "generic_red_mapped_windows": int(ambiguous_red_mapped),
    "zip_path": str(OUT_ZIP),
    "zip_sha256": zip_hash,
    "hard_quality_failures": hard_quality_failures,
    "hard_issue_count": hard_issue_count,
    "warning_count": warning_count,
    "info_count": info_count,
    "issue_count": int(len(issues_df)),
    "ready_for_v79_tracking_preparation": bool(hard_issue_count == 0 and mapped_count > 0),
    "claim_scope": "video_mapped_annotation_window_table_only",
    "generated_at": datetime.now().isoformat(timespec="seconds"),
}])

safe_to_csv(decision, OUT_DECISION)

OUT_NOTE.write_text(
    "# v78 Final Video-Mapped Annotation Window Table\n\n"
    f"- v78 decision: {decision.iloc[0]['v78_decision']}\n"
    f"- Input clean annotation windows: {input_count}\n"
    f"- Mapped annotation windows: {mapped_count}\n"
    f"- Unresolved annotation windows: {unresolved_count}\n"
    f"- Mapped ratio: {mapped_ratio:.4f}\n"
    f"- Unique mapped videos: {unique_video_count}\n"
    f"- Behaviour classes in mapped rows: {behaviour_class_count}\n"
    f"- Generic red mapped windows: {ambiguous_red_mapped}\n"
    f"- Hard issues: {hard_issue_count}\n"
    f"- Ready for v79 tracking preparation: {bool(hard_issue_count == 0 and mapped_count > 0)}\n\n"
    "This stage maps cleaned annotation windows to video files and timestamps. It does not run tracking or model training.\n"
)

progress_row = pd.DataFrame([{
    "date": datetime.now().date().isoformat(),
    "stage": "v78",
    "task_name": "Final video-mapped annotation-window table",
    "status": "PASS" if hard_issue_count == 0 else "NEEDS_FIX",
    "input_summary": str(V77B_CLEAN),
    "output_summary": str(PKG),
    "hard_issues": hard_issue_count,
    "warnings": warning_count,
    "next_action": "Review unresolved mappings and prepare tracking in v79." if hard_issue_count == 0 else "Fix v78 hard issues.",
}])

if OUT_PROGRESS.exists():
    old = pd.read_csv(OUT_PROGRESS)
    progress = pd.concat([old, progress_row], ignore_index=True)
else:
    progress = progress_row

safe_to_csv(progress, OUT_PROGRESS)

print("=== v78 decision ===")
print(decision.to_string(index=False))

print("\n=== mapped behaviour distribution ===")
print(behav.to_string(index=False))

print("\n=== mapped colour distribution ===")
print(colour.to_string(index=False))

print("\n=== mapping quality distribution ===")
if len(final_df):
    print(final_df.groupby("mapping_quality").size().reset_index(name="count").to_string(index=False))
else:
    print("No mapped rows.")

print("\n=== unresolved sample ===")
print(unresolved_df.head(20).to_string(index=False) if len(unresolved_df) else "No unresolved rows.")

print("\n=== QA ===")
print(qa.to_string(index=False))

print("\n=== issues ===")
print(issues_df.to_string(index=False) if len(issues_df) else "No issues found.")
