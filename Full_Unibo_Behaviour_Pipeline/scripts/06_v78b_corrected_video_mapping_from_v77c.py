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

V77C = FULL / "outputs" / "v77c_layout_aware_excel_decoder"
V77C_PKG = V77C / "Full_Unibo_Layout_Aware_Excel_Decoder"
V77C_DECISION = V77C / "v77c_decision_summary.csv"
V77C_ROWS = V77C_PKG / "v77c_layout_aware_annotation_windows.csv"

CONFIG = FULL / "config"
MANUAL_CAMERA_MAP = CONFIG / "camera_code_mapping.csv"

OUT = FULL / "outputs" / "v78b_corrected_video_mapping_from_v77c"
PKG = OUT / "Full_Unibo_Corrected_Video_Mapping_From_v77c"
NOTES = FULL / "notes"
REPORTS = FULL / "reports"
PROGRESS = FULL / "progress"

for p in [OUT, PKG, NOTES, REPORTS, PROGRESS, CONFIG]:
    p.mkdir(parents=True, exist_ok=True)

OUT_MAPPED = PKG / "v78b_corrected_video_mapped_annotation_windows.csv"
OUT_UNRESOLVED = PKG / "v78b_unresolved_annotation_windows.csv"
OUT_CANDIDATES = PKG / "v78b_mapping_candidates_long.csv"
OUT_CAMERA_TEMPLATE = PKG / "v78b_camera_code_mapping_template.csv"
OUT_VIDEO_COVERAGE = PKG / "v78b_video_coverage_summary.csv"
OUT_BEHAV = PKG / "v78b_mapped_behaviour_distribution.csv"
OUT_COLOUR = PKG / "v78b_mapped_colour_identity_distribution.csv"
OUT_QA = PKG / "v78b_quality_checks.csv"
OUT_README = PKG / "README_v78b_Corrected_Video_Mapping.md"
OUT_MANIFEST = PKG / "v78b_manifest.json"

OUT_DECISION = OUT / "v78b_decision_summary.csv"
OUT_ISSUES = OUT / "v78b_issues.csv"
OUT_ZIP = OUT / "Full_Unibo_Corrected_Video_Mapping_From_v77c.zip"
OUT_SHA = OUT / "Full_Unibo_Corrected_Video_Mapping_From_v77c.sha256"
OUT_NOTE = NOTES / "v78b_corrected_video_mapping_from_v77c_notes.md"
OUT_REPORT = REPORTS / "v78b_corrected_video_mapping_from_v77c_report.md"
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


def video_start_end_sec(video):
    start_min = to_int(video.get("parsed_start_minute", ""), None)
    end_min = to_int(video.get("parsed_end_minute_estimate", ""), None)

    if start_min is None or end_min is None:
        return None, None

    return start_min * 60, end_min * 60


def window_inside_video(annotation, video):
    a_start = to_int(annotation.get("absolute_start_sec_of_day", ""), None)
    a_end = to_int(annotation.get("absolute_end_sec_of_day", ""), None)
    v_start, v_end = video_start_end_sec(video)

    if a_start is None or a_end is None or v_start is None or v_end is None:
        return False

    return v_start <= a_start and a_end <= v_end


def manual_map_lookup(manual_df, annotation, video):
    if manual_df is None or len(manual_df) == 0:
        return False, ""

    needed_cols = {"date", "camera", "pen", "video_camera_code"}
    if not needed_cols.issubset(set(manual_df.columns)):
        return False, "manual_map_missing_required_columns"

    a_date = clean(annotation.get("date", ""))
    a_camera = clean(annotation.get("camera", ""))
    a_pen = clean(annotation.get("pen", ""))
    v_code = clean(video.get("parsed_camera_code", ""))

    matches = manual_df[
        (manual_df["date"].astype(str).map(clean) == a_date)
        & (manual_df["camera"].astype(str).map(clean) == a_camera)
        & (manual_df["pen"].astype(str).map(clean) == a_pen)
        & (manual_df["video_camera_code"].astype(str).map(clean) == v_code)
    ]

    if len(matches) > 0:
        return True, "manual_camera_code_mapping_match"

    return False, ""


def score_candidate(annotation, video, manual_df):
    score = 0
    reasons = []

    a_date = clean(annotation.get("date", ""))
    a_camera = clean(annotation.get("camera", ""))
    a_pen = clean(annotation.get("pen", ""))

    v_date = clean(video.get("parsed_date", ""))
    v_camera = clean(video.get("parsed_camera", ""))
    v_pen = clean(video.get("parsed_pen", ""))
    v_type = clean(video.get("video_name_type", ""))
    v_code = clean(video.get("parsed_camera_code", ""))

    if not window_inside_video(annotation, video):
        return 0, "window_not_inside_video_interval", "not_candidate"

    # Friendly video names contain TLC/pen/time. These are trusted even if no date is in filename.
    if v_type == "friendly_tlc_time_range":
        if a_camera and v_camera and a_camera == v_camera:
            score += 40
            reasons.append("friendly_camera_match")
        if a_pen and v_pen and a_pen == v_pen:
            score += 40
            reasons.append("friendly_pen_match")
        if score >= 80:
            score += 30
            reasons.append("window_inside_friendly_video")
            return score, ";".join(reasons), "trusted_friendly_tlc_mapping"
        return 0, "friendly_video_camera_or_pen_mismatch", "not_candidate"

    # Encoded video names contain date/time/camera-code but not TLC/pen.
    if v_type == "encoded_camera_datetime":
        if a_date and v_date and a_date == v_date:
            score += 40
            reasons.append("encoded_date_match")
        else:
            return 0, "encoded_date_mismatch_or_missing", "not_candidate"

        score += 25
        reasons.append("window_inside_encoded_video")

        manual_ok, manual_reason = manual_map_lookup(manual_df, annotation, video)
        if manual_ok:
            score += 60
            reasons.append(manual_reason)
            return score, ";".join(reasons), "trusted_manual_camera_code_mapping"

        if v_code:
            reasons.append("requires_camera_code_mapping")
            return score, ";".join(reasons), "requires_camera_code_mapping"

    return 0, "unsupported_video_type", "not_candidate"


def choose_candidate(candidates):
    if not candidates:
        return None, "unresolved_no_candidate"

    trusted = [c for c in candidates if c["candidate_mapping_type"].startswith("trusted")]

    if trusted:
        trusted = sorted(trusted, key=lambda x: x["match_score"], reverse=True)
        best_score = trusted[0]["match_score"]
        best = [c for c in trusted if c["match_score"] == best_score]

        if len(best) == 1:
            return best[0], "mapped"
        return None, "unresolved_multiple_trusted_candidates"

    # Do not force encoded camera-code candidates.
    return None, "unresolved_requires_camera_code_mapping"


issues = []

for p in [V76_VIDEOS, V77C_DECISION, V77C_ROWS]:
    if not p.exists():
        issues.append({
            "item": str(p),
            "issue_type": "hard_missing_required_input",
            "issue_detail": "v78b requires video inventory and v77c layout-aware annotations.",
            "severity": "hard",
        })

if issues:
    issues_df = pd.DataFrame(issues)
    safe_to_csv(issues_df, OUT_ISSUES)
    raise SystemExit("Missing required inputs.")

videos = read_csv_clean(V76_VIDEOS)
v77c_decision = read_csv_clean(V77C_DECISION)
annotations = read_csv_clean(V77C_ROWS)

if len(v77c_decision) == 0 or not bool_true(v77c_decision.iloc[0].get("ready_for_corrected_video_mapping", "")):
    issues.append({
        "item": "v77c_decision",
        "issue_type": "hard_v77c_not_ready",
        "issue_detail": "v77c must be ready before corrected mapping.",
        "severity": "hard",
    })

manual_df = None
manual_map_loaded = False

if MANUAL_CAMERA_MAP.exists():
    manual_df = read_csv_clean(MANUAL_CAMERA_MAP)
    manual_map_loaded = True

video_records = [r.to_dict() for _, r in videos.iterrows()]

candidate_rows = []
mapped_rows = []
unresolved_rows = []

for _, a in annotations.iterrows():
    a_dict = a.to_dict()
    local = []

    for v in video_records:
        score, reasons, mapping_type = score_candidate(a_dict, v, manual_df)

        if score <= 0:
            continue

        v_start, v_end = video_start_end_sec(v)
        a_start = to_int(a_dict.get("absolute_start_sec_of_day", ""), 0)
        a_end = to_int(a_dict.get("absolute_end_sec_of_day", ""), 0)

        c = {
            "annotation_window_id": clean(a_dict.get("annotation_window_id", "")),
            "date": clean(a_dict.get("date", "")),
            "sheet_name": clean(a_dict.get("sheet_name", "")),
            "camera": clean(a_dict.get("camera", "")),
            "pen": clean(a_dict.get("pen", "")),
            "canonical_colour_identity": clean(a_dict.get("canonical_colour_identity", "")),
            "behaviour_code": clean(a_dict.get("behaviour_code", "")),
            "absolute_start_hhmmss": clean(a_dict.get("absolute_start_hhmmss", "")),
            "absolute_end_hhmmss": clean(a_dict.get("absolute_end_hhmmss", "")),
            "absolute_start_sec_of_day": a_start,
            "absolute_end_sec_of_day": a_end,
            "candidate_video_path": clean(v.get("video_path", "")),
            "candidate_video_filename": clean(v.get("video_filename", "")),
            "candidate_video_type": clean(v.get("video_name_type", "")),
            "candidate_video_date": clean(v.get("parsed_date", "")),
            "candidate_video_camera": clean(v.get("parsed_camera", "")),
            "candidate_video_pen": clean(v.get("parsed_pen", "")),
            "candidate_video_camera_code": clean(v.get("parsed_camera_code", "")),
            "candidate_video_start_hhmm": clean(v.get("parsed_start_hhmm", "")),
            "candidate_video_end_hhmm": clean(v.get("parsed_end_hhmm_estimate", "")),
            "candidate_video_start_sec_of_day": v_start if v_start is not None else "",
            "candidate_video_end_sec_of_day": v_end if v_end is not None else "",
            "candidate_mapping_type": mapping_type,
            "match_score": score,
            "match_reasons": reasons,
        }

        candidate_rows.append(c)
        local.append(c)

    chosen, status = choose_candidate(local)

    if chosen is None:
        u = dict(a_dict)
        u.update({
            "mapping_status": status,
            "candidate_count": len(local),
            "trusted_candidate_count": sum(1 for x in local if x["candidate_mapping_type"].startswith("trusted")),
            "requires_camera_code_mapping_candidate_count": sum(1 for x in local if x["candidate_mapping_type"] == "requires_camera_code_mapping"),
            "candidate_video_codes": ";".join(sorted(set([x["candidate_video_camera_code"] for x in local if x["candidate_video_camera_code"]]))),
            "candidate_video_filenames": ";".join([x["candidate_video_filename"] for x in local[:30]]),
        })
        unresolved_rows.append(u)
        continue

    video_start_sec = to_int(chosen.get("candidate_video_start_sec_of_day", ""), 0)

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
        "video_relative_start_sec": int(chosen["absolute_start_sec_of_day"]) - video_start_sec,
        "video_relative_end_sec": int(chosen["absolute_end_sec_of_day"]) - video_start_sec,
        "mapping_status": "mapped",
        "mapping_quality": chosen["candidate_mapping_type"],
        "mapping_score": chosen["match_score"],
        "mapping_reasons": chosen["match_reasons"],
    })

    mapped_rows.append(f)

candidates_df = pd.DataFrame(candidate_rows)
mapped_df = pd.DataFrame(mapped_rows)
unresolved_df = pd.DataFrame(unresolved_rows)

safe_to_csv(candidates_df, OUT_CANDIDATES)
safe_to_csv(mapped_df, OUT_MAPPED)
safe_to_csv(unresolved_df, OUT_UNRESOLVED)

# Camera-code mapping template for unresolved encoded cases.
template_rows = []

if len(unresolved_df):
    encoded_candidates = candidates_df[candidates_df["candidate_mapping_type"] == "requires_camera_code_mapping"].copy() if len(candidates_df) else pd.DataFrame()

    if len(encoded_candidates):
        group_cols = ["date", "camera", "pen"]
        for key, g in encoded_candidates.groupby(group_cols):
            date, camera, pen = key
            template_rows.append({
                "date": date,
                "camera": camera,
                "pen": pen,
                "video_camera_code": "",
                "candidate_video_camera_codes": ";".join(sorted(set(g["candidate_video_camera_code"].astype(str)))),
                "candidate_video_count": int(g["candidate_video_path"].nunique()),
                "candidate_video_examples": ";".join(sorted(set(g["candidate_video_filename"].astype(str)))[:20]),
                "manual_resolution_status": "fill_video_camera_code_then_save_as_Full_Unibo_Behaviour_Pipeline/config/camera_code_mapping.csv",
            })

camera_template = pd.DataFrame(template_rows)

if len(camera_template) == 0:
    camera_template = pd.DataFrame(columns=[
        "date", "camera", "pen", "video_camera_code",
        "candidate_video_camera_codes", "candidate_video_count",
        "candidate_video_examples", "manual_resolution_status"
    ])

safe_to_csv(camera_template, OUT_CAMERA_TEMPLATE)

if len(mapped_df):
    behav = mapped_df.groupby("behaviour_code").size().reset_index(name="mapped_count").sort_values("mapped_count", ascending=False)
    colour = mapped_df.groupby("canonical_colour_identity").size().reset_index(name="mapped_count").sort_values("mapped_count", ascending=False)
    coverage = mapped_df.groupby(["video_filename", "mapping_quality"]).size().reset_index(name="mapped_rows").sort_values(["video_filename"])
else:
    behav = pd.DataFrame(columns=["behaviour_code", "mapped_count"])
    colour = pd.DataFrame(columns=["canonical_colour_identity", "mapped_count"])
    coverage = pd.DataFrame(columns=["video_filename", "mapping_quality", "mapped_rows"])

safe_to_csv(behav, OUT_BEHAV)
safe_to_csv(colour, OUT_COLOUR)
safe_to_csv(coverage, OUT_VIDEO_COVERAGE)

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

input_count = len(annotations)
mapped_count = len(mapped_df)
unresolved_count = len(unresolved_df)
candidate_count = len(candidates_df)
mapped_ratio = mapped_count / input_count if input_count else 0
unique_mapped_videos = int(mapped_df["video_path"].nunique()) if len(mapped_df) else 0
behaviour_class_count = int(mapped_df["behaviour_code"].nunique()) if len(mapped_df) else 0
colour_count = int(mapped_df["canonical_colour_identity"].nunique()) if len(mapped_df) else 0
template_count = len(camera_template)

relative_ok = True
duration_ok = True

if len(mapped_df):
    relative_ok = bool((mapped_df["video_relative_start_sec"].astype(float) >= 0).all())
    duration_ok = bool(((mapped_df["video_relative_end_sec"].astype(float) - mapped_df["video_relative_start_sec"].astype(float)) == 10).all())

add_qa("v77c_input_rows", ">0", input_count, input_count > 0, "hard", "v77c annotation windows must exist.")
add_qa("mapping_candidates_created", ">0", candidate_count, candidate_count > 0, "hard", "Candidate mappings should be created.")
add_qa("trusted_mapped_rows_created", ">0", mapped_count, mapped_count > 0, "hard", "At least trusted friendly or manual mappings should be created.")
add_qa("mapped_windows_keep_10_sec_duration", True, duration_ok, duration_ok, "hard", "Mapped windows must remain 10 seconds.")
add_qa("mapped_video_relative_start_nonnegative", True, relative_ok, relative_ok, "hard", "Mapped video-relative starts must be non-negative.")
add_qa("unresolved_rows_reported", ">=0", unresolved_count, unresolved_count >= 0, "info", "Unresolved rows are expected until camera-code mapping is resolved.")
add_qa("camera_code_mapping_template_created", ">=0", template_count, template_count >= 0, "info", "Template supports manual camera-code mapping.")
add_qa("manual_camera_map_loaded", "True/False", manual_map_loaded, True, "info", "Reports whether manual camera-code mapping was used.")

qa = pd.DataFrame(qa_rows)
safe_to_csv(qa, OUT_QA)

hard_quality_failures = int(((qa["severity"] == "hard") & (~qa["passed"])).sum())

if hard_quality_failures:
    issues.append({
        "item": "v78b_quality_checks",
        "issue_type": "hard_corrected_video_mapping_failed",
        "issue_detail": f"{hard_quality_failures} hard QA checks failed.",
        "severity": "hard",
    })

if unresolved_count > 0:
    issues.append({
        "item": "unresolved_mapping",
        "issue_type": "info_unresolved_windows_require_camera_code_mapping",
        "issue_detail": f"{unresolved_count} windows remain unresolved; fill camera_code_mapping.csv before full tracking.",
        "severity": "info",
    })

issues.append({
    "item": "scope",
    "issue_type": "info_corrected_mapping_only",
    "issue_detail": "v78b maps v77c annotation windows to videos where trusted mapping exists. It does not run tracking or model training.",
    "severity": "info",
})

issues_df = pd.DataFrame(issues, columns=["item", "issue_type", "issue_detail", "severity"])
safe_to_csv(issues_df, OUT_ISSUES)

hard_issue_count = int((issues_df["severity"] == "hard").sum())
warning_count = int((issues_df["severity"] == "warning").sum()) if "warning" in set(issues_df["severity"]) else 0
info_count = int((issues_df["severity"] == "info").sum())

ready_for_full_tracking = bool(hard_issue_count == 0 and unresolved_count == 0 and mapped_count > 0)
ready_for_mapping_resolution = bool(hard_issue_count == 0 and unresolved_count > 0)

manifest = {
    "version": "v78b_corrected_video_mapping_from_v77c",
    "created_at": datetime.now().isoformat(timespec="seconds"),
    "input_annotation_windows": int(input_count),
    "mapped_annotation_windows": int(mapped_count),
    "unresolved_annotation_windows": int(unresolved_count),
    "mapped_ratio": float(mapped_ratio),
    "unique_mapped_videos": int(unique_mapped_videos),
    "camera_code_mapping_template_rows": int(template_count),
    "manual_camera_map_loaded": bool(manual_map_loaded),
    "hard_issue_count": int(hard_issue_count),
    "ready_for_full_tracking": ready_for_full_tracking,
    "claim_boundary": "corrected video mapping only; no tracking/model",
}

OUT_MANIFEST.write_text(json.dumps(manifest, indent=2, ensure_ascii=False))

readme = f"""# v78b Corrected Video Mapping from v77c

## Purpose

v78b maps v77c layout-aware annotation windows to video files.

Trusted mappings:
- friendly TLC videos with camera + pen + time match,
- encoded videos only when a manual camera-code mapping file exists.

Unresolved encoded mappings are not forced.

## Main counts

- Input annotation windows: {input_count}
- Mapped annotation windows: {mapped_count}
- Unresolved annotation windows: {unresolved_count}
- Mapped ratio: {mapped_ratio:.6f}
- Unique mapped videos: {unique_mapped_videos}
- Behaviour classes in mapped rows: {behaviour_class_count}
- Colour identities in mapped rows: {colour_count}
- Camera-code mapping template rows: {template_count}
- Manual camera map loaded: {manual_map_loaded}
- Hard issues: {hard_issue_count}

## Next step

If unresolved rows remain, fill:
`Full_Unibo_Behaviour_Pipeline/config/camera_code_mapping.csv`

using the template:
`v78b_camera_code_mapping_template.csv`

Then rerun v78b.
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
    "v78b_decision": "corrected_video_mapping_completed" if hard_issue_count == 0 else "corrected_video_mapping_has_blocking_issues",
    "input_annotation_windows": int(input_count),
    "mapped_annotation_windows": int(mapped_count),
    "unresolved_annotation_windows": int(unresolved_count),
    "mapped_ratio": round(mapped_ratio, 6),
    "unique_mapped_videos": int(unique_mapped_videos),
    "behaviour_class_count_in_mapped": int(behaviour_class_count),
    "colour_identity_count_in_mapped": int(colour_count),
    "camera_code_mapping_template_rows": int(template_count),
    "manual_camera_map_loaded": bool(manual_map_loaded),
    "zip_path": str(OUT_ZIP),
    "zip_sha256": zip_hash,
    "hard_quality_failures": int(hard_quality_failures),
    "hard_issue_count": int(hard_issue_count),
    "warning_count": int(warning_count),
    "info_count": int(info_count),
    "issue_count": int(len(issues_df)),
    "ready_for_mapping_resolution": ready_for_mapping_resolution,
    "ready_for_v79_full_tracking_preparation": ready_for_full_tracking,
    "claim_scope": "corrected_video_mapping_only",
    "generated_at": datetime.now().isoformat(timespec="seconds"),
}])

safe_to_csv(decision, OUT_DECISION)

OUT_NOTE.write_text(
    "# v78b Corrected Video Mapping from v77c\n\n"
    f"- v78b decision: {decision.iloc[0]['v78b_decision']}\n"
    f"- Input annotation windows: {input_count}\n"
    f"- Mapped annotation windows: {mapped_count}\n"
    f"- Unresolved annotation windows: {unresolved_count}\n"
    f"- Mapped ratio: {mapped_ratio:.6f}\n"
    f"- Unique mapped videos: {unique_mapped_videos}\n"
    f"- Behaviour classes in mapped rows: {behaviour_class_count}\n"
    f"- Colour identities in mapped rows: {colour_count}\n"
    f"- Camera-code mapping template rows: {template_count}\n"
    f"- Manual camera map loaded: {manual_map_loaded}\n"
    f"- Hard issues: {hard_issue_count}\n"
    f"- Ready for mapping resolution: {ready_for_mapping_resolution}\n"
    f"- Ready for v79 full tracking preparation: {ready_for_full_tracking}\n\n"
    "Do not proceed to full tracking until unresolved mapping is resolved or explicitly scoped.\n"
)

progress_row = pd.DataFrame([{
    "date": datetime.now().date().isoformat(),
    "stage": "v78b",
    "task_name": "Corrected video mapping from v77c",
    "status": "PASS_NEEDS_MAPPING_RESOLUTION" if hard_issue_count == 0 and unresolved_count > 0 else ("PASS" if hard_issue_count == 0 else "NEEDS_FIX"),
    "input_summary": str(V77C_ROWS),
    "output_summary": str(PKG),
    "hard_issues": hard_issue_count,
    "warnings": warning_count,
    "next_action": "Resolve camera-code mapping and rerun v78b." if unresolved_count > 0 else "Proceed to v79 full tracking preparation.",
}])

if OUT_PROGRESS.exists():
    old = pd.read_csv(OUT_PROGRESS)
    progress = pd.concat([old, progress_row], ignore_index=True)
else:
    progress = progress_row

safe_to_csv(progress, OUT_PROGRESS)

print("=== v78b decision ===")
print(decision.to_string(index=False))

print("\n=== mapped behaviour distribution ===")
print(behav.to_string(index=False))

print("\n=== mapped colour distribution ===")
print(colour.to_string(index=False))

print("\n=== video coverage ===")
print(coverage.head(50).to_string(index=False))

print("\n=== camera-code mapping template sample ===")
print(camera_template.head(50).to_string(index=False))

print("\n=== QA ===")
print(qa.to_string(index=False))

print("\n=== issues ===")
print(issues_df.to_string(index=False))
