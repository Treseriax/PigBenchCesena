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
V76_EXCEL_FILES = V76_PKG / "v76_all_excel_files_inventory.csv"

V77 = FULL / "outputs" / "v77_full_excel_annotation_schema_decode"
V77_PKG = V77 / "Full_Unibo_Excel_Annotation_Schema_Decode"
V77_DECISION = V77 / "v77_decision_summary.csv"
V77_NORMALIZED = V77_PKG / "v77_normalized_annotation_candidates.csv"

OUT = FULL / "outputs" / "v77b_cleaned_annotation_window_candidates"
PKG = OUT / "Full_Unibo_Cleaned_Annotation_Window_Candidates"
NOTES = FULL / "notes"
REPORTS = FULL / "reports"
PROGRESS = FULL / "progress"

for p in [OUT, PKG, NOTES, REPORTS, PROGRESS]:
    p.mkdir(parents=True, exist_ok=True)

OUT_CLEAN = PKG / "v77b_cleaned_annotation_window_candidates.csv"
OUT_EXCLUDED = PKG / "v77b_excluded_annotation_candidates.csv"
OUT_DUP_POLICY = PKG / "v77b_duplicate_excel_policy.csv"
OUT_BEHAV = PKG / "v77b_clean_behaviour_distribution.csv"
OUT_COLOUR = PKG / "v77b_clean_colour_identity_distribution.csv"
OUT_WINDOWS = PKG / "v77b_window_time_distribution.csv"
OUT_QA = PKG / "v77b_quality_checks.csv"
OUT_README = PKG / "README_v77b_Cleaned_Annotation_Window_Candidates.md"
OUT_MANIFEST = PKG / "v77b_manifest.json"

OUT_DECISION = OUT / "v77b_decision_summary.csv"
OUT_ISSUES = OUT / "v77b_issues.csv"
OUT_ZIP = OUT / "Full_Unibo_Cleaned_Annotation_Window_Candidates.zip"
OUT_SHA = OUT / "Full_Unibo_Cleaned_Annotation_Window_Candidates.sha256"
OUT_NOTE = NOTES / "v77b_cleaned_annotation_window_candidates_notes.md"
OUT_REPORT = REPORTS / "v77b_cleaned_annotation_window_candidates_report.md"
OUT_PROGRESS = PROGRESS / "full_unibo_pipeline_progress_log.csv"

BEHAVIOURS = {"PI", "SI", "LAI", "STI", "NU", "BE", "DE", "AN", "IN", "IA", "BOX"}
SPECIFIC_IDENTITIES = {"blue", "green", "purple", "red_neck", "red_tail", "no_colour", "cyan", "pink"}


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


def to_int(x, default=None):
    try:
        if clean(x) == "":
            return default
        return int(float(x))
    except Exception:
        return default


def hhmmss_from_minute_and_sec(total_minute, extra_sec):
    if total_minute is None or extra_sec is None:
        return ""

    total_sec = int(total_minute) * 60 + int(extra_sec)
    h = total_sec // 3600
    m = (total_sec % 3600) // 60
    s = total_sec % 60
    return f"{h:02d}:{m:02d}:{s:02d}"


def is_header_or_summary(row):
    preview = clean(row.get("row_values_preview", "")).lower()
    colour = clean(row.get("detected_colour_identity", ""))
    r = to_int(row.get("row", ""), 9999)

    header_terms = [
        "fascia oraria",
        "periodo di osservazione",
        "somma",
        "camera",
        "room",
        "pen",
        "tesi",
    ]

    if colour:
        return False

    if r is not None and r <= 4:
        return True

    for term in header_terms:
        if term in preview:
            return True

    return False


issues = []

for p in [V77_DECISION, V77_NORMALIZED, V76_EXCEL_FILES]:
    if not p.exists():
        issues.append({
            "item": str(p),
            "issue_type": "hard_missing_required_input",
            "issue_detail": "v77b requires v76/v77 outputs.",
            "severity": "hard",
        })

if issues:
    issues_df = pd.DataFrame(issues)
    safe_to_csv(issues_df, OUT_ISSUES)
    decision = pd.DataFrame([{
        "v77b_decision": "cleaned_annotation_window_candidates_blocked",
        "hard_issue_count": int((issues_df["severity"] == "hard").sum()),
        "ready_for_v78_final_annotation_window_table": False,
        "generated_at": datetime.now().isoformat(timespec="seconds"),
    }])
    safe_to_csv(decision, OUT_DECISION)
    print(decision.to_string(index=False))
    raise SystemExit(1)


v77_decision = read_csv_clean(V77_DECISION)
norm = read_csv_clean(V77_NORMALIZED)
excel_files = read_csv_clean(V76_EXCEL_FILES)

if len(v77_decision) == 0 or not bool_true(v77_decision.iloc[0].get("ready_for_v78_final_annotation_window_table", "")):
    issues.append({
        "item": "v77_decision",
        "issue_type": "hard_v77_not_ready",
        "issue_detail": "v77 must be ready before v77b.",
        "severity": "hard",
    })

# Duplicate Excel policy: keep one path for each SHA.
# Prefer files under /excel/ because that is the organized annotation folder.
policy_rows = []

if "sha256" in excel_files.columns:
    for sha, g in excel_files.groupby("sha256"):
        rows = g.copy()
        rows["prefer_score"] = rows["excel_path"].map(lambda p: 1 if "/excel/" in str(p) else 0)
        rows = rows.sort_values(["prefer_score", "excel_path"], ascending=[False, True])
        canonical = rows.iloc[0]["excel_path"]

        for _, r in rows.iterrows():
            policy_rows.append({
                "excel_path": r["excel_path"],
                "excel_filename": r["excel_filename"],
                "sha256": sha,
                "canonical_excel_path_for_sha": canonical,
                "is_canonical_copy": clean(r["excel_path"]) == clean(canonical),
                "duplicate_group_size": len(rows),
            })
else:
    for p in sorted(norm["excel_path"].unique()):
        policy_rows.append({
            "excel_path": p,
            "excel_filename": Path(p).name,
            "sha256": sha256_file(Path(p)) if Path(p).exists() else "",
            "canonical_excel_path_for_sha": p,
            "is_canonical_copy": True,
            "duplicate_group_size": 1,
        })

dup_policy = pd.DataFrame(policy_rows)
safe_to_csv(dup_policy.drop(columns=["prefer_score"], errors="ignore"), OUT_DUP_POLICY)

canonical_by_path = {}
if len(dup_policy):
    for _, r in dup_policy.iterrows():
        canonical_by_path[clean(r["excel_path"])] = clean(r["canonical_excel_path_for_sha"])

clean_rows = []
excluded_rows = []

for _, row in norm.iterrows():
    d = row.to_dict()

    excel_path = clean(d.get("excel_path", ""))
    canonical_path = canonical_by_path.get(excel_path, excel_path)
    is_canonical = excel_path == canonical_path

    behaviour = clean(d.get("behaviour_code", "")).upper()
    colour = clean(d.get("detected_colour_identity", ""))

    period_min = to_int(d.get("observation_period_offset_min", ""), None)
    slot_start_min = to_int(d.get("time_slot_start_minute", ""), None)
    slot_end_min = to_int(d.get("time_slot_end_minute", ""), None)

    reason = ""

    if not is_canonical:
        reason = "duplicate_excel_copy"
    elif behaviour not in BEHAVIOURS:
        reason = "not_target_behaviour"
    elif is_header_or_summary(d):
        reason = "header_or_summary_row"
    elif not colour:
        reason = "missing_colour_identity"
    elif period_min is None:
        reason = "missing_observation_period"
    elif slot_start_min is None or slot_end_min is None:
        reason = "missing_time_slot"
    else:
        reason = "clean"

    if reason != "clean":
        d["exclusion_reason"] = reason
        d["canonical_excel_path_for_sha"] = canonical_path
        excluded_rows.append(d)
        continue

    window_start_sec_within_slot = period_min * 60
    window_end_sec_within_slot = window_start_sec_within_slot + 10
    absolute_start_minute = slot_start_min + period_min
    absolute_end_minute = absolute_start_minute
    absolute_start_hhmmss = hhmmss_from_minute_and_sec(slot_start_min, window_start_sec_within_slot)
    absolute_end_hhmmss = hhmmss_from_minute_and_sec(slot_start_min, window_end_sec_within_slot)

    identity_status = "specific_identity"
    if colour == "red":
        identity_status = "ambiguous_red_identity_needs_resolution"
    elif colour not in SPECIFIC_IDENTITIES:
        identity_status = "nonstandard_identity"

    window_id = (
        f"{clean(d.get('date',''))}__{clean(d.get('sheet_name','')).replace(' ','_')}"
        f"__{clean(d.get('time_slot_start_hhmm','')).replace(':','')}"
        f"__p{period_min:02d}__{colour}"
        f"__r{clean(d.get('row',''))}_c{clean(d.get('col',''))}"
    )

    d.update({
        "clean_annotation_window_id": window_id,
        "canonical_excel_path_for_sha": canonical_path,
        "canonical_colour_identity": colour,
        "identity_resolution_status": identity_status,
        "window_duration_sec": 10,
        "fixed_window_start_sec_within_slot": window_start_sec_within_slot,
        "fixed_window_end_sec_within_slot": window_end_sec_within_slot,
        "absolute_start_minute_of_day": absolute_start_minute,
        "absolute_end_minute_of_day": absolute_end_minute,
        "absolute_start_hhmmss": absolute_start_hhmmss,
        "absolute_end_hhmmss": absolute_end_hhmmss,
        "cleaning_status": "clean_annotation_window_candidate",
    })

    clean_rows.append(d)

clean_df = pd.DataFrame(clean_rows)
excluded_df = pd.DataFrame(excluded_rows)

safe_to_csv(clean_df, OUT_CLEAN)
safe_to_csv(excluded_df, OUT_EXCLUDED)

if len(clean_df):
    behav_dist = clean_df.groupby("behaviour_code").size().reset_index(name="clean_count").sort_values("clean_count", ascending=False)
    colour_dist = clean_df.groupby("canonical_colour_identity").size().reset_index(name="clean_count").sort_values("clean_count", ascending=False)
    window_dist = (
        clean_df.groupby(["date", "sheet_name", "time_slot_start_hhmm"])
        .size()
        .reset_index(name="clean_rows")
        .sort_values(["date", "sheet_name", "time_slot_start_hhmm"])
    )
else:
    behav_dist = pd.DataFrame(columns=["behaviour_code", "clean_count"])
    colour_dist = pd.DataFrame(columns=["canonical_colour_identity", "clean_count"])
    window_dist = pd.DataFrame(columns=["date", "sheet_name", "time_slot_start_hhmm", "clean_rows"])

safe_to_csv(behav_dist, OUT_BEHAV)
safe_to_csv(colour_dist, OUT_COLOUR)
safe_to_csv(window_dist, OUT_WINDOWS)

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

input_rows = len(norm)
duplicate_excluded = int((excluded_df["exclusion_reason"] == "duplicate_excel_copy").sum()) if len(excluded_df) else 0
header_excluded = int((excluded_df["exclusion_reason"] == "header_or_summary_row").sum()) if len(excluded_df) else 0
missing_colour_excluded = int((excluded_df["exclusion_reason"] == "missing_colour_identity").sum()) if len(excluded_df) else 0
ambiguous_red = int((clean_df["identity_resolution_status"] == "ambiguous_red_identity_needs_resolution").sum()) if len(clean_df) else 0

no_header_in_clean = True
if len(clean_df):
    no_header_in_clean = not clean_df.apply(lambda r: is_header_or_summary(r.to_dict()), axis=1).any()

all_duration_10 = bool((clean_df["window_duration_sec"] == 10).all()) if len(clean_df) else False
all_have_colour = bool(clean_df["canonical_colour_identity"].astype(str).str.len().gt(0).all()) if len(clean_df) else False
all_canonical_copy = bool(clean_df["excel_path"].eq(clean_df["canonical_excel_path_for_sha"]).all()) if len(clean_df) else False
behaviour_class_count = int(clean_df["behaviour_code"].nunique()) if len(clean_df) else 0

add_qa("v77_input_rows", ">0", input_rows, input_rows > 0, "hard", "v77 normalized candidates must exist.")
add_qa("clean_rows_created", ">0", len(clean_df), len(clean_df) > 0, "hard", "Clean annotation-window candidates must be created.")
add_qa("excluded_rows_created", ">0", len(excluded_df), len(excluded_df) > 0, "info", "Some candidates should be excluded/reviewed.")
add_qa("duplicate_excel_policy_created", ">0", len(dup_policy), len(dup_policy) > 0, "hard", "Duplicate Excel policy should exist.")
add_qa("duplicate_excel_rows_excluded", ">=0", duplicate_excluded, duplicate_excluded >= 0, "info", "Duplicate exact Excel copies are excluded from clean table.")
add_qa("header_rows_excluded", ">0", header_excluded, header_excluded > 0, "hard", "Header/summary behaviour tokens should be excluded.")
add_qa("all_clean_have_colour_identity", True, all_have_colour, all_have_colour, "hard", "Clean rows must have pig colour/identity.")
add_qa("no_header_rows_in_clean", True, no_header_in_clean, no_header_in_clean, "hard", "Clean rows should not contain header/summary rows.")
add_qa("all_clean_window_duration_10_sec", True, all_duration_10, all_duration_10, "hard", "All clean annotation windows should be 10 seconds.")
add_qa("all_clean_from_canonical_excel_copy", True, all_canonical_copy, all_canonical_copy, "hard", "Clean rows should not use duplicate Excel copies.")
add_qa("behaviour_classes_preserved", ">=10", behaviour_class_count, behaviour_class_count >= 10, "hard", "Most/all behaviour classes should remain after cleaning.")
add_qa("ambiguous_red_identity_count_reported", ">=0", ambiguous_red, ambiguous_red >= 0, "info", "Ambiguous red identities are flagged for later identity resolution.")

qa = pd.DataFrame(qa_rows)
safe_to_csv(qa, OUT_QA)

hard_quality_failures = int(((qa["severity"] == "hard") & (~qa["passed"])).sum())

if hard_quality_failures:
    issues.append({
        "item": "v77b_quality_checks",
        "issue_type": "hard_cleaning_quality_failed",
        "issue_detail": f"{hard_quality_failures} hard QA checks failed.",
        "severity": "hard",
    })

if ambiguous_red > 0:
    issues.append({
        "item": "identity_resolution",
        "issue_type": "info_ambiguous_red_identity_exists",
        "issue_detail": f"{ambiguous_red} clean rows have generic red identity and may need later red_neck/red_tail resolution if required.",
        "severity": "info",
    })

if missing_colour_excluded > 0:
    issues.append({
        "item": "missing_colour_identity",
        "issue_type": "info_missing_colour_candidates_excluded",
        "issue_detail": f"{missing_colour_excluded} candidates were excluded because no colour/identity could be attached.",
        "severity": "info",
    })

issues.append({
    "item": "scope",
    "issue_type": "info_cleaning_only",
    "issue_detail": "v77b creates cleaned annotation-window candidates. It does not yet perform final video mapping, tracking, or GT fusion.",
    "severity": "info",
})

issues_df = pd.DataFrame(issues, columns=["item", "issue_type", "issue_detail", "severity"])
safe_to_csv(issues_df, OUT_ISSUES)

hard_issue_count = int((issues_df["severity"] == "hard").sum())
warning_count = int((issues_df["severity"] == "warning").sum()) if "warning" in set(issues_df["severity"]) else 0
info_count = int((issues_df["severity"] == "info").sum())

manifest = {
    "version": "v77b_cleaned_annotation_window_candidates",
    "created_at": datetime.now().isoformat(timespec="seconds"),
    "input_v77_rows": int(input_rows),
    "clean_annotation_window_candidates": int(len(clean_df)),
    "excluded_candidates": int(len(excluded_df)),
    "duplicate_excluded": int(duplicate_excluded),
    "header_excluded": int(header_excluded),
    "missing_colour_excluded": int(missing_colour_excluded),
    "behaviour_class_count": int(behaviour_class_count),
    "window_duration_sec": 10,
    "hard_issue_count": int(hard_issue_count),
    "claim_boundary": "cleaned annotation-window candidates only; not final video-mapped GT",
}

OUT_MANIFEST.write_text(json.dumps(manifest, indent=2, ensure_ascii=False))

readme = f"""# v77b Cleaned Annotation Window Candidates

## Purpose

v77b cleans the v77 normalized annotation candidates before final video mapping.

It:
- removes duplicate Excel file copies,
- excludes header/summary rows,
- excludes candidates without colour/identity,
- fixes observation windows to 10 seconds,
- produces clean pig-level annotation-window candidates for v78.

## Main counts

- Input v77 rows: {input_rows}
- Clean rows: {len(clean_df)}
- Excluded rows: {len(excluded_df)}
- Duplicate Excel rows excluded: {duplicate_excluded}
- Header/summary rows excluded: {header_excluded}
- Missing-colour rows excluded: {missing_colour_excluded}
- Behaviour classes preserved: {behaviour_class_count}
- Ambiguous generic red rows: {ambiguous_red}
- Hard issues: {hard_issue_count}

## Boundary

This is not final moving GT yet.
v78 should resolve final video mapping and create the finalized annotation-window table.
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
    "v77b_decision": "cleaned_annotation_window_candidates_completed" if hard_issue_count == 0 else "cleaned_annotation_window_candidates_has_blocking_issues",
    "input_v77_rows": int(input_rows),
    "clean_annotation_window_candidates": int(len(clean_df)),
    "excluded_candidates": int(len(excluded_df)),
    "duplicate_excel_rows_excluded": int(duplicate_excluded),
    "header_or_summary_rows_excluded": int(header_excluded),
    "missing_colour_identity_rows_excluded": int(missing_colour_excluded),
    "ambiguous_red_identity_rows": int(ambiguous_red),
    "behaviour_class_count": int(behaviour_class_count),
    "window_duration_sec": 10,
    "zip_path": str(OUT_ZIP),
    "zip_sha256": zip_hash,
    "hard_quality_failures": hard_quality_failures,
    "hard_issue_count": hard_issue_count,
    "warning_count": warning_count,
    "info_count": info_count,
    "issue_count": int(len(issues_df)),
    "ready_for_v78_final_annotation_window_table": bool(hard_issue_count == 0),
    "claim_scope": "cleaned_annotation_window_candidates_only",
    "generated_at": datetime.now().isoformat(timespec="seconds"),
}])

safe_to_csv(decision, OUT_DECISION)

OUT_NOTE.write_text(
    "# v77b Cleaned Annotation Window Candidates\n\n"
    f"- v77b decision: {decision.iloc[0]['v77b_decision']}\n"
    f"- Input v77 rows: {input_rows}\n"
    f"- Clean annotation-window candidates: {len(clean_df)}\n"
    f"- Excluded candidates: {len(excluded_df)}\n"
    f"- Duplicate Excel rows excluded: {duplicate_excluded}\n"
    f"- Header/summary rows excluded: {header_excluded}\n"
    f"- Missing-colour rows excluded: {missing_colour_excluded}\n"
    f"- Ambiguous generic red rows: {ambiguous_red}\n"
    f"- Behaviour classes: {behaviour_class_count}\n"
    f"- Window duration fixed to: 10 seconds\n"
    f"- Hard issues: {hard_issue_count}\n"
    f"- Ready for v78 final annotation-window table: {bool(hard_issue_count == 0)}\n\n"
    "This stage cleans annotation-window candidates and fixes the 10-second window semantics. It does not create final video-mapped GT yet.\n"
)

progress_row = pd.DataFrame([{
    "date": datetime.now().date().isoformat(),
    "stage": "v77b",
    "task_name": "Cleaned annotation-window candidates",
    "status": "PASS" if hard_issue_count == 0 else "NEEDS_FIX",
    "input_summary": str(V77_NORMALIZED),
    "output_summary": str(PKG),
    "hard_issues": hard_issue_count,
    "warnings": warning_count,
    "next_action": "Create final video-mapped annotation-window table in v78." if hard_issue_count == 0 else "Fix v77b hard issues.",
}])

if OUT_PROGRESS.exists():
    old = pd.read_csv(OUT_PROGRESS)
    progress = pd.concat([old, progress_row], ignore_index=True)
else:
    progress = progress_row

safe_to_csv(progress, OUT_PROGRESS)

print("=== v77b decision ===")
print(decision.to_string(index=False))

print("\n=== clean behaviour distribution ===")
print(behav_dist.to_string(index=False))

print("\n=== clean colour distribution ===")
print(colour_dist.to_string(index=False))

print("\n=== excluded reason distribution ===")
if len(excluded_df):
    print(excluded_df.groupby("exclusion_reason").size().reset_index(name="count").to_string(index=False))
else:
    print("No excluded rows.")

print("\n=== QA ===")
print(qa.to_string(index=False))

print("\n=== issues ===")
print(issues_df.to_string(index=False) if len(issues_df) else "No issues found.")
