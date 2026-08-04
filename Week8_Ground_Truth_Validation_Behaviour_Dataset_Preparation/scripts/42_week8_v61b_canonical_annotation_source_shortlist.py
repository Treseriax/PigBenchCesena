from pathlib import Path
from datetime import datetime
import csv
import json
import pandas as pd


ROOT = Path.home() / "PigBench"
W6 = ROOT / "Week6_Unibo_Dataset_Validation"
W7 = ROOT / "Week7_Dataset_Validation_Identity_Tracking_Behaviour_Preparation"
W8 = ROOT / "Week8_Ground_Truth_Validation_Behaviour_Dataset_Preparation"
RAW = Path("/work/pig/datasets/Unibo")

OUT = W8 / "outputs" / "v61b_canonical_annotation_source_shortlist"
NOTES = W8 / "notes"
REPORTS = W8 / "reports"
PROGRESS = W8 / "progress"

for p in [OUT, NOTES, REPORTS, PROGRESS]:
    p.mkdir(parents=True, exist_ok=True)

OUT_ALL = OUT / "week8_v61b_all_nonderived_annotation_candidates.csv"
OUT_SHORTLIST = OUT / "week8_v61b_canonical_annotation_shortlist.csv"
OUT_PREVIEW = OUT / "week8_v61b_candidate_previews.csv"
OUT_DECISION = OUT / "week8_v61b_decision_summary.csv"
OUT_ISSUES = OUT / "week8_v61b_issues.csv"
OUT_NOTE = NOTES / "week8_v61b_canonical_annotation_source_shortlist_notes.md"
OUT_REPORT = REPORTS / "week8_v61b_canonical_annotation_source_shortlist_report.md"
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


def is_derived_path(path):
    p = str(path).lower()
    derived_tokens = [
        "/outputs/",
        "/reports/",
        "/notes/",
        "/progress/",
        "/validation/",
        "final_audit_package",
        "final_delivery",
        "behaviour_label_fusion",
        "primary_split",
        "propagated_ground_truth",
        "v52",
        "v53",
        "v60",
        "v61",
        "tracking",
        "hybrid",
        "crop",
        "feature",
        "baseline",
    ]
    return any(tok in p for tok in derived_tokens)


def score_original_candidate(path, sheet, cols, sample_text, row_count):
    ptxt = str(path).lower()
    name = path.name.lower()
    cols_l = " ".join([str(c).lower() for c in cols])
    sample_l = sample_text.lower()

    score = 0

    if path.suffix.lower() in [".xlsx", ".xls"]:
        score += 15

    if not is_derived_path(path):
        score += 20

    positive_path_tokens = [
        "annot", "annotation", "annotations",
        "behaviour", "behavior",
        "ethogram", "excel",
        "unibo", "labels", "label",
        "observation", "groundtruth", "ground_truth", "gt",
    ]
    for tok in positive_path_tokens:
        if tok in ptxt:
            score += 6

    positive_col_tokens = [
        "video", "file", "time", "timestamp", "start", "end", "frame",
        "pig", "animal", "id", "identity",
        "colour", "color",
        "behaviour", "behavior", "activity", "label", "code",
    ]
    for tok in positive_col_tokens:
        if tok in cols_l:
            score += 4

    colour_values = [
        "blue", "green", "red", "pink", "purple", "cyan",
        "yellow", "orange", "black", "white"
    ]
    for tok in colour_values:
        if tok in sample_l:
            score += 3

    behaviour_values = [
        "sti", "lai", "box", "an", "nu", "pi", "in", "si", "de", "be", "ia"
    ]
    for tok in behaviour_values:
        if f" {tok} " in f" {sample_l} " or f",{tok}," in sample_l:
            score += 2

    if row_count and row_count >= 50:
        score += 4

    if row_count and row_count >= 300:
        score += 2

    negative_tokens = [
        "week7_behaviour_label_fusion",
        "week8_v45",
        "week8_v52",
        "week8_v53",
        "split_datasets",
        "train.csv",
        "val.csv",
        "test.csv",
    ]
    for tok in negative_tokens:
        if tok in name or tok in ptxt:
            score -= 20

    return score


def read_candidate(path):
    rows = []
    ext = path.suffix.lower()

    try:
        if ext in [".xlsx", ".xls"]:
            xl = pd.ExcelFile(path)
            for sheet in xl.sheet_names:
                try:
                    df = pd.read_excel(path, sheet_name=sheet, nrows=30)
                    cols = [str(c) for c in df.columns.tolist()]
                    sample_text = " ".join(df.astype(str).fillna("").head(10).values.flatten().tolist())
                    score = score_original_candidate(path, sheet, cols, sample_text, len(df))
                    rows.append({
                        "path": str(path),
                        "file_name": path.name,
                        "file_kind": "excel",
                        "sheet": sheet,
                        "size_mb": round(path.stat().st_size / (1024*1024), 3),
                        "columns": " | ".join(cols),
                        "sample_preview": sample_text[:1200],
                        "candidate_score": score,
                        "read_status": "ok",
                        "error": "",
                    })
                except Exception as e:
                    rows.append(error_row(path, sheet, "excel", e))

        elif ext in [".csv", ".tsv"]:
            sep = "\t" if ext == ".tsv" else ","
            df = pd.read_csv(path, sep=sep, nrows=30)
            cols = [str(c) for c in df.columns.tolist()]
            sample_text = " ".join(df.astype(str).fillna("").head(10).values.flatten().tolist())
            row_count = sum(1 for _ in open(path, "r", encoding="utf-8", errors="ignore")) - 1
            score = score_original_candidate(path, "", cols, sample_text, row_count)
            rows.append({
                "path": str(path),
                "file_name": path.name,
                "file_kind": "csv_or_tsv",
                "sheet": "",
                "size_mb": round(path.stat().st_size / (1024*1024), 3),
                "columns": " | ".join(cols),
                "sample_preview": sample_text[:1200],
                "candidate_score": score,
                "read_status": "ok",
                "error": "",
            })

        elif ext == ".json":
            data = json.loads(path.read_text(errors="ignore"))
            if isinstance(data, list):
                preview = json.dumps(data[:5], ensure_ascii=False)
                cols = list(data[0].keys()) if data and isinstance(data[0], dict) else []
                row_count = len(data)
            elif isinstance(data, dict):
                preview = json.dumps(data, ensure_ascii=False)[:1500]
                cols = list(data.keys())
                row_count = 1
            else:
                preview = str(data)[:1500]
                cols = []
                row_count = 0

            score = score_original_candidate(path, "", cols, preview, row_count)
            rows.append({
                "path": str(path),
                "file_name": path.name,
                "file_kind": "json",
                "sheet": "",
                "size_mb": round(path.stat().st_size / (1024*1024), 3),
                "columns": " | ".join([str(c) for c in cols]),
                "sample_preview": preview[:1200],
                "candidate_score": score,
                "read_status": "ok",
                "error": "",
            })

    except Exception as e:
        rows.append(error_row(path, "", "unknown", e))

    return rows


def error_row(path, sheet, kind, e):
    return {
        "path": str(path),
        "file_name": path.name,
        "file_kind": kind,
        "sheet": sheet,
        "size_mb": round(path.stat().st_size / (1024*1024), 3) if path.exists() else "",
        "columns": "",
        "sample_preview": "",
        "candidate_score": -999,
        "read_status": "error",
        "error": str(e)[:500],
    }


search_roots = [RAW, W6, W7, W8]
extensions = {".xlsx", ".xls", ".csv", ".tsv", ".json"}

candidate_files = []

for root in search_roots:
    if not root.exists():
        continue

    for path in root.rglob("*"):
        if not path.is_file():
            continue
        if path.suffix.lower() not in extensions:
            continue

        size_mb = path.stat().st_size / (1024 * 1024)
        if size_mb > 50:
            continue

        # Hard exclude generated outputs for canonical source search.
        if is_derived_path(path):
            continue

        candidate_files.append(path)

rows = []
for path in candidate_files:
    rows.extend(read_candidate(path))

all_df = pd.DataFrame(rows)

if len(all_df):
    all_df = all_df.sort_values(["candidate_score", "path", "sheet"], ascending=[False, True, True])
else:
    all_df = pd.DataFrame(columns=[
        "path", "file_name", "file_kind", "sheet", "size_mb",
        "columns", "sample_preview", "candidate_score", "read_status", "error"
    ])

safe_to_csv(all_df, OUT_ALL)

shortlist = all_df[
    (all_df["read_status"] == "ok")
    & (all_df["candidate_score"] >= 20)
].head(30).copy()

if len(shortlist) == 0:
    shortlist = all_df[all_df["read_status"] == "ok"].head(30).copy()

safe_to_csv(shortlist, OUT_SHORTLIST)

preview = shortlist[[
    "candidate_score",
    "path",
    "sheet",
    "file_kind",
    "columns",
    "sample_preview",
]].copy()

safe_to_csv(preview, OUT_PREVIEW)

issues = []

if len(shortlist) == 0:
    issues.append({
        "item": "canonical_annotation_shortlist",
        "issue_type": "hard_no_nonderived_annotation_candidates",
        "issue_detail": "No non-derived annotation candidates found after excluding outputs.",
        "severity": "hard",
    })

issues_df = pd.DataFrame(issues, columns=["item", "issue_type", "issue_detail", "severity"])
safe_to_csv(issues_df, OUT_ISSUES)

hard_issue_count = int((issues_df["severity"] == "hard").sum()) if len(issues_df) else 0

decision = pd.DataFrame([{
    "v61b_decision": "canonical_annotation_source_shortlist_completed" if hard_issue_count == 0 else "canonical_annotation_source_shortlist_blocked",
    "nonderived_candidate_file_count": len(candidate_files),
    "candidate_row_count": len(all_df),
    "shortlist_row_count": len(shortlist),
    "hard_issue_count": hard_issue_count,
    "warning_count": 0,
    "issue_count": len(issues_df),
    "ready_for_manual_canonical_source_selection": bool(hard_issue_count == 0),
    "generated_at": datetime.now().isoformat(timespec="seconds"),
}])

safe_to_csv(decision, OUT_DECISION)

OUT_NOTE.write_text(
    "# Week 8 v61b Canonical Annotation Source Shortlist\n\n"
    f"- v61b decision: {decision.iloc[0]['v61b_decision']}\n"
    f"- Non-derived candidate files: {len(candidate_files)}\n"
    f"- Candidate rows: {len(all_df)}\n"
    f"- Shortlist rows: {len(shortlist)}\n"
    f"- Ready for manual canonical source selection: {hard_issue_count == 0}\n\n"
    "This step excludes generated Week7/Week8 outputs and searches for original annotation files.\n"
)

OUT_REPORT.write_text(
    "# Week 8 v61b Canonical Annotation Source Shortlist Report\n\n"
    f"Decision: {decision.iloc[0]['v61b_decision']}\n\n"
    f"Shortlist: `{OUT_SHORTLIST}`\n\n"
    f"Previews: `{OUT_PREVIEW}`\n"
)

progress_row = pd.DataFrame([{
    "date": datetime.now().date().isoformat(),
    "stage": "v61b",
    "task_name": "Canonical annotation source shortlist",
    "status": "PASS" if hard_issue_count == 0 else "BLOCKED",
    "input_summary": "Non-derived annotation candidates from RAW/W6/W7/W8",
    "output_summary": str(OUT),
    "hard_issues": hard_issue_count,
    "warnings": 0,
    "next_action": "Inspect shortlist and select canonical annotation file.",
}])

if OUT_PROGRESS.exists():
    old = pd.read_csv(OUT_PROGRESS)
    progress = pd.concat([old, progress_row], ignore_index=True)
else:
    progress = progress_row

safe_to_csv(progress, OUT_PROGRESS)

print("Saved:")
print(OUT_ALL)
print(OUT_SHORTLIST)
print(OUT_PREVIEW)
print(OUT_DECISION)
print(OUT_ISSUES)
print(OUT_NOTE)

print()
print("=== v61b decision ===")
print(decision.to_string(index=False))

print()
print("=== v61b shortlist ===")
if len(shortlist):
    print(shortlist[[
        "candidate_score",
        "path",
        "sheet",
        "file_kind",
        "columns",
        "sample_preview",
    ]].head(20).to_string(index=False))
else:
    print("No shortlist candidates.")

print()
print("=== v61b issues ===")
if len(issues_df):
    print(issues_df.to_string(index=False))
else:
    print("No issues found.")
