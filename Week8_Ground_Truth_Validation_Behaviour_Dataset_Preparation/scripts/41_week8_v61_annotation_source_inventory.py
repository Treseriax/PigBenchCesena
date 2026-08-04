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

OUT = W8 / "outputs" / "v61_annotation_source_inventory"
NOTES = W8 / "notes"
REPORTS = W8 / "reports"
PROGRESS = W8 / "progress"

for p in [OUT, NOTES, REPORTS, PROGRESS]:
    p.mkdir(parents=True, exist_ok=True)

OUT_INVENTORY = OUT / "week8_v61_annotation_source_inventory.csv"
OUT_TOP_CANDIDATES = OUT / "week8_v61_top_annotation_candidates.csv"
OUT_DECISION = OUT / "week8_v61_decision_summary.csv"
OUT_ISSUES = OUT / "week8_v61_issues.csv"
OUT_NOTE = NOTES / "week8_v61_annotation_source_inventory_notes.md"
OUT_REPORT = REPORTS / "week8_v61_annotation_source_inventory_report.md"
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


def keyword_hits(cols, keywords):
    hits = []
    for c in cols:
        cl = c.lower()
        if any(k in cl for k in keywords):
            hits.append(c)
    return hits


def score_candidate(path, sheet, cols, row_count, sample_text):
    ptxt = str(path).lower()
    cols_l = " ".join([c.lower() for c in cols])
    text = (ptxt + " " + cols_l + " " + sample_text.lower())

    score = 0

    for k in ["annotation", "annot", "behaviour", "behavior", "excel", "label", "ground", "gt"]:
        if k in ptxt:
            score += 4

    for k in ["colour", "color", "pig", "animal", "behaviour", "behavior", "activity", "time", "video"]:
        if k in cols_l:
            score += 5

    for k in ["blue", "green", "red", "pink", "purple", "cyan", "yellow", "orange"]:
        if k in text:
            score += 2

    for k in ["bbox", "x1", "y1", "x2", "y2", "xmin", "ymin", "xmax", "ymax"]:
        if k in cols_l:
            score += 3

    if "outputs" in ptxt:
        score -= 3

    if "v52" in ptxt or "v53" in ptxt:
        score -= 4

    if row_count and row_count > 10:
        score += 2

    return score


def inspect_csv(path):
    rows = []
    try:
        sep = "\t" if path.suffix.lower() == ".tsv" else None
        if sep:
            df = pd.read_csv(path, sep=sep, nrows=20)
            row_count_est = sum(1 for _ in open(path, "r", encoding="utf-8", errors="ignore")) - 1
        else:
            df = pd.read_csv(path, nrows=20)
            row_count_est = sum(1 for _ in open(path, "r", encoding="utf-8", errors="ignore")) - 1

        cols = [str(c) for c in df.columns.tolist()]
        sample_text = " ".join(df.astype(str).head(5).fillna("").values.flatten().tolist())
        rows.append(make_row(path, "", cols, row_count_est, sample_text, "csv_or_tsv"))
    except Exception as e:
        rows.append(error_row(path, "", "csv_or_tsv", str(e)))
    return rows


def inspect_excel(path):
    rows = []
    try:
        xl = pd.ExcelFile(path)
        for sheet in xl.sheet_names[:10]:
            try:
                df = pd.read_excel(path, sheet_name=sheet, nrows=20)
                cols = [str(c) for c in df.columns.tolist()]
                sample_text = " ".join(df.astype(str).head(5).fillna("").values.flatten().tolist())
                rows.append(make_row(path, sheet, cols, None, sample_text, "excel"))
            except Exception as e:
                rows.append(error_row(path, sheet, "excel", str(e)))
    except Exception as e:
        rows.append(error_row(path, "", "excel", str(e)))
    return rows


def inspect_json(path):
    rows = []
    try:
        data = json.loads(path.read_text(errors="ignore"))
        cols = []
        sample_text = ""

        if isinstance(data, list) and data:
            if isinstance(data[0], dict):
                cols = list(data[0].keys())
                sample_text = json.dumps(data[:3], ensure_ascii=False)[:3000]
                row_count = len(data)
            else:
                sample_text = json.dumps(data[:5], ensure_ascii=False)[:3000]
                row_count = len(data)
        elif isinstance(data, dict):
            cols = list(data.keys())
            sample_text = json.dumps(data, ensure_ascii=False)[:3000]
            row_count = 1
        else:
            row_count = 0

        rows.append(make_row(path, "", [str(c) for c in cols], row_count, sample_text, "json"))
    except Exception as e:
        rows.append(error_row(path, "", "json", str(e)))
    return rows


def make_row(path, sheet, cols, row_count, sample_text, file_kind):
    colour_cols = keyword_hits(cols, ["colour", "color"])
    pig_cols = keyword_hits(cols, ["pig", "animal", "identity", "id"])
    behaviour_cols = keyword_hits(cols, ["behav", "activity", "action", "label"])
    time_cols = keyword_hits(cols, ["time", "timestamp", "start", "end", "frame", "scan"])
    bbox_cols = keyword_hits(cols, ["bbox", "box", "x1", "y1", "x2", "y2", "xmin", "ymin", "xmax", "ymax"])

    score = score_candidate(path, sheet, cols, row_count or 0, sample_text)

    return {
        "path": str(path),
        "file_name": path.name,
        "file_kind": file_kind,
        "sheet": sheet,
        "size_mb": round(path.stat().st_size / (1024 * 1024), 3) if path.exists() else "",
        "row_count_estimate": row_count if row_count is not None else "",
        "column_count": len(cols),
        "columns": " | ".join(cols),
        "colour_columns": " | ".join(colour_cols),
        "pig_identity_columns": " | ".join(pig_cols),
        "behaviour_columns": " | ".join(behaviour_cols),
        "time_frame_columns": " | ".join(time_cols),
        "bbox_columns": " | ".join(bbox_cols),
        "candidate_score": score,
        "read_status": "ok",
        "error": "",
    }


def error_row(path, sheet, file_kind, error):
    return {
        "path": str(path),
        "file_name": path.name,
        "file_kind": file_kind,
        "sheet": sheet,
        "size_mb": round(path.stat().st_size / (1024 * 1024), 3) if path.exists() else "",
        "row_count_estimate": "",
        "column_count": "",
        "columns": "",
        "colour_columns": "",
        "pig_identity_columns": "",
        "behaviour_columns": "",
        "time_frame_columns": "",
        "bbox_columns": "",
        "candidate_score": -999,
        "read_status": "error",
        "error": error[:500],
    }


search_roots = [W6, W7, W8, RAW]
extensions = {".csv", ".tsv", ".xlsx", ".xls", ".json"}

files = []

for root in search_roots:
    if not root.exists():
        continue

    for path in root.rglob("*"):
        if not path.is_file():
            continue
        if path.suffix.lower() not in extensions:
            continue

        # Skip very large non-annotation files.
        size_mb = path.stat().st_size / (1024 * 1024)
        if size_mb > 100:
            continue

        files.append(path)

rows = []

for path in files:
    ext = path.suffix.lower()
    if ext in [".csv", ".tsv"]:
        rows.extend(inspect_csv(path))
    elif ext in [".xlsx", ".xls"]:
        rows.extend(inspect_excel(path))
    elif ext == ".json":
        rows.extend(inspect_json(path))

inventory = pd.DataFrame(rows)

if len(inventory):
    inventory = inventory.sort_values(["candidate_score", "path"], ascending=[False, True])
else:
    inventory = pd.DataFrame(columns=[
        "path", "file_name", "file_kind", "sheet", "size_mb", "row_count_estimate",
        "column_count", "columns", "colour_columns", "pig_identity_columns",
        "behaviour_columns", "time_frame_columns", "bbox_columns",
        "candidate_score", "read_status", "error"
    ])

safe_to_csv(inventory, OUT_INVENTORY)

top = inventory[inventory["read_status"] == "ok"].head(30).copy()
safe_to_csv(top, OUT_TOP_CANDIDATES)

issues = []
if len(top) == 0:
    issues.append({
        "item": "annotation_inventory",
        "issue_type": "hard_no_readable_candidates",
        "issue_detail": "No readable annotation candidates found.",
        "severity": "hard",
    })

strong_candidates = top[
    (top["colour_columns"].astype(str) != "")
    & (top["pig_identity_columns"].astype(str) != "")
    & (
        (top["behaviour_columns"].astype(str) != "")
        | (top["time_frame_columns"].astype(str) != "")
    )
]

if len(strong_candidates) == 0:
    issues.append({
        "item": "annotation_inventory",
        "issue_type": "warning_no_obvious_colour_identity_behaviour_candidate",
        "issue_detail": "No top candidate clearly contains colour + pig identity + behaviour/time columns.",
        "severity": "warning",
    })

issues_df = pd.DataFrame(issues, columns=["item", "issue_type", "issue_detail", "severity"])
safe_to_csv(issues_df, OUT_ISSUES)

hard_issue_count = int((issues_df["severity"] == "hard").sum()) if len(issues_df) else 0
warning_count = int((issues_df["severity"] == "warning").sum()) if len(issues_df) else 0

decision = pd.DataFrame([{
    "v61_decision": "annotation_source_inventory_completed" if hard_issue_count == 0 else "annotation_source_inventory_blocked",
    "search_root_count": len([p for p in search_roots if p.exists()]),
    "candidate_file_count": len(files),
    "inventory_row_count": int(len(inventory)),
    "top_candidate_count": int(len(top)),
    "strong_candidate_count": int(len(strong_candidates)),
    "hard_issue_count": hard_issue_count,
    "warning_count": warning_count,
    "issue_count": int(len(issues_df)),
    "ready_for_v62_canonical_colour_identity_mapping": bool(hard_issue_count == 0),
    "generated_at": datetime.now().isoformat(timespec="seconds"),
}])
safe_to_csv(decision, OUT_DECISION)

OUT_NOTE.write_text(
    "# Week 8 v61 Annotation Source Inventory\n\n"
    f"- v61 decision: {decision.iloc[0]['v61_decision']}\n"
    f"- Candidate files inspected: {len(files)}\n"
    f"- Inventory rows: {len(inventory)}\n"
    f"- Top candidates: {len(top)}\n"
    f"- Strong candidates: {len(strong_candidates)}\n"
    f"- Hard issues: {hard_issue_count}\n"
    f"- Warnings: {warning_count}\n\n"
    "Next: choose the original annotation file that contains canonical pig colour/identity labels.\n"
)

OUT_REPORT.write_text(
    "# Week 8 v61 Annotation Source Inventory Report\n\n"
    f"Decision: {decision.iloc[0]['v61_decision']}\n\n"
    f"Inventory: `{OUT_INVENTORY}`\n\n"
    f"Top candidates: `{OUT_TOP_CANDIDATES}`\n"
)

progress_row = pd.DataFrame([{
    "date": datetime.now().date().isoformat(),
    "stage": "v61",
    "task_name": "Annotation source inventory",
    "status": "PASS" if hard_issue_count == 0 else "BLOCKED",
    "input_summary": "Week6/Week7/Week8 folders and raw Unibo dataset",
    "output_summary": str(OUT),
    "hard_issues": hard_issue_count,
    "warnings": warning_count,
    "next_action": "Select canonical annotation file for v62 colour/identity mapping.",
}])

if OUT_PROGRESS.exists():
    old = pd.read_csv(OUT_PROGRESS)
    progress = pd.concat([old, progress_row], ignore_index=True)
else:
    progress = progress_row

safe_to_csv(progress, OUT_PROGRESS)

print("Saved:")
print(OUT_INVENTORY)
print(OUT_TOP_CANDIDATES)
print(OUT_DECISION)
print(OUT_ISSUES)
print(OUT_NOTE)

print()
print("=== v61 decision ===")
print(decision.to_string(index=False))

print()
print("=== v61 issues ===")
if len(issues_df):
    print(issues_df.to_string(index=False))
else:
    print("No issues found.")

print()
print("=== v61 top candidates ===")
cols = [
    "candidate_score",
    "path",
    "sheet",
    "row_count_estimate",
    "columns",
    "colour_columns",
    "pig_identity_columns",
    "behaviour_columns",
    "time_frame_columns",
    "bbox_columns",
]
print(top[cols].head(20).to_string(index=False))
