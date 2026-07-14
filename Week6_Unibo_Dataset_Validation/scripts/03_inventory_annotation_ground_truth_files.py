from pathlib import Path
import json
import csv
import pandas as pd


PROJECT_ROOT = Path.home() / "PigBench"
W6 = PROJECT_ROOT / "Week6_Unibo_Dataset_Validation"
OUT = W6 / "outputs/unified_ground_truth"
NOTES = W6 / "notes"

OUT.mkdir(parents=True, exist_ok=True)
NOTES.mkdir(parents=True, exist_ok=True)

SEARCH_ROOTS = [
    Path("/work/pig/datasets/Unibo"),
    Path("/work/pig/datasets"),
    PROJECT_ROOT,
]

SEARCH_ROOTS = [p for p in SEARCH_ROOTS if p.exists()]

EXTENSIONS = {".xlsx", ".xls", ".csv", ".json", ".txt", ".xml", ".yaml", ".yml"}

KEYWORDS = [
    "annotation",
    "annotations",
    "annot",
    "ground",
    "truth",
    "gt",
    "label",
    "labels",
    "behaviour",
    "behavior",
    "ethogram",
    "excel",
    "bbox",
    "box",
    "pig",
    "colour",
    "color",
    "scan",
    "window",
]

EXCLUDE_PARTS = {
    ".git",
    "__pycache__",
    "Week4_5_Final_Package",
    "final_outputs/Week4_5_Final_Package",
    "data/public_datasets/edinburgh/annotations",
    "data/public_datasets/edinburgh/samples",
}


def is_excluded(path: Path):
    s = str(path)
    return any(part in s for part in EXCLUDE_PARTS)


def keyword_score(path: Path):
    s = str(path).lower()
    hits = [k for k in KEYWORDS if k in s]
    return len(hits), ", ".join(hits)


def inspect_csv(path: Path):
    try:
        df = pd.read_csv(path, nrows=5)
        return {
            "schema_status": "readable_csv",
            "columns": ", ".join(map(str, df.columns.tolist())),
            "sample_rows": len(df),
        }
    except Exception as e:
        return {
            "schema_status": f"csv_read_error: {type(e).__name__}",
            "columns": "",
            "sample_rows": "",
        }


def inspect_excel(path: Path):
    try:
        xl = pd.ExcelFile(path)
        sheets = xl.sheet_names
        first_cols = ""
        if sheets:
            df = pd.read_excel(path, sheet_name=sheets[0], nrows=5)
            first_cols = ", ".join(map(str, df.columns.tolist()))
        return {
            "schema_status": "readable_excel",
            "columns": first_cols,
            "sheet_names": ", ".join(sheets),
            "sample_rows": 5 if sheets else 0,
        }
    except Exception as e:
        return {
            "schema_status": f"excel_read_error: {type(e).__name__}",
            "columns": "",
            "sheet_names": "",
            "sample_rows": "",
        }


def inspect_json(path: Path):
    try:
        with open(path) as f:
            data = json.load(f)

        if isinstance(data, dict):
            top = ", ".join(map(str, list(data.keys())[:30]))
            length = len(data)
            kind = "dict"
        elif isinstance(data, list):
            top = ""
            length = len(data)
            kind = "list"
            if data and isinstance(data[0], dict):
                top = ", ".join(map(str, list(data[0].keys())[:30]))
        else:
            top = ""
            length = ""
            kind = type(data).__name__

        return {
            "schema_status": f"readable_json_{kind}",
            "columns": top,
            "json_length": length,
        }
    except Exception as e:
        return {
            "schema_status": f"json_read_error: {type(e).__name__}",
            "columns": "",
            "json_length": "",
        }


def inspect_text(path: Path):
    try:
        text = path.read_text(errors="ignore")
        lines = text.splitlines()
        preview = " | ".join(line.strip() for line in lines[:3])[:500]
        return {
            "schema_status": "readable_text",
            "columns": preview,
            "line_count": len(lines),
        }
    except Exception as e:
        return {
            "schema_status": f"text_read_error: {type(e).__name__}",
            "columns": "",
            "line_count": "",
        }


rows = []

for root in SEARCH_ROOTS:
    for path in sorted(root.rglob("*")):
        if not path.is_file():
            continue
        if is_excluded(path):
            continue
        if path.suffix.lower() not in EXTENSIONS:
            continue

        try:
            stat = path.stat()
        except Exception:
            continue

        score, hits = keyword_score(path)

        row = {
            "absolute_path": str(path),
            "search_root": str(root),
            "filename": path.name,
            "suffix": path.suffix.lower(),
            "parent_folder": str(path.parent),
            "size_bytes": stat.st_size,
            "size_kb": round(stat.st_size / 1024, 2),
            "keyword_score": score,
            "keyword_hits": hits,
            "schema_status": "",
            "columns": "",
            "sheet_names": "",
            "sample_rows": "",
            "json_length": "",
            "line_count": "",
        }

        ext = path.suffix.lower()
        if ext == ".csv":
            row.update(inspect_csv(path))
        elif ext in {".xlsx", ".xls"}:
            row.update(inspect_excel(path))
        elif ext == ".json":
            row.update(inspect_json(path))
        elif ext in {".txt", ".xml", ".yaml", ".yml"}:
            row.update(inspect_text(path))

        rows.append(row)

df = pd.DataFrame(rows)

inventory_path = OUT / "annotation_ground_truth_file_inventory.csv"
df.to_csv(inventory_path, index=False)

# Candidate GT files: keyword score high or readable columns with relevant fields.
if len(df):
    relevant_column_terms = ["frame", "bbox", "box", "behaviour", "behavior", "label", "pig", "colour", "color", "track", "id", "timestamp"]
    def relevant(row):
        combined = (str(row.get("columns", "")) + " " + str(row.get("absolute_path", ""))).lower()
        return row["keyword_score"] > 0 or any(t in combined for t in relevant_column_terms)

    candidates = df[df.apply(relevant, axis=1)].copy()
    candidates = candidates.sort_values(["keyword_score", "size_bytes"], ascending=[False, False])
else:
    candidates = pd.DataFrame()

candidate_path = OUT / "annotation_ground_truth_candidate_files.csv"
candidates.to_csv(candidate_path, index=False)

suffix_summary = (
    df.groupby("suffix")
    .agg(file_count=("absolute_path", "count"), total_size_mb=("size_bytes", lambda x: round(x.sum() / (1024*1024), 3)))
    .reset_index()
    .sort_values("file_count", ascending=False)
    if len(df) else pd.DataFrame(columns=["suffix", "file_count", "total_size_mb"])
)

suffix_summary_path = OUT / "annotation_file_suffix_summary.csv"
suffix_summary.to_csv(suffix_summary_path, index=False)

root_summary = (
    df.groupby("search_root")
    .size()
    .reset_index(name="file_count")
    .sort_values("file_count", ascending=False)
    if len(df) else pd.DataFrame(columns=["search_root", "file_count"])
)

root_summary_path = OUT / "annotation_file_search_root_summary.csv"
root_summary.to_csv(root_summary_path, index=False)

note_path = NOTES / "annotation_ground_truth_file_inventory.md"

with open(note_path, "w") as f:
    f.write("# Annotation / Ground-Truth File Inventory\n\n")

    f.write("## Purpose\n\n")
    f.write(
        "This inventory searches for possible annotation, ground-truth, label, Excel, CSV, and JSON files related to the Unibo dataset. "
        "The goal is to identify which files can be used to build the Week 6 unified ground-truth table.\n\n"
    )

    f.write("## Search roots\n\n")
    for r in SEARCH_ROOTS:
        f.write(f"- `{r}`\n")
    f.write("\n")

    f.write("## Suffix summary\n\n")
    f.write(suffix_summary.to_markdown(index=False) if len(suffix_summary) else "No annotation-like files found.")
    f.write("\n\n")

    f.write("## Search root summary\n\n")
    f.write(root_summary.to_markdown(index=False) if len(root_summary) else "No annotation-like files found.")
    f.write("\n\n")

    f.write("## Candidate files\n\n")
    if len(candidates):
        cols = ["absolute_path", "suffix", "size_kb", "keyword_score", "keyword_hits", "schema_status", "columns", "sheet_names"]
        f.write(candidates[cols].head(80).to_markdown(index=False))
    else:
        f.write("No candidate annotation files identified.")
    f.write("\n\n")

    f.write("## Interpretation\n\n")
    f.write(
        "The next step is to select the actual Unibo ground-truth files from these candidates. "
        "Only files with behaviour labels, frame/timestamp information, pig colour/ID, and bounding boxes should be used for the unified ground-truth table.\n"
    )

print("Saved:")
print(inventory_path)
print(candidate_path)
print(suffix_summary_path)
print(root_summary_path)
print(note_path)

print()
print("=== Suffix summary ===")
print(suffix_summary.to_string(index=False) if len(suffix_summary) else "No annotation-like files found.")

print()
print("=== Candidate annotation files first 50 ===")
if len(candidates):
    cols = ["absolute_path", "suffix", "size_kb", "keyword_score", "keyword_hits", "schema_status", "columns", "sheet_names"]
    print(candidates[cols].head(50).to_string(index=False))
else:
    print("No candidate annotation files identified.")
