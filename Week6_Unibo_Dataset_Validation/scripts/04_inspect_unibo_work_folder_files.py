from pathlib import Path
import json
import csv
import pandas as pd


WORK_UNIBO = Path("/work/pig/datasets/Unibo")
PROJECT_ROOT = Path.home() / "PigBench"
W6 = PROJECT_ROOT / "Week6_Unibo_Dataset_Validation"
OUT_STATS = W6 / "outputs/dataset_statistics"
OUT_GT = W6 / "outputs/unified_ground_truth"
NOTES = W6 / "notes"

OUT_STATS.mkdir(parents=True, exist_ok=True)
OUT_GT.mkdir(parents=True, exist_ok=True)
NOTES.mkdir(parents=True, exist_ok=True)

if not WORK_UNIBO.exists():
    raise FileNotFoundError(WORK_UNIBO)


def clean_text(x, limit=1800):
    if x is None:
        return ""
    s = str(x)
    s = s.replace("\r", " ").replace("\n", " ").replace("\t", " ")
    s = " ".join(s.split())
    return s[:limit]


def safe_to_csv(df, path):
    df.to_csv(
        path,
        index=False,
        quoting=csv.QUOTE_ALL,
        escapechar="\\",
        lineterminator="\n"
    )


rows = []

for p in sorted(WORK_UNIBO.iterdir()):
    if not p.is_file():
        continue

    stat = p.stat()

    row = {
        "absolute_path": str(p),
        "filename": p.name,
        "suffix": p.suffix.lower(),
        "size_bytes": stat.st_size,
        "size_kb": round(stat.st_size / 1024, 2),
        "size_mb": round(stat.st_size / (1024 * 1024), 3),
        "file_role_guess": "",
        "read_status": "",
        "sheets_or_top_keys": "",
        "columns_or_preview": "",
        "row_count_or_length": "",
    }

    suffix = p.suffix.lower()

    if suffix == ".mp4":
        row["file_role_guess"] = "raw_video"
        row["read_status"] = "not_inspected_here"

    elif suffix in [".xlsx", ".xls"]:
        row["file_role_guess"] = "possible_excel_annotation_or_metadata"
        try:
            xl = pd.ExcelFile(p)
            row["read_status"] = "readable_excel"
            row["sheets_or_top_keys"] = clean_text(", ".join(xl.sheet_names))

            previews = []
            total_rows = 0

            for sheet in xl.sheet_names:
                df_sheet = pd.read_excel(p, sheet_name=sheet)
                total_rows += len(df_sheet)
                previews.append(
                    f"{sheet}: columns={list(df_sheet.columns)} rows={len(df_sheet)}"
                )

            row["columns_or_preview"] = clean_text(" | ".join(previews))
            row["row_count_or_length"] = total_rows

        except Exception as e:
            row["read_status"] = clean_text(f"excel_read_error: {type(e).__name__}: {e}")

    elif suffix == ".csv":
        row["file_role_guess"] = "possible_csv_annotation_or_metadata"
        try:
            df_csv = pd.read_csv(p)
            row["read_status"] = "readable_csv"
            row["columns_or_preview"] = clean_text(", ".join(map(str, df_csv.columns)))
            row["row_count_or_length"] = len(df_csv)
        except Exception as e:
            row["read_status"] = clean_text(f"csv_read_error: {type(e).__name__}: {e}")

    elif suffix == ".json":
        row["file_role_guess"] = "possible_json_annotation_or_metadata"
        try:
            with open(p) as f:
                data = json.load(f)

            if isinstance(data, dict):
                row["read_status"] = "readable_json_dict"
                row["sheets_or_top_keys"] = clean_text(", ".join(map(str, list(data.keys())[:40])))
                row["row_count_or_length"] = len(data)
            elif isinstance(data, list):
                row["read_status"] = "readable_json_list"
                row["row_count_or_length"] = len(data)
                if data and isinstance(data[0], dict):
                    row["sheets_or_top_keys"] = clean_text(", ".join(map(str, list(data[0].keys())[:40])))
            else:
                row["read_status"] = clean_text(f"readable_json_{type(data).__name__}")

        except Exception as e:
            row["read_status"] = clean_text(f"json_read_error: {type(e).__name__}: {e}")

    else:
        row["file_role_guess"] = "other_file"
        try:
            text = p.read_text(errors="ignore")
            row["read_status"] = "readable_text"
            row["columns_or_preview"] = clean_text(" | ".join(text.splitlines()[:8]))
            row["row_count_or_length"] = len(text.splitlines())
        except Exception as e:
            row["read_status"] = clean_text(f"text_read_error: {type(e).__name__}: {e}")

    rows.append(row)

files_df = pd.DataFrame(rows)

all_files_path = OUT_STATS / "work_unibo_folder_file_inventory.csv"
safe_to_csv(files_df, all_files_path)

non_video = files_df[files_df["suffix"] != ".mp4"].copy()
non_video_path = OUT_GT / "work_unibo_non_video_files_inspection.csv"
safe_to_csv(non_video, non_video_path)

preview_rows = []

for _, r in non_video.iterrows():
    p = Path(r["absolute_path"])
    suffix = p.suffix.lower()

    if suffix in [".xlsx", ".xls"]:
        try:
            xl = pd.ExcelFile(p)
            for sheet in xl.sheet_names:
                df_sheet = pd.read_excel(p, sheet_name=sheet)
                preview_rows.append({
                    "file": str(p),
                    "sheet": sheet,
                    "rows": len(df_sheet),
                    "columns": clean_text(", ".join(map(str, df_sheet.columns))),
                    "preview": clean_text(df_sheet.head(5).to_json(orient="records", force_ascii=False), 3000),
                })
        except Exception as e:
            preview_rows.append({
                "file": str(p),
                "sheet": "",
                "rows": "",
                "columns": "",
                "preview": clean_text(f"error: {type(e).__name__}: {e}"),
            })

    elif suffix == ".csv":
        try:
            df_csv = pd.read_csv(p)
            preview_rows.append({
                "file": str(p),
                "sheet": "",
                "rows": len(df_csv),
                "columns": clean_text(", ".join(map(str, df_csv.columns))),
                "preview": clean_text(df_csv.head(5).to_json(orient="records", force_ascii=False), 3000),
            })
        except Exception as e:
            preview_rows.append({
                "file": str(p),
                "sheet": "",
                "rows": "",
                "columns": "",
                "preview": clean_text(f"error: {type(e).__name__}: {e}"),
            })

preview_df = pd.DataFrame(preview_rows)
preview_path = OUT_GT / "work_unibo_annotation_file_previews.csv"
safe_to_csv(preview_df, preview_path)

note_path = NOTES / "work_unibo_folder_file_inspection.md"

with open(note_path, "w") as f:
    f.write("# Work Unibo Folder File Inspection\n\n")

    f.write("## Purpose\n\n")
    f.write(
        "This note inspects the direct contents of `/work/pig/datasets/Unibo`, which contains the raw Week 6 Unibo videos. "
        "The goal is to identify whether the same folder also contains annotation, Excel, CSV, JSON, or metadata files needed for ground-truth extraction.\n\n"
    )

    f.write("## Direct file inventory summary\n\n")
    if len(files_df):
        summary = files_df.groupby(["suffix", "file_role_guess"]).size().reset_index(name="file_count")
        f.write(summary.to_markdown(index=False))
    else:
        f.write("No files found.")
    f.write("\n\n")

    f.write("## Non-video files\n\n")
    if len(non_video):
        cols = [
            "absolute_path",
            "suffix",
            "size_kb",
            "file_role_guess",
            "read_status",
            "sheets_or_top_keys",
            "columns_or_preview",
            "row_count_or_length",
        ]
        f.write(non_video[cols].to_markdown(index=False))
    else:
        f.write("No non-video files found directly inside `/work/pig/datasets/Unibo`.")
    f.write("\n\n")

    f.write("## Annotation file previews\n\n")
    if len(preview_df):
        f.write(preview_df.to_markdown(index=False))
    else:
        f.write("No readable Excel/CSV annotation previews were generated.")
    f.write("\n\n")

    f.write("## Interpretation\n\n")
    f.write(
        "If non-video files include Excel/CSV/JSON annotations with frame, timestamp, pig colour/ID, or behaviour columns, "
        "they should become the primary source for the Week 6 unified ground-truth table. If not, the pipeline will rely on "
        "the existing Week 3 corrected scan-window annotations and the raw videos will be used mainly for visualization, feature extraction, and segmentation tests.\n"
    )

print("Saved:")
print(all_files_path)
print(non_video_path)
print(preview_path)
print(note_path)

print()
print("=== Direct /work Unibo file inventory summary ===")
print(files_df.groupby(["suffix", "file_role_guess"]).size().reset_index(name="file_count").to_string(index=False))

print()
print("=== Non-video files ===")
if len(non_video):
    cols = ["absolute_path", "suffix", "size_kb", "file_role_guess", "read_status", "sheets_or_top_keys", "columns_or_preview", "row_count_or_length"]
    print(non_video[cols].to_string(index=False))
else:
    print("No non-video files found.")

print()
print("=== Annotation previews ===")
print(preview_df.to_string(index=False) if len(preview_df) else "No annotation previews generated.")
