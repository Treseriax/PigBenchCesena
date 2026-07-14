from pathlib import Path
import zipfile
import pandas as pd
from collections import Counter, defaultdict


ROOT = Path("Week4_5_Behaviour_Representation")
ZIP_PATH = ROOT / "data/public_datasets/edinburgh/samples/pigs161119.zip"
OUT = ROOT / "outputs/dataset_comparison"
NOTES = ROOT / "notes/public_dataset_notes"

OUT.mkdir(parents=True, exist_ok=True)
NOTES.mkdir(parents=True, exist_ok=True)

if not ZIP_PATH.exists():
    raise FileNotFoundError(ZIP_PATH)

rows = []

with zipfile.ZipFile(ZIP_PATH, "r") as z:
    infos = z.infolist()

    for info in infos:
        name = info.filename
        parts = Path(name).parts

        clip_id = ""
        if len(parts) >= 2 and parts[0].startswith("PIGS"):
            clip_id = parts[1]

        suffix = Path(name).suffix.lower()
        file_name = Path(name).name

        rows.append({
            "path": name,
            "clip_id": clip_id,
            "file_name": file_name,
            "suffix": suffix,
            "size_bytes": info.file_size,
            "compressed_size_bytes": info.compress_size,
            "is_dir": name.endswith("/")
        })

df = pd.DataFrame(rows)

inventory_path = OUT / "edinburgh_pigs161119_zip_inventory.csv"
df.to_csv(inventory_path, index=False)

file_df = df[~df["is_dir"]].copy()

clip_summary = (
    file_df
    .groupby("clip_id")
    .agg(
        file_count=("path", "count"),
        total_size_bytes=("size_bytes", "sum"),
        has_color_mp4=("file_name", lambda x: int("color.mp4" in set(x))),
        has_depth_mp4=("file_name", lambda x: int("depth.mp4" in set(x))),
        has_background_png=("file_name", lambda x: int("background.png" in set(x))),
        has_background_depth_png=("file_name", lambda x: int("background_depth.png" in set(x))),
        has_mask_png=("file_name", lambda x: int("mask.png" in set(x))),
        has_times_txt=("file_name", lambda x: int("times.txt" in set(x))),
        has_output_json=("file_name", lambda x: int("output.json" in set(x))),
    )
    .reset_index()
)

clip_summary_path = OUT / "edinburgh_pigs161119_clip_summary.csv"
clip_summary.to_csv(clip_summary_path, index=False)

suffix_summary = (
    file_df
    .groupby(["suffix", "file_name"])
    .agg(
        count=("path", "count"),
        total_size_bytes=("size_bytes", "sum")
    )
    .reset_index()
    .sort_values(["suffix", "file_name"])
)

suffix_summary_path = OUT / "edinburgh_pigs161119_filetype_summary.csv"
suffix_summary.to_csv(suffix_summary_path, index=False)

output_json_paths = file_df[file_df["file_name"] == "output.json"]["path"].tolist()

note_path = NOTES / "edinburgh_pigs161119_sample_zip_inspection.md"

with open(note_path, "w") as f:
    f.write("# Edinburgh `pigs161119.zip` Sample Inspection\n\n")

    f.write("## Sample file\n\n")
    f.write(f"- Zip path: `{ZIP_PATH}`\n")
    f.write(f"- Total entries: `{len(df)}`\n")
    f.write(f"- File entries: `{len(file_df)}`\n")
    f.write(f"- Directory entries: `{int(df['is_dir'].sum())}`\n\n")

    f.write("## Clip summary\n\n")
    f.write(clip_summary.to_markdown(index=False))
    f.write("\n\n")

    f.write("## File type summary\n\n")
    f.write(suffix_summary.to_markdown(index=False))
    f.write("\n\n")

    f.write("## Annotation file check\n\n")
    if output_json_paths:
        f.write("`output.json` files were found:\n\n")
        for p in output_json_paths:
            f.write(f"- `{p}`\n")
    else:
        f.write(
            "`output.json` was not found inside this sample zip based on zip inventory. "
            "This suggests that this sample package contains raw source video/depth/background/mask files, "
            "while ground-truth or automatic result annotations may be distributed separately.\n"
        )

print("Saved:")
print(inventory_path)
print(clip_summary_path)
print(suffix_summary_path)
print(note_path)

print()
print("=== Clip summary ===")
print(clip_summary.to_string(index=False))

print()
print("=== File type summary ===")
print(suffix_summary.to_string(index=False))

print()
print("=== output.json paths ===")
if output_json_paths:
    for p in output_json_paths:
        print(p)
else:
    print("No output.json found in this zip.")
