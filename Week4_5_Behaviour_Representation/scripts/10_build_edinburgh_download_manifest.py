from pathlib import Path
import json
import re
import pandas as pd


ROOT = Path("Week4_5_Behaviour_Representation")
EDIN = ROOT / "data/public_datasets/edinburgh"
OUT = ROOT / "outputs/dataset_comparison"
NOTES = ROOT / "notes/public_dataset_notes"

listing_path = EDIN / "edinburgh_gdrive_folder_listing.json"
context_path = OUT / "edinburgh_zip_link_context.txt"

with open(listing_path) as f:
    listing = json.load(f)

list_rows = []
for item in listing:
    path = item.get("path", "")
    url = item.get("url", "")
    m = re.search(r"id=([^&]+)", url)
    file_id = m.group(1) if m else ""
    list_rows.append({
        "path": path,
        "file_name": Path(path).name,
        "file_id": file_id,
        "url": url
    })

listing_df = pd.DataFrame(list_rows)

context_rows = []
if context_path.exists():
    for line in context_path.read_text(errors="ignore").splitlines():
        # Example:
        # 137:<tr><td>16/11/19</td><td><a href="...">pigs161119.zip</a></td><td>2.6</td><td>10</td><td>67-76</td></tr>
        m = re.search(
            r"<td>([^<]+)</td><td><a href=\"([^\"]+)\">([^<]+)</a></td><td>([^<]+)</td><td>([^<]+)</td><td>([^<]+)</td>",
            line
        )
        if m:
            context_rows.append({
                "date_label": m.group(1),
                "html_link": m.group(2),
                "file_name": m.group(3),
                "listed_size": m.group(4),
                "num_clips": m.group(5),
                "clip_index_range": m.group(6)
            })

context_df = pd.DataFrame(context_rows)

manifest = listing_df.merge(context_df, on="file_name", how="left")

# Put downloadable pig zip files first, smaller listed sizes first
manifest["listed_size_num"] = pd.to_numeric(manifest["listed_size"], errors="coerce")
manifest["num_clips_num"] = pd.to_numeric(manifest["num_clips"], errors="coerce")
manifest["is_pig_zip"] = manifest["file_name"].str.startswith("pigs") & manifest["file_name"].str.endswith(".zip")

manifest = manifest.sort_values(
    ["is_pig_zip", "listed_size_num", "num_clips_num"],
    ascending=[False, True, True]
)

manifest_path = OUT / "edinburgh_download_manifest.csv"
manifest.to_csv(manifest_path, index=False)

note_path = NOTES / "edinburgh_download_manifest_summary.md"
with open(note_path, "w") as f:
    f.write("# Edinburgh Download Manifest Summary\n\n")
    f.write("## Purpose\n\n")
    f.write("This note maps Edinburgh Google Drive file IDs to dataset filenames and source-page metadata.\n\n")
    f.write("## Recommended sample\n\n")
    sample = manifest[manifest["is_pig_zip"]].head(1)
    if len(sample):
        s = sample.iloc[0]
        f.write(f"- Recommended first sample: `{s['file_name']}`\n")
        f.write(f"- File ID: `{s['file_id']}`\n")
        f.write(f"- Listed size: `{s['listed_size']}`\n")
        f.write(f"- Number of clips: `{s['num_clips']}`\n")
        f.write(f"- Clip index range: `{s['clip_index_range']}`\n\n")

    f.write("## Top small pig zip candidates\n\n")
    cols = ["file_name", "file_id", "listed_size", "num_clips", "clip_index_range"]
    f.write(manifest[manifest["is_pig_zip"]][cols].head(8).to_markdown(index=False))
    f.write("\n\n")

print("Saved:", manifest_path)
print("Saved:", note_path)
print()
print("=== Top candidates ===")
cols = ["file_name", "file_id", "listed_size", "num_clips", "clip_index_range", "url"]
print(manifest[manifest["is_pig_zip"]][cols].head(10).to_string(index=False))
