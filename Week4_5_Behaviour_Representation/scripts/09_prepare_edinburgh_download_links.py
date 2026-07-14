from pathlib import Path
import re
import pandas as pd


ROOT = Path("Week4_5_Behaviour_Representation")
OUT = ROOT / "outputs/dataset_comparison"
DATA = ROOT / "data/public_datasets/edinburgh"

links_csv = OUT / "public_dataset_source_links.csv"
links = pd.read_csv(links_csv)

edin = links[
    (links["dataset"] == "edinburgh") &
    (links["link_type"] == "google_drive")
].copy()

rows = []

for _, row in edin.iterrows():
    link = row["link"]

    file_id = None
    folder_id = None

    m_file = re.search(r"/file/d/([^/]+)", link)
    m_folder = re.search(r"/drive/folders/([^/?]+)", link)

    if m_file:
        file_id = m_file.group(1)
        link_kind = "file"
    elif m_folder:
        folder_id = m_folder.group(1)
        link_kind = "folder"
    else:
        link_kind = "unknown"

    rows.append({
        "link_kind": link_kind,
        "file_id": file_id,
        "folder_id": folder_id,
        "original_link": link,
        "gdown_file_command": f"gdown --id {file_id}" if file_id else "",
        "gdown_folder_command": f"gdown --folder {link}" if folder_id else "",
    })

df = pd.DataFrame(rows)

out_path = OUT / "edinburgh_google_drive_ids.csv"
df.to_csv(out_path, index=False)

print("Saved:", out_path)
print()
print(df.to_string(index=False))
