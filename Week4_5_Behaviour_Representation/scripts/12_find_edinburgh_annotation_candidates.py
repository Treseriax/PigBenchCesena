from pathlib import Path
import json
import re
import pandas as pd


ROOT = Path("Week4_5_Behaviour_Representation")
EDIN = ROOT / "data/public_datasets/edinburgh"
OUT = ROOT / "outputs/dataset_comparison"
NOTES = ROOT / "notes/public_dataset_notes"

listing_path = EDIN / "edinburgh_gdrive_folder_listing.json"

with open(listing_path) as f:
    listing = json.load(f)

rows = []

for item in listing:
    path = item.get("path", "")
    url = item.get("url", "")
    m = re.search(r"id=([^&]+)", url)
    file_id = m.group(1) if m else ""

    low = path.lower()

    candidate_reason = []
    if "annot" in low:
        candidate_reason.append("annotation keyword")
    if "result" in low:
        candidate_reason.append("result keyword")
    if low.endswith(".json"):
        candidate_reason.append("json file")
    if low.endswith(".tar") or low.endswith(".tar.gz") or low.endswith(".zip"):
        candidate_reason.append("archive file")

    rows.append({
        "path": path,
        "file_name": Path(path).name,
        "file_id": file_id,
        "url": url,
        "candidate_reason": "; ".join(candidate_reason),
        "is_annotation_candidate": int(any(r in candidate_reason for r in ["annotation keyword", "result keyword", "json file"]))
    })

df = pd.DataFrame(rows)

out_all = OUT / "edinburgh_all_gdrive_listing_structured.csv"
out_candidates = OUT / "edinburgh_annotation_candidates.csv"

df.to_csv(out_all, index=False)
df[df["is_annotation_candidate"] == 1].to_csv(out_candidates, index=False)

note_path = NOTES / "edinburgh_annotation_candidate_summary.md"

with open(note_path, "w") as f:
    f.write("# Edinburgh Annotation Candidate Summary\n\n")
    f.write("## Purpose\n\n")
    f.write(
        "The downloaded `pigs161119.zip` sample contains raw source clips but no `output.json`. "
        "This note identifies possible annotation/result packages from the Google Drive folder listing.\n\n"
    )

    f.write("## Candidate files\n\n")
    candidates = df[df["is_annotation_candidate"] == 1]
    if len(candidates):
        f.write(candidates[["file_name", "file_id", "url", "candidate_reason"]].to_markdown(index=False))
        f.write("\n\n")
    else:
        f.write("No explicit annotation candidates found in folder listing.\n\n")

    f.write("## Interpretation\n\n")
    f.write(
        "If `annotated.tar` exists, it is likely the first package to inspect for ground-truth annotations. "
        "`results_dataset.tar.gz` is likely related to automatically detected/tracked results and may be much larger. "
        "The next safe step is to inspect or download the smaller annotation candidate first if possible.\n"
    )

print("Saved:")
print(out_all)
print(out_candidates)
print(note_path)

print()
print("=== Annotation candidates ===")
cand = df[df["is_annotation_candidate"] == 1]
if len(cand):
    print(cand[["file_name", "file_id", "url", "candidate_reason"]].to_string(index=False))
else:
    print("No explicit annotation candidates found.")
