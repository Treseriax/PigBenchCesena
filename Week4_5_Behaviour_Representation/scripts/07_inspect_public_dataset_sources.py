from pathlib import Path
import re
import pandas as pd


ROOT = Path("Week4_5_Behaviour_Representation")
DATA = ROOT / "data/public_datasets"
OUT = ROOT / "outputs/dataset_comparison"
NOTES = ROOT / "notes/public_dataset_notes"

OUT.mkdir(parents=True, exist_ok=True)
NOTES.mkdir(parents=True, exist_ok=True)

edinburgh_html_path = DATA / "edinburgh/edinburgh_pigdata_page.html"
aggressive_readme_path = DATA / "aggressive_pig_chicken/pig-and-chicken-behavior-dataset/README.md"

# -----------------------------
# Edinburgh HTML inspection
# -----------------------------
html = edinburgh_html_path.read_text(errors="ignore")

hrefs = re.findall(r'href="([^"]+)"', html)
google_drive_links = [h for h in hrefs if "drive.google.com" in h]
pdf_links = [h for h in hrefs if h.lower().endswith(".pdf") or "/PAPERS/" in h]
local_links = [h for h in hrefs if "drive.google.com" not in h and not h.startswith("http")]

# Crude text cleanup for page notes
text = re.sub(r"<script.*?</script>", " ", html, flags=re.S | re.I)
text = re.sub(r"<style.*?</style>", " ", text, flags=re.S | re.I)
text = re.sub(r"<[^>]+>", "\n", text)
text = re.sub(r"\n\s*\n+", "\n", text)
text = "\n".join([line.strip() for line in text.splitlines() if line.strip()])

# Find potentially useful keyword lines
keywords = [
    "behaviour", "behavior", "annotation", "json", "video",
    "tracking", "mask", "dataset", "pig", "posture", "feeding",
    "drinking", "standing", "lying", "walking"
]

keyword_lines = []
for line in text.splitlines():
    low = line.lower()
    if any(k in low for k in keywords):
        keyword_lines.append(line)

# -----------------------------
# Aggressive README inspection
# -----------------------------
if aggressive_readme_path.exists():
    readme = aggressive_readme_path.read_text(errors="ignore")
else:
    readme = ""

readme_links = re.findall(r'https?://[^\s\)\]]+', readme)
readme_drive_links = [l for l in readme_links if "drive.google.com" in l or "pan.baidu" in l or "figshare" in l]
readme_keyword_lines = []
for line in readme.splitlines():
    low = line.lower()
    if any(k in low for k in keywords + ["download", "class", "clip", "duration", "aggressive", "chicken"]):
        readme_keyword_lines.append(line.strip())

# -----------------------------
# Save link inventories
# -----------------------------
rows = []

for link in google_drive_links:
    rows.append({
        "dataset": "edinburgh",
        "link_type": "google_drive",
        "link": link
    })

for link in pdf_links:
    rows.append({
        "dataset": "edinburgh",
        "link_type": "paper_or_pdf",
        "link": link
    })

for link in local_links:
    rows.append({
        "dataset": "edinburgh",
        "link_type": "local_or_anchor",
        "link": link
    })

for link in readme_links:
    rows.append({
        "dataset": "aggressive_pig_chicken",
        "link_type": "readme_external",
        "link": link
    })

links_df = pd.DataFrame(rows)
links_csv = OUT / "public_dataset_source_links.csv"
links_df.to_csv(links_csv, index=False)

# -----------------------------
# Save notes
# -----------------------------
edinburgh_note = NOTES / "edinburgh_source_page_inspection.md"
with open(edinburgh_note, "w") as f:
    f.write("# Edinburgh Pig Behaviour Dataset — Source Page Inspection\n\n")
    f.write(f"Source HTML: `{edinburgh_html_path}`\n\n")
    f.write("## Link summary\n\n")
    f.write(f"- Total href links: {len(hrefs)}\n")
    f.write(f"- Google Drive links: {len(google_drive_links)}\n")
    f.write(f"- Paper/PDF links: {len(pdf_links)}\n")
    f.write(f"- Local/anchor links: {len(local_links)}\n\n")

    f.write("## Google Drive links\n\n")
    for i, link in enumerate(google_drive_links, start=1):
        f.write(f"{i}. {link}\n")

    f.write("\n## Paper/PDF links\n\n")
    for i, link in enumerate(pdf_links, start=1):
        f.write(f"{i}. {link}\n")

    f.write("\n## Keyword lines from page\n\n")
    for line in keyword_lines[:200]:
        f.write(f"- {line}\n")

aggressive_note = NOTES / "aggressive_pig_chicken_readme_inspection.md"
with open(aggressive_note, "w") as f:
    f.write("# Aggressive Pig/Chicken Dataset — README Inspection\n\n")
    f.write(f"README: `{aggressive_readme_path}`\n\n")
    f.write("## README links\n\n")
    if readme_links:
        for i, link in enumerate(readme_links, start=1):
            f.write(f"{i}. {link}\n")
    else:
        f.write("No external links found in README.\n")

    f.write("\n## Keyword lines from README\n\n")
    if readme_keyword_lines:
        for line in readme_keyword_lines[:200]:
            f.write(f"- {line}\n")
    else:
        f.write("No keyword lines found.\n")

    f.write("\n## Full README preview\n\n")
    f.write("```text\n")
    f.write(readme[:5000])
    f.write("\n```\n")

# -----------------------------
# Console output
# -----------------------------
print("=== Edinburgh page ===")
print("Total href links:", len(hrefs))
print("Google Drive links:", len(google_drive_links))
print("PDF links:", len(pdf_links))
print()
print("First 20 Google Drive links:")
for link in google_drive_links[:20]:
    print(link)

print()
print("=== Aggressive README ===")
print("README exists:", aggressive_readme_path.exists())
print("External README links:", len(readme_links))
for link in readme_links:
    print(link)

print()
print("Saved:")
print(links_csv)
print(edinburgh_note)
print(aggressive_note)
