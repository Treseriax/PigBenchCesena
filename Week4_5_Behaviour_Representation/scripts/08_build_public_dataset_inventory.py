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

html = edinburgh_html_path.read_text(errors="ignore")
readme = aggressive_readme_path.read_text(errors="ignore") if aggressive_readme_path.exists() else ""

# Clean visible HTML text
text = re.sub(r"<script.*?</script>", " ", html, flags=re.S | re.I)
text = re.sub(r"<style.*?</style>", " ", text, flags=re.S | re.I)
text = re.sub(r"<[^>]+>", "\n", text)
text = re.sub(r"\n\s*\n+", "\n", text)
lines = [line.strip() for line in text.splitlines() if line.strip()]

# Find source zip names mentioned on Edinburgh page
zip_names = []
for line in lines:
    if re.search(r"pigs\d+.*\.zip", line):
        zip_names.append(line.strip())

# Behaviour/format related lines
important_terms = [
    "behavior label",
    "behaviour label",
    "bounding boxes",
    "tracking identifier",
    "7200",
    "labeled frames",
    "color.mp4",
    "depth.mp4",
    "output.json",
    "mask",
    "1800",
    "5 minutes",
    "registered color",
    "depth videos",
    "FRAMELIST",
    "behavior",
    "behaviour",
]

edinburgh_important = []
for line in lines:
    low = line.lower()
    if any(term.lower() in low for term in important_terms):
        edinburgh_important.append(line)

# Dataset comparison inventory
rows = [
    {
        "dataset": "Edinburgh Pig Behaviour Dataset",
        "species": "pig",
        "source_access": "Public web page with Google Drive links",
        "video_content": "23 days over 6 weeks of daytime video from a nearly overhead camera; registered colour and depth video",
        "animals": "Mostly 8 growing pigs in one pen",
        "video_format": "color.mp4 and depth.mp4 in clip folders; each clip is described as 5 minutes / 1800 frames",
        "annotation_format": "output.json with bounding boxes, persistent tracking identifier, and behaviour label",
        "annotation_granularity": "Frame-level descriptors for tracked pigs; 7200 labelled frames with 8 labelled pigs",
        "tracking_info": "Persistent tracking identifier included in ground-truth/detection JSON",
        "labels": "Behaviour labels are included, exact label set to be confirmed by downloading/inspecting output.json",
        "advantages": "Strong for behaviour representation because it includes video, depth, masks/pen area, bounding boxes, tracking IDs, and behaviour labels",
        "limitations": "Single pigpen and one camera setup; full download requires Google Drive access; automatic labels/results may need careful validation",
        "suitability_for_our_project": "High suitability for dataset comparison and trajectory/representation analysis; useful reference dataset before transferring to Unibo data",
        "download_status": "Source page downloaded; full data not yet downloaded"
    },
    {
        "dataset": "Aggressive Pig/Chicken Behaviour Dataset",
        "species": "pig and chicken",
        "source_access": "GitHub README with Baidu Pan links",
        "video_content": "Dataset associated with pig/chicken behaviour recognition; exact files require external Baidu download",
        "animals": "Pig aggression dataset plus additional pig posture and caged chicken feeding datasets mentioned",
        "video_format": "Not visible in GitHub repo; requires dataset download",
        "annotation_format": "Not visible in GitHub repo; requires dataset download",
        "annotation_granularity": "Likely clip-level for aggression recognition based on TSM paper, but needs downloaded dataset confirmation",
        "tracking_info": "No tracking information visible in README",
        "labels": "Aggression/non-aggression for TSM paper; README also mentions pig lying/standing/sitting and chicken feeding datasets",
        "advantages": "Useful as a video-based behaviour recognition benchmark, especially for aggressive behaviour and temporal modelling",
        "limitations": "Data not stored directly in GitHub; Baidu access may require manual download; metadata is limited in README",
        "suitability_for_our_project": "Medium suitability; useful for video classification comparison, less directly useful for trajectory representation unless annotations are available",
        "download_status": "README downloaded/cloned; full data not yet downloaded"
    },
    {
        "dataset": "Unibo corrected scan-window dataset",
        "species": "pig",
        "source_access": "Our Week 3 corrected outputs",
        "video_content": "Six manually aligned scan-window segments corresponding to 09:00, 09:10, 09:20, 09:30, 09:40, 09:50 observations",
        "animals": "Tracked pigs in Unibo videos",
        "video_format": "Corrected scan-window video segments and visualization videos",
        "annotation_format": "Corrected JSON with frame, track ID, bbox, segment, Excel scan-window labels, and conservative identity policy",
        "annotation_granularity": "Frame-level tracking plus segment-level behaviour labels from Excel scan sampling",
        "tracking_info": "YOLOv8-s detection + ByteTrack tracking; frame-level bbox and track IDs",
        "labels": "Ethogram labels from Excel: eating, lying, standing, exploring, interaction, BOX/out of view, etc.",
        "advantages": "Directly connected to our project; already has tracking, behaviour labels, visualization, trajectory features, and ROI prototype",
        "limitations": "Identity mapping remains conservative; scan-window labels are segment-level; snout/skeleton keypoints are not available yet",
        "suitability_for_our_project": "Primary internal dataset for trajectory and ROI feature engineering",
        "download_status": "Available locally from Week 3 outputs"
    }
]

inventory = pd.DataFrame(rows)
inventory_path = OUT / "dataset_inventory_comparison.csv"
inventory.to_csv(inventory_path, index=False)

zip_df = pd.DataFrame({"edinburgh_zip_or_file_line": zip_names})
zip_path = OUT / "edinburgh_zip_file_mentions.csv"
zip_df.to_csv(zip_path, index=False)

important_df = pd.DataFrame({"edinburgh_important_line": edinburgh_important})
important_path = OUT / "edinburgh_important_page_lines.csv"
important_df.to_csv(important_path, index=False)

# Markdown report
report_path = NOTES / "public_dataset_inventory_summary.md"

with open(report_path, "w") as f:
    f.write("# Public Dataset Inventory Summary\n\n")

    f.write("## Purpose\n\n")
    f.write(
        "This note summarizes the public datasets requested in the Week 4 task sheet "
        "and compares them with the internal Unibo corrected scan-window dataset.\n\n"
    )

    f.write("## Dataset comparison table\n\n")
    f.write(inventory.to_markdown(index=False))
    f.write("\n\n")

    f.write("## Edinburgh dataset extracted facts\n\n")
    f.write(
        "- The Edinburgh page describes 23 days over 6 weeks of daytime pig video.\n"
        "- The video was captured from a nearly overhead camera.\n"
        "- Registered colour and depth videos are available.\n"
        "- Most frames show 8 pigs.\n"
        "- Ground-truth data includes axis-aligned bounding boxes, persistent tracking identifiers, and behaviour labels.\n"
        "- The page states a total of 7200 labelled frames, each with 8 labelled pigs.\n"
        "- Each clip folder contains `color.mp4`, `depth.mp4`, and `output.json`.\n"
        "- Each subfolder is described as one 5-minute video clip with 1800 frames.\n\n"
    )

    f.write("## Aggressive pig/chicken dataset extracted facts\n\n")
    f.write(
        "- The GitHub repository contains only README and metadata.\n"
        "- The README provides Baidu Pan links and password `1234`.\n"
        "- The main dataset is associated with the paper *Efficient Aggressive Behavior Recognition of Pigs Based on Temporal Shift Module*.\n"
        "- Additional links mention pig lying/standing/sitting posture data and caged chicken feeding data.\n"
        "- Full structure and annotation format require manual or successful Baidu download.\n\n"
    )

    f.write("## Current decision\n\n")
    f.write(
        "The Edinburgh dataset is the stronger immediate target for structured inspection because its source page already describes "
        "video files, JSON annotations, tracking IDs, and behaviour labels. The aggressive pig/chicken dataset remains useful as "
        "a video-classification reference, but full inspection depends on successful download from Baidu Pan.\n"
    )

print("Saved:")
print(inventory_path)
print(zip_path)
print(important_path)
print(report_path)

print()
print("=== Inventory table ===")
print(inventory[["dataset", "annotation_format", "annotation_granularity", "tracking_info", "suitability_for_our_project"]].to_string(index=False))

print()
print("=== Edinburgh zip/file mentions ===")
print(zip_df.head(40).to_string(index=False))
