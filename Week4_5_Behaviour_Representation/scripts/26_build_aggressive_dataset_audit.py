from pathlib import Path
import re
import pandas as pd


ROOT = Path("Week4_5_Behaviour_Representation")
OUT = ROOT / "outputs/dataset_comparison"
NOTES = ROOT / "notes/public_dataset_notes"

OUT.mkdir(parents=True, exist_ok=True)
NOTES.mkdir(parents=True, exist_ok=True)

REPO = ROOT / "data/public_datasets/aggressive_pig_chicken/pig-and-chicken-behavior-dataset"
README = REPO / "README.md"

repo_exists = REPO.exists()
readme_exists = README.exists()

readme_text = README.read_text(errors="ignore") if readme_exists else ""

# Extract possible URLs / Baidu mentions / password-like strings
urls = re.findall(r"https?://\S+", readme_text)
baidu_lines = [line.strip() for line in readme_text.splitlines() if "baidu" in line.lower() or "pan.baidu" in line.lower()]
password_lines = [line.strip() for line in readme_text.splitlines() if "password" in line.lower() or "pwd" in line.lower() or "提取" in line]

file_rows = []
if repo_exists:
    for p in sorted(REPO.rglob("*")):
        if p.is_file():
            file_rows.append({
                "relative_path": str(p.relative_to(REPO)),
                "size_bytes": p.stat().st_size,
                "suffix": p.suffix
            })

file_inventory = pd.DataFrame(file_rows)
file_inventory_path = OUT / "aggressive_dataset_repo_file_inventory.csv"
file_inventory.to_csv(file_inventory_path, index=False)

link_rows = []
for u in urls:
    link_rows.append({
        "extracted_url": u,
        "contains_baidu": "baidu" in u.lower(),
        "interpretation": "External dataset/download link" if "baidu" in u.lower() else "Related URL from README"
    })

link_table = pd.DataFrame(link_rows)
link_table_path = OUT / "aggressive_dataset_readme_links.csv"
link_table.to_csv(link_table_path, index=False)

audit_rows = [
    {
        "criterion": "Repository accessible locally",
        "status": "yes" if repo_exists else "no",
        "evidence": str(REPO)
    },
    {
        "criterion": "README available",
        "status": "yes" if readme_exists else "no",
        "evidence": str(README)
    },
    {
        "criterion": "Actual dataset files visible in GitHub clone",
        "status": "no" if len(file_inventory) <= 2 else "partial",
        "evidence": f"{len(file_inventory)} files found in repository clone"
    },
    {
        "criterion": "External Baidu links mentioned",
        "status": "yes" if baidu_lines or any('baidu' in u.lower() for u in urls) else "unknown",
        "evidence": "Baidu-related lines or URLs found in README" if baidu_lines or any('baidu' in u.lower() for u in urls) else "No Baidu line detected automatically"
    },
    {
        "criterion": "Annotation format directly inspectable",
        "status": "no",
        "evidence": "No annotation files are visible in the cloned repository"
    },
    {
        "criterion": "Video clip duration directly inspectable",
        "status": "no",
        "evidence": "No video files are visible in the cloned repository"
    },
    {
        "criterion": "Behaviour classes directly inspectable from repository",
        "status": "partial",
        "evidence": "README and TSM article indicate aggressive/non-aggressive style usage, but full class files are not visible locally"
    },
    {
        "criterion": "Suitability for trajectory/ROI feature engineering",
        "status": "low",
        "evidence": "Dataset appears more suitable for video-based behaviour recognition than trajectory-based representation because no tracking files are visible"
    },
    {
        "criterion": "Suitability for video-temporal representation review",
        "status": "medium-high",
        "evidence": "Useful as a TSM/aggression-recognition reference dataset"
    },
]

audit = pd.DataFrame(audit_rows)
audit_path = OUT / "aggressive_dataset_access_audit.csv"
audit.to_csv(audit_path, index=False)

note_path = NOTES / "aggressive_dataset_access_audit.md"

with open(note_path, "w") as f:
    f.write("# Aggressive Pig/Chicken Dataset Access Audit\n\n")

    f.write("## Purpose\n\n")
    f.write(
        "This note strengthens the public dataset exploration section by documenting what could and could not be inspected "
        "from the aggressive pig/chicken behaviour dataset repository. The goal is to be explicit about dataset accessibility, "
        "annotation visibility, and why this dataset is treated as a video-temporal recognition reference rather than a trajectory/ROI feature dataset.\n\n"
    )

    f.write("## Repository file inventory\n\n")
    if len(file_inventory):
        f.write(file_inventory.to_markdown(index=False))
    else:
        f.write("No local files found in the cloned repository.")
    f.write("\n\n")

    f.write("## Extracted README links\n\n")
    if len(link_table):
        f.write(link_table.to_markdown(index=False))
    else:
        f.write("No URLs automatically extracted from README.")
    f.write("\n\n")

    f.write("## Baidu / external access evidence\n\n")
    if baidu_lines:
        for line in baidu_lines:
            f.write(f"- {line}\n")
    else:
        f.write("No Baidu-specific line was automatically extracted, but previous inspection indicated that the full data is externally distributed.\n")
    f.write("\n\n")

    f.write("## Audit table\n\n")
    f.write(audit.to_markdown(index=False))
    f.write("\n\n")

    f.write("## Interpretation\n\n")
    f.write(
        "The cloned GitHub repository does not provide directly inspectable video files, annotation files, or tracking outputs. "
        "Therefore, this dataset should not be used as the main source for trajectory or ROI feature engineering. "
        "It is more appropriate as a video-based behaviour recognition reference, especially for aggressive/non-aggressive temporal modelling. "
        "The main internal feature-engineering dataset remains the Unibo corrected scan-window dataset, while Edinburgh remains the stronger public "
        "reference for annotation structure and tracking-related behaviour representation.\n"
    )

print("Saved:")
print(file_inventory_path)
print(link_table_path)
print(audit_path)
print(note_path)

print()
print("=== Aggressive dataset audit ===")
print(audit.to_string(index=False))
