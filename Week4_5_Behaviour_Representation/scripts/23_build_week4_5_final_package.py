from pathlib import Path
import shutil
import zipfile
import os


ROOT = Path("Week4_5_Behaviour_Representation")
FINAL = ROOT / "final_outputs"
PKG = FINAL / "Week4_5_Final_Package"
ZIP_PATH = FINAL / "Week4_5_Behaviour_Representation_Final_Package.zip"

if PKG.exists():
    shutil.rmtree(PKG)

PKG.mkdir(parents=True, exist_ok=True)

copy_items = [
    ("final_outputs", ROOT / "final_outputs"),
    ("outputs", ROOT / "outputs"),
    ("notes", ROOT / "notes"),
    ("scripts", ROOT / "scripts"),
]

exclude_names = {
    "Week4_5_Final_Package",
    "Week4_5_Behaviour_Representation_Final_Package.zip",
}

exclude_suffixes = {
    ".tar",
    ".zip",
}

def should_exclude(path: Path):
    if path.name in exclude_names:
        return True

    # Exclude raw downloaded archives, but keep generated final package zip outside package.
    if path.suffix.lower() in exclude_suffixes:
        return True

    # Exclude very large raw dataset folder content from public_datasets if accidentally encountered.
    if "data/public_datasets/edinburgh/samples" in str(path):
        return True
    if "data/public_datasets/edinburgh/annotations" in str(path):
        return True

    return False

def copy_tree_filtered(src: Path, dst: Path):
    dst.mkdir(parents=True, exist_ok=True)

    for item in src.rglob("*"):
        if should_exclude(item):
            continue

        rel = item.relative_to(src)
        target = dst / rel

        if item.is_dir():
            target.mkdir(parents=True, exist_ok=True)
        elif item.is_file():
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(item, target)

for label, src in copy_items:
    if src.exists():
        copy_tree_filtered(src, PKG / label)

# Add small dataset metadata files, but not raw archives.
metadata_dir = PKG / "data_metadata"
metadata_dir.mkdir(parents=True, exist_ok=True)

small_metadata_files = [
    ROOT / "data/public_datasets/edinburgh/edinburgh_pigdata_page.html",
    ROOT / "data/public_datasets/edinburgh/edinburgh_gdrive_folder_listing.json",
    ROOT / "data/public_datasets/aggressive_pig_chicken/pig-and-chicken-behavior-dataset/README.md",
    ROOT / "data/public_datasets/aggressive_pig_chicken/pig-and-chicken-behavior-dataset/LICENSE",
]

for src in small_metadata_files:
    if src.exists():
        target = metadata_dir / src.name
        shutil.copy2(src, target)

readme = PKG / "README_Week4_5_Final_Package.md"
readme.write_text(
    "# Week 4-5 Final Package\n\n"
    "This package contains the Week 4-5 behaviour representation and feature engineering deliverables.\n\n"
    "## Included\n\n"
    "- Final report draft and validation files\n"
    "- Trajectory feature CSVs and trajectory visualizations\n"
    "- ROI/resource feature CSVs and ROI visualizations\n"
    "- Public dataset comparison outputs\n"
    "- Edinburgh annotation inspection outputs\n"
    "- Representation survey tables\n"
    "- Skeleton, mesh, and embedding review notes\n"
    "- Reproducibility scripts\n\n"
    "## Excluded\n\n"
    "Large raw public dataset archives are intentionally excluded:\n\n"
    "- `pigs161119.zip`\n"
    "- `annotated.tar`\n"
    "- other large raw dataset archives\n\n"
    "These files are not required for reviewing the final Week 4-5 deliverables and would make the package unnecessarily large.\n"
)

# Create package manifest
manifest_path = PKG / "PACKAGE_MANIFEST.tsv"
with open(manifest_path, "w") as f:
    f.write("size_bytes\trelative_path\n")
    for p in sorted(PKG.rglob("*")):
        if p.is_file():
            f.write(f"{p.stat().st_size}\t{p.relative_to(PKG)}\n")

# Zip package
if ZIP_PATH.exists():
    ZIP_PATH.unlink()

with zipfile.ZipFile(ZIP_PATH, "w", compression=zipfile.ZIP_DEFLATED) as z:
    for p in sorted(PKG.rglob("*")):
        if p.is_file():
            z.write(p, p.relative_to(FINAL))

print("Package folder:", PKG)
print("ZIP path:", ZIP_PATH)
print("ZIP size MB:", round(ZIP_PATH.stat().st_size / (1024 * 1024), 2))

print("\nTop-level package contents:")
for p in sorted(PKG.iterdir()):
    print("-", p.name)
