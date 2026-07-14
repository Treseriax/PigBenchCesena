from pathlib import Path
import shutil
import zipfile


ROOT = Path("Week4_5_Behaviour_Representation")
FINAL = ROOT / "final_outputs"
PKG = FINAL / "Week4_5_Final_Package"
ZIP_PATH = FINAL / "Week4_5_Behaviour_Representation_Final_Package.zip"

# Clean old partial package
if PKG.exists():
    shutil.rmtree(PKG)

if ZIP_PATH.exists():
    ZIP_PATH.unlink()

PKG.mkdir(parents=True, exist_ok=True)

def copy_file(src: Path, dst: Path):
    if src.exists() and src.is_file():
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, dst)
        return True
    return False

def copy_tree(src: Path, dst: Path, exclude_suffixes=None):
    exclude_suffixes = exclude_suffixes or set()
    if not src.exists():
        return

    for item in src.rglob("*"):
        if item.is_dir():
            continue

        if item.suffix.lower() in exclude_suffixes:
            continue

        rel = item.relative_to(src)
        target = dst / rel
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(item, target)

# 1) Final report files only
final_report_dir = PKG / "final_report"
final_report_files = [
    FINAL / "Week4_5_Final_Report_Draft.md",
    FINAL / "Week4_5_Final_Report_Outline.md",
    FINAL / "Week4_5_Final_Report_Validation.md",
]

for src in final_report_files:
    copy_file(src, final_report_dir / src.name)

# 2) Outputs, notes, scripts
copy_tree(ROOT / "outputs", PKG / "outputs")
copy_tree(ROOT / "notes", PKG / "notes")
copy_tree(ROOT / "scripts", PKG / "scripts")

# 3) Small metadata only, no large raw archives
metadata_dir = PKG / "data_metadata"
metadata_dir.mkdir(parents=True, exist_ok=True)

metadata_files = [
    ROOT / "data/public_datasets/edinburgh/edinburgh_pigdata_page.html",
    ROOT / "data/public_datasets/edinburgh/edinburgh_gdrive_folder_listing.json",
    ROOT / "data/public_datasets/aggressive_pig_chicken/pig-and-chicken-behavior-dataset/README.md",
    ROOT / "data/public_datasets/aggressive_pig_chicken/pig-and-chicken-behavior-dataset/LICENSE",
]

for src in metadata_files:
    copy_file(src, metadata_dir / src.name)

# 4) README
readme = PKG / "README_Week4_5_Final_Package.md"
readme.write_text(
    "# Week 4-5 Final Package\n\n"
    "This package contains the Week 4-5 behaviour representation and feature engineering deliverables.\n\n"
    "## Included\n\n"
    "- Final report draft, outline, and validation files\n"
    "- Trajectory feature CSVs and trajectory visualizations\n"
    "- ROI/resource feature CSVs and ROI visualizations\n"
    "- Public dataset comparison outputs\n"
    "- Edinburgh annotation inspection outputs\n"
    "- Representation survey tables\n"
    "- Skeleton, mesh, and embedding review notes\n"
    "- Reproducibility scripts\n\n"
    "## Excluded\n\n"
    "Large raw public dataset archives are intentionally excluded, including `pigs161119.zip` and `annotated.tar`.\n"
)

# 5) Manifest
manifest_path = PKG / "PACKAGE_MANIFEST.tsv"
with open(manifest_path, "w") as f:
    f.write("size_bytes\trelative_path\n")
    for p in sorted(PKG.rglob("*")):
        if p.is_file():
            f.write(f"{p.stat().st_size}\t{p.relative_to(PKG)}\n")

# 6) ZIP
with zipfile.ZipFile(ZIP_PATH, "w", compression=zipfile.ZIP_DEFLATED) as z:
    for p in sorted(PKG.rglob("*")):
        if p.is_file():
            z.write(p, p.relative_to(FINAL))

print("Package folder:", PKG)
print("Package folder size MB:", round(sum(p.stat().st_size for p in PKG.rglob('*') if p.is_file()) / (1024 * 1024), 2))
print("ZIP path:", ZIP_PATH)
print("ZIP size MB:", round(ZIP_PATH.stat().st_size / (1024 * 1024), 2))
print("Manifest:", manifest_path)

print("\nTop-level package contents:")
for p in sorted(PKG.iterdir()):
    print("-", p.name)
