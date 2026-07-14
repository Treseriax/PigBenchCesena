from pathlib import Path
import csv
import hashlib
import shutil
import zipfile
from datetime import datetime

import pandas as pd


ROOT = Path.home() / "PigBench"
W6 = ROOT / "Week6_Unibo_Dataset_Validation"

V2_DIR = W6 / "final_delivery_package_v2" / "Week6_Unibo_Dataset_Validation_Final_Package_v2"

PKG_ROOT = W6 / "final_delivery_package_v3"
PKG_NAME = "Week6_Unibo_Dataset_Validation_Final_Package_v3"
PKG_DIR = PKG_ROOT / PKG_NAME
ZIP_PATH = PKG_ROOT / f"{PKG_NAME}.zip"

STATS = W6 / "outputs" / "dataset_statistics"
FEAT = W6 / "outputs" / "feature_extractors"
VIS = W6 / "outputs" / "visual_label_check"
NOTES = W6 / "notes"
SCRIPTS = W6 / "scripts"
PRES = W6 / "final_presentation"

PKG_ROOT.mkdir(parents=True, exist_ok=True)


def safe_to_csv(df, path):
    df.to_csv(
        path,
        index=False,
        quoting=csv.QUOTE_ALL,
        escapechar="\\",
        lineterminator="\n",
    )


def copy_one(src, dst_rel):
    src = Path(src)
    dst = PKG_DIR / dst_rel
    dst.parent.mkdir(parents=True, exist_ok=True)

    if not src.exists():
        return {
            "source": str(src),
            "package_path": str(dst_rel),
            "status": "missing",
            "size_mb": "",
        }

    if not src.is_file():
        return {
            "source": str(src),
            "package_path": str(dst_rel),
            "status": "not_file",
            "size_mb": "",
        }

    shutil.copy2(src, dst)

    return {
        "source": str(src),
        "package_path": str(dst_rel),
        "status": "copied",
        "size_mb": round(dst.stat().st_size / (1024 * 1024), 3),
    }


if not V2_DIR.exists():
    raise FileNotFoundError(f"Final Package v2 directory not found: {V2_DIR}")

if PKG_DIR.exists():
    shutil.rmtree(PKG_DIR)

shutil.copytree(V2_DIR, PKG_DIR)

added = []

# ------------------------------------------------------------------
# 1. Updated presentation
# ------------------------------------------------------------------
presentation_pdf = PRES / "Week6_Unibo_Dataset_Validation_Presentation_v2_with_Segment_Anything_Model.pdf"
added.append(copy_one(
    presentation_pdf,
    Path("presentation") / presentation_pdf.name,
))

presentation_note = NOTES / "week6_presentation_v2_with_segment_anything_model_notes.md"
presentation_summary = STATS / "week6_presentation_v2_generation_summary.csv"

for p in [presentation_note, presentation_summary]:
    if p.exists():
        folder = "00_report_notes" if p.suffix == ".md" else "01_dataset_statistics"
        added.append(copy_one(p, Path(folder) / p.name))


# ------------------------------------------------------------------
# 2. Segment Anything Model outputs
# ------------------------------------------------------------------
sam_note_files = [
    NOTES / "week6_sam_box_prompt_smoke_test_notes.md",
    NOTES / "week6_sam_box_prompt_full_segmentation_notes.md",
]

sam_stat_files = [
    STATS / "week6_sam_box_prompt_smoke_test_summary.csv",
    STATS / "week6_sam_box_prompt_smoke_test_results.csv",
    STATS / "week6_sam_box_prompt_segmentation_quality_summary.csv",
    STATS / "week6_sam_box_prompt_segmentation_status_summary.csv",
    STATS / "week6_sam_box_prompt_segmentation_recovery_summary.csv",
    STATS / "week6_sam_vs_grabcut_otsu_segmentation_comparison.csv",
    STATS / "week6_sam_vs_grabcut_otsu_segmentation_comparison_detailed.csv",
]

sam_feature_files = [
    FEAT / "week6_sam_box_prompt_segmentation_features.csv",
    FEAT / "week6_sam_box_prompt_segmentation_features.json",
    FEAT / "week6_sam_box_prompt_segmentation_frame_summary.csv",
]

sam_visual_files = [
    VIS / "week6_sam_box_prompt_segmentation_contact_sheet.jpg",
]

for p in sam_note_files:
    added.append(copy_one(p, Path("05_advanced_segmentation_segment_anything_model") / "notes" / p.name))

for p in sam_stat_files:
    added.append(copy_one(p, Path("05_advanced_segmentation_segment_anything_model") / "statistics" / p.name))

for p in sam_feature_files:
    added.append(copy_one(p, Path("05_advanced_segmentation_segment_anything_model") / "feature_extractors" / p.name))

for p in sam_visual_files:
    added.append(copy_one(p, Path("05_advanced_segmentation_segment_anything_model") / "visualizations" / p.name))


# ------------------------------------------------------------------
# 3. Segment Anything Model masks and overlays
# ------------------------------------------------------------------
sam_mask_dir = VIS / "sam_box_prompt_full_segmentation" / "masks"
sam_overlay_dir = VIS / "sam_box_prompt_full_segmentation" / "frame_overlays"

if sam_mask_dir.exists():
    for p in sorted(sam_mask_dir.glob("*.png")):
        added.append(copy_one(
            p,
            Path("05_advanced_segmentation_segment_anything_model") / "masks" / p.name,
        ))

if sam_overlay_dir.exists():
    for p in sorted(sam_overlay_dir.glob("*.jpg")):
        added.append(copy_one(
            p,
            Path("05_advanced_segmentation_segment_anything_model") / "frame_overlays" / p.name,
        ))


# ------------------------------------------------------------------
# 4. New scripts
# ------------------------------------------------------------------
script_names = [
    "49_interface_dependency_free_smoke_test.py",
    "50_sam_box_prompt_smoke_test.py",
    "51_sam_box_prompt_full_segmentation.py",
    "52_finalize_sam_box_prompt_outputs_after_comparison_error.py",
    "53_build_updated_presentation_pdf_v2_with_sam.py",
    "54_build_final_delivery_package_v3_with_presentation_and_sam.py",
]

for name in script_names:
    p = SCRIPTS / name
    if p.exists():
        added.append(copy_one(p, Path("scripts") / name))


# ------------------------------------------------------------------
# 5. Raw data note
# ------------------------------------------------------------------
raw_note_path = PKG_DIR / "raw_data_location.txt"
raw_note_path.write_text(
    "Raw Unibo videos are not included in this delivery package.\n\n"
    "Reason:\n"
    "The raw videos were already uploaded by the instructor on the server. "
    "They should not be duplicated inside this derived-output package.\n\n"
    "Server location:\n"
    "/work/pig/datasets/Unibo\n\n"
    "This package contains derived outputs, including unified ground truth files, "
    "scanpoint frames, detector outputs, feature tables, embeddings, segmentation outputs, "
    "visual quality control artifacts, scripts, notes, shared tracker, and presentation files.\n"
)


# ------------------------------------------------------------------
# 6. Interface usage note
# ------------------------------------------------------------------
interface_note_path = PKG_DIR / "interface_usage_notes.md"
interface_note_path.write_text(
    "# Interface Usage Notes\n\n"
    "The package includes a dependency-free static visualization viewer.\n\n"
    "From the Week 6 project folder on the server:\n\n"
    "```bash\n"
    "cd ~/PigBench/Week6_Unibo_Dataset_Validation\n"
    "python -m http.server 8506\n"
    "```\n\n"
    "Then open:\n\n"
    "```text\n"
    "http://localhost:8506/interface_demo/week6_static_visualization_viewer.html\n"
    "```\n\n"
    "If using Visual Studio Code Remote, forward port 8506 from the Ports panel.\n\n"
    "To stop the server, press Control + C in the terminal.\n"
    "Do not use Control + Z because it only suspends the process.\n"
)


# ------------------------------------------------------------------
# 7. Improved README
# ------------------------------------------------------------------
readme_path = PKG_DIR / "README.md"
readme_path.write_text(
    "# Week 6 Unibo Dataset Validation Final Package v3\n\n"
    f"Generated: `{datetime.now().isoformat(timespec='seconds')}`\n\n"
    "## Start here\n\n"
    "Open these files first:\n\n"
    "1. `presentation/Week6_Unibo_Dataset_Validation_Presentation_v2_with_Segment_Anything_Model.pdf`\n"
    "2. `00_report_notes/week6_final_report_v2.md`\n"
    "3. `00_report_notes/week6_final_executive_summary_v2.md`\n"
    "4. `package_manifest_v3.csv`\n\n"
    "## What is inside this package?\n\n"
    "This package contains the final derived outputs for Week 6:\n\n"
    "- Unified ground truth tables in Comma-Separated Values format and JavaScript Object Notation format.\n"
    "- Scanpoint frame index and frame-label alignment tables.\n"
    "- Recommended training, validation, and test split protocol.\n"
    "- Detection outputs and detector quality control flags.\n"
    "- Feature extractor outputs.\n"
    "- Detector-backed learned crop embeddings.\n"
    "- Classical GrabCut and Otsu segmentation baseline outputs.\n"
    "- Segment Anything Model box-prompt segmentation outputs.\n"
    "- Visualization interface demo files.\n"
    "- Contact sheets and overlay images.\n"
    "- Shared Excel task tracker.\n"
    "- Final presentation.\n"
    "- Scripts used for reproducibility.\n\n"
    "## Key counts\n\n"
    "- Manual behaviour labels: 432\n"
    "- Scanpoint frames: 72\n"
    "- Detector bounding boxes: 540\n"
    "- Detector-backed embedding rows: 540\n"
    "- Detector-backed embedding dimension: 896\n"
    "- Classical segmentation feature rows: 540\n"
    "- Segment Anything Model masks: 540\n"
    "- Segment Anything Model frame overlays: 72\n"
    "- Segment Anything Model failed rows: 0\n"
    "- Recommended split: 288 training labels, 72 validation labels, 72 test labels\n\n"
    "## Raw data\n\n"
    "Raw videos are not included in this package because they already exist on the server.\n\n"
    "Server location:\n\n"
    "```text\n"
    "/work/pig/datasets/Unibo\n"
    "```\n\n"
    "See also:\n\n"
    "```text\n"
    "raw_data_location.txt\n"
    "```\n\n"
    "## Visualization interface\n\n"
    "See:\n\n"
    "```text\n"
    "interface_usage_notes.md\n"
    "```\n\n"
    "## Important interpretation\n\n"
    "Manual labels, automatic detector bounding boxes, automatic segmentation masks, and candidate colour identity evidence are kept separate.\n\n"
    "The Segment Anything Model masks are automatic model outputs, not manual segmentation ground truth.\n"
)


# ------------------------------------------------------------------
# 8. Reports, manifest, checksums
# ------------------------------------------------------------------
added_df = pd.DataFrame(added)
safe_to_csv(added_df, PKG_DIR / "package_v3_added_files_report.csv")

manifest_rows = []
for p in sorted(PKG_DIR.rglob("*")):
    if p.is_file():
        manifest_rows.append({
            "relative_path": str(p.relative_to(PKG_DIR)),
            "size_bytes": p.stat().st_size,
            "size_mb": round(p.stat().st_size / (1024 * 1024), 3),
        })

manifest = pd.DataFrame(manifest_rows)
safe_to_csv(manifest, PKG_DIR / "package_manifest_v3.csv")

checksum_rows = []
for p in sorted(PKG_DIR.rglob("*")):
    if p.is_file():
        h = hashlib.sha256()
        with open(p, "rb") as f:
            for chunk in iter(lambda: f.read(1024 * 1024), b""):
                h.update(chunk)
        checksum_rows.append({
            "relative_path": str(p.relative_to(PKG_DIR)),
            "sha256": h.hexdigest(),
            "size_bytes": p.stat().st_size,
        })

checksums = pd.DataFrame(checksum_rows)
safe_to_csv(checksums, PKG_DIR / "checksums_sha256_v3.csv")

with open(PKG_DIR / "checksums_sha256_v3.txt", "w") as f:
    for _, r in checksums.iterrows():
        f.write(f"{r['sha256']}  {r['relative_path']}\n")

summary = pd.DataFrame([
    {"metric": "package_name", "value": PKG_NAME},
    {"metric": "generated_at", "value": datetime.now().isoformat(timespec="seconds")},
    {"metric": "package_dir", "value": str(PKG_DIR)},
    {"metric": "zip_path", "value": str(ZIP_PATH)},
    {"metric": "manifest_file_count", "value": len(manifest)},
    {"metric": "added_file_rows", "value": len(added_df)},
    {
        "metric": "missing_added_files",
        "value": int((added_df["status"] == "missing").sum()) if len(added_df) else 0,
    },
    {
        "metric": "sam_mask_files_in_package",
        "value": len(list((PKG_DIR / "05_advanced_segmentation_segment_anything_model" / "masks").glob("*.png"))),
    },
    {
        "metric": "sam_overlay_files_in_package",
        "value": len(list((PKG_DIR / "05_advanced_segmentation_segment_anything_model" / "frame_overlays").glob("*.jpg"))),
    },
])
safe_to_csv(summary, PKG_DIR / "package_build_summary_v3.csv")

validation_path = PKG_DIR / "Week6_Final_Package_v3_Validation.md"
validation_path.write_text(
    "# Week 6 Final Package v3 Validation\n\n"
    "## Summary\n\n"
    + summary.to_markdown(index=False)
    + "\n\n"
    "## Added files status\n\n"
    + (added_df["status"].value_counts(dropna=False).to_frame("count").to_markdown() if len(added_df) else "No added files recorded.")
    + "\n\n"
    "## Decision\n\n"
    "Final Package v3 includes Final Package v2 contents plus updated presentation, raw data location note, "
    "interface usage note, improved README, Segment Anything Model smoke test outputs, full Segment Anything Model "
    "segmentation outputs, mask images, overlay images, and comparison against the classical GrabCut and Otsu baseline.\n"
)


# ------------------------------------------------------------------
# 9. Zip and zip test
# ------------------------------------------------------------------
if ZIP_PATH.exists():
    ZIP_PATH.unlink()

with zipfile.ZipFile(ZIP_PATH, "w", compression=zipfile.ZIP_DEFLATED) as zf:
    for p in sorted(PKG_DIR.rglob("*")):
        if p.is_file():
            zf.write(p, arcname=str(Path(PKG_NAME) / p.relative_to(PKG_DIR)))

with zipfile.ZipFile(ZIP_PATH, "r") as zf:
    bad_file = zf.testzip()

zip_hash = hashlib.sha256()
with open(ZIP_PATH, "rb") as f:
    for chunk in iter(lambda: f.read(1024 * 1024), b""):
        zip_hash.update(chunk)

zip_info = pd.DataFrame([
    {
        "zip_path": str(ZIP_PATH),
        "zip_size_mb": round(ZIP_PATH.stat().st_size / (1024 * 1024), 3),
        "zip_test_result": "OK" if bad_file is None else f"BAD:{bad_file}",
        "sha256": zip_hash.hexdigest(),
    }
])

safe_to_csv(zip_info, PKG_ROOT / "Week6_Unibo_Dataset_Validation_Final_Package_v3_zip_info.csv")

print("Saved package directory:")
print(PKG_DIR)
print()
print("Saved zip:")
print(ZIP_PATH)
print()
print("=== Package summary ===")
print(summary.to_string(index=False))
print()
print("=== Added file status ===")
print(added_df["status"].value_counts(dropna=False).to_string() if len(added_df) else "No added files.")
print()
print("=== Zip info ===")
print(zip_info.to_string(index=False))
