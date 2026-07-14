from pathlib import Path
import shutil
import hashlib
import zipfile
import csv
import pandas as pd
from datetime import datetime


ROOT = Path.home() / "PigBench"
W6 = ROOT / "Week6_Unibo_Dataset_Validation"

OUT = W6 / "outputs"
STATS = OUT / "dataset_statistics"
GT = OUT / "unified_ground_truth"
FEAT = OUT / "feature_extractors"
VIS = OUT / "visual_label_check"
NOTES = W6 / "notes"

PACKAGE_ROOT = W6 / "final_delivery_package"
PACKAGE_DIR = PACKAGE_ROOT / "Week6_Unibo_Dataset_Validation_Final_Package"
ZIP_PATH = PACKAGE_ROOT / "Week6_Unibo_Dataset_Validation_Final_Package.zip"

PACKAGE_ROOT.mkdir(parents=True, exist_ok=True)

if PACKAGE_DIR.exists():
    shutil.rmtree(PACKAGE_DIR)

PACKAGE_DIR.mkdir(parents=True, exist_ok=True)


def sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def copy_item(src, dst_rel):
    src = Path(src)
    dst = PACKAGE_DIR / dst_rel
    dst.parent.mkdir(parents=True, exist_ok=True)

    if not src.exists():
        return {
            "status": "missing",
            "source": str(src),
            "package_path": str(dst_rel),
            "size_mb": "",
            "sha256": "",
        }

    shutil.copy2(src, dst)

    return {
        "status": "copied",
        "source": str(src),
        "package_path": str(dst_rel),
        "size_mb": round(dst.stat().st_size / (1024 * 1024), 3),
        "sha256": sha256_file(dst),
    }


items = [
    # Report notes
    (NOTES / "week6_final_executive_summary.md", "00_report_notes/week6_final_executive_summary.md"),
    (NOTES / "week6_final_report_outline.md", "00_report_notes/week6_final_report_outline.md"),
    (NOTES / "week6_feature_extractor_comparison_final_clean_v3_notes.md", "00_report_notes/week6_feature_extractor_comparison_final_clean_v3_notes.md"),
    (NOTES / "week6_postfix_final_consistency_audit_notes.md", "00_report_notes/week6_postfix_final_consistency_audit_notes.md"),
    (NOTES / "week6_recommended_split_v2_notes.md", "00_report_notes/week6_recommended_split_v2_notes.md"),

    # Ground truth
    (GT / "week6_unified_ground_truth_v2_with_recommended_split_v2.csv", "01_ground_truth/week6_unified_ground_truth_v2_with_recommended_split_v2.csv"),
    (GT / "week6_scanpoint_frame_index.csv", "01_ground_truth/week6_scanpoint_frame_index.csv"),
    (GT / "week6_scanpoint_frame_labels_long.csv", "01_ground_truth/week6_scanpoint_frame_labels_long.csv"),

    # Statistics and split
    (STATS / "week6_final_deliverable_inventory.csv", "02_statistics_split_qc/week6_final_deliverable_inventory.csv"),
    (STATS / "week6_final_dataset_overview_metrics.csv", "02_statistics_split_qc/week6_final_dataset_overview_metrics.csv"),
    (STATS / "week6_final_behaviour_distribution.csv", "02_statistics_split_qc/week6_final_behaviour_distribution.csv"),
    (STATS / "week6_final_rare_behaviour_classes.csv", "02_statistics_split_qc/week6_final_rare_behaviour_classes.csv"),
    (STATS / "week6_recommended_split_v2_summary.csv", "02_statistics_split_qc/week6_recommended_split_v2_summary.csv"),
    (STATS / "week6_recommended_split_v2_video_level.csv", "02_statistics_split_qc/week6_recommended_split_v2_video_level.csv"),
    (STATS / "week6_recommended_split_v2_behaviour_coverage.csv", "02_statistics_split_qc/week6_recommended_split_v2_behaviour_coverage.csv"),
    (STATS / "week6_postfix_final_consistency_audit.csv", "02_statistics_split_qc/week6_postfix_final_consistency_audit.csv"),
    (STATS / "week6_postfix_final_split_summary.csv", "02_statistics_split_qc/week6_postfix_final_split_summary.csv"),
    (STATS / "week6_postfix_final_detection_summary.csv", "02_statistics_split_qc/week6_postfix_final_detection_summary.csv"),
    (STATS / "week6_detection_qc_conservative_subset_summary.csv", "02_statistics_split_qc/week6_detection_qc_conservative_subset_summary.csv"),

    # Feature extractors
    (FEAT / "week6_yolov8s_all_scanpoint_detections_with_qc_flags.csv", "03_feature_extractors/week6_yolov8s_all_scanpoint_detections_with_qc_flags.csv"),
    (FEAT / "week6_yolov8s_conservative_detections_score_ge_0_50.csv", "03_feature_extractors/week6_yolov8s_conservative_detections_score_ge_0_50.csv"),
    (FEAT / "week6_crop_colour_marker_features.csv", "03_feature_extractors/week6_crop_colour_marker_features.csv"),
    (FEAT / "week6_candidate_bbox_to_colour_assignments.csv", "03_feature_extractors/week6_candidate_bbox_to_colour_assignments.csv"),
    (FEAT / "week6_feature_extractor_comparison_final_clean_v3.csv", "03_feature_extractors/week6_feature_extractor_comparison_final_clean_v3.csv"),
    (FEAT / "week6_feature_extractor_recommendations_final_clean_v3.csv", "03_feature_extractors/week6_feature_extractor_recommendations_final_clean_v3.csv"),

    # Visual deliverables
    (VIS / "marker_bbox_overlay_v3_video/week6_marker_bbox_overlay_v3_slideshow.mp4", "04_visualizations/week6_marker_bbox_overlay_v3_slideshow.mp4"),
    (VIS / "bbox_count_warning_qc/all_bbox_count_warning_frames_contact_sheet.jpg", "04_visualizations/all_bbox_count_warning_frames_contact_sheet.jpg"),
    (VIS / "bbox_count_warning_qc/low_bbox_count_frames_contact_sheet.jpg", "04_visualizations/low_bbox_count_frames_contact_sheet.jpg"),
    (VIS / "bbox_count_warning_qc/high_bbox_count_frames_contact_sheet.jpg", "04_visualizations/high_bbox_count_frames_contact_sheet.jpg"),
]

manifest_rows = []

for src, dst_rel in items:
    manifest_rows.append(copy_item(src, dst_rel))

manifest = pd.DataFrame(manifest_rows)
manifest_path = PACKAGE_DIR / "MANIFEST.csv"
manifest.to_csv(
    manifest_path,
    index=False,
    quoting=csv.QUOTE_ALL,
    escapechar="\\",
    lineterminator="\n"
)

checksums_path = PACKAGE_DIR / "CHECKSUMS_SHA256.txt"

with open(checksums_path, "w") as f:
    for _, row in manifest.iterrows():
        if row["status"] == "copied":
            f.write(f"{row['sha256']}  {row['package_path']}\n")

readme_path = PACKAGE_DIR / "README.md"

missing_count = int((manifest["status"] == "missing").sum())

with open(readme_path, "w") as f:
    f.write("# Week 6 Unibo Dataset Validation Final Package\n\n")
    f.write(f"Generated: {datetime.now().isoformat(timespec='seconds')}\n\n")

    f.write("## Contents\n\n")
    f.write("- `00_report_notes/`: executive summary, report outline, audit and split notes.\n")
    f.write("- `01_ground_truth/`: final recommended GT table and scanpoint frame/label alignment tables.\n")
    f.write("- `02_statistics_split_qc/`: dataset statistics, recommended split v2, final audit, detector QC summaries.\n")
    f.write("- `03_feature_extractors/`: YOLOv8-s bbox features, conservative subset, colour-marker features, feature extractor comparison.\n")
    f.write("- `04_visualizations/`: marker-bbox demo slideshow and bbox warning QC contact sheets.\n")
    f.write("- `MANIFEST.csv`: package file list with copy status and checksums.\n")
    f.write("- `CHECKSUMS_SHA256.txt`: SHA256 checksums for copied files.\n\n")

    f.write("## Key result summary\n\n")
    f.write("- Final manual labels: 432\n")
    f.write("- Scanpoint frames: 72\n")
    f.write("- Manual labels per frame: 6\n")
    f.write("- YOLOv8-s primary bboxes: 540\n")
    f.write("- Medium/high marker candidates: 196\n")
    f.write("- Recommended split v2: train 288, val 72, test 72 labels\n")
    f.write("- Final post-fix audit: 0 failed checks\n\n")

    f.write("## Important limitations\n\n")
    f.write("- Labels come from one Excel sheet/day/camera-pen context.\n")
    f.write("- 15:00-19:00 c-token video mappings are candidate recovered mappings and should remain medium confidence.\n")
    f.write("- Bboxes are automatic detector outputs, not manual bbox annotations.\n")
    f.write("- Bbox-to-colour identity is candidate-level only; red markers are ambiguous and no_color cannot be marker-detected.\n")
    f.write("- Rare behaviour classes remain limited.\n\n")

    f.write("## Missing files\n\n")
    if missing_count == 0:
        f.write("No missing files in selected final package.\n")
    else:
        f.write(f"{missing_count} selected package files were missing. See MANIFEST.csv.\n")

# Zip package.
if ZIP_PATH.exists():
    ZIP_PATH.unlink()

with zipfile.ZipFile(ZIP_PATH, "w", compression=zipfile.ZIP_DEFLATED) as zf:
    for path in sorted(PACKAGE_DIR.rglob("*")):
        if path.is_file():
            zf.write(path, path.relative_to(PACKAGE_ROOT))

package_summary = pd.DataFrame([
    {
        "package_dir": str(PACKAGE_DIR),
        "zip_path": str(ZIP_PATH),
        "zip_size_mb": round(ZIP_PATH.stat().st_size / (1024 * 1024), 3),
        "selected_files": len(manifest),
        "missing_files": missing_count,
    }
])

summary_path = PACKAGE_ROOT / "final_package_summary.csv"
package_summary.to_csv(summary_path, index=False)

print("Saved package:")
print(PACKAGE_DIR)
print(ZIP_PATH)
print(summary_path)
print()
print("=== Package summary ===")
print(package_summary.to_string(index=False))
print()
print("=== Missing files ===")
print(manifest[manifest["status"] == "missing"].to_string(index=False) if missing_count else "None")
