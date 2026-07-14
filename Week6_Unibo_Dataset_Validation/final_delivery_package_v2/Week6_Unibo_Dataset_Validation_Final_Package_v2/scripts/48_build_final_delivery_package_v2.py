from pathlib import Path
import csv
import hashlib
import shutil
import zipfile
import pandas as pd
from datetime import datetime


ROOT = Path.home() / "PigBench"
W6 = ROOT / "Week6_Unibo_Dataset_Validation"

PKG_ROOT = W6 / "final_delivery_package_v2"
PKG_NAME = "Week6_Unibo_Dataset_Validation_Final_Package_v2"
PKG_DIR = PKG_ROOT / PKG_NAME
ZIP_PATH = PKG_ROOT / f"{PKG_NAME}.zip"

STATS = W6 / "outputs/dataset_statistics"
GT = W6 / "outputs/unified_ground_truth"
FEAT = W6 / "outputs/feature_extractors"
VIS = W6 / "outputs/visual_label_check"
NOTES = W6 / "notes"
INTERFACE = W6 / "interface_demo"
TRACKER = W6 / "shared_tracker"
SCRIPTS = W6 / "scripts"

PKG_ROOT.mkdir(parents=True, exist_ok=True)

if PKG_DIR.exists():
    shutil.rmtree(PKG_DIR)

PKG_DIR.mkdir(parents=True, exist_ok=True)


def safe_to_csv(df, path):
    df.to_csv(
        path,
        index=False,
        quoting=csv.QUOTE_ALL,
        escapechar="\\",
        lineterminator="\n"
    )


def rel_to_w6(path):
    return Path(path).relative_to(W6)


def copy_file(src, dst_rel=None):
    src = Path(src)

    if not src.exists():
        return {
            "source": str(src),
            "package_path": "",
            "status": "missing",
            "size_mb": "",
        }

    if dst_rel is None:
        dst_rel = rel_to_w6(src)

    dst = PKG_DIR / dst_rel
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dst)

    return {
        "source": str(src),
        "package_path": str(dst.relative_to(PKG_DIR)),
        "status": "copied",
        "size_mb": round(dst.stat().st_size / (1024 * 1024), 3),
    }


def copy_glob(base, pattern, dst_prefix=None):
    rows = []

    base = Path(base)

    for src in sorted(base.glob(pattern)):
        if src.is_file():
            if dst_prefix is None:
                dst_rel = rel_to_w6(src)
            else:
                dst_rel = Path(dst_prefix) / src.name

            rows.append(copy_file(src, dst_rel))

    return rows


manifest_rows = []

# ------------------------------------------------------------
# 1) Documentation / notes
# ------------------------------------------------------------
note_files = [
    NOTES / "week6_final_report_v2.md",
    NOTES / "week6_final_executive_summary_v2.md",
    NOTES / "week6_final_package_v2_readme_draft.md",
    NOTES / "week6_final_task_sheet_compliance_audit_v2_notes.md",
    NOTES / "week6_quality_hardening_audit_v2_notes.md",
    NOTES / "week6_detector_backbone_crop_embeddings_notes.md",
    NOTES / "week6_segmentation_visual_qc_manual_review_notes.md",
    NOTES / "week6_segmentation_visual_qc_package_notes.md",
    NOTES / "week6_preliminary_segmentation_baseline_notes.md",
    NOTES / "week6_shared_excel_task_tracker_notes.md",
    NOTES / "week6_learned_embedding_route_smoke_test_notes.md",
    NOTES / "week6_model_weight_capability_audit_notes.md",
]

for p in note_files:
    manifest_rows.append(copy_file(p))

# README.md from draft.
readme_src = NOTES / "week6_final_package_v2_readme_draft.md"
readme_dst = PKG_DIR / "README.md"

if readme_src.exists():
    shutil.copy2(readme_src, readme_dst)
    manifest_rows.append({
        "source": str(readme_src),
        "package_path": "README.md",
        "status": "copied",
        "size_mb": round(readme_dst.stat().st_size / (1024 * 1024), 3),
    })

# ------------------------------------------------------------
# 2) Unified GT and metadata
# ------------------------------------------------------------
gt_files = [
    GT / "week6_unified_ground_truth_v2_with_recommended_split_v2.csv",
    GT / "week6_unified_ground_truth_v2_with_recommended_split_v2.json",
    GT / "week6_unified_ground_truth_v2_with_recommended_split_v2_schema.json",
    GT / "week6_scanpoint_annotations_nested_for_viewer.json",
    GT / "week6_scanpoint_frame_index.csv",
    GT / "week6_scanpoint_frame_labels_long.csv",
]

for p in gt_files:
    manifest_rows.append(copy_file(p))

# ------------------------------------------------------------
# 3) Dataset statistics, compliance, QC, split summaries
# ------------------------------------------------------------
stats_patterns = [
    "week6_final_*",
    "week6_recommended_split_v2_*",
    "week6_postfix_final_consistency_audit.csv",
    "week6_task_sheet_*v2.csv",
    "week6_quality_hardening_*v2.csv",
    "week6_quality_hardening_v2_text_cleanup_report.csv",
    "week6_note_typo_bad_term_hits_v2.csv",
    "week6_segmentation_visual_qc_*.csv",
    "week6_preliminary_segmentation_*.csv",
    "week6_segmentation_resource_audit.csv",
    "week6_detector_backbone_crop_embedding_*.csv",
    "week6_model_*",
    "week6_local_model_weight_inventory.csv",
    "week6_shared_tracker_*.csv",
    "week6_lightweight_feature_extractor_test_summary.csv",
    "week6_camera_pen_crate_video_summary.csv",
]

for pattern in stats_patterns:
    manifest_rows.extend(copy_glob(STATS, pattern))

# ------------------------------------------------------------
# 4) Feature extractor outputs
# ------------------------------------------------------------
feature_files = [
    FEAT / "week6_yolov8s_all_scanpoint_detections_with_qc_flags.csv",
    FEAT / "week6_yolov8s_conservative_detections_score_ge_0_50.csv",
    FEAT / "week6_yolov8s_conservative_detections_score_ge_0_70.csv",
    FEAT / "week6_crop_colour_marker_features.csv",
    FEAT / "week6_candidate_bbox_to_colour_assignments.csv",
    FEAT / "week6_feature_extractor_comparison_final_clean_v3.csv",
    FEAT / "week6_feature_extractor_recommendations_final_clean_v3.csv",
    FEAT / "week6_bbox_geometry_features.csv",
    FEAT / "week6_group_spatial_features_per_frame.csv",
    FEAT / "week6_coarse_roi_resource_proxy_features.csv",
    FEAT / "week6_coarse_roi_resource_proxy_features_per_frame.csv",
    FEAT / "week6_crop_descriptor_baseline_features.csv",
    FEAT / "week6_crop_descriptor_plus_marker_features.csv",
    FEAT / "week6_trajectory_feature_feasibility_report.csv",
    FEAT / "week6_detector_backbone_crop_embeddings_896.csv",
    FEAT / "week6_detector_backbone_crop_embeddings_896.npy",
    FEAT / "week6_detector_backbone_crop_embedding_metadata.csv",
    FEAT / "week6_detector_backbone_crop_embedding_pca_features.csv",
    FEAT / "week6_detector_backbone_crop_embedding_smoke_test_v2_preview.csv",
    FEAT / "week6_preliminary_bbox_guided_segmentation_features.csv",
    FEAT / "week6_preliminary_bbox_guided_segmentation_features.json",
    FEAT / "week6_preliminary_segmentation_frame_summary.csv",
]

for p in feature_files:
    manifest_rows.append(copy_file(p))

# ------------------------------------------------------------
# 5) Interface demo
# ------------------------------------------------------------
if INTERFACE.exists():
    for src in sorted(INTERFACE.glob("*")):
        if src.is_file():
            manifest_rows.append(copy_file(src))

# ------------------------------------------------------------
# 6) Shared tracker
# ------------------------------------------------------------
if TRACKER.exists():
    for src in sorted(TRACKER.glob("*")):
        if src.is_file():
            manifest_rows.append(copy_file(src))

# ------------------------------------------------------------
# 7) Visual QC artifacts
# Include contact sheets, HTML reviewers, slideshow videos, and selected overlay dirs.
# ------------------------------------------------------------
visual_files = [
    VIS / "week6_preliminary_segmentation_contact_sheet.jpg",
    VIS / "segmentation_visual_qc/segmentation_visual_qc_high_risk_contact_sheet.jpg",
    VIS / "segmentation_visual_qc/segmentation_visual_qc_manual_review_sample_contact_sheet.jpg",
    VIS / "segmentation_visual_qc/segmentation_visual_qc_representative_contact_sheet.jpg",
    VIS / "segmentation_visual_qc/segmentation_visual_qc_static_reviewer.html",
    VIS / "bbox_count_warning_qc/all_bbox_count_warning_frames_contact_sheet.jpg",
    VIS / "marker_bbox_overlay_v3_video/week6_marker_bbox_overlay_v3_slideshow.mp4",
]

for p in visual_files:
    manifest_rows.append(copy_file(p))

# Include all segmentation overlay images because they are final QC artifacts.
seg_overlay_dir = VIS / "preliminary_segmentation_baseline"

if seg_overlay_dir.exists():
    for src in sorted(seg_overlay_dir.glob("*_preliminary_segmentation_overlay.jpg")):
        manifest_rows.append(copy_file(src))

# Include all scanpoint frames only as compact derived frames, not raw videos.
scanpoint_dir = VIS / "all_scanpoint_frames"

if scanpoint_dir.exists():
    for src in sorted(scanpoint_dir.glob("*.jpg")):
        manifest_rows.append(copy_file(src))

# ------------------------------------------------------------
# 8) Scripts for reproducibility
# Include Week 6 scripts used after quality hardening and finalization.
# ------------------------------------------------------------
script_patterns = [
    "31_task_sheet_compliance_audit.py",
    "32_generate_recommended_gt_json_and_metadata_summary.py",
    "33_create_minimal_visualization_interface_demo.py",
    "34_build_lightweight_feature_extractor_tests.py",
    "35_build_preliminary_bbox_guided_segmentation_baseline.py",
    "36_generate_week6_shared_excel_task_tracker.py",
    "37_final_task_sheet_compliance_audit_v2.py",
    "38_quality_hardening_audit.py",
    "39_model_weight_capability_audit.py",
    "40_learned_embedding_route_smoke_test.py",
    "41_diagnose_crop_loading_for_embeddings.py",
    "42_detector_backbone_crop_embedding_smoke_test_v2.py",
    "43_extract_full_detector_backbone_crop_embeddings.py",
    "44_build_segmentation_visual_qc_package.py",
    "45_record_segmentation_manual_visual_qc_result.py",
    "46_quality_hardening_audit_v2_after_fixes.py",
    "47_build_final_report_notes_v2.py",
    "48_build_final_delivery_package_v2.py",
]

for name in script_patterns:
    manifest_rows.append(copy_file(SCRIPTS / name))

# ------------------------------------------------------------
# 9) Manifest and checksums
# ------------------------------------------------------------
manifest = pd.DataFrame(manifest_rows)
manifest = manifest.drop_duplicates(subset=["package_path", "source"], keep="first")

manifest_path = PKG_DIR / "package_manifest_v2.csv"
safe_to_csv(manifest, manifest_path)

# Checksum files inside package.
checksum_rows = []

for path in sorted(PKG_DIR.rglob("*")):
    if path.is_file():
        h = hashlib.sha256()
        with open(path, "rb") as f:
            for chunk in iter(lambda: f.read(1024 * 1024), b""):
                h.update(chunk)

        checksum_rows.append({
            "relative_path": str(path.relative_to(PKG_DIR)),
            "sha256": h.hexdigest(),
            "size_bytes": path.stat().st_size,
            "size_mb": round(path.stat().st_size / (1024 * 1024), 3),
        })

checksums = pd.DataFrame(checksum_rows)
checksums_path = PKG_DIR / "checksums_sha256_v2.csv"
safe_to_csv(checksums, checksums_path)

# Also simple sha256sum-compatible text.
sha_txt_path = PKG_DIR / "checksums_sha256_v2.txt"

with open(sha_txt_path, "w") as f:
    for _, r in checksums.iterrows():
        f.write(f"{r['sha256']}  {r['relative_path']}\n")

# Add package build summary.
copied = manifest[manifest["status"] == "copied"]
missing = manifest[manifest["status"] != "copied"]

summary = pd.DataFrame([
    {
        "metric": "package_name",
        "value": PKG_NAME,
    },
    {
        "metric": "generated_at",
        "value": datetime.now().isoformat(timespec="seconds"),
    },
    {
        "metric": "copied_files",
        "value": len(copied),
    },
    {
        "metric": "missing_files",
        "value": len(missing),
    },
    {
        "metric": "package_dir",
        "value": str(PKG_DIR),
    },
    {
        "metric": "zip_path",
        "value": str(ZIP_PATH),
    },
])

summary_path = PKG_DIR / "package_build_summary_v2.csv"
safe_to_csv(summary, summary_path)

# ------------------------------------------------------------
# 10) Zip package
# ------------------------------------------------------------
if ZIP_PATH.exists():
    ZIP_PATH.unlink()

with zipfile.ZipFile(ZIP_PATH, "w", compression=zipfile.ZIP_DEFLATED) as zf:
    for path in sorted(PKG_DIR.rglob("*")):
        if path.is_file():
            zf.write(path, arcname=str(Path(PKG_NAME) / path.relative_to(PKG_DIR)))

# Zip checksum outside.
zip_sha = hashlib.sha256()

with open(ZIP_PATH, "rb") as f:
    for chunk in iter(lambda: f.read(1024 * 1024), b""):
        zip_sha.update(chunk)

zip_info = pd.DataFrame([
    {
        "zip_path": str(ZIP_PATH),
        "zip_size_mb": round(ZIP_PATH.stat().st_size / (1024 * 1024), 3),
        "sha256": zip_sha.hexdigest(),
    }
])

zip_info_path = PKG_ROOT / "Week6_Unibo_Dataset_Validation_Final_Package_v2_zip_info.csv"
safe_to_csv(zip_info, zip_info_path)

print("Saved package dir:")
print(PKG_DIR)

print()
print("Saved zip:")
print(ZIP_PATH)

print()
print("=== Package summary ===")
print(summary.to_string(index=False))

print()
print("=== Missing files ===")
if len(missing):
    print(missing.to_string(index=False))
else:
    print("None")

print()
print("=== Zip info ===")
print(zip_info.to_string(index=False))
