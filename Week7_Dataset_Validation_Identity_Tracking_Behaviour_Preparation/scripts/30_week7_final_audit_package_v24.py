from pathlib import Path
from datetime import datetime
import csv
import hashlib
import shutil
import zipfile
import pandas as pd


ROOT = Path.home() / "PigBench"
W7 = ROOT / "Week7_Dataset_Validation_Identity_Tracking_Behaviour_Preparation"

OUT_ROOT = W7 / "outputs" / "final_audit_package_v24"
PACKAGE_ROOT = OUT_ROOT / "Week7_Final_Audit_Package_v24"
ZIP_PATH = OUT_ROOT / "Week7_Final_Audit_Package_v24.zip"

OUT_ROOT.mkdir(parents=True, exist_ok=True)

if PACKAGE_ROOT.exists():
    shutil.rmtree(PACKAGE_ROOT)
PACKAGE_ROOT.mkdir(parents=True, exist_ok=True)

MANIFEST = PACKAGE_ROOT / "manifest_v24.csv"
README = PACKAGE_ROOT / "README_Week7_Final_Audit_Package_v24.md"
SUMMARY = PACKAGE_ROOT / "package_summary_v24.csv"
NOTE = W7 / "notes" / "week7_final_audit_package_v24_notes.md"

ITEMS = [
    # Notes
    ("notes/week7_final_colour_identity_v17_fixed_notes.md", W7 / "notes" / "week7_final_colour_identity_v17_fixed_notes.md"),
    ("notes/week7_behaviour_label_fusion_v18c_verified_crosswalk_notes.md", W7 / "notes" / "week7_behaviour_label_fusion_v18c_verified_crosswalk_notes.md"),
    ("notes/week7_final_fusion_qa_dataset_stats_v19_notes.md", W7 / "notes" / "week7_final_fusion_qa_dataset_stats_v19_notes.md"),
    ("notes/week7_primary_split_v21_notes.md", W7 / "notes" / "week7_primary_split_v21_notes.md"),
    ("notes/week7_model_ready_crop_dataset_v22_notes.md", W7 / "notes" / "week7_model_ready_crop_dataset_v22_notes.md"),
    ("notes/week7_crop_visual_qa_contact_sheets_v23_notes.md", W7 / "notes" / "week7_crop_visual_qa_contact_sheets_v23_notes.md"),
    ("notes/week7_final_colour_coded_bbox_overlays_v23c_fixed_notes.md", W7 / "notes" / "week7_final_colour_coded_bbox_overlays_v23c_fixed_notes.md"),

    # v17 colour identity
    ("01_colour_identity_v17/week7_final_colour_identity_v17_fixed_summary.csv", W7 / "outputs" / "colour_identity" / "final_colour_identity_v17_fixed" / "week7_final_colour_identity_v17_fixed_summary.csv"),
    ("01_colour_identity_v17/week7_final_colour_identity_v17_fixed_locked_assignments.csv", W7 / "outputs" / "colour_identity" / "final_colour_identity_v17_fixed" / "week7_final_colour_identity_v17_fixed_locked_assignments.csv"),
    ("01_colour_identity_v17/week7_final_colour_identity_v17_fixed_frame_qa.csv", W7 / "outputs" / "colour_identity" / "final_colour_identity_v17_fixed" / "week7_final_colour_identity_v17_fixed_frame_qa.csv"),

    # v18c behaviour fusion
    ("02_behaviour_fusion_v18c/week7_behaviour_label_fusion_v18c_summary.csv", W7 / "outputs" / "behaviour_label_fusion_v18c_verified_crosswalk" / "week7_behaviour_label_fusion_v18c_summary.csv"),
    ("02_behaviour_fusion_v18c/week7_behaviour_label_fusion_v18c_visual_to_behaviour_pig_id_crosswalk.csv", W7 / "outputs" / "behaviour_label_fusion_v18c_verified_crosswalk" / "week7_behaviour_label_fusion_v18c_visual_to_behaviour_pig_id_crosswalk.csv"),
    ("02_behaviour_fusion_v18c/week7_behaviour_label_fusion_v18c_fused_pig_colour_behaviour.csv", W7 / "outputs" / "behaviour_label_fusion_v18c_verified_crosswalk" / "week7_behaviour_label_fusion_v18c_fused_pig_colour_behaviour.csv"),
    ("02_behaviour_fusion_v18c/week7_behaviour_label_fusion_v18c_box_level_dataset.csv", W7 / "outputs" / "behaviour_label_fusion_v18c_verified_crosswalk" / "week7_behaviour_label_fusion_v18c_box_level_dataset.csv"),
    ("02_behaviour_fusion_v18c/week7_behaviour_label_fusion_v18c_behaviour_distribution.csv", W7 / "outputs" / "behaviour_label_fusion_v18c_verified_crosswalk" / "week7_behaviour_label_fusion_v18c_behaviour_distribution.csv"),

    # v19 final QA/stats
    ("03_final_stats_v19/week7_final_fusion_qa_v19_summary.csv", W7 / "outputs" / "final_fusion_qa_dataset_stats_v19" / "week7_final_fusion_qa_v19_summary.csv"),
    ("03_final_stats_v19/week7_final_fusion_qa_v19_behaviour_distribution.csv", W7 / "outputs" / "final_fusion_qa_dataset_stats_v19" / "week7_final_fusion_qa_v19_behaviour_distribution.csv"),
    ("03_final_stats_v19/week7_final_fusion_qa_v19_behaviour_imbalance_report.csv", W7 / "outputs" / "final_fusion_qa_dataset_stats_v19" / "week7_final_fusion_qa_v19_behaviour_imbalance_report.csv"),
    ("03_final_stats_v19/week7_final_training_ready_dataset_v19_locked.csv", W7 / "outputs" / "final_fusion_qa_dataset_stats_v19" / "week7_final_training_ready_dataset_v19_locked.csv"),

    # v21 primary split
    ("04_primary_split_v21/README_primary_split_v21.md", W7 / "outputs" / "primary_split_v21" / "README_primary_split_v21.md"),
    ("04_primary_split_v21/week7_primary_split_v21_summary.csv", W7 / "outputs" / "primary_split_v21" / "week7_primary_split_v21_summary.csv"),
    ("04_primary_split_v21/week7_primary_split_v21_all.csv", W7 / "outputs" / "primary_split_v21" / "week7_primary_split_v21_all.csv"),
    ("04_primary_split_v21/week7_primary_split_v21_train.csv", W7 / "outputs" / "primary_split_v21" / "week7_primary_split_v21_train.csv"),
    ("04_primary_split_v21/week7_primary_split_v21_val.csv", W7 / "outputs" / "primary_split_v21" / "week7_primary_split_v21_val.csv"),
    ("04_primary_split_v21/week7_primary_split_v21_test.csv", W7 / "outputs" / "primary_split_v21" / "week7_primary_split_v21_test.csv"),
    ("04_primary_split_v21/week7_primary_split_v21_class_distribution.csv", W7 / "outputs" / "primary_split_v21" / "week7_primary_split_v21_class_distribution.csv"),
    ("04_primary_split_v21/week7_primary_split_v21_video_leakage_report.csv", W7 / "outputs" / "primary_split_v21" / "week7_primary_split_v21_video_leakage_report.csv"),

    # v22 crop dataset metadata and crops
    ("05_model_ready_crops_v22/README_model_ready_crop_dataset_v22.md", W7 / "outputs" / "model_ready_crop_dataset_v22" / "README_model_ready_crop_dataset_v22.md"),
    ("05_model_ready_crops_v22/week7_model_ready_crop_dataset_v22_summary.csv", W7 / "outputs" / "model_ready_crop_dataset_v22" / "week7_model_ready_crop_dataset_v22_summary.csv"),
    ("05_model_ready_crops_v22/week7_model_ready_crop_dataset_v22_index.csv", W7 / "outputs" / "model_ready_crop_dataset_v22" / "week7_model_ready_crop_dataset_v22_index.csv"),
    ("05_model_ready_crops_v22/week7_model_ready_crop_dataset_v22_class_distribution.csv", W7 / "outputs" / "model_ready_crop_dataset_v22" / "week7_model_ready_crop_dataset_v22_class_distribution.csv"),
    ("05_model_ready_crops_v22/week7_model_ready_crop_dataset_v22_split_distribution.csv", W7 / "outputs" / "model_ready_crop_dataset_v22" / "week7_model_ready_crop_dataset_v22_split_distribution.csv"),
    ("05_model_ready_crops_v22/crops_by_split_and_behaviour", W7 / "outputs" / "model_ready_crop_dataset_v22" / "crops_by_split_and_behaviour"),

    # v23 crop visual QA
    ("06_crop_visual_qa_v23/week7_crop_visual_qa_v23_summary.csv", W7 / "outputs" / "crop_visual_qa_contact_sheets_v23" / "week7_crop_visual_qa_v23_summary.csv"),
    ("06_crop_visual_qa_v23/week7_crop_visual_qa_v23_issues.csv", W7 / "outputs" / "crop_visual_qa_contact_sheets_v23" / "week7_crop_visual_qa_v23_issues.csv"),
    ("06_crop_visual_qa_v23/contact_sheets", W7 / "outputs" / "crop_visual_qa_contact_sheets_v23" / "contact_sheets"),

    # v23c final colour-coded overlays
    ("07_final_colour_coded_overlays_v23c/week7_final_colour_coded_bbox_overlays_v23c_summary.csv", W7 / "outputs" / "final_colour_coded_bbox_overlays_v23c_fixed" / "week7_final_colour_coded_bbox_overlays_v23c_summary.csv"),
    ("07_final_colour_coded_overlays_v23c/week7_final_colour_coded_bbox_overlays_v23c_frame_qa.csv", W7 / "outputs" / "final_colour_coded_bbox_overlays_v23c_fixed" / "week7_final_colour_coded_bbox_overlays_v23c_frame_qa.csv"),
    ("07_final_colour_coded_overlays_v23c/week7_final_colour_coded_bbox_overlays_v23c_issues.csv", W7 / "outputs" / "final_colour_coded_bbox_overlays_v23c_fixed" / "week7_final_colour_coded_bbox_overlays_v23c_issues.csv"),
    ("07_final_colour_coded_overlays_v23c/contact_sheets", W7 / "outputs" / "final_colour_coded_bbox_overlays_v23c_fixed" / "contact_sheets"),
    ("07_final_colour_coded_overlays_v23c/frame_overlays", W7 / "outputs" / "final_colour_coded_bbox_overlays_v23c_fixed" / "frame_overlays"),
]


def sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def copy_item(rel_dest, src):
    dest = PACKAGE_ROOT / rel_dest
    src = Path(src)

    if not src.exists():
        return {
            "relative_path": rel_dest,
            "source_path": str(src),
            "status": "missing",
            "type": "",
            "file_count": 0,
            "total_bytes": 0,
            "sha256": "",
        }

    dest.parent.mkdir(parents=True, exist_ok=True)

    if src.is_dir():
        if dest.exists():
            shutil.rmtree(dest)
        shutil.copytree(src, dest)

        files = [p for p in dest.rglob("*") if p.is_file()]
        total_bytes = sum(p.stat().st_size for p in files)

        return {
            "relative_path": rel_dest,
            "source_path": str(src),
            "status": "copied",
            "type": "directory",
            "file_count": len(files),
            "total_bytes": total_bytes,
            "sha256": "",
        }

    shutil.copy2(src, dest)

    return {
        "relative_path": rel_dest,
        "source_path": str(src),
        "status": "copied",
        "type": "file",
        "file_count": 1,
        "total_bytes": dest.stat().st_size,
        "sha256": sha256_file(dest),
    }


manifest_rows = []
for rel_dest, src in ITEMS:
    manifest_rows.append(copy_item(rel_dest, src))

manifest = pd.DataFrame(manifest_rows)
manifest.to_csv(MANIFEST, index=False, quoting=csv.QUOTE_ALL, lineterminator="\n")


def read_single_row_csv(path):
    try:
        p = Path(path)
        if p.exists():
            df = pd.read_csv(p)
            if len(df):
                return df.iloc[0].to_dict()
    except Exception:
        pass
    return {}


v17 = read_single_row_csv(W7 / "outputs" / "colour_identity" / "final_colour_identity_v17_fixed" / "week7_final_colour_identity_v17_fixed_summary.csv")
v18c = read_single_row_csv(W7 / "outputs" / "behaviour_label_fusion_v18c_verified_crosswalk" / "week7_behaviour_label_fusion_v18c_summary.csv")
v19 = read_single_row_csv(W7 / "outputs" / "final_fusion_qa_dataset_stats_v19" / "week7_final_fusion_qa_v19_summary.csv")
v21 = read_single_row_csv(W7 / "outputs" / "primary_split_v21" / "week7_primary_split_v21_summary.csv")
v22 = read_single_row_csv(W7 / "outputs" / "model_ready_crop_dataset_v22" / "week7_model_ready_crop_dataset_v22_summary.csv")
v23 = read_single_row_csv(W7 / "outputs" / "crop_visual_qa_contact_sheets_v23" / "week7_crop_visual_qa_v23_summary.csv")
v23c = read_single_row_csv(W7 / "outputs" / "final_colour_coded_bbox_overlays_v23c_fixed" / "week7_final_colour_coded_bbox_overlays_v23c_summary.csv")

summary = pd.DataFrame([{
    "package_name": "Week7_Final_Audit_Package_v24",
    "generated_at": datetime.now().isoformat(timespec="seconds"),
    "copied_items": int((manifest["status"] == "copied").sum()),
    "missing_items": int((manifest["status"] == "missing").sum()),
    "package_files": int(sum(manifest["file_count"])),
    "package_total_bytes": int(sum(manifest["total_bytes"])),

    "v17_total_boxes": v17.get("total_boxes", ""),
    "v17_usable_colour_identity_boxes": v17.get("usable_colour_identity_boxes", ""),
    "v17_ready_for_behaviour_fusion": v17.get("ready_for_behaviour_fusion", ""),

    "v18c_matched_behaviour_boxes": v18c.get("matched_behaviour_boxes", ""),
    "v18c_ready_training_rows": v18c.get("ready_for_behaviour_model_training_rows", ""),

    "v19_training_ready_rows": v19.get("training_ready_rows", ""),
    "v19_hard_issue_count": v19.get("hard_issue_count", ""),
    "v19_ready_for_split_design": v19.get("ready_for_split_design", ""),

    "v21_primary_split": v21.get("primary_split_name", ""),
    "v21_train_rows": v21.get("train_rows", ""),
    "v21_val_rows": v21.get("val_rows", ""),
    "v21_test_rows": v21.get("test_rows", ""),
    "v21_same_frame_leakage": v21.get("same_frame_leakage", ""),
    "v21_same_video_leakage": v21.get("same_video_leakage", ""),

    "v22_expected_rows": v22.get("expected_rows", ""),
    "v22_exported_crops": v22.get("exported_crops", ""),
    "v22_ready_for_feature_extraction": v22.get("ready_for_feature_extraction", ""),

    "v23_contact_sheets_created": v23.get("contact_sheets_created", ""),
    "v23_issue_count": v23.get("issue_count", ""),

    "v23c_frames_processed": v23c.get("frames_processed", ""),
    "v23c_drawn_boxes": v23c.get("drawn_boxes", ""),
    "v23c_ready_for_report_visuals": v23c.get("ready_for_report_visuals", ""),
}])

summary.to_csv(SUMMARY, index=False, quoting=csv.QUOTE_ALL, lineterminator="\n")

README.write_text(f"""# Week 7 Final Audit Package v24

## Purpose

This package collects the validated outputs from the Week 7 dataset validation, identity tracking preparation, and behaviour preparation workflow.

It is intended as an evidence package before moving to tracking, temporal consistency, clip-level representation, feature extraction, or baseline modelling.

## Main pipeline status

1. Final corrected boxes and colour identity were locked.
2. Visual marker colours were mapped to annotation-specific behaviour pig IDs through a verified crosswalk.
3. Behaviour labels were fused with final pig boxes.
4. Dataset QA/statistics were generated.
5. A primary frame-stratified split was locked.
6. Model-ready crops were exported.
7. Crop visual QA contact sheets were generated.
8. Final colour-coded bounding box overlays were generated.

## Key results

- v17 usable colour identities: `{summary.iloc[0].get('v17_usable_colour_identity_boxes', '')}`
- v18c matched behaviour boxes: `{summary.iloc[0].get('v18c_matched_behaviour_boxes', '')}`
- v19 training-ready rows: `{summary.iloc[0].get('v19_training_ready_rows', '')}`
- v21 split: train `{summary.iloc[0].get('v21_train_rows', '')}`, val `{summary.iloc[0].get('v21_val_rows', '')}`, test `{summary.iloc[0].get('v21_test_rows', '')}`
- v22 exported crops: `{summary.iloc[0].get('v22_exported_crops', '')}`
- v23c drawn boxes: `{summary.iloc[0].get('v23c_drawn_boxes', '')}`

## Important limitations

- The dataset is small and imbalanced.
- Rare behaviours such as IA, BE, and DE have very few examples.
- The primary split is frame-stratified, not strict video-level generalization.
- Some behaviour classes are weakly represented by single-frame crops.
- Unknown / not-visible / uncertain identities are retained and documented rather than forced into incorrect labels.

## Folder guide

- `01_colour_identity_v17/`: final locked visual colour identity.
- `02_behaviour_fusion_v18c/`: verified visual-colour to behaviour-pig-ID fusion.
- `03_final_stats_v19/`: final QA and dataset statistics.
- `04_primary_split_v21/`: locked primary train/validation/test split.
- `05_model_ready_crops_v22/`: model-ready crop images and index.
- `06_crop_visual_qa_v23/`: crop visual QA contact sheets.
- `07_final_colour_coded_overlays_v23c/`: report-ready colour-coded bounding box overlays.
- `notes/`: methodological notes for each stage.

## Manifest

See `manifest_v24.csv` for copied files and SHA256 checksums.
""")

NOTE.write_text(
    "# Week 7 Final Audit Package v24\n\n"
    "## Purpose\n\n"
    "This step packages the validated Week 7 outputs into one final audit/evidence folder and zip archive.\n\n"
    "## Summary\n\n"
    f"- Package root: `{PACKAGE_ROOT}`\n"
    f"- Zip path: `{ZIP_PATH}`\n"
    f"- Copied items: `{int(summary.iloc[0]['copied_items'])}`\n"
    f"- Missing items: `{int(summary.iloc[0]['missing_items'])}`\n"
    f"- Package files: `{int(summary.iloc[0]['package_files'])}`\n"
    f"- v18c matched behaviour boxes: `{summary.iloc[0].get('v18c_matched_behaviour_boxes', '')}`\n"
    f"- v19 training-ready rows: `{summary.iloc[0].get('v19_training_ready_rows', '')}`\n"
    f"- v22 exported crops: `{summary.iloc[0].get('v22_exported_crops', '')}`\n"
    f"- v23c drawn boxes: `{summary.iloc[0].get('v23c_drawn_boxes', '')}`\n\n"
    "## Outputs\n\n"
    f"- README: `{README}`\n"
    f"- Summary: `{SUMMARY}`\n"
    f"- Manifest: `{MANIFEST}`\n"
    f"- Zip archive: `{ZIP_PATH}`\n"
)

if ZIP_PATH.exists():
    ZIP_PATH.unlink()

with zipfile.ZipFile(ZIP_PATH, "w", compression=zipfile.ZIP_DEFLATED) as zf:
    for p in PACKAGE_ROOT.rglob("*"):
        if p.is_file():
            zf.write(p, p.relative_to(OUT_ROOT))

print("Saved package root:")
print(PACKAGE_ROOT)
print()
print("Saved zip:")
print(ZIP_PATH)
print()
print("=== v24 package summary ===")
print(summary.to_string(index=False))
print()
print("=== missing items ===")
missing = manifest[manifest["status"] == "missing"]
if len(missing):
    print(missing.to_string(index=False))
else:
    print("No missing items.")
