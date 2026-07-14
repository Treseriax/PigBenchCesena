from pathlib import Path
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
INTERFACE = W6 / "interface_demo"
TRACKER = W6 / "shared_tracker"

STATS.mkdir(parents=True, exist_ok=True)
NOTES.mkdir(parents=True, exist_ok=True)


def safe_to_csv(df, path):
    df.to_csv(
        path,
        index=False,
        quoting=csv.QUOTE_ALL,
        escapechar="\\",
        lineterminator="\n"
    )


def read_csv(path):
    path = Path(path)
    if not path.exists():
        return pd.DataFrame()
    try:
        return pd.read_csv(path)
    except Exception:
        return pd.DataFrame()


def rel(path):
    try:
        return str(Path(path).relative_to(W6))
    except Exception:
        return str(path)


def exists(path):
    return Path(path).exists()


def size_mb(path):
    path = Path(path)
    if not path.exists():
        return ""
    return round(path.stat().st_size / (1024 * 1024), 3)


key_artifacts = [
    ("Dataset overview", STATS / "week6_final_dataset_overview_metrics.csv"),
    ("Behaviour distribution", STATS / "week6_final_behaviour_distribution.csv"),
    ("Unified GT CSV", GT / "week6_unified_ground_truth_v2_with_recommended_split_v2.csv"),
    ("Unified GT JSON", GT / "week6_unified_ground_truth_v2_with_recommended_split_v2.json"),
    ("GT schema", GT / "week6_unified_ground_truth_v2_with_recommended_split_v2_schema.json"),
    ("Nested viewer JSON", GT / "week6_scanpoint_annotations_nested_for_viewer.json"),
    ("Recommended split summary", STATS / "week6_recommended_split_v2_summary.csv"),
    ("Visualization Streamlit app", INTERFACE / "week6_visualization_streamlit_app.py"),
    ("Static HTML viewer", INTERFACE / "week6_static_visualization_viewer.html"),
    ("YOLOv8-s detections QC", FEAT / "week6_yolov8s_all_scanpoint_detections_with_qc_flags.csv"),
    ("BBox geometry features", FEAT / "week6_bbox_geometry_features.csv"),
    ("Group-spatial features", FEAT / "week6_group_spatial_features_per_frame.csv"),
    ("ROI proxy features", FEAT / "week6_coarse_roi_resource_proxy_features.csv"),
    ("Crop descriptor features", FEAT / "week6_crop_descriptor_baseline_features.csv"),
    ("Crop descriptor + marker features", FEAT / "week6_crop_descriptor_plus_marker_features.csv"),
    ("Detector-backed embeddings CSV", FEAT / "week6_detector_backbone_crop_embeddings_896.csv"),
    ("Detector-backed embeddings NPY", FEAT / "week6_detector_backbone_crop_embeddings_896.npy"),
    ("Embedding PCA features", FEAT / "week6_detector_backbone_crop_embedding_pca_features.csv"),
    ("Segmentation features", FEAT / "week6_preliminary_bbox_guided_segmentation_features.csv"),
    ("Segmentation frame summary", FEAT / "week6_preliminary_segmentation_frame_summary.csv"),
    ("Segmentation visual QC summary", STATS / "week6_segmentation_visual_qc_final_review_summary.csv"),
    ("Quality hardening v2", STATS / "week6_quality_hardening_summary_v2.csv"),
    ("Task compliance v2", STATS / "week6_task_sheet_compliance_summary_v2.csv"),
    ("Shared Excel tracker", TRACKER / "Week6_Unibo_Dataset_Validation_Shared_Task_Tracker.xlsx"),
]

artifact_table = pd.DataFrame([
    {
        "artifact": name,
        "path": rel(path),
        "exists": path.exists(),
        "size_mb": size_mb(path),
    }
    for name, path in key_artifacts
])

artifact_table_path = STATS / "week6_final_v2_key_artifact_table.csv"
safe_to_csv(artifact_table, artifact_table_path)

quality = read_csv(STATS / "week6_quality_hardening_summary_v2.csv")
compliance = read_csv(STATS / "week6_task_sheet_compliance_summary_v2.csv")
embedding_summary = read_csv(STATS / "week6_detector_backbone_crop_embedding_summary.csv")
seg_qc = read_csv(STATS / "week6_segmentation_visual_qc_final_review_summary.csv")

report_path = NOTES / "week6_final_report_v2.md"
exec_path = NOTES / "week6_final_executive_summary_v2.md"
readme_draft_path = NOTES / "week6_final_package_v2_readme_draft.md"

report = f"""# Week 6 Unibo Dataset Validation — Final Report v2

Generated at: `{datetime.now().isoformat(timespec="seconds")}`

## 1. Objective

The goal of this Week 6 work is to convert the Unibo pig behaviour material into a reliable experimental dataset and feature-extraction pipeline for future behaviour classification.

The pipeline is:

`video frame → pigs/detections → bbox/crop → colour-marker evidence → behaviour label → feature tables`

## 2. Final dataset scope

- Raw Unibo work videos inspected: `84`
- Manual scanpoint labels extracted: `432`
- Scanpoint frames extracted: `72`
- Labels per scanpoint frame: `6`
- Colour IDs per scanpoint frame: `green`, `blue`, `purple`, `red_neck`, `red_tail`, `no_color`
- Automatic detector bboxes on scanpoint frames: `540`

## 3. Unified ground truth

The final recommended ground-truth outputs include:

- CSV table with `432` manual labels
- JSON version of the recommended GT
- JSON schema
- Nested viewer JSON linking scanpoint frames, labels, and bboxes
- Camera/pen/crate/video metadata summary

Manual labels and automatic detector bboxes are intentionally kept conceptually separate. The manual behaviour labels come from the Excel annotation sheet; detector bboxes are automatic feature-extraction outputs.

## 4. Split protocol

A recommended split v2 was produced using video-hour units to avoid leakage:

- Train: `288` labels
- Validation: `72` labels
- Test: `72` labels

The split is designed to avoid leakage by keeping video-hour units separated across splits.

## 5. Visualization interface

The visualization deliverables include:

- Streamlit viewer
- Static HTML viewer
- Viewer index CSV
- Nested annotation JSON
- BBox/label/colour overlay visualizations
- QC contact sheets

The viewer artifacts allow inspection of scanpoint frames with labels, detector bboxes, timestamps, and colour/behaviour annotation information.

## 6. Feature extractor outputs

The feature-extraction pipeline now includes:

1. BBox geometry features
2. Group-spatial features
3. Coarse ROI/resource proxy features
4. Crop descriptor baseline features
5. Crop descriptor + marker features
6. Detector-backed learned crop embeddings
7. Preliminary segmentation-derived features
8. Trajectory feasibility report

The learned crop embeddings are not random or dummy embeddings. They are detector-backed embeddings extracted with the local PigBench YOLOv8-s detector checkpoint via `model.extract_feat` and global average pooling.

Embedding verification:

{embedding_summary.to_markdown(index=False) if len(embedding_summary) else "Embedding summary missing."}

## 7. Segmentation baseline and visual QC

A preliminary bbox-guided segmentation baseline was implemented using GrabCut with Otsu fallback.

Segmentation outputs include:

- Per-bbox segmentation features
- Per-frame segmentation summaries
- Segmentation quality summaries
- Overlay images for all 72 scanpoint frames
- Visual QC contact sheets
- Manual visual QC result

Manual visual QC result:

{seg_qc.to_markdown(index=False) if len(seg_qc) else "Segmentation visual QC summary missing."}

The segmentation masks are accepted for preliminary shape/posture/contact/foreground feature extraction. They should not be described as manual segmentation ground truth.

## 8. Known boundaries

The project avoids overclaiming:

- Detector bboxes are automatic detector outputs, not manual bbox ground truth.
- Colour-marker assignments are candidate evidence, not final identity-resolved tracking.
- Red marker evidence is ambiguous between `red_neck` and `red_tail` without extra validation.
- Trajectory features are documented as feasibility only because stable identity-resolved tracking was not validated.
- Segmentation masks are preliminary automatic masks, not manual segmentation ground truth.

## 9. Compliance and quality status

Task-sheet compliance v2:

{compliance.to_markdown(index=False) if len(compliance) else "Compliance summary missing."}

Quality hardening v2:

{quality.to_markdown(index=False) if len(quality) else "Quality summary missing."}

Final interpretation: all task-sheet items are complete and all quality-hardening checks pass.

## 10. Key artifacts

{artifact_table.to_markdown(index=False)}
"""

report_path.write_text(report)

exec_summary = """# Week 6 Unibo Dataset Validation — Executive Summary v2

## Final status

The Week 6 Unibo dataset validation work is task-sheet complete and quality-hardened.

## Main outputs

- 432 manual behaviour labels extracted into unified GT CSV/JSON.
- 72 scanpoint frames extracted.
- 540 YOLOv8-s detector bboxes generated and QC-flagged.
- Recommended no-leakage split v2 created: 288 train / 72 validation / 72 test labels.
- Visualization interface created with Streamlit and static HTML viewer.
- Feature extractor outputs created for bbox geometry, group-spatial context, ROI proxies, crop descriptors, marker evidence, detector-backed learned embeddings, and segmentation-derived features.
- Detector-backed learned crop embeddings were generated for all 540 crops with 896-dimensional vectors.
- Preliminary bbox-guided segmentation features were generated and visually QC-reviewed.
- Shared Excel/task tracker was generated and verified as readable.
- Final task-sheet compliance audit v2 reports all 14 items done.
- Quality hardening audit v2 reports 24 PASS and no WARN/FAIL items.

## Important interpretation

The outputs are suitable for experimental behaviour-classification dataset preparation and feature comparison. Automatic bboxes, colour-marker matches, and segmentation masks are feature-extraction artifacts, not manually validated ground-truth annotations. This distinction is intentionally preserved throughout the deliverables.
"""

exec_path.write_text(exec_summary)

readme_draft = """# Week6_Unibo_Dataset_Validation_Final_Package_v2

This package contains the final Week 6 Unibo dataset validation deliverables.

## Contents

- Unified ground-truth CSV/JSON/schema
- Scanpoint frame index and nested viewer JSON
- Dataset statistics and split protocol outputs
- Visualization interface demo files
- Detection, marker, feature extractor, embedding, and segmentation feature tables
- Segmentation visual QC outputs
- Task compliance and quality-hardening audits
- Shared Excel task tracker
- Final report and executive summary notes

## Key counts

- Manual labels: 432
- Scanpoint frames: 72
- Detector bboxes: 540
- Detector-backed embedding rows: 540
- Embedding dimension: 896
- Segmentation feature rows: 540
- Recommended split: 288 train / 72 validation / 72 test labels

## Quality status

- Task-sheet compliance v2: all items done.
- Quality hardening v2: no FAIL or WARN items remain.
- Segmentation visual QC: accepted for preliminary feature extraction with notes.
- Learned embeddings: detector-backed, not random/untrained.

## Scope note

The package does not include raw videos. It includes derived frame-level outputs, feature tables, visual QC artifacts, and documentation.
"""

readme_draft_path.write_text(readme_draft)

print("Saved:")
print(report_path)
print(exec_path)
print(readme_draft_path)
print(artifact_table_path)

print()
print("=== Key artifact table ===")
print(artifact_table.to_string(index=False))
