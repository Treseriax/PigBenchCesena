# Week 6 Unibo Dataset Validation — Final Report v2

Generated at: `2026-07-05T00:20:01`

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

| metric                    | value                                               | interpretation                                                           |
|:--------------------------|:----------------------------------------------------|:-------------------------------------------------------------------------|
| input_detection_rows      | 540                                                 | All detector rows considered for crop embedding extraction.              |
| successful_embedding_rows | 540                                                 | Rows with usable crop and finite detector-backbone embedding.            |
| failed_crop_rows          | 0                                                   | Rows skipped due to crop loading/crop quality problems.                  |
| embedding_dim             | 896                                                 | Detector-backbone pooled feature dimension.                              |
| embedding_method          | mmdet_yolov8s_pig_detector_backbone_global_avg_pool | Local PigBench YOLOv8-s detector checkpoint, not random/untrained model. |
| device                    | cuda:0                                              | Inference device.                                                        |
| pca_components            | 16                                                  | Compact PCA features generated for downstream visualization/comparison.  |

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

| metric                            | value                                                  | interpretation                                                                         |
|:----------------------------------|:-------------------------------------------------------|:---------------------------------------------------------------------------------------|
| manual_sample_reviewed_frames     | 33                                                     | Frames included in manual visual QC sample table.                                      |
| high_risk_sheet_score             | 4                                                      | Human visual review score for high-risk contact sheet.                                 |
| segmentation_visual_qc_verdict    | accepted_for_preliminary_feature_extraction_with_notes | Final visual QC verdict for current preliminary segmentation baseline.                 |
| fix_required_before_final_package | False                                                  | No segmentation fix is required before final package based on high-risk visual review. |
| recommended_segmentation_use      | preliminary_feature_extraction                         | Use for shape/posture/contact/foreground feature analysis, not as manual mask GT.      |

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

| status   |   count |
|:---------|--------:|
| done     |      14 |

Quality hardening v2:

| status   | severity   |   count |
|:---------|:-----------|--------:|
| PASS     | none       |      24 |

Final interpretation: all task-sheet items are complete and all quality-hardening checks pass.

## 10. Key artifacts

| artifact                          | path                                                                                             | exists   |   size_mb |
|:----------------------------------|:-------------------------------------------------------------------------------------------------|:---------|----------:|
| Dataset overview                  | outputs/dataset_statistics/week6_final_dataset_overview_metrics.csv                              | True     |     0.001 |
| Behaviour distribution            | outputs/dataset_statistics/week6_final_behaviour_distribution.csv                                | True     |     0.001 |
| Unified GT CSV                    | outputs/unified_ground_truth/week6_unified_ground_truth_v2_with_recommended_split_v2.csv         | True     |     0.419 |
| Unified GT JSON                   | outputs/unified_ground_truth/week6_unified_ground_truth_v2_with_recommended_split_v2.json        | True     |     0.863 |
| GT schema                         | outputs/unified_ground_truth/week6_unified_ground_truth_v2_with_recommended_split_v2_schema.json | True     |     0.008 |
| Nested viewer JSON                | outputs/unified_ground_truth/week6_scanpoint_annotations_nested_for_viewer.json                  | True     |     0.41  |
| Recommended split summary         | outputs/dataset_statistics/week6_recommended_split_v2_summary.csv                                | True     |     0     |
| Visualization Streamlit app       | interface_demo/week6_visualization_streamlit_app.py                                              | True     |     0.002 |
| Static HTML viewer                | interface_demo/week6_static_visualization_viewer.html                                            | True     |     0.036 |
| YOLOv8-s detections QC            | outputs/feature_extractors/week6_yolov8s_all_scanpoint_detections_with_qc_flags.csv              | True     |     0.427 |
| BBox geometry features            | outputs/feature_extractors/week6_bbox_geometry_features.csv                                      | True     |     0.22  |
| Group-spatial features            | outputs/feature_extractors/week6_group_spatial_features_per_frame.csv                            | True     |     0.023 |
| ROI proxy features                | outputs/feature_extractors/week6_coarse_roi_resource_proxy_features.csv                          | True     |     0.141 |
| Crop descriptor features          | outputs/feature_extractors/week6_crop_descriptor_baseline_features.csv                           | True     |     0.137 |
| Crop descriptor + marker features | outputs/feature_extractors/week6_crop_descriptor_plus_marker_features.csv                        | True     |     0.176 |
| Detector-backed embeddings CSV    | outputs/feature_extractors/week6_detector_backbone_crop_embeddings_896.csv                       | True     |    10.48  |
| Detector-backed embeddings NPY    | outputs/feature_extractors/week6_detector_backbone_crop_embeddings_896.npy                       | True     |     1.846 |
| Embedding PCA features            | outputs/feature_extractors/week6_detector_backbone_crop_embedding_pca_features.csv               | True     |     0.261 |
| Segmentation features             | outputs/feature_extractors/week6_preliminary_bbox_guided_segmentation_features.csv               | True     |     0.262 |
| Segmentation frame summary        | outputs/feature_extractors/week6_preliminary_segmentation_frame_summary.csv                      | True     |     0.012 |
| Segmentation visual QC summary    | outputs/dataset_statistics/week6_segmentation_visual_qc_final_review_summary.csv                 | True     |     0.001 |
| Quality hardening v2              | outputs/dataset_statistics/week6_quality_hardening_summary_v2.csv                                | True     |     0     |
| Task compliance v2                | outputs/dataset_statistics/week6_task_sheet_compliance_summary_v2.csv                            | True     |     0     |
| Shared Excel tracker              | shared_tracker/Week6_Unibo_Dataset_Validation_Shared_Task_Tracker.xlsx                           | True     |     0.013 |
