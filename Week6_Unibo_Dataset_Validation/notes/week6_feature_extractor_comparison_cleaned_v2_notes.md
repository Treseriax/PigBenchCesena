# Week 6 Feature Extractor Comparison Cleaned v2

## Purpose

This cleaned v2 version fixes minor wording/spacing issues in the feature extractor comparison table. It does not change any counts, paths, implementation status, or technical interpretation.

## Feature extractor comparison

| extractor_family                    | implemented   | main_output                                                                                                                                   |   record_count | feature_examples                                                                        | strength                                                       | limitation                                                             | recommended_use                                     |
|:------------------------------------|:--------------|:----------------------------------------------------------------------------------------------------------------------------------------------|---------------:|:----------------------------------------------------------------------------------------|:---------------------------------------------------------------|:-----------------------------------------------------------------------|:----------------------------------------------------|
| Manual Excel scan-sampling labels   | yes           | /home/oyavuz/PigBench/Week6_Unibo_Dataset_Validation/outputs/unified_ground_truth/week6_unified_ground_truth_v2_with_recommended_split_v2.csv |            432 | behaviour_code, behaviour_label, colour_id, timestamp, recommended_split_v2             | Reliable manual behaviour labels at scan-sampling timestamps   | Not dense frame-level labels; no manual bbox                           | Primary label source                                |
| Scanpoint frame extraction          | yes           | /home/oyavuz/PigBench/Week6_Unibo_Dataset_Validation/outputs/unified_ground_truth/week6_scanpoint_frame_index.csv                             |             72 | scan_frame_id, frame_index, timestamp_sec_in_video, frame_image_path                    | Connects labels to actual video frames                         | Only 72 scanpoint frames, not full video                               | Visual QC and detector input                        |
| YOLOv8-s bbox detector              | yes           | /home/oyavuz/PigBench/Week6_Unibo_Dataset_Validation/outputs/feature_extractors/week6_yolov8s_all_scanpoint_detections_with_qc_flags.csv      |            540 | x1, y1, x2, y2, bbox_area, score, bbox_count_issue_type                                 | Provides pig bounding boxes for all 72 frames                  | Automatic detector output; not manual GT; bbox-to-colour ID unresolved | BBox geometry, crop extraction, visual overlays     |
| Conservative bbox subset            | yes           | /home/oyavuz/PigBench/Week6_Unibo_Dataset_Validation/outputs/feature_extractors/week6_yolov8s_conservative_detections_score_ge_0_50.csv       |            447 | score >= 0.50 detections                                                                | Cleaner higher-confidence boxes while still covering 72 frames | May miss occluded pigs                                                 | Cleaner crop/embedding features                     |
| Crop colour-marker features         | yes           | /home/oyavuz/PigBench/Week6_Unibo_Dataset_Validation/outputs/feature_extractors/week6_crop_colour_marker_features.csv                         |            540 | green_score, blue_score, purple_score, red_score, best_marker_colour, marker_confidence | Candidate bbox-to-colour association                           | Red is ambiguous; no_color cannot be marker-detected                   | Candidate identity/colour QC, not final identity GT |
| Candidate bbox-to-colour assignment | yes           | /home/oyavuz/PigBench/Week6_Unibo_Dataset_Validation/outputs/feature_extractors/week6_candidate_bbox_to_colour_assignments.csv                |            196 | candidate_colour_id, assignment_status, marker_confidence                               | Links high/medium marker evidence to candidate colour IDs      | Needs visual/manual confirmation                                       | Candidate association table                         |
| Segmentation features               | not yet       | nan                                                                                                                                           |            nan | mask area, body contour, posture shape                                                  | Could improve body-shape and posture representation            | Requires segmentation model or manual masks                            | Future extension                                    |
| Embedding features                  | not yet       | nan                                                                                                                                           |            nan | CNN/ViT crop embeddings, temporal embeddings                                            | Useful for learned representation comparison                   | Less interpretable; needs controlled validation                        | Optional future model input                         |

## Recommendation ranking

|   rank | feature_set                          | status    | reason                                                 |
|-------:|:-------------------------------------|:----------|:-------------------------------------------------------|
|      1 | Manual labels + scanpoint frames     | core      | Ground truth alignment foundation                      |
|      2 | YOLOv8 bbox geometry + QC flags      | core      | Interpretable object-level features                    |
|      3 | Crop colour-marker features          | candidate | Useful for colour ID association but not final GT      |
|      4 | Conservative bbox subset score>=0.50 | optional  | Cleaner features when precision matters                |
|      5 | Segmentation/embedding features      | future    | Promising but outside current validated implementation |

## Interpretation

The validated Week 6 feature pipeline remains: manual scan-sampling labels, scanpoint frames, YOLOv8-s bbox geometry with QC flags, and crop colour-marker candidate features. Segmentation and embedding features remain future extensions.
