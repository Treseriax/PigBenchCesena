# Week 6 Final Report Outline

## 1. Objective

Validate the Unibo pig behaviour dataset workflow and build a reliable foundation for future behaviour classification. The pipeline links raw videos, manual Excel scan-sampling labels, scanpoint frames, pig detections, visual QC, and candidate feature extractors.

## 2. Dataset and Raw Video Inventory

- Report raw video count, naming families, camera tokens, and direct TLC/c-token distinction.
- Use dataset overview metrics and raw naming analysis tables.

## 3. Manual Annotation Parsing

- Explain Excel structure: six colour IDs, 12 hourly blocks, six scan points per hour.
- Explain behaviour code mapping and final 432-label GT table.

## 4. Video Linkage and Recovered c-token Mapping

- Explain direct TLC matches for 07:00-15:00.
- Explain candidate recovered c-token videos for 15:00-19:00.
- Keep candidate recovered mappings marked as medium confidence.

## 5. Scanpoint Frame Extraction

- Explain 72 extracted scanpoint frames.
- Explain frame-to-label alignment: each scan frame has six manual labels.

## 6. YOLOv8-s Detection and BBox Features

- Explain detector setup, 540 detected bboxes, and mean bbox count per frame.
- Explain why bbox outputs are automatic detector outputs, not manual bbox ground truth.
- Include QC flags and conservative score >= 0.50 subset.

## 7. Crop Colour-Marker Candidate Features

- Explain green/blue/purple/red HSV marker features.
- Explain candidate bbox-to-colour assignment.
- State limitations: red is ambiguous between red_neck/red_tail; no_color cannot be marker-detected.

## 8. Visual QC Outputs

- Include bbox+label overlay v2.
- Include marker-bbox overlay v3.
- Include low/high bbox-count contact sheets.

## 9. Split Protocol

- Use recommended split v2.
- Explain video-hour level no-leakage design.
- Explain train/val/test sizes: 288/72/72 labels.
- Explain behaviour coverage: train 11/11, val 10/11, test 10/11.

## 10. Feature Extractor Comparison

- Compare manual labels, scanpoint frames, YOLOv8 bbox geometry, conservative bbox subset, crop marker features, and candidate bbox-to-colour assignment.
- Keep segmentation and embedding features as future extensions unless implemented later.

## 11. Consistency Audits and Limitations

- Report final post-fix audit: 0 failed checks.
- Discuss small dataset limitations, rare behaviours, candidate video mappings, automatic detector boxes, and unresolved identity mapping.

## 12. Conclusion

The Week 6 pipeline is ready for reporting and future experiments. It provides a controlled dataset validation workflow, aligned GT labels, visualizations, detector-based bbox features, candidate marker features, and a no-leakage recommended split.

## Key Deliverables

| category                     | priority   | exists   | path                                                                                               |   size_mb | description                                                                                             | report_use                               |
|:-----------------------------|:-----------|:---------|:---------------------------------------------------------------------------------------------------|----------:|:--------------------------------------------------------------------------------------------------------|:-----------------------------------------|
| Unified ground truth         | core       | True     | outputs/unified_ground_truth/week6_unified_ground_truth_v2_with_recommended_split_v2.csv           |     0.419 | Final recommended unified GT table with manual labels, recovered video paths, and recommended split v2. | Main dataset table                       |
| Split protocol               | core       | True     | outputs/dataset_statistics/week6_recommended_split_v2_summary.csv                                  |     0     | Final recommended train/val/test summary.                                                               | Split protocol section                   |
| Split protocol               | core       | True     | outputs/dataset_statistics/week6_recommended_split_v2_video_level.csv                              |     0.002 | Video-hour level split table; prevents frame/video leakage.                                             | Split protocol section                   |
| Split protocol               | core       | True     | outputs/dataset_statistics/week6_recommended_split_v2_behaviour_coverage.csv                       |     0     | Behaviour coverage by train/val/test.                                                                   | Split limitations and coverage           |
| Dataset statistics           | core       | True     | outputs/dataset_statistics/week6_final_dataset_overview_metrics.csv                                |     0.001 | Final dataset overview metrics.                                                                         | Dataset statistics section               |
| Dataset statistics           | core       | True     | outputs/dataset_statistics/week6_final_behaviour_distribution.csv                                  |     0.001 | Behaviour class distribution.                                                                           | Class distribution table                 |
| Dataset statistics           | core       | True     | outputs/dataset_statistics/week6_final_rare_behaviour_classes.csv                                  |     0     | Rare behaviour classes requiring special handling.                                                      | Dataset limitations                      |
| Frame extraction             | core       | True     | outputs/unified_ground_truth/week6_scanpoint_frame_index.csv                                       |     0.026 | 72 extracted scanpoint frames linked to timestamps and videos.                                          | Frame-label alignment section            |
| Manual labels                | core       | True     | outputs/unified_ground_truth/week6_scanpoint_frame_labels_long.csv                                 |     0.154 | Long-format table containing six manual colour-ID labels per scanpoint frame.                           | Ground-truth parsing section             |
| Detection features           | core       | True     | outputs/feature_extractors/week6_yolov8s_all_scanpoint_detections_with_qc_flags.csv                |     0.427 | YOLOv8-s detection table with QC flags.                                                                 | BBox feature extractor section           |
| Detection features           | optional   | True     | outputs/feature_extractors/week6_yolov8s_conservative_detections_score_ge_0_50.csv                 |     0.352 | Conservative detection subset for cleaner downstream feature extraction.                                | Optional conservative analysis           |
| Marker features              | candidate  | True     | outputs/feature_extractors/week6_crop_colour_marker_features.csv                                   |     0.563 | Crop-based green/blue/purple/red marker evidence for each bbox.                                         | Candidate identity association section   |
| Marker features              | candidate  | True     | outputs/feature_extractors/week6_candidate_bbox_to_colour_assignments.csv                          |     0.132 | High/medium-confidence candidate bbox-to-colour assignments.                                            | Candidate identity association section   |
| Feature extractor comparison | core       | True     | outputs/feature_extractors/week6_feature_extractor_comparison_final_clean_v3.csv                   |     0.003 | Final cleaned comparison of implemented and future feature extractor families.                          | Feature extractor comparison section     |
| Feature extractor comparison | core       | True     | outputs/feature_extractors/week6_feature_extractor_recommendations_final_clean_v3.csv              |     0     | Ranked recommendation of feature sets.                                                                  | Feature extractor recommendation section |
| QC                           | core       | True     | outputs/dataset_statistics/week6_postfix_final_consistency_audit.csv                               |     0.002 | Final post-fix consistency audit with all checks passing.                                               | Validation and QC section                |
| QC                           | core       | True     | outputs/dataset_statistics/week6_detection_qc_conservative_subset_summary.csv                      |     0     | Detection QC and conservative subset summary.                                                           | Detector QC section                      |
| Visualization                | core       | True     | outputs/visual_label_check/marker_bbox_overlay_v3_video/week6_marker_bbox_overlay_v3_slideshow.mp4 |     6.764 | Slideshow video showing bbox, marker candidates, and manual labels.                                     | Demo/video deliverable                   |
| Visualization                | core       | True     | outputs/visual_label_check/bbox_count_warning_qc/all_bbox_count_warning_frames_contact_sheet.jpg   |     0.925 | Contact sheet of low/high bbox-count QC frames.                                                         | Detector QC visual evidence              |

## Missing Deliverables

No missing key deliverables.
