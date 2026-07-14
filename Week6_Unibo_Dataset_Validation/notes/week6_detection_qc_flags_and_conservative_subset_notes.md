# Week 6 Detection QC Flags and Conservative Subset Notes

## Purpose

This step adds QC flags to the YOLOv8-s detection outputs without overwriting or deleting the primary detector table. It also creates conservative score-filtered subsets for optional downstream analysis.

## Decision

The primary detector output remains the threshold 0.25 table. A global threshold change is not adopted because low-bbox frames would lose additional pigs at higher thresholds, while lower thresholds increase false-positive risk. Warning frames are instead explicitly flagged for visual QC and optional conservative analysis.

## Outputs

- QC flagged detections: `/home/oyavuz/PigBench/Week6_Unibo_Dataset_Validation/outputs/feature_extractors/week6_yolov8s_all_scanpoint_detections_with_qc_flags.csv`
- Conservative score >= 0.50 subset: `/home/oyavuz/PigBench/Week6_Unibo_Dataset_Validation/outputs/feature_extractors/week6_yolov8s_conservative_detections_score_ge_0_50.csv`
- Conservative score >= 0.70 subset: `/home/oyavuz/PigBench/Week6_Unibo_Dataset_Validation/outputs/feature_extractors/week6_yolov8s_conservative_detections_score_ge_0_70.csv`
- Frame QC summary: `/home/oyavuz/PigBench/Week6_Unibo_Dataset_Validation/outputs/dataset_statistics/week6_detection_qc_frame_summary.csv`
- Issue/score-band summary: `/home/oyavuz/PigBench/Week6_Unibo_Dataset_Validation/outputs/dataset_statistics/week6_detection_qc_issue_score_band_summary.csv`
- Conservative subset summary: `/home/oyavuz/PigBench/Week6_Unibo_Dataset_Validation/outputs/dataset_statistics/week6_detection_qc_conservative_subset_summary.csv`

## Conservative subset summary

| subset                     |   detection_count |   frame_count_with_detections |   mean_bboxes_per_frame |   low_bbox_warning_frames |   high_bbox_warning_frames |
|:---------------------------|------------------:|------------------------------:|------------------------:|--------------------------:|---------------------------:|
| primary_score_ge_0_25      |               540 |                            72 |                   7.5   |                        10 |                          4 |
| conservative_score_ge_0_50 |               447 |                            72 |                   6.208 |                           |                            |
| conservative_score_ge_0_70 |               376 |                            72 |                   5.222 |                           |                            |

## Issue and score-band summary

| bbox_count_issue_type   | detection_score_band       |   detection_count |
|:------------------------|:---------------------------|------------------:|
| high_bbox_count_warning | high_score_ge_0_50         |                 4 |
| high_bbox_count_warning | low_accepted_score_ge_0_25 |                11 |
| high_bbox_count_warning | medium_score_ge_0_35       |                10 |
| high_bbox_count_warning | very_high_score_ge_0_70    |                26 |
| low_bbox_count_warning  | high_score_ge_0_50         |                 7 |
| low_bbox_count_warning  | low_accepted_score_ge_0_25 |                 2 |
| low_bbox_count_warning  | medium_score_ge_0_35       |                 1 |
| low_bbox_count_warning  | very_high_score_ge_0_70    |                39 |
| normal_bbox_count       | high_score_ge_0_50         |                60 |
| normal_bbox_count       | low_accepted_score_ge_0_25 |                34 |
| normal_bbox_count       | medium_score_ge_0_35       |                35 |
| normal_bbox_count       | very_high_score_ge_0_70    |               311 |

## Interpretation

The primary table should be used when recall is important. The conservative score >= 0.50 subset can be used for cleaner crop/embedding features when precision is more important. The score >= 0.70 subset is stricter and may miss occluded pigs. Low/high bbox-count frames should remain documented as detector QC limitations rather than treated as manual annotation errors.
