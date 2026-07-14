# Week 6 Bbox Warning Threshold and Duplicate Analysis

## Purpose

This controlled analysis reruns detector inference only on bbox-count warning frames and checks whether the warnings are sensitive to score threshold or caused by duplicate overlapping detections.

## Outputs

- Raw detections >=0.05: `/home/oyavuz/PigBench/Week6_Unibo_Dataset_Validation/outputs/feature_extractors/week6_bbox_warning_raw_detections_threshold005.csv`
- Threshold sensitivity: `/home/oyavuz/PigBench/Week6_Unibo_Dataset_Validation/outputs/dataset_statistics/week6_bbox_warning_threshold_sensitivity.csv`
- Threshold summary: `/home/oyavuz/PigBench/Week6_Unibo_Dataset_Validation/outputs/dataset_statistics/week6_bbox_warning_threshold_sensitivity_summary.csv`
- Duplicate overlap analysis: `/home/oyavuz/PigBench/Week6_Unibo_Dataset_Validation/outputs/dataset_statistics/week6_bbox_warning_duplicate_overlap_analysis.csv`
- Duplicate overlap summary: `/home/oyavuz/PigBench/Week6_Unibo_Dataset_Validation/outputs/dataset_statistics/week6_bbox_warning_duplicate_overlap_summary.csv`
- Visualizations: `/home/oyavuz/PigBench/Week6_Unibo_Dataset_Validation/outputs/visual_label_check/bbox_warning_threshold_duplicate_analysis`

## Threshold sensitivity summary

| warning_type    |   threshold |   frame_count |   mean_bbox_count |   min_bbox_count |   max_bbox_count |   frames_lt_6 |   frames_gt_10 |
|:----------------|------------:|--------------:|------------------:|-----------------:|-----------------:|--------------:|---------------:|
| high_bbox_count |        0.1  |             4 |             14    |               13 |               15 |             0 |              4 |
| high_bbox_count |        0.15 |             4 |             13    |               12 |               14 |             0 |              4 |
| high_bbox_count |        0.25 |             4 |             12.75 |               12 |               14 |             0 |              4 |
| high_bbox_count |        0.35 |             4 |             10    |                7 |               13 |             0 |              2 |
| high_bbox_count |        0.5  |             4 |              7.5  |                4 |               12 |             1 |              1 |
| high_bbox_count |        0.7  |             4 |              6.5  |                4 |                9 |             1 |              0 |
| low_bbox_count  |        0.1  |            10 |              6.5  |                4 |               11 |             3 |              1 |
| low_bbox_count  |        0.15 |            10 |              5.9  |                4 |               10 |             5 |              0 |
| low_bbox_count  |        0.25 |            10 |              4.9  |                4 |                5 |            10 |              0 |
| low_bbox_count  |        0.35 |            10 |              4.7  |                4 |                5 |            10 |              0 |
| low_bbox_count  |        0.5  |            10 |              4.6  |                3 |                5 |            10 |              0 |
| low_bbox_count  |        0.7  |            10 |              3.9  |                3 |                5 |            10 |              0 |

## Duplicate overlap summary

| warning_type    |   threshold |   frame_count |   total_duplicate_pairs_iou_ge_0_50 |   total_duplicate_pairs_iou_ge_0_70 |   mean_max_pairwise_iou |   max_pairwise_iou |
|:----------------|------------:|--------------:|------------------------------------:|------------------------------------:|------------------------:|-------------------:|
| high_bbox_count |        0.25 |             4 |                                   9 |                                   0 |                0.671792 |           0.684947 |
| high_bbox_count |        0.5  |             4 |                                   0 |                                   0 |                0.139727 |           0.293664 |
| low_bbox_count  |        0.25 |            10 |                                   0 |                                   0 |                0.213089 |           0.414517 |
| low_bbox_count  |        0.5  |            10 |                                   0 |                                   0 |                0.191186 |           0.414517 |

## Interpretation guide

- If low bbox frames reach six boxes only at very low thresholds, extra boxes may be low-confidence and should not automatically be accepted.
- If high bbox frames remain high even at 0.50 or 0.70, the issue may be duplicate/false-positive detections or multiple body-part boxes.
- If many IoU>=0.50 duplicate pairs exist, post-filtering/NMS may help.
- If duplicate overlap is low, high counts may come from distinct false positives rather than simple duplicates.
