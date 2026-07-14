# Week 7 Adaptive Upper-Pen Region of Interest v3 with Segment Anything Model Mask Overlap

## Reason for v3

The ground truth pen is not always located at a fixed top-right screen position. The correct generalization is to focus on the upper annotated pen region and to use mask-overlap rather than only bounding-box centre checks. This avoids including lower adjacent-pen pigs when their bounding boxes cross candidate boundaries.

## Method

Multiple upper-pen Region of Interest candidates were generated. For each detection, the Segment Anything Model mask was intersected with each candidate Region of Interest. A detection is treated as inside a candidate when at least 55 percent of its mask area lies inside the candidate. For each scanpoint frame, a candidate is selected using a score that favours about six inside pigs while penalizing over-selection.

## Candidate global summary

| roi_name                         |   inside_count |   outside_count |   inside_fraction |   mean_inside_per_frame |
|:---------------------------------|---------------:|----------------:|------------------:|------------------------:|
| upper_pen_y360_v3                |            338 |             202 |          0.625926 |                 4.69444 |
| upper_pen_fence_diagonal_y390_v3 |            374 |             166 |          0.692593 |                 5.19444 |
| upper_pen_y390_v3                |            386 |             154 |          0.714815 |                 5.36111 |
| upper_pen_fence_diagonal_y420_v3 |            397 |             143 |          0.735185 |                 5.51389 |
| upper_pen_fence_diagonal_y450_v3 |            411 |             129 |          0.761111 |                 5.70833 |
| upper_pen_y420_v3                |            415 |             125 |          0.768519 |                 5.76389 |

## Adaptive selected summary

- Total detections: `540`
- Inside selected adaptive Region of Interest: `400`
- Outside selected adaptive Region of Interest: `140`
- Mean inside per frame: `5.556`
- Median inside per frame: `5.500`

## Visual review files

- Selected adaptive contact sheet: `/home/oyavuz/PigBench/Week7_Dataset_Validation_Identity_Tracking_Behaviour_Preparation/outputs/visualizations/week7_adaptive_upper_pen_roi_v3_selected_contact_sheet.jpg`
- Candidate review sheet: `/home/oyavuz/PigBench/Week7_Dataset_Validation_Identity_Tracking_Behaviour_Preparation/outputs/visualizations/week7_adaptive_upper_pen_roi_v3_candidate_review_sheet.jpg`
- Candidate contact index: `/home/oyavuz/PigBench/Week7_Dataset_Validation_Identity_Tracking_Behaviour_Preparation/outputs/visualizations/week7_adaptive_upper_pen_roi_v3_contact_sheet_index.csv`

## Important interpretation

This v3 output should be visually reviewed before it is finalized. If it performs better than fixed v1/v2 polygons, it should supersede previous Region of Interest decisions.

## Outputs

- `/home/oyavuz/PigBench/Week7_Dataset_Validation_Identity_Tracking_Behaviour_Preparation/outputs/roi_and_crate_mapping/week7_adaptive_upper_pen_roi_v3_candidate_definitions.csv`
- `/home/oyavuz/PigBench/Week7_Dataset_Validation_Identity_Tracking_Behaviour_Preparation/outputs/roi_and_crate_mapping/week7_adaptive_upper_pen_roi_v3_candidate_definitions.json`
- `/home/oyavuz/PigBench/Week7_Dataset_Validation_Identity_Tracking_Behaviour_Preparation/outputs/roi_and_crate_mapping/week7_adaptive_upper_pen_roi_v3_mask_overlap_per_detection.csv`
- `/home/oyavuz/PigBench/Week7_Dataset_Validation_Identity_Tracking_Behaviour_Preparation/outputs/roi_and_crate_mapping/week7_adaptive_upper_pen_roi_v3_frame_selection.csv`
- `/home/oyavuz/PigBench/Week7_Dataset_Validation_Identity_Tracking_Behaviour_Preparation/outputs/roi_and_crate_mapping/week7_bbox_roi_assignment_adaptive_upper_pen_v3.csv`
- `/home/oyavuz/PigBench/Week7_Dataset_Validation_Identity_Tracking_Behaviour_Preparation/outputs/roi_and_crate_mapping/week7_roi_frame_summary_adaptive_upper_pen_v3.csv`
- `/home/oyavuz/PigBench/Week7_Dataset_Validation_Identity_Tracking_Behaviour_Preparation/outputs/roi_and_crate_mapping/week7_adaptive_upper_pen_roi_v3_manual_review_template.csv`
- `/home/oyavuz/PigBench/Week7_Dataset_Validation_Identity_Tracking_Behaviour_Preparation/outputs/visualizations/week7_adaptive_upper_pen_roi_v3_selected_contact_sheet.jpg`
- `/home/oyavuz/PigBench/Week7_Dataset_Validation_Identity_Tracking_Behaviour_Preparation/outputs/visualizations/week7_adaptive_upper_pen_roi_v3_candidate_review_sheet.jpg`
