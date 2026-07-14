# Week 7 Semantic Ground Truth Pen Region of Interest v1

## Purpose

This step records the annotated Region of Interest as a semantic ground truth pen decision. The user confirmed that the Region of Interest should correspond to the upper/right pen where the visible annotated pigs are located. Two coordinate candidates are saved for review: a broad recommended candidate and a stricter upper candidate.

## Region of Interest definitions

| roi_name                     | roi_type   | points_json                                                                  |   frame_width |   frame_height | source                                         | interpretation                                                                                                              | human_confirmation_status                         |   area_pixels |   area_fraction_of_frame |
|:-----------------------------|:-----------|:-----------------------------------------------------------------------------|--------------:|---------------:|:-----------------------------------------------|:----------------------------------------------------------------------------------------------------------------------------|:--------------------------------------------------|--------------:|-------------------------:|
| broad_semantic_gt_pen_roi_v1 | polygon    | [[115, 0], [704, 0], [704, 576], [85, 576], [85, 420], [135, 315], [135, 0]] |           704 |            576 | human semantic confirmation from contact sheet | Recommended initial ROI because it covers the upper/right annotated pen while retaining pigs visible lower in the same pen. | confirmed_semantic_region_needs_coordinate_review |        338169 |                 0.833947 |
| strict_upper_gt_pen_roi_v1   | polygon    | [[140, 0], [704, 0], [704, 355], [170, 355], [170, 0]]                       |           704 |            576 | strict upper/right candidate                   | More aggressive ROI for testing; may exclude pigs if the annotated pen extends downward.                                    | candidate_for_comparison                          |        189570 |                 0.467492 |

## Detection assignment summary

| roi_name                     |   inside_count |   outside_count |   inside_fraction |
|:-----------------------------|---------------:|----------------:|------------------:|
| broad_semantic_gt_pen_roi_v1 |            505 |              35 |          0.935185 |
| strict_upper_gt_pen_roi_v1   |            299 |             241 |          0.553704 |

## Visual review files

- Broad candidate contact sheet: `/home/oyavuz/PigBench/Week7_Dataset_Validation_Identity_Tracking_Behaviour_Preparation/outputs/visualizations/week7_semantic_gt_pen_roi_v1_broad_contact_sheet.jpg`
- Strict candidate contact sheet: `/home/oyavuz/PigBench/Week7_Dataset_Validation_Identity_Tracking_Behaviour_Preparation/outputs/visualizations/week7_semantic_gt_pen_roi_v1_strict_contact_sheet.jpg`
- Comparison contact sheet: `/home/oyavuz/PigBench/Week7_Dataset_Validation_Identity_Tracking_Behaviour_Preparation/outputs/visualizations/week7_semantic_gt_pen_roi_v1_comparison_contact_sheet.jpg`

## Interpretation

The broad semantic Region of Interest is the recommended initial candidate because it keeps the upper/right pen while reducing influence from adjacent or outside areas. However, the coordinates should still be visually confirmed before colour identity matching and ground truth association are finalized.

## Outputs

- `/home/oyavuz/PigBench/Week7_Dataset_Validation_Identity_Tracking_Behaviour_Preparation/outputs/roi_and_crate_mapping/week7_semantic_gt_pen_roi_v1_definitions.csv`
- `/home/oyavuz/PigBench/Week7_Dataset_Validation_Identity_Tracking_Behaviour_Preparation/outputs/roi_and_crate_mapping/week7_semantic_gt_pen_roi_v1_definitions.json`
- `/home/oyavuz/PigBench/Week7_Dataset_Validation_Identity_Tracking_Behaviour_Preparation/outputs/roi_and_crate_mapping/week7_bbox_roi_assignment_semantic_gt_pen_v1.csv`
- `/home/oyavuz/PigBench/Week7_Dataset_Validation_Identity_Tracking_Behaviour_Preparation/outputs/roi_and_crate_mapping/week7_roi_frame_summary_semantic_gt_pen_v1.csv`
- `/home/oyavuz/PigBench/Week7_Dataset_Validation_Identity_Tracking_Behaviour_Preparation/outputs/roi_and_crate_mapping/week7_semantic_roi_v1_manual_review_template.csv`
- `/home/oyavuz/PigBench/Week7_Dataset_Validation_Identity_Tracking_Behaviour_Preparation/outputs/visualizations/week7_semantic_gt_pen_roi_v1_broad_contact_sheet.jpg`
- `/home/oyavuz/PigBench/Week7_Dataset_Validation_Identity_Tracking_Behaviour_Preparation/outputs/visualizations/week7_semantic_gt_pen_roi_v1_strict_contact_sheet.jpg`
- `/home/oyavuz/PigBench/Week7_Dataset_Validation_Identity_Tracking_Behaviour_Preparation/outputs/visualizations/week7_semantic_gt_pen_roi_v1_comparison_contact_sheet.jpg`
