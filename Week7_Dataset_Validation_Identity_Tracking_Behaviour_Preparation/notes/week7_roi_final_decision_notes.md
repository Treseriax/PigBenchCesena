# Week 7 Final Ground Truth Pen Region of Interest Decision

## Decision

The final Region of Interest is selected as `final_gt_pen_roi_v1`, based on the previous `broad_semantic_gt_pen_roi_v1` candidate.

## Reason

Visual inspection showed that the annotated ground truth pen corresponds to the upper/right pen where the visible annotated pigs are located. The strict upper candidate was rejected because it excluded too many valid pigs from the same pen. The broad candidate better preserves all pigs visible in the ground truth pen while still reducing influence from adjacent or outside areas.

## Detection assignment summary

- Total detections: `540`
- Inside final Region of Interest: `505`
- Outside final Region of Interest: `35`
- Inside fraction: `0.9352`

## Important interpretation

Detections outside the final Region of Interest should not be used for ground-truth colour or behaviour matching unless manually justified. They may correspond to adjacent pens, partial animals, or detections not associated with the annotated ground truth pen.

## Outputs

- `/home/oyavuz/PigBench/Week7_Dataset_Validation_Identity_Tracking_Behaviour_Preparation/outputs/roi_and_crate_mapping/week7_final_gt_pen_roi_definition.csv`
- `/home/oyavuz/PigBench/Week7_Dataset_Validation_Identity_Tracking_Behaviour_Preparation/outputs/roi_and_crate_mapping/week7_final_gt_pen_roi_definition.json`
- `/home/oyavuz/PigBench/Week7_Dataset_Validation_Identity_Tracking_Behaviour_Preparation/outputs/roi_and_crate_mapping/week7_bbox_roi_assignment_final.csv`
- `/home/oyavuz/PigBench/Week7_Dataset_Validation_Identity_Tracking_Behaviour_Preparation/outputs/roi_and_crate_mapping/week7_roi_frame_summary_final.csv`
- `/home/oyavuz/PigBench/Week7_Dataset_Validation_Identity_Tracking_Behaviour_Preparation/outputs/visualizations/week7_final_gt_pen_roi_overlay_contact_sheet.jpg`
