# Week 7 Ground Truth Pen Detection Selection v4

## Reason for v4

Adaptive ROI v3 improved the Region of Interest but still selected some detections from the wrong pen. The problem is not only geometric Region of Interest selection; the task needs frame-level selection of detections that correspond to the six manually annotated pigs.

## Method

For each scanpoint frame, detections are scored using Segment Anything Model mask overlap with the adaptive upper-pen Region of Interest, detector confidence, and relative vertical position. A duplicate filtering step similar to non-maximum suppression is used, and up to six detections are selected for ground-truth colour and behaviour matching.

## Summary

- Total detections: `540`
- Selected for ground-truth pen matching: `421`
- Ignored detections: `119`
- Frames with exactly six selected detections: `62`
- Frames with fewer than six selected detections: `10`
- Frames with more than six selected detections: `0`

## Important interpretation

This file should be used for colour and behaviour matching instead of raw Region of Interest membership alone. The selected detections are the best automatic estimate of which detector boxes belong to the annotated ground-truth pen. Manual visual review is still required before treating this as final.

## Outputs

- `/home/oyavuz/PigBench/Week7_Dataset_Validation_Identity_Tracking_Behaviour_Preparation/outputs/roi_and_crate_mapping/week7_gt_pen_detection_selection_v4.csv`
- `/home/oyavuz/PigBench/Week7_Dataset_Validation_Identity_Tracking_Behaviour_Preparation/outputs/roi_and_crate_mapping/week7_gt_pen_detection_selection_v4_frame_summary.csv`
- `/home/oyavuz/PigBench/Week7_Dataset_Validation_Identity_Tracking_Behaviour_Preparation/outputs/roi_and_crate_mapping/week7_gt_pen_detection_selection_v4_manual_review_template.csv`
- `/home/oyavuz/PigBench/Week7_Dataset_Validation_Identity_Tracking_Behaviour_Preparation/outputs/visualizations/week7_gt_pen_detection_selection_v4_contact_sheet.jpg`
- `/home/oyavuz/PigBench/Week7_Dataset_Validation_Identity_Tracking_Behaviour_Preparation/outputs/visualizations/week7_gt_pen_detection_selection_v4_problem_frames_contact_sheet.jpg`
