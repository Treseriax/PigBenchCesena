# Week 7 GT Pen Selection v6 Final-Safe Layer

## Purpose

This step creates a final-safe automatic subset for colour matching and separates all uncertain detections into a manual review layer. The purpose is to avoid silently using wrong-pen detections for behaviour-label association.

## Main rule

Use only `strict_auto_safe_for_colour_matching_v6 = True` for automatic colour matching.

## Summary

- Total detections: `540`
- Strict automatic safe detections: `320`
- Selected detections requiring review: `101`
- Ignored high-risk detections requiring review: `21`
- Ignored detections: `98`
- Frames requiring manual review: `64` out of `72`

## Interpretation

The strict automatic subset may contain fewer than six detections per frame. This is intentional: it is safer to leave a pig slot unresolved than to assign a wrong-pen detection to a behaviour label. Manual overrides can later add or reject detections using the manual override template.

## Outputs

- `/home/oyavuz/PigBench/Week7_Dataset_Validation_Identity_Tracking_Behaviour_Preparation/outputs/roi_and_crate_mapping/week7_gt_pen_detection_selection_v6_final_safe_layer.csv`
- `/home/oyavuz/PigBench/Week7_Dataset_Validation_Identity_Tracking_Behaviour_Preparation/outputs/roi_and_crate_mapping/week7_gt_pen_detection_selection_v6_strict_auto_safe_for_colour_matching.csv`
- `/home/oyavuz/PigBench/Week7_Dataset_Validation_Identity_Tracking_Behaviour_Preparation/outputs/roi_and_crate_mapping/week7_gt_pen_detection_selection_v6_manual_review_required.csv`
- `/home/oyavuz/PigBench/Week7_Dataset_Validation_Identity_Tracking_Behaviour_Preparation/outputs/roi_and_crate_mapping/week7_gt_pen_detection_selection_v6_frame_summary.csv`
- `/home/oyavuz/PigBench/Week7_Dataset_Validation_Identity_Tracking_Behaviour_Preparation/outputs/roi_and_crate_mapping/week7_gt_pen_detection_selection_v6_manual_override_template.csv`
- `/home/oyavuz/PigBench/Week7_Dataset_Validation_Identity_Tracking_Behaviour_Preparation/outputs/visualizations/week7_gt_pen_selection_v6_all_frames_contact_sheet.jpg`
- `/home/oyavuz/PigBench/Week7_Dataset_Validation_Identity_Tracking_Behaviour_Preparation/outputs/visualizations/week7_gt_pen_selection_v6_review_required_contact_sheet.jpg`
- `/home/oyavuz/PigBench/Week7_Dataset_Validation_Identity_Tracking_Behaviour_Preparation/outputs/visualizations/week7_gt_pen_selection_v6_static_review.html`
