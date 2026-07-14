# Week 7 GT Pen Selection v5 Confidence Review Package

## Purpose

This step does not blindly finalize the v4 automatic selection. Instead, it separates clean selected detections from selected detections that need review and ignored detections that remain suspicious.

## Summary

- Total detections: `540`
- Clean selected detections: `354`
- Selected detections needing review: `67`
- Ignored detections: `98`
- Ignored high-risk detections: `21`
- Frames needing review: `54` out of `72`

## Interpretation

For strict colour matching, use only `clean_selected` detections. For exploratory matching, `selected_needs_review` can be included but should be visually checked. Wrong-pen detections should not be silently used for behaviour-label association.

## Visual review files

- All review contact sheet: `/home/oyavuz/PigBench/Week7_Dataset_Validation_Identity_Tracking_Behaviour_Preparation/outputs/visualizations/week7_gt_pen_selection_v5_all_review_contact_sheet.jpg`
- Risky frames contact sheet: `/home/oyavuz/PigBench/Week7_Dataset_Validation_Identity_Tracking_Behaviour_Preparation/outputs/visualizations/week7_gt_pen_selection_v5_risky_frames_contact_sheet.jpg`
- Static review HTML: `/home/oyavuz/PigBench/Week7_Dataset_Validation_Identity_Tracking_Behaviour_Preparation/outputs/visualizations/week7_gt_pen_selection_v5_static_review.html`
- Individual frame review directory: `/home/oyavuz/PigBench/Week7_Dataset_Validation_Identity_Tracking_Behaviour_Preparation/outputs/visualizations/gt_pen_selection_v5_review_frames`

## Outputs

- `/home/oyavuz/PigBench/Week7_Dataset_Validation_Identity_Tracking_Behaviour_Preparation/outputs/roi_and_crate_mapping/week7_gt_pen_detection_selection_v5_confidence_review.csv`
- `/home/oyavuz/PigBench/Week7_Dataset_Validation_Identity_Tracking_Behaviour_Preparation/outputs/roi_and_crate_mapping/week7_gt_pen_detection_selection_v5_frame_summary.csv`
- `/home/oyavuz/PigBench/Week7_Dataset_Validation_Identity_Tracking_Behaviour_Preparation/outputs/roi_and_crate_mapping/week7_gt_pen_detection_selection_v5_risky_detections.csv`
- `/home/oyavuz/PigBench/Week7_Dataset_Validation_Identity_Tracking_Behaviour_Preparation/outputs/roi_and_crate_mapping/week7_gt_pen_detection_selection_v5_manual_override_template.csv`
- `/home/oyavuz/PigBench/Week7_Dataset_Validation_Identity_Tracking_Behaviour_Preparation/outputs/visualizations/week7_gt_pen_selection_v5_all_review_contact_sheet.jpg`
- `/home/oyavuz/PigBench/Week7_Dataset_Validation_Identity_Tracking_Behaviour_Preparation/outputs/visualizations/week7_gt_pen_selection_v5_risky_frames_contact_sheet.jpg`
- `/home/oyavuz/PigBench/Week7_Dataset_Validation_Identity_Tracking_Behaviour_Preparation/outputs/visualizations/week7_gt_pen_selection_v5_static_review.html`
