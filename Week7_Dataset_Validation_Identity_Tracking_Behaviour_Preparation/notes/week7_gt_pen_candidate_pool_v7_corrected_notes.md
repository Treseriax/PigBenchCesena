# Week 7 GT Pen Candidate Pool v7 Corrected

## Purpose

This step merges v6 strict/review detections with low-threshold detector recall candidates. New low-threshold detections are not accepted automatically; they are added as manual recall candidates only.

## Key rule

Only `v7_candidate_class = strict_auto_safe` can be used automatically for colour matching. `existing_selected_review_required`, `existing_ignored_high_risk_review`, and `new_low_threshold_recall_candidate` require manual confirmation.

## Summary

- Total candidate pool rows: `618`
- Strict automatic safe detections: `320`
- Manual review required candidates: `200`
- New low-threshold recall candidates included: `78`
- Priority new recall candidates: `38`
- Frames requiring manual review: `68` out of `72`
- Frames still missing slots after strict + priority new candidates: `49`

## Interpretation

Low-threshold detection expands recall but introduces false positives. Therefore v7 is a corrected candidate pool, not a final automatic label-matching table. The next step is manual confirmation of review candidates or generation of missing manual boxes where the detector still fails.

## Outputs

- `/home/oyavuz/PigBench/Week7_Dataset_Validation_Identity_Tracking_Behaviour_Preparation/outputs/roi_and_crate_mapping/week7_gt_pen_candidate_pool_v7_corrected.csv`
- `/home/oyavuz/PigBench/Week7_Dataset_Validation_Identity_Tracking_Behaviour_Preparation/outputs/roi_and_crate_mapping/week7_gt_pen_candidate_pool_v7_strict_auto_safe.csv`
- `/home/oyavuz/PigBench/Week7_Dataset_Validation_Identity_Tracking_Behaviour_Preparation/outputs/roi_and_crate_mapping/week7_gt_pen_candidate_pool_v7_manual_review_required.csv`
- `/home/oyavuz/PigBench/Week7_Dataset_Validation_Identity_Tracking_Behaviour_Preparation/outputs/roi_and_crate_mapping/week7_gt_pen_candidate_pool_v7_priority_new_recall_candidates.csv`
- `/home/oyavuz/PigBench/Week7_Dataset_Validation_Identity_Tracking_Behaviour_Preparation/outputs/roi_and_crate_mapping/week7_gt_pen_candidate_pool_v7_frame_summary.csv`
- `/home/oyavuz/PigBench/Week7_Dataset_Validation_Identity_Tracking_Behaviour_Preparation/outputs/roi_and_crate_mapping/week7_gt_pen_candidate_pool_v7_manual_override_template.csv`
- `/home/oyavuz/PigBench/Week7_Dataset_Validation_Identity_Tracking_Behaviour_Preparation/outputs/visualizations/week7_gt_pen_candidate_pool_v7_all_frames_contact_sheet.jpg`
- `/home/oyavuz/PigBench/Week7_Dataset_Validation_Identity_Tracking_Behaviour_Preparation/outputs/visualizations/week7_gt_pen_candidate_pool_v7_review_required_contact_sheet.jpg`
- `/home/oyavuz/PigBench/Week7_Dataset_Validation_Identity_Tracking_Behaviour_Preparation/outputs/visualizations/week7_gt_pen_candidate_pool_v7_static_review.html`
