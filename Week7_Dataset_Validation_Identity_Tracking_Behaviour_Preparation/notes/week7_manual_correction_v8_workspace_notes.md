# Week 7 Manual Correction Workspace v8

## Purpose

The automatic ROI/detection pipeline was not reliable enough to finalize colour/behaviour matching. Some wrong-pen pigs were still marked as safe, while some correct GT-pen pigs were missed by the detector. Therefore, this workspace supports a human-in-the-loop correction step.

## Correction rule

Final colour matching must use the manually corrected GT-pen box set, not raw detector output. A candidate is kept only if visual review confirms that it belongs to the annotated GT pen. If a true GT-pen pig is missing, a manual bbox should be added.

## Summary

- Candidate decision rows: `618`
- Strict auto-safe candidates to verify: `320`
- Manual review candidates: `200`
- New low-threshold recall candidates: `78`
- Manual add rows prepared: `103`
- High-priority frames with missing slots: `49`

## Manual workflow

1. Open the static HTML review page.
2. For each frame, inspect whether each candidate belongs to the annotated GT pen.
3. Fill `week7_manual_correction_v8_candidate_decisions_template.csv`.
4. If a correct GT-pen pig is missing, fill `week7_manual_correction_v8_manual_add_boxes_template.csv`.
5. After manual edits, run the next script to compile final corrected boxes.

## Outputs

- `/home/oyavuz/PigBench/Week7_Dataset_Validation_Identity_Tracking_Behaviour_Preparation/outputs/roi_and_crate_mapping/week7_manual_correction_v8_candidate_decisions_template.csv`
- `/home/oyavuz/PigBench/Week7_Dataset_Validation_Identity_Tracking_Behaviour_Preparation/outputs/roi_and_crate_mapping/week7_manual_correction_v8_manual_add_boxes_template.csv`
- `/home/oyavuz/PigBench/Week7_Dataset_Validation_Identity_Tracking_Behaviour_Preparation/outputs/roi_and_crate_mapping/week7_manual_correction_v8_frame_checklist.csv`
- `/home/oyavuz/PigBench/Week7_Dataset_Validation_Identity_Tracking_Behaviour_Preparation/outputs/roi_and_crate_mapping/week7_manual_correction_v8_final_corrected_boxes_EMPTY_TEMPLATE.csv`
- `/home/oyavuz/PigBench/Week7_Dataset_Validation_Identity_Tracking_Behaviour_Preparation/outputs/visualizations/week7_manual_correction_v8_review_image_index.csv`
- `/home/oyavuz/PigBench/Week7_Dataset_Validation_Identity_Tracking_Behaviour_Preparation/outputs/visualizations/week7_manual_correction_v8_all_frames_contact_sheet.jpg`
- `/home/oyavuz/PigBench/Week7_Dataset_Validation_Identity_Tracking_Behaviour_Preparation/outputs/visualizations/week7_manual_correction_v8_priority_review_contact_sheet.jpg`
- `/home/oyavuz/PigBench/Week7_Dataset_Validation_Identity_Tracking_Behaviour_Preparation/outputs/visualizations/week7_manual_correction_v8_static_review.html`
