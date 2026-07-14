# Week 7 Final Corrected GT-Pen Boxes v11

## Purpose

This step validates the manually corrected GT-pen pig boxes from the custom v10 editor. These boxes replace the unreliable automatic ROI/detector selection for downstream colour matching.

## Summary

- Total scanpoint frames: `72`
- Complete frames with six kept boxes: `69`
- Incomplete frames: `3`
- Total final corrected boxes: `429`
- Total missing vs six-per-frame expectation: `3`
- Incomplete frame IDs: `scanframe_0026, scanframe_0053, scanframe_0056`

## Interpretation

Frames with fewer than six boxes are accepted only when the missing pig is not visible or not reliably annotatable. It is safer to keep five true GT-pen boxes than to add a wrong-pen or hallucinated box.

## Main output for colour matching

- `/home/oyavuz/PigBench/Week7_Dataset_Validation_Identity_Tracking_Behaviour_Preparation/outputs/final_corrected_gt_pen_boxes_v11/week7_final_corrected_gt_pen_boxes_v11_for_colour_matching.csv`

## QA outputs

- Frame QA summary: `/home/oyavuz/PigBench/Week7_Dataset_Validation_Identity_Tracking_Behaviour_Preparation/outputs/final_corrected_gt_pen_boxes_v11/week7_final_corrected_gt_pen_boxes_v11_frame_qa_summary.csv`
- Decision audit: `/home/oyavuz/PigBench/Week7_Dataset_Validation_Identity_Tracking_Behaviour_Preparation/outputs/final_corrected_gt_pen_boxes_v11/week7_final_corrected_gt_pen_boxes_v11_decision_audit.csv`
- All frame contact sheet: `/home/oyavuz/PigBench/Week7_Dataset_Validation_Identity_Tracking_Behaviour_Preparation/outputs/final_corrected_gt_pen_boxes_v11/week7_final_corrected_gt_pen_boxes_v11_all_frames_contact_sheet.jpg`
- Incomplete frame contact sheet: `/home/oyavuz/PigBench/Week7_Dataset_Validation_Identity_Tracking_Behaviour_Preparation/outputs/final_corrected_gt_pen_boxes_v11/week7_final_corrected_gt_pen_boxes_v11_incomplete_frames_contact_sheet.jpg`
- Static QA HTML: `/home/oyavuz/PigBench/Week7_Dataset_Validation_Identity_Tracking_Behaviour_Preparation/outputs/final_corrected_gt_pen_boxes_v11/week7_final_corrected_gt_pen_boxes_v11_static_qa.html`
