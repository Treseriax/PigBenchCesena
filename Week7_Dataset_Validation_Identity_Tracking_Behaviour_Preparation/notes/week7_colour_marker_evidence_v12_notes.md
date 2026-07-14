# Week 7 Colour Marker Evidence v12

## Purpose

This step extracts crops and candidate colour-marker evidence from the manually corrected GT-pen pig boxes. It prepares the input for colour identity assignment.

## Inputs

- Final corrected boxes: `/home/oyavuz/PigBench/Week7_Dataset_Validation_Identity_Tracking_Behaviour_Preparation/outputs/final_corrected_gt_pen_boxes_v11/week7_final_corrected_gt_pen_boxes_v11_for_colour_matching.csv`

## Outputs

- Per-box features: `/home/oyavuz/PigBench/Week7_Dataset_Validation_Identity_Tracking_Behaviour_Preparation/outputs/colour_identity/colour_marker_evidence_v12/week7_colour_marker_evidence_v12_per_box_features.csv`
- Marker candidates: `/home/oyavuz/PigBench/Week7_Dataset_Validation_Identity_Tracking_Behaviour_Preparation/outputs/colour_identity/colour_marker_evidence_v12/week7_colour_marker_evidence_v12_marker_candidates.csv`
- Frame summary: `/home/oyavuz/PigBench/Week7_Dataset_Validation_Identity_Tracking_Behaviour_Preparation/outputs/colour_identity/colour_marker_evidence_v12/week7_colour_marker_evidence_v12_frame_summary.csv`
- Contact sheet: `/home/oyavuz/PigBench/Week7_Dataset_Validation_Identity_Tracking_Behaviour_Preparation/outputs/colour_identity/colour_marker_evidence_v12/week7_colour_marker_evidence_v12_frame_overlay_contact_sheet.jpg`
- Static HTML: `/home/oyavuz/PigBench/Week7_Dataset_Validation_Identity_Tracking_Behaviour_Preparation/outputs/colour_identity/colour_marker_evidence_v12/week7_colour_marker_evidence_v12_static_review.html`

## Interpretation

The marker candidates are not final labels. They are automatically extracted visual evidence for the next colour matching step.
