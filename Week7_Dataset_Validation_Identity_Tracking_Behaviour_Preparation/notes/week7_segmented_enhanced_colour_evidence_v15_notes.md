# Week 7 Segmented Enhanced Colour Evidence v15

## Purpose

This step improves colour marker evidence by using segmentation masks and contrast/saturation enhancement. It is designed to reduce background false positives and reveal weak pig marker colours.

## Method

- Use SAM box-prompt masks when available; otherwise fallback to GrabCut/box-fill.
- Keep only pig-mask pixels inside each final corrected GT-pen box.
- Generate masked crop, LAB CLAHE crop, HSV saturation crop, and combined enhancement crop.
- Search only valid marker colours: blue, green, cyan, red, pink, purple.
- Ignore orange/yellow as invalid marker colours.

## Summary

- Total boxes processed: `429`
- High confidence evidence: `277`
- Medium confidence evidence: `53`
- Low confidence evidence: `99`
- Boxes recommended for review: `99`
- Frames recommended for review: `56`
- Segmentation method: `sam_vit_b_cpu`

## Outputs

- Box/mask QA: `/home/oyavuz/PigBench/Week7_Dataset_Validation_Identity_Tracking_Behaviour_Preparation/outputs/colour_identity/segmented_enhanced_colour_evidence_v15/week7_segmented_enhanced_colour_evidence_v15_box_mask_qa.csv`
- Colour scores: `/home/oyavuz/PigBench/Week7_Dataset_Validation_Identity_Tracking_Behaviour_Preparation/outputs/colour_identity/segmented_enhanced_colour_evidence_v15/week7_segmented_enhanced_colour_evidence_v15_colour_scores.csv`
- Marker candidates: `/home/oyavuz/PigBench/Week7_Dataset_Validation_Identity_Tracking_Behaviour_Preparation/outputs/colour_identity/segmented_enhanced_colour_evidence_v15/week7_segmented_enhanced_colour_evidence_v15_marker_candidates.csv`
- Frame summary: `/home/oyavuz/PigBench/Week7_Dataset_Validation_Identity_Tracking_Behaviour_Preparation/outputs/colour_identity/segmented_enhanced_colour_evidence_v15/week7_segmented_enhanced_colour_evidence_v15_frame_summary.csv`
- Low-confidence review table: `/home/oyavuz/PigBench/Week7_Dataset_Validation_Identity_Tracking_Behaviour_Preparation/outputs/colour_identity/segmented_enhanced_colour_evidence_v15/week7_segmented_enhanced_colour_evidence_v15_low_confidence_review.csv`
- Static HTML: `/home/oyavuz/PigBench/Week7_Dataset_Validation_Identity_Tracking_Behaviour_Preparation/outputs/colour_identity/segmented_enhanced_colour_evidence_v15/week7_segmented_enhanced_colour_evidence_v15_static_review.html`
