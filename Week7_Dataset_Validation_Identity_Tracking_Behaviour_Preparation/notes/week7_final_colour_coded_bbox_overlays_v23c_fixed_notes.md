# Week 7 Final Colour-coded BBox Overlays v23c Fixed

## Purpose

This fixed step redraws final corrected bounding boxes using final visual marker colours. Unlike v23b, this version uses Week6 behaviour-label image paths as a fallback for frames where all identities are unknown/not_visible.

## Colour policy

- blue boxes: visual marker colour `blue`
- green boxes: visual marker colour `green`
- cyan boxes: visual marker colour `cyan`, mapped to behaviour pig ID `no_color`
- red boxes: visual marker colour `red`, mapped to behaviour pig ID `red_neck`
- pink/magenta boxes: visual marker colour `pink`, mapped to behaviour pig ID `red_tail`
- purple boxes: visual marker colour `purple`
- grey boxes: unknown / not_visible / uncertain identity

## Summary

- Frames processed: `72`
- Total box rows: `429`
- Drawn boxes: `429`
- Valid colour boxes: `374`
- Unknown colour boxes: `55`
- Issue count: `0`
- Ready for report visuals: `True`

## Outputs

- Frame overlays: `/home/oyavuz/PigBench/Week7_Dataset_Validation_Identity_Tracking_Behaviour_Preparation/outputs/final_colour_coded_bbox_overlays_v23c_fixed/frame_overlays`
- Contact sheets: `/home/oyavuz/PigBench/Week7_Dataset_Validation_Identity_Tracking_Behaviour_Preparation/outputs/final_colour_coded_bbox_overlays_v23c_fixed/contact_sheets`
- Overlay index: `/home/oyavuz/PigBench/Week7_Dataset_Validation_Identity_Tracking_Behaviour_Preparation/outputs/final_colour_coded_bbox_overlays_v23c_fixed/week7_final_colour_coded_bbox_overlays_v23c_index.csv`
- Frame QA: `/home/oyavuz/PigBench/Week7_Dataset_Validation_Identity_Tracking_Behaviour_Preparation/outputs/final_colour_coded_bbox_overlays_v23c_fixed/week7_final_colour_coded_bbox_overlays_v23c_frame_qa.csv`
- Contact sheet index: `/home/oyavuz/PigBench/Week7_Dataset_Validation_Identity_Tracking_Behaviour_Preparation/outputs/final_colour_coded_bbox_overlays_v23c_fixed/week7_final_colour_coded_bbox_overlays_v23c_contact_sheet_index.csv`
- Issues: `/home/oyavuz/PigBench/Week7_Dataset_Validation_Identity_Tracking_Behaviour_Preparation/outputs/final_colour_coded_bbox_overlays_v23c_fixed/week7_final_colour_coded_bbox_overlays_v23c_issues.csv`
- Summary: `/home/oyavuz/PigBench/Week7_Dataset_Validation_Identity_Tracking_Behaviour_Preparation/outputs/final_colour_coded_bbox_overlays_v23c_fixed/week7_final_colour_coded_bbox_overlays_v23c_summary.csv`
