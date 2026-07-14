# Week 7 Behaviour Label Fusion v18

## Purpose

This step fuses final corrected pig boxes, final locked colour identity, and original behaviour labels into the main `Pig → Colour Marker → Behaviour Label` dataset.

## Matching rule

Known colour identities are matched to behaviour labels by `scan_frame_id + canonical colour`. `not_visible` and `uncertain` identities are retained in the box-level dataset but are not used as safe behaviour matches.

## Summary

- Total corrected boxes: `429`
- Usable colour identity boxes: `374`
- Unknown identity boxes: `55`
- Matched behaviour boxes: `185`
- Ready training rows: `185`
- Unmatched behaviour labels: `31`
- Duplicate label keys detected: `0`

## Outputs

- Fused ready dataset: `/home/oyavuz/PigBench/Week7_Dataset_Validation_Identity_Tracking_Behaviour_Preparation/outputs/behaviour_label_fusion_v18/week7_behaviour_label_fusion_v18_fused_pig_colour_behaviour.csv`
- Box-level dataset: `/home/oyavuz/PigBench/Week7_Dataset_Validation_Identity_Tracking_Behaviour_Preparation/outputs/behaviour_label_fusion_v18/week7_behaviour_label_fusion_v18_box_level_dataset.csv`
- Unmatched identity boxes: `/home/oyavuz/PigBench/Week7_Dataset_Validation_Identity_Tracking_Behaviour_Preparation/outputs/behaviour_label_fusion_v18/week7_behaviour_label_fusion_v18_unmatched_identity_boxes.csv`
- Unmatched behaviour labels: `/home/oyavuz/PigBench/Week7_Dataset_Validation_Identity_Tracking_Behaviour_Preparation/outputs/behaviour_label_fusion_v18/week7_behaviour_label_fusion_v18_unmatched_behaviour_labels.csv`
- Frame summary: `/home/oyavuz/PigBench/Week7_Dataset_Validation_Identity_Tracking_Behaviour_Preparation/outputs/behaviour_label_fusion_v18/week7_behaviour_label_fusion_v18_frame_summary.csv`
- Behaviour distribution: `/home/oyavuz/PigBench/Week7_Dataset_Validation_Identity_Tracking_Behaviour_Preparation/outputs/behaviour_label_fusion_v18/week7_behaviour_label_fusion_v18_behaviour_distribution.csv`
- Colour/status distribution: `/home/oyavuz/PigBench/Week7_Dataset_Validation_Identity_Tracking_Behaviour_Preparation/outputs/behaviour_label_fusion_v18/week7_behaviour_label_fusion_v18_colour_status_distribution.csv`
- Summary: `/home/oyavuz/PigBench/Week7_Dataset_Validation_Identity_Tracking_Behaviour_Preparation/outputs/behaviour_label_fusion_v18/week7_behaviour_label_fusion_v18_summary.csv`
