# Week 7 Behaviour Label Fusion v18b

## Purpose

This corrected fusion step uses robust colour-name normalization across `colour_id`, `colour_raw`, and `pig_id` before matching behaviour labels to final colour identities.

## Summary

- Total corrected boxes: `429`
- Usable colour identity boxes: `374`
- Unknown identity boxes: `55`
- Total behaviour label rows: `432`
- Valid-colour behaviour label rows: `288`
- Matched behaviour boxes: `246`
- Ready training rows: `246`
- Unmatched behaviour labels: `42`
- Duplicate label keys detected: `72`

## Outputs

- Label colour audit: `/home/oyavuz/PigBench/Week7_Dataset_Validation_Identity_Tracking_Behaviour_Preparation/outputs/behaviour_label_fusion_v18b/week7_behaviour_label_fusion_v18b_label_colour_audit.csv`
- Fused ready dataset: `/home/oyavuz/PigBench/Week7_Dataset_Validation_Identity_Tracking_Behaviour_Preparation/outputs/behaviour_label_fusion_v18b/week7_behaviour_label_fusion_v18b_fused_pig_colour_behaviour.csv`
- Box-level dataset: `/home/oyavuz/PigBench/Week7_Dataset_Validation_Identity_Tracking_Behaviour_Preparation/outputs/behaviour_label_fusion_v18b/week7_behaviour_label_fusion_v18b_box_level_dataset.csv`
- Unmatched identity boxes: `/home/oyavuz/PigBench/Week7_Dataset_Validation_Identity_Tracking_Behaviour_Preparation/outputs/behaviour_label_fusion_v18b/week7_behaviour_label_fusion_v18b_unmatched_identity_boxes.csv`
- Unmatched behaviour labels: `/home/oyavuz/PigBench/Week7_Dataset_Validation_Identity_Tracking_Behaviour_Preparation/outputs/behaviour_label_fusion_v18b/week7_behaviour_label_fusion_v18b_unmatched_behaviour_labels.csv`
- Frame summary: `/home/oyavuz/PigBench/Week7_Dataset_Validation_Identity_Tracking_Behaviour_Preparation/outputs/behaviour_label_fusion_v18b/week7_behaviour_label_fusion_v18b_frame_summary.csv`
- Behaviour distribution: `/home/oyavuz/PigBench/Week7_Dataset_Validation_Identity_Tracking_Behaviour_Preparation/outputs/behaviour_label_fusion_v18b/week7_behaviour_label_fusion_v18b_behaviour_distribution.csv`
- Summary: `/home/oyavuz/PigBench/Week7_Dataset_Validation_Identity_Tracking_Behaviour_Preparation/outputs/behaviour_label_fusion_v18b/week7_behaviour_label_fusion_v18b_summary.csv`
