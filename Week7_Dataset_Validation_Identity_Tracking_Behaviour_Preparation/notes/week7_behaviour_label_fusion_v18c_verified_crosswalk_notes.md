# Week 7 Behaviour Label Fusion v18c Verified Crosswalk

## Purpose

This step fuses final visual marker colours with behaviour labels using a manually verified crosswalk from visual marker colour to annotation-specific behaviour pig ID.

## Crosswalk

- blue → blue
- green → green
- purple → purple
- red → red_neck / red_head
- pink → red_tail
- cyan → no_color

## Important

The visual marker colour and the behaviour annotation pig ID are stored separately. This preserves the visual evidence layer while allowing correct behaviour-label fusion.

## Summary

- Total corrected boxes: `429`
- Usable visual colour identity boxes: `374`
- Unknown identity boxes: `55`
- Total behaviour label rows: `432`
- Valid behaviour pig ID label rows: `432`
- Matched behaviour boxes: `374`
- Ready training rows: `374`
- Unmatched behaviour labels: `58`
- Duplicate label keys detected: `0`

## Outputs

- Crosswalk: `/home/oyavuz/PigBench/Week7_Dataset_Validation_Identity_Tracking_Behaviour_Preparation/outputs/behaviour_label_fusion_v18c_verified_crosswalk/week7_behaviour_label_fusion_v18c_visual_to_behaviour_pig_id_crosswalk.csv`
- Fused ready dataset: `/home/oyavuz/PigBench/Week7_Dataset_Validation_Identity_Tracking_Behaviour_Preparation/outputs/behaviour_label_fusion_v18c_verified_crosswalk/week7_behaviour_label_fusion_v18c_fused_pig_colour_behaviour.csv`
- Box-level dataset: `/home/oyavuz/PigBench/Week7_Dataset_Validation_Identity_Tracking_Behaviour_Preparation/outputs/behaviour_label_fusion_v18c_verified_crosswalk/week7_behaviour_label_fusion_v18c_box_level_dataset.csv`
- Unmatched identity boxes: `/home/oyavuz/PigBench/Week7_Dataset_Validation_Identity_Tracking_Behaviour_Preparation/outputs/behaviour_label_fusion_v18c_verified_crosswalk/week7_behaviour_label_fusion_v18c_unmatched_identity_boxes.csv`
- Unmatched behaviour labels: `/home/oyavuz/PigBench/Week7_Dataset_Validation_Identity_Tracking_Behaviour_Preparation/outputs/behaviour_label_fusion_v18c_verified_crosswalk/week7_behaviour_label_fusion_v18c_unmatched_behaviour_labels.csv`
- Frame summary: `/home/oyavuz/PigBench/Week7_Dataset_Validation_Identity_Tracking_Behaviour_Preparation/outputs/behaviour_label_fusion_v18c_verified_crosswalk/week7_behaviour_label_fusion_v18c_frame_summary.csv`
- Behaviour distribution: `/home/oyavuz/PigBench/Week7_Dataset_Validation_Identity_Tracking_Behaviour_Preparation/outputs/behaviour_label_fusion_v18c_verified_crosswalk/week7_behaviour_label_fusion_v18c_behaviour_distribution.csv`
- Status distribution: `/home/oyavuz/PigBench/Week7_Dataset_Validation_Identity_Tracking_Behaviour_Preparation/outputs/behaviour_label_fusion_v18c_verified_crosswalk/week7_behaviour_label_fusion_v18c_status_distribution.csv`
- Summary: `/home/oyavuz/PigBench/Week7_Dataset_Validation_Identity_Tracking_Behaviour_Preparation/outputs/behaviour_label_fusion_v18c_verified_crosswalk/week7_behaviour_label_fusion_v18c_summary.csv`
