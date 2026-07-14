# Week 7 Final Fusion QA + Dataset Statistics v19

## Purpose

This step audits the v18c fused dataset and locks a clean training-ready table for later split design, tracking, and clip-level representation.

## Inputs

- v18c fused ready dataset: `/home/oyavuz/PigBench/Week7_Dataset_Validation_Identity_Tracking_Behaviour_Preparation/outputs/behaviour_label_fusion_v18c_verified_crosswalk/week7_behaviour_label_fusion_v18c_fused_pig_colour_behaviour.csv`
- v18c box-level dataset: `/home/oyavuz/PigBench/Week7_Dataset_Validation_Identity_Tracking_Behaviour_Preparation/outputs/behaviour_label_fusion_v18c_verified_crosswalk/week7_behaviour_label_fusion_v18c_box_level_dataset.csv`
- v18c summary: `/home/oyavuz/PigBench/Week7_Dataset_Validation_Identity_Tracking_Behaviour_Preparation/outputs/behaviour_label_fusion_v18c_verified_crosswalk/week7_behaviour_label_fusion_v18c_summary.csv`

## QA policy

Hard issues include duplicate training-ready boxes, missing behaviour labels, invalid visual colours, invalid behaviour pig IDs, or duplicate behaviour pig IDs within the same frame. Soft issues are expected exclusions such as `not_visible` or `uncertain` identity.

## Summary

- Training-ready rows: `374`
- Box-level rows: `429`
- Excluded rows: `55`
- Hard issue count: `0`
- Behaviour class count: `11`
- Majority behaviour count: `151`
- Minority behaviour count: `3`
- Frames with training rows: `70` / `72`
- Ready for split design: `True`

## Outputs

- Locked training-ready dataset: `/home/oyavuz/PigBench/Week7_Dataset_Validation_Identity_Tracking_Behaviour_Preparation/outputs/final_fusion_qa_dataset_stats_v19/week7_final_training_ready_dataset_v19_locked.csv`
- Summary: `/home/oyavuz/PigBench/Week7_Dataset_Validation_Identity_Tracking_Behaviour_Preparation/outputs/final_fusion_qa_dataset_stats_v19/week7_final_fusion_qa_v19_summary.csv`
- Behaviour distribution: `/home/oyavuz/PigBench/Week7_Dataset_Validation_Identity_Tracking_Behaviour_Preparation/outputs/final_fusion_qa_dataset_stats_v19/week7_final_fusion_qa_v19_behaviour_distribution.csv`
- Behaviour imbalance report: `/home/oyavuz/PigBench/Week7_Dataset_Validation_Identity_Tracking_Behaviour_Preparation/outputs/final_fusion_qa_dataset_stats_v19/week7_final_fusion_qa_v19_behaviour_imbalance_report.csv`
- Visual colour distribution: `/home/oyavuz/PigBench/Week7_Dataset_Validation_Identity_Tracking_Behaviour_Preparation/outputs/final_fusion_qa_dataset_stats_v19/week7_final_fusion_qa_v19_visual_colour_distribution.csv`
- Behaviour pig ID distribution: `/home/oyavuz/PigBench/Week7_Dataset_Validation_Identity_Tracking_Behaviour_Preparation/outputs/final_fusion_qa_dataset_stats_v19/week7_final_fusion_qa_v19_behaviour_pig_id_distribution.csv`
- Video distribution: `/home/oyavuz/PigBench/Week7_Dataset_Validation_Identity_Tracking_Behaviour_Preparation/outputs/final_fusion_qa_dataset_stats_v19/week7_final_fusion_qa_v19_video_distribution.csv`
- Frame coverage: `/home/oyavuz/PigBench/Week7_Dataset_Validation_Identity_Tracking_Behaviour_Preparation/outputs/final_fusion_qa_dataset_stats_v19/week7_final_fusion_qa_v19_frame_coverage.csv`
- Excluded rows: `/home/oyavuz/PigBench/Week7_Dataset_Validation_Identity_Tracking_Behaviour_Preparation/outputs/final_fusion_qa_dataset_stats_v19/week7_final_fusion_qa_v19_excluded_rows.csv`
- Hard issues: `/home/oyavuz/PigBench/Week7_Dataset_Validation_Identity_Tracking_Behaviour_Preparation/outputs/final_fusion_qa_dataset_stats_v19/week7_final_fusion_qa_v19_hard_issues.csv`
- Soft issues: `/home/oyavuz/PigBench/Week7_Dataset_Validation_Identity_Tracking_Behaviour_Preparation/outputs/final_fusion_qa_dataset_stats_v19/week7_final_fusion_qa_v19_soft_issues.csv`
