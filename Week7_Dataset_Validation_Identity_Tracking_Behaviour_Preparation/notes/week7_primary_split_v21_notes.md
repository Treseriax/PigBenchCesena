# Week 7 Primary Split v21

## Purpose

This step locks the selected primary split from v20. The selected split is `frame_stratified`.

## Rationale

The dataset is small and imbalanced, so the primary objective is preserving behaviour-class coverage while keeping each scan frame intact. `frame_stratified` has no same-frame leakage and provides the best class coverage among the tested candidates. It does have same-video leakage, so it should not be interpreted as a strict video-level generalization split.

## Summary

- Total rows: `374`
- Train rows: `252`
- Validation rows: `63`
- Test rows: `59`
- Same-frame leakage: `0`
- Same-video leakage: `11`
- Validation missing classes: `IA`
- Test missing classes: ``

## Outputs

- All rows: `/home/oyavuz/PigBench/Week7_Dataset_Validation_Identity_Tracking_Behaviour_Preparation/outputs/primary_split_v21/week7_primary_split_v21_all.csv`
- Train: `/home/oyavuz/PigBench/Week7_Dataset_Validation_Identity_Tracking_Behaviour_Preparation/outputs/primary_split_v21/week7_primary_split_v21_train.csv`
- Validation: `/home/oyavuz/PigBench/Week7_Dataset_Validation_Identity_Tracking_Behaviour_Preparation/outputs/primary_split_v21/week7_primary_split_v21_val.csv`
- Test: `/home/oyavuz/PigBench/Week7_Dataset_Validation_Identity_Tracking_Behaviour_Preparation/outputs/primary_split_v21/week7_primary_split_v21_test.csv`
- Summary: `/home/oyavuz/PigBench/Week7_Dataset_Validation_Identity_Tracking_Behaviour_Preparation/outputs/primary_split_v21/week7_primary_split_v21_summary.csv`
- Class distribution: `/home/oyavuz/PigBench/Week7_Dataset_Validation_Identity_Tracking_Behaviour_Preparation/outputs/primary_split_v21/week7_primary_split_v21_class_distribution.csv`
- Frame distribution: `/home/oyavuz/PigBench/Week7_Dataset_Validation_Identity_Tracking_Behaviour_Preparation/outputs/primary_split_v21/week7_primary_split_v21_frame_distribution.csv`
- Video leakage report: `/home/oyavuz/PigBench/Week7_Dataset_Validation_Identity_Tracking_Behaviour_Preparation/outputs/primary_split_v21/week7_primary_split_v21_video_leakage_report.csv`
- README: `/home/oyavuz/PigBench/Week7_Dataset_Validation_Identity_Tracking_Behaviour_Preparation/outputs/primary_split_v21/README_primary_split_v21.md`
