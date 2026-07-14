# Week 7 Split Strategy Design v20

## Purpose

This step creates and evaluates candidate train/validation/test split strategies from the v19 locked training-ready dataset.

## Strategies

- `frame_stratified`: frame-level units, class-aware greedy assignment.
- `video_aware`: video-level units, class-aware greedy assignment.
- `temporal`: chronological frame-level split.

## Summary

- Input rows: `374`
- Input frames: `70`
- Input videos: `12`
- Behaviour classes: `11`
- Recommended candidate: `frame_stratified`

## Outputs

- Frame-stratified assignments: `/home/oyavuz/PigBench/Week7_Dataset_Validation_Identity_Tracking_Behaviour_Preparation/outputs/split_strategy_design_v20/week7_split_strategy_design_v20_frame_stratified_assignments.csv`
- Video-aware assignments: `/home/oyavuz/PigBench/Week7_Dataset_Validation_Identity_Tracking_Behaviour_Preparation/outputs/split_strategy_design_v20/week7_split_strategy_design_v20_video_aware_assignments.csv`
- Temporal assignments: `/home/oyavuz/PigBench/Week7_Dataset_Validation_Identity_Tracking_Behaviour_Preparation/outputs/split_strategy_design_v20/week7_split_strategy_design_v20_temporal_assignments.csv`
- Candidate summary: `/home/oyavuz/PigBench/Week7_Dataset_Validation_Identity_Tracking_Behaviour_Preparation/outputs/split_strategy_design_v20/week7_split_strategy_design_v20_candidate_summary.csv`
- Class distribution by candidate: `/home/oyavuz/PigBench/Week7_Dataset_Validation_Identity_Tracking_Behaviour_Preparation/outputs/split_strategy_design_v20/week7_split_strategy_design_v20_class_distribution_by_candidate.csv`
- Leakage report: `/home/oyavuz/PigBench/Week7_Dataset_Validation_Identity_Tracking_Behaviour_Preparation/outputs/split_strategy_design_v20/week7_split_strategy_design_v20_leakage_report.csv`
- Recommendation: `/home/oyavuz/PigBench/Week7_Dataset_Validation_Identity_Tracking_Behaviour_Preparation/outputs/split_strategy_design_v20/week7_split_strategy_design_v20_recommendation.md`
- Split datasets: `/home/oyavuz/PigBench/Week7_Dataset_Validation_Identity_Tracking_Behaviour_Preparation/outputs/split_strategy_design_v20/split_datasets`
