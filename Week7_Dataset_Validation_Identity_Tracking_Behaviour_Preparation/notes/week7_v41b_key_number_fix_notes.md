# Week 7 v41b Key Number Fix

## Summary

- Corrected metric: `model_ready_clip_rows`
- Corrected value: `70`
- Fixed zip: `/home/oyavuz/PigBench/Week7_Dataset_Validation_Identity_Tracking_Behaviour_Preparation/outputs/week7_final_report_ready_summary_visual_package_v41/Week7_Final_Report_Ready_Summary_Visual_Package_v41b_FIXED.zip`
- Fixed zip SHA256: `fd0034fbf88ac4b26fea036c484276f9f9c0c6f83b7902f5dafcd091775fe95b`
- Hard issue count: `0`
- Report-ready fixed package: `True`

## Reason

The original v41 key numbers table accidentally wrote clip_temporal_test_micro_f1 into the model_ready_clip_rows row.
The correct value is taken from v39 decision summary.
