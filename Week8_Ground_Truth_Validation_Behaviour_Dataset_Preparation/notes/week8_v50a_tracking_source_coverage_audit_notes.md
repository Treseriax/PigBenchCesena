# Week 8 v50a Tracking Source Coverage Audit

## Summary

- v50a decision: existing_tracking_source_partial_full_72_tracking_run_needed
- Selected tracking source: /home/oyavuz/PigBench/Week7_Dataset_Validation_Identity_Tracking_Behaviour_Preparation/outputs/dense_polygon_filtered_tracking_v28d/week7_dense_polygon_filtered_tracking_v28d_tracks.csv
- Tracking rows total: 987
- Valid tracking bbox rows: 987
- Covered clips: 5/72
- Missing tracking clips: 67
- Coverage ratio: 0.0694
- Hard issue count: 0
- Warning count: 1
- Ready for v50b full 72-clip tracking run: True
- Ready for v50c tracking integration: False

## Interpretation

Existing tracking should only be used for final visual validation if it covers all 72 clips with usable frame, track, and bbox columns. If coverage is partial, the next step is a full 72-clip tracking run.
