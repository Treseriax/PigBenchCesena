Week 8 v50a Tracking Source Coverage Audit Report

Decision:
existing_tracking_source_partial_full_72_tracking_run_needed

Selected tracking source:
/home/oyavuz/PigBench/Week7_Dataset_Validation_Identity_Tracking_Behaviour_Preparation/outputs/dense_polygon_filtered_tracking_v28d/week7_dense_polygon_filtered_tracking_v28d_tracks.csv

Coverage:
- Tracking rows total: 987
- Valid tracking bbox rows: 987
- Covered clips: 5/72
- Missing tracking clips: 67
- Coverage ratio: 0.0694

Detected columns:
- scan column: scan_frame_id
- frame column: frame_index
- track column: dense_track_id
- bbox columns: x1, y1, x2, y2

Interpretation:
v49c validated anchor-frame annotations. v50a checks whether existing tracking outputs are sufficient to replace fixed anchor boxes with tracking-refined per-frame boxes.

If existing tracking covers only a subset of the 72 clips, it should not be integrated as the final Week 8 viewer source. The next step should be a CPU-safe full 72-clip tracking run or a clearly scoped partial demonstration.

Outputs:
- Tracking source candidates: /home/oyavuz/PigBench/Week8_Ground_Truth_Validation_Behaviour_Dataset_Preparation/outputs/v50a_tracking_source_coverage_audit/week8_v50a_tracking_source_candidates.csv
- Column inventory: /home/oyavuz/PigBench/Week8_Ground_Truth_Validation_Behaviour_Dataset_Preparation/outputs/v50a_tracking_source_coverage_audit/week8_v50a_tracking_column_inventory.csv
- Coverage by clip: /home/oyavuz/PigBench/Week8_Ground_Truth_Validation_Behaviour_Dataset_Preparation/outputs/v50a_tracking_source_coverage_audit/week8_v50a_tracking_coverage_by_clip.csv
- Normalized sample: /home/oyavuz/PigBench/Week8_Ground_Truth_Validation_Behaviour_Dataset_Preparation/outputs/v50a_tracking_source_coverage_audit/week8_v50a_normalized_tracking_sample.csv
- Issues: /home/oyavuz/PigBench/Week8_Ground_Truth_Validation_Behaviour_Dataset_Preparation/outputs/v50a_tracking_source_coverage_audit/week8_v50a_issues.csv

Next:
- If ready_for_v50b_full_72_tracking_run=True, run full 72-clip tracking.
- If ready_for_v50c_tracking_integration=True, integrate existing tracking into the visualizer.
