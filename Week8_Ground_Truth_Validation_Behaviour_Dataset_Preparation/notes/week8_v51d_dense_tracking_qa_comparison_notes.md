# Week 8 v51d Dense Tracking QA Comparison

## Summary

- v51d decision: dense_tracking_qa_completed_selected_source_ready_for_visualizer
- Selected tracking source: v51c_nms_corrected_dense
- Selection reason: NMS-corrected dense tracking keeps recall close while reducing detections/tracks.
- v51b strong/moderate recall: 0.7972
- v51c strong/moderate recall: 0.7925
- v51b weak-inclusive recall: 0.8508
- v51c weak-inclusive recall: 0.8462
- v51b tracking rows: 161319
- v51c tracking rows: 150034
- v51b unique track sum: 1862
- v51c unique track sum: 1382
- Row reduction: 11285
- Track reduction: 480
- Hard issue count: 0
- Warning count: 0
- Ready for v52 visualizer: True

## Important limitation

Tracking IDs are not final pig identities. The selected tracking source is for improved full-clip visualization and review.

Gallery: /home/oyavuz/PigBench/Week8_Ground_Truth_Validation_Behaviour_Dataset_Preparation/outputs/v51d_dense_tracking_qa_comparison/index.html
