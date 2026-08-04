Week 8 v51d Dense Tracking QA Comparison Report

Decision: dense_tracking_qa_completed_selected_source_ready_for_visualizer
Selected source: v51c_nms_corrected_dense
Selection reason: NMS-corrected dense tracking keeps recall close while reducing detections/tracks.

v51b strong/moderate recall: 0.7972
v51c strong/moderate recall: 0.7925
v51b weak-inclusive recall: 0.8508
v51c weak-inclusive recall: 0.8462
v51b tracking rows: 161319
v51c tracking rows: 150034
v51b unique track sum: 1862
v51c unique track sum: 1382
Row reduction: 11285
Track reduction: 480

Interpretation:
This QA compares raw dense tracking and NMS-corrected dense tracking against anchor GT objects. The selected source should be used in the next visualizer, but tracking IDs are not final pig identities.
