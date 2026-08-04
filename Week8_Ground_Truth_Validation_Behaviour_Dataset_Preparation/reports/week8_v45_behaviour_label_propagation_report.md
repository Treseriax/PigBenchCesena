# Week 8 v45 Behaviour Label Propagation Report

## Decision

- v45 decision: behaviour_label_propagation_completed
- Input clips: 72
- Processed clips: 72
- Total frames generated: 17977
- Clip-object rows: 429
- Frame-object rows: 107113
- Clip objects with behaviour labels: 374
- Clip objects training-ready: 374
- Clip objects missing behaviour: 55
- Unmapped clips: 2
- Hard issue count: 0
- Warning count: 0
- Ready for v46 propagation QA: True

## Propagation rule

Each annotated scanpoint defines a 10-second observation interval. Pig-level behaviour labels are propagated to every frame within that interval.

## Important scope note

The current v45 geometry source is scanpoint-anchor boxes repeated across the interval. This is sufficient for label propagation and validation indexing. Tracking-refined per-frame boxes can be integrated later through the Week 8 identity/tracking improvement stage.

## Outputs

- Clip-object annotations: /home/oyavuz/PigBench/Week8_Ground_Truth_Validation_Behaviour_Dataset_Preparation/outputs/propagated_ground_truth/v45_label_propagation/week8_v45_clip_object_propagated_annotations.csv
- Frame-object annotations: /home/oyavuz/PigBench/Week8_Ground_Truth_Validation_Behaviour_Dataset_Preparation/outputs/propagated_ground_truth/v45_label_propagation/week8_v45_frame_object_propagated_annotations.csv
- Frame-level JSONL: /home/oyavuz/PigBench/Week8_Ground_Truth_Validation_Behaviour_Dataset_Preparation/outputs/propagated_ground_truth/v45_label_propagation/week8_v45_frame_level_ground_truth.jsonl
- Clip-level JSON: /home/oyavuz/PigBench/Week8_Ground_Truth_Validation_Behaviour_Dataset_Preparation/outputs/propagated_ground_truth/v45_label_propagation/week8_v45_clip_level_ground_truth.json
- Clip propagation summary: /home/oyavuz/PigBench/Week8_Ground_Truth_Validation_Behaviour_Dataset_Preparation/outputs/propagated_ground_truth/v45_label_propagation/week8_v45_clip_propagation_summary.csv
- QA summary: /home/oyavuz/PigBench/Week8_Ground_Truth_Validation_Behaviour_Dataset_Preparation/outputs/propagated_ground_truth/v45_label_propagation/week8_v45_propagation_qa_summary.csv
- Issues: /home/oyavuz/PigBench/Week8_Ground_Truth_Validation_Behaviour_Dataset_Preparation/outputs/propagated_ground_truth/v45_label_propagation/week8_v45_propagation_issues.csv

## Next step

v46 should perform propagation QA and consistency checks before the visualization interface is built.
