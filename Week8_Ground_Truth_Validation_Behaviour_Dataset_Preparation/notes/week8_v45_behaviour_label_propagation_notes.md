# Week 8 v45 Behaviour Label Propagation

## Summary

- v45 decision: behaviour_label_propagation_completed
- Processed clips: 72 / 72
- Total frames generated: 17977
- Clip-object rows: 429
- Frame-object rows: 107113
- Clip objects with behaviour: 374
- Clip objects training-ready: 374
- Clip objects missing behaviour: 55
- Unmapped clips: 2
- Hard issue count: 0
- Warning count: 0
- Ready for v46 propagation QA: True

## Key scope note

Behaviour labels are propagated over the full annotated 10-second interval. Bounding boxes are scanpoint-anchor boxes repeated across the interval in v45; tracking-refined boxes can be integrated in a later identity/tracking refinement step.

## Outputs

- Clip-object CSV: /home/oyavuz/PigBench/Week8_Ground_Truth_Validation_Behaviour_Dataset_Preparation/outputs/propagated_ground_truth/v45_label_propagation/week8_v45_clip_object_propagated_annotations.csv
- Frame-object CSV: /home/oyavuz/PigBench/Week8_Ground_Truth_Validation_Behaviour_Dataset_Preparation/outputs/propagated_ground_truth/v45_label_propagation/week8_v45_frame_object_propagated_annotations.csv
- Frame-level JSONL: /home/oyavuz/PigBench/Week8_Ground_Truth_Validation_Behaviour_Dataset_Preparation/outputs/propagated_ground_truth/v45_label_propagation/week8_v45_frame_level_ground_truth.jsonl
- Clip-level JSON: /home/oyavuz/PigBench/Week8_Ground_Truth_Validation_Behaviour_Dataset_Preparation/outputs/propagated_ground_truth/v45_label_propagation/week8_v45_clip_level_ground_truth.json
- QA summary: /home/oyavuz/PigBench/Week8_Ground_Truth_Validation_Behaviour_Dataset_Preparation/outputs/propagated_ground_truth/v45_label_propagation/week8_v45_propagation_qa_summary.csv
- Issues: /home/oyavuz/PigBench/Week8_Ground_Truth_Validation_Behaviour_Dataset_Preparation/outputs/propagated_ground_truth/v45_label_propagation/week8_v45_propagation_issues.csv
