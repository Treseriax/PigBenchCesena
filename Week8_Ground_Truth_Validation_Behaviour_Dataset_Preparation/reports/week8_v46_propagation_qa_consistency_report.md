# Week 8 v46 Propagation QA and Consistency Report

## Decision

- v46 decision: propagation_qa_passed
- Ready for v47 visualization interface: True
- Hard issue count: 0
- Warning count: 0
- Info count: 2

## QA results

| Check | Passed |
|---|---:|
| Global counts | True |
| Per-clip frame/object counts | True |
| Per-object propagation counts | True |
| Label stability across propagated frames | True |
| BBox validity | True |
| Timestamp consistency | True |
| JSONL consistency | True |

## Main counts

- Clip count: 72
- Clip-object rows: 429
- Frame-object rows: 107113
- JSONL frame lines: 17977
- Missing behaviour objects preserved: 55
- Unmapped clips preserved: 2

## Interpretation

The propagated ground-truth dataset is internally consistent if all hard checks pass. Missing behaviour objects and unmapped clips are expected preserved cases, not failures.

## Scope note

Bounding boxes in v45/v46 are scanpoint-anchor boxes repeated across the interval. This is appropriate for validating label propagation and building the visual interface. Tracking-refined per-frame boxes remain a later refinement step.

## Outputs

- Consistency checks: /home/oyavuz/PigBench/Week8_Ground_Truth_Validation_Behaviour_Dataset_Preparation/outputs/propagated_ground_truth/v46_propagation_qa/week8_v46_propagation_consistency_checks.csv
- Object/frame count consistency: /home/oyavuz/PigBench/Week8_Ground_Truth_Validation_Behaviour_Dataset_Preparation/outputs/propagated_ground_truth/v46_propagation_qa/week8_v46_object_frame_count_consistency.csv
- Clip count consistency: /home/oyavuz/PigBench/Week8_Ground_Truth_Validation_Behaviour_Dataset_Preparation/outputs/propagated_ground_truth/v46_propagation_qa/week8_v46_clip_count_consistency.csv
- Label stability: /home/oyavuz/PigBench/Week8_Ground_Truth_Validation_Behaviour_Dataset_Preparation/outputs/propagated_ground_truth/v46_propagation_qa/week8_v46_label_stability_by_object.csv
- Timestamp QA: /home/oyavuz/PigBench/Week8_Ground_Truth_Validation_Behaviour_Dataset_Preparation/outputs/propagated_ground_truth/v46_propagation_qa/week8_v46_timestamp_consistency.csv
- BBox QA: /home/oyavuz/PigBench/Week8_Ground_Truth_Validation_Behaviour_Dataset_Preparation/outputs/propagated_ground_truth/v46_propagation_qa/week8_v46_bbox_consistency.csv
- JSONL QA: /home/oyavuz/PigBench/Week8_Ground_Truth_Validation_Behaviour_Dataset_Preparation/outputs/propagated_ground_truth/v46_propagation_qa/week8_v46_jsonl_consistency.csv
- Validation issues: /home/oyavuz/PigBench/Week8_Ground_Truth_Validation_Behaviour_Dataset_Preparation/validation/week8_v46_propagation_validation_issues.csv
