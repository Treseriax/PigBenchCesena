# Week 8 v52c Full Hybrid Tracking QA Report

## Decision

- Decision: full_hybrid_tracking_qa_completed
- Ready for v52d tracking-integrated visualizer: True

## Global metrics

- Clip count: 72
- Tracking rows total: 107113
- Mean stable tracklet ratio: 0.7582
- Mean recall fallback ratio: 0.1385
- Mean missing ratio: 0.1033
- Mean draw-ok ratio: 0.8967
- Mean review-needed ratio: 0.2418

## Quality tiers

- strong_visual_tracking_support: 33
- usable_with_review: 30
- limited_review_required: 7
- challenging_low_tracking_support: 2

## Interpretation

The full hybrid tracking layer provides strong visual tracking support for many clips, while preserving fallback and missing cases explicitly for manual review. It should be used as a tracking-assisted validation layer, not as fully automatic ground-truth replacement.

## Output files

- Clip QA: /home/oyavuz/PigBench/Week8_Ground_Truth_Validation_Behaviour_Dataset_Preparation/outputs/v52c_full_hybrid_tracking_qa/week8_v52c_clip_quality_assessment.csv
- Object QA: /home/oyavuz/PigBench/Week8_Ground_Truth_Validation_Behaviour_Dataset_Preparation/outputs/v52c_full_hybrid_tracking_qa/week8_v52c_object_quality_assessment.csv
- Manual review queue: /home/oyavuz/PigBench/Week8_Ground_Truth_Validation_Behaviour_Dataset_Preparation/outputs/v52c_full_hybrid_tracking_qa/week8_v52c_manual_review_queue.csv
- Successful cases: /home/oyavuz/PigBench/Week8_Ground_Truth_Validation_Behaviour_Dataset_Preparation/outputs/v52c_full_hybrid_tracking_qa/week8_v52c_successful_cases.csv
- Challenging cases: /home/oyavuz/PigBench/Week8_Ground_Truth_Validation_Behaviour_Dataset_Preparation/outputs/v52c_full_hybrid_tracking_qa/week8_v52c_challenging_cases.csv