# Week 8 v64b2 Assignment Consistency Fix

- v64b2 decision: assignment_consistency_fix_completed
- Backup created: /home/oyavuz/PigBench/Week8_Ground_Truth_Validation_Behaviour_Dataset_Preparation/validation/week8_v63b_manual_gt_v2_assignments_backup_before_v64b2_20260720_213224.csv
- Patched rows: 2
- Patch log: /home/oyavuz/PigBench/Week8_Ground_Truth_Validation_Behaviour_Dataset_Preparation/outputs/v64b2_assignment_consistency_fix/week8_v64b2_assignment_consistency_fix_log.csv

Rule applied: fix_required and unknown_pending_review rows cannot be use_for_classification. They are set to pending_review. red_exclude rows are set to exclude_from_classification.
