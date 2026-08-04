# Week 8 v43c Final Input Lock

## Summary

- v43c decision: `final_input_lock_passed`
- Locked input count: `12`
- Behaviour fusion rows: `429`
- Training-ready rows: `374`
- Clip index rows: `72`
- Clip-level index rows: `72`
- Hard issue count: `0`
- Ready for v44 ground-truth schema: `True`

## Final lock decisions

- Use v18c box-level fusion as the main 429-row identity + behaviour reference.
- Use v21 primary split as the canonical 374-row training-ready set.
- Use v26 clip extraction index as the 72-clip interval reference.
- Keep v17 frame QA as supporting QA, not box-level identity.

## Outputs

- Final locked inputs JSON: `/home/oyavuz/PigBench/Week8_Ground_Truth_Validation_Behaviour_Dataset_Preparation/outputs/v43_setup_input_audit/week8_v43c_final_locked_inputs.json`
- Audit CSV: `/home/oyavuz/PigBench/Week8_Ground_Truth_Validation_Behaviour_Dataset_Preparation/outputs/v43_setup_input_audit/week8_v43c_final_locked_input_audit.csv`
- Decision CSV: `/home/oyavuz/PigBench/Week8_Ground_Truth_Validation_Behaviour_Dataset_Preparation/outputs/v43_setup_input_audit/week8_v43c_decision_summary.csv`
- Issues CSV: `/home/oyavuz/PigBench/Week8_Ground_Truth_Validation_Behaviour_Dataset_Preparation/outputs/v43_setup_input_audit/week8_v43c_issues.csv`
- Report: `/home/oyavuz/PigBench/Week8_Ground_Truth_Validation_Behaviour_Dataset_Preparation/outputs/v43_setup_input_audit/week8_v43c_final_input_lock_report.md`
