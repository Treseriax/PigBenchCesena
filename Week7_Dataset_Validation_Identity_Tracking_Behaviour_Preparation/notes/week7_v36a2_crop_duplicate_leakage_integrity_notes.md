# Week 7 v36a2 Crop Duplicate / Leakage / Sample Integrity Audit

## Summary

- Original baseline rows: `288`
- Unique crop paths: `143`
- Duplicate groups: `80`
- Duplicate rows total: `225`
- Duplicate extra rows removed in dedup index: `145`
- Deduplicated baseline rows: `143`
- Cross-split leakage groups: `0`
- Label conflict groups: `0`
- Duplicate sample ID count: `0`
- Hard issue count: `0`
- Warning count: `1`
- Ready for v36b full crop feature extraction: `True`

## Outputs

- Duplicate groups: `/home/oyavuz/PigBench/Week7_Dataset_Validation_Identity_Tracking_Behaviour_Preparation/outputs/week7_crop_duplicate_leakage_integrity_audit_v36a2/week7_v36a2_duplicate_crop_path_groups.csv`
- Duplicate rows: `/home/oyavuz/PigBench/Week7_Dataset_Validation_Identity_Tracking_Behaviour_Preparation/outputs/week7_crop_duplicate_leakage_integrity_audit_v36a2/week7_v36a2_duplicate_crop_path_rows.csv`
- Conflicts: `/home/oyavuz/PigBench/Week7_Dataset_Validation_Identity_Tracking_Behaviour_Preparation/outputs/week7_crop_duplicate_leakage_integrity_audit_v36a2/week7_v36a2_leakage_or_label_conflict_groups.csv`
- Deduplicated index: `/home/oyavuz/PigBench/Week7_Dataset_Validation_Identity_Tracking_Behaviour_Preparation/outputs/week7_crop_duplicate_leakage_integrity_audit_v36a2/week7_v36a2_deduplicated_crop_baseline_ready_subset.csv`
- Removed duplicate rows: `/home/oyavuz/PigBench/Week7_Dataset_Validation_Identity_Tracking_Behaviour_Preparation/outputs/week7_crop_duplicate_leakage_integrity_audit_v36a2/week7_v36a2_removed_duplicate_rows.csv`
- Original vs dedup counts: `/home/oyavuz/PigBench/Week7_Dataset_Validation_Identity_Tracking_Behaviour_Preparation/outputs/week7_crop_duplicate_leakage_integrity_audit_v36a2/week7_v36a2_original_vs_dedup_class_split_counts.csv`
- Recommended v36b config: `/home/oyavuz/PigBench/Week7_Dataset_Validation_Identity_Tracking_Behaviour_Preparation/outputs/week7_crop_duplicate_leakage_integrity_audit_v36a2/week7_v36a2_recommended_v36b_feature_extraction_config.json`
- Report: `/home/oyavuz/PigBench/Week7_Dataset_Validation_Identity_Tracking_Behaviour_Preparation/outputs/week7_crop_duplicate_leakage_integrity_audit_v36a2/week7_v36a2_crop_duplicate_leakage_integrity_report.md`
- Decision: `/home/oyavuz/PigBench/Week7_Dataset_Validation_Identity_Tracking_Behaviour_Preparation/outputs/week7_crop_duplicate_leakage_integrity_audit_v36a2/week7_v36a2_crop_duplicate_leakage_integrity_decision_summary.csv`
- Issues: `/home/oyavuz/PigBench/Week7_Dataset_Validation_Identity_Tracking_Behaviour_Preparation/outputs/week7_crop_duplicate_leakage_integrity_audit_v36a2/week7_v36a2_crop_duplicate_leakage_integrity_issues.csv`
