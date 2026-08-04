# Week 7 v39 Clip Multi-Label Temporal Baseline Dry-Run

## Summary

- Model-ready clip rows: `70`
- Train clips: `47`
- Val clips: `12`
- Test clips: `11`
- Label count: `11`
- Feature count: `59`
- Models evaluated: `label_prior_majority_multilabel, one_vs_rest_nearest_centroid_multilabel`
- Centroid test micro-F1: `0.4819`
- Centroid test macro-F1: `0.3036`
- Centroid test sample-F1: `0.4890`
- Centroid test hamming loss: `0.3554`
- Hard issue count: `0`
- Warning count: `2`
- Ready for v40 evidence package: `True`

## Important interpretation

This is a clip-level multi-label temporal baseline sanity check, not a final behaviour classifier.

## Outputs

- Data audit: `/home/oyavuz/PigBench/Week7_Dataset_Validation_Identity_Tracking_Behaviour_Preparation/outputs/week7_clip_multilabel_temporal_baseline_dryrun_v39/week7_v39_clip_multilabel_data_audit.csv`
- Predictions: `/home/oyavuz/PigBench/Week7_Dataset_Validation_Identity_Tracking_Behaviour_Preparation/outputs/week7_clip_multilabel_temporal_baseline_dryrun_v39/week7_v39_clip_multilabel_predictions.csv`
- Metrics: `/home/oyavuz/PigBench/Week7_Dataset_Validation_Identity_Tracking_Behaviour_Preparation/outputs/week7_clip_multilabel_temporal_baseline_dryrun_v39/week7_v39_clip_multilabel_metrics.csv`
- Per-label metrics: `/home/oyavuz/PigBench/Week7_Dataset_Validation_Identity_Tracking_Behaviour_Preparation/outputs/week7_clip_multilabel_temporal_baseline_dryrun_v39/week7_v39_clip_multilabel_per_label_metrics.csv`
- Label confusion: `/home/oyavuz/PigBench/Week7_Dataset_Validation_Identity_Tracking_Behaviour_Preparation/outputs/week7_clip_multilabel_temporal_baseline_dryrun_v39/week7_v39_clip_multilabel_label_confusion.csv`
- Model summary: `/home/oyavuz/PigBench/Week7_Dataset_Validation_Identity_Tracking_Behaviour_Preparation/outputs/week7_clip_multilabel_temporal_baseline_dryrun_v39/week7_v39_clip_multilabel_model_summary.csv`
- Report: `/home/oyavuz/PigBench/Week7_Dataset_Validation_Identity_Tracking_Behaviour_Preparation/outputs/week7_clip_multilabel_temporal_baseline_dryrun_v39/week7_v39_clip_multilabel_temporal_baseline_report.md`
- Decision: `/home/oyavuz/PigBench/Week7_Dataset_Validation_Identity_Tracking_Behaviour_Preparation/outputs/week7_clip_multilabel_temporal_baseline_dryrun_v39/week7_v39_clip_multilabel_temporal_baseline_decision_summary.csv`
- Issues: `/home/oyavuz/PigBench/Week7_Dataset_Validation_Identity_Tracking_Behaviour_Preparation/outputs/week7_clip_multilabel_temporal_baseline_dryrun_v39/week7_v39_clip_multilabel_temporal_baseline_issues.csv`
- v40 config: `/home/oyavuz/PigBench/Week7_Dataset_Validation_Identity_Tracking_Behaviour_Preparation/outputs/week7_clip_multilabel_temporal_baseline_dryrun_v39/week7_v39_recommended_v40_behaviour_temporal_evidence_package_config.json`
