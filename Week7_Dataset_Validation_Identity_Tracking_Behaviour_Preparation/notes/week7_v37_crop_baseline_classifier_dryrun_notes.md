# Week 7 v37 Crop Baseline Classifier Dry-Run

## Summary

- Feature rows used: `143`
- Train rows: `100`
- Val rows: `23`
- Test rows: `20`
- Classes: `AN, BOX, LAI, STI`
- Models evaluated: `majority_train_class, nearest_centroid_lightweight_features`
- Nearest centroid test accuracy: `0.1000`
- Nearest centroid test macro-F1: `0.0769`
- Hard issue count: `0`
- Warning count: `2`
- Ready for v38 clip feature preparation: `True`

## Important interpretation

This is a crop-only lightweight baseline sanity check, not a final behaviour classifier.

## Outputs

- Predictions: `/home/oyavuz/PigBench/Week7_Dataset_Validation_Identity_Tracking_Behaviour_Preparation/outputs/week7_crop_baseline_classifier_dryrun_v37/week7_v37_crop_baseline_predictions.csv`
- Metrics: `/home/oyavuz/PigBench/Week7_Dataset_Validation_Identity_Tracking_Behaviour_Preparation/outputs/week7_crop_baseline_classifier_dryrun_v37/week7_v37_crop_baseline_metrics.csv`
- Per-class metrics: `/home/oyavuz/PigBench/Week7_Dataset_Validation_Identity_Tracking_Behaviour_Preparation/outputs/week7_crop_baseline_classifier_dryrun_v37/week7_v37_crop_baseline_per_class_metrics.csv`
- Confusion matrices: `/home/oyavuz/PigBench/Week7_Dataset_Validation_Identity_Tracking_Behaviour_Preparation/outputs/week7_crop_baseline_classifier_dryrun_v37/week7_v37_crop_baseline_confusion_matrices.csv`
- Model summary: `/home/oyavuz/PigBench/Week7_Dataset_Validation_Identity_Tracking_Behaviour_Preparation/outputs/week7_crop_baseline_classifier_dryrun_v37/week7_v37_crop_baseline_model_summary.csv`
- Data audit: `/home/oyavuz/PigBench/Week7_Dataset_Validation_Identity_Tracking_Behaviour_Preparation/outputs/week7_crop_baseline_classifier_dryrun_v37/week7_v37_crop_baseline_data_audit.csv`
- Report: `/home/oyavuz/PigBench/Week7_Dataset_Validation_Identity_Tracking_Behaviour_Preparation/outputs/week7_crop_baseline_classifier_dryrun_v37/week7_v37_crop_baseline_classifier_dryrun_report.md`
- Decision: `/home/oyavuz/PigBench/Week7_Dataset_Validation_Identity_Tracking_Behaviour_Preparation/outputs/week7_crop_baseline_classifier_dryrun_v37/week7_v37_crop_baseline_classifier_dryrun_decision_summary.csv`
- Issues: `/home/oyavuz/PigBench/Week7_Dataset_Validation_Identity_Tracking_Behaviour_Preparation/outputs/week7_crop_baseline_classifier_dryrun_v37/week7_v37_crop_baseline_classifier_dryrun_issues.csv`
- v38 config: `/home/oyavuz/PigBench/Week7_Dataset_Validation_Identity_Tracking_Behaviour_Preparation/outputs/week7_crop_baseline_classifier_dryrun_v37/week7_v37_recommended_v38_clip_feature_preparation_config.json`
