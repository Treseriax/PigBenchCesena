# Week 7 Detector / Tracker Quality Audit v27c

## Purpose

This step documents v27b dry-run quality and decides whether full tracking is justified.

## Summary

- Detector dry-run pass: `True`
- Simple IoU tracker final pass: `False`
- Clips with moderate/high fragmentation: `2`
- Clips with moderate/high ROI leakage: `5`
- Recommended next step: `ROI-filtered detector/tracker dry-run before full 72-clip tracking`
- Ready for v28 ROI-filtered dry-run: `True`

## Outputs

- Decision summary: `/home/oyavuz/PigBench/Week7_Dataset_Validation_Identity_Tracking_Behaviour_Preparation/outputs/detector_tracker_quality_audit_v27c/week7_detector_tracker_quality_audit_v27c_decision_summary.csv`
- Clip quality: `/home/oyavuz/PigBench/Week7_Dataset_Validation_Identity_Tracking_Behaviour_Preparation/outputs/detector_tracker_quality_audit_v27c/week7_detector_tracker_quality_audit_v27c_clip_quality.csv`
- Frame quality: `/home/oyavuz/PigBench/Week7_Dataset_Validation_Identity_Tracking_Behaviour_Preparation/outputs/detector_tracker_quality_audit_v27c/week7_detector_tracker_quality_audit_v27c_frame_quality.csv`
- Visual review template: `/home/oyavuz/PigBench/Week7_Dataset_Validation_Identity_Tracking_Behaviour_Preparation/outputs/detector_tracker_quality_audit_v27c/week7_detector_tracker_quality_audit_v27c_visual_review_template.csv`
- Limitations: `/home/oyavuz/PigBench/Week7_Dataset_Validation_Identity_Tracking_Behaviour_Preparation/outputs/detector_tracker_quality_audit_v27c/week7_detector_tracker_quality_audit_v27c_limitations.md`
- Issues: `/home/oyavuz/PigBench/Week7_Dataset_Validation_Identity_Tracking_Behaviour_Preparation/outputs/detector_tracker_quality_audit_v27c/week7_detector_tracker_quality_audit_v27c_issues.csv`
