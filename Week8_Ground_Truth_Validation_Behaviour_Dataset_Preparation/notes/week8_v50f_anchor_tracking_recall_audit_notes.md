# Week 8 v50f Anchor Tracking Recall Audit

## Summary

- v50f decision: anchor_tracking_recall_audit_completed_use_tracking_as_optional_layer
- Anchor objects: 429
- Filtered strong matches: 277
- Filtered moderate matches: 24
- Raw matches removed by ROI: 0
- Weak filtered matches: 30
- Detector/sampling misses: 98
- Filtered anchor recall: 0.7016
- Raw anchor recall with weak/ROI-removed: 0.7716
- Good recall clips: 11
- Moderate recall clips: 41
- Low recall clips: 20
- Review queue rows: 128
- Hard issue count: 0
- Warning count: 2
- Ready for v51 tracking-integrated visualizer: True

## Important limitation

Tracking boxes should be shown as an optional visualization layer, not as complete ground truth and not as final pig identity.

Gallery: /home/oyavuz/PigBench/Week8_Ground_Truth_Validation_Behaviour_Dataset_Preparation/outputs/v50f_anchor_tracking_recall_audit/index.html
