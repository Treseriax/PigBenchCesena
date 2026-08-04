# Week 8 v52b2 Tracklet-level GT Linking Full

## Summary

- v52b2 decision: tracklet_level_gt_linking_full_completed
- Full clips: 72
- Full objects: 429
- Tracking rows total: 107113
- Mean matched object ratio: 0.8699
- Mean tracklet detection ratio: 0.7130
- Mean missing ratio: 0.2418
- Mean draw-ok ratio: 0.7582
- Hard issue count: 0
- Warning count: 0
- Ready for v52b full tracklet-level linking: True

## Interpretation

v52a2/v52a3 can jump identity frame-by-frame. v52b2 instead locks each anchor GT object onto a selected detector tracklet, improving identity stability at the cost of possible missing gaps.

Gallery: /home/oyavuz/PigBench/Week8_Ground_Truth_Validation_Behaviour_Dataset_Preparation/outputs/v52b2_tracklet_level_gt_linking_full/index.html
