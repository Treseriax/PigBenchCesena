# Week 8 v52a4 Tracklet-level GT Linking Pilot

## Summary

- v52a4 decision: tracklet_level_gt_linking_pilot_completed
- Pilot clips: 14
- Pilot objects: 84
- Tracking rows total: 20970
- Mean matched object ratio: 0.7262
- Mean tracklet detection ratio: 0.5665
- Mean missing ratio: 0.4047
- Mean draw-ok ratio: 0.5953
- Hard issue count: 0
- Warning count: 0
- Ready for v52b full tracklet-level linking: True

## Interpretation

v52a2/v52a3 can jump identity frame-by-frame. v52a4 instead locks each anchor GT object onto a selected detector tracklet, improving identity stability at the cost of possible missing gaps.

Gallery: /home/oyavuz/PigBench/Week8_Ground_Truth_Validation_Behaviour_Dataset_Preparation/outputs/v52a4_tracklet_level_gt_linking_pilot/index.html
