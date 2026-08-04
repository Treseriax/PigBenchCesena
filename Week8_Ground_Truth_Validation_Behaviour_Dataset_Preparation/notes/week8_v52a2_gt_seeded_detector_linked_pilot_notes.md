# Week 8 v52a2 GT-seeded Detector-linked Tracking Pilot

## Summary

- v52a2 decision: gt_seeded_detector_linked_pilot_completed
- Pilot clips: 14
- Pilot objects: 84
- Tracking rows total: 20970
- Mean detector-linked ratio: 0.6698
- Mean missing ratio: 0.2259
- Mean draw-ok ratio: 0.7741
- Hard issue count: 0
- Warning count: 0
- Ready for v52b full GT-seeded detector-linked tracking: True

## Interpretation

v52a failed because OpenCV tracker propagation did not work. v52a2 instead uses anchor GT identities and links them through v51c dense detections frame by frame.

Gallery: /home/oyavuz/PigBench/Week8_Ground_Truth_Validation_Behaviour_Dataset_Preparation/outputs/v52a2_gt_seeded_detector_linked_pilot/index.html
