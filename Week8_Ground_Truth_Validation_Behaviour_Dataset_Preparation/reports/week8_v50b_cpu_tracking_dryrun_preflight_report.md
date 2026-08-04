Week 8 v50b CPU Tracking Dry-run Preflight Report

Decision: cpu_tracking_dryrun_passed
Selected config: /home/oyavuz/PigBench/detection/configs/yolov8/yolov8_s.py
Selected checkpoint: /home/oyavuz/PigBench/detection/data/pretrained_weights/yolov8_pigs/yolov8_s.pth
Device: cpu
Processed clips: 3
Tracking rows: 1015
Unique tracks: 27
Hard issues: 0
Warnings: 0

Purpose:
This dry-run confirms whether the detector and simple IoU tracker can run safely before launching a full 72-clip tracking pass.

Next:
If ready_for_v50c_full_72_tracking is True, run the full 72-clip tracking stage with resume support.
