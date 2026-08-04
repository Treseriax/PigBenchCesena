Week 8 v51a Tracking Maximization Sweep Report

Decision: tracking_maximization_sweep_completed
Test clips: 12
Candidates tested: 16

Selected parameters:
- frame_stride: 2
- score_thr: 0.1
- tracker_iou_thr: 0.15
- nms_iou_thr: 0.55
- max_missed: 35

Selected performance on test subset:
- filtered recall strong/moderate: 0.6111
- weak-inclusive recall: 0.7500
- miss rate: 0.2500
- duplicate pressure: 1.2778
- quality score: 0.6169

Interpretation:
This sweep maximizes tracking before full integration. It tests denser frame sampling, lower score thresholds, NMS, and more permissive tracker IoU. The selected candidate should be used for v51b full 72-clip dense tracking.
