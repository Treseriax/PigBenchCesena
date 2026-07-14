# Week 6 YOLOv8-s All Scanpoint Detection Notes

## Purpose

This step runs the PigBench YOLOv8-s detector on all 72 extracted manual scan-sampling frames. The resulting bounding boxes will be used for bbox visualization, crop-based features, colour-marker exploration, and later bbox-to-label association.

## Detector

- Config: `/home/oyavuz/PigBench/detection/configs/yolov8/yolov8_s.py`
- Checkpoint: `/home/oyavuz/PigBench/detection/data/pretrained_weights/yolov8_pigs/yolov8_s.pth`
- Device: `cuda:0`
- Score threshold: `0.25`

## Outputs

- Detections CSV: `/home/oyavuz/PigBench/Week6_Unibo_Dataset_Validation/outputs/feature_extractors/week6_yolov8s_all_scanpoint_detections.csv`
- Frame summary CSV: `/home/oyavuz/PigBench/Week6_Unibo_Dataset_Validation/outputs/feature_extractors/week6_yolov8s_all_scanpoint_frame_summary.csv`
- Detection visualizations: `/home/oyavuz/PigBench/Week6_Unibo_Dataset_Validation/outputs/visual_label_check/yolov8s_all_scanpoint_detections`

## Status summary

| status   |   frame_count |   total_detections |   mean_detections_per_frame |   min_detections |   max_detections |
|:---------|--------------:|-------------------:|----------------------------:|-----------------:|-----------------:|
| ok       |            72 |                540 |                         7.5 |                4 |               14 |

## Detection score summary

|   num_detections |   mean_score |   median_score |   min_score |   max_score |   mean_bbox_area |   median_bbox_area |
|-----------------:|-------------:|---------------:|------------:|------------:|-----------------:|-------------------:|
|              540 |     0.744028 |       0.831413 |    0.266454 |    0.955514 |          13358.1 |            11159.1 |

## Interpretation

These detections provide pig bounding boxes for the scan-sampling frames, but they are not yet assigned to manual colour IDs. The next step is to combine detection bboxes with the manual colour/behaviour label panel for visual QC, then attempt crop-based colour-marker association.
