# Week 7 Detector Recall Expansion with Low Threshold

## Purpose

This step reruns the pig detector on the 72 scanpoint frames to recover possible missed pigs. The motivation is that some correct ground-truth pen pigs were not detected in the previous detection table, while some wrong-pen detections were still marked as strict-safe.

## Model

- Config: `/home/oyavuz/PigBench/detection/configs/yolov8/yolov8_s.py`
- Checkpoint: `/home/oyavuz/PigBench/detection/data/pretrained_weights/yolov8_pigs/yolov8_s.pth`
- Device: `cpu`

## Threshold summary

|   threshold |   total_detections |   mean_detections_per_frame |   median_detections_per_frame |   min_detections_per_frame |   max_detections_per_frame |   new_candidates_vs_week6 |
|------------:|-------------------:|----------------------------:|------------------------------:|---------------------------:|---------------------------:|--------------------------:|
|        0.05 |                721 |                    10.0139  |                           9.5 |                          4 |                         16 |                       141 |
|        0.1  |                637 |                     8.84722 |                           8   |                          4 |                         15 |                        78 |
|        0.15 |                594 |                     8.25    |                           8   |                          4 |                         14 |                        43 |
|        0.2  |                565 |                     7.84722 |                           8   |                          4 |                         14 |                        17 |
|        0.25 |                540 |                     7.5     |                           7   |                          4 |                         14 |                         0 |
|        0.3  |                518 |                     7.19444 |                           7   |                          4 |                         14 |                         0 |
|        0.5  |                447 |                     6.20833 |                           6   |                          3 |                         12 |                         0 |

## New candidates

- New low-threshold candidates with score >= 0.05: `141`
- Contact sheet: `/home/oyavuz/PigBench/Week7_Dataset_Validation_Identity_Tracking_Behaviour_Preparation/outputs/visualizations/week7_detector_recall_expansion_problem_frames_contact_sheet.jpg`
- Contact sheet generated: `True`

## Interpretation

New low-threshold detections should not be accepted automatically. They are candidate boxes that may recover missed true pigs, but they also introduce more false positives. The next step is to merge useful new candidates into a v7 corrected candidate pool with manual review.

## Outputs

- `/home/oyavuz/PigBench/Week7_Dataset_Validation_Identity_Tracking_Behaviour_Preparation/outputs/roi_and_crate_mapping/week7_detector_recall_expansion_low_threshold_detections.csv`
- `/home/oyavuz/PigBench/Week7_Dataset_Validation_Identity_Tracking_Behaviour_Preparation/outputs/roi_and_crate_mapping/week7_detector_recall_expansion_threshold_summary.csv`
- `/home/oyavuz/PigBench/Week7_Dataset_Validation_Identity_Tracking_Behaviour_Preparation/outputs/roi_and_crate_mapping/week7_detector_recall_expansion_new_candidates.csv`
- `/home/oyavuz/PigBench/Week7_Dataset_Validation_Identity_Tracking_Behaviour_Preparation/outputs/roi_and_crate_mapping/week7_detector_recall_expansion_frame_summary.csv`
- `/home/oyavuz/PigBench/Week7_Dataset_Validation_Identity_Tracking_Behaviour_Preparation/outputs/dataset_statistics/week7_detector_recall_expansion_model_audit.csv`
- `/home/oyavuz/PigBench/Week7_Dataset_Validation_Identity_Tracking_Behaviour_Preparation/outputs/visualizations/week7_detector_recall_expansion_problem_frames_contact_sheet.jpg`
