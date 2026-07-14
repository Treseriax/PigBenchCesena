# Week 6 Detector-Backbone Crop Embedding Smoke Test v2

## Purpose

This step tests whether the locally available PigBench YOLOv8-s detector checkpoint can produce detector-backbone crop embeddings. This avoids generating dummy/untrained embeddings.

## Smoke test result

| check                 | status   | details                                                                         |
|:----------------------|:---------|:--------------------------------------------------------------------------------|
| environment           | PASS     | cv2=True; mmdet=True; config_exists=True; checkpoint_exists=True; device=cuda:0 |
| usable_crop_selection | PASS     | usable_crops=5                                                                  |
| model_load            | PASS     | class=YOLODetector; has_extract_feat=True                                       |
| embedding_extraction  | PASS     | embedding_dims=[896, 896, 896, 896, 896]; finite_ok=True                        |

## Embedding preview

|   sample_id | scan_frame_id   |   det_id | crop_shape    |   embedding_dim |   embedding_mean |   embedding_std |   embedding_l2_norm | feature_maps_type   | status   |
|------------:|:----------------|---------:|:--------------|----------------:|-----------------:|----------------:|--------------------:|:--------------------|:---------|
|           0 | scanframe_0015  |        0 | (158, 224, 3) |             896 |          1.07616 |         2.48416 |             81.0367 | tuple               | ok       |
|           1 | scanframe_0039  |        0 | (130, 154, 3) |             896 |          1.28157 |         3.01116 |             97.9577 | tuple               | ok       |
|           2 | scanframe_0039  |        1 | (113, 77, 3)  |             896 |          1.2979  |         2.86147 |             94.0523 | tuple               | ok       |
|           3 | scanframe_0000  |        0 | (156, 251, 3) |             896 |          1.24642 |         2.88604 |             94.1008 | tuple               | ok       |
|           4 | scanframe_0034  |        0 | (111, 202, 3) |             896 |          1.09109 |         2.49859 |             81.6109 | tuple               | ok       |

## Decision

| decision                                       | value                                               | reason                                                                                           |
|:-----------------------------------------------|:----------------------------------------------------|:-------------------------------------------------------------------------------------------------|
| can_run_full_detector_backbone_crop_embeddings | True                                                | Detector model loaded and extract_feat produced finite fixed-length embeddings for sample crops. |
| selected_method                                | mmdet_yolov8s_pig_detector_backbone_global_avg_pool | Uses locally available pig detector checkpoint, not random/untrained embeddings.                 |

## Interpretation

The detector-backed embedding route is usable. Next step: extract embeddings for all 540 detector crops and save a full feature table.
