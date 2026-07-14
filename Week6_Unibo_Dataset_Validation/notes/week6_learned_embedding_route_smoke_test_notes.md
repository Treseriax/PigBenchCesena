# Week 6 Learned Embedding Route Smoke Test

## Purpose

This smoke test checks which learned crop embedding route can be used without overclaiming. Untrained models are tested only as a control and should not be used as final learned semantic embeddings.

## Smoke test results

| route                                | status   | pretrained              | embedding_dim   | device   | evidence                                                         | recommendation                                         |
|:-------------------------------------|:---------|:------------------------|:----------------|:---------|:-----------------------------------------------------------------|:-------------------------------------------------------|
| test_crop                            | FAIL     |                         |                 | cuda:0   | RuntimeError: No usable crop found.                              | Fix crop loading first.                                |
| mmdet_yolov8s_detector_backbone_load | PASS     | pig_detector_checkpoint | unknown         | cuda:0   | loaded detector model; has_extract_feat=True; class=YOLODetector | Candidate for final learned crop embedding extraction. |

## Decision

| decision                               | value                                | reason                                                                |
|:---------------------------------------|:-------------------------------------|:----------------------------------------------------------------------|
| selected_embedding_route               | mmdet_yolov8s_detector_backbone_load | A pretrained or detector-checkpoint-backed route passed smoke test.   |
| can_generate_final_embedding_table_now | True                                 | Only true if a pretrained or detector-checkpoint-backed route passed. |

## Interpretation

The selected route is `mmdet_yolov8s_detector_backbone_load`. Next step is to run this route over all 540 bbox crops and save the embedding table with a clear method note.
