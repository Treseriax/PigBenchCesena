# Detector / Bbox Resource Audit Notes

## Purpose

This audit checks which detection packages, checkpoints, configs, and inference scripts are available for linking bounding boxes to the Week 6 scan-sampling frames.

## Python package audit

| package     | status   | version      |   returncode | stderr                                             |
|:------------|:---------|:-------------|-------------:|:---------------------------------------------------|
| torch       | FOUND    | 2.0.0+cu118  |            0 |                                                    |
| torchvision | FOUND    | 0.15.1+cu118 |            0 |                                                    |
| cv2         | FOUND    | 4.11.0       |            0 |                                                    |
| mmdet       | FOUND    | 3.3.0        |            0 |                                                    |
| mmcv        | FOUND    | 2.0.0        |            0 |                                                    |
| mmengine    | FOUND    | 0.10.7       |            0 |                                                    |
| ultralytics | MISSING  |              |            1 | Traceback (most recent call last):                 |
|             |          |              |              |   File "<string>", line 4, in <module>             |
|             |          |              |              | ModuleNotFoundError: No module named 'ultralytics' |
| numpy       | FOUND    | 1.26.4       |            0 |                                                    |
| pandas      | FOUND    | 2.3.3        |            0 |                                                    |

## PigBench detector resources

| path                                                                                                    | relative_path                                                                     | suffix   |   size_mb |   keyword_score | resource_guess   |
|:--------------------------------------------------------------------------------------------------------|:----------------------------------------------------------------------------------|:---------|----------:|----------------:|:-----------------|
| /home/oyavuz/PigBench/detection/data/pretrained_weights/yolov8_pigs/yolov8_s.pth                        | detection/data/pretrained_weights/yolov8_pigs/yolov8_s.pth                        | .pth     |    42.699 |               4 | checkpoint       |
| /home/oyavuz/PigBench/detection/configs/yolov8/yolov8_s.py                                              | detection/configs/yolov8/yolov8_s.py                                              | .py      |     0.011 |               5 | config_or_script |
| /home/oyavuz/PigBench/detection/configs/yolov8/yolov8_m.py                                              | detection/configs/yolov8/yolov8_m.py                                              | .py      |     0.002 |               5 | config_or_script |
| /home/oyavuz/PigBench/detection/configs/yolov8/yolov8_l.py                                              | detection/configs/yolov8/yolov8_l.py                                              | .py      |     0.001 |               5 | config_or_script |
| /home/oyavuz/PigBench/detection/configs/yolov8/yolov8_x.py                                              | detection/configs/yolov8/yolov8_x.py                                              | .py      |     0.001 |               5 | config_or_script |
| /home/oyavuz/PigBench/detection/outputs/yolov8_s_official_test/yolov8_s.py                              | detection/outputs/yolov8_s_official_test/yolov8_s.py                              | .py      |     0.019 |               4 | config_or_script |
| /home/oyavuz/PigBench/tracking/motrv2/detector/co_detr/configs/co_dino_r50_lsj.py                       | tracking/motrv2/detector/co_detr/configs/co_dino_r50_lsj.py                       | .py      |     0.014 |               4 | config_or_script |
| /home/oyavuz/PigBench/detection/configs/yolox/yolox_s.py                                                | detection/configs/yolox/yolox_s.py                                                | .py      |     0.01  |               4 | config_or_script |
| /home/oyavuz/PigBench/Week2_Tracking_Study/scripts/03_detect_sequence_yolov8s.py                        | Week2_Tracking_Study/scripts/03_detect_sequence_yolov8s.py                        | .py      |     0.004 |               4 | config_or_script |
| /home/oyavuz/PigBench/tracking/motrv2/detector/co_detr/configs/co_dino_swin.py                          | tracking/motrv2/detector/co_detr/configs/co_dino_swin.py                          | .py      |     0.004 |               4 | config_or_script |
| /home/oyavuz/PigBench/Week3_Behaviour_Dataset/final_outputs/scripts/08_detect_unibo_sequence_yolov8s.py | Week3_Behaviour_Dataset/final_outputs/scripts/08_detect_unibo_sequence_yolov8s.py | .py      |     0.004 |               4 | config_or_script |
| /home/oyavuz/PigBench/Week3_Behaviour_Dataset/scripts/08_detect_unibo_sequence_yolov8s.py               | Week3_Behaviour_Dataset/scripts/08_detect_unibo_sequence_yolov8s.py               | .py      |     0.004 |               4 | config_or_script |
| /home/oyavuz/PigBench/tracking/motrv2/detector/co_detr/configs/co_dino_r50.py                           | tracking/motrv2/detector/co_detr/configs/co_dino_r50.py                           | .py      |     0.003 |               4 | config_or_script |
| /home/oyavuz/PigBench/detection/configs/yolox/yolox_p5_tta.py                                           | detection/configs/yolox/yolox_p5_tta.py                                           | .py      |     0.002 |               4 | config_or_script |
| /home/oyavuz/PigBench/detection/configs/yolox/yolox_s_rtmdet_hyp.py                                     | detection/configs/yolox/yolox_s_rtmdet_hyp.py                                     | .py      |     0.002 |               4 | config_or_script |
| /home/oyavuz/PigBench/detection/configs/yolox/yolox_x.py                                                | detection/configs/yolox/yolox_x.py                                                | .py      |     0.001 |               4 | config_or_script |
| /home/oyavuz/PigBench/detection/configs/yolox/yolox_l.py                                                | detection/configs/yolox/yolox_l.py                                                | .py      |     0.001 |               4 | config_or_script |
| /home/oyavuz/PigBench/detection/configs/yolox/yolox_m_rtmdet_hyp.py                                     | detection/configs/yolox/yolox_m_rtmdet_hyp.py                                     | .py      |     0.001 |               4 | config_or_script |
| /home/oyavuz/PigBench/tracking/motrv2/detector/co_detr/configs/__init__.py                              | tracking/motrv2/detector/co_detr/configs/__init__.py                              | .py      |     0     |               4 | config_or_script |
| /home/oyavuz/PigBench/detection/outputs/yolov8_s_official_test/20260609_234514/20260609_234514.json     | detection/outputs/yolov8_s_official_test/20260609_234514/20260609_234514.json     | .json    |     0     |               4 | config_or_script |
| /home/oyavuz/PigBench/detection/configs/co_detr/model/transformer.py                                    | detection/configs/co_detr/model/transformer.py                                    | .py      |     0.056 |               3 | config_or_script |
| /home/oyavuz/PigBench/tracking/motrv2/detector/co_detr/model/transformer.py                             | tracking/motrv2/detector/co_detr/model/transformer.py                             | .py      |     0.056 |               3 | config_or_script |
| /home/oyavuz/PigBench/tracking/boxmot/detector/co_detr/model/transformer.py                             | tracking/boxmot/detector/co_detr/model/transformer.py                             | .py      |     0.056 |               3 | config_or_script |
| /home/oyavuz/PigBench/detection/configs/co_detr/model/co_dino_head.py                                   | detection/configs/co_detr/model/co_dino_head.py                                   | .py      |     0.028 |               3 | config_or_script |
| /home/oyavuz/PigBench/tracking/motrv2/detector/co_detr/model/co_dino_head.py                            | tracking/motrv2/detector/co_detr/model/co_dino_head.py                            | .py      |     0.028 |               3 | config_or_script |
| /home/oyavuz/PigBench/tracking/boxmot/detector/co_detr/model/co_dino_head.py                            | tracking/boxmot/detector/co_detr/model/co_dino_head.py                            | .py      |     0.028 |               3 | config_or_script |
| /home/oyavuz/PigBench/detection/configs/co_detr/co_dino_r50_lsj.py                                      | detection/configs/co_detr/co_dino_r50_lsj.py                                      | .py      |     0.014 |               3 | config_or_script |
| /home/oyavuz/PigBench/tracking/boxmot/detector/co_detr/co_dino_r50_lsj.py                               | tracking/boxmot/detector/co_detr/co_dino_r50_lsj.py                               | .py      |     0.014 |               3 | config_or_script |
| /home/oyavuz/PigBench/detection/configs/co_detr/model/codetr.py                                         | detection/configs/co_detr/model/codetr.py                                         | .py      |     0.013 |               3 | config_or_script |
| /home/oyavuz/PigBench/tracking/motrv2/detector/co_detr/model/codetr.py                                  | tracking/motrv2/detector/co_detr/model/codetr.py                                  | .py      |     0.013 |               3 | config_or_script |
| /home/oyavuz/PigBench/tracking/boxmot/detector/co_detr/model/codetr.py                                  | tracking/boxmot/detector/co_detr/model/codetr.py                                  | .py      |     0.013 |               3 | config_or_script |
| /home/oyavuz/PigBench/Week6_Unibo_Dataset_Validation/scripts/12_audit_detector_bbox_resources.py        | Week6_Unibo_Dataset_Validation/scripts/12_audit_detector_bbox_resources.py        | .py      |     0.006 |               3 | config_or_script |
| /home/oyavuz/PigBench/detection/configs/co_detr/model/co_atss_head.py                                   | detection/configs/co_detr/model/co_atss_head.py                                   | .py      |     0.006 |               3 | config_or_script |
| /home/oyavuz/PigBench/tracking/motrv2/detector/co_detr/model/co_atss_head.py                            | tracking/motrv2/detector/co_detr/model/co_atss_head.py                            | .py      |     0.006 |               3 | config_or_script |
| /home/oyavuz/PigBench/tracking/boxmot/detector/co_detr/model/co_atss_head.py                            | tracking/boxmot/detector/co_detr/model/co_atss_head.py                            | .py      |     0.006 |               3 | config_or_script |
| /home/oyavuz/PigBench/detection/configs/co_detr/model/co_roi_head.py                                    | detection/configs/co_detr/model/co_roi_head.py                                    | .py      |     0.005 |               3 | config_or_script |
| /home/oyavuz/PigBench/tracking/motrv2/detector/co_detr/model/co_roi_head.py                             | tracking/motrv2/detector/co_detr/model/co_roi_head.py                             | .py      |     0.005 |               3 | config_or_script |
| /home/oyavuz/PigBench/tracking/boxmot/detector/co_detr/model/co_roi_head.py                             | tracking/boxmot/detector/co_detr/model/co_roi_head.py                             | .py      |     0.005 |               3 | config_or_script |
| /home/oyavuz/PigBench/detection/train_mmdet.py                                                          | detection/train_mmdet.py                                                          | .py      |     0.004 |               3 | config_or_script |
| /home/oyavuz/PigBench/detection/train_mmyolo.py                                                         | detection/train_mmyolo.py                                                         | .py      |     0.004 |               3 | config_or_script |
| /home/oyavuz/PigBench/detection/configs/co_detr/co_dino_swin.py                                         | detection/configs/co_detr/co_dino_swin.py                                         | .py      |     0.004 |               3 | config_or_script |
| /home/oyavuz/PigBench/tracking/boxmot/detector/co_detr/co_dino_swin.py                                  | tracking/boxmot/detector/co_detr/co_dino_swin.py                                  | .py      |     0.004 |               3 | config_or_script |
| /home/oyavuz/PigBench/detection/configs/co_detr/co_dino_r50.py                                          | detection/configs/co_detr/co_dino_r50.py                                          | .py      |     0.003 |               3 | config_or_script |
| /home/oyavuz/PigBench/tracking/boxmot/detector/co_detr/co_dino_r50.py                                   | tracking/boxmot/detector/co_detr/co_dino_r50.py                                   | .py      |     0.003 |               3 | config_or_script |
| /home/oyavuz/PigBench/detection/tools/inference/visualization_utils.py                                  | detection/tools/inference/visualization_utils.py                                  | .py      |     0.002 |               3 | config_or_script |
| /home/oyavuz/PigBench/detection/configs/co_detr/model/__init__.py                                       | detection/configs/co_detr/model/__init__.py                                       | .py      |     0.001 |               3 | config_or_script |
| /home/oyavuz/PigBench/tracking/motrv2/detector/co_detr/model/__init__.py                                | tracking/motrv2/detector/co_detr/model/__init__.py                                | .py      |     0.001 |               3 | config_or_script |
| /home/oyavuz/PigBench/tracking/boxmot/detector/co_detr/model/__init__.py                                | tracking/boxmot/detector/co_detr/model/__init__.py                                | .py      |     0.001 |               3 | config_or_script |
| /home/oyavuz/PigBench/detection/configs/__init__.py                                                     | detection/configs/__init__.py                                                     | .py      |     0     |               3 | config_or_script |
| /home/oyavuz/PigBench/detection/configs/co_detr/__init__.py                                             | detection/configs/co_detr/__init__.py                                             | .py      |     0     |               3 | config_or_script |
| /home/oyavuz/PigBench/tracking/motrv2/detector/__init__.py                                              | tracking/motrv2/detector/__init__.py                                              | .py      |     0     |               3 | config_or_script |
| /home/oyavuz/PigBench/tracking/motrv2/detector/co_detr/__init__.py                                      | tracking/motrv2/detector/co_detr/__init__.py                                      | .py      |     0     |               3 | config_or_script |
| /home/oyavuz/PigBench/tracking/boxmot/detector/__init__.py                                              | tracking/boxmot/detector/__init__.py                                              | .py      |     0     |               3 | config_or_script |
| /home/oyavuz/PigBench/tracking/boxmot/detector/co_detr/__init__.py                                      | tracking/boxmot/detector/co_detr/__init__.py                                      | .py      |     0     |               3 | config_or_script |
| /home/oyavuz/PigBench/data/official_dataverse/PigDetect/train.json                                      | data/official_dataverse/PigDetect/train.json                                      | .json    |    11.33  |               2 | config_or_script |
| /home/oyavuz/PigBench/detection/data/PigDetect/train.json                                               | detection/data/PigDetect/train.json                                               | .json    |    11.33  |               2 | config_or_script |
| /home/oyavuz/PigBench/data/official_dataverse/PigDetect/test.json                                       | data/official_dataverse/PigDetect/test.json                                       | .json    |     1.808 |               2 | config_or_script |
| /home/oyavuz/PigBench/detection/data/PigDetect/test.json                                                | detection/data/PigDetect/test.json                                                | .json    |     1.808 |               2 | config_or_script |
| /home/oyavuz/PigBench/data/official_dataverse/PigDetect/val.json                                        | data/official_dataverse/PigDetect/val.json                                        | .json    |     1.201 |               2 | config_or_script |
| /home/oyavuz/PigBench/detection/data/PigDetect/val.json                                                 | detection/data/PigDetect/val.json                                                 | .json    |     1.201 |               2 | config_or_script |
| /home/oyavuz/PigBench/tracking/motip/engines/inference_engine.py                                        | tracking/motip/engines/inference_engine.py                                        | .py      |     0.017 |               2 | config_or_script |
| /home/oyavuz/PigBench/tracking/motrv2/engines/inference_engine.py                                       | tracking/motrv2/engines/inference_engine.py                                       | .py      |     0.012 |               2 | config_or_script |
| /home/oyavuz/PigBench/tracking/boxmot/boxmot/appearance/backbones/clip/config/defaults.py               | tracking/boxmot/boxmot/appearance/backbones/clip/config/defaults.py               | .py      |     0.008 |               2 | config_or_script |
| /home/oyavuz/PigBench/tracking/boxmot/engines/inference_engine.py                                       | tracking/boxmot/engines/inference_engine.py                                       | .py      |     0.007 |               2 | config_or_script |
| /home/oyavuz/PigBench/tracking/boxmot/boxmot/appearance/backbones/clip/config/defaults_base.py          | tracking/boxmot/boxmot/appearance/backbones/clip/config/defaults_base.py          | .py      |     0.006 |               2 | config_or_script |
| /home/oyavuz/PigBench/detection/test.py                                                                 | detection/test.py                                                                 | .py      |     0.005 |               2 | config_or_script |
| /home/oyavuz/PigBench/detection/tools/download/download.py                                              | detection/tools/download/download.py                                              | .py      |     0.005 |               2 | config_or_script |
| /home/oyavuz/PigBench/tracking/motrv2/datasets/inference_datasets.py                                    | tracking/motrv2/datasets/inference_datasets.py                                    | .py      |     0.004 |               2 | config_or_script |
| /home/oyavuz/PigBench/tracking/motip/configs/motip.yaml                                                 | tracking/motip/configs/motip.yaml                                                 | .yaml    |     0.004 |               2 | config_or_script |
| /home/oyavuz/PigBench/tracking/tools/download/restructure_pigdetect.py                                  | tracking/tools/download/restructure_pigdetect.py                                  | .py      |     0.003 |               2 | config_or_script |
| /home/oyavuz/PigBench/tracking/motip/configs/base.yaml                                                  | tracking/motip/configs/base.yaml                                                  | .yaml    |     0.003 |               2 | config_or_script |
| /home/oyavuz/PigBench/tracking/motip/configs/detr.yaml                                                  | tracking/motip/configs/detr.yaml                                                  | .yaml    |     0.003 |               2 | config_or_script |
| /home/oyavuz/PigBench/tracking/motip/configs/utils.py                                                   | tracking/motip/configs/utils.py                                                   | .py      |     0.002 |               2 | config_or_script |
| /home/oyavuz/PigBench/tracking/motrv2/util/checkpoint.py                                                | tracking/motrv2/util/checkpoint.py                                                | .py      |     0.002 |               2 | config_or_script |
| /home/oyavuz/PigBench/tracking/motrv2/configs/base.yaml                                                 | tracking/motrv2/configs/base.yaml                                                 | .yaml    |     0.002 |               2 | config_or_script |
| /home/oyavuz/PigBench/tracking/motrv2/configs/motrv2.yaml                                               | tracking/motrv2/configs/motrv2.yaml                                               | .yaml    |     0.002 |               2 | config_or_script |
| /home/oyavuz/PigBench/detection/tools/download/restructure.py                                           | detection/tools/download/restructure.py                                           | .py      |     0.001 |               2 | config_or_script |
| /home/oyavuz/PigBench/tracking/boxmot/boxmot/trackers/strongsort/sort/detection.py                      | tracking/boxmot/boxmot/trackers/strongsort/sort/detection.py                      | .py      |     0.001 |               2 | config_or_script |
| /home/oyavuz/PigBench/Week2_Tracking_Study/configs/botsort_no_reid.yaml                                 | Week2_Tracking_Study/configs/botsort_no_reid.yaml                                 | .yaml    |     0.001 |               2 | config_or_script |
| /home/oyavuz/PigBench/Week2_Tracking_Study/configs/deepocsort_no_reid.yaml                              | Week2_Tracking_Study/configs/deepocsort_no_reid.yaml                              | .yaml    |     0.001 |               2 | config_or_script |

## /work detector resources

| path                                                                                   | suffix   |   size_mb |   keyword_score | resource_guess       |
|:---------------------------------------------------------------------------------------|:---------|----------:|----------------:|:---------------------|
| /work/pig/datasets/PigDetect-IoU-YOLOv8/PigDetect-IoU-YOLOv8/YOLOv8-IoU-x.pt           | .pt      |   130.389 |               4 | checkpoint           |
| /work/pig/datasets/PigDetect-IoU-YOLOv8/PigDetect-IoU-YOLOv8/YOLOv8-IoU-x.pt           | .pt      |   130.389 |               4 | checkpoint           |
| /work/pig/datasets/PigDetect-IoU-RT-DETR/rtdetr-IoU-x.pt                               | .pt      |   129.109 |               4 | checkpoint           |
| /work/pig/datasets/PigDetect-YOLOv8/PigDetect-YOLOv8/YOLOv8-x.pt                       | .pt      |   129.109 |               4 | checkpoint           |
| /work/pig/datasets/PigDetect-IoU-RT-DETR/rtdetr-IoU-x.pt                               | .pt      |   129.109 |               4 | checkpoint           |
| /work/pig/datasets/PigDetect-YOLOv8/PigDetect-YOLOv8/YOLOv8-x.pt                       | .pt      |   129.109 |               4 | checkpoint           |
| /work/pig/datasets/PigDetect-RT-DETR/PigDetect-RT-DETR/rtdetr-x.pt                     | .pt      |   129.099 |               4 | checkpoint           |
| /work/pig/datasets/PigDetect-RT-DETR/PigDetect-RT-DETR/rtdetr-IoU-x.pt                 | .pt      |   129.099 |               4 | checkpoint           |
| /work/pig/datasets/PigDetect-RT-DETR/PigDetect-RT-DETR/rtdetr-x.pt                     | .pt      |   129.099 |               4 | checkpoint           |
| /work/pig/datasets/PigDetect-RT-DETR/PigDetect-RT-DETR/rtdetr-IoU-x.pt                 | .pt      |   129.099 |               4 | checkpoint           |
| /work/pig/datasets/PigDetect-IoU-YOLOv8/PigDetect-IoU-YOLOv8/YOLOv8-IoU-l.pt           | .pt      |    83.591 |               4 | checkpoint           |
| /work/pig/datasets/PigDetect-IoU-YOLOv8/PigDetect-IoU-YOLOv8/YOLOv8-IoU-l.pt           | .pt      |    83.591 |               4 | checkpoint           |
| /work/pig/datasets/PigDetect-IoU-RT-DETR/rtdetr-IoU-l.pt                               | .pt      |    63.092 |               4 | checkpoint           |
| /work/pig/datasets/PigDetect-YOLOv8/PigDetect-YOLOv8/YOLOv8-l.pt                       | .pt      |    63.092 |               4 | checkpoint           |
| /work/pig/datasets/PigDetect-IoU-RT-DETR/rtdetr-IoU-l.pt                               | .pt      |    63.092 |               4 | checkpoint           |
| /work/pig/datasets/PigDetect-YOLOv8/PigDetect-YOLOv8/YOLOv8-l.pt                       | .pt      |    63.092 |               4 | checkpoint           |
| /work/pig/datasets/PigDetect-RT-DETR/PigDetect-RT-DETR/rtdetr-IoU-l.pt                 | .pt      |    63.08  |               4 | checkpoint           |
| /work/pig/datasets/PigDetect-RT-DETR/PigDetect-RT-DETR/rtdetr-l.pt                     | .pt      |    63.08  |               4 | checkpoint           |
| /work/pig/datasets/PigDetect-RT-DETR/PigDetect-RT-DETR/rtdetr-IoU-l.pt                 | .pt      |    63.08  |               4 | checkpoint           |
| /work/pig/datasets/PigDetect-RT-DETR/PigDetect-RT-DETR/rtdetr-l.pt                     | .pt      |    63.08  |               4 | checkpoint           |
| /work/pig/datasets/PigDetect-IoU-YOLOv8/PigDetect-IoU-YOLOv8/YOLOv8-IoU-m.pt           | .pt      |    49.624 |               4 | checkpoint           |
| /work/pig/datasets/PigDetect-YOLOv8/PigDetect-YOLOv8/YOLOv8-m.pt                       | .pt      |    49.624 |               4 | checkpoint           |
| /work/pig/datasets/PigDetect-IoU-YOLOv8/PigDetect-IoU-YOLOv8/YOLOv8-IoU-m.pt           | .pt      |    49.624 |               4 | checkpoint           |
| /work/pig/datasets/PigDetect-YOLOv8/PigDetect-YOLOv8/YOLOv8-m.pt                       | .pt      |    49.624 |               4 | checkpoint           |
| /work/pig/datasets/PigDetect-IoU-YOLOv8/PigDetect-IoU-YOLOv8/YOLOv8-IoU-s.pt           | .pt      |    21.48  |               4 | checkpoint           |
| /work/pig/datasets/PigDetect-YOLOv8/PigDetect-YOLOv8/YOLOv8-s.pt                       | .pt      |    21.48  |               4 | checkpoint           |
| /work/pig/datasets/PigDetect-IoU-YOLOv8/PigDetect-IoU-YOLOv8/YOLOv8-IoU-s.pt           | .pt      |    21.48  |               4 | checkpoint           |
| /work/pig/datasets/PigDetect-YOLOv8/PigDetect-YOLOv8/YOLOv8-s.pt                       | .pt      |    21.48  |               4 | checkpoint           |
| /work/pig/datasets/PigDetect-IoU-YOLOv8/PigDetect-IoU-YOLOv8/YOLOv8-IoU-n.pt           | .pt      |     5.965 |               4 | checkpoint           |
| /work/pig/datasets/PigDetect-YOLOv8/PigDetect-YOLOv8/YOLOv8-n.pt                       | .pt      |     5.965 |               4 | checkpoint           |
| /work/pig/datasets/PigDetect-IoU-YOLOv8/PigDetect-IoU-YOLOv8/YOLOv8-IoU-n.pt           | .pt      |     5.965 |               4 | checkpoint           |
| /work/pig/datasets/PigDetect-YOLOv8/PigDetect-YOLOv8/YOLOv8-n.pt                       | .pt      |     5.965 |               4 | checkpoint           |
| /work/pig/datasets/PigDetect-High_IoU/detect_harder/train/labels/annotations_coco.json | .json    |     8.424 |               2 | config_or_annotation |
| /work/pig/datasets/PigDetect-High_IoU/detect_harder/train/labels/annotations_coco.json | .json    |     8.424 |               2 | config_or_annotation |
| /work/pig/datasets/PigDetect-Random/detect/train/labels/annotations_coco.json          | .json    |     8.195 |               2 | config_or_annotation |
| /work/pig/datasets/PigDetect-Random/detect/train/labels/annotations_coco.json          | .json    |     8.195 |               2 | config_or_annotation |
| /work/pig/datasets/PigDetect-Random/detect/valid/labels/annotations_coco.json          | .json    |     0.919 |               2 | config_or_annotation |
| /work/pig/datasets/PigDetect-Random/detect/valid/labels/annotations_coco.json          | .json    |     0.919 |               2 | config_or_annotation |
| /work/pig/datasets/PigDetect-High_IoU/detect_harder/valid/labels/annotations_coco.json | .json    |     0.916 |               2 | config_or_annotation |
| /work/pig/datasets/PigDetect-High_IoU/detect_harder/valid/labels/annotations_coco.json | .json    |     0.916 |               2 | config_or_annotation |
| /work/pig/datasets/PigDetect-High_IoU/detect_harder/test/labels/annotations_coco.json  | .json    |     0.668 |               2 | config_or_annotation |
| /work/pig/datasets/PigDetect-Random/detect/test/labels/annotations_coco.json           | .json    |     0.668 |               2 | config_or_annotation |
| /work/pig/datasets/PigDetect-High_IoU/detect_harder/test/labels/annotations_coco.json  | .json    |     0.668 |               2 | config_or_annotation |
| /work/pig/datasets/PigDetect-Random/detect/test/labels/annotations_coco.json           | .json    |     0.668 |               2 | config_or_annotation |
| /work/pig/code/PigBench/tracking/motip/configs/motip.yaml                              | .yaml    |     0.004 |               2 | config_or_annotation |
| /work/pig/code/PigBench/tracking/motip/configs/base.yaml                               | .yaml    |     0.003 |               2 | config_or_annotation |
| /work/pig/code/PigBench/tracking/motip/configs/detr.yaml                               | .yaml    |     0.003 |               2 | config_or_annotation |
| /work/pig/code/PigBench/tracking/motrv2/configs/base.yaml                              | .yaml    |     0.002 |               2 | config_or_annotation |
| /work/pig/code/PigBench/tracking/motrv2/configs/motrv2.yaml                            | .yaml    |     0.002 |               2 | config_or_annotation |
| /work/pig/code/PigBench/tracking/boxmot/configs/base.yaml                              | .yaml    |     0.001 |               2 | config_or_annotation |
| /work/pig/code/PigBench/tracking/boxmot/configs/trackers/ocsort.yaml                   | .yaml    |     0.001 |               2 | config_or_annotation |
| /work/pig/code/PigBench/tracking/boxmot/configs/trackers/imprassoc.yaml                | .yaml    |     0.001 |               2 | config_or_annotation |
| /work/pig/code/PigBench/tracking/boxmot/configs/trackers/botsort.yaml                  | .yaml    |     0.001 |               2 | config_or_annotation |
| /work/pig/code/PigBench/tracking/boxmot/configs/trackers/deepocsort.yaml               | .yaml    |     0.001 |               2 | config_or_annotation |
| /work/pig/datasets/PigDetect-High_IoU/detect_harder/data.yaml                          | .yaml    |     0     |               2 | config_or_annotation |
| /work/pig/datasets/PigDetect-Random/detect/data.yaml                                   | .yaml    |     0     |               2 | config_or_annotation |
| /work/pig/datasets/PigDetect-High_IoU/detect_harder/data.yaml                          | .yaml    |     0     |               2 | config_or_annotation |
| /work/pig/datasets/PigDetect-Random/detect/data.yaml                                   | .yaml    |     0     |               2 | config_or_annotation |
| /work/pig/code/PigBench/tracking/boxmot/configs/strongsort.yaml                        | .yaml    |     0     |               2 | config_or_annotation |
| /work/pig/code/PigBench/tracking/boxmot/configs/ocsort.yaml                            | .yaml    |     0     |               2 | config_or_annotation |
| /work/pig/code/PigBench/tracking/boxmot/configs/imprassoc.yaml                         | .yaml    |     0     |               2 | config_or_annotation |
| /work/pig/code/PigBench/tracking/boxmot/configs/botsort.yaml                           | .yaml    |     0     |               2 | config_or_annotation |
| /work/pig/code/PigBench/tracking/boxmot/configs/deepocsort.yaml                        | .yaml    |     0     |               2 | config_or_annotation |
| /work/pig/code/PigBench/tracking/boxmot/configs/bytetrack.yaml                         | .yaml    |     0     |               2 | config_or_annotation |
| /work/pig/code/PigBench/tracking/boxmot/configs/trackers/strongsort.yaml               | .yaml    |     0     |               2 | config_or_annotation |
| /work/pig/code/PigBench/tracking/boxmot/configs/trackers/bytetrack.yaml                | .yaml    |     0     |               2 | config_or_annotation |
| /work/pig/datasets/training_information.yaml                                           | .yaml    |     0.002 |               1 | config_or_annotation |
| /work/pig/training_information.yaml                                                    | .yaml    |     0.002 |               1 | config_or_annotation |
| /work/pig/datasets/training_information.yaml                                           | .yaml    |     0.002 |               1 | config_or_annotation |

## Interpretation

The next step is to select the most reliable detector route. If MMDetection and the PigBench YOLOv8 checkpoint/config are available, we can run inference on the 72 extracted scanpoint frames. If not, we will fall back to a simpler available detector route or document bbox extraction as pending.
