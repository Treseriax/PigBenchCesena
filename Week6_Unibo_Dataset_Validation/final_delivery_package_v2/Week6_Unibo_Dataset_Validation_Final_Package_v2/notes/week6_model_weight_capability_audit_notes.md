# Week 6 Model / Weight Capability Audit

## Purpose

This audit checks which model packages, weights, and configs are available locally before attempting ideal segmentation or learned embedding extraction.

## Package capability

| package          | available   | version      | possible_use                                           |
|:-----------------|:------------|:-------------|:-------------------------------------------------------|
| torch            | True        | 2.0.0+cu118  | embedding extraction / model inference                 |
| torchvision      | True        | 0.15.1+cu118 | ResNet/ViT style crop embeddings if weights available  |
| cv2              | True        | 4.11.0       | segmentation baseline / image processing               |
| numpy            | True        | 1.26.4       |                                                        |
| pandas           | True        | 2.3.3        |                                                        |
| mmdet            | True        | 3.3.0        | detector and possible model backbone features          |
| mmcv             | True        | 2.0.0        | mmdet dependency                                       |
| mmengine         | True        | 0.10.7       | mmdet dependency                                       |
| ultralytics      | False       |              | YOLO-seg / YOLO embeddings if installed                |
| segment_anything | False       |              | SAM segmentation if installed with weights             |
| sam2             | False       |              | SAM2 segmentation if installed with weights            |
| detectron2       | False       |              | Mask R-CNN segmentation if installed                   |
| timm             | True        | 1.0.27       | ViT/DINO/ConvNeXt embeddings if installed with weights |
| transformers     | False       |              | DINOv2/CLIP/ViT if installed with weights              |
| sklearn          | True        | 1.7.2        | feature normalization/PCA/UMAP if needed               |
| PIL              | True        | 12.2.0       | image loading                                          |

## Local weight inventory summary

| path                                                                                                                                                |   size_mb | keyword_hits   | candidate_type                 |
|:----------------------------------------------------------------------------------------------------------------------------------------------------|----------:|:---------------|:-------------------------------|
| /work/pig/datasets/PigDetect-IoU-YOLOv8/PigDetect-IoU-YOLOv8/YOLOv8-IoU-x.pt                                                                        |   130.389 | yolov8,yolo    | detector_or_tracking_candidate |
| /work/pig/datasets/PigDetect-IoU-RT-DETR/rtdetr-IoU-x.pt                                                                                            |   129.109 | rtdetr,rt-detr | detector_or_tracking_candidate |
| /work/pig/datasets/PigDetect-YOLOv8/PigDetect-YOLOv8/YOLOv8-x.pt                                                                                    |   129.109 | yolov8,yolo    | detector_or_tracking_candidate |
| /work/pig/datasets/PigDetect-RT-DETR/PigDetect-RT-DETR/rtdetr-IoU-x.pt                                                                              |   129.099 | rtdetr,rt-detr | detector_or_tracking_candidate |
| /work/pig/datasets/PigDetect-RT-DETR/PigDetect-RT-DETR/rtdetr-x.pt                                                                                  |   129.099 | rtdetr,rt-detr | detector_or_tracking_candidate |
| /work/pig/datasets/PigDetect-IoU-YOLOv8/PigDetect-IoU-YOLOv8/YOLOv8-IoU-l.pt                                                                        |    83.591 | yolov8,yolo    | detector_or_tracking_candidate |
| /work/pig/datasets/PigDetect-IoU-RT-DETR/rtdetr-IoU-l.pt                                                                                            |    63.092 | rtdetr,rt-detr | detector_or_tracking_candidate |
| /work/pig/datasets/PigDetect-YOLOv8/PigDetect-YOLOv8/YOLOv8-l.pt                                                                                    |    63.092 | yolov8,yolo    | detector_or_tracking_candidate |
| /work/pig/datasets/PigDetect-RT-DETR/PigDetect-RT-DETR/rtdetr-IoU-l.pt                                                                              |    63.08  | rtdetr,rt-detr | detector_or_tracking_candidate |
| /work/pig/datasets/PigDetect-RT-DETR/PigDetect-RT-DETR/rtdetr-l.pt                                                                                  |    63.08  | rtdetr,rt-detr | detector_or_tracking_candidate |
| /work/pig/datasets/PigDetect-IoU-YOLOv8/PigDetect-IoU-YOLOv8/YOLOv8-IoU-m.pt                                                                        |    49.624 | yolov8,yolo    | detector_or_tracking_candidate |
| /work/pig/datasets/PigDetect-YOLOv8/PigDetect-YOLOv8/YOLOv8-m.pt                                                                                    |    49.624 | yolov8,yolo    | detector_or_tracking_candidate |
| detection/data/pretrained_weights/yolov8_pigs/yolov8_s.pth                                                                                          |    42.699 | yolov8,yolo    | detector_or_tracking_candidate |
| /work/pig/datasets/PigDetect-IoU-YOLOv8/PigDetect-IoU-YOLOv8/YOLOv8-IoU-s.pt                                                                        |    21.48  | yolov8,yolo    | detector_or_tracking_candidate |
| /work/pig/datasets/PigDetect-YOLOv8/PigDetect-YOLOv8/YOLOv8-s.pt                                                                                    |    21.48  | yolov8,yolo    | detector_or_tracking_candidate |
| /work/pig/datasets/PigDetect-IoU-YOLOv8/PigDetect-IoU-YOLOv8/YOLOv8-IoU-n.pt                                                                        |     5.965 | yolov8,yolo    | detector_or_tracking_candidate |
| /work/pig/datasets/PigDetect-YOLOv8/PigDetect-YOLOv8/YOLOv8-n.pt                                                                                    |     5.965 | yolov8,yolo    | detector_or_tracking_candidate |
| /home/oyavuz/miniconda3/pkgs/torchvision-0.15.0-py310_cu118/info/test/test/expect/ModelTester.test_fasterrcnn_mobilenet_v3_large_320_fpn_expect.pkl |     0.004 | faster         | detector_or_tracking_candidate |
| /home/oyavuz/miniconda3/pkgs/torchvision-0.15.0-py310_cu118/info/test/test/expect/ModelTester.test_fasterrcnn_mobilenet_v3_large_fpn_expect.pkl     |     0.004 | faster         | detector_or_tracking_candidate |
| /work/models/Pytorch/RetinaFace/Resnet50_Final.pth                                                                                                  |   104.425 | resnet         | embedding_candidate            |
| /home/oyavuz/miniconda3/pkgs/torchvision-0.15.0-py310_cu118/info/test/test/expect/ModelTester.test_deeplabv3_resnet101_expect.pkl                   |     0.04  | resnet         | embedding_candidate            |
| /home/oyavuz/miniconda3/pkgs/torchvision-0.15.0-py310_cu118/info/test/test/expect/ModelTester.test_deeplabv3_resnet50_expect.pkl                    |     0.04  | resnet         | embedding_candidate            |
| /home/oyavuz/miniconda3/pkgs/torchvision-0.15.0-py310_cu118/info/test/test/expect/ModelTester.test_fcn_resnet101_expect.pkl                         |     0.04  | resnet         | embedding_candidate            |
| /home/oyavuz/miniconda3/pkgs/torchvision-0.15.0-py310_cu118/info/test/test/expect/ModelTester.test_fcn_resnet50_expect.pkl                          |     0.04  | resnet         | embedding_candidate            |
| /home/oyavuz/miniconda3/pkgs/torchvision-0.15.0-py310_cu118/info/test/test/expect/ModelTester.test_retinanet_resnet50_fpn_expect.pkl                |     0.009 | resnet         | embedding_candidate            |
| /home/oyavuz/miniconda3/pkgs/torchvision-0.15.0-py310_cu118/info/test/test/expect/ModelTester.test_retinanet_resnet50_fpn_v2_expect.pkl             |     0.009 | resnet         | embedding_candidate            |
| /home/oyavuz/miniconda3/pkgs/torchvision-0.15.0-py310_cu118/info/test/test/expect/ModelTester.test_fasterrcnn_resnet50_fpn_expect.pkl               |     0.004 | resnet,faster  | embedding_candidate            |
| /home/oyavuz/miniconda3/pkgs/torchvision-0.15.0-py310_cu118/info/test/test/expect/ModelTester.test_fasterrcnn_resnet50_fpn_v2_expect.pkl            |     0.004 | resnet,faster  | embedding_candidate            |
| /home/oyavuz/miniconda3/pkgs/torchvision-0.15.0-py310_cu118/info/test/test/expect/ModelTester.test_fcos_resnet50_fpn_expect.pkl                     |     0.003 | resnet         | embedding_candidate            |
| /home/oyavuz/miniconda3/pkgs/torchvision-0.15.0-py310_cu118/info/test/test/expect/ModelTester.test_keypointrcnn_resnet50_fpn_expect.pkl             |     0.003 | resnet         | embedding_candidate            |
| /home/oyavuz/miniconda3/pkgs/torchvision-0.15.0-py310_cu118/info/test/test/expect/ModelTester.test_convnext_base_expect.pkl                         |     0.001 | convnext       | embedding_candidate            |
| /home/oyavuz/miniconda3/pkgs/torchvision-0.15.0-py310_cu118/info/test/test/expect/ModelTester.test_convnext_large_expect.pkl                        |     0.001 | convnext       | embedding_candidate            |
| /home/oyavuz/miniconda3/pkgs/torchvision-0.15.0-py310_cu118/info/test/test/expect/ModelTester.test_convnext_small_expect.pkl                        |     0.001 | convnext       | embedding_candidate            |
| /home/oyavuz/miniconda3/pkgs/torchvision-0.15.0-py310_cu118/info/test/test/expect/ModelTester.test_convnext_tiny_expect.pkl                         |     0.001 | convnext       | embedding_candidate            |
| /home/oyavuz/miniconda3/pkgs/torchvision-0.15.0-py310_cu118/info/test/test/expect/ModelTester.test_maxvit_t_expect.pkl                              |     0.001 | vit            | embedding_candidate            |
| /home/oyavuz/miniconda3/pkgs/torchvision-0.15.0-py310_cu118/info/test/test/expect/ModelTester.test_mvit_v1_b_expect.pkl                             |     0.001 | vit            | embedding_candidate            |
| /home/oyavuz/miniconda3/pkgs/torchvision-0.15.0-py310_cu118/info/test/test/expect/ModelTester.test_mvit_v2_s_expect.pkl                             |     0.001 | vit            | embedding_candidate            |
| /home/oyavuz/miniconda3/pkgs/torchvision-0.15.0-py310_cu118/info/test/test/expect/ModelTester.test_resnet101_expect.pkl                             |     0.001 | resnet         | embedding_candidate            |
| /home/oyavuz/miniconda3/pkgs/torchvision-0.15.0-py310_cu118/info/test/test/expect/ModelTester.test_resnet152_expect.pkl                             |     0.001 | resnet         | embedding_candidate            |
| /home/oyavuz/miniconda3/pkgs/torchvision-0.15.0-py310_cu118/info/test/test/expect/ModelTester.test_resnet18_expect.pkl                              |     0.001 | resnet         | embedding_candidate            |
| /home/oyavuz/miniconda3/pkgs/torchvision-0.15.0-py310_cu118/info/test/test/expect/ModelTester.test_resnet18_quantized_expect.pkl                    |     0.001 | resnet         | embedding_candidate            |
| /home/oyavuz/miniconda3/pkgs/torchvision-0.15.0-py310_cu118/info/test/test/expect/ModelTester.test_resnet34_expect.pkl                              |     0.001 | resnet         | embedding_candidate            |
| /home/oyavuz/miniconda3/pkgs/torchvision-0.15.0-py310_cu118/info/test/test/expect/ModelTester.test_resnet50_expect.pkl                              |     0.001 | resnet         | embedding_candidate            |
| /home/oyavuz/miniconda3/pkgs/torchvision-0.15.0-py310_cu118/info/test/test/expect/ModelTester.test_resnet50_quantized_expect.pkl                    |     0.001 | resnet         | embedding_candidate            |
| /home/oyavuz/miniconda3/pkgs/torchvision-0.15.0-py310_cu118/info/test/test/expect/ModelTester.test_swin3d_b_expect.pkl                              |     0.001 | swin           | embedding_candidate            |
| /home/oyavuz/miniconda3/pkgs/torchvision-0.15.0-py310_cu118/info/test/test/expect/ModelTester.test_swin3d_s_expect.pkl                              |     0.001 | swin           | embedding_candidate            |
| /home/oyavuz/miniconda3/pkgs/torchvision-0.15.0-py310_cu118/info/test/test/expect/ModelTester.test_swin3d_t_expect.pkl                              |     0.001 | swin           | embedding_candidate            |
| /home/oyavuz/miniconda3/pkgs/torchvision-0.15.0-py310_cu118/info/test/test/expect/ModelTester.test_swin_b_expect.pkl                                |     0.001 | swin           | embedding_candidate            |
| /home/oyavuz/miniconda3/pkgs/torchvision-0.15.0-py310_cu118/info/test/test/expect/ModelTester.test_swin_s_expect.pkl                                |     0.001 | swin           | embedding_candidate            |
| /home/oyavuz/miniconda3/pkgs/torchvision-0.15.0-py310_cu118/info/test/test/expect/ModelTester.test_swin_t_expect.pkl                                |     0.001 | swin           | embedding_candidate            |

## Config inventory summary

| path                                                                                                  | keyword_hits           | candidate_type            |
|:------------------------------------------------------------------------------------------------------|:-----------------------|:--------------------------|
| Week2_Tracking_Study/configs/botsort_no_reid.yaml                                                     | bot,reid               | detector_config_candidate |
| Week2_Tracking_Study/configs/deepocsort_no_reid.yaml                                                  | reid                   | detector_config_candidate |
| Week2_Tracking_Study/scripts/03_detect_sequence_yolov8s.py                                            | yolo,yolov8            | detector_config_candidate |
| Week2_Tracking_Study/scripts/04_track_with_boxmot.py                                                  | bytetrack,reid,tracker | detector_config_candidate |
| Week3_Behaviour_Dataset/final_outputs/scripts/08_detect_unibo_sequence_yolov8s.py                     | yolo,yolov8            | detector_config_candidate |
| Week3_Behaviour_Dataset/final_outputs/scripts/09_track_unibo_with_boxmot.py                           | bytetrack,reid,tracker | detector_config_candidate |
| Week3_Behaviour_Dataset/scripts/08_detect_unibo_sequence_yolov8s.py                                   | yolo,yolov8            | detector_config_candidate |
| Week3_Behaviour_Dataset/scripts/09_track_unibo_with_boxmot.py                                         | bytetrack,reid,tracker | detector_config_candidate |
| Week6_Unibo_Dataset_Validation/outputs/feature_extractors/week6_crop_colour_marker_features.json      | yolo,yolov8            | detector_config_candidate |
| Week6_Unibo_Dataset_Validation/outputs/feature_extractors/week6_yolov8s_all_scanpoint_detections.json | yolo,yolov8            | detector_config_candidate |
| Week6_Unibo_Dataset_Validation/scripts/13_smoke_test_yolov8s_detector_on_scanpoint.py                 | yolo,yolov8            | detector_config_candidate |
| Week6_Unibo_Dataset_Validation/scripts/14_run_yolov8s_detector_on_all_scanpoints.py                   | yolo,yolov8            | detector_config_candidate |
| Week6_Unibo_Dataset_Validation/scripts/17_make_marker_bbox_overlay_visualization_v3.py                | bot,yolo,yolov8        | detector_config_candidate |
| Week6_Unibo_Dataset_Validation/scripts/22_make_bbox_count_warning_qc_contact_sheets.py                | bot                    | detector_config_candidate |
| Week6_Unibo_Dataset_Validation/scripts/25_week6_postfix_final_consistency_audit.py                    | yolo,yolov8            | detector_config_candidate |
| Week6_Unibo_Dataset_Validation/scripts/30_build_week6_final_delivery_package.py                       | yolo,yolov8            | detector_config_candidate |
| Week6_Unibo_Dataset_Validation/scripts/33_create_minimal_visualization_interface_demo.py              | bot,yolo,yolov8        | detector_config_candidate |
| Week6_Unibo_Dataset_Validation/scripts/34_build_lightweight_feature_extractor_tests.py                | bot,yolo,yolov8        | detector_config_candidate |
| Week6_Unibo_Dataset_Validation/scripts/36_generate_week6_shared_excel_task_tracker.py                 | tracker                | detector_config_candidate |
| detection/configs/yolov8/yolov8_l.py                                                                  | yolo,yolov8            | detector_config_candidate |
| detection/configs/yolov8/yolov8_l.py                                                                  | yolo,yolov8            | detector_config_candidate |
| detection/configs/yolov8/yolov8_m.py                                                                  | yolo,yolov8            | detector_config_candidate |
| detection/configs/yolov8/yolov8_m.py                                                                  | yolo,yolov8            | detector_config_candidate |
| detection/configs/yolov8/yolov8_s.py                                                                  | yolo,yolov8            | detector_config_candidate |
| detection/configs/yolov8/yolov8_s.py                                                                  | yolo,yolov8            | detector_config_candidate |
| detection/configs/yolov8/yolov8_x.py                                                                  | yolo,yolov8            | detector_config_candidate |
| detection/configs/yolov8/yolov8_x.py                                                                  | yolo,yolov8            | detector_config_candidate |
| detection/configs/yolox/yolox_l.py                                                                    | yolo                   | detector_config_candidate |
| detection/configs/yolox/yolox_l.py                                                                    | yolo                   | detector_config_candidate |
| detection/configs/yolox/yolox_m_rtmdet_hyp.py                                                         | yolo                   | detector_config_candidate |
| detection/configs/yolox/yolox_m_rtmdet_hyp.py                                                         | yolo                   | detector_config_candidate |
| detection/configs/yolox/yolox_p5_tta.py                                                               | yolo                   | detector_config_candidate |
| detection/configs/yolox/yolox_p5_tta.py                                                               | yolo                   | detector_config_candidate |
| detection/configs/yolox/yolox_s.py                                                                    | yolo                   | detector_config_candidate |
| detection/configs/yolox/yolox_s.py                                                                    | yolo                   | detector_config_candidate |
| detection/configs/yolox/yolox_s_rtmdet_hyp.py                                                         | yolo                   | detector_config_candidate |
| detection/configs/yolox/yolox_s_rtmdet_hyp.py                                                         | yolo                   | detector_config_candidate |
| detection/configs/yolox/yolox_x.py                                                                    | yolo                   | detector_config_candidate |
| detection/configs/yolox/yolox_x.py                                                                    | yolo                   | detector_config_candidate |
| detection/outputs/yolov8_s_official_test/20260609_234514/20260609_234514.json                         | yolo,yolov8            | detector_config_candidate |
| detection/outputs/yolov8_s_official_test/yolov8_s.py                                                  | yolo,yolov8            | detector_config_candidate |
| detection/train_mmyolo.py                                                                             | yolo                   | detector_config_candidate |
| tracking/boxmot/TrackEval/scripts/comparison_plots.py                                                 | tracker                | detector_config_candidate |
| tracking/boxmot/TrackEval/scripts/run_rob_mots.py                                                     | tracker                | detector_config_candidate |
| tracking/boxmot/TrackEval/tests/test_all_quick.py                                                     | tracker                | detector_config_candidate |
| tracking/boxmot/TrackEval/tests/test_davis.py                                                         | tracker                | detector_config_candidate |
| tracking/boxmot/TrackEval/tests/test_metrics.py                                                       | tracker                | detector_config_candidate |
| tracking/boxmot/TrackEval/tests/test_mot17.py                                                         | tracker                | detector_config_candidate |
| tracking/boxmot/TrackEval/tests/test_mots.py                                                          | tracker                | detector_config_candidate |
| tracking/boxmot/TrackEval/trackeval/_timing.py                                                        | tracker                | detector_config_candidate |

## Capability decision table

| capability                          | status    | evidence                                                       | recommended_next_step                                                                                                                     |
|:------------------------------------|:----------|:---------------------------------------------------------------|:------------------------------------------------------------------------------------------------------------------------------------------|
| ideal_segmentation_model            | not_ready | seg_pkg_available=False; seg_weight_available=True             | Use preliminary bbox-guided baseline unless user provides YOLO-seg/SAM/MaskRCNN package+weights.                                          |
| torchvision_crop_embedding          | possible  | torch=True; torchvision=True; embedding_weight_available=True  | Try torchvision model feature extraction. Prefer pretrained local weights if available; otherwise do not overclaim pretrained embeddings. |
| timm_or_transformers_embedding      | possible  | timm=True; transformers=False; embedding_weight_available=True | Search local pretrained model weights and test embedding extraction.                                                                      |
| detector_backbone_or_yolo_embedding | possible  | detector_weight_available=True                                 | Investigate detector backbone feature extraction from available checkpoint.                                                               |

## Interpretation

If ideal segmentation packages/weights are unavailable, the current bbox-guided GrabCut/Otsu segmentation remains a preliminary baseline. For learned crop embeddings, we should only claim pretrained/learned semantic embeddings if a trained model or local pretrained weights are available. Otherwise we can produce non-pretrained descriptors but should not overclaim them.
