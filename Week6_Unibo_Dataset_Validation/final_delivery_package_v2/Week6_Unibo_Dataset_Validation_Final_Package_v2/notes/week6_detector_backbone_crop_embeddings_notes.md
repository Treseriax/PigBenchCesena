# Week 6 Detector-Backbone Crop Embeddings

## Purpose

This step extracts learned crop embeddings for all detector bboxes using the locally available PigBench YOLOv8-s detector checkpoint. The method uses `model.extract_feat` and global average pooling over feature maps. This is not a random/untrained embedding route.

## Method

- Config: `/home/oyavuz/PigBench/detection/configs/yolov8/yolov8_s.py`
- Checkpoint: `/home/oyavuz/PigBench/detection/data/pretrained_weights/yolov8_pigs/yolov8_s.pth`
- Embedding method: `mmdet_yolov8s_pig_detector_backbone_global_avg_pool`
- Crop source: detector bbox crops from scanpoint frames
- Feature pooling: adaptive global average pooling per feature map, concatenated

## Outputs

- Wide embedding CSV: `/home/oyavuz/PigBench/Week6_Unibo_Dataset_Validation/outputs/feature_extractors/week6_detector_backbone_crop_embeddings_896.csv`
- Embedding NPY: `/home/oyavuz/PigBench/Week6_Unibo_Dataset_Validation/outputs/feature_extractors/week6_detector_backbone_crop_embeddings_896.npy`
- Metadata CSV: `/home/oyavuz/PigBench/Week6_Unibo_Dataset_Validation/outputs/feature_extractors/week6_detector_backbone_crop_embedding_metadata.csv`
- PCA feature CSV: `/home/oyavuz/PigBench/Week6_Unibo_Dataset_Validation/outputs/feature_extractors/week6_detector_backbone_crop_embedding_pca_features.csv`
- Summary CSV: `/home/oyavuz/PigBench/Week6_Unibo_Dataset_Validation/outputs/dataset_statistics/week6_detector_backbone_crop_embedding_summary.csv`
- Verification CSV: `/home/oyavuz/PigBench/Week6_Unibo_Dataset_Validation/outputs/dataset_statistics/week6_detector_backbone_crop_embedding_verification.csv`

## Summary

| metric                    | value                                               | interpretation                                                           |
|:--------------------------|:----------------------------------------------------|:-------------------------------------------------------------------------|
| input_detection_rows      | 540                                                 | All detector rows considered for crop embedding extraction.              |
| successful_embedding_rows | 540                                                 | Rows with usable crop and finite detector-backbone embedding.            |
| failed_crop_rows          | 0                                                   | Rows skipped due to crop loading/crop quality problems.                  |
| embedding_dim             | 896                                                 | Detector-backbone pooled feature dimension.                              |
| embedding_method          | mmdet_yolov8s_pig_detector_backbone_global_avg_pool | Local PigBench YOLOv8-s detector checkpoint, not random/untrained model. |
| device                    | cuda:0                                              | Inference device.                                                        |
| pca_components            | 16                                                  | Compact PCA features generated for downstream visualization/comparison.  |

## Verification

| check               | status   |   observed |   expected |
|:--------------------|:---------|-----------:|-----------:|
| embedding_count_540 | PASS     |        540 |        540 |
| embedding_dim_896   | PASS     |        896 |        896 |
| finite_embeddings   | PASS     |       True |       True |
| npy_exists          | PASS     |       True |       True |
| wide_csv_exists     | PASS     |       True |       True |

## Interpretation

The detector-backbone embeddings provide a learned crop representation derived from the pig detector checkpoint. They can be used as a feature extractor comparison input alongside bbox geometry, ROI proxies, crop descriptors, marker features, and segmentation features. They should be described as detector-backed crop embeddings, not as DINO/CLIP/SAM features.
