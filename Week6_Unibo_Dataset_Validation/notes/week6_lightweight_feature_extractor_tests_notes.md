# Week 6 Lightweight Feature Extractor Tests

## Purpose

This step expands Task 5 beyond detector boxes and marker features. It creates lightweight, auditable feature extractor outputs for bbox geometry, group-spatial context, coarse ROI/resource proxies, crop descriptors, and trajectory feasibility.

## Outputs

| feature_test                | status                 | output                                                                                                                          |   record_count | interpretation                                                                          |
|:----------------------------|:-----------------------|:--------------------------------------------------------------------------------------------------------------------------------|---------------:|:----------------------------------------------------------------------------------------|
| bbox_geometry               | implemented            | /home/oyavuz/PigBench/Week6_Unibo_Dataset_Validation/outputs/feature_extractors/week6_bbox_geometry_features.csv                |            540 | Per-detection geometric features for bbox size, location, and normalized area.          |
| group_spatial               | implemented            | /home/oyavuz/PigBench/Week6_Unibo_Dataset_Validation/outputs/feature_extractors/week6_group_spatial_features_per_frame.csv      |             72 | Per-frame group distribution, pairwise distance, overlap, and spatial spread features.  |
| coarse_roi_resource_proxy   | implemented            | /home/oyavuz/PigBench/Week6_Unibo_Dataset_Validation/outputs/feature_extractors/week6_coarse_roi_resource_proxy_features.csv    |            540 | Coarse image-position ROI proxy; not a manually mapped physical resource annotation.    |
| crop_descriptor_baseline    | implemented            | /home/oyavuz/PigBench/Week6_Unibo_Dataset_Validation/outputs/feature_extractors/week6_crop_descriptor_baseline_features.csv     |            540 | Simple crop colour/brightness/edge descriptors for downstream comparison.               |
| crop_descriptor_plus_marker | implemented            | /home/oyavuz/PigBench/Week6_Unibo_Dataset_Validation/outputs/feature_extractors/week6_crop_descriptor_plus_marker_features.csv  |            540 | Crop descriptors plus marker evidence when available.                                   |
| trajectory                  | feasibility_documented | /home/oyavuz/PigBench/Week6_Unibo_Dataset_Validation/outputs/feature_extractors/week6_trajectory_feature_feasibility_report.csv |              2 | Trajectory features are not final because stable identity association is not validated. |

## Interpretation

- BBox geometry features are useful for object-level size and location analysis.
- Group-spatial features summarize group distribution, overlap, and interaction/contact proxies at scanpoint-frame level.
- Coarse ROI/resource features are image-position proxies only; they are not manually calibrated feeder/drinker/enrichment zones.
- Crop descriptor baselines provide simple colour/brightness/edge features and can be compared with marker features.
- Final trajectory features are intentionally not claimed because stable identity-resolved tracking is not validated yet.

## Key counts

- BBox geometry rows: `540`
- Group-spatial frame rows: `72`
- ROI proxy rows: `540`
- Crop descriptor rows: `540`
- Crop descriptor rows with available crop: `540`
- Trajectory feasibility rows: `2`
