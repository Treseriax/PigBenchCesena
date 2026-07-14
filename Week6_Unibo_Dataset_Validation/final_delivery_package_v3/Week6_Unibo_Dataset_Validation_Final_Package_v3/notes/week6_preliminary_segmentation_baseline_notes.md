# Week 6 Preliminary Segmentation Baseline

## Purpose

This step addresses the segmentation requirement with a preliminary bbox-guided segmentation baseline. The method uses detector bbox crops and applies GrabCut with an Otsu fallback to estimate foreground masks.

## Important scope note

These masks are automatic preliminary segmentation outputs, not manual segmentation ground truth. They are intended for early shape/posture/contact/ROI feature exploration and visual QC.

## Segmentation resource audit

| resource         | available   | interpretation                                                                    |
|:-----------------|:------------|:----------------------------------------------------------------------------------|
| opencv_cv2       | True        | Required for bbox-guided GrabCut/Otsu segmentation baseline.                      |
| ultralytics      | False       | Would be useful for YOLO-seg if installed and segmentation weights are available. |
| segment_anything | False       | Would be useful for SAM baseline if installed and weights are available.          |
| sam2             | False       | Would be useful for SAM2 baseline if installed and weights are available.         |
| detectron2       | False       | Would be useful for Mask R-CNN style instance segmentation if installed.          |
| ffmpeg           | True        | Useful for video export but not required for static segmentation overlays.        |

## Outputs

- Segmentation feature CSV: `/home/oyavuz/PigBench/Week6_Unibo_Dataset_Validation/outputs/feature_extractors/week6_preliminary_bbox_guided_segmentation_features.csv`
- Segmentation feature JSON: `/home/oyavuz/PigBench/Week6_Unibo_Dataset_Validation/outputs/feature_extractors/week6_preliminary_bbox_guided_segmentation_features.json`
- Per-frame segmentation summary: `/home/oyavuz/PigBench/Week6_Unibo_Dataset_Validation/outputs/feature_extractors/week6_preliminary_segmentation_frame_summary.csv`
- Quality summary: `/home/oyavuz/PigBench/Week6_Unibo_Dataset_Validation/outputs/dataset_statistics/week6_preliminary_segmentation_quality_summary.csv`
- Status summary: `/home/oyavuz/PigBench/Week6_Unibo_Dataset_Validation/outputs/dataset_statistics/week6_preliminary_segmentation_status_summary.csv`
- Overlay directory: `/home/oyavuz/PigBench/Week6_Unibo_Dataset_Validation/outputs/visual_label_check/preliminary_segmentation_baseline`
- Contact sheet: `/home/oyavuz/PigBench/Week6_Unibo_Dataset_Validation/outputs/visual_label_check/week6_preliminary_segmentation_contact_sheet.jpg`

## Quality summary

| metric                           |   value | interpretation                                                                        |
|:---------------------------------|--------:|:--------------------------------------------------------------------------------------|
| total_segmentation_records       |     540 | One preliminary segmentation attempt per detector bbox.                               |
| successful_or_fallback_masks     |     540 | Masks with usable foreground contour from GrabCut or Otsu fallback.                   |
| success_rate                     |       1 | Approximate technical mask generation rate; not accuracy against manual segmentation. |
| overlay_images                   |      72 | One preliminary segmentation overlay per scanpoint frame.                             |
| manual_segmentation_gt_available |       0 | No manual segmentation GT is available in this pipeline.                              |

## Segmentation status summary

| segmentation_status         |   count |
|:----------------------------|--------:|
| grabcut_success             |     279 |
| otsu_fallback_after_grabcut |     261 |

## Interpretation

The segmentation baseline provides mask area, mask aspect ratio, contour perimeter, shape extent, solidity, orientation, and foreground colour statistics. These features may improve shape/posture/contact estimates compared with bbox-only features, but they require visual/manual validation before being treated as reliable labels.
