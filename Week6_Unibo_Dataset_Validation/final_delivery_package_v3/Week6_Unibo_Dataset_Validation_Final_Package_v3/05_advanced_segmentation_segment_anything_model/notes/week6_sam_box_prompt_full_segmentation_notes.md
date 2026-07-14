# Week 6 Segment Anything Model Full Box-Prompt Segmentation

## Purpose

This step applies Segment Anything Model to all pig detector bounding boxes. Each detector bounding box is used as a box prompt, producing one automatic segmentation mask per detection.

## Recovery note

The original full run completed Segment Anything Model mask generation for all detections, but the final post-processing script stopped during baseline comparison because of a pandas column-name collision. This recovery step did not rerun Segment Anything Model inference. It finalized the already generated masks, overlays, feature tables, comparison table, status table, and documentation.

## Quality summary

| metric                            | value              |
|:----------------------------------|:-------------------|
| total_input_detections_expected   | 540                |
| successful_sam_masks              | 540                |
| failed_sam_masks                  | 0                  |
| success_rate                      | 1.0                |
| frame_overlay_files               | 72                 |
| mask_png_files                    | 540                |
| mean_sam_predicted_iou_score      | 0.940495749645763  |
| median_sam_predicted_iou_score    | 0.9525234997272491 |
| mean_mask_area_fraction_of_bbox   | 0.5095501059598596 |
| median_mask_area_fraction_of_bbox | 0.5021001939285491 |
| device                            | cpu                |
| sam_model_type                    | vit_b              |

## Status checks

| item                        | status   | evidence                                                                                                                              |
|:----------------------------|:---------|:--------------------------------------------------------------------------------------------------------------------------------------|
| sam_feature_table_exists    | PASS     | /home/oyavuz/PigBench/Week6_Unibo_Dataset_Validation/outputs/feature_extractors/week6_sam_box_prompt_segmentation_features.csv        |
| sam_feature_rows            | PASS     | 540                                                                                                                                   |
| sam_mask_png_files          | PASS     | 540                                                                                                                                   |
| sam_frame_summary_rows      | PASS     | 72                                                                                                                                    |
| sam_frame_overlay_files     | PASS     | 72                                                                                                                                    |
| sam_failed_rows             | PASS     | 0                                                                                                                                     |
| sam_vs_baseline_comparison  | PASS     | /home/oyavuz/PigBench/Week6_Unibo_Dataset_Validation/outputs/dataset_statistics/week6_sam_vs_grabcut_otsu_segmentation_comparison.csv |
| sam_contact_sheet           | PASS     | /home/oyavuz/PigBench/Week6_Unibo_Dataset_Validation/outputs/visual_label_check/week6_sam_box_prompt_segmentation_contact_sheet.jpg   |
| previous_exception_recorded | INFO     | /home/oyavuz/PigBench/Week6_Unibo_Dataset_Validation/outputs/dataset_statistics/week6_sam_box_prompt_segmentation_exception.txt       |

## Comparison with classical GrabCut/Otsu baseline

| comparison                        | status   |   evidence |
|:----------------------------------|:---------|-----------:|
| matched_rows                      | PASS     |  540       |
| sam_rows                          | INFO     |  540       |
| baseline_rows                     | INFO     |  540       |
| mean_sam_mask_area_pixels         | INFO     | 6366.58    |
| mean_baseline_mask_area_pixels    | INFO     | 5098.37    |
| median_sam_to_baseline_area_ratio | INFO     |    1.281   |
| mean_sam_to_baseline_area_ratio   | INFO     |    1.60838 |

## Outputs

- Segment Anything Model feature table: `/home/oyavuz/PigBench/Week6_Unibo_Dataset_Validation/outputs/feature_extractors/week6_sam_box_prompt_segmentation_features.csv`
- Segment Anything Model JSON features: `/home/oyavuz/PigBench/Week6_Unibo_Dataset_Validation/outputs/feature_extractors/week6_sam_box_prompt_segmentation_features.json`
- Frame summary: `/home/oyavuz/PigBench/Week6_Unibo_Dataset_Validation/outputs/feature_extractors/week6_sam_box_prompt_segmentation_frame_summary.csv`
- Quality summary: `/home/oyavuz/PigBench/Week6_Unibo_Dataset_Validation/outputs/dataset_statistics/week6_sam_box_prompt_segmentation_quality_summary.csv`
- Status summary: `/home/oyavuz/PigBench/Week6_Unibo_Dataset_Validation/outputs/dataset_statistics/week6_sam_box_prompt_segmentation_status_summary.csv`
- Baseline comparison: `/home/oyavuz/PigBench/Week6_Unibo_Dataset_Validation/outputs/dataset_statistics/week6_sam_vs_grabcut_otsu_segmentation_comparison.csv`
- Detailed baseline comparison: `/home/oyavuz/PigBench/Week6_Unibo_Dataset_Validation/outputs/dataset_statistics/week6_sam_vs_grabcut_otsu_segmentation_comparison_detailed.csv`
- Mask directory: `/home/oyavuz/PigBench/Week6_Unibo_Dataset_Validation/outputs/visual_label_check/sam_box_prompt_full_segmentation/masks`
- Overlay directory: `/home/oyavuz/PigBench/Week6_Unibo_Dataset_Validation/outputs/visual_label_check/sam_box_prompt_full_segmentation/frame_overlays`
- Contact sheet: `/home/oyavuz/PigBench/Week6_Unibo_Dataset_Validation/outputs/visual_label_check/week6_sam_box_prompt_segmentation_contact_sheet.jpg`

## Interpretation

The full Segment Anything Model box-prompt segmentation is complete. It produced one automatic Segment Anything Model mask per detector bounding box, for 540 masks across 72 scanpoint frames. This is a stronger foundation-model-based segmentation route in addition to the earlier classical GrabCut/Otsu baseline. The masks remain automatic segmentation outputs, not manual segmentation ground truth.
