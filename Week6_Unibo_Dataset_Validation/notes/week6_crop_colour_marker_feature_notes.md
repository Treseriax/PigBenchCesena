# Week 6 Crop Colour Marker Feature Notes

## Purpose

This step scores each YOLOv8-s pig detection crop for visible colour-marker evidence. The goal is to explore candidate bbox-to-colour-ID association, not to create final identity ground truth.

## Inputs

- Detections: `/home/oyavuz/PigBench/Week6_Unibo_Dataset_Validation/outputs/feature_extractors/week6_yolov8s_all_scanpoint_detections.csv`
- Manual labels: `/home/oyavuz/PigBench/Week6_Unibo_Dataset_Validation/outputs/unified_ground_truth/week6_scanpoint_frame_labels_long.csv`

## Outputs

- Crop marker features: `/home/oyavuz/PigBench/Week6_Unibo_Dataset_Validation/outputs/feature_extractors/week6_crop_colour_marker_features.csv`
- Candidate bbox-to-colour assignments: `/home/oyavuz/PigBench/Week6_Unibo_Dataset_Validation/outputs/feature_extractors/week6_candidate_bbox_to_colour_assignments.csv`
- Marker summary: `/home/oyavuz/PigBench/Week6_Unibo_Dataset_Validation/outputs/feature_extractors/week6_crop_colour_marker_summary.csv`
- Frame summary: `/home/oyavuz/PigBench/Week6_Unibo_Dataset_Validation/outputs/feature_extractors/week6_crop_colour_marker_frame_summary.csv`
- Debug crops: `/home/oyavuz/PigBench/Week6_Unibo_Dataset_Validation/outputs/visual_label_check/colour_marker_crop_debug`

## Marker summary

| best_marker_colour   | marker_confidence   |   detection_count |   mean_best_score |   mean_margin |   mean_detection_score |
|:---------------------|:--------------------|------------------:|------------------:|--------------:|-----------------------:|
| blue                 | high                |                66 |        0.0700225  |   0.0547008   |               0.679712 |
| green                | high                |                32 |        0.0554607  |   0.0415972   |               0.686509 |
| red                  | high                |                30 |        0.0471601  |   0.0374596   |               0.710514 |
| purple               | high                |                16 |        0.051012   |   0.0300606   |               0.752434 |
| green                | low                 |                31 |        0.00776332 |   0.00438121  |               0.765229 |
| blue                 | low                 |                19 |        0.0103513  |   0.0037693   |               0.788478 |
| red                  | low                 |                11 |        0.0120817  |   0.00431425  |               0.806451 |
| purple               | low                 |                 8 |        0.00965338 |   0.00452631  |               0.85715  |
| blue                 | medium              |                15 |        0.0181503  |   0.00702063  |               0.75657  |
| green                | medium              |                13 |        0.0152959  |   0.0105775   |               0.724304 |
| purple               | medium              |                12 |        0.0207437  |   0.00720682  |               0.811969 |
| red                  | medium              |                12 |        0.014944   |   0.0112251   |               0.71636  |
| no_marker_detected   | none                |               275 |        0.00107892 |   0.000878595 |               0.756566 |

## Interpretation

Green, blue, purple, and red markers are detected as colour evidence inside each bbox crop. Red evidence is ambiguous because the Excel labels contain both red_neck and red_tail. The no_color pig cannot be identified by a colour marker. Therefore these outputs should be used as candidate assignments and visual QC aids rather than final ground truth.
