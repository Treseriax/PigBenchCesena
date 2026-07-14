# Week 7 Refined Ground Truth Pen Region of Interest v2 Candidates

## Reason for refinement

The previous broad semantic Region of Interest included too many pigs from the lower adjacent pen. This is not acceptable for ground-truth colour or behaviour matching because the Week 7 objective is to focus on the annotated ground truth pen only.

## Candidate ranking

| roi_name                             |   inside_count |   outside_count |   inside_fraction |   mean_inside_per_frame |   median_inside_per_frame |   min_inside_per_frame |   max_inside_per_frame |   frames_under_4_inside |   frames_over_8_inside |   target_total_inside |   target_mean_per_frame |   heuristic_score_lower_is_better | needs_visual_review   |
|:-------------------------------------|---------------:|----------------:|------------------:|------------------------:|--------------------------:|-----------------------:|-----------------------:|------------------------:|-----------------------:|----------------------:|------------------------:|----------------------------------:|:----------------------|
| wide_top_right_pen_roi_v2            |            391 |             149 |          0.724074 |                 5.43056 |                         5 |                      3 |                     10 |                       2 |                      3 |                   432 |                       6 |                           77.3889 | True                  |
| balanced_upper_pen_roi_v2            |            377 |             163 |          0.698148 |                 5.23611 |                         5 |                      2 |                      9 |                       3 |                      3 |                   432 |                       6 |                          100.278  | True                  |
| fence_line_upper_pen_roi_v2          |            359 |             181 |          0.664815 |                 4.98611 |                         5 |                      2 |                      9 |                       8 |                      3 |                   432 |                       6 |                          148.278  | True                  |
| conservative_main_pen_roi_v2         |            330 |             210 |          0.611111 |                 4.58333 |                         4 |                      2 |                      9 |                      15 |                      2 |                   432 |                       6 |                          215.333  | True                  |
| strict_upper_gt_pen_roi_v1_reference |            299 |             241 |          0.553704 |                 4.15278 |                         4 |                      2 |                      9 |                      24 |                      1 |                   432 |                       6 |                          294.944  | True                  |

## Candidate definitions

| roi_name                             | roi_type   | points_json                                                                    |   frame_width |   frame_height |   area_pixels |   area_fraction_of_frame | source                                                                         | interpretation                                                             |
|:-------------------------------------|:-----------|:-------------------------------------------------------------------------------|--------------:|---------------:|--------------:|-------------------------:|:-------------------------------------------------------------------------------|:---------------------------------------------------------------------------|
| strict_upper_gt_pen_roi_v1_reference | polygon    | [[140, 0], [704, 0], [704, 355], [170, 355], [170, 0]]                         |           704 |            576 |        189570 |                 0.467492 | refined human-guided candidate after broad v1 included too many lower-pen pigs | Candidate for annotated upper/right ground truth pen. Needs visual review. |
| balanced_upper_pen_roi_v2            | polygon    | [[125, 0], [704, 0], [704, 430], [185, 430], [165, 365], [125, 325], [125, 0]] |           704 |            576 |        244920 |                 0.603989 | refined human-guided candidate after broad v1 included too many lower-pen pigs | Candidate for annotated upper/right ground truth pen. Needs visual review. |
| fence_line_upper_pen_roi_v2          | polygon    | [[120, 0], [704, 0], [704, 400], [165, 400], [150, 340], [120, 310], [120, 0]] |           704 |            576 |        230900 |                 0.569415 | refined human-guided candidate after broad v1 included too many lower-pen pigs | Candidate for annotated upper/right ground truth pen. Needs visual review. |
| conservative_main_pen_roi_v2         | polygon    | [[135, 0], [704, 0], [704, 385], [190, 385], [170, 335], [135, 315], [135, 0]] |           704 |            576 |        216465 |                 0.533817 | refined human-guided candidate after broad v1 included too many lower-pen pigs | Candidate for annotated upper/right ground truth pen. Needs visual review. |
| wide_top_right_pen_roi_v2            | polygon    | [[110, 0], [704, 0], [704, 455], [220, 455], [170, 375], [120, 325], [110, 0]] |           704 |            576 |        260095 |                 0.641412 | refined human-guided candidate after broad v1 included too many lower-pen pigs | Candidate for annotated upper/right ground truth pen. Needs visual review. |

## Visual review files

| roi_name                             | contact_sheet_path                                                                                                                                                                     | generated   |
|:-------------------------------------|:---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|:------------|
| strict_upper_gt_pen_roi_v1_reference | /home/oyavuz/PigBench/Week7_Dataset_Validation_Identity_Tracking_Behaviour_Preparation/outputs/visualizations/roi_v2_candidates/strict_upper_gt_pen_roi_v1_reference_contact_sheet.jpg | True        |
| balanced_upper_pen_roi_v2            | /home/oyavuz/PigBench/Week7_Dataset_Validation_Identity_Tracking_Behaviour_Preparation/outputs/visualizations/roi_v2_candidates/balanced_upper_pen_roi_v2_contact_sheet.jpg            | True        |
| fence_line_upper_pen_roi_v2          | /home/oyavuz/PigBench/Week7_Dataset_Validation_Identity_Tracking_Behaviour_Preparation/outputs/visualizations/roi_v2_candidates/fence_line_upper_pen_roi_v2_contact_sheet.jpg          | True        |
| conservative_main_pen_roi_v2         | /home/oyavuz/PigBench/Week7_Dataset_Validation_Identity_Tracking_Behaviour_Preparation/outputs/visualizations/roi_v2_candidates/conservative_main_pen_roi_v2_contact_sheet.jpg         | True        |
| wide_top_right_pen_roi_v2            | /home/oyavuz/PigBench/Week7_Dataset_Validation_Identity_Tracking_Behaviour_Preparation/outputs/visualizations/roi_v2_candidates/wide_top_right_pen_roi_v2_contact_sheet.jpg            | True        |

- Combined review sheet: `/home/oyavuz/PigBench/Week7_Dataset_Validation_Identity_Tracking_Behaviour_Preparation/outputs/visualizations/week7_refined_gt_pen_roi_v2_combined_review_sheet.jpg`
- Combined review sheet generated: `True`

## Required decision

Select the candidate that best includes the annotated upper/right pen while excluding the lower adjacent pen. The heuristic ranking is useful, but visual review has priority.

## Outputs

- `/home/oyavuz/PigBench/Week7_Dataset_Validation_Identity_Tracking_Behaviour_Preparation/outputs/roi_and_crate_mapping/week7_refined_gt_pen_roi_v2_candidate_definitions.csv`
- `/home/oyavuz/PigBench/Week7_Dataset_Validation_Identity_Tracking_Behaviour_Preparation/outputs/roi_and_crate_mapping/week7_refined_gt_pen_roi_v2_candidate_definitions.json`
- `/home/oyavuz/PigBench/Week7_Dataset_Validation_Identity_Tracking_Behaviour_Preparation/outputs/roi_and_crate_mapping/week7_bbox_roi_assignment_refined_candidates_v2.csv`
- `/home/oyavuz/PigBench/Week7_Dataset_Validation_Identity_Tracking_Behaviour_Preparation/outputs/roi_and_crate_mapping/week7_roi_frame_summary_refined_candidates_v2.csv`
- `/home/oyavuz/PigBench/Week7_Dataset_Validation_Identity_Tracking_Behaviour_Preparation/outputs/roi_and_crate_mapping/week7_refined_gt_pen_roi_v2_candidate_ranking.csv`
- `/home/oyavuz/PigBench/Week7_Dataset_Validation_Identity_Tracking_Behaviour_Preparation/outputs/roi_and_crate_mapping/week7_refined_gt_pen_roi_v2_manual_review_template.csv`
- `/home/oyavuz/PigBench/Week7_Dataset_Validation_Identity_Tracking_Behaviour_Preparation/outputs/visualizations/week7_refined_gt_pen_roi_v2_contact_sheet_index.csv`
- `/home/oyavuz/PigBench/Week7_Dataset_Validation_Identity_Tracking_Behaviour_Preparation/outputs/visualizations/week7_refined_gt_pen_roi_v2_combined_review_sheet.jpg`
