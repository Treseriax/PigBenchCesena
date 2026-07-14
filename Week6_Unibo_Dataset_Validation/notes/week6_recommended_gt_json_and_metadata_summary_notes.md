# Week 6 Recommended GT JSON and Metadata Summary

## Purpose

This step closes the Task 2 JSON gap and strengthens the Task 1 dataset metadata summary. It creates a recommended GT JSON, schema file, nested viewer annotation JSON, and an explicit camera/pen/crate/video summary table.

## Outputs

- Recommended GT JSON: `/home/oyavuz/PigBench/Week6_Unibo_Dataset_Validation/outputs/unified_ground_truth/week6_unified_ground_truth_v2_with_recommended_split_v2.json`
- GT schema JSON: `/home/oyavuz/PigBench/Week6_Unibo_Dataset_Validation/outputs/unified_ground_truth/week6_unified_ground_truth_v2_with_recommended_split_v2_schema.json`
- GT schema fields CSV: `/home/oyavuz/PigBench/Week6_Unibo_Dataset_Validation/outputs/dataset_statistics/week6_recommended_gt_schema_fields.csv`
- Nested viewer annotation JSON: `/home/oyavuz/PigBench/Week6_Unibo_Dataset_Validation/outputs/unified_ground_truth/week6_scanpoint_annotations_nested_for_viewer.json`
- Camera/pen/crate/video summary: `/home/oyavuz/PigBench/Week6_Unibo_Dataset_Validation/outputs/dataset_statistics/week6_camera_pen_crate_video_summary.csv`
- Verification: `/home/oyavuz/PigBench/Week6_Unibo_Dataset_Validation/outputs/dataset_statistics/week6_recommended_gt_json_generation_verification.csv`

## Verification

| check_name                       | status   |   observed |   expected |
|:---------------------------------|:---------|-----------:|-----------:|
| recommended_gt_json_exists       | PASS     |       True |       True |
| recommended_gt_json_record_count | PASS     |        432 |        432 |
| schema_json_exists               | PASS     |       True |       True |
| nested_viewer_json_frame_count   | PASS     |         72 |         72 |
| camera_pen_crate_summary_exists  | PASS     |       True |       True |

## Camera/pen/crate/video summary preview

| date       | camera_id   | crate_id   | pen_id   | video_id            | video_match_status               | video_mapping_confidence         | hour_start   | hour_end   | recommended_split_v2   |   label_count |   unique_colour_ids |   unique_behaviour_codes |   scanpoint_frame_count |   total_bboxes |   mean_bboxes_per_frame |
|:-----------|:------------|:-----------|:---------|:--------------------|:---------------------------------|:---------------------------------|:-------------|:-----------|:-----------------------|--------------:|--------------------:|-------------------------:|------------------------:|---------------:|------------------------:|
| 2021-07-22 | TLC1        | B1         | B1       | TLC 1 -B1 0700-0800 | matched_tlc_hour_video           | high                             | 07:00        | 08:00      | train                  |            36 |                   6 |                        7 |                       6 |             41 |                 6.83333 |
| 2021-07-22 | TLC1        | B1         | B1       | TLC1 B1 800-900     | matched_tlc_hour_video           | high                             | 08:00        | 09:00      | test                   |            36 |                   6 |                        7 |                       6 |             51 |                 8.5     |
| 2021-07-22 | TLC1        | B1         | B1       | TLC1 B1 900-1000    | matched_tlc_hour_video           | high                             | 09:00        | 10:00      | train                  |            36 |                   6 |                        7 |                       6 |             50 |                 8.33333 |
| 2021-07-22 | TLC1        | B1         | B1       | TLC1 B1 1000-1100   | matched_tlc_hour_video           | high                             | 10:00        | 11:00      | train                  |            36 |                   6 |                        8 |                       6 |             39 |                 6.5     |
| 2021-07-22 | TLC1        | B1         | B1       | TLC1 B1 1100-1200   | matched_tlc_hour_video           | high                             | 11:00        | 12:00      | train                  |            36 |                   6 |                        4 |                       6 |             41 |                 6.83333 |
| 2021-07-22 | TLC1        | B1         | B1       | TLC1 B1 1200-1300   | matched_tlc_hour_video           | high                             | 12:00        | 13:00      | val                    |            36 |                   6 |                        9 |                       6 |             37 |                 6.16667 |
| 2021-07-22 | TLC1        | B1         | B1       | TLC1 B1 1300-1400   | matched_tlc_hour_video           | high                             | 13:00        | 14:00      | train                  |            36 |                   6 |                        6 |                       6 |             40 |                 6.66667 |
| 2021-07-22 | TLC1        | B1         | B1       | TLC1 B1 1400-1500   | matched_tlc_hour_video           | high                             | 14:00        | 15:00      | train                  |            36 |                   6 |                        8 |                       6 |             55 |                 9.16667 |
| 2021-07-22 | TLC1        | B1         | B1       | c0001210722150000   | candidate_recovered_ctoken_video | medium_needs_visual_confirmation | 15:00        | 16:00      | val                    |            36 |                   6 |                        5 |                       6 |             39 |                 6.5     |
| 2021-07-22 | TLC1        | B1         | B1       | c0001210722160000   | candidate_recovered_ctoken_video | medium_needs_visual_confirmation | 16:00        | 17:00      | train                  |            36 |                   6 |                        5 |                       6 |             53 |                 8.83333 |
| 2021-07-22 | TLC1        | B1         | B1       | c0001210722170000   | candidate_recovered_ctoken_video | medium_needs_visual_confirmation | 17:00        | 18:00      | test                   |            36 |                   6 |                        7 |                       6 |             52 |                 8.66667 |
| 2021-07-22 | TLC1        | B1         | B1       | c0001210722180000   | candidate_recovered_ctoken_video | medium_needs_visual_confirmation | 18:00        | 19:00      | train                  |            36 |                   6 |                        5 |                       6 |             42 |                 7       |

## Interpretation

The recommended GT is now available in both CSV and JSON format. The nested viewer JSON is prepared for visualization interfaces because it groups each scanpoint frame with its six manual labels and detector bboxes. Detector bboxes remain separate automatic outputs and should not be confused with manual bbox ground truth.
