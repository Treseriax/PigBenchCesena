# Week 7 Region of Interest and Crate Mapping Candidate

## Purpose

This step creates initial Region of Interest candidates for the annotated pen or crate. The candidates are not automatically treated as final because the annotated Region of Interest is a semantic ground-truth decision, not only a pixel-level inference.

## Region of Interest candidates

| roi_candidate_name          | roi_type   |   x1 |   y1 |   x2 |   y2 |   frame_width |   frame_height | source                             | interpretation                                                                                          | requires_human_confirmation   |   area_pixels |   area_fraction_of_frame |
|:----------------------------|:-----------|-----:|-----:|-----:|-----:|--------------:|---------------:|:-----------------------------------|:--------------------------------------------------------------------------------------------------------|:------------------------------|--------------:|-------------------------:|
| global_full_frame_roi       | rectangle  |    0 |    0 |  704 |  576 |           704 |            576 | full frame default candidate       | Use only if the camera view corresponds to the annotated pen or crate.                                  | True                          |        405504 |                 1        |
| global_detection_extent_roi | rectangle  |    0 |   20 |  704 |  576 |           704 |            576 | global detector extent with margin | Useful as an animal-activity Region of Interest candidate, but it may not cover the full annotated pen. | True                          |        391424 |                 0.965278 |

## Detection assignment summary

| roi_candidate_name          |   inside_count |   outside_count |   inside_fraction |
|:----------------------------|---------------:|----------------:|------------------:|
| global_full_frame_roi       |            540 |               0 |                 1 |
| global_detection_extent_roi |            540 |               0 |                 1 |

## Crate / camera / hour initial summary

| initial_crate_or_camera_token   |   scanpoint_frames |   videos | summary_type          |
|:--------------------------------|-------------------:|---------:|:----------------------|
| B1                              |                  6 |        1 | crate_or_camera_token |
| B1|LC1_B1                       |                 42 |        7 | crate_or_camera_token |
| c0001                           |                 24 |        4 | crate_or_camera_token |
| 07                              |                  6 |        1 | hour_token            |
| 08                              |                  6 |        1 | hour_token            |
| 09                              |                  6 |        1 | hour_token            |
| 10                              |                  6 |        1 | hour_token            |
| 11                              |                  6 |        1 | hour_token            |
| 12                              |                  6 |        1 | hour_token            |
| 13                              |                  6 |        1 | hour_token            |
| 14                              |                  6 |        1 | hour_token            |
| 15                              |                  6 |        1 | hour_token            |
| 16                              |                  6 |        1 | hour_token            |
| 17                              |                  6 |        1 | hour_token            |
| 18                              |                  6 |        1 | hour_token            |

## Contact sheet

- `/home/oyavuz/PigBench/Week7_Dataset_Validation_Identity_Tracking_Behaviour_Preparation/outputs/visualizations/week7_roi_candidate_overlay_contact_sheet.jpg`
- Contact sheet generated: `True`

## Required review

Please inspect the contact sheet and decide which Region of Interest candidate best corresponds to the annotated pen. If neither candidate is correct, manually adjust the coordinates in the manual review template.

## Outputs

- `/home/oyavuz/PigBench/Week7_Dataset_Validation_Identity_Tracking_Behaviour_Preparation/outputs/roi_and_crate_mapping/week7_roi_candidate_definitions.csv`
- `/home/oyavuz/PigBench/Week7_Dataset_Validation_Identity_Tracking_Behaviour_Preparation/outputs/roi_and_crate_mapping/week7_roi_candidate_definitions.json`
- `/home/oyavuz/PigBench/Week7_Dataset_Validation_Identity_Tracking_Behaviour_Preparation/outputs/roi_and_crate_mapping/week7_bbox_roi_assignment_candidate.csv`
- `/home/oyavuz/PigBench/Week7_Dataset_Validation_Identity_Tracking_Behaviour_Preparation/outputs/roi_and_crate_mapping/week7_roi_frame_summary_candidate.csv`
- `/home/oyavuz/PigBench/Week7_Dataset_Validation_Identity_Tracking_Behaviour_Preparation/outputs/roi_and_crate_mapping/week7_roi_manual_review_template.csv`
- `/home/oyavuz/PigBench/Week7_Dataset_Validation_Identity_Tracking_Behaviour_Preparation/outputs/dataset_statistics/week7_crate_pen_video_initial_summary.csv`
- `/home/oyavuz/PigBench/Week7_Dataset_Validation_Identity_Tracking_Behaviour_Preparation/outputs/visualizations/week7_roi_candidate_overlay_contact_sheet.jpg`
