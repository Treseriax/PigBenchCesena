# Edinburgh vs Unibo Annotation Format Comparison

## Purpose

This note compares the Edinburgh manually annotated `output.json` format with the Week 3 corrected Unibo scan-window annotation JSON.

## Comparison table

| aspect                                | Edinburgh manually annotated dataset                                                                               | Unibo corrected scan-window dataset                                                                                                                                         |
|:--------------------------------------|:-------------------------------------------------------------------------------------------------------------------|:----------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| Data source                           | Public Edinburgh annotated.tar ground-truth package                                                                | Our Week 3 corrected scan-window outputs                                                                                                                                    |
| Main annotation file                  | output.json inside each annotated sequence folder                                                                  | corrected_scan_window_behaviour_annotations.json                                                                                                                            |
| Top-level JSON structure              | dict with keys: videoFileName, fullVideoFilePath, stepSize, config, objects                                        | dict with keys: dataset_name, source_video, method_note, annotation_policy, segments                                                                                        |
| Object / track representation         | objects list; each object corresponds to one tracked pig; inspected sample has 8 objects                           | frame-level track records with track IDs; detected unique track-id values: 35                                                                                               |
| Frame range / temporal representation | Ground-truth frameNumber range 0–599; source page notes GT corresponds to every third raw video frame              | Detected frame value range 1–500; corrected scan-window segments are aligned to Excel observation windows                                                                   |
| Bounding box format                   | bbox object with x, y, width, height                                                                               | bbox-like frame-level coordinates from YOLO/ByteTrack tracking output                                                                                                       |
| Behaviour labels                      | standing, walk, investigating, sleep, sitting, drink, playwithtoy, lying                                           | BOX, out_of_view_or_box, STI, lying_sternal, NU, eating, None, identity_not_verified, AN, exploring_or_sniffing, PI, standing_inactive, IN, interaction, LAI, lying_lateral |
| Behaviour granularity                 | Manual ground-truth frame descriptors; behaviour forward-propagated between changes according to source-page notes | Segment/window-level Excel ethogram labels attached to frame-level tracks                                                                                                   |
| Identity handling                     | Persistent object IDs in ground-truth object list                                                                  | ByteTrack IDs available; true pig identity remains conservative / not fully verified                                                                                        |
| Depth information                     | Depth video and background_depth are available                                                                     | RGB/video based; no depth information                                                                                                                                       |
| Strength for Week 4                   | Strong public reference for bbox + tracking + behaviour annotation format                                          | Primary internal dataset for trajectory, ROI, and representation prototype                                                                                                  |
| Main limitation                       | Different camera/pen setup; only selected sequences manually ground-truthed                                        | Labels are scan-window level, identity mapping remains conservative, no snout/skeleton/depth yet                                                                            |

## Unibo schema summary

| field                     | value                                                                                                               |
|:--------------------------|:--------------------------------------------------------------------------------------------------------------------|
| top_level_type            | dict                                                                                                                |
| top_level_keys            | dataset_name, source_video, method_note, annotation_policy, segments                                                |
| top_level_length_if_list  |                                                                                                                     |
| dict_count_recursive      | 188366                                                                                                              |
| list_count_recursive      | 95399                                                                                                               |
| bbox_like_count           | 0                                                                                                                   |
| unique_segments           | 514                                                                                                                 |
| unique_track_ids          | 35                                                                                                                  |
| min_frame                 | 1                                                                                                                   |
| max_frame                 | 500                                                                                                                 |
| frame_value_count         | 2450                                                                                                                |
| behaviour_fields_detected | behaviour_code, behaviour_label                                                                                     |
| identity_fields_detected  | identity_verified_rows, identity_unverified_rows, pig_id, colour, identity_status, assigned_colour, assigned_pig_id |

## Unibo most common keys

| key                                |   count |
|:-----------------------------------|--------:|
| behaviour_code                     |  162674 |
| behaviour_label                    |  162674 |
| source                             |  162674 |
| pig_id                             |  139440 |
| colour                             |  139440 |
| track_id                           |   23234 |
| bbox_xyxy                          |   23234 |
| bbox_xywh                          |   23234 |
| centroid                           |   23234 |
| score                              |   23234 |
| identity_status                    |   23234 |
| assigned_colour                    |   23234 |
| assigned_pig_id                    |   23234 |
| behaviour                          |   23234 |
| available_segment_behaviour_labels |   23234 |
| confidence                         |   23234 |
| excel_interval_start               |    2456 |
| excel_interval_end                 |    2456 |
| frame                              |    2450 |
| segment_time_sec                   |    2450 |
| video_time_sec                     |    2450 |
| tracks                             |    2450 |
| segment_id                         |       6 |
| source_video_segment_start_sec     |       6 |
| source_video_segment_end_sec       |       6 |
| available_excel_behaviour_labels   |       6 |
| frames                             |       6 |
| dataset_name                       |       1 |
| source_video                       |       1 |
| method_note                        |       1 |

## Unibo behaviour/value counts

| behaviour_or_behavior_value   |   count |
|:------------------------------|--------:|
| BOX                           |   48882 |
| out_of_view_or_box            |   48882 |
| STI                           |   31082 |
| lying_sternal                 |   31082 |
| NU                            |   25700 |
| eating                        |   25700 |
| None                          |   23234 |
| identity_not_verified         |   23234 |
| AN                            |   21269 |
| exploring_or_sniffing         |   21269 |
| PI                            |    5354 |
| standing_inactive             |    5354 |
| IN                            |    4482 |
| interaction                   |    4482 |
| LAI                           |    2671 |
| lying_lateral                 |    2671 |

## Interpretation

The Edinburgh dataset provides a clean public reference for manual ground-truth annotation with persistent object IDs, bounding boxes, and frame-level behaviour descriptors. The Unibo dataset is more directly connected to the internship task and contains corrected scan-window alignment, tracking outputs, and behaviour labels, but its labels are coarser because they are attached at scan-window level rather than fully manual frame-level annotation. Together, the two datasets are complementary for Week 4 behaviour representation analysis.
