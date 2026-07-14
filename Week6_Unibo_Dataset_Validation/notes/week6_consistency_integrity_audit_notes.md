# Week 6 Consistency and Integrity Audit

## Purpose

This audit checks the integrity of the Week 6 pipeline before moving to additional feature extractors. It validates GT counts, scanpoint frames, label-frame alignment, video path existence, split leakage, detection alignment, and marker candidate consistency.

## Audit result summary

- Failed checks: `0`
- Warning checks: `2`
- Total checks: `18`

## Checks

| check_name                                             | status             | observed                               | expected                                                   | severity   | note                                                                                                                      |
|:-------------------------------------------------------|:-------------------|:---------------------------------------|:-----------------------------------------------------------|:-----------|:--------------------------------------------------------------------------------------------------------------------------|
| gt_total_records_is_432                                | PASS               | 432                                    | 432                                                        | critical   | Unified GT should preserve all Excel scan-sampling labels.                                                                |
| scanpoint_frames_is_72                                 | PASS               | 72                                     | 72                                                         | critical   | 12 hours x 6 scan points per hour.                                                                                        |
| frame_labels_is_432                                    | PASS               | 432                                    | 432                                                        | critical   | Each of 72 frames should have six manual labels.                                                                          |
| detections_exist                                       | PASS               | 540                                    | >0                                                         | critical   | YOLOv8-s detection output should not be empty.                                                                            |
| marker_features_match_detection_count                  | PASS               | 540                                    | 540                                                        | critical   | Each detection bbox should have one crop-marker feature row.                                                              |
| no_duplicate_record_id                                 | PASS               | 0                                      | 0                                                          | critical   | record_id should be unique.                                                                                               |
| no_duplicate_scan_frame_id                             | PASS               | 0                                      | 0                                                          | critical   | scan_frame_id should be unique.                                                                                           |
| six_manual_labels_per_scan_frame                       | PASS               | 0 bad frames; 72 labelled frames       | 0 bad frames; 72 labelled frames                           | critical   | Each scan frame should contain six colour-ID behaviour labels.                                                            |
| all_frame_images_exist                                 | PASS               | 0                                      | 0                                                          | critical   | All extracted scanpoint frame image paths should exist.                                                                   |
| all_gt_video_paths_exist                               | PASS               | 0                                      | 0                                                          | critical   | All GT video paths should exist after v2 recovery.                                                                        |
| all_gt_rows_have_split                                 | PASS               | 0                                      | 0                                                          | critical   | Every GT row should have train/val/test split.                                                                            |
| no_video_id_split_leakage                              | PASS               | 0                                      | 0                                                          | critical   | A video/hour unit must not appear in multiple splits.                                                                     |
| behaviour_coverage_per_split_warning                   | PASS_WITH_WARNINGS | max missing behaviours in a split = 3  | 0 ideal, but rare classes may be absent                    | warning    | Rare behaviours may not appear in every split because dataset is small.                                                   |
| detections_cover_all_scan_frames                       | PASS               | 0                                      | 0                                                          | critical   | Each extracted frame should have detector rows. If a frame has zero detections, it should still be documented separately. |
| no_extra_detection_scan_frame_ids                      | PASS               | 0                                      | 0                                                          | critical   | Detection scan_frame_id values should all exist in frame index.                                                           |
| bbox_count_reasonable_warning                          | PASS_WITH_WARNINGS | 10 frames <6 boxes; 4 frames >10 boxes | around 6 pigs, but occlusion/duplicate detections possible | warning    | Detector bboxes are automatic, so counts can differ from six manual pig labels.                                           |
| candidate_ctoken_rows_have_medium_confidence           | PASS               | 0                                      | 0                                                          | important  | Recovered c-token videos should remain marked as candidate/medium confidence.                                             |
| candidate_assignment_count_matches_medium_high_markers | PASS               | 196                                    | 196                                                        | important  | Candidate bbox-to-colour assignment table should contain medium/high marker rows.                                         |

## Split summary

| split   |   rows |   videos |   colours |   behaviours |
|:--------|-------:|---------:|----------:|-------------:|
| test    |     72 |        2 |         6 |            9 |
| train   |    288 |        8 |         6 |           11 |
| val     |     72 |        2 |         6 |            8 |

## Split behaviour coverage

| split   |   present_behaviour_count |   total_behaviour_count | missing_behaviours   |   missing_count |
|:--------|--------------------------:|------------------------:|:---------------------|----------------:|
| test    |                         9 |                      11 | NU,SI                |               2 |
| train   |                        11 |                      11 |                      |               0 |
| val     |                         8 |                      11 | BE,DE,IA             |               3 |

## Low bbox-count frames

| scan_frame_id   |   bbox_count |
|:----------------|-------------:|
| scanframe_0002  |            4 |
| scanframe_0008  |            5 |
| scanframe_0011  |            5 |
| scanframe_0020  |            5 |
| scanframe_0021  |            5 |
| scanframe_0023  |            5 |
| scanframe_0024  |            5 |
| scanframe_0041  |            5 |
| scanframe_0066  |            5 |
| scanframe_0068  |            5 |

## High bbox-count frames

| scan_frame_id   |   bbox_count |
|:----------------|-------------:|
| scanframe_0035  |           13 |
| scanframe_0038  |           14 |
| scanframe_0059  |           12 |
| scanframe_0060  |           12 |

## Interpretation

No critical integrity failures were found. Warning-level issues should still be reviewed, especially split behaviour coverage and frames with unusually low/high bbox counts. If only warnings remain, the pipeline can continue to feature extractor comparison.
