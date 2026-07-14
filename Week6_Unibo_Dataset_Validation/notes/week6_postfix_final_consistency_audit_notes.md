# Week 6 Post-fix Final Consistency Audit

## Purpose

This audit verifies the pipeline after adopting recommended split v2 and adding detection QC flags/conservative subsets. It is intended as the final checkpoint before moving to feature extractor comparison.

## Result summary

- Failed checks: `0`
- Total checks: `16`

## Checks

| check_name                                                  | status   | observed                                                                              | expected                                     | severity   | note                                                                                                |
|:------------------------------------------------------------|:---------|:--------------------------------------------------------------------------------------|:---------------------------------------------|:-----------|:----------------------------------------------------------------------------------------------------|
| recommended_gt_exists_and_has_432_records                   | PASS     | 432                                                                                   | 432                                          | critical   | Recommended split v2 GT must preserve all manual Excel labels.                                      |
| recommended_split_v2_column_exists                          | PASS     | True                                                                                  | True                                         | critical   | Recommended split v2 file should contain recommended_split_v2 column.                               |
| recommended_split_v2_counts_are_288_72_72                   | PASS     | {'train': 288, 'test': 72, 'val': 72}                                                 | {'train': 288, 'val': 72, 'test': 72}        | critical   | Recommended split should be 8/2/2 video-hour units.                                                 |
| recommended_split_v2_has_no_video_hour_leakage              | PASS     | 0                                                                                     | 0                                            | critical   | No hourly video unit should appear in multiple splits.                                              |
| recommended_split_v2_behaviour_coverage_expected            | PASS     | train=11, val=10, test=10                                                             | train=11, val=10, test=10                    | important  | Optimized split should improve val/test coverage while keeping full train coverage.                 |
| scanpoint_frames_still_72                                   | PASS     | 72                                                                                    | 72                                           | critical   | Scanpoint frame index should remain unchanged.                                                      |
| frame_labels_still_432                                      | PASS     | 432                                                                                   | 432                                          | critical   | Frame-label long table should keep six labels per scanpoint frame.                                  |
| six_labels_per_scan_frame_still_true                        | PASS     | 0                                                                                     | 0                                            | critical   | Every scan frame should still have six manual labels.                                               |
| primary_detection_count_still_540                           | PASS     | 540                                                                                   | 540                                          | important  | Primary detection table should remain unchanged.                                                    |
| qc_detection_count_matches_primary                          | PASS     | 540                                                                                   | 540                                          | critical   | QC flag table should not add/drop detections.                                                       |
| qc_detection_columns_exist                                  | PASS     |                                                                                       | no missing QC columns                        | critical   | QC flag columns are needed for later feature extraction decisions.                                  |
| qc_warning_frame_counts_expected                            | PASS     | {'high_bbox_count_warning': 4, 'low_bbox_count_warning': 10, 'normal_bbox_count': 58} | 10 low warning frames, 4 high warning frames | important  | QC flag table should preserve the warning frame audit result.                                       |
| conservative_050_covers_all_72_frames                       | PASS     | 72                                                                                    | 72                                           | important  | Score >=0.50 subset should still cover every scanpoint frame.                                       |
| conservative_070_covers_all_72_frames                       | PASS     | 72                                                                                    | 72                                           | warning    | Score >=0.70 subset coverage is useful but strict; missing frames would be a warning, not critical. |
| marker_features_match_primary_detections                    | PASS     | 540                                                                                   | 540                                          | critical   | Each primary detection should still have one marker feature row.                                    |
| candidate_bbox_colour_assignments_match_medium_high_markers | PASS     | 196                                                                                   | 196                                          | important  | Candidate colour assignments should equal high/medium marker detections.                            |

## Final split summary

| split   |   labels |   video_units |   behaviours |   colours |   direct_tlc_records |   candidate_ctoken_records |
|:--------|---------:|--------------:|-------------:|----------:|---------------------:|---------------------------:|
| test    |       72 |             2 |           10 |         6 |                   36 |                         36 |
| train   |      288 |             8 |           11 |         6 |                  216 |                         72 |
| val     |       72 |             2 |           10 |         6 |                   36 |                         36 |

## Final detection summary

| table                      |   detections |   frames |   mean_bboxes_per_frame |
|:---------------------------|-------------:|---------:|------------------------:|
| primary_score_ge_0_25      |          540 |       72 |                   7.5   |
| qc_flagged_primary         |          540 |       72 |                   7.5   |
| conservative_score_ge_0_50 |          447 |       72 |                   6.208 |
| conservative_score_ge_0_70 |          376 |       72 |                   5.222 |

## Interpretation

All post-fix consistency checks passed. The Week 6 pipeline is ready for the next stage: feature extractor comparison and final report packaging. The primary detector table remains unchanged, while QC flags and conservative subsets are available for safer downstream analysis.
