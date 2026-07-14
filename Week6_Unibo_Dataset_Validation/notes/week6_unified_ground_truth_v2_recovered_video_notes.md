# Week 6 Unified Ground Truth v2 with Recovered c-token Videos

## Purpose

This version updates the unified ground-truth table by linking the previously unmatched 15:00-19:00 labels to candidate c-token videos. The mapping is based on TLC-to-c-token visual similarity and should be treated as candidate evidence until visually confirmed.

## Summary

| metric                                   |   value |
|:-----------------------------------------|--------:|
| total_records                            |     432 |
| direct_tlc_video_matched_records         |     288 |
| candidate_recovered_ctoken_video_records |     144 |
| still_unmatched_video_records            |       0 |
| records_with_any_video_path              |     432 |
| bbox_available_records                   |       0 |

## Video match status distribution

| video_match_status               | video_mapping_confidence         |   count |
|:---------------------------------|:---------------------------------|--------:|
| candidate_recovered_ctoken_video | medium_needs_visual_confirmation |     144 |
| matched_tlc_hour_video           | high                             |     288 |

## Quality flag distribution

| quality_flag                                 |   count |
|:---------------------------------------------|--------:|
| label_with_video_match_no_bbox               |     288 |
| label_with_candidate_recovered_video_no_bbox |     144 |

## Interpretation

All 432 labels now have a video path. The 07:00-15:00 labels are direct TLC filename matches and have high confidence. The 15:00-19:00 labels are recovered candidate c-token matches and have medium confidence until the contact sheet is visually inspected. Bounding boxes are still not linked at this stage.
