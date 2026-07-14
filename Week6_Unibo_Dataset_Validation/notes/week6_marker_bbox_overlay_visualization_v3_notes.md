# Week 6 Marker-Bbox Overlay Visualization v3 Notes

## Purpose

This visualization overlays YOLOv8-s pig bounding boxes and crop-based colour-marker candidates on all 72 scanpoint frames. The side panel preserves the manual Excel colour IDs and behaviour labels for visual comparison.

## Outputs

- Overlay image directory: `/home/oyavuz/PigBench/Week6_Unibo_Dataset_Validation/outputs/visual_label_check/marker_bbox_overlay_v3`
- Slideshow video: `/home/oyavuz/PigBench/Week6_Unibo_Dataset_Validation/outputs/visual_label_check/marker_bbox_overlay_v3_video/week6_marker_bbox_overlay_v3_slideshow.mp4`
- Summary CSV: `/home/oyavuz/PigBench/Week6_Unibo_Dataset_Validation/outputs/visual_label_check/marker_bbox_overlay_v3_summary.csv`
- Status summary CSV: `/home/oyavuz/PigBench/Week6_Unibo_Dataset_Validation/outputs/visual_label_check/marker_bbox_overlay_v3_status_summary.csv`

## Status summary

| status   | video_match_status               |   frame_count |   total_bboxes |   total_manual_labels |   high_marker_candidates |   medium_marker_candidates |   low_marker_candidates |   no_marker |   green_candidates |   blue_candidates |   purple_candidates |   red_candidates |
|:---------|:---------------------------------|--------------:|---------------:|----------------------:|-------------------------:|---------------------------:|------------------------:|------------:|-------------------:|------------------:|--------------------:|-----------------:|
| written  | candidate_recovered_ctoken_video |            24 |            186 |                   144 |                       22 |                          6 |                      15 |         143 |                  4 |                 6 |                   1 |               17 |
| written  | matched_tlc_hour_video           |            48 |            354 |                   288 |                      122 |                         46 |                      54 |         132 |                 41 |                75 |                  27 |               25 |

## Interpretation

The visualization should be used for visual quality control. Green, blue, purple, and red marker detections are candidate bbox-to-colour links. Red remains ambiguous because the manual labels distinguish red_neck and red_tail, while HSV marker detection only detects red. No-color pigs cannot be recovered through colour-marker evidence. Therefore v3 is a strong visual QC artifact but not final identity-resolved ground truth.
