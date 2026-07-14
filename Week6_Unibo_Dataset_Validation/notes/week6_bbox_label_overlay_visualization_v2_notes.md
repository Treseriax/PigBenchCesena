# Week 6 Bbox + Label Overlay Visualization v2 Notes

## Purpose

This visualization combines YOLOv8-s pig detections with manual Excel scan-sampling behaviour labels. It provides a visual quality-control bridge between video frames, detector bboxes, pig colour IDs, and behaviour labels.

## Outputs

- Overlay image directory: `/home/oyavuz/PigBench/Week6_Unibo_Dataset_Validation/outputs/visual_label_check/bbox_label_overlay_v2`
- Slideshow video: `/home/oyavuz/PigBench/Week6_Unibo_Dataset_Validation/outputs/visual_label_check/bbox_label_overlay_v2_video/week6_bbox_label_overlay_v2_slideshow.mp4`
- Overlay summary CSV: `/home/oyavuz/PigBench/Week6_Unibo_Dataset_Validation/outputs/visual_label_check/bbox_label_overlay_v2_summary.csv`
- Status summary CSV: `/home/oyavuz/PigBench/Week6_Unibo_Dataset_Validation/outputs/visual_label_check/bbox_label_overlay_v2_status_summary.csv`

## Status summary

| status   | video_match_status               |   frame_count |   total_bboxes |   mean_bboxes_per_frame |   min_bboxes |   max_bboxes |   total_manual_labels |
|:---------|:---------------------------------|--------------:|---------------:|------------------------:|-------------:|-------------:|----------------------:|
| written  | candidate_recovered_ctoken_video |            24 |            186 |                   7.75  |            5 |           12 |                   144 |
| written  | matched_tlc_hour_video           |            48 |            354 |                   7.375 |            4 |           14 |                   288 |

## Interpretation

The detector bboxes are available for all scanpoint frames. The manual labels are shown as a side panel because the exact bbox-to-colour-ID association has not yet been solved. The next step is crop-based colour-marker analysis, where each detected pig crop is scored for green/blue/purple/red marker evidence and then linked to manual colour IDs when possible.
