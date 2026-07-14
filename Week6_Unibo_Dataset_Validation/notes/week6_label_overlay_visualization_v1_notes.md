# Week 6 Label Overlay Visualization v1 Notes

## Purpose

This visualization checks whether the unified ground-truth labels are correctly linked to video frames. Because bounding boxes are not yet linked, this v1 visualization displays the frame and a side panel with pig colour IDs and manual behaviour labels.

## Outputs

- Frame directory: `/home/oyavuz/PigBench/Week6_Unibo_Dataset_Validation/outputs/visual_label_check/label_overlay_frames_v1`
- Slideshow video: `/home/oyavuz/PigBench/Week6_Unibo_Dataset_Validation/outputs/visual_label_check/label_overlay_videos_v1/week6_label_overlay_v1_scanpoint_slideshow.mp4`
- Summary CSV: `/home/oyavuz/PigBench/Week6_Unibo_Dataset_Validation/outputs/visual_label_check/label_overlay_v1_frame_summary.csv`

## Summary

| video_id            | timestamp           |   frame_index | status   | output_frame                                                                                                                                           |   num_labels | video_match_status               |
|:--------------------|:--------------------|--------------:|:---------|:-------------------------------------------------------------------------------------------------------------------------------------------------------|-------------:|:---------------------------------|
| TLC 1 -B1 0700-0800 | 2021-07-22T07:00:00 |             0 | written  | /home/oyavuz/PigBench/Week6_Unibo_Dataset_Validation/outputs/visual_label_check/label_overlay_frames_v1/20210722_070000_TLC_1_-B1_0700-0800_labels.jpg |            6 | matched_tlc_hour_video           |
| TLC1 B1 1000-1100   | 2021-07-22T10:00:00 |             0 | written  | /home/oyavuz/PigBench/Week6_Unibo_Dataset_Validation/outputs/visual_label_check/label_overlay_frames_v1/20210722_100000_TLC1_B1_1000-1100_labels.jpg   |            6 | matched_tlc_hour_video           |
| TLC1 B1 1100-1200   | 2021-07-22T11:00:00 |             0 | written  | /home/oyavuz/PigBench/Week6_Unibo_Dataset_Validation/outputs/visual_label_check/label_overlay_frames_v1/20210722_110000_TLC1_B1_1100-1200_labels.jpg   |            6 | matched_tlc_hour_video           |
| TLC1 B1 1200-1300   | 2021-07-22T12:00:00 |             0 | written  | /home/oyavuz/PigBench/Week6_Unibo_Dataset_Validation/outputs/visual_label_check/label_overlay_frames_v1/20210722_120000_TLC1_B1_1200-1300_labels.jpg   |            6 | matched_tlc_hour_video           |
| TLC1 B1 1300-1400   | 2021-07-22T13:00:00 |             0 | written  | /home/oyavuz/PigBench/Week6_Unibo_Dataset_Validation/outputs/visual_label_check/label_overlay_frames_v1/20210722_130000_TLC1_B1_1300-1400_labels.jpg   |            6 | matched_tlc_hour_video           |
| TLC1 B1 1400-1500   | 2021-07-22T14:00:00 |             0 | written  | /home/oyavuz/PigBench/Week6_Unibo_Dataset_Validation/outputs/visual_label_check/label_overlay_frames_v1/20210722_140000_TLC1_B1_1400-1500_labels.jpg   |            6 | matched_tlc_hour_video           |
| TLC1 B1 800-900     | 2021-07-22T08:00:00 |             0 | written  | /home/oyavuz/PigBench/Week6_Unibo_Dataset_Validation/outputs/visual_label_check/label_overlay_frames_v1/20210722_080000_TLC1_B1_800-900_labels.jpg     |            6 | matched_tlc_hour_video           |
| TLC1 B1 900-1000    | 2021-07-22T09:00:00 |             0 | written  | /home/oyavuz/PigBench/Week6_Unibo_Dataset_Validation/outputs/visual_label_check/label_overlay_frames_v1/20210722_090000_TLC1_B1_900-1000_labels.jpg    |            6 | matched_tlc_hour_video           |
| c0001210722150000   | 2021-07-22T15:00:00 |             0 | written  | /home/oyavuz/PigBench/Week6_Unibo_Dataset_Validation/outputs/visual_label_check/label_overlay_frames_v1/20210722_150000_c0001210722150000_labels.jpg   |            6 | candidate_recovered_ctoken_video |
| c0001210722160000   | 2021-07-22T16:00:00 |             0 | written  | /home/oyavuz/PigBench/Week6_Unibo_Dataset_Validation/outputs/visual_label_check/label_overlay_frames_v1/20210722_160000_c0001210722160000_labels.jpg   |            6 | candidate_recovered_ctoken_video |
| c0001210722170000   | 2021-07-22T17:00:00 |             0 | written  | /home/oyavuz/PigBench/Week6_Unibo_Dataset_Validation/outputs/visual_label_check/label_overlay_frames_v1/20210722_170000_c0001210722170000_labels.jpg   |            6 | candidate_recovered_ctoken_video |
| c0001210722180000   | 2021-07-22T18:00:00 |             0 | written  | /home/oyavuz/PigBench/Week6_Unibo_Dataset_Validation/outputs/visual_label_check/label_overlay_frames_v1/20210722_180000_c0001210722180000_labels.jpg   |            6 | candidate_recovered_ctoken_video |

## Interpretation

The generated frames provide a visual check for timestamp/video matching. This is not yet the final bbox+label visualization requested by the task sheet. The next step is to link detector/tracker bounding boxes to these scan-sampling timestamps and then render bbox + colour ID + behaviour label overlays.
