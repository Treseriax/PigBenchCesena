# Week 6 Scanpoint Frame Extraction Notes

## Purpose

This step extracts one video frame for each unique manual Excel scan-sampling timestamp. These frames are the basis for visual quality control, detector bbox extraction, colour-marker analysis, and segmentation tests.

## Outputs

- Frame directory: `/home/oyavuz/PigBench/Week6_Unibo_Dataset_Validation/outputs/visual_label_check/all_scanpoint_frames`
- Scan frame index: `/home/oyavuz/PigBench/Week6_Unibo_Dataset_Validation/outputs/unified_ground_truth/week6_scanpoint_frame_index.csv`
- Long label table per scan frame: `/home/oyavuz/PigBench/Week6_Unibo_Dataset_Validation/outputs/unified_ground_truth/week6_scanpoint_frame_labels_long.csv`
- Extraction status summary: `/home/oyavuz/PigBench/Week6_Unibo_Dataset_Validation/outputs/unified_ground_truth/week6_scanpoint_frame_extraction_status_summary.csv`

## Extraction status summary

| extraction_status   | video_match_status               |   frame_count |
|:--------------------|:---------------------------------|--------------:|
| ok                  | candidate_recovered_ctoken_video |            24 |
| ok                  | matched_tlc_hour_video           |            48 |

## Interpretation

Each extracted frame should have six manual pig-colour labels from the Excel sheet. Bounding boxes are still pending and will be added after running or linking a pig detector/tracker.
