# Week 6 MP4 Video Inventory Notes

## Purpose

This note documents the first Week 6 dataset inventory step. The goal is to identify all MP4 videos currently available under the PigBench project folder and extract basic metadata for later Unibo dataset analysis.

## Summary

| metric                               |   value |
|:-------------------------------------|--------:|
| total_mp4_videos_found               |  41     |
| total_size_gb                        |   0.355 |
| unique_parent_folders                |  29     |
| videos_with_inferred_camera_id       |   0     |
| videos_with_inferred_crate_or_pen_id |   0     |
| videos_with_inferred_scan_window     |  24     |
| videos_with_metadata                 |  41     |

## Parent folder counts

| parent_folder                                                                                                             |   video_count |
|:--------------------------------------------------------------------------------------------------------------------------|--------------:|
| /home/oyavuz/PigBench/Week3_Behaviour_Dataset/final_outputs_corrected_scan_windows/videos                                 |             6 |
| /home/oyavuz/PigBench/Week2_Tracking_Study/final_outputs/videos                                                           |             4 |
| /home/oyavuz/PigBench/Week3_Behaviour_Dataset/data/videos                                                                 |             3 |
| /home/oyavuz/PigBench/Week3_Behaviour_Dataset/final_outputs/videos                                                        |             3 |
| /home/oyavuz/PigBench/Week3_Behaviour_Dataset/outputs/scan_window_aligned/tracking/scan_09_10_bytetrack                   |             1 |
| /home/oyavuz/PigBench/Week3_Behaviour_Dataset/outputs/tracking/UniboVid2_sample_bytetrack                                 |             1 |
| /home/oyavuz/PigBench/Week3_Behaviour_Dataset/outputs/scan_window_aligned/visualization/corrected_scan_windows/scan_09_50 |             1 |
| /home/oyavuz/PigBench/Week3_Behaviour_Dataset/outputs/scan_window_aligned/visualization/corrected_scan_windows/scan_09_40 |             1 |
| /home/oyavuz/PigBench/Week3_Behaviour_Dataset/outputs/scan_window_aligned/visualization/corrected_scan_windows/scan_09_30 |             1 |
| /home/oyavuz/PigBench/Week3_Behaviour_Dataset/outputs/scan_window_aligned/visualization/corrected_scan_windows/scan_09_20 |             1 |
| /home/oyavuz/PigBench/Week3_Behaviour_Dataset/outputs/scan_window_aligned/visualization/corrected_scan_windows/scan_09_10 |             1 |
| /home/oyavuz/PigBench/Week3_Behaviour_Dataset/outputs/scan_window_aligned/visualization/corrected_scan_windows/scan_09_00 |             1 |
| /home/oyavuz/PigBench/Week3_Behaviour_Dataset/outputs/scan_window_aligned/tracking/scan_09_50_bytetrack                   |             1 |
| /home/oyavuz/PigBench/Week3_Behaviour_Dataset/outputs/scan_window_aligned/tracking/scan_09_40_bytetrack                   |             1 |
| /home/oyavuz/PigBench/Week3_Behaviour_Dataset/outputs/scan_window_aligned/tracking/scan_09_30_bytetrack                   |             1 |
| /home/oyavuz/PigBench/Week3_Behaviour_Dataset/outputs/scan_window_aligned/tracking/scan_09_20_bytetrack                   |             1 |
| /home/oyavuz/PigBench/Week3_Behaviour_Dataset/outputs/scan_window_aligned/detections/scan_09_50                           |             1 |
| /home/oyavuz/PigBench/Week3_Behaviour_Dataset/outputs/scan_window_aligned/tracking/scan_09_00_bytetrack                   |             1 |
| /home/oyavuz/PigBench/Week2_Tracking_Study/outputs/detections/pigtrack0028_yolov8s                                        |             1 |
| /home/oyavuz/PigBench/Week3_Behaviour_Dataset/outputs/scan_window_aligned/detections/scan_09_40                           |             1 |
| /home/oyavuz/PigBench/Week3_Behaviour_Dataset/outputs/scan_window_aligned/detections/scan_09_30                           |             1 |
| /home/oyavuz/PigBench/Week3_Behaviour_Dataset/outputs/scan_window_aligned/detections/scan_09_20                           |             1 |
| /home/oyavuz/PigBench/Week3_Behaviour_Dataset/outputs/scan_window_aligned/detections/scan_09_10                           |             1 |
| /home/oyavuz/PigBench/Week3_Behaviour_Dataset/outputs/scan_window_aligned/detections/scan_09_00                           |             1 |
| /home/oyavuz/PigBench/Week3_Behaviour_Dataset/outputs/detections/UniboVid2_sample_yolov8s                                 |             1 |
| /home/oyavuz/PigBench/Week2_Tracking_Study/outputs/tracking_videos/pigtrack0028_deepocsort                                |             1 |
| /home/oyavuz/PigBench/Week2_Tracking_Study/outputs/tracking_videos/pigtrack0028_bytetrack                                 |             1 |
| /home/oyavuz/PigBench/Week2_Tracking_Study/outputs/tracking_videos/pigtrack0028_botsort                                   |             1 |
| /home/oyavuz/PigBench/Week3_Behaviour_Dataset/outputs/visualization_interface/UniboVid2_sample                            |             1 |

## Interpretation

This inventory is the first step before extracting ground-truth labels, building a unified annotation table, and proposing train/test splits. If crate, pen, camera, or session IDs cannot be inferred from filenames or folders, they will need to be mapped manually or extracted from associated annotation/metadata files.
