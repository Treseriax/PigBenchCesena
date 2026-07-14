# Week 6 All-Server MP4 Video Inventory Notes

## Purpose

This corrected inventory scans shared server work folders in addition to the user's home PigBench folder. The previous inventory only scanned ~/PigBench and therefore mostly captured generated demo/tracking videos.

## Summary

| metric                               | value                        |
|:-------------------------------------|:-----------------------------|
| search_roots                         | /work, /home/oyavuz/PigBench |
| total_mp4_videos_found               | 125                          |
| total_size_gb                        | 27.453                       |
| unique_parent_folders                | 30                           |
| videos_under_work                    | 84                           |
| videos_under_home_project            | 41                           |
| videos_with_metadata                 | 125                          |
| videos_with_inferred_camera_id       | 84                           |
| videos_with_inferred_crate_or_pen_id | 0                            |
| videos_with_inferred_scan_window     | 24                           |

## Search root counts

| search_root           |   video_count |
|:----------------------|--------------:|
| /work                 |            84 |
| /home/oyavuz/PigBench |            41 |

## Top parent folders

| parent_folder                                                                                                             |   video_count |
|:--------------------------------------------------------------------------------------------------------------------------|--------------:|
| /work/pig/datasets/Unibo                                                                                                  |            84 |
| /home/oyavuz/PigBench/Week3_Behaviour_Dataset/final_outputs_corrected_scan_windows/videos                                 |             6 |
| /home/oyavuz/PigBench/Week2_Tracking_Study/final_outputs/videos                                                           |             4 |
| /home/oyavuz/PigBench/Week3_Behaviour_Dataset/data/videos                                                                 |             3 |
| /home/oyavuz/PigBench/Week3_Behaviour_Dataset/final_outputs/videos                                                        |             3 |
| /home/oyavuz/PigBench/Week2_Tracking_Study/outputs/tracking_videos/pigtrack0028_bytetrack                                 |             1 |
| /home/oyavuz/PigBench/Week3_Behaviour_Dataset/outputs/scan_window_aligned/tracking/scan_09_30_bytetrack                   |             1 |
| /home/oyavuz/PigBench/Week3_Behaviour_Dataset/outputs/visualization_interface/UniboVid2_sample                            |             1 |
| /home/oyavuz/PigBench/Week3_Behaviour_Dataset/outputs/tracking/UniboVid2_sample_bytetrack                                 |             1 |
| /home/oyavuz/PigBench/Week3_Behaviour_Dataset/outputs/scan_window_aligned/visualization/corrected_scan_windows/scan_09_50 |             1 |
| /home/oyavuz/PigBench/Week3_Behaviour_Dataset/outputs/scan_window_aligned/visualization/corrected_scan_windows/scan_09_40 |             1 |
| /home/oyavuz/PigBench/Week3_Behaviour_Dataset/outputs/scan_window_aligned/visualization/corrected_scan_windows/scan_09_30 |             1 |
| /home/oyavuz/PigBench/Week3_Behaviour_Dataset/outputs/scan_window_aligned/visualization/corrected_scan_windows/scan_09_20 |             1 |
| /home/oyavuz/PigBench/Week3_Behaviour_Dataset/outputs/scan_window_aligned/visualization/corrected_scan_windows/scan_09_10 |             1 |
| /home/oyavuz/PigBench/Week3_Behaviour_Dataset/outputs/scan_window_aligned/visualization/corrected_scan_windows/scan_09_00 |             1 |
| /home/oyavuz/PigBench/Week3_Behaviour_Dataset/outputs/scan_window_aligned/tracking/scan_09_50_bytetrack                   |             1 |
| /home/oyavuz/PigBench/Week3_Behaviour_Dataset/outputs/scan_window_aligned/tracking/scan_09_40_bytetrack                   |             1 |
| /home/oyavuz/PigBench/Week3_Behaviour_Dataset/outputs/scan_window_aligned/tracking/scan_09_10_bytetrack                   |             1 |
| /home/oyavuz/PigBench/Week3_Behaviour_Dataset/outputs/scan_window_aligned/tracking/scan_09_20_bytetrack                   |             1 |
| /home/oyavuz/PigBench/Week2_Tracking_Study/outputs/tracking_videos/pigtrack0028_deepocsort                                |             1 |
| /home/oyavuz/PigBench/Week2_Tracking_Study/outputs/detections/pigtrack0028_yolov8s                                        |             1 |
| /home/oyavuz/PigBench/Week3_Behaviour_Dataset/outputs/scan_window_aligned/detections/scan_09_50                           |             1 |
| /home/oyavuz/PigBench/Week3_Behaviour_Dataset/outputs/scan_window_aligned/detections/scan_09_40                           |             1 |
| /home/oyavuz/PigBench/Week3_Behaviour_Dataset/outputs/scan_window_aligned/detections/scan_09_30                           |             1 |
| /home/oyavuz/PigBench/Week3_Behaviour_Dataset/outputs/scan_window_aligned/detections/scan_09_20                           |             1 |
| /home/oyavuz/PigBench/Week3_Behaviour_Dataset/outputs/scan_window_aligned/detections/scan_09_10                           |             1 |
| /home/oyavuz/PigBench/Week3_Behaviour_Dataset/outputs/scan_window_aligned/detections/scan_09_00                           |             1 |
| /home/oyavuz/PigBench/Week3_Behaviour_Dataset/outputs/detections/UniboVid2_sample_yolov8s                                 |             1 |
| /home/oyavuz/PigBench/Week2_Tracking_Study/outputs/tracking_videos/pigtrack0028_botsort                                   |             1 |
| /home/oyavuz/PigBench/Week3_Behaviour_Dataset/outputs/scan_window_aligned/tracking/scan_09_00_bytetrack                   |             1 |

## Candidate raw work videos

| absolute_path                                    |   size_mb |   width |   height |     fps |   frame_count |   duration_sec |
|:-------------------------------------------------|----------:|--------:|---------:|--------:|--------------:|---------------:|
| /work/pig/datasets/Unibo/TLC 1 -B1 0700-0800.mp4 |   221.284 |     704 |      576 | 24.9997 |         89999 |        3600    |
| /work/pig/datasets/Unibo/TLC1 B1 1000-1100.mp4   |   115.063 |     704 |      576 | 24.9997 |         89999 |        3600    |
| /work/pig/datasets/Unibo/TLC1 B1 1100-1200.mp4   |   191.184 |     704 |      576 | 24.9997 |         89999 |        3600    |
| /work/pig/datasets/Unibo/TLC1 B1 1200-1300.mp4   |   278.039 |     704 |      576 | 24.9997 |         89999 |        3600    |
| /work/pig/datasets/Unibo/TLC1 B1 1300-1400.mp4   |   314.858 |     704 |      576 | 24.9992 |         89997 |        3600    |
| /work/pig/datasets/Unibo/TLC1 B1 1400-1500.mp4   |   364.757 |     704 |      576 | 24.9997 |         89999 |        3600    |
| /work/pig/datasets/Unibo/TLC1 B1 800-900.mp4     |   337.307 |     704 |      576 | 24.7797 |         89207 |        3600    |
| /work/pig/datasets/Unibo/TLC1 B1 900-1000.mp4    |   241.958 |     704 |      576 | 24.8961 |         89626 |        3600    |
| /work/pig/datasets/Unibo/c0000210722090000.mp4   |   715.027 |     704 |      576 | 25      |         90042 |        3601.68 |
| /work/pig/datasets/Unibo/c0000210722150000.mp4   |   729.076 |     704 |      576 | 25      |         90002 |        3600.08 |
| /work/pig/datasets/Unibo/c0000210722160000.mp4   |   763.906 |     704 |      576 | 25      |         90052 |        3602.08 |
| /work/pig/datasets/Unibo/c0000210722170000.mp4   |   877.617 |     704 |      576 | 25      |         90002 |        3600.08 |
| /work/pig/datasets/Unibo/c0000210722180000.mp4   |    58.263 |     704 |      576 | 24.8889 |         89600 |        3600    |
| /work/pig/datasets/Unibo/c0000210722190000.mp4   |    27.878 |     704 |      576 | 24.9967 |         89988 |        3600    |
| /work/pig/datasets/Unibo/c0001210722065004.mp4   |    19.2   |     704 |      576 | 24.934  |         15105 |         605.8  |
| /work/pig/datasets/Unibo/c0001210722070000.mp4   |    63.913 |     704 |      576 | 24.9053 |         89659 |        3600    |
| /work/pig/datasets/Unibo/c0001210722080000.mp4   |   188.633 |     704 |      576 | 24.97   |         89892 |        3600    |
| /work/pig/datasets/Unibo/c0001210722090000.mp4   |   123.559 |     704 |      576 | 24.9844 |         89944 |        3600    |
| /work/pig/datasets/Unibo/c0001210722100000.mp4   |    70.019 |     704 |      576 | 24.9997 |         89999 |        3600    |
| /work/pig/datasets/Unibo/c0001210722110000.mp4   |   112.26  |     704 |      576 | 24.9408 |         89787 |        3600    |
| /work/pig/datasets/Unibo/c0001210722120000.mp4   |   102.827 |     704 |      576 | 24.9858 |         89949 |        3600    |
| /work/pig/datasets/Unibo/c0001210722130000.mp4   |   165.095 |     704 |      576 | 24.9997 |         89999 |        3600    |
| /work/pig/datasets/Unibo/c0001210722140000.mp4   |   176.823 |     704 |      576 | 24.9994 |         89998 |        3600    |
| /work/pig/datasets/Unibo/c0001210722150000.mp4   |   105.672 |     704 |      576 | 24.9978 |         89992 |        3600    |
| /work/pig/datasets/Unibo/c0001210722160000.mp4   |   133.841 |     704 |      576 | 24.9997 |         89999 |        3600    |
| /work/pig/datasets/Unibo/c0001210722170000.mp4   |   110.687 |     704 |      576 | 24.985  |         89946 |        3600    |
| /work/pig/datasets/Unibo/c0001210722180000.mp4   |    57.675 |     704 |      576 | 24.9969 |         89989 |        3600    |
| /work/pig/datasets/Unibo/c0001210722190000.mp4   |    13.426 |     704 |      576 | 24.9861 |         89950 |        3600    |
| /work/pig/datasets/Unibo/c0002210722065004.mp4   |    71.95  |     704 |      576 | 24.9997 |         89999 |        3600    |
| /work/pig/datasets/Unibo/c0002210722070000.mp4   |   482.776 |     704 |      576 | 24.9997 |         89999 |        3600    |
| /work/pig/datasets/Unibo/c0002210722080000.mp4   |   515.274 |     704 |      576 | 24.9994 |         89998 |        3600    |
| /work/pig/datasets/Unibo/c0002210722090000.mp4   |   416.942 |     704 |      576 | 24.9994 |         89998 |        3600    |
| /work/pig/datasets/Unibo/c0002210722100000.mp4   |   390.445 |     704 |      576 | 24.9994 |         89998 |        3600    |
| /work/pig/datasets/Unibo/c0002210722110000.mp4   |   351.706 |     704 |      576 | 24.9997 |         89999 |        3600    |
| /work/pig/datasets/Unibo/c0002210722120000.mp4   |   425.138 |     704 |      576 | 24.9828 |         89938 |        3600    |
| /work/pig/datasets/Unibo/c0002210722130000.mp4   |   445.565 |     704 |      576 | 24.9997 |         89999 |        3600    |
| /work/pig/datasets/Unibo/c0002210722140000.mp4   |   487.653 |     704 |      576 | 24.9997 |         89999 |        3600    |
| /work/pig/datasets/Unibo/c0002210722150000.mp4   |   456.043 |     704 |      576 | 24.9997 |         89999 |        3600    |
| /work/pig/datasets/Unibo/c0002210722160000.mp4   |   456.797 |     704 |      576 | 24.9992 |         89997 |        3600    |
| /work/pig/datasets/Unibo/c0002210722170000.mp4   |   426.747 |     704 |      576 | 24.9997 |         89999 |        3600    |
| /work/pig/datasets/Unibo/c0002210722180000.mp4   |   393.039 |     704 |      576 | 24.9997 |         89999 |        3600    |
| /work/pig/datasets/Unibo/c0002210722190000.mp4   |    91.355 |     704 |      576 | 24.9997 |         89999 |        3600    |
| /work/pig/datasets/Unibo/c0003210722065004.mp4   |    93.7   |     704 |      576 | 24.9997 |         89999 |        3600    |
| /work/pig/datasets/Unibo/c0003210722070000.mp4   |   555.147 |     704 |      576 | 24.9983 |         89994 |        3600    |
| /work/pig/datasets/Unibo/c0003210722080000.mp4   |   567.075 |     704 |      576 | 24.9989 |         89996 |        3600    |
| /work/pig/datasets/Unibo/c0003210722090000.mp4   |   473.533 |     704 |      576 | 24.9989 |         89996 |        3600    |
| /work/pig/datasets/Unibo/c0003210722100000.mp4   |   478.589 |     704 |      576 | 24.9997 |         89999 |        3600    |
| /work/pig/datasets/Unibo/c0003210722110000.mp4   |   478.347 |     704 |      576 | 24.9992 |         89997 |        3600    |
| /work/pig/datasets/Unibo/c0003210722120000.mp4   |   518.328 |     704 |      576 | 24.9992 |         89997 |        3600    |
| /work/pig/datasets/Unibo/c0003210722130000.mp4   |   521.387 |     704 |      576 | 24.9997 |         89999 |        3600    |

## Interpretation

For Week 6, the important videos are likely the raw MP4 files under the shared work folder, not generated visualizations. The next step is to inspect the candidate raw work videos and identify which ones correspond to Unibo crates/pens, cameras, and annotation files.
