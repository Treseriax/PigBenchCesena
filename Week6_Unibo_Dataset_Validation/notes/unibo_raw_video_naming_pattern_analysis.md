# Unibo Raw Video Naming Pattern Analysis

## Purpose

This note analyses the naming conventions of the 84 raw MP4 videos found under `/work/pig/datasets/Unibo`. The goal is to infer possible camera, pen/crate, date, and time information before building the unified ground-truth table.

## Naming family summary

| naming_family         |   video_count |   total_size_gb |   mean_duration_sec |
|:----------------------|--------------:|----------------:|--------------------:|
| c_token_datetime_like |            76 |          25.082 |             3560.65 |
| TLC_B_time_range      |             8 |           2.016 |             3600    |

## Camera token summary

| naming_family         | parsed_camera_id   |   video_count |
|:----------------------|:-------------------|--------------:|
| TLC_B_time_range      | TLC1               |             8 |
| c_token_datetime_like | c0000              |             6 |
| c_token_datetime_like | c0001              |            14 |
| c_token_datetime_like | c0002              |            14 |
| c_token_datetime_like | c0003              |            14 |
| c_token_datetime_like | c0100              |            14 |
| c_token_datetime_like | c0101              |            14 |

## Pen/box token summary

| naming_family         | parsed_pen_or_box_id   |   video_count |
|:----------------------|:-----------------------|--------------:|
| TLC_B_time_range      | B1                     |             8 |
| c_token_datetime_like |                        |            76 |

## First parsed videos

| filename                | naming_family         | parsed_camera_id   | parsed_pen_or_box_id   |   parsed_date_token | parsed_start_time   | parsed_end_time   |   size_mb |   duration_sec |
|:------------------------|:----------------------|:-------------------|:-----------------------|--------------------:|:--------------------|:------------------|----------:|---------------:|
| TLC 1 -B1 0700-0800.mp4 | TLC_B_time_range      | TLC1               | B1                     |                     | 07:00               | 08:00             |   221.284 |        3600    |
| TLC1 B1 1000-1100.mp4   | TLC_B_time_range      | TLC1               | B1                     |                     | 10:00               | 11:00             |   115.063 |        3600    |
| TLC1 B1 1100-1200.mp4   | TLC_B_time_range      | TLC1               | B1                     |                     | 11:00               | 12:00             |   191.184 |        3600    |
| TLC1 B1 1200-1300.mp4   | TLC_B_time_range      | TLC1               | B1                     |                     | 12:00               | 13:00             |   278.039 |        3600    |
| TLC1 B1 1300-1400.mp4   | TLC_B_time_range      | TLC1               | B1                     |                     | 13:00               | 14:00             |   314.858 |        3600    |
| TLC1 B1 1400-1500.mp4   | TLC_B_time_range      | TLC1               | B1                     |                     | 14:00               | 15:00             |   364.757 |        3600    |
| TLC1 B1 800-900.mp4     | TLC_B_time_range      | TLC1               | B1                     |                     | 08:00               | 09:00             |   337.307 |        3600    |
| TLC1 B1 900-1000.mp4    | TLC_B_time_range      | TLC1               | B1                     |                     | 09:00               | 10:00             |   241.958 |        3600    |
| c0000210722090000.mp4   | c_token_datetime_like | c0000              |                        |              210722 | 09:00:00            |                   |   715.027 |        3601.68 |
| c0000210722150000.mp4   | c_token_datetime_like | c0000              |                        |              210722 | 15:00:00            |                   |   729.076 |        3600.08 |
| c0000210722160000.mp4   | c_token_datetime_like | c0000              |                        |              210722 | 16:00:00            |                   |   763.906 |        3602.08 |
| c0000210722170000.mp4   | c_token_datetime_like | c0000              |                        |              210722 | 17:00:00            |                   |   877.617 |        3600.08 |
| c0000210722180000.mp4   | c_token_datetime_like | c0000              |                        |              210722 | 18:00:00            |                   |    58.263 |        3600    |
| c0000210722190000.mp4   | c_token_datetime_like | c0000              |                        |              210722 | 19:00:00            |                   |    27.878 |        3600    |
| c0001210722065004.mp4   | c_token_datetime_like | c0001              |                        |              210722 | 06:50:04            |                   |    19.2   |         605.8  |
| c0001210722070000.mp4   | c_token_datetime_like | c0001              |                        |              210722 | 07:00:00            |                   |    63.913 |        3600    |
| c0001210722080000.mp4   | c_token_datetime_like | c0001              |                        |              210722 | 08:00:00            |                   |   188.633 |        3600    |
| c0001210722090000.mp4   | c_token_datetime_like | c0001              |                        |              210722 | 09:00:00            |                   |   123.559 |        3600    |
| c0001210722100000.mp4   | c_token_datetime_like | c0001              |                        |              210722 | 10:00:00            |                   |    70.019 |        3600    |
| c0001210722110000.mp4   | c_token_datetime_like | c0001              |                        |              210722 | 11:00:00            |                   |   112.26  |        3600    |
| c0001210722120000.mp4   | c_token_datetime_like | c0001              |                        |              210722 | 12:00:00            |                   |   102.827 |        3600    |
| c0001210722130000.mp4   | c_token_datetime_like | c0001              |                        |              210722 | 13:00:00            |                   |   165.095 |        3600    |
| c0001210722140000.mp4   | c_token_datetime_like | c0001              |                        |              210722 | 14:00:00            |                   |   176.823 |        3600    |
| c0001210722150000.mp4   | c_token_datetime_like | c0001              |                        |              210722 | 15:00:00            |                   |   105.672 |        3600    |
| c0001210722160000.mp4   | c_token_datetime_like | c0001              |                        |              210722 | 16:00:00            |                   |   133.841 |        3600    |
| c0001210722170000.mp4   | c_token_datetime_like | c0001              |                        |              210722 | 17:00:00            |                   |   110.687 |        3600    |
| c0001210722180000.mp4   | c_token_datetime_like | c0001              |                        |              210722 | 18:00:00            |                   |    57.675 |        3600    |
| c0001210722190000.mp4   | c_token_datetime_like | c0001              |                        |              210722 | 19:00:00            |                   |    13.426 |        3600    |
| c0002210722065004.mp4   | c_token_datetime_like | c0002              |                        |              210722 | 06:50:04            |                   |    71.95  |        3600    |
| c0002210722070000.mp4   | c_token_datetime_like | c0002              |                        |              210722 | 07:00:00            |                   |   482.776 |        3600    |
| c0002210722080000.mp4   | c_token_datetime_like | c0002              |                        |              210722 | 08:00:00            |                   |   515.274 |        3600    |
| c0002210722090000.mp4   | c_token_datetime_like | c0002              |                        |              210722 | 09:00:00            |                   |   416.942 |        3600    |
| c0002210722100000.mp4   | c_token_datetime_like | c0002              |                        |              210722 | 10:00:00            |                   |   390.445 |        3600    |
| c0002210722110000.mp4   | c_token_datetime_like | c0002              |                        |              210722 | 11:00:00            |                   |   351.706 |        3600    |
| c0002210722120000.mp4   | c_token_datetime_like | c0002              |                        |              210722 | 12:00:00            |                   |   425.138 |        3600    |
| c0002210722130000.mp4   | c_token_datetime_like | c0002              |                        |              210722 | 13:00:00            |                   |   445.565 |        3600    |
| c0002210722140000.mp4   | c_token_datetime_like | c0002              |                        |              210722 | 14:00:00            |                   |   487.653 |        3600    |
| c0002210722150000.mp4   | c_token_datetime_like | c0002              |                        |              210722 | 15:00:00            |                   |   456.043 |        3600    |
| c0002210722160000.mp4   | c_token_datetime_like | c0002              |                        |              210722 | 16:00:00            |                   |   456.797 |        3600    |
| c0002210722170000.mp4   | c_token_datetime_like | c0002              |                        |              210722 | 17:00:00            |                   |   426.747 |        3600    |

## Interpretation

The TLC-style filenames appear to explicitly encode a camera token, a B/box token, and a time range. The c-token filenames appear to encode camera-like and date/time-like tokens, but their exact semantics should be confirmed using annotation files or supervisor metadata before being treated as ground truth.
