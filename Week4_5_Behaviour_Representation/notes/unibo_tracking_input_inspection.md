# Unibo Tracking Input Inspection

## Input files

- Tracks: `Week4_5_Behaviour_Representation/data/unibo_inputs/all_scan_windows_bytetrack_tracks_with_excel_window.csv`
- Behaviour labels: `Week4_5_Behaviour_Representation/data/unibo_inputs/scan_window_behaviour_labels.csv`
- Tracking summary: `Week4_5_Behaviour_Representation/data/unibo_inputs/scan_window_tracking_with_time_summary.csv`

## Tracks shape

- Rows: 23234
- Columns: 27

## Segment overview

| segment_id   |   frames |   track_rows |   unique_track_ids |   min_frame |   max_frame |   min_segment_time |   max_segment_time |   mean_tracks_per_frame |   mean_score |   mean_bbox_area | excel_interval_start   | excel_interval_end   |
|:-------------|---------:|-------------:|-------------------:|------------:|------------:|-------------------:|-------------------:|------------------------:|-------------:|-----------------:|:-----------------------|:---------------------|
| scan_09_00   |      500 |         5139 |                 33 |           1 |         500 |                  0 |              19.96 |                10.278   |     0.795995 |          61689.3 | 09:00:00               | 09:00:10             |
| scan_09_10   |      375 |         3294 |                 19 |           1 |         375 |                  0 |              14.96 |                 8.784   |     0.730806 |          46438.6 | 09:10:00               | 09:10:10             |
| scan_09_20   |      250 |         2682 |                 19 |           1 |         250 |                  0 |               9.96 |                10.728   |     0.714827 |          68142   | 09:20:00               | 09:20:10             |
| scan_09_30   |      500 |         4968 |                 32 |           1 |         500 |                  0 |              19.96 |                 9.936   |     0.664336 |          44870.4 | 09:30:00               | 09:30:10             |
| scan_09_40   |      500 |         4481 |                 15 |           1 |         500 |                  0 |              19.96 |                 8.962   |     0.815961 |          56044.6 | 09:40:00               | 09:40:10             |
| scan_09_50   |      325 |         2670 |                 10 |           1 |         325 |                  0 |              12.96 |                 8.21538 |     0.84972  |          53977.1 | 09:50:00               | 09:50:10             |

## Data quality note

No missing values were found in the tracking CSV.

## Interpretation

The Week 3 corrected tracking outputs contain frame-level bounding boxes, centroids, track IDs, segment IDs, and Excel scan-window timing. These are suitable for trajectory-based behaviour representation features such as speed, acceleration, displacement, trajectory length, turning angle, occupancy maps, heatmaps, nearest-neighbour distance, local density, and ROI time.
