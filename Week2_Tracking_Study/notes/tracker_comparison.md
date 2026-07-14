# Tracker Comparison on pigtrack0028

## Sequence information

- Sequence: pigtrack0028
- Frames: 60
- Resolution: 1280 x 800
- FPS: 10

## Ground Truth

- GT pig IDs: 12
- GT boxes: 720

## Detector used

- Detector: YOLOv8-s
- Config: detection/configs/yolov8/yolov8_s.py
- Checkpoint: detection/data/pretrained_weights/yolov8_pigs/yolov8_s.pth
- Detection threshold: 0.05
- Total detections: 845

## Tracker comparison table

| Tracker | Frames | Track rows | Unique track IDs | Main outputs |
|---|---:|---:|---:|---|
| ByteTrack | 60 | 694 | 14 | CSV + video + centroid tracks + heatmap + occupancy map |
| BoT-SORT | 60 | 695 | 14 | CSV + video + centroid tracks + heatmap + occupancy map |
| DeepOCSORT | 60 | 697 | 16 | CSV + video + centroid tracks + heatmap + occupancy map |

## Preliminary interpretation

All three trackers successfully produced trajectory CSV files, annotated tracking videos, centroid tracks, heatmaps, and occupancy maps.

Observations:

- Ground truth contains 12 pig identities and 720 annotated boxes.
- ByteTrack produced 694 track rows and 14 unique track IDs.
- BoT-SORT produced 695 track rows and 14 unique track IDs.
- DeepOCSORT produced 697 track rows and 16 unique track IDs.
- ByteTrack and BoT-SORT produced very similar results.
- DeepOCSORT produced slightly more track IDs, which may indicate more track fragmentation or extra identity creation.
- All trackers produced slightly fewer track rows than the ground-truth box count, suggesting some missed associations or filtered detections.

## Qualitative notes

### ByteTrack

- Good baseline tracker.
- Produced stable results on most frames.
- Generated 14 IDs instead of the 12 GT identities, suggesting some fragmentation or extra tracks.

### BoT-SORT

- Similar numerical performance to ByteTrack.
- Ran successfully without ReID.
- Generated 14 IDs, same as ByteTrack.
- Useful as a motion-based baseline when ReID weights are unavailable.

### DeepOCSORT

- Required a small code patch because the implementation initialized the ReID backend even when embedding was disabled.
- Ran with appearance embedding disabled.
- Produced 16 unique IDs, more than ByteTrack and BoT-SORT.
- This may suggest more fragmentation under the no-ReID setting.

## Current conclusion

For pigtrack0028, ByteTrack and BoT-SORT look like the stronger first baselines because they produced fewer extra identities than DeepOCSORT. DeepOCSORT can be tested again later if a valid ReID model weight is available.

## Next steps

1. Visually inspect tracking videos and centroid-track images.
2. Identify clear examples of ID switches, missing tracks, or fragmented tracks.
3. Add qualitative observations to the report.
4. Connect trajectory outputs to behaviour-analysis features such as zone occupancy, movement path, and activity level.
