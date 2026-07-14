# Week 2 Tracking Pipeline Progress Notes

## 1. Project Goal

The goal of Week 2 is to build a pig tracking pipeline that converts pig detections into individual trajectories. The intended workflow is:

Pig video / image sequence
→ Object detection
→ Multi-object tracking
→ Trajectory CSV
→ Trajectory visualizations
→ Preparation for behaviour analysis

## 2. Unibo Video Investigation

The original Unibo recording was found at:

/work/pig/datasets/Unibo/c0000210722090000.vgz

The .vgz file was investigated using Linux file inspection, hex header inspection, FFprobe, and OpenCV.

Findings:

- The file command identified it as generic data.
- The header starts with XGV.
- FFprobe could not parse the file and returned an invalid data error.
- OpenCV VideoCapture could not open the file.

Conclusion:

The Unibo .vgz file does not appear to be directly readable using standard Python, OpenCV, or FFmpeg tools. A proprietary viewer/export workflow or MP4 screen-recording workaround is likely required.

## 3. Alternative Dataset Used

Because the Unibo .vgz file could not be directly processed, the official PigTrack dataset was used to develop and test the tracking pipeline.

The selected test sequence is:

pigtrack0028

Sequence information:

- Frame rate: 10 FPS
- Sequence length: 60 frames
- Image size: 1280 x 800
- Pen ID: 4
- Day/night flag: 0, daytime recording

## 4. Ground Truth Data

The PigTrack sequence uses MOT-style annotation format:

frame, track_id, x, y, width, height, confidence, class_id, visibility

Ground truth summary for pigtrack0028:

- Number of frames: 60
- Number of pig IDs: 12
- Number of ground-truth boxes: 720

Generated ground-truth outputs:

- GT trajectory CSV
- GT centroid tracks
- GT heatmap
- GT occupancy map
- GT first-frame bounding-box overlay

## 5. Detection Pipeline

The official YOLOv8-s PigBench checkpoint was used for object detection.

Detector:

YOLOv8-s

Config:

detection/configs/yolov8/yolov8_s.py

Checkpoint:

detection/data/pretrained_weights/yolov8_pigs/yolov8_s.pth

Detection output for pigtrack0028:

- Number of frames: 60
- Detection threshold: 0.05
- Total detections: 845
- Detection CSV created
- Annotated detection video created

## 6. ByteTrack Pipeline

ByteTrack was applied to the YOLOv8-s detection CSV using BoxMOT.

Tracker:

ByteTrack

Tracker config:

tracking/boxmot/configs/trackers/bytetrack.yaml

ByteTrack output for pigtrack0028:

- Track rows: 694
- Unique track IDs: 14
- Tracking CSV created
- Annotated tracking video created
- Centroid tracks created
- Heatmap created
- Occupancy map created

## 7. Preliminary Observation

The ground truth contains 12 pigs and 720 boxes. ByteTrack produced 14 unique track IDs and 694 track rows.

This suggests that the tracker is mostly working, but there may be some track fragmentation, missed associations, or duplicate/extra IDs.

## 8. Current Status

Completed:

- Week2_Tracking_Study folder structure
- Unibo .vgz investigation
- PigTrack sequence extraction
- Ground-truth trajectory visualization
- YOLOv8-s detection
- ByteTrack tracking
- ByteTrack trajectory visualization

Next steps:

1. Test BoT-SORT.
2. Test DeepOCSORT.
3. Compare trackers.
4. Prepare summary table.
5. Connect trajectory outputs to behaviour-analysis features.
