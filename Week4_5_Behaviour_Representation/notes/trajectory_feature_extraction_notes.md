# Trajectory Feature Extraction Notes

## Purpose

This step converts Week 3 corrected tracking outputs into interpretable trajectory-based behaviour representation features.

## Input

- `Week4_5_Behaviour_Representation/data/unibo_inputs/all_scan_windows_bytetrack_tracks_with_excel_window.csv`

## Feature families

### Track-level motion features

- Speed in pixels per second
- Normalized speed using image diagonal
- Acceleration in pixels per second squared
- Step displacement
- Trajectory length
- Net displacement
- Straightness index
- Turning angle
- Stationary ratio
- Fast motion ratio

### Group/social features

- Visible track count per frame
- Group centroid
- Group spread
- Nearest-neighbour distance
- Local density within 180 px radius

## Interpretation examples

- High speed and acceleration may indicate active movement or interaction.
- Low speed and high stationary ratio may indicate resting or lying behaviour.
- Low nearest-neighbour distance and high local density may indicate social contact or crowding.
- High group spread means animals are spatially dispersed.
- Low group spread means animals are clustered.

## Limitations

The features are based on centroid and bounding-box trajectories only. The current tracking data does not include snout keypoints, rotated bounding boxes, or skeletons. Therefore, feeding/drinking resource detection is approximated only in later ROI-based steps.
