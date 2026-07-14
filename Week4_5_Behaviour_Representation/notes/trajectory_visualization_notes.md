# Trajectory Visualization Notes

## Purpose

This step generates visual examples of trajectory-derived behaviour representations from the Week 3 corrected tracking outputs.

## Generated visualizations

- Dominant track trajectory plots
- Centroid heatmaps
- Normalized occupancy maps
- Speed-over-time plots
- Group/social feature plots over time
- Cross-segment summary plots for speed, stationary ratio, group spread, and local density

## Interpretation

Trajectory plots show movement paths of dominant track IDs. Heatmaps and occupancy maps show where pigs spend more time. Speed-over-time plots reveal active or stationary periods. Group feature plots summarize social/spatial patterns such as visible track count and local density.

## Important limitation

Some speed spikes may be caused by tracking fragmentation, ID switches, or bounding-box jitter. Therefore, speed plots clip extreme values for visualization only, and future robust feature extraction should include smoothing or outlier filtering.
