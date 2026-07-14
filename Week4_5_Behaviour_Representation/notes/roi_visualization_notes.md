# ROI Visualization Notes

## Purpose

This step visualizes the initial manually defined ROI zones and summarizes zone occupancy patterns across scan-window segments.

## Generated visualizations

- ROI overlay images for each scan-window segment
- Resource candidate zone instance ratio by segment
- Resource candidate zone track-seconds by segment
- Spatial grid occupancy heatmaps
- Dominant spatial grid zone distribution

## Interpretation

The ROI overlays are used to visually inspect whether approximate zones match meaningful areas in the pen. Occupancy charts summarize how much tracked pig centroids fall inside each zone. These features are useful as interpretable spatial behaviour representations.

## Connection to resource-based behaviour detection

The Larsen et al. feeding/drinking paper uses precise feeder and drinker ROIs with head/snout information. In our current prototype, ROI features are based only on centroid and bounding-box tracking. Therefore, they should be treated as resource/contact candidate features rather than final feeding or drinking detections.
