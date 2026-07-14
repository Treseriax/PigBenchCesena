# Week 4 Start Notes — Behaviour Representation and Feature Engineering

## Objective

The goal of Week 4 is to move from dataset construction to behaviour representation. We are not yet training a final behaviour classifier. Instead, we focus on extracting robust, interpretable, and transferable behavioural descriptors from video, detections, tracks, and annotations.

## Main representation families

1. Trajectory-based features
2. ROI/resource-based features
3. Social interaction features
4. Skeleton/keypoint-based features
5. Mesh-based features
6. Embedding/video-based features

## Important new reading

The Larsen et al. 2026 paper on feeding and drinking behaviour shows that large-scale behavioural data can be produced efficiently by combining computer vision with ethological instantaneous sampling. The paper uses pig detection, rotated bounding boxes, head direction, snout keypoints, feeder/drinker ROIs, IoU, distance-in-pixels, and RGB filtering to estimate feeder and drinker occupation.

This supports a Week 4 feature engineering direction based on:
- ROI occupation
- distance to feeder/drinker
- time spent in resource zones
- occupancy duration
- sampling interval evaluation
- pen-level resource use patterns

## First implementation priority

Use existing Week 3 corrected tracking outputs to extract trajectory and ROI-style features:
- speed
- acceleration
- displacement
- trajectory length
- turning angle
- occupancy maps
- heatmaps
- time spent in manually defined zones
- nearest-neighbour distance
- local density

## Conservative note

The current Unibo tracking data does not yet include snout keypoints or rotated bounding boxes. Therefore, Week 4 will first implement centroid/bounding-box based trajectory and ROI features. Snout/keypoint-based resource detection will be discussed as a future extension.
