# ROI Visual Quality Control

## Visual inspection result

The ROI overlay montage was visually inspected across all six corrected scan-window segments.

## Decision

The initial ROI definitions are acceptable for a first Week 4 feature engineering prototype, but they should not be interpreted as final feeder or drinker detectors.

## Observations

1. The coarse spatial grid zones are usable for general spatial behaviour representation.
2. The central_activity_candidate zone covers the main visible pen area where many pigs are located.
3. The front_gate_lower_candidate zone captures the lower/front barrier area and may be useful as a contact/resting/spatial-use candidate zone.
4. The upper_wall_resource_candidate and right_wall_candidate zones are intentionally approximate and should not be interpreted as exact resources.
5. The current ROI features are centroid/bounding-box based because the available Unibo tracking output does not include snout keypoints, rotated bounding boxes, or skeleton keypoints.

## Scientific limitation

Compared with Larsen et al. 2026, which uses rotated bounding boxes, head direction, snout keypoints, feeder/drinker ROIs, IoU, and distance-to-snout rules, the current ROI prototype is less behaviour-specific. Therefore, it should be reported as a spatial/resource/contact representation prototype rather than as a final feeding or drinking behaviour detector.

## Future refinement

Future versions can improve ROI behaviour detection by:
- defining precise feeder and drinker coordinates;
- using anterior body region instead of centroid only;
- adding snout keypoints or skeleton estimation;
- validating ROI-based events against manual labels.
