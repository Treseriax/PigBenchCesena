# ROI / Resource-Based Feature Extraction Notes

## Purpose

This step creates a first ROI-based behaviour representation prototype from Week 3 corrected tracking outputs. It estimates how much time each track spends inside coarse spatial zones and approximate resource/contact candidate zones.

## Connection to Larsen et al. 2026

The Larsen et al. 2026 feeding/drinking paper uses pig detection, rotated bounding boxes, head direction, snout keypoints, and feeder/drinker ROIs to estimate resource occupation. Our current Unibo tracking data does not contain snout keypoints or rotated bounding boxes, so this prototype uses centroid and bounding-box based ROI approximation only.

## Generated feature files

- `roi_frame_level_features.csv`
- `roi_track_level_summary.csv`
- `roi_track_dominant_zone_summary.csv`
- `roi_zone_segment_summary.csv`
- `roi_resource_candidate_segment_summary.csv`
- `manual_roi_zones_initial_absolute.csv`
- `manual_roi_zones_initial.json`

## Interpretation

High time spent in a zone may indicate resting, activity, resource use, or social contact, depending on the zone location. Resource candidate zones are not final feeder/drinker labels; they are approximate regions for feature engineering and should be visually refined.

## Limitation

Without snout keypoints and rotated bounding boxes, this approach cannot reliably distinguish actual feeding/drinking from simply being near a feeder/drinker area. It should therefore be treated as a representation prototype, not a final behaviour detector.
