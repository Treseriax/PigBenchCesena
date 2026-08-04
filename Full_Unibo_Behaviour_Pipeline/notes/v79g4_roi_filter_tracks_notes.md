# v79g4 ROI Filter Tracks

- Decision: roi_filtered_tracks_created
- Valid ROI polygons: 36
- Input track rows: 16233
- Kept track rows: 12015
- Rejected track rows: 4218
- Kept track ratio: 0.740159
- Input detection rows: 16233
- Kept detection rows: 12015
- Rejected detection rows: 4218
- Unique tracklets with ROI hit: 1632
- Hard issues: 0
- Ready for ROI-filtered clip fusion: True

This stage removes track/detection rows whose bbox centroid falls outside the target-pen ROI. It does not assign colour identity.
