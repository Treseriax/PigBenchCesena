# v78k3 TLC-to-c-code Mapping Resolver

This package creates a TLC camera-level mapping matrix.

Known:
- TLC1 is locked to GROUP_A / c0002 primary, c0000 duplicate, c0100 top-view.
- TLC2-TLC6 require manual/external evidence before mapping.

Boundary:
- This does not provide pen-level ROI.
- This does not run tracking.
- This does not create final behaviour labels.
