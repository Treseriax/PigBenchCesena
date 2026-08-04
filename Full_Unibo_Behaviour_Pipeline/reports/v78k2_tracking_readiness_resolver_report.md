# v78k2 Tracking Readiness Resolver

This package classifies each target by readiness for tracking.

It separates:

1. Camera/pen association readiness
2. TLC-to-encoded-video mapping readiness
3. Frame-level pen ROI readiness
4. Final tracking readiness

Important claim boundary:
This stage does not run tracking, does not create final behaviour labels, and does not claim full tracking readiness.
