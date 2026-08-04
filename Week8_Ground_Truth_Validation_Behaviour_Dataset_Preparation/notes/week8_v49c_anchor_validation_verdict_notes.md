# Week 8 v49c Anchor Validation Verdict

## Summary

- v49c decision: anchor_validation_verdict_recorded
- Verdict: anchor-frame annotations validated
- Clip count: 72
- Saved anchor overlays: 72
- BBox coordinate fix needed: False
- Annotation mapping repair needed: False
- Tracking-refined per-frame boxes needed: True
- Ready for v50: True

## Interpretation

Exact anchor overlays look correct. The earlier mismatch is not a resolution/coordinate-system problem. It is caused by using fixed scanpoint-anchor boxes during full-clip playback while pigs move.

## Next

Proceed to v50: tracking-refined per-frame boxes and identity review.
