Week 8 v49c Anchor Validation Verdict Report

Decision:
v49c decision: anchor_validation_verdict_recorded

Verdict:
Anchor-frame annotations are validated.

Evidence:
- v49a found no coordinate-system mismatch.
- v49a found no bbox outside video bounds.
- v49a found no invalid basic bbox geometry.
- v49b generated exact anchor-frame overlays for 72/72 clips.
- Manual visual inspection confirmed that exact-anchor overlays are now correct.

Interpretation:
The earlier visual mismatch in the browser viewer is not a coordinate-system failure. It is mainly caused by using scanpoint-anchor bounding boxes across 10-second clips where pigs move.

Next action:
Proceed to tracking-refined per-frame boxes and identity review. The validation interface should keep anchor-frame mode as the reliable reference, while full-clip playback needs per-frame tracking boxes.

Important limitation:
Anchor-frame bbox correctness does not mean full-clip bbox correctness. Full-clip validation still requires tracking-refined boxes.
