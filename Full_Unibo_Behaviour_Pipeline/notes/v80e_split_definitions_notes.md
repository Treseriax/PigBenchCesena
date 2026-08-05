# v80e Leakage-safe Split Definitions

- Decision: leakage_safe_splits_created
- Split A: grouped video-level split, no same video across train/val/test.
- Split B: cross-camera/pen split with TLC6/M4 held out as test and TLC2/B6 as validation.
- Split A video leakage count: 0
- Split B video leakage count: 0
- Ready for v80f frame-based baseline: True

These split definitions apply to the current validated 36-video / 2768-clip label-bearing subset. They do not claim completed behaviour annotations for all 84 videos.
