# v80f-4 Frame-based Baseline on Split B

- Decision: frame_based_baseline_splitB_completed
- Split B protocol: cross-camera/pen generalization, TLC6/M4 held out as test.
- Train frame rows: 107682
- Val frame rows: 9319
- Test frame rows: 25321
- Best test clip aggregation: majority_vote
- Best test clip macro F1: 0.048607
- Best test clip accuracy: 0.058411
- Hard issues: 0
- Ready for v80f-5 frame baseline comparison: True

This is a cross-camera frame-proxy baseline. It evaluates generalization to a held-out camera/pen, but it is not an RGB CNN and not a production classifier.
