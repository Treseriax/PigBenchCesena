# v80f-3 Frame-based Baseline on Split A

- Decision: frame_based_baseline_splitA_completed
- Model type: logistic regression on ROI bbox/frame-proxy features
- Train frame rows: 114967
- Val frame rows: 9915
- Test frame rows: 17440
- Best test clip aggregation: average_probability
- Best test clip macro F1: 0.042168
- Best test clip accuracy: 0.091463
- Hard issues: 0
- Ready for v80f-4 Split B generalization eval: True

This is a frame-proxy baseline, not an RGB image CNN and not a production classifier. Clip-level results are produced using majority voting and average probability aggregation.
