# Week 8 v47 Visualization Interface MVP Report

Decision:
- v47 decision: visualization_interface_mvp_created
- Clip count: 72
- Existing clip paths: 72
- Object count: 429
- Hard issue count: 0
- Warning count: 0
- Ready for manual visual review: True

Interface features:
- Video player
- Clip selector
- Canvas overlay
- Bounding box display
- Pig ID, colour identity and behaviour label display
- Timestamp and estimated frame display
- Checkbox layer toggles
- Manual issue note export

Run:
cd ~/PigBench
python Week8_Ground_Truth_Validation_Behaviour_Dataset_Preparation/interface/week8_visualizer_server_v47.py --port 8513

Open:
http://127.0.0.1:8513/

Scope:
This is the MVP visual validation interface. It uses v45 propagated ground truth. The v45 bounding boxes are scanpoint-anchor boxes repeated across the interval.
