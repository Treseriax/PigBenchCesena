# Week 8 v47 Behaviour Dataset Visualizer Quickstart

Purpose:
This interface is the Week 8 MVP visual validation tool. It loads extracted 10-second clips and the propagated ground-truth annotation generated in v45.

Supported functions:
- video playback
- bounding-box overlays
- pig ID / colour identity / behaviour label display
- timestamp and estimated frame display
- checkbox layer toggles
- manual validation note export

Run from the server:
cd ~/PigBench
python Week8_Ground_Truth_Validation_Behaviour_Dataset_Preparation/interface/week8_visualizer_server_v47.py --port 8513

Open:
http://127.0.0.1:8513/

If port forwarding is needed from your local computer:
ssh -L 8513:127.0.0.1:8513 oyavuz@137.204.72.3

Stop:
Press Ctrl+C in the terminal running the server.

Validation notes:
/home/oyavuz/PigBench/Week8_Ground_Truth_Validation_Behaviour_Dataset_Preparation/validation/week8_v47_visual_validation_notes.csv

Scope:
v47 is an MVP visual validation interface. It uses the v45 propagated annotations. The v45 bounding boxes are scanpoint-anchor boxes repeated across the interval. Tracking-refined boxes can be integrated in a later refinement step.
