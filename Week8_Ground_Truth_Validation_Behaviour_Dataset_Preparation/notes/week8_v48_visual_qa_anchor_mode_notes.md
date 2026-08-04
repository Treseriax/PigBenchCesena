# Week 8 v48 Visual QA Anchor Mode

## Summary

- v48 decision: visual_qa_anchor_mode_created
- Clip count: 72
- Existing clip paths: 72
- Object count: 429
- Hard issue count: 0
- Ready for anchor-frame visual review: True

## Main interpretation

v47 works technically, but scanpoint-anchor boxes can drift during full-clip playback. v48 adds anchor-frame review mode and makes this limitation explicit.

## Run

cd ~/PigBench
python Week8_Ground_Truth_Validation_Behaviour_Dataset_Preparation/interface/week8_visualizer_server_v48.py --port 8514

Open http://127.0.0.1:8514/
