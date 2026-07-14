# Week 6 Visualization Interface Demo

## Purpose

This demo addresses the visualization interface requirement. It loads the nested scanpoint annotation JSON and displays frame metadata, manual colour-ID behaviour labels, detector bboxes, and overlay visualizations.

## Files

- Streamlit app: `/home/oyavuz/PigBench/Week6_Unibo_Dataset_Validation/interface_demo/week6_visualization_streamlit_app.py`
- Static HTML viewer: `/home/oyavuz/PigBench/Week6_Unibo_Dataset_Validation/interface_demo/week6_static_visualization_viewer.html`
- Interface index CSV: `/home/oyavuz/PigBench/Week6_Unibo_Dataset_Validation/interface_demo/week6_visualization_interface_index.csv`
- Source JSON: `/home/oyavuz/PigBench/Week6_Unibo_Dataset_Validation/outputs/unified_ground_truth/week6_scanpoint_annotations_nested_for_viewer.json`

## How to run Streamlit

```bash
cd ~/PigBench
streamlit run Week6_Unibo_Dataset_Validation/interface_demo/week6_visualization_streamlit_app.py --server.port 8504
```

## Interpretation

The demo is intended for visual quality control and dataset exploration. It does not claim final identity-resolved bbox-to-colour ground truth. Manual labels are shown separately from detector bboxes because bbox-to-colour assignment remains candidate-level.
