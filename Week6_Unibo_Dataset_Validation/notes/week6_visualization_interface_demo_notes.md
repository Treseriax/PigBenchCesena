# Week 6 Visualization Interface Demo Notes

## Purpose

This step creates a minimal visualization interface demo using the nested scanpoint annotation JSON. It closes the interface gap by providing both a Streamlit app and a static HTML viewer.

## Outputs

| artifact               | path                                                                            | exists   | purpose                                                  |
|:-----------------------|:--------------------------------------------------------------------------------|:---------|:---------------------------------------------------------|
| nested_annotation_json | outputs/unified_ground_truth/week6_scanpoint_annotations_nested_for_viewer.json | True     | Frame-level JSON with manual labels and detector bboxes. |
| streamlit_app          | interface_demo/week6_visualization_streamlit_app.py                             | True     | Interactive interface demo.                              |
| static_html_viewer     | interface_demo/week6_static_visualization_viewer.html                           | True     | Portable static visualization demo.                      |
| interface_index_csv    | interface_demo/week6_visualization_interface_index.csv                          | True     | Compact index of 72 scan frames for the viewer.          |

## Viewer coverage

- Indexed scan frames: `72`
- Frames with overlay image: `72`
- Total manual labels listed: `432`
- Total detector bboxes listed: `540`

## Interpretation

The visualization demo loads scanpoint annotation JSON and displays frame-level metadata, manual behaviour labels, detector bboxes, and overlay images. It is a dataset exploration/QC interface rather than a final annotation editing tool.
