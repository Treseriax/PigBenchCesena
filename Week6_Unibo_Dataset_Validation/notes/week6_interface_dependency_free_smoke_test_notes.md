# Week 6 Interface Dependency-Free Smoke Test

## Purpose

Streamlit is not installed in the current environment. This smoke test validates the dependency-free static interface deliverable instead.

## Result summary

| metric                       |   value |
|:-----------------------------|--------:|
| checks_total                 |      10 |
| pass_count                   |      10 |
| warn_count                   |       0 |
| fail_count                   |       0 |
| streamlit_runtime_required   |   False |
| static_html_smoke_test_ready |    True |

## Checks

| check                               | status   | evidence                                                                                                                                                                                                     | recommendation   |
|:------------------------------------|:---------|:-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|:-----------------|
| static_html_viewer_exists           | PASS     | interface_demo/week6_static_visualization_viewer.html                                                                                                                                                        | No action.       |
| streamlit_app_file_exists           | PASS     | interface_demo/week6_visualization_streamlit_app.py                                                                                                                                                          | No action.       |
| interface_index_exists              | PASS     | interface_demo/week6_visualization_interface_index.csv                                                                                                                                                       | No action.       |
| nested_json_exists                  | PASS     | outputs/unified_ground_truth/week6_scanpoint_annotations_nested_for_viewer.json                                                                                                                              | No action.       |
| static_html_contains_expected_terms | PASS     | found_terms=['Week', 'frame', 'annotation', 'bbox', 'behaviour']                                                                                                                                             | No action.       |
| interface_index_rows_72             | PASS     | rows=72                                                                                                                                                                                                      | No action.       |
| interface_index_columns             | PASS     | columns=['scan_frame_id', 'timestamp', 'video_id', 'video_match_status', 'manual_label_count', 'detector_bbox_count', 'behaviour_summary', 'frame_image_path', 'overlay_image_path', 'overlay_image_exists'] | No action.       |
| nested_json_parseable               | PASS     | type=dict; top_keys=['dataset', 'version', 'frame_count', 'notes', 'frames']; serialized_len=305487                                                                                                          | No action.       |
| interface_index_image_path_columns  | PASS     | image_path_cols=['frame_image_path', 'overlay_image_path']                                                                                                                                                   | No action.       |
| interface_image_paths_readable      | PASS     | [{'column': 'frame_image_path', 'total_rows': 72, 'resolved_paths': 72, 'cv2_readable_images': 72}, {'column': 'overlay_image_path', 'total_rows': 72, 'resolved_paths': 72, 'cv2_readable_images': 72}]     | No action.       |

## Remaining risks

No FAIL or WARN items remain for the static interface smoke test.

## How to serve the static interface

```bash
cd ~/PigBench/Week6_Unibo_Dataset_Validation
python -m http.server 8505 --directory interface_demo
```

Then open:

`http://localhost:8505/week6_static_visualization_viewer.html`

