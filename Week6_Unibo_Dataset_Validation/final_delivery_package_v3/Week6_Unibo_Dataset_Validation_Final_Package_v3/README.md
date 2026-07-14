# Week 6 Unibo Dataset Validation Final Package v3

Generated: `2026-07-06T00:43:53`

## Start here

Open these files first:

1. `presentation/Week6_Unibo_Dataset_Validation_Presentation_v2_with_Segment_Anything_Model.pdf`
2. `00_report_notes/week6_final_report_v2.md`
3. `00_report_notes/week6_final_executive_summary_v2.md`
4. `package_manifest_v3.csv`

## What is inside this package?

This package contains the final derived outputs for Week 6:

- Unified ground truth tables in Comma-Separated Values format and JavaScript Object Notation format.
- Scanpoint frame index and frame-label alignment tables.
- Recommended training, validation, and test split protocol.
- Detection outputs and detector quality control flags.
- Feature extractor outputs.
- Detector-backed learned crop embeddings.
- Classical GrabCut and Otsu segmentation baseline outputs.
- Segment Anything Model box-prompt segmentation outputs.
- Visualization interface demo files.
- Contact sheets and overlay images.
- Shared Excel task tracker.
- Final presentation.
- Scripts used for reproducibility.

## Key counts

- Manual behaviour labels: 432
- Scanpoint frames: 72
- Detector bounding boxes: 540
- Detector-backed embedding rows: 540
- Detector-backed embedding dimension: 896
- Classical segmentation feature rows: 540
- Segment Anything Model masks: 540
- Segment Anything Model frame overlays: 72
- Segment Anything Model failed rows: 0
- Recommended split: 288 training labels, 72 validation labels, 72 test labels

## Raw data

Raw videos are not included in this package because they already exist on the server.

Server location:

```text
/work/pig/datasets/Unibo
```

See also:

```text
raw_data_location.txt
```

## Visualization interface

See:

```text
interface_usage_notes.md
```

## Important interpretation

Manual labels, automatic detector bounding boxes, automatic segmentation masks, and candidate colour identity evidence are kept separate.

The Segment Anything Model masks are automatic model outputs, not manual segmentation ground truth.
