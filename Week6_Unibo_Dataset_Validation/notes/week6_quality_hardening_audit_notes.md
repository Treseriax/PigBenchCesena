# Week 6 Quality Hardening Audit

## Purpose

The task-sheet compliance audit confirms that required outputs exist. This quality-hardening audit checks whether any technically complete item still needs extra validation before final package v2.

## Summary

| status   | severity   |   count |
|:---------|:-----------|--------:|
| PASS     | none       |      18 |
| WARN     | medium     |       2 |

## Remaining risks / warnings

| area         | check_name                           | status   | severity   | evidence                                                                            | recommendation                                                                        |
|:-------------|:-------------------------------------|:---------|:-----------|:------------------------------------------------------------------------------------|:--------------------------------------------------------------------------------------|
| Segmentation | segmentation_visual_qc_needed        | WARN     | medium     | Automatic masks exist, but visual/manual quality scoring has not been recorded yet. | Create segmentation visual QC sampling table with selected overlay frames.            |
| Feature      | learned_crop_embedding_output_exists | WARN     | medium     | No learned embedding table found.                                                   | Add a lightweight learned crop embedding baseline if torchvision/torch are available. |

## Interpretation

The project is task-sheet complete, but the listed quality warnings should be resolved or explicitly validated before final package v2.
