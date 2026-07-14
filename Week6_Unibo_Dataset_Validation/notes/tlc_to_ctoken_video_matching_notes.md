# TLC to c-token Video Matching Notes

## Purpose

The Excel annotations cover 07:00-19:00, but only TLC-named videos were directly matched for 07:00-15:00. This step compares overlapping TLC videos with c-token videos from the same hours to infer which c-token camera corresponds to TLC1/B1.

## Camera similarity summary

| ctoken_camera_token   |   comparisons |   mean_corr |   mean_mad |   mean_similarity_score |   max_similarity_score |
|:----------------------|--------------:|------------:|-----------:|------------------------:|-----------------------:|
| c0001                 |             8 |    0.525862 |    27.274  |               0.418905  |               0.5459   |
| c0000                 |             1 |    0.471034 |    43.9944 |               0.298507  |               0.298507 |
| c0100                 |             8 |    0.386981 |    38.5568 |               0.235778  |               0.42153  |
| c0101                 |             8 |    0.377761 |    46.1535 |               0.196767  |               0.405771 |
| c0002                 |             8 |    0.259371 |    52.2295 |               0.054549  |               0.449223 |
| c0003                 |             8 |    0.205863 |    56.663  |              -0.0163448 |               0.185647 |

## Recommended c-token camera

`c0001`

## Candidate recovered videos for 15:00-19:00

| target_hour_start   | recommended_ctoken_camera   | recommended_video_path                         | recommended_video_filename   | status                    |
|:--------------------|:----------------------------|:-----------------------------------------------|:-----------------------------|:--------------------------|
| 15:00:00            | c0001                       | /work/pig/datasets/Unibo/c0001210722150000.mp4 | c0001210722150000.mp4        | candidate_recovered_video |
| 16:00:00            | c0001                       | /work/pig/datasets/Unibo/c0001210722160000.mp4 | c0001210722160000.mp4        | candidate_recovered_video |
| 17:00:00            | c0001                       | /work/pig/datasets/Unibo/c0001210722170000.mp4 | c0001210722170000.mp4        | candidate_recovered_video |
| 18:00:00            | c0001                       | /work/pig/datasets/Unibo/c0001210722180000.mp4 | c0001210722180000.mp4        | candidate_recovered_video |

## Contact sheets

- `/home/oyavuz/PigBench/Week6_Unibo_Dataset_Validation/outputs/visual_label_check/tlc_ctoken_matching_contact_sheets/tlc_vs_c0000_contact_sheet.jpg`
- `/home/oyavuz/PigBench/Week6_Unibo_Dataset_Validation/outputs/visual_label_check/tlc_ctoken_matching_contact_sheets/tlc_vs_c0001_contact_sheet.jpg`
- `/home/oyavuz/PigBench/Week6_Unibo_Dataset_Validation/outputs/visual_label_check/tlc_ctoken_matching_contact_sheets/tlc_vs_c0100_contact_sheet.jpg`

## Interpretation

The highest-similarity c-token camera can be used as a candidate mapping for unmatched afternoon labels, but the contact sheet should be visually inspected before treating this mapping as confirmed ground truth.
