# Week 6 Dataset Statistics and Split Protocol

## Purpose

This note consolidates Week 6 dataset statistics after raw video inventory, Excel ground-truth parsing, video linkage, scanpoint frame extraction, YOLOv8-s bbox detection, and crop-based colour-marker analysis.

## Dataset overview metrics

| section             | metric                               |     value |
|:--------------------|:-------------------------------------|----------:|
| raw_video_inventory | raw_work_mp4_videos                  |   84      |
| raw_video_inventory | raw_work_total_size_gb               |   27.098  |
| raw_video_inventory | mean_video_duration_sec              | 3564.4    |
| video_naming        | naming_families                      |    2      |
| video_naming        | camera_tokens                        |    7      |
| ground_truth        | unified_gt_records                   |  432      |
| ground_truth        | unique_video_slots                   |   12      |
| ground_truth        | unique_colour_ids                    |    6      |
| ground_truth        | unique_behaviour_codes               |   11      |
| ground_truth        | direct_tlc_records                   |  288      |
| ground_truth        | candidate_recovered_ctoken_records   |  144      |
| ground_truth        | records_with_video_path              |  432      |
| ground_truth        | bbox_available_in_gt                 |    0      |
| scanpoint_frames    | scanpoint_frames                     |   72      |
| scanpoint_frames    | ok_extracted_frames                  |   72      |
| detections          | total_yolov8s_bboxes                 |  540      |
| detections          | mean_bboxes_per_frame                |    7.5    |
| detections          | min_bboxes_per_frame                 |    4      |
| detections          | max_bboxes_per_frame                 |   14      |
| detections          | mean_detection_score                 |    0.744  |
| detections          | median_detection_score               |    0.8314 |
| marker_candidates   | analysed_bbox_crops                  |  540      |
| marker_candidates   | high_marker_candidates               |  144      |
| marker_candidates   | medium_marker_candidates             |   52      |
| marker_candidates   | low_marker_candidates                |   69      |
| marker_candidates   | no_marker_detected                   |  275      |
| marker_candidates   | medium_or_high_candidate_assignments |  196      |

## Behaviour class distribution

| behaviour_code   | behaviour_label                                        |   count |   percentage |
|:-----------------|:-------------------------------------------------------|--------:|-------------:|
| STI              | Stesi inattivi sternali - Laying in a sternal position |     169 |        39.12 |
| LAI              | Stesi inattivi laterali - Laying in a lateral position |      69 |        15.97 |
| BOX              | INTERAZIONE BOX - Interaction with pen equipment       |      51 |        11.81 |
| AN               | Annusano, grufolano - Sniffing - rooting               |      50 |        11.57 |
| NU               | Si nutrono - Eating                                    |      23 |         5.32 |
| PI               | In piedi inattivi - Standing inactive                  |      22 |         5.09 |
| IN               | INTERAZIONE NEUTRA - Neutral interaction               |      20 |         4.63 |
| SI               | Seduti inattivi - Sitting inactive                     |      12 |         2.78 |
| DE               | Deambulano - Walking                                   |       6 |         1.39 |
| BE               | Bevono - Drinking                                      |       5 |         1.16 |
| IA               | INTERAZIONE AGGRESSIVA - Aggressive interaction        |       5 |         1.16 |

## Rare behaviour classes

| behaviour_code   | behaviour_label                                 |   count |   percentage |
|:-----------------|:------------------------------------------------|--------:|-------------:|
| SI               | Seduti inattivi - Sitting inactive              |      12 |         2.78 |
| DE               | Deambulano - Walking                            |       6 |         1.39 |
| BE               | Bevono - Drinking                               |       5 |         1.16 |
| IA               | INTERAZIONE AGGRESSIVA - Aggressive interaction |       5 |         1.16 |

## Colour ID distribution

| colour_id   | colour_raw             |   count |
|:------------|:-----------------------|--------:|
| blue        | blu - blue             |      72 |
| green       | verde - green          |      72 |
| no_color    | # - no color           |      72 |
| purple      | viola - purple         |      72 |
| red_neck    | rosso testa - red neck |      72 |
| red_tail    | rosso coda - red tail  |      72 |

## Hourly annotation distribution

| hour_start   | hour_end   | video_id            | video_match_status               |   label_count |   unique_colours |   unique_behaviours |
|:-------------|:-----------|:--------------------|:---------------------------------|--------------:|-----------------:|--------------------:|
| 07:00        | 08:00      | TLC 1 -B1 0700-0800 | matched_tlc_hour_video           |            36 |                6 |                   7 |
| 08:00        | 09:00      | TLC1 B1 800-900     | matched_tlc_hour_video           |            36 |                6 |                   7 |
| 09:00        | 10:00      | TLC1 B1 900-1000    | matched_tlc_hour_video           |            36 |                6 |                   7 |
| 10:00        | 11:00      | TLC1 B1 1000-1100   | matched_tlc_hour_video           |            36 |                6 |                   8 |
| 11:00        | 12:00      | TLC1 B1 1100-1200   | matched_tlc_hour_video           |            36 |                6 |                   4 |
| 12:00        | 13:00      | TLC1 B1 1200-1300   | matched_tlc_hour_video           |            36 |                6 |                   9 |
| 13:00        | 14:00      | TLC1 B1 1300-1400   | matched_tlc_hour_video           |            36 |                6 |                   6 |
| 14:00        | 15:00      | TLC1 B1 1400-1500   | matched_tlc_hour_video           |            36 |                6 |                   8 |
| 15:00        | 16:00      | c0001210722150000   | candidate_recovered_ctoken_video |            36 |                6 |                   5 |
| 16:00        | 17:00      | c0001210722160000   | candidate_recovered_ctoken_video |            36 |                6 |                   5 |
| 17:00        | 18:00      | c0001210722170000   | candidate_recovered_ctoken_video |            36 |                6 |                   7 |
| 18:00        | 19:00      | c0001210722180000   | candidate_recovered_ctoken_video |            36 |                6 |                   5 |

## Detection frame summary

|        | scan_frame_id   | video_match_status     |   bbox_count |   mean_score |   max_score |   mean_bbox_area |
|:-------|:----------------|:-----------------------|-------------:|-------------:|------------:|-----------------:|
| count  | 72              | 72                     |     72       |   72         |   72        |            72    |
| unique | 72              | 2                      |    nan       |  nan         |  nan        |           nan    |
| top    | scanframe_0000  | matched_tlc_hour_video |    nan       |  nan         |  nan        |           nan    |
| freq   | 1               | 48                     |    nan       |  nan         |  nan        |           nan    |
| mean   | nan             | nan                    |      7.5     |    0.756122  |    0.924435 |         13585.1  |
| std    | nan             | nan                    |      1.95008 |    0.0908793 |    0.019004 |          2564.81 |
| min    | nan             | nan                    |      4       |    0.511223  |    0.849499 |          8053.09 |
| 25%    | nan             | nan                    |      6       |    0.713731  |    0.914482 |         12047.9  |
| 50%    | nan             | nan                    |      7       |    0.772435  |    0.92689  |         13702.6  |
| 75%    | nan             | nan                    |      8       |    0.825496  |    0.938092 |         14724.2  |
| max    | nan             | nan                    |     14       |    0.903863  |    0.955514 |         20247.9  |

## Marker candidate distribution

| best_marker_colour   | marker_confidence   |   detection_count |   mean_marker_score |   mean_margin |   mean_detection_score |
|:---------------------|:--------------------|------------------:|--------------------:|--------------:|-----------------------:|
| blue                 | high                |                66 |          0.0700225  |   0.0547008   |               0.679712 |
| green                | high                |                32 |          0.0554607  |   0.0415972   |               0.686509 |
| red                  | high                |                30 |          0.0471601  |   0.0374596   |               0.710514 |
| purple               | high                |                16 |          0.051012   |   0.0300606   |               0.752434 |
| green                | low                 |                31 |          0.00776332 |   0.00438121  |               0.765229 |
| blue                 | low                 |                19 |          0.0103513  |   0.0037693   |               0.788478 |
| red                  | low                 |                11 |          0.0120817  |   0.00431425  |               0.806451 |
| purple               | low                 |                 8 |          0.00965338 |   0.00452631  |               0.85715  |
| blue                 | medium              |                15 |          0.0181503  |   0.00702063  |               0.75657  |
| green                | medium              |                13 |          0.0152959  |   0.0105775   |               0.724304 |
| purple               | medium              |                12 |          0.0207437  |   0.00720682  |               0.811969 |
| red                  | medium              |                12 |          0.014944   |   0.0112251   |               0.71636  |
| no_marker_detected   | none                |               275 |          0.00107892 |   0.000878595 |               0.756566 |

## Split protocol

The recommended split is video/hour-level, not frame-level. All labels from the same hourly video remain in the same split. This avoids leakage where frames from the same video hour appear in both training and evaluation sets.

### Split summary

| split   |   video_units |   label_count |   unique_colours |   unique_behaviours |   direct_tlc_records |   candidate_ctoken_records |   label_percentage |
|:--------|--------------:|--------------:|-----------------:|--------------------:|---------------------:|---------------------------:|-------------------:|
| test    |             2 |            72 |                6 |                   9 |                   36 |                         36 |              16.67 |
| train   |             8 |           288 |                6 |                  11 |                  216 |                         72 |              66.67 |
| val     |             2 |            72 |                6 |                   8 |                   36 |                         36 |              16.67 |

### Video-level split table

| split   | video_id            | hour_start   | hour_end   | video_match_status               | video_mapping_confidence         |   label_count |   unique_colours |   unique_behaviours |
|:--------|:--------------------|:-------------|:-----------|:---------------------------------|:---------------------------------|--------------:|-----------------:|--------------------:|
| test    | TLC1 B1 1200-1300   | 12:00        | 13:00      | matched_tlc_hour_video           | high                             |            36 |                6 |                   9 |
| test    | c0001210722180000   | 18:00        | 19:00      | candidate_recovered_ctoken_video | medium_needs_visual_confirmation |            36 |                6 |                   5 |
| train   | TLC 1 -B1 0700-0800 | 07:00        | 08:00      | matched_tlc_hour_video           | high                             |            36 |                6 |                   7 |
| train   | TLC1 B1 800-900     | 08:00        | 09:00      | matched_tlc_hour_video           | high                             |            36 |                6 |                   7 |
| train   | TLC1 B1 1000-1100   | 10:00        | 11:00      | matched_tlc_hour_video           | high                             |            36 |                6 |                   8 |
| train   | TLC1 B1 1100-1200   | 11:00        | 12:00      | matched_tlc_hour_video           | high                             |            36 |                6 |                   4 |
| train   | TLC1 B1 1300-1400   | 13:00        | 14:00      | matched_tlc_hour_video           | high                             |            36 |                6 |                   6 |
| train   | TLC1 B1 1400-1500   | 14:00        | 15:00      | matched_tlc_hour_video           | high                             |            36 |                6 |                   8 |
| train   | c0001210722160000   | 16:00        | 17:00      | candidate_recovered_ctoken_video | medium_needs_visual_confirmation |            36 |                6 |                   5 |
| train   | c0001210722170000   | 17:00        | 18:00      | candidate_recovered_ctoken_video | medium_needs_visual_confirmation |            36 |                6 |                   7 |
| val     | TLC1 B1 900-1000    | 09:00        | 10:00      | matched_tlc_hour_video           | high                             |            36 |                6 |                   7 |
| val     | c0001210722150000   | 15:00        | 16:00      | candidate_recovered_ctoken_video | medium_needs_visual_confirmation |            36 |                6 |                   5 |

## Important limitations

- Current labels come from one Excel sheet, one day, one camera/pen context. This is useful for Week 6 validation and prototyping, but not enough for a final generalization benchmark.
- The 15:00-19:00 video linkage is candidate recovered c-token mapping and should remain marked as medium confidence until visually confirmed.
- Bounding boxes are detector outputs, not manual ground-truth boxes.
- Crop marker features provide candidate bbox-to-colour association only. Red is ambiguous between red_neck and red_tail, and no_color cannot be recovered by marker detection.
- Rare classes such as drinking, walking, aggressive interaction, and sitting inactive need special handling or grouping for future classification experiments.

## Interpretation

At this stage, the Week 6 dataset is suitable for dataset validation, visualization demos, bbox extraction, feature extractor testing, and preliminary split-protocol design. It is not yet a fully identity-resolved frame-level behaviour dataset. The next technical step is to build feature extractor comparison tables from trajectory/bbox/crop/marker/embedding-style descriptors.
