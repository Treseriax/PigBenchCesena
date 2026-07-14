# Week 6 Split Optimizer Notes

## Purpose

This step searches alternative train/val/test splits at video-hour level. It does not overwrite the current split. It creates an optimized candidate split that minimizes missing behaviour classes in validation/test while preserving no-leakage video-hour grouping.

## Constraints

- Train/val/test split unit is the hourly video, not individual frames.
- Train has 8 hourly videos, validation has 2, and test has 2.
- Validation and test must each include at least one direct TLC video and one candidate recovered c-token video.
- Lower score is better. The score prioritizes full train behaviour coverage, then fewer missing validation/test behaviours, then fewer missing rare behaviours.

## Best candidate score

|   score |   train_missing_count |   val_missing_count |   test_missing_count | train_missing_behaviours   | val_missing_behaviours   | test_missing_behaviours   |   train_rare_missing_count |   val_rare_missing_count |   test_rare_missing_count | train_rare_missing   | val_rare_missing   | test_rare_missing   |   train_behaviour_count |   val_behaviour_count |   test_behaviour_count | train_units                                                                                                                                                                                              | val_units                                         | test_units                                      |   train_direct_units |   train_candidate_units |   val_direct_units |   val_candidate_units |   test_direct_units |   test_candidate_units |
|--------:|----------------------:|--------------------:|---------------------:|:---------------------------|:-------------------------|:--------------------------|---------------------------:|-------------------------:|--------------------------:|:---------------------|:-------------------|:--------------------|------------------------:|----------------------:|-----------------------:|:---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|:--------------------------------------------------|:------------------------------------------------|---------------------:|------------------------:|-------------------:|----------------------:|--------------------:|-----------------------:|
|     550 |                     0 |                   1 |                    1 |                            | NU                       | IA                        |                          0 |                        0 |                         1 |                      |                    | IA                  |                      11 |                    10 |                     10 | TLC 1 -B1 0700-0800__07:00|TLC1 B1 1000-1100__10:00|TLC1 B1 1100-1200__11:00|TLC1 B1 1300-1400__13:00|TLC1 B1 1400-1500__14:00|TLC1 B1 900-1000__09:00|c0001210722160000__16:00|c0001210722180000__18:00 | TLC1 B1 1200-1300__12:00|c0001210722150000__15:00 | TLC1 B1 800-900__08:00|c0001210722170000__17:00 |                    6 |                       2 |                  1 |                     1 |                   1 |                      1 |

## Optimized split summary

| split   |   video_units |   label_count |   unique_colours |   unique_behaviours |   direct_tlc_records |   candidate_ctoken_records |   label_percentage |
|:--------|--------------:|--------------:|-----------------:|--------------------:|---------------------:|---------------------------:|-------------------:|
| test    |             2 |            72 |                6 |                  10 |                   36 |                         36 |              16.67 |
| train   |             8 |           288 |                6 |                  11 |                  216 |                         72 |              66.67 |
| val     |             2 |            72 |                6 |                  10 |                   36 |                         36 |              16.67 |

## Optimized behaviour coverage

| split   |   present_behaviour_count |   total_behaviour_count |   missing_behaviour_count | missing_behaviours   |
|:--------|--------------------------:|------------------------:|--------------------------:|:---------------------|
| test    |                        10 |                      11 |                         1 | IA                   |
| train   |                        11 |                      11 |                         0 |                      |
| val     |                        10 |                      11 |                         1 | NU                   |

## Current vs optimized comparison

| split   |   current_missing_count | current_missing_behaviours   |   optimized_missing_count | optimized_missing_behaviours   |
|:--------|------------------------:|:-----------------------------|--------------------------:|:-------------------------------|
| train   |                       0 |                              |                         0 |                                |
| val     |                       3 | BE,DE,IA                     |                         1 | NU                             |
| test    |                       2 | NU,SI                        |                         1 | IA                             |

## Optimized video-level split

| split   | split_unit_id              | video_id            | hour_start   | hour_end   | video_match_status               | video_mapping_confidence         |   label_count |   unique_colours |   unique_behaviours | behaviours                    |
|:--------|:---------------------------|:--------------------|:-------------|:-----------|:---------------------------------|:---------------------------------|--------------:|-----------------:|--------------------:|:------------------------------|
| test    | TLC1 B1 800-900__08:00     | TLC1 B1 800-900     | 08:00        | 09:00      | matched_tlc_hour_video           | high                             |            36 |                6 |                   7 | AN,IN,LAI,NU,PI,SI,STI        |
| test    | c0001210722170000__17:00   | c0001210722170000   | 17:00        | 18:00      | candidate_recovered_ctoken_video | medium_needs_visual_confirmation |            36 |                6 |                   7 | AN,BE,BOX,DE,LAI,SI,STI       |
| train   | TLC 1 -B1 0700-0800__07:00 | TLC 1 -B1 0700-0800 | 07:00        | 08:00      | matched_tlc_hour_video           | high                             |            36 |                6 |                   7 | AN,DE,IA,LAI,PI,SI,STI        |
| train   | TLC1 B1 900-1000__09:00    | TLC1 B1 900-1000    | 09:00        | 10:00      | matched_tlc_hour_video           | high                             |            36 |                6 |                   7 | AN,BOX,IN,LAI,NU,PI,STI       |
| train   | TLC1 B1 1000-1100__10:00   | TLC1 B1 1000-1100   | 10:00        | 11:00      | matched_tlc_hour_video           | high                             |            36 |                6 |                   8 | AN,BE,BOX,DE,LAI,PI,SI,STI    |
| train   | TLC1 B1 1100-1200__11:00   | TLC1 B1 1100-1200   | 11:00        | 12:00      | matched_tlc_hour_video           | high                             |            36 |                6 |                   4 | AN,BOX,LAI,STI                |
| train   | TLC1 B1 1300-1400__13:00   | TLC1 B1 1300-1400   | 13:00        | 14:00      | matched_tlc_hour_video           | high                             |            36 |                6 |                   6 | AN,IA,IN,LAI,SI,STI           |
| train   | TLC1 B1 1400-1500__14:00   | TLC1 B1 1400-1500   | 14:00        | 15:00      | matched_tlc_hour_video           | high                             |            36 |                6 |                   8 | AN,BE,BOX,IN,NU,PI,SI,STI     |
| train   | c0001210722160000__16:00   | c0001210722160000   | 16:00        | 17:00      | candidate_recovered_ctoken_video | medium_needs_visual_confirmation |            36 |                6 |                   5 | AN,BOX,IN,LAI,STI             |
| train   | c0001210722180000__18:00   | c0001210722180000   | 18:00        | 19:00      | candidate_recovered_ctoken_video | medium_needs_visual_confirmation |            36 |                6 |                   5 | AN,BOX,DE,LAI,STI             |
| val     | TLC1 B1 1200-1300__12:00   | TLC1 B1 1200-1300   | 12:00        | 13:00      | matched_tlc_hour_video           | high                             |            36 |                6 |                   9 | AN,BE,BOX,DE,IA,IN,LAI,PI,STI |
| val     | c0001210722150000__15:00   | c0001210722150000   | 15:00        | 16:00      | candidate_recovered_ctoken_video | medium_needs_visual_confirmation |            36 |                6 |                   5 | AN,BOX,LAI,SI,STI             |

## Interpretation

If the optimized candidate reduces validation/test missing behaviours without breaking train coverage and without video leakage, it can replace the initial split in the next controlled step. If improvement is minimal, the current split can be kept and the missing rare classes should be documented as a dataset limitation.
