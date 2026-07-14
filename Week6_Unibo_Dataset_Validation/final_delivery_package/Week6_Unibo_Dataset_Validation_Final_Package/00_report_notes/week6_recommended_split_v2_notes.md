# Week 6 Recommended Split v2 Notes

## Purpose

This file adopts the optimized split candidate as the recommended Week 6 split v2. The original split files are preserved; this v2 split is saved as a separate recommended file.

## Rationale

The initial split had 3 missing behaviour classes in validation and 2 missing behaviour classes in test. The optimized split reduces this to 1 missing behaviour in validation and 1 missing behaviour in test, while preserving full train behaviour coverage and video-hour level no-leakage grouping.

## Split summary

| split   |   video_units |   label_count |   unique_colours |   unique_behaviours |   direct_tlc_records |   candidate_ctoken_records |   label_percentage |
|:--------|--------------:|--------------:|-----------------:|--------------------:|---------------------:|---------------------------:|-------------------:|
| test    |             2 |            72 |                6 |                  10 |                   36 |                         36 |              16.67 |
| train   |             8 |           288 |                6 |                  11 |                  216 |                         72 |              66.67 |
| val     |             2 |            72 |                6 |                  10 |                   36 |                         36 |              16.67 |

## Behaviour coverage

| split   |   present_behaviour_count |   total_behaviour_count |   missing_behaviour_count | missing_behaviours   |
|:--------|--------------------------:|------------------------:|--------------------------:|:---------------------|
| test    |                        10 |                      11 |                         1 | IA                   |
| train   |                        11 |                      11 |                         0 |                      |
| val     |                        10 |                      11 |                         1 | NU                   |

## Video-level split

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

## Leakage check

No video-hour split leakage detected.

## Interpretation

This split is the recommended split for reporting and future experiments. Because the dataset is small and contains rare behaviours, validation/test cannot cover all behaviour classes simultaneously under the current 8/2/2 video-hour split. The remaining missing classes should be documented as a dataset limitation.
