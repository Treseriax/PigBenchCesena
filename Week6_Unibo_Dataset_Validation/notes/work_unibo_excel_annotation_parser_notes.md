# Work Unibo Excel Annotation Parser Notes

## Purpose

This parser converts the manual Excel scan-sampling annotation sheet into a long-format table. Each row corresponds to one pig colour/ID at one observation timestamp with one behaviour code.

## Input file

- `/work/pig/datasets/Unibo/Giorno 1 - 22_7_2021 tlc1 FASCIA 9-10.xlsx`

## Parsed structure

- Rows 3-4 define the 07:00-13:00 hour blocks and 10-minute observation points.
- Rows 6-11 contain pig-colour behaviour annotations for 07:00-13:00.
- Rows 14-15 define the 13:00-19:00 hour blocks and 10-minute observation points.
- Rows 17-22 contain pig-colour behaviour annotations for 13:00-19:00.

## Behaviour distribution

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

## Colour distribution

| colour_id   | colour_raw             |   count |
|:------------|:-----------------------|--------:|
| blue        | blu - blue             |      72 |
| green       | verde - green          |      72 |
| no_color    | # - no color           |      72 |
| purple      | viola - purple         |      72 |
| red_neck    | rosso testa - red neck |      72 |
| red_tail    | rosso coda - red tail  |      72 |

## Video linkage summary

| hour_start   | hour_end   | matched_video_status       | matched_video_filename   |   annotation_count |
|:-------------|:-----------|:---------------------------|:-------------------------|-------------------:|
| 07:00        | 08:00      | matched_tlc_hour_video     | TLC 1 -B1 0700-0800.mp4  |                 36 |
| 08:00        | 09:00      | matched_tlc_hour_video     | TLC1 B1 800-900.mp4      |                 36 |
| 09:00        | 10:00      | matched_tlc_hour_video     | TLC1 B1 900-1000.mp4     |                 36 |
| 10:00        | 11:00      | matched_tlc_hour_video     | TLC1 B1 1000-1100.mp4    |                 36 |
| 11:00        | 12:00      | matched_tlc_hour_video     | TLC1 B1 1100-1200.mp4    |                 36 |
| 12:00        | 13:00      | matched_tlc_hour_video     | TLC1 B1 1200-1300.mp4    |                 36 |
| 13:00        | 14:00      | matched_tlc_hour_video     | TLC1 B1 1300-1400.mp4    |                 36 |
| 14:00        | 15:00      | matched_tlc_hour_video     | TLC1 B1 1400-1500.mp4    |                 36 |
| 15:00        | 16:00      | no_matching_tlc_hour_video |                          |                 36 |
| 16:00        | 17:00      | no_matching_tlc_hour_video |                          |                 36 |
| 17:00        | 18:00      | no_matching_tlc_hour_video |                          |                 36 |
| 18:00        | 19:00      | no_matching_tlc_hour_video |                          |                 36 |

## Missing / linkage report

| check                                  |   value | issue                                                                |
|:---------------------------------------|--------:|:---------------------------------------------------------------------|
| total_annotations                      |     432 |                                                                      |
| missing_behaviour_label_mapping        |       0 | Behaviour code not found in extracted/fallback mapping               |
| annotations_without_matching_tlc_video |     144 | No TLC hourly MP4 found for that hour block                          |
| annotations_with_matching_tlc_video    |     288 |                                                                      |
| bbox_available_at_this_stage           |       0 | This Excel parser extracts labels only; bbox linkage is a later step |

## Interpretation

The Excel file provides manual scan-sampling behaviour labels for six colour-coded pigs across 12 hours. Hourly TLC videos are currently matched for the available TLC1 B1 video files. Annotations without a matching TLC video are still preserved in the label table, but they cannot yet be directly visualized or linked to frame indices unless the corresponding raw videos are identified.
