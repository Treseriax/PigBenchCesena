# Week 6 Unified Ground Truth v1 Notes

## Purpose

This table converts the parsed Unibo Excel scan-sampling annotations into the unified Week 6 ground-truth format. It is label-only at this stage: behaviour labels are available, but bounding boxes are not yet linked.

## Summary

| metric                        | value                   |
|:------------------------------|:------------------------|
| total_unified_records         | 432                     |
| unique_videos_or_video_slots  | 12                      |
| records_with_matched_video    | 288                     |
| records_without_matched_video | 144                     |
| bbox_available_records        | 0                       |
| unique_pigs_colours           | 6                       |
| unique_behaviour_codes        | 11                      |
| annotation_granularity        | 10_minute_scan_sampling |

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

## Colour distribution

| colour_id   | colour_raw             |   count |
|:------------|:-----------------------|--------:|
| blue        | blu - blue             |      72 |
| green       | verde - green          |      72 |
| no_color    | # - no color           |      72 |
| purple      | viola - purple         |      72 |
| red_neck    | rosso testa - red neck |      72 |
| red_tail    | rosso coda - red tail  |      72 |

## Missing / ambiguous label report

| issue_type      |   count | interpretation                                                                                                                           |
|:----------------|--------:|:-----------------------------------------------------------------------------------------------------------------------------------------|
| missing_bbox    |     432 | Manual Excel annotations provide behaviour labels but not bounding boxes; bbox must be linked later from detector/tracker outputs.       |
| unmatched_video |     144 | Labels from 15:00-19:00 currently have no matching TLC hourly video filename; candidate c-token videos must be inspected before linking. |
| rare_classes    |       4 | Some behaviour classes have very few labels and may be unsuitable for train/test split without grouping or special handling.             |

## Interpretation

The unified table is suitable as a first machine-learning label source, but it should be interpreted as scan-sampling labels rather than dense frame-level labels. For visual verification and feature extraction, matched TLC video records can be used immediately. Unmatched afternoon labels should only be linked after confirming which raw videos correspond to those hours.
