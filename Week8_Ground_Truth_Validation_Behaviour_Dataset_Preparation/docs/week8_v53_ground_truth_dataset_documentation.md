# Week 8 Ground Truth Dataset Documentation

## Dataset scope

The Week 8 ground truth dataset is not the complete raw Unibo video archive. It is a validated subset built from annotated observation windows.

- Validated scanframes: 72
- Validated clips: 72
- Source videos represented in the validated subset: 12
- Clip duration policy: 10-second observation window per scanframe
- Clip-object annotations: 429
- Usable behaviour-labelled objects: 374
- Missing or unmapped behaviour objects: 55

Each scanframe corresponds to one annotation reference point. A 10-second clip was extracted around that observation window, and the behaviour label is propagated across the interval for the corresponding pig identity.

## Main data levels

### Clip level

Each clip represents one annotated observation window.

Typical fields:

- scan_frame_id
- video_id
- clip_path
- frame_count
- fps_used
- observation interval metadata

### Clip-object level

Each clip-object row represents one pig/object at the annotation anchor level.

Typical fields:

- scan_frame_id
- final_box_id
- visual marker colour
- behaviour pig ID
- behaviour code
- anchor bounding box
- behaviour propagation source

### Frame-object tracking-support level

The full hybrid tracking-support layer expands clip-object annotations to frame-object rows.

- Frame-object rows: 107113
- Stable tracklet ratio: 0.7582
- Recall fallback ratio: 0.1385
- Missing ratio: 0.1033
- Draw-ok ratio: 0.8967
- Review-needed ratio: 0.2418

Hybrid source interpretation:

- stable_tracklet: primary temporal tracking evidence
- recall_fallback: fallback detection-linked evidence; review-needed
- missing: no reliable box drawn; preserved explicitly

## Behaviour distribution

| behaviour_code_clean   |   clip_object_count |
|:-----------------------|--------------------:|
| STI                    |                 151 |
| LAI                    |                  55 |
| MISSING_OR_UNMAPPED    |                  55 |
| BOX                    |                  44 |
| AN                     |                  38 |
| NU                     |                  23 |
| PI                     |                  20 |
| IN                     |                  18 |
| SI                     |                  12 |
| BE                     |                   5 |
| DE                     |                   5 |
| IA                     |                   3 |

## Source video / scanframe summary

| video_id            |   scanframe_count | first_scanframe   | last_scanframe   |   total_object_frame_rows |   mean_draw_ok_ratio |   mean_missing_ratio |   mean_review_needed_ratio |
|:--------------------|------------------:|:------------------|:-----------------|--------------------------:|---------------------:|---------------------:|---------------------------:|
| TLC 1 -B1 0700-0800 |                 6 | scanframe_0000    | scanframe_0005   |                      8988 |             0.841619 |            0.158381  |                   0.294211 |
| TLC1 B1 1000-1100   |                 6 | scanframe_0006    | scanframe_0011   |                      8988 |             0.900233 |            0.0997666 |                   0.178494 |
| TLC1 B1 1100-1200   |                 6 | scanframe_0012    | scanframe_0017   |                      8988 |             0.77587  |            0.22413   |                   0.298894 |
| TLC1 B1 1200-1300   |                 6 | scanframe_0018    | scanframe_0023   |                      8988 |             0.931847 |            0.0681531 |                   0.226057 |
| TLC1 B1 1300-1400   |                 6 | scanframe_0024    | scanframe_0029   |                      8732 |             0.873138 |            0.126862  |                   0.248391 |
| TLC1 B1 1400-1500   |                 6 | scanframe_0030    | scanframe_0035   |                      8988 |             0.969262 |            0.0307376 |                   0.274021 |
| TLC1 B1 800-900     |                 6 | scanframe_0036    | scanframe_0041   |                      8988 |             0.949535 |            0.0504654 |                   0.162395 |
| TLC1 B1 900-1000    |                 6 | scanframe_0042    | scanframe_0047   |                      8988 |             0.964731 |            0.0352686 |                   0.15573  |
| c0001210722150000   |                 6 | scanframe_0048    | scanframe_0053   |                      8738 |             0.9139   |            0.0860995 |                   0.282412 |
| c0001210722160000   |                 6 | scanframe_0054    | scanframe_0059   |                      8739 |             0.950114 |            0.0498862 |                   0.135979 |
| c0001210722170000   |                 6 | scanframe_0060    | scanframe_0065   |                      9000 |             0.884222 |            0.115778  |                   0.291333 |
| c0001210722180000   |                 6 | scanframe_0066    | scanframe_0071   |                      8988 |             0.806499 |            0.193501  |                   0.353261 |

## Classification usage policy

This dataset is suitable for baseline and pilot classification experiments after validation. It should not be used to claim a robust production-level behaviour classifier. Rare classes remain limited and should be handled carefully.
