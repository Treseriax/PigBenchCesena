# Week 8 v69c Error Analysis

- v69c decision: error_analysis_completed
- Prediction rows: 150
- Error rows: 111
- Correct rows: 39
- Current split errors: 63
- Group-aware split errors: 48
- Hard issues: 0
- Ready for v70a improved baseline: True

Top confusions:

             split_policy crop_type       model_type actual predicted  error_count
current_recommended_split context10 nearest_centroid    STI        IN            7
current_recommended_split context10 nearest_centroid    STI       BOX            6
current_recommended_split context10 nearest_centroid    LAI       STI            4
current_recommended_split context10 nearest_centroid    STI        SI            4
current_recommended_split context10 nearest_centroid     AN       BOX            3
current_recommended_split context10 nearest_centroid    LAI       BOX            3
current_recommended_split context10 nearest_centroid     NU       LAI            3
current_recommended_split context10 nearest_centroid    LAI        IN            2
current_recommended_split context10 nearest_centroid     NU       BOX            2
current_recommended_split context10 nearest_centroid    STI        IA            2

Weakest class summaries:

              split_policy crop_type       model_type actual  support  correct  mean_confidence  errors   recall
 current_recommended_split context10 nearest_centroid     BE        1        0         0.002912       1 0.000000
 current_recommended_split context10 nearest_centroid     DE        2        0         0.005523       2 0.000000
 current_recommended_split context10 nearest_centroid     IA        2        0         0.003708       2 0.000000
 current_recommended_split context10 nearest_centroid    STI       30        3         0.003803      27 0.100000
 current_recommended_split context10 nearest_centroid     NU       11        2         0.003429       9 0.181818
 current_recommended_split context10 nearest_centroid    LAI       14        3         0.003591      11 0.214286
 current_recommended_split context10 nearest_centroid     PI        3        1         0.004181       2 0.333333
 current_recommended_split context10 nearest_centroid     AN        6        2         0.005197       4 0.333333
 current_recommended_split context10 nearest_centroid    BOX        9        4         0.003881       5 0.444444
selected_group_aware_split     tight             knn3     BE        1        0         0.666667       1 0.000000
selected_group_aware_split     tight             knn3     DE        1        0         0.333333       1 0.000000
selected_group_aware_split     tight             knn3     SI        1        0         0.333333       1 0.000000
