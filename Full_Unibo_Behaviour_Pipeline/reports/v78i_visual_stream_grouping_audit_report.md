# v78i Visual Stream Grouping Audit

## Purpose

This stage records manual visual observations and computes same-hour visual similarity between encoded c-code videos and friendly TLC reference videos.

## Key manual observations

- TLC1 anchor and c0002 are visually the same.
- c0000 appears visually the same as c0002.
- c0001 and c0003 appear visually similar / same.
- c0100 appears to be the top-view of the c0000/c0002 visual group.
- c0101 appears to be the top-view of the c0001/c0003 visual group.
- No visible label or camera name was found.

## Interpretation

A strict one-to-one TLC→c-code mapping is not safe. The encoded c-codes appear to form visual stream groups and overhead/side-view relationships.

## Outputs

- `v78i_human_visual_observations.csv`
- `v78i_pairwise_visual_similarity.csv`
- `v78i_tlc1_anchor_similarity_by_hour.csv`
- `v78i_visual_group_decision_table.csv`
- `v78i_visual_stream_grouping_board.html`

## Next

Build v78j time-aware visual-stream resolver instead of forcing fixed TLC→c-code mapping.
