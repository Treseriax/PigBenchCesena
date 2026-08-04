# v78j Time-Aware Visual Stream Resolver

## Purpose

v78j replaces the unsafe fixed TLC→c-code mapping with a visual-stream resolver.

## Visual groups

- GROUP_A: c0000/c0002 side or duplicate view, c0100 top view. TLC1/c0002 anchor locked.
- GROUP_B: c0001/c0003 side or duplicate view, c0101 top view.

## Manual action

Fill:

`/home/oyavuz/PigBench/Full_Unibo_Behaviour_Pipeline/outputs/v78j_time_aware_visual_stream_resolver/Full_Unibo_Time_Aware_Visual_Stream_Resolver/v78j_MANUAL_FILL_visual_stream_resolution.csv`

Allowed `selected_visual_group_id`:
- GROUP_A
- GROUP_B
- NOT_VISIBLE
- UNRESOLVED

Allowed `selected_view_type`:
- side
- top
- side_and_top
- not_visible
- unresolved

TLC1 rows are prefilled and locked from the visual anchor. Other rows must be resolved by visual evidence or explicitly marked unresolved/not visible.

## Important

This stage does not run tracking and does not finalize mappings.
