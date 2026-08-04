# v78h Deep Camera Mapping Forensics

## Purpose

This stage exhausts non-permission-based evidence before asking for access:
1. MP4 metadata / ffprobe scan
2. MP4 header string scan
3. Deep Excel scan: sheet titles, hidden sheets, comments, hyperlinks, formulas, cell values
4. Readable small-file keyword scan
5. Visual camera board for human inspection

## Counts

- Video rows: 84
- Encoded videos: 76
- Friendly reference videos: 8
- Encoded camera codes: 6
- MP4 metadata hits: 1028
- MP4 header string hits: 36771
- Excel deep hits: 4136
- Readable file hits: 552284
- Visual frames created: 37
- Hard issues: 0

## Visual board

Open:

`/home/oyavuz/PigBench/Full_Unibo_Behaviour_Pipeline/outputs/v78h_deep_camera_mapping_forensics/Full_Unibo_Deep_Camera_Mapping_Forensics/v78h_visual_camera_board.html`

Look for:
- camera overlay text
- wall labels
- pen labels
- physical signs
- geometry that connects to TLC naming
- whether c0002 remains visually consistent with TLC1 reference

This stage does not apply mappings.
