# v78b Corrected Video Mapping from v77c

## Purpose

v78b maps v77c layout-aware annotation windows to video files.

Trusted mappings:
- friendly TLC videos with camera + pen + time match,
- encoded videos only when a manual camera-code mapping file exists.

Unresolved encoded mappings are not forced.

## Main counts

- Input annotation windows: 15733
- Mapped annotation windows: 905
- Unresolved annotation windows: 14828
- Mapped ratio: 0.057522
- Unique mapped videos: 12
- Behaviour classes in mapped rows: 11
- Colour identities in mapped rows: 6
- Camera-code mapping template rows: 13
- Manual camera map loaded: True
- Hard issues: 0

## Next step

If unresolved rows remain, fill:
`Full_Unibo_Behaviour_Pipeline/config/camera_code_mapping.csv`

using the template:
`v78b_camera_code_mapping_template.csv`

Then rerun v78b.
