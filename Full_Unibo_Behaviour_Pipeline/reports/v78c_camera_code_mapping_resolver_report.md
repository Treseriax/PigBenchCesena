# v78c Camera-Code Mapping Resolver

## Purpose

v78c resolves the bottleneck discovered in v78b: encoded video filenames use camera codes, while Excel sheets use TLC/camera/pen names.

This stage creates:
- unresolved mapping groups,
- clean camera-code mapping template,
- encoded video camera-code inventory,
- visual contact sheets for manual review.

## Main counts

- Unresolved annotation rows from v78b: 15296
- Unresolved groups: 42
- Template rows: 42
- Encoded camera-code count: 6
- Visual thumbnails created: 71
- Hard issues: 0

## Manual next step

Inspect:

`/home/oyavuz/PigBench/Full_Unibo_Behaviour_Pipeline/outputs/v78c_camera_code_mapping_resolver/Full_Unibo_Camera_Code_Mapping_Resolver/v78c_visual_atlas_index.html`

Fill:

`/home/oyavuz/PigBench/Full_Unibo_Behaviour_Pipeline/config/camera_code_mapping_TEMPLATE_TO_FILL.csv`

Then save/copy the filled file as:

`Full_Unibo_Behaviour_Pipeline/config/camera_code_mapping.csv`

and rerun v78b.
