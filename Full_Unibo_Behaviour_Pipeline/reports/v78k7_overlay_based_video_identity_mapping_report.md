# v78k7 Overlay-Based Video Identity Mapping

This stage uses the visible top-left video overlay as evidence for video identity.

Evidence hierarchy:
1. Overlay text such as `TLC 2 B6`
2. Filename c-code/date/time encoding
3. Manual visual confirmation from overlay crop board

This stage supersedes attempts to infer TLC-to-c-code mapping without reading the video overlay.
