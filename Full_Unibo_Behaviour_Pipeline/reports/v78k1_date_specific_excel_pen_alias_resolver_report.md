# v78k1 Date-Specific Excel Pen Alias Resolver

This stage fixes v78k0 by resolving `big` / `small` sheet aliases using the actual Excel row-1 Camera / Room / Pen metadata.

Important:
- `big` and `small` are not literal pen names.
- They are sheet aliases and must be resolved per date and TLC camera.
- This stage resolves camera/pen association, not frame-level spatial ROI.
