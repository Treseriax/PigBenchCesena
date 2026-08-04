# v78k4 Annotation Source-of-Truth Rebuild

## Purpose

This stage treats Excel as the annotation source of truth.

It answers:

- Which sheet is the annotation unit?
- Which TLC / Camera / Room / Pen does each sheet describe?
- How should `big` and `small` be interpreted?
- Which parts are annotation truth and which parts are video/spatial mapping?

## Key rule

Do not look for B1/B3 visually in the frame during annotation decoding.

The sheet itself defines the annotation target.

Example:

- `TLC 1 big` resolves to `TLC1 / B1` using row-1 metadata.
- `TLC 1 B3` resolves to `TLC1 / B3`.
- `TLC 2 B4` resolves to `TLC2 / B4`.

## Boundary

This stage does not solve encoded c-code mapping and does not solve frame-level ROI.
