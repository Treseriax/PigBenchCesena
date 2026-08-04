# v78k0 Excel Camera/Pen Pattern Resolver

## Purpose

This stage extracts the Camera / Room / Pen pattern directly from the Excel annotation sheets.

It resolves which TLC/camera is associated with which pen labels, without relying on visual guessing.

## Canonical pattern expected

- TLC1 -> B1 / B3
- TLC2 -> B6 / B4
- TLC3 -> C1 / C3
- TLC4 -> C6 / C4
- TLC5 -> M1 / M2
- TLC6 -> M4 / M3

## Important boundary

This stage resolves camera/pen association from Excel, but it does **not** provide frame-level ROI coordinates.

If a video frame contains multiple pens, pen-specific spatial filtering still requires ROI or layout evidence.
