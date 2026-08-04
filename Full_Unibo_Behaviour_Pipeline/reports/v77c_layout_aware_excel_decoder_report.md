# v77c Layout-Aware Excel Decoder

## Purpose

v77c replaces the earlier column-nearest decoding logic with a sequential layout-aware decoder.

For each pig colour row, it reads the behaviour sequence and maps sequence index to:
- hourly slot,
- observation period offset,
- 10-second window.

## Main counts

- Decoded annotation windows: 15733
- Canonical sheets processed: 43
- Behaviour classes: 11
- Colour identities: 6
- Generic red rows: 0
- Hard issues: 0

## Boundary

This is not video-mapped GT yet.
v78b/v78c should map these corrected windows to videos.
