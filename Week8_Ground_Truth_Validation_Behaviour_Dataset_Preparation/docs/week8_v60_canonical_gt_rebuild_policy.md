# Week 8 v60 Canonical GT Rebuild Policy

## Decision

The previous task-sheet-oriented pipeline is frozen. It remains useful as a diagnostic and visualization pipeline, but it is not accepted as final ground truth.

## Reason

Manual validation notes revealed substantial issues:

- missing pigs
- wrong bounding boxes
- identity switches
- possible wrong or missing GT anchors
- conflicting manual notes for some scanframes

Therefore, classification is paused until a canonical GT v2 is created.

## New source-of-truth rules

1. Original annotation file is the canonical source for pig colour / identity labels.
2. Visual colour guesses are not canonical.
3. Previous suggested colour mappings are deprecated until verified against the annotation file.
4. Tracking is a helper layer only.
5. Classification can only use clips marked GOLD or explicitly accepted SILVER in GT v2.
6. RED / FIX_REQUIRED clips are excluded from classification until corrected.

## GT v2 clip status

- GOLD: reliable bbox, identity, colour and behaviour.
- SILVER: usable with minor limitations.
- RED: not usable for classification.
- FIX_REQUIRED: potentially usable after manual correction.
- UNKNOWN: not yet reviewed.

## Next step

Run v61 annotation source inventory and identify the original annotation file containing pig IDs, colour labels and behaviour labels.
