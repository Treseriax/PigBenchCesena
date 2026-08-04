# Week 8 v63a Canonical GT v2 Schema

## Purpose

This schema replaces the previous task-sheet-oriented GT assumption with a canonical manual GT v2 workflow.

## Source of truth hierarchy

1. Canonical colour and behaviour labels: original Excel annotation extracted in v62b2.
2. Candidate bounding boxes: existing v45 anchor boxes, not final GT.
3. Candidate tracking: helper layer only, not final GT.
4. Manual GT v2 assignment: final source for bbox-to-colour identity association.

## Manual GT v2 status values

Recommended values:

- gold_usable
- silver_usable_with_caution
- red_exclude
- fix_required
- unknown_pending_review

## Manual bbox status values

Recommended values:

- bbox_ok
- bbox_wrong
- bbox_missing
- bbox_extra_false_positive
- bbox_needs_manual_redraw
- bbox_uncertain

## Manual identity status values

Recommended values:

- identity_confirmed
- identity_uncertain
- identity_switch
- identity_not_visible
- identity_missing

## Classification use values

Recommended values:

- use_for_classification
- use_for_classification_with_caution
- exclude_from_classification
- pending_review
