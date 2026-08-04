# Week 8 Annotation Rules v44

## 1. Behaviour label propagation

The behaviour observation corresponds to a 10-second annotated observation window.

Therefore, the pig-level behaviour label from the annotation is propagated to every frame inside the corresponding annotated clip interval.

## 2. Identity rule

Pig identity is represented through the colour-marker identity and its behaviour-annotation pig ID crosswalk.

Valid visual marker colours:

blue, green, cyan, red, pink, purple

Verified crosswalk:

blue -> blue
green -> green
cyan -> no_color
red -> red_neck
pink -> red_tail
purple -> purple

## 3. Unknown / uncertain identities

Unknown, not-visible, uncertain, or unassigned identities must not be silently forced into a valid identity.

They must be preserved as validation states or review-required cases.

## 4. Bounding boxes

Bounding boxes use [x1, y1, x2, y2] pixel coordinates.

The scanpoint-level corrected boxes are the geometry anchor. Frame-level propagation may later use tracking boxes where reliable tracking evidence is available.

## 5. Validation flags

Supported issue flags include:

colour_error
behaviour_error
identity_switch
missing_label
wrong_bbox
occlusion
uncertain
temporal_inconsistency
review_required

## 6. Dataset claim scope

The Week 8 dataset is a validated and inspectable ground-truth preparation dataset.

It is not a final behaviour classifier and not a production-grade multi-object tracking system.
