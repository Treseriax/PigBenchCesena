# Week8 Final Delivery Package

## Scope

This package is the final Week8 delivery index and summary package.

It does not duplicate raw Unibo videos or large intermediate image folders.
Large generated artifacts are referenced by path and SHA256 in `week8_v74_zip_index.csv`.

## Final GT status

- Reviewed manual GT objects: 432
- Strict-gold objects for classification: 372
- Caution objects: 3
- Nonusable / excluded objects: 57
- Scanframes: 72

## Final modeling status

Final champion: frozen_yolov8_detector_embedding_baseline

Current split:
- crop/config: tight / knn3_cosine
- accuracy: 0.3717948717948718
- macro-F1: 0.1916040100250627
- delta macro-F1 vs v69b: 0.0539166097060835

Group-aware split:
- crop/config: context10 / ridge_pca128_lam100.0
- accuracy: 0.2638888888888889
- macro-F1: 0.1503464594373685
- delta macro-F1 vs v69b: 0.0237397328306419

## Claim boundary

This is a validated GT + baseline modeling delivery.
It is not a production behaviour classifier.
Manual GT v2 remains the source of truth.
