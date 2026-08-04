# Week 8 Final Modeling Package

## Final modeling decision

The final Week8 modeling champion is v72b: frozen YOLOv8 detector embedding baseline.

## Champion results

Current split:
- crop: tight
- config: knn3_cosine
- test accuracy: 0.3718
- test macro-F1: 0.1916
- delta macro-F1 vs v69b: 0.0539

Group-aware split:
- crop: context10
- config: ridge_pca128_lam100.0
- test accuracy: 0.2639
- test macro-F1: 0.1503
- delta macro-F1 vs v69b: 0.0237

## Claim boundary

This is a frozen embedding baseline, not a production behaviour classifier.
No GT is modified by model outputs.
Manual GT v2 remains the source of truth.

## Error inspection

v72c created a visual interface for manual error inspection.
