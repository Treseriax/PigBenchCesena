from pathlib import Path
import csv
import pandas as pd


ROOT = Path.home() / "PigBench"
W6 = ROOT / "Week6_Unibo_Dataset_Validation"

OUT_STATS = W6 / "outputs/dataset_statistics"
OUT_FEAT = W6 / "outputs/feature_extractors"
OUT_GT = W6 / "outputs/unified_ground_truth"
NOTES = W6 / "notes"

OUT_STATS.mkdir(parents=True, exist_ok=True)
OUT_FEAT.mkdir(parents=True, exist_ok=True)
NOTES.mkdir(parents=True, exist_ok=True)


def safe_to_csv(df, path):
    df.to_csv(path, index=False, quoting=csv.QUOTE_ALL, escapechar="\\", lineterminator="\n")


def exists(path):
    return path.exists()


files = {
    "gt": OUT_GT / "week6_unified_ground_truth_v2_with_recommended_split_v2.csv",
    "frames": OUT_GT / "week6_scanpoint_frame_index.csv",
    "detections": OUT_FEAT / "week6_yolov8s_all_scanpoint_detections_with_qc_flags.csv",
    "conservative_050": OUT_FEAT / "week6_yolov8s_conservative_detections_score_ge_0_50.csv",
    "markers": OUT_FEAT / "week6_crop_colour_marker_features.csv",
    "candidates": OUT_FEAT / "week6_candidate_bbox_to_colour_assignments.csv",
}

gt = pd.read_csv(files["gt"])
frames = pd.read_csv(files["frames"])
detections = pd.read_csv(files["detections"])
conservative_050 = pd.read_csv(files["conservative_050"])
markers = pd.read_csv(files["markers"])
candidates = pd.read_csv(files["candidates"])

comparison = pd.DataFrame([
    {
        "extractor_family": "Manual Excel scan-sampling labels",
        "implemented": "yes",
        "main_output": str(files["gt"]),
        "record_count": len(gt),
        "feature_examples": "behaviour_code, behaviour_label, colour_id, timestamp, recommended_split_v2",
        "strength": "Reliable manual behaviour labels at scan-sampling timestamps",
        "limitation": "Not dense frame-level labels; no manual bbox",
        "recommended_use": "Primary label source"
    },
    {
        "extractor_family": "Scanpoint frame extraction",
        "implemented": "yes",
        "main_output": str(files["frames"]),
        "record_count": len(frames),
        "feature_examples": "scan_frame_id, frame_index, timestamp_sec_in_video, frame_image_path",
        "strength": "Connects labels to actual video frames",
        "limitation": "Only 72 scanpoint frames, not full video",
        "recommended_use": "Visual QC and detector input"
    },
    {
        "extractor_family": "YOLOv8-s bbox detector",
        "implemented": "yes",
        "main_output": str(files["detections"]),
        "record_count": len(detections),
        "feature_examples": "x1, y1, x2, y2, bbox_area, score, bbox_count_issue_type",
        "strength": "Provides pig bounding boxes for all 72 frames",
        "limitation": "Automatic detector output; not manual GT; bbox-to-colour ID unresolved",
        "recommended_use": "BBox geometry, crop extraction, visual overlays"
    },
    {
        "extractor_family": "Conservative bbox subset",
        "implemented": "yes",
        "main_output": str(files["conservative_050"]),
        "record_count": len(conservative_050),
        "feature_examples": "score >= 0.50 detections",
        "strength": "Cleaner higher-confidence boxes while still covering 72 frames",
        "limitation": "May miss occluded pigs",
        "recommended_use": "Cleaner crop/embedding features"
    },
    {
        "extractor_family": "Crop colour-marker features",
        "implemented": "yes",
        "main_output": str(files["markers"]),
        "record_count": len(markers),
        "feature_examples": "green_score, blue_score, purple_score, red_score, best_marker_colour, marker_confidence",
        "strength": "Candidate bbox-to-colour association",
        "limitation": "Red is ambiguous; no_color cannot be marker-detected",
        "recommended_use": "Candidate identity/colour QC, not final identity GT"
    },
    {
        "extractor_family": "Candidate bbox-to-colour assignment",
        "implemented": "yes",
        "main_output": str(files["candidates"]),
        "record_count": len(candidates),
        "feature_examples": "candidate_colour_id, assignment_status, marker_confidence",
        "strength": "Links high/medium marker evidence to candidate colour IDs",
        "limitation": "Needs visual/manual confirmation",
        "recommended_use": "Candidate association table"
    },
    {
        "extractor_family": "Segmentation features",
        "implemented": "not yet",
        "main_output": "",
        "record_count": "",
        "feature_examples": "mask area, body contour, posture shape",
        "strength": "Could improve body-shape and posture representation",
        "limitation": "Requires segmentation model or manual masks",
        "recommended_use": "Future extension"
    },
    {
        "extractor_family": "Embedding features",
        "implemented": "not yet",
        "main_output": "",
        "record_count": "",
        "feature_examples": "CNN/ViT crop embeddings, temporal embeddings",
        "strength": "Useful for learned representation comparison",
        "limitation": "Less interpretable; needs controlled validation",
        "recommended_use": "Optional future model input"
    },
])

comparison_path = OUT_FEAT / "week6_feature_extractor_comparison.csv"
safe_to_csv(comparison, comparison_path)

recommendation = pd.DataFrame([
    {"rank": 1, "feature_set": "Manual labels + scanpoint frames", "status": "core", "reason": "Ground truth alignment foundation"},
    {"rank": 2, "feature_set": "YOLOv8 bbox geometry + QC flags", "status": "core", "reason": "Interpretable object-level features"},
    {"rank": 3, "feature_set": "Crop colour-marker features", "status": "candidate", "reason": "Useful for colour ID association but not final GT"},
    {"rank": 4, "feature_set": "Conservative bbox subset score>=0.50", "status": "optional", "reason": "Cleaner features when precision matters"},
    {"rank": 5, "feature_set": "Segmentation/embedding features", "status": "future", "reason": "Promising but outside current validated implementation"},
])

recommendation_path = OUT_FEAT / "week6_feature_extractor_recommendations.csv"
safe_to_csv(recommendation, recommendation_path)

note_path = NOTES / "week6_feature_extractor_comparison_notes.md"

with open(note_path, "w") as f:
    f.write("# Week 6 Feature Extractor Comparison\n\n")
    f.write("## Purpose\n\n")
    f.write("This note compares implemented and future feature extractor families after the validated Week 6 pipeline.\n\n")

    f.write("## Feature extractor comparison\n\n")
    f.write(comparison.to_markdown(index=False))
    f.write("\n\n")

    f.write("## Recommendation ranking\n\n")
    f.write(recommendation.to_markdown(index=False))
    f.write("\n\n")

    f.write("## Interpretation\n\n")
    f.write(
        "The strongest validated feature pipeline is manual scan-sampling labels linked to scanpoint frames, "
        "YOLOv8-s bbox geometry with QC flags, and crop colour-marker candidate features. "
        "Segmentation and embedding features remain future extensions unless additional implementation time is available.\n"
    )

print("Saved:")
print(comparison_path)
print(recommendation_path)
print(note_path)

print("\n=== Feature extractor comparison ===")
print(comparison.to_string(index=False))

print("\n=== Recommendations ===")
print(recommendation.to_string(index=False))
