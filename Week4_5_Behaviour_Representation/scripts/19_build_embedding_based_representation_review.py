from pathlib import Path
import pandas as pd


ROOT = Path("Week4_5_Behaviour_Representation")
OUT = ROOT / "outputs/representation_tables"
NOTES = ROOT / "notes"

OUT.mkdir(parents=True, exist_ok=True)
NOTES.mkdir(parents=True, exist_ok=True)

embedding_rows = [
    {
        "embedding_family": "Detector backbone features",
        "examples": "YOLOv8, RT-DETR, Co-DETR / Co-DINO style detector features",
        "input_unit": "Full frame, detected pig crop, or detector region feature",
        "output_representation": "Feature vector from object detector backbone/neck/head",
        "what_it_represents": "Appearance, local posture cues, object context, detector-level visual patterns",
        "advantages": "Already close to our detection pipeline; can be linked to track IDs and frame-level detections",
        "limitations": "Detector features are not necessarily behaviour-specific; extracting them may require modifying inference code",
        "best_use_case": "Track-level appearance descriptors, crop retrieval, future behaviour classifier input",
        "our_project_status": "Survey stage; possible future extension after trajectory/ROI features",
        "priority": "Medium"
    },
    {
        "embedding_family": "Self-supervised image embeddings",
        "examples": "DINOv2-style visual embeddings",
        "input_unit": "Pig crop or full frame",
        "output_representation": "General visual feature vector learned without behaviour labels",
        "what_it_represents": "Appearance, shape, posture-like visual patterns, scene context",
        "advantages": "Useful when labelled behaviour data is limited; can support clustering and similarity search",
        "limitations": "Less interpretable than hand-crafted features; domain shift from generic images to overhead pig footage is possible",
        "best_use_case": "Unsupervised grouping of pig crops, finding visually similar behaviour/posture examples",
        "our_project_status": "Survey stage; implementation optional",
        "priority": "Medium"
    },
    {
        "embedding_family": "Vision-language embeddings",
        "examples": "CLIP, SigLIP",
        "input_unit": "Image crop or full frame",
        "output_representation": "Image-language aligned latent vector",
        "what_it_represents": "Semantic visual similarity and broad image concepts",
        "advantages": "Can support image retrieval and weak semantic grouping without training from scratch",
        "limitations": "May not understand fine-grained livestock behaviours; overhead farm footage may not match web-scale training data",
        "best_use_case": "Exploratory retrieval, visual clustering, qualitative dataset browsing",
        "our_project_status": "Low-priority survey item",
        "priority": "Low-Medium"
    },
    {
        "embedding_family": "Video clip embeddings",
        "examples": "VideoMAE, Video Swin, TimeSformer-like models",
        "input_unit": "Short video clip or tracklet crop sequence",
        "output_representation": "Temporal latent vector",
        "what_it_represents": "Motion, posture change, temporal context, interaction dynamics",
        "advantages": "Better than still-image embeddings for behaviours that unfold over time",
        "limitations": "Needs clip extraction pipeline; less interpretable; may need labelled data or fine-tuning",
        "best_use_case": "Aggression, interaction, active movement, lying/standing transitions",
        "our_project_status": "Survey stage; not immediate implementation",
        "priority": "Medium"
    },
    {
        "embedding_family": "End-to-end temporal classification features",
        "examples": "Temporal Shift Module models, clip classifiers",
        "input_unit": "Fixed-length labelled video clip",
        "output_representation": "Behaviour class prediction or intermediate temporal feature vector",
        "what_it_represents": "Task-specific temporal behaviour pattern",
        "advantages": "Strong for clip-level behaviours such as aggression vs non-aggression",
        "limitations": "Requires labelled clips; can be less interpretable than trajectory/ROI/skeleton features",
        "best_use_case": "Future aggressive behaviour recognition or short social-event classification",
        "our_project_status": "Literature review only; useful as future comparison baseline",
        "priority": "Medium"
    },
    {
        "embedding_family": "Tracklet-level embeddings",
        "examples": "Crop sequence embeddings, ReID-style embeddings, appearance descriptors",
        "input_unit": "Cropped sequence for one track ID",
        "output_representation": "Per-track or per-tracklet identity/appearance vector",
        "what_it_represents": "Visual identity, colour marker, body appearance, temporal consistency",
        "advantages": "Could help identity association and reduce track fragmentation",
        "limitations": "Identity labels are not fully verified in current Unibo data; colour markers and occlusion complicate learning",
        "best_use_case": "Future ReID, identity verification, track merging",
        "our_project_status": "Potential future direction after conservative identity mapping",
        "priority": "Medium"
    }
]

embeddings = pd.DataFrame(embedding_rows)
embedding_csv = OUT / "embedding_based_representation_review.csv"
embeddings.to_csv(embedding_csv, index=False)

feature_rows = [
    {
        "feature_level": "Frame-level crop embedding",
        "input": "Single pig crop from one frame",
        "aggregation": "None or average over nearby frames",
        "possible_use": "Visual posture/appearance clustering"
    },
    {
        "feature_level": "Track-level embedding",
        "input": "All crops belonging to one track ID",
        "aggregation": "Mean, median, or temporal pooling",
        "possible_use": "Track identity consistency, ReID, appearance comparison"
    },
    {
        "feature_level": "Segment-level embedding",
        "input": "All frame or track embeddings within one scan-window segment",
        "aggregation": "Mean pooling, histogram, cluster distribution",
        "possible_use": "Segment-level behaviour representation"
    },
    {
        "feature_level": "Clip-level video embedding",
        "input": "Short video clip or fixed-length frame sequence",
        "aggregation": "Temporal model output",
        "possible_use": "Aggression, interaction, or movement classification"
    },
    {
        "feature_level": "Hybrid representation",
        "input": "Trajectory + ROI + visual embedding",
        "aggregation": "Concatenate normalized feature groups",
        "possible_use": "Future interpretable + visual behaviour classifier"
    }
]

features = pd.DataFrame(feature_rows)
features_csv = OUT / "embedding_feature_levels.csv"
features.to_csv(features_csv, index=False)

risk_rows = [
    {
        "risk": "Low interpretability",
        "explanation": "Embedding dimensions are not directly meaningful like speed or ROI time.",
        "mitigation": "Use embeddings together with interpretable trajectory and ROI features."
    },
    {
        "risk": "Domain shift",
        "explanation": "Models pretrained on generic images/videos may not fit overhead pig-pen footage.",
        "mitigation": "Validate on Unibo and Edinburgh samples; use clustering and manual inspection before relying on labels."
    },
    {
        "risk": "Label scarcity",
        "explanation": "Good supervised embeddings require reliable behaviour labels.",
        "mitigation": "Start with self-supervised or pretrained embeddings; fine-tune only after stronger labels exist."
    },
    {
        "risk": "Track fragmentation",
        "explanation": "Embedding aggregation over wrong track IDs can mix identities.",
        "mitigation": "Use conservative identity handling and track quality filtering."
    },
    {
        "risk": "Computational cost",
        "explanation": "Video embeddings and large models can be GPU-heavy.",
        "mitigation": "Start with small samples and crop-level embeddings before full video processing."
    }
]

risks = pd.DataFrame(risk_rows)
risks_csv = OUT / "embedding_risks_and_mitigation.csv"
risks.to_csv(risks_csv, index=False)

recommendation_rows = [
    {
        "rank": 1,
        "recommendation": "Do not start with embeddings as the primary representation",
        "reason": "Trajectory, ROI, and social-spatial features are already implemented and more interpretable."
    },
    {
        "rank": 2,
        "recommendation": "Use embeddings as a complementary feature layer",
        "reason": "Visual embeddings can capture posture/appearance patterns that hand-crafted motion features miss."
    },
    {
        "rank": 3,
        "recommendation": "Start with crop-level or track-level embeddings before full video models",
        "reason": "Crop/track embeddings are easier to integrate with existing detection and tracking outputs."
    },
    {
        "rank": 4,
        "recommendation": "Use video embeddings later for temporal behaviours",
        "reason": "Clip models are more appropriate for aggression, interaction, and rapid behaviour changes."
    }
]

recommendations = pd.DataFrame(recommendation_rows)
recommendations_csv = OUT / "embedding_recommendations.csv"
recommendations.to_csv(recommendations_csv, index=False)

note_path = NOTES / "embedding_based_representation_review.md"

with open(note_path, "w") as f:
    f.write("# Embedding-Based Representation Review\n\n")

    f.write("## Purpose\n\n")
    f.write(
        "This review evaluates whether visual embeddings can represent pig behaviour beyond hand-crafted "
        "trajectory, ROI, social-spatial, skeleton, and mesh features. Embeddings are not implemented at this stage; "
        "they are evaluated as a future feature layer.\n\n"
    )

    f.write("## Embedding families\n\n")
    f.write(embeddings.to_markdown(index=False))
    f.write("\n\n")

    f.write("## Feature levels\n\n")
    f.write(features.to_markdown(index=False))
    f.write("\n\n")

    f.write("## Risks and mitigation\n\n")
    f.write(risks.to_markdown(index=False))
    f.write("\n\n")

    f.write("## Recommendations\n\n")
    f.write(recommendations.to_markdown(index=False))
    f.write("\n\n")

    f.write("## Interpretation\n\n")
    f.write(
        "Embedding-based representations are promising because they can capture appearance, posture, context, and temporal cues "
        "that are difficult to hand-design. However, they are less interpretable and may suffer from domain shift. "
        "For this project, embeddings should be treated as a complementary future layer. The immediate priority remains "
        "trajectory, ROI/resource, and social-spatial features, followed by skeleton/keypoint feasibility testing.\n"
    )

print("Saved:")
print(embedding_csv)
print(features_csv)
print(risks_csv)
print(recommendations_csv)
print(note_path)

print()
print("=== Embedding recommendations ===")
print(recommendations.to_string(index=False))
