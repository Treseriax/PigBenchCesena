from pathlib import Path
import pandas as pd


ROOT = Path("Week4_5_Behaviour_Representation")
OUT = ROOT / "outputs/representation_tables"
NOTES = ROOT / "notes"

OUT.mkdir(parents=True, exist_ok=True)
NOTES.mkdir(parents=True, exist_ok=True)

rows = [
    {
        "representation_family": "Trajectory-based",
        "input_data": "Detection/tracking outputs: track_id, frame, bbox, centroid",
        "output_representation": "Motion descriptors",
        "example_features": "speed, acceleration, displacement, trajectory length, turning angle, stationary ratio, fast-motion ratio",
        "advantages": "Interpretable; already implementable from YOLOv8-s + ByteTrack outputs; transferable if normalized",
        "limitations": "Sensitive to ID switches, fragmented tracks, camera perspective, and bbox jitter",
        "our_current_status": "Implemented in Week 4; frame-level, track-level, segment-level, and group-level features generated",
        "best_use_case": "General activity level, exploration, inactivity/resting proxy, movement pattern comparison",
        "priority_for_next_phase": "High"
    },
    {
        "representation_family": "ROI / resource-based",
        "input_data": "Tracking outputs plus manually defined pen/resource zones",
        "output_representation": "Spatial occupation descriptors",
        "example_features": "time spent in ROI, zone instance ratio, zone track-seconds, distance to ROI center, dominant spatial zone",
        "advantages": "Highly interpretable; connected to ethological resource-use concepts such as feeder/drinker occupation or pen-area use",
        "limitations": "Requires accurate ROI definition; centroid-only approximation cannot prove actual feeding/drinking without snout/head information",
        "our_current_status": "Implemented as first centroid/bbox-based prototype with ROI overlays and occupancy summaries",
        "best_use_case": "Feeding/drinking candidate zones, resting/contact zones, pen-use patterns",
        "priority_for_next_phase": "High"
    },
    {
        "representation_family": "Social / group-spatial",
        "input_data": "Multiple tracked pigs per frame",
        "output_representation": "Group interaction and proximity descriptors",
        "example_features": "nearest-neighbour distance, local density, group spread, visible tracks per frame",
        "advantages": "Useful for interaction, clustering, aggression/contact, and group-level welfare indicators",
        "limitations": "Needs reliable multi-animal tracking; close contact and occlusion can create ID switches",
        "our_current_status": "Partially implemented through group frame features and group segment summary",
        "best_use_case": "Interaction, clustering, crowding, possible aggression/contact indicators",
        "priority_for_next_phase": "High"
    },
    {
        "representation_family": "Shape-based",
        "input_data": "Silhouettes, masks, contours, or segmented pig blobs",
        "output_representation": "Geometric descriptors",
        "example_features": "area, aspect ratio, contour compactness, orientation, elongation, posture-like shape indicators",
        "advantages": "More posture-aware than centroid trajectories; can help distinguish standing/lying-like shapes if segmentation is reliable",
        "limitations": "Requires masks/segmentation; difficult in occlusion and overlapping pigs; can be sensitive to lighting/camera changes",
        "our_current_status": "Not implemented yet; feasible if masks or segmentation outputs are added",
        "best_use_case": "Standing vs lying proxy, posture changes, body orientation",
        "priority_for_next_phase": "Medium"
    },
    {
        "representation_family": "Skeleton-based",
        "input_data": "Animal keypoints from pose estimation frameworks",
        "output_representation": "Joint configurations and temporal pose descriptors",
        "example_features": "body posture, head orientation, limb positions, joint angles, temporal pose changes",
        "advantages": "More behaviour-specific than bbox trajectory; useful for posture, head direction, feeding/drinking, lying/standing analysis",
        "limitations": "Requires keypoint annotation or suitable pretrained model; multi-pig occlusion is challenging; pig-specific pose labels may be needed",
        "our_current_status": "Survey/review stage only; DeepLabCut, SLEAP, Animal-Pose/AP-10K need feasibility review",
        "best_use_case": "Posture recognition, head direction, feeding/drinking refinement, fine-grained movement",
        "priority_for_next_phase": "Medium-High"
    },
    {
        "representation_family": "Mesh-based",
        "input_data": "3D reconstruction, SMAL parameters, or mesh estimation from images",
        "output_representation": "3D shape + posture representation",
        "example_features": "body pose, body shape, back curvature, orientation, surface deformation, posture sequence descriptors",
        "advantages": "Richest geometric representation; potentially more transferable across viewpoints if 3D estimation is reliable",
        "limitations": "Computationally heavy; pig-specific mesh validation is uncertain; implementation not required at this stage",
        "our_current_status": "Feasibility review stage; AniMer/SMAL is the main literature reference",
        "best_use_case": "Future high-quality posture and biomechanics descriptors",
        "priority_for_next_phase": "Low-Medium"
    },
    {
        "representation_family": "Embedding-based",
        "input_data": "Images, crops, video clips, detector features, or self-supervised visual features",
        "output_representation": "Latent vectors",
        "example_features": "YOLO/RT-DETR/Co-DINO feature vectors, DINOv2 embeddings, CLIP/SigLIP embeddings, VideoMAE/Video Swin clip embeddings",
        "advantages": "Can capture appearance, posture, context, and subtle visual cues not hand-designed manually",
        "limitations": "Less interpretable; domain shift risk; requires careful validation and possibly labelled data",
        "our_current_status": "Survey stage; implementation optional",
        "best_use_case": "Future behaviour classification, crop/clip retrieval, semi-supervised representation learning",
        "priority_for_next_phase": "Medium"
    },
    {
        "representation_family": "End-to-end video / temporal model",
        "input_data": "Short video clips or frame sequences",
        "output_representation": "Predicted behaviour class or learned temporal features",
        "example_features": "TSM/VideoMAE/Video Swin temporal features, aggressive vs non-aggressive clip prediction",
        "advantages": "Strong for behaviours requiring temporal context, such as aggression or rapid social interaction",
        "limitations": "Less interpretable; needs labelled clips; may fail under occlusion or ambiguous overlapping behaviour",
        "our_current_status": "Literature review only; TSM aggressive behaviour paper is the main reference",
        "best_use_case": "Aggression, tail biting, short interaction events, clip-level behaviour recognition",
        "priority_for_next_phase": "Medium"
    },
]

survey = pd.DataFrame(rows)

survey_csv = OUT / "behaviour_representation_survey.csv"
survey.to_csv(survey_csv, index=False)

recommendations = pd.DataFrame([
    {
        "rank": 1,
        "recommended_family": "Trajectory + group-spatial features",
        "reason": "Already available from tracking outputs; interpretable; directly implemented and visualized",
        "next_action": "Add smoothing/outlier filtering and compare features across behaviour-labelled scan windows"
    },
    {
        "rank": 2,
        "recommended_family": "ROI / resource-based features",
        "reason": "Strong interpretability and directly connected to ethological resource-use concepts",
        "next_action": "Refine ROI definitions and later add head/snout/keypoint information if possible"
    },
    {
        "rank": 3,
        "recommended_family": "Skeleton/keypoint features",
        "reason": "Best next step for posture and head-orientation features, especially feeding/drinking and lying/standing",
        "next_action": "Review DeepLabCut, SLEAP, Animal-Pose/AP-10K and estimate annotation effort"
    },
    {
        "rank": 4,
        "recommended_family": "Embedding-based crop/clip features",
        "reason": "Useful for future classification and transfer learning, but less interpretable",
        "next_action": "Start with detector crop embeddings or DINOv2-style image embeddings after feature survey"
    },
    {
        "rank": 5,
        "recommended_family": "Mesh-based features",
        "reason": "Powerful but heavy; currently better as feasibility discussion than implementation",
        "next_action": "Summarize AniMer/SMAL and discuss pig/livestock applicability"
    },
])

recommendation_csv = OUT / "representation_recommendation_ranking.csv"
recommendations.to_csv(recommendation_csv, index=False)

note_path = NOTES / "behaviour_representation_survey.md"

with open(note_path, "w") as f:
    f.write("# Behaviour Representation Survey\n\n")

    f.write("## Purpose\n\n")
    f.write(
        "The goal of this survey is to compare behaviour representation families before training "
        "a final classifier. The focus is on what information can be extracted from videos, tracking outputs, "
        "ROIs, pose, mesh, and embeddings to represent pig behaviour in an interpretable and transferable way.\n\n"
    )

    f.write("## Representation families\n\n")
    f.write(survey.to_markdown(index=False))
    f.write("\n\n")

    f.write("## Recommendation ranking for next phase\n\n")
    f.write(recommendations.to_markdown(index=False))
    f.write("\n\n")

    f.write("## Interpretation\n\n")
    f.write(
        "For the immediate next phase, trajectory, ROI, and social/group-spatial features are the strongest candidates "
        "because they can be extracted from the current Unibo tracking outputs. Skeleton and embedding representations "
        "are promising next steps but require either keypoint estimation or additional feature extraction pipelines. "
        "Mesh-based representations are scientifically interesting and may provide richer posture information, but they "
        "are computationally heavier and should remain a feasibility review at this stage.\n\n"
    )

    f.write("## Connection to completed work\n\n")
    f.write(
        "- Trajectory features were already extracted from the corrected Unibo scan-window tracking outputs.\n"
        "- ROI/resource-style features were implemented as a first centroid/bbox-based prototype.\n"
        "- Edinburgh annotation inspection showed a clean public reference format with persistent objects, bounding boxes, and behaviour labels.\n"
        "- Unibo remains the primary project dataset for representation feature engineering.\n"
    )

print("Saved:")
print(survey_csv)
print(recommendation_csv)
print(note_path)

print()
print("=== Recommendation ranking ===")
print(recommendations.to_string(index=False))
