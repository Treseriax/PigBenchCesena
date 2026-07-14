from pathlib import Path
import pandas as pd


ROOT = Path("Week4_5_Behaviour_Representation")
OUT = ROOT / "outputs/representation_tables"
NOTES = ROOT / "notes"

OUT.mkdir(parents=True, exist_ok=True)
NOTES.mkdir(parents=True, exist_ok=True)

framework_rows = [
    {
        "framework_or_dataset": "DeepLabCut",
        "type": "Pose estimation framework",
        "main_idea": "User-defined keypoints are manually labelled on selected frames, then a model is trained to predict those keypoints on new frames.",
        "required_annotations": "Manual keypoint labels for pig body parts such as snout, ears, shoulders/back, tail base, legs, or other project-defined landmarks.",
        "multi_animal_support": "Supports multi-animal workflows, but performance depends on clear keypoint definitions and enough labelled examples under occlusion.",
        "expected_outputs": "Per-frame keypoint coordinates, likelihood/confidence scores, and trajectories for each labelled body part.",
        "pig_suitability": "High if we can annotate representative pig frames from Unibo videos; useful for head direction, posture, feeding/drinking refinement.",
        "integration_with_our_pipeline": "Can be added after detection/tracking. Pig crops or full frames can be labelled, and predicted keypoints can be linked to track IDs.",
        "possible_features": "head orientation, snout-to-ROI distance, body axis, posture state, temporal pose changes",
        "main_limitations": "Requires manual annotation effort; occlusion and overlapping pigs may reduce reliability.",
        "priority": "High for future refinement"
    },
    {
        "framework_or_dataset": "SLEAP",
        "type": "Pose estimation framework",
        "main_idea": "Multi-animal pose tracking framework designed for animal pose estimation and tracking.",
        "required_annotations": "Skeleton definition plus labelled keypoints on representative frames; multi-animal instances should be labelled if using full-frame multi-pig mode.",
        "multi_animal_support": "Strong multi-animal focus; suitable for multiple animals in one frame, depending on visibility and occlusion.",
        "expected_outputs": "Keypoint coordinates, instance tracks, confidence scores, and temporal pose tracks.",
        "pig_suitability": "Promising for group-housed pigs because it is designed around animal pose and multi-animal settings.",
        "integration_with_our_pipeline": "Could either run directly on full frames or after detection. Outputs can be matched to ByteTrack IDs using bbox/keypoint overlap.",
        "possible_features": "limb positions, body orientation, head direction, lying/standing posture proxy, social contact pose patterns",
        "main_limitations": "Needs labelled pig examples; overlapping animals and low image quality may still be difficult.",
        "priority": "High for multi-pig pose feasibility"
    },
    {
        "framework_or_dataset": "Animal-Pose / AP-10K",
        "type": "Animal pose dataset / pretrained pose-estimation resource",
        "main_idea": "Provides general animal keypoint data that may support pretrained or transferable pose models.",
        "required_annotations": "For direct use, no new annotations may be needed if a pretrained model works; for pig-specific robustness, fine-tuning labels are likely needed.",
        "multi_animal_support": "Depends on the model used with the dataset; often combined with detector + pose-estimator pipelines.",
        "expected_outputs": "Standard animal keypoints, confidence scores, and possible body pose descriptors.",
        "pig_suitability": "Useful as a starting point, but pig-specific performance must be validated because farm pigs, overhead camera, and occlusion differ from generic animal images.",
        "integration_with_our_pipeline": "Could be tested on Unibo frames as a pretrained baseline, then compared with custom DeepLabCut/SLEAP if needed.",
        "possible_features": "generic body pose, body axis, rough head/tail orientation, posture descriptors",
        "main_limitations": "Domain shift risk; keypoint taxonomy may not match our desired pig-specific landmarks.",
        "priority": "Medium"
    }
]

frameworks = pd.DataFrame(framework_rows)
framework_csv = OUT / "skeleton_framework_review.csv"
frameworks.to_csv(framework_csv, index=False)

feature_rows = [
    {
        "feature_group": "Head / snout features",
        "required_keypoints": "snout, head center, ears or neck/back point",
        "example_features": "snout-to-feeder distance, snout-to-drinker distance, head orientation angle",
        "behaviour_relevance": "Important for feeding, drinking, exploring/sniffing",
        "connection_to_current_work": "Improves the current centroid-only ROI prototype"
    },
    {
        "feature_group": "Body axis / posture features",
        "required_keypoints": "head/neck, back, tail base, shoulders or hips",
        "example_features": "body axis angle, body length proxy, posture orientation, elongation",
        "behaviour_relevance": "Useful for standing, lying, sitting, and posture changes",
        "connection_to_current_work": "Adds posture information missing from bbox trajectory features"
    },
    {
        "feature_group": "Limb / movement features",
        "required_keypoints": "legs/feet if visible",
        "example_features": "limb displacement, step-like motion, temporal pose changes",
        "behaviour_relevance": "Potentially useful for walking, running, active movement",
        "connection_to_current_work": "Can complement speed and acceleration features"
    },
    {
        "feature_group": "Social pose features",
        "required_keypoints": "snout/head and body axis for multiple pigs",
        "example_features": "head-to-body distance between pigs, facing angle, contact-like proximity",
        "behaviour_relevance": "Useful for interaction, aggression, exploration, social contact",
        "connection_to_current_work": "Extends nearest-neighbour/local-density features with orientation"
    }
]

features = pd.DataFrame(feature_rows)
features_csv = OUT / "skeleton_possible_features.csv"
features.to_csv(features_csv, index=False)

recommendation_rows = [
    {
        "rank": 1,
        "recommendation": "Start with a small custom pig keypoint set",
        "reason": "A small keypoint set is easier to annotate and directly supports ROI/head-direction features.",
        "suggested_keypoints": "snout, left ear, right ear, shoulder/back center, tail base"
    },
    {
        "rank": 2,
        "recommendation": "Use SLEAP or DeepLabCut for feasibility testing",
        "reason": "Both are suitable candidates for animal pose estimation; testing on a few Unibo frames is more realistic than full deployment immediately.",
        "suggested_keypoints": "same small keypoint set"
    },
    {
        "rank": 3,
        "recommendation": "Use Animal-Pose/AP-10K as a pretrained baseline only",
        "reason": "Generic animal pose resources may suffer from domain shift on overhead pig-pen footage.",
        "suggested_keypoints": "depends on available pretrained taxonomy"
    },
    {
        "rank": 4,
        "recommendation": "Do not replace trajectory/ROI features yet",
        "reason": "Trajectory and ROI features are already implemented and interpretable; skeleton features should refine them, not replace them.",
        "suggested_keypoints": "snout and body axis keypoints are most useful"
    }
]

recommendations = pd.DataFrame(recommendation_rows)
recommendations_csv = OUT / "skeleton_framework_recommendations.csv"
recommendations.to_csv(recommendations_csv, index=False)

note_path = NOTES / "skeleton_framework_review.md"

with open(note_path, "w") as f:
    f.write("# Skeleton Framework Review\n\n")

    f.write("## Purpose\n\n")
    f.write(
        "This review evaluates whether skeleton/keypoint-based representations could be integrated "
        "into the pig behaviour analysis pipeline. The focus is not on immediate full implementation, "
        "but on required annotations, multi-animal support, expected outputs, and suitability for pigs.\n\n"
    )

    f.write("## Framework comparison\n\n")
    f.write(frameworks.to_markdown(index=False))
    f.write("\n\n")

    f.write("## Possible skeleton-derived features\n\n")
    f.write(features.to_markdown(index=False))
    f.write("\n\n")

    f.write("## Recommendations\n\n")
    f.write(recommendations.to_markdown(index=False))
    f.write("\n\n")

    f.write("## Interpretation\n\n")
    f.write(
        "Skeleton representations are most useful as a refinement layer above the current trajectory "
        "and ROI features. The current ROI prototype uses centroids and bounding boxes, which cannot "
        "confirm true feeding or drinking. A snout/head keypoint would make distance-to-resource "
        "features more behaviour-specific. However, skeleton methods require annotation effort and "
        "may struggle with occlusion in group-housed pigs. Therefore, the best next step is a small "
        "feasibility test with a limited pig-specific keypoint set rather than a full pose-estimation deployment.\n"
    )

print("Saved:")
print(framework_csv)
print(features_csv)
print(recommendations_csv)
print(note_path)

print()
print("=== Skeleton framework review ===")
print(frameworks.to_string(index=False))

print()
print("=== Skeleton recommendations ===")
print(recommendations.to_string(index=False))
