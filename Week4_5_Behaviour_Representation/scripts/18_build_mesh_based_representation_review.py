from pathlib import Path
import pandas as pd


ROOT = Path("Week4_5_Behaviour_Representation")
OUT = ROOT / "outputs/representation_tables"
NOTES = ROOT / "notes"

OUT.mkdir(parents=True, exist_ok=True)
NOTES.mkdir(parents=True, exist_ok=True)

mesh_rows = [
    {
        "topic": "SMAL model",
        "description": "SMAL is a parametric animal body model used to represent quadruped body shape and pose.",
        "behaviour_representation_value": "It can encode body pose, body shape, orientation, posture, and potentially deformation-like descriptors.",
        "advantage_over_skeleton": "Skeletons only provide keypoints, while SMAL/mesh can provide full-body surface and shape information.",
        "limitation": "SMAL quality may vary by species; pig-specific validation would be needed before trusting mesh features.",
        "use_in_our_project": "Useful as a feasibility discussion and future direction, not immediate implementation."
    },
    {
        "topic": "Mesh reconstruction",
        "description": "Mesh reconstruction estimates a full 3D animal body surface from image evidence.",
        "behaviour_representation_value": "A mesh sequence can describe posture transitions, body orientation, lying/standing configuration, and shape changes over time.",
        "advantage_over_skeleton": "Can preserve richer body geometry than sparse keypoints.",
        "limitation": "Hard under occlusion, low image quality, overhead farm cameras, and overlapping pigs.",
        "use_in_our_project": "Potentially useful later for posture and body-orientation descriptors if a robust model is available."
    },
    {
        "topic": "AniMer architecture",
        "description": "AniMer uses a high-capacity Transformer-based design for animal pose and shape estimation, including a ViT encoder and SMAL-related decoder/regression components.",
        "behaviour_representation_value": "The predicted pose, shape, camera, and family-aware features could become high-level behavioural descriptors.",
        "advantage_over_skeleton": "It can represent both pose and full-body shape instead of only joint/keypoint locations.",
        "limitation": "Implementation complexity is high and may require GPU-heavy inference and domain adaptation.",
        "use_in_our_project": "Best treated as a literature-backed feasibility option for Week 4-5."
    },
    {
        "topic": "Cross-species generalization",
        "description": "AniMer focuses on multi-species quadruped pose and shape estimation using family-aware learning.",
        "behaviour_representation_value": "This is relevant to the task question of robust representations across farm layout, camera setup, environment, and possibly species.",
        "advantage_over_skeleton": "A learned shape space may provide more transferable posture descriptors than manually defined 2D keypoints alone.",
        "limitation": "General quadruped performance does not guarantee robust performance on farm pigs from overhead camera footage.",
        "use_in_our_project": "Discuss as promising but requiring validation on pig-specific frames."
    },
    {
        "topic": "Synthetic 3D data",
        "description": "AniMer introduces CtrlAni3D, a synthetic dataset with SMAL-aligned labels to improve training diversity.",
        "behaviour_representation_value": "Synthetic mesh-labelled data can help train models when real animal 3D labels are scarce.",
        "advantage_over_skeleton": "Can provide full 3D supervision, not just 2D keypoints.",
        "limitation": "Synthetic-to-real domain gap may still exist, especially for farm camera footage.",
        "use_in_our_project": "Useful as background for feasibility analysis, not something we implement now."
    }
]

mesh_review = pd.DataFrame(mesh_rows)
mesh_review_csv = OUT / "mesh_based_representation_review.csv"
mesh_review.to_csv(mesh_review_csv, index=False)

descriptor_rows = [
    {
        "descriptor_family": "Posture descriptors",
        "mesh_signals": "body pose parameters, body axis, surface orientation",
        "possible_features": "standing-like posture, lying-like posture, sitting-like posture, posture transition events",
        "behaviour_use": "Standing, lying, sitting, resting"
    },
    {
        "descriptor_family": "Orientation descriptors",
        "mesh_signals": "head/body direction, camera-relative orientation, body axis",
        "possible_features": "heading angle, orientation stability, facing another pig, facing feeder/drinker ROI",
        "behaviour_use": "Exploration, feeding/drinking candidate, social interaction"
    },
    {
        "descriptor_family": "Shape descriptors",
        "mesh_signals": "shape parameters, body surface, back curvature",
        "possible_features": "elongation, body compactness, back curvature, body volume proxy",
        "behaviour_use": "Lying vs standing, posture state, abnormal posture screening"
    },
    {
        "descriptor_family": "Temporal mesh descriptors",
        "mesh_signals": "mesh sequence over frames",
        "possible_features": "pose velocity, pose acceleration, shape stability, posture transition frequency",
        "behaviour_use": "Movement, activity, interaction, rapid events"
    },
    {
        "descriptor_family": "Social mesh descriptors",
        "mesh_signals": "mesh position and orientation for multiple pigs",
        "possible_features": "body-to-body distance, facing angle, overlap/contact proxy, relative orientation",
        "behaviour_use": "Interaction, aggression/contact, clustering"
    }
]

descriptors = pd.DataFrame(descriptor_rows)
descriptors_csv = OUT / "mesh_possible_behaviour_descriptors.csv"
descriptors.to_csv(descriptors_csv, index=False)

feasibility_rows = [
    {
        "criterion": "Implementation requirement",
        "assessment": "High",
        "explanation": "Mesh methods require a specialised model, GPU inference, and possibly adaptation to pig footage."
    },
    {
        "criterion": "Annotation requirement",
        "assessment": "Medium to high",
        "explanation": "Direct mesh annotation is difficult; validation may require manual checks or keypoint/segmentation proxies."
    },
    {
        "criterion": "Interpretability",
        "assessment": "High",
        "explanation": "If reliable, mesh parameters can provide meaningful posture, body-shape, and orientation descriptors."
    },
    {
        "criterion": "Robustness to occlusion",
        "assessment": "Low to medium",
        "explanation": "Group-housed pigs overlap heavily, which can reduce mesh reconstruction reliability."
    },
    {
        "criterion": "Suitability for immediate Week 4 implementation",
        "assessment": "Low",
        "explanation": "The assignment asks for feasibility analysis; implementation is not required at this stage."
    },
    {
        "criterion": "Suitability for future research",
        "assessment": "Medium to high",
        "explanation": "Mesh representation could become valuable after trajectory, ROI, and skeleton features are established."
    }
]

feasibility = pd.DataFrame(feasibility_rows)
feasibility_csv = OUT / "mesh_feasibility_assessment.csv"
feasibility.to_csv(feasibility_csv, index=False)

note_path = NOTES / "mesh_based_representation_review.md"

with open(note_path, "w") as f:
    f.write("# Mesh-Based Representation Review\n\n")

    f.write("## Purpose\n\n")
    f.write(
        "This review evaluates mesh-based animal behaviour representations for the Week 4-5 assignment. "
        "The goal is not to implement a mesh model immediately, but to understand what mesh reconstruction "
        "could provide beyond trajectory, ROI, and skeleton features.\n\n"
    )

    f.write("## Main review table\n\n")
    f.write(mesh_review.to_markdown(index=False))
    f.write("\n\n")

    f.write("## Possible mesh-derived behavioural descriptors\n\n")
    f.write(descriptors.to_markdown(index=False))
    f.write("\n\n")

    f.write("## Feasibility assessment\n\n")
    f.write(feasibility.to_markdown(index=False))
    f.write("\n\n")

    f.write("## Advantages over skeletons\n\n")
    f.write(
        "Skeletons represent animals with sparse keypoints, while mesh-based methods can represent the full body surface, "
        "body shape, body orientation, and posture. This can provide richer descriptors for lying, standing, body curvature, "
        "and contact-like interactions. Meshes can also support 3D reasoning if camera calibration and model predictions are reliable.\n\n"
    )

    f.write("## Computational requirements\n\n")
    f.write(
        "Mesh reconstruction is more computationally expensive than trajectory and ROI features. It may require GPU inference, "
        "specialised pretrained models, and careful validation. For group-housed pigs, occlusion and overlapping bodies are major practical obstacles.\n\n"
    )

    f.write("## Applicability to pigs and livestock\n\n")
    f.write(
        "Mesh-based representation is promising for livestock because it can describe posture and body shape more richly than bounding boxes. "
        "However, farm pigs under overhead cameras differ from many generic quadruped datasets. Therefore, pig-specific validation is necessary before "
        "using mesh features as reliable behavioural descriptors.\n\n"
    )

    f.write("## Recommendation\n\n")
    f.write(
        "For the current project phase, mesh-based representation should remain a feasibility review and future direction. "
        "The immediate implementation priority should remain trajectory, ROI/resource, and skeleton/keypoint features. "
        "Mesh representations may become useful later for high-quality posture and biomechanics descriptors.\n"
    )

print("Saved:")
print(mesh_review_csv)
print(descriptors_csv)
print(feasibility_csv)
print(note_path)

print()
print("=== Mesh feasibility assessment ===")
print(feasibility.to_string(index=False))
