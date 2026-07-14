# Mesh-Based Representation Review

## Purpose

This review evaluates mesh-based animal behaviour representations for the Week 4-5 assignment. The goal is not to implement a mesh model immediately, but to understand what mesh reconstruction could provide beyond trajectory, ROI, and skeleton features.

## Main review table

| topic                        | description                                                                                                                                                        | behaviour_representation_value                                                                                                       | advantage_over_skeleton                                                                                           | limitation                                                                                                     | use_in_our_project                                                                                    |
|:-----------------------------|:-------------------------------------------------------------------------------------------------------------------------------------------------------------------|:-------------------------------------------------------------------------------------------------------------------------------------|:------------------------------------------------------------------------------------------------------------------|:---------------------------------------------------------------------------------------------------------------|:------------------------------------------------------------------------------------------------------|
| SMAL model                   | SMAL is a parametric animal body model used to represent quadruped body shape and pose.                                                                            | It can encode body pose, body shape, orientation, posture, and potentially deformation-like descriptors.                             | Skeletons only provide keypoints, while SMAL/mesh can provide full-body surface and shape information.            | SMAL quality may vary by species; pig-specific validation would be needed before trusting mesh features.       | Useful as a feasibility discussion and future direction, not immediate implementation.                |
| Mesh reconstruction          | Mesh reconstruction estimates a full 3D animal body surface from image evidence.                                                                                   | A mesh sequence can describe posture transitions, body orientation, lying/standing configuration, and shape changes over time.       | Can preserve richer body geometry than sparse keypoints.                                                          | Hard under occlusion, low image quality, overhead farm cameras, and overlapping pigs.                          | Potentially useful later for posture and body-orientation descriptors if a robust model is available. |
| AniMer architecture          | AniMer uses a high-capacity Transformer-based design for animal pose and shape estimation, including a ViT encoder and SMAL-related decoder/regression components. | The predicted pose, shape, camera, and family-aware features could become high-level behavioural descriptors.                        | It can represent both pose and full-body shape instead of only joint/keypoint locations.                          | Implementation complexity is high and may require GPU-heavy inference and domain adaptation.                   | Best treated as a literature-backed feasibility option for Week 4-5.                                  |
| Cross-species generalization | AniMer focuses on multi-species quadruped pose and shape estimation using family-aware learning.                                                                   | This is relevant to the task question of robust representations across farm layout, camera setup, environment, and possibly species. | A learned shape space may provide more transferable posture descriptors than manually defined 2D keypoints alone. | General quadruped performance does not guarantee robust performance on farm pigs from overhead camera footage. | Discuss as promising but requiring validation on pig-specific frames.                                 |
| Synthetic 3D data            | AniMer introduces CtrlAni3D, a synthetic dataset with SMAL-aligned labels to improve training diversity.                                                           | Synthetic mesh-labelled data can help train models when real animal 3D labels are scarce.                                            | Can provide full 3D supervision, not just 2D keypoints.                                                           | Synthetic-to-real domain gap may still exist, especially for farm camera footage.                              | Useful as background for feasibility analysis, not something we implement now.                        |

## Possible mesh-derived behavioural descriptors

| descriptor_family         | mesh_signals                                                | possible_features                                                                          | behaviour_use                                                |
|:--------------------------|:------------------------------------------------------------|:-------------------------------------------------------------------------------------------|:-------------------------------------------------------------|
| Posture descriptors       | body pose parameters, body axis, surface orientation        | standing-like posture, lying-like posture, sitting-like posture, posture transition events | Standing, lying, sitting, resting                            |
| Orientation descriptors   | head/body direction, camera-relative orientation, body axis | heading angle, orientation stability, facing another pig, facing feeder/drinker ROI        | Exploration, feeding/drinking candidate, social interaction  |
| Shape descriptors         | shape parameters, body surface, back curvature              | elongation, body compactness, back curvature, body volume proxy                            | Lying vs standing, posture state, abnormal posture screening |
| Temporal mesh descriptors | mesh sequence over frames                                   | pose velocity, pose acceleration, shape stability, posture transition frequency            | Movement, activity, interaction, rapid events                |
| Social mesh descriptors   | mesh position and orientation for multiple pigs             | body-to-body distance, facing angle, overlap/contact proxy, relative orientation           | Interaction, aggression/contact, clustering                  |

## Feasibility assessment

| criterion                                       | assessment     | explanation                                                                                                 |
|:------------------------------------------------|:---------------|:------------------------------------------------------------------------------------------------------------|
| Implementation requirement                      | High           | Mesh methods require a specialised model, GPU inference, and possibly adaptation to pig footage.            |
| Annotation requirement                          | Medium to high | Direct mesh annotation is difficult; validation may require manual checks or keypoint/segmentation proxies. |
| Interpretability                                | High           | If reliable, mesh parameters can provide meaningful posture, body-shape, and orientation descriptors.       |
| Robustness to occlusion                         | Low to medium  | Group-housed pigs overlap heavily, which can reduce mesh reconstruction reliability.                        |
| Suitability for immediate Week 4 implementation | Low            | The assignment asks for feasibility analysis; implementation is not required at this stage.                 |
| Suitability for future research                 | Medium to high | Mesh representation could become valuable after trajectory, ROI, and skeleton features are established.     |

## Advantages over skeletons

Skeletons represent animals with sparse keypoints, while mesh-based methods can represent the full body surface, body shape, body orientation, and posture. This can provide richer descriptors for lying, standing, body curvature, and contact-like interactions. Meshes can also support 3D reasoning if camera calibration and model predictions are reliable.

## Computational requirements

Mesh reconstruction is more computationally expensive than trajectory and ROI features. It may require GPU inference, specialised pretrained models, and careful validation. For group-housed pigs, occlusion and overlapping bodies are major practical obstacles.

## Applicability to pigs and livestock

Mesh-based representation is promising for livestock because it can describe posture and body shape more richly than bounding boxes. However, farm pigs under overhead cameras differ from many generic quadruped datasets. Therefore, pig-specific validation is necessary before using mesh features as reliable behavioural descriptors.

## Recommendation

For the current project phase, mesh-based representation should remain a feasibility review and future direction. The immediate implementation priority should remain trajectory, ROI/resource, and skeleton/keypoint features. Mesh representations may become useful later for high-quality posture and biomechanics descriptors.
