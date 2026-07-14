from pathlib import Path
import pandas as pd


ROOT = Path("Week4_5_Behaviour_Representation")
OUT = ROOT / "outputs/final_report_inputs"
FINAL = ROOT / "final_outputs"
NOTES = ROOT / "notes"
PUBLIC_NOTES = NOTES / "public_dataset_notes"
REP = ROOT / "outputs/representation_tables"
TRAJ = ROOT / "outputs/trajectory_features"
ROI = ROOT / "outputs/roi_features"
DATASET = ROOT / "outputs/dataset_comparison"

FINAL.mkdir(parents=True, exist_ok=True)

def read_text(path, fallback=""):
    p = Path(path)
    if p.exists():
        return p.read_text(errors="ignore")
    return fallback

def csv_table(path, max_rows=None):
    p = Path(path)
    if not p.exists():
        return f"`{p}` not found.\n"
    df = pd.read_csv(p)
    if max_rows is not None:
        df = df.head(max_rows)
    return df.to_markdown(index=False)

def section(title):
    return f"\n\n# {title}\n\n"

def subsection(title):
    return f"\n\n## {title}\n\n"

report = []

report.append("# Week 4–5 Behaviour Representation and Feature Engineering Report\n\n")

report.append(
    "This report summarizes the Week 4–5 work on pig behaviour representation and feature engineering. "
    "The focus is not final classifier training. Instead, the work compares representation families, "
    "builds interpretable trajectory and ROI features from the corrected Unibo scan-window tracking outputs, "
    "and reviews public datasets and future representation layers such as skeletons, meshes, and embeddings.\n"
)

report.append(section("1. Objective and Scope"))
report.append(
    "The Week 4–5 objective is to understand how pig behaviour can be represented before training a final model. "
    "The main question is which representations are robust enough for different cameras, pen layouts, environmental conditions, "
    "and potentially different animal datasets. The work therefore combines practical feature engineering on the Unibo data "
    "with a structured review of public datasets and representation methods.\n"
)

report.append(section("2. Completed Internal Dataset Preparation"))
report.append(
    "The internal dataset used in this phase is the corrected Unibo scan-window dataset created from Week 3. "
    "It contains six scan-window segments aligned to Excel observation intervals: scan_09_00, scan_09_10, scan_09_20, "
    "scan_09_30, scan_09_40, and scan_09_50. The dataset includes YOLOv8-s detection, ByteTrack tracking, bounding boxes, "
    "track IDs, segment-level behaviour labels, and conservative identity handling.\n"
)

report.append(subsection("Unibo annotation summary"))
report.append(csv_table(DATASET / "unibo_annotation_schema_summary_fixed.csv"))

report.append(section("3. Trajectory-Based Feature Engineering"))
report.append(
    "Trajectory-based representation uses frame-level tracking outputs to compute motion descriptors. "
    "The implemented features include centroid movement, speed, acceleration, displacement, trajectory length, turning angle, "
    "stationary ratio, fast-motion ratio, and straightness. These features are interpretable and can be computed directly "
    "from bounding boxes and track IDs.\n"
)

report.append(subsection("Segment motion summary"))
report.append(csv_table(TRAJ / "trajectory_segment_motion_summary.csv"))

report.append(subsection("Group-spatial summary"))
report.append(csv_table(TRAJ / "trajectory_group_segment_summary.csv"))

report.append(
    "\nThe trajectory outputs show differences between scan windows. For example, scan_09_50 is comparatively stationary, "
    "while scan_09_20 and scan_09_00 show higher motion and larger speed spikes. These spikes are useful to report, but they "
    "must be interpreted carefully because tracking jitter, ID switches, or bounding-box jumps can inflate instantaneous speed.\n"
)

report.append(section("4. Trajectory Visualizations"))
report.append(
    "The trajectory visualization stage generated dominant-track trajectory plots, centroid heatmaps, normalized occupancy maps, "
    "speed-over-time plots, group feature plots, and cross-segment summary plots. These figures provide visual support for the "
    "trajectory representation layer.\n\n"
)
report.append("Key visualization folder:\n\n")
report.append("```text\noutputs/visualizations/trajectory/\n```\n")

report.append(section("5. ROI / Resource-Based Feature Engineering"))
report.append(
    "ROI/resource-based representation describes where pigs spend time inside manually defined spatial zones. "
    "This is inspired by resource-use behaviour analysis, but the current implementation is intentionally conservative: "
    "it uses centroid and bounding-box tracking only, not snout keypoints or rotated body boxes. Therefore, the ROI features "
    "are reported as coarse spatial/resource/contact candidate features, not final feeding or drinking detectors.\n"
)

report.append(subsection("Resource candidate segment summary"))
report.append(csv_table(ROI / "roi_resource_candidate_segment_summary.csv"))

report.append(section("6. ROI Visual Quality Control"))
report.append(
    "ROI overlay images were generated for each corrected scan-window segment and combined into a montage. "
    "The visual inspection showed that the zones are usable for a first feature-engineering prototype, but not precise enough "
    "to be interpreted as exact feeder or drinker regions.\n\n"
)
report.append("Key visualization:\n\n")
report.append("```text\noutputs/visualizations/roi/all_segments_roi_overlay_montage.jpg\n```\n")

report.append(section("7. Public Dataset Exploration"))
report.append(
    "The public dataset exploration compared the Edinburgh Pig Behaviour Dataset, the aggressive pig/chicken dataset, "
    "and the internal Unibo corrected scan-window dataset. The Edinburgh dataset is the strongest public reference for "
    "manual annotation structure because it includes video, depth, masks, persistent object IDs, bounding boxes, and behaviour labels. "
    "The aggressive pig/chicken dataset is useful as a video-classification reference, but its full data access depends on external Baidu links.\n"
)

report.append(subsection("Dataset inventory comparison"))
report.append(csv_table(DATASET / "dataset_inventory_comparison.csv"))

report.append(section("8. Edinburgh Dataset Sample and Annotation Inspection"))
report.append(
    "A raw Edinburgh sample package, `pigs161119.zip`, was manually downloaded and inspected. "
    "It contains raw source clips with colour video, depth video, background images, masks, calibration-related files, and time files, "
    "but no `output.json`. The separate `annotated.tar` package was then inspected and confirmed to contain 12 manually ground-truthed "
    "`output.json` files.\n"
)

report.append(subsection("Raw sample clip summary"))
report.append(csv_table(DATASET / "edinburgh_pigs161119_clip_summary.csv"))

report.append(subsection("Edinburgh output.json schema summary"))
report.append(csv_table(DATASET / "edinburgh_sample_output_json_schema_summary.csv"))

report.append(subsection("Edinburgh sample behaviour counts"))
report.append(csv_table(DATASET / "edinburgh_sample_behaviour_counts.csv"))

report.append(section("9. Edinburgh vs Unibo Annotation Format Comparison"))
report.append(
    "The Edinburgh manually annotated format and the Unibo corrected scan-window format are complementary. "
    "Edinburgh provides cleaner manual frame-level annotation with persistent object IDs. Unibo is more project-specific "
    "and contains corrected scan-window alignment plus detector/tracker outputs, but its behaviour labels are attached at scan-window level.\n"
)

report.append(csv_table(DATASET / "edinburgh_vs_unibo_annotation_format_comparison_fixed.csv"))

report.append(section("10. Behaviour Representation Survey"))
report.append(
    "The representation survey compares trajectory-based, ROI/resource-based, social/group-spatial, shape-based, skeleton-based, "
    "mesh-based, embedding-based, and end-to-end temporal model representations.\n"
)

report.append(subsection("Representation families"))
report.append(csv_table(REP / "behaviour_representation_survey.csv"))

report.append(subsection("Representation recommendation ranking"))
report.append(csv_table(REP / "representation_recommendation_ranking.csv"))

report.append(section("11. Skeleton Framework Review"))
report.append(
    "Skeleton/keypoint methods can refine the current trajectory and ROI features by adding snout/head position, body axis, posture, "
    "and orientation information. DeepLabCut and SLEAP are suitable candidates for a small feasibility test, while Animal-Pose/AP-10K "
    "can be considered as a pretrained baseline with domain-shift limitations.\n"
)

report.append(subsection("Skeleton framework comparison"))
report.append(csv_table(REP / "skeleton_framework_review.csv"))

report.append(subsection("Skeleton possible features"))
report.append(csv_table(REP / "skeleton_possible_features.csv"))

report.append(section("12. Mesh-Based Representation Review"))
report.append(
    "Mesh-based representation, especially AniMer/SMAL-style pose and shape estimation, can provide richer full-body posture and shape descriptors "
    "than sparse keypoints. However, mesh reconstruction is computationally heavy and uncertain under pig-specific overhead farm footage, so it remains "
    "a future feasibility direction rather than an immediate implementation target.\n"
)

report.append(subsection("Mesh feasibility assessment"))
report.append(csv_table(REP / "mesh_feasibility_assessment.csv"))

report.append(subsection("Mesh possible descriptors"))
report.append(csv_table(REP / "mesh_possible_behaviour_descriptors.csv"))

report.append(section("13. Embedding-Based Representation Review"))
report.append(
    "Embedding-based representations can capture appearance, posture, context, and temporal cues that hand-crafted features may miss. "
    "However, they are less interpretable and may suffer from domain shift. For this project, embeddings should be complementary to trajectory, ROI, "
    "and social-spatial features rather than the primary representation.\n"
)

report.append(subsection("Embedding families"))
report.append(csv_table(REP / "embedding_based_representation_review.csv"))

report.append(subsection("Embedding risks and mitigation"))
report.append(csv_table(REP / "embedding_risks_and_mitigation.csv"))

report.append(section("14. Final Recommendations"))
report.append(
    "The immediate next phase should prioritize trajectory, group-spatial, and ROI/resource-based features because they are already implemented, "
    "interpretable, and directly derived from the current tracking outputs. Skeleton/keypoint features should be the next refinement layer, especially "
    "for snout-to-resource distance and head/body orientation. Embedding and mesh representations should remain future extensions unless additional "
    "labelled data, validation time, and computational resources are available.\n"
)

report.append(subsection("Recommended next steps"))
report.append(
    "1. Apply smoothing and outlier filtering to trajectory features.\n"
    "2. Compare trajectory and ROI features across behaviour-labelled scan windows.\n"
    "3. Refine ROI definitions using visual inspection and, later, snout/head keypoints.\n"
    "4. Run a small skeleton feasibility test with a minimal pig keypoint set.\n"
    "5. Keep embeddings and mesh representations as future feature layers.\n"
)

report.append(section("15. Deliverable Inventory"))
report.append(
    "A structured deliverable inventory was created to track all generated CSV files, notes, and visualizations.\n"
)

report.append(csv_table(OUT / "week4_5_deliverable_inventory.csv", max_rows=80))

report.append("\n\n# Conclusion\n\n")
report.append(
    "The Week 4–5 work establishes a strong representation-oriented foundation for pig behaviour analysis. "
    "The project now has implemented trajectory features, group-spatial features, ROI/resource candidate features, visualizations, "
    "public dataset exploration, Edinburgh annotation inspection, and structured reviews of skeleton, mesh, and embedding representations. "
    "This provides a clear path toward future behaviour classification while preserving interpretability and scientific caution.\n"
)

final_report = FINAL / "Week4_5_Final_Report_Draft.md"
final_report.write_text("".join(report))

print("Saved:", final_report)
print("Size bytes:", final_report.stat().st_size)
