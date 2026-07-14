from pathlib import Path
import pandas as pd


ROOT = Path("Week4_5_Behaviour_Representation")
OUT = ROOT / "outputs/final_report_inputs"
NOTES = ROOT / "notes"
FINAL = ROOT / "final_outputs"

OUT.mkdir(parents=True, exist_ok=True)
NOTES.mkdir(parents=True, exist_ok=True)
FINAL.mkdir(parents=True, exist_ok=True)

task_rows = [
    {
        "task_sheet_item": "Task 1 - Edinburgh Pig Behaviour Dataset",
        "required_work": "Study videos, tracking information, behavioural annotations, data formats, use cases, labels, granularity, advantages and limitations",
        "what_we_completed": "Downloaded/inspected source page, raw sample zip, annotated.tar, output.json schema, behaviour counts, and Edinburgh vs Unibo format comparison",
        "main_outputs": "edinburgh_pigs161119_sample_zip_inspection.md; edinburgh_output_json_schema_inspection.md; edinburgh_vs_unibo_annotation_format_comparison_fixed.csv",
        "status": "Completed"
    },
    {
        "task_sheet_item": "Task 1 - Aggressive Behaviour Dataset",
        "required_work": "Study dataset structure, annotation format, clip duration, behaviour classes, suitability for video-based recognition, compare with Edinburgh",
        "what_we_completed": "Cloned/inspected repository and README; added explicit access audit; documented that full video/annotation files are not visible locally and dataset is better treated as video-temporal recognition reference",
        "main_outputs": "aggressive_dataset_access_audit.md; aggressive_dataset_access_audit.csv; dataset_inventory_comparison.csv",
        "status": "Completed with access limitation"
    },
    {
        "task_sheet_item": "Task 2 - Behaviour Representation Survey",
        "required_work": "Compare trajectory, shape, skeleton, mesh, and embedding representations",
        "what_we_completed": "Created structured representation survey including trajectory, ROI/resource, social/group-spatial, shape, skeleton, mesh, embedding, and end-to-end temporal models",
        "main_outputs": "behaviour_representation_survey.md; behaviour_representation_survey.csv; representation_recommendation_ranking.csv",
        "status": "Completed"
    },
    {
        "task_sheet_item": "Task 3 - Trajectory Features",
        "required_work": "Investigate speed, acceleration, displacement, trajectory length, turning angle, occupancy maps, heatmaps, time in ROI, nearest-neighbour distance, local density; implement prototype and visual examples",
        "what_we_completed": "Implemented trajectory feature extraction, group-spatial features, ROI/resource features, and generated trajectory/ROI visualizations",
        "main_outputs": "outputs/trajectory_features/; outputs/visualizations/trajectory/; outputs/roi_features/; outputs/visualizations/roi/",
        "status": "Completed"
    },
    {
        "task_sheet_item": "Task 4 - Skeleton-Based Representations",
        "required_work": "Review DeepLabCut, SLEAP, Animal-Pose/AP-10K; required annotations, multi-animal support, suitability for pigs, expected outputs, integration",
        "what_we_completed": "Created skeleton framework comparison, possible skeleton-derived features, and recommendation table",
        "main_outputs": "skeleton_framework_review.md; skeleton_framework_review.csv; skeleton_possible_features.csv",
        "status": "Completed"
    },
    {
        "task_sheet_item": "Task 5 - Mesh-Based Representations",
        "required_work": "Study AniMer/SMAL, mesh reconstruction, pose estimation, descriptors, advantages over skeletons, computational requirements, livestock applicability; implementation not required",
        "what_we_completed": "Created mesh-based representation review, possible mesh descriptors, and feasibility assessment",
        "main_outputs": "mesh_based_representation_review.md; mesh_based_representation_review.csv; mesh_feasibility_assessment.csv",
        "status": "Completed"
    },
    {
        "task_sheet_item": "Task 6 - Deep Embeddings",
        "required_work": "Investigate YOLO/RT-DETR/Co-DINO features, DINOv2, CLIP, SigLIP, VideoMAE, Video Swin; representation meaning, extraction, applications; implementation optional",
        "what_we_completed": "Created embedding representation review, feature levels, risks/mitigation, and recommendations",
        "main_outputs": "embedding_based_representation_review.md; embedding_based_representation_review.csv; embedding_risks_and_mitigation.csv",
        "status": "Completed"
    },
    {
        "task_sheet_item": "Final Deliverables",
        "required_work": "Dataset comparison report, representation survey, trajectory examples, skeleton review, mesh review, embedding review, recommendation document",
        "what_we_completed": "All deliverables were included in final report draft, PDF, and final ZIP package",
        "main_outputs": "Week4_5_Final_Report_Draft.pdf; Week4_5_Behaviour_Representation_Final_Package.zip",
        "status": "Completed"
    },
]

task_df = pd.DataFrame(task_rows)
task_csv = OUT / "task_sheet_compliance_matrix.csv"
task_df.to_csv(task_csv, index=False)

reading_rows = [
    {
        "required_reading": "Comparative Analysis of Methods for Automated Detection of Pig Behaviour from Visual Imaging",
        "focus_requested": "engineered behavioural features; feature importance; generalization across pen layouts",
        "how_used_in_our_work": "Supports the argument that engineered detection/tracklet features are valuable when labelled data is limited and generalization is a concern",
        "connected_report_sections": "Trajectory features; ROI/resource features; Final recommendations",
        "status": "Summarized through representation discussion"
    },
    {
        "required_reading": "HABLer",
        "focus_requested": "behaviour quantifiers; AI-assisted annotation; behaviour representation concepts",
        "how_used_in_our_work": "Supports the idea that quantitative behavioural metrics and expert-guided correction can help build behaviour labels and representation pipelines",
        "connected_report_sections": "Behaviour representation survey; future annotation/validation discussion",
        "status": "Summarized through representation discussion"
    },
    {
        "required_reading": "AniMer",
        "focus_requested": "mesh-based animal representation; SMAL model; cross-species generalization",
        "how_used_in_our_work": "Used as the core source for mesh-based representation feasibility and future posture/shape descriptors",
        "connected_report_sections": "Mesh-based representation review",
        "status": "Completed"
    },
    {
        "required_reading": "Efficient Aggressive Behavior Recognition of Pigs Based on Temporal Shift Module",
        "focus_requested": "video-based behaviour recognition; temporal modelling; strengths and weaknesses of end-to-end approaches",
        "how_used_in_our_work": "Used to position the aggressive dataset and TSM-style models as video-temporal references rather than trajectory/ROI feature datasets",
        "connected_report_sections": "Embedding review; end-to-end video/temporal model discussion; aggressive dataset audit",
        "status": "Completed"
    },
    {
        "required_reading": "Larsen 2026 computer vision + ethology feeding/drinking paper",
        "focus_requested": "not in original four-item list but useful for ROI/resource-based behaviour representation",
        "how_used_in_our_work": "Used to motivate ROI/resource occupation features and explain why snout/head keypoints would improve centroid-only ROI features",
        "connected_report_sections": "ROI/resource features; skeleton review; final recommendations",
        "status": "Additional supporting reading"
    },
]

reading_df = pd.DataFrame(reading_rows)
reading_csv = OUT / "required_reading_usage_matrix.csv"
reading_df.to_csv(reading_csv, index=False)

gap_rows = [
    {
        "possible_gap": "Aggressive dataset not fully downloaded",
        "risk_level": "Medium",
        "mitigation_added": "Created explicit access audit and documented that full data is externally distributed / not directly inspectable from repository",
        "remaining_risk": "If supervisor expects full download, manual Baidu access may still be needed"
    },
    {
        "possible_gap": "No final behaviour classifier",
        "risk_level": "Low",
        "mitigation_added": "Task sheet explicitly says final classifier is not the objective at this stage",
        "remaining_risk": "None for Week 4-5"
    },
    {
        "possible_gap": "Skeleton/mesh/embedding not implemented",
        "risk_level": "Low",
        "mitigation_added": "Task sheet asks for investigation/feasibility; implementation is not required for mesh and optional for embeddings",
        "remaining_risk": "Future work only"
    },
    {
        "possible_gap": "ROI features are coarse",
        "risk_level": "Low-Medium",
        "mitigation_added": "Report clearly states centroid/bbox ROI features are candidate zones, not final feeding/drinking detectors",
        "remaining_risk": "Snout/keypoint-based refinement needed later"
    },
]

gap_df = pd.DataFrame(gap_rows)
gap_csv = OUT / "remaining_gap_analysis.csv"
gap_df.to_csv(gap_csv, index=False)

note_path = FINAL / "Week4_5_Task_Compliance_Addendum.md"

with open(note_path, "w") as f:
    f.write("# Week 4-5 Task Compliance Addendum\n\n")

    f.write("## Purpose\n\n")
    f.write(
        "This addendum explicitly maps the Week 4-5 task sheet requirements to the outputs generated in this project. "
        "It also documents the remaining limitations and how they were mitigated in the final deliverables.\n\n"
    )

    f.write("## Task sheet compliance matrix\n\n")
    f.write(task_df.to_markdown(index=False))
    f.write("\n\n")

    f.write("## Required reading usage matrix\n\n")
    f.write(reading_df.to_markdown(index=False))
    f.write("\n\n")

    f.write("## Remaining gap analysis\n\n")
    f.write(gap_df.to_markdown(index=False))
    f.write("\n\n")

    f.write("## Final interpretation\n\n")
    f.write(
        "The Week 4-5 deliverables are complete for a representation and feature-engineering stage. "
        "The only meaningful limitation is that the aggressive pig/chicken dataset could not be fully inspected beyond repository metadata without external Baidu access. "
        "This limitation is documented and does not invalidate the main trajectory/ROI representation pipeline, because the aggressive dataset is primarily useful for video-temporal recognition rather than tracking-based feature engineering.\n"
    )

print("Saved:")
print(task_csv)
print(reading_csv)
print(gap_csv)
print(note_path)

print()
print("=== Task compliance ===")
print(task_df.to_string(index=False))

print()
print("=== Gap analysis ===")
print(gap_df.to_string(index=False))
