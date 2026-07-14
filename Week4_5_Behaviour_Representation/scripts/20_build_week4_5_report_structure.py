from pathlib import Path
import pandas as pd


ROOT = Path("Week4_5_Behaviour_Representation")
OUT = ROOT / "outputs/final_report_inputs"
FINAL = ROOT / "final_outputs"
NOTES = ROOT / "notes"

OUT.mkdir(parents=True, exist_ok=True)
FINAL.mkdir(parents=True, exist_ok=True)
NOTES.mkdir(parents=True, exist_ok=True)

sections = [
    {
        "section_no": "1",
        "section_title": "Introduction and Week 4–5 Objective",
        "purpose": "Explain that the goal is behaviour representation and feature engineering, not final classifier training.",
        "main_inputs": "Week4-5 assignment sheet, previous Week 3 corrected dataset",
        "status": "Ready"
    },
    {
        "section_no": "2",
        "section_title": "Completed Internal Dataset Preparation",
        "purpose": "Summarize the Unibo corrected scan-window dataset inherited from Week 3.",
        "main_inputs": "corrected_scan_window_behaviour_annotations.json, tracking CSVs",
        "status": "Ready"
    },
    {
        "section_no": "3",
        "section_title": "Trajectory-Based Feature Engineering",
        "purpose": "Describe speed, acceleration, trajectory length, turning angle, stationary ratio, and group-spatial features.",
        "main_inputs": "trajectory_frame_level_features.csv, trajectory_track_level_summary.csv, trajectory_group_segment_summary.csv",
        "status": "Ready"
    },
    {
        "section_no": "4",
        "section_title": "Trajectory Visualizations",
        "purpose": "Present trajectory plots, heatmaps, occupancy maps, speed plots, and group feature plots.",
        "main_inputs": "outputs/visualizations/trajectory/",
        "status": "Ready"
    },
    {
        "section_no": "5",
        "section_title": "ROI / Resource-Based Feature Engineering",
        "purpose": "Explain centroid/bbox-based ROI prototype and its limitation compared with snout/keypoint-based resource detection.",
        "main_inputs": "roi_frame_level_features.csv, roi_track_level_summary.csv, roi_resource_candidate_segment_summary.csv",
        "status": "Ready"
    },
    {
        "section_no": "6",
        "section_title": "ROI Visual Quality Control",
        "purpose": "Show that ROI overlays were visually inspected and accepted as coarse candidate zones, not final feeder/drinker detectors.",
        "main_inputs": "all_segments_roi_overlay_montage.jpg, roi_visual_quality_control.md",
        "status": "Ready"
    },
    {
        "section_no": "7",
        "section_title": "Public Dataset Exploration",
        "purpose": "Compare Edinburgh, aggressive pig/chicken dataset, and internal Unibo data.",
        "main_inputs": "dataset_inventory_comparison.csv, public_dataset_inventory_summary.md",
        "status": "Ready"
    },
    {
        "section_no": "8",
        "section_title": "Edinburgh Dataset Sample and Annotation Inspection",
        "purpose": "Document raw source package inspection, annotated.tar inspection, and output.json schema.",
        "main_inputs": "edinburgh_pigs161119_sample_zip_inspection.md, edinburgh_output_json_schema_inspection.md",
        "status": "Ready"
    },
    {
        "section_no": "9",
        "section_title": "Edinburgh vs Unibo Annotation Format Comparison",
        "purpose": "Compare public manual ground truth and our corrected scan-window annotation structure.",
        "main_inputs": "edinburgh_vs_unibo_annotation_format_comparison_fixed.md",
        "status": "Ready"
    },
    {
        "section_no": "10",
        "section_title": "Behaviour Representation Survey",
        "purpose": "Compare trajectory, ROI, social-spatial, shape, skeleton, mesh, embedding, and video-temporal representations.",
        "main_inputs": "behaviour_representation_survey.md",
        "status": "Ready"
    },
    {
        "section_no": "11",
        "section_title": "Skeleton Framework Review",
        "purpose": "Review DeepLabCut, SLEAP, and Animal-Pose/AP-10K feasibility.",
        "main_inputs": "skeleton_framework_review.md",
        "status": "Ready"
    },
    {
        "section_no": "12",
        "section_title": "Mesh-Based Representation Review",
        "purpose": "Review AniMer/SMAL-style mesh representation and future feasibility.",
        "main_inputs": "mesh_based_representation_review.md",
        "status": "Ready"
    },
    {
        "section_no": "13",
        "section_title": "Embedding-Based Representation Review",
        "purpose": "Review detector features, self-supervised embeddings, crop/track/video embeddings, and risks.",
        "main_inputs": "embedding_based_representation_review.md",
        "status": "Ready"
    },
    {
        "section_no": "14",
        "section_title": "Final Recommendations",
        "purpose": "Recommend trajectory + group-spatial and ROI features as immediate priority, skeleton as next refinement, embeddings/mesh as future layers.",
        "main_inputs": "representation_recommendation_ranking.csv, skeleton recommendations, embedding recommendations, mesh feasibility",
        "status": "Ready"
    },
]

sections_df = pd.DataFrame(sections)
sections_csv = OUT / "week4_5_report_section_plan.csv"
sections_df.to_csv(sections_csv, index=False)

# Inventory important output files
inventory_rows = []

patterns = [
    ("trajectory_features", ROOT / "outputs/trajectory_features", "*"),
    ("trajectory_visualizations", ROOT / "outputs/visualizations/trajectory", "*"),
    ("roi_features", ROOT / "outputs/roi_features", "*"),
    ("roi_visualizations", ROOT / "outputs/visualizations/roi", "*"),
    ("dataset_comparison", ROOT / "outputs/dataset_comparison", "*"),
    ("representation_tables", ROOT / "outputs/representation_tables", "*"),
    ("notes", ROOT / "notes", "*.md"),
    ("public_dataset_notes", ROOT / "notes/public_dataset_notes", "*.md"),
]

for category, folder, pattern in patterns:
    if folder.exists():
        for p in sorted(folder.glob(pattern)):
            if p.is_file():
                inventory_rows.append({
                    "category": category,
                    "relative_path": str(p.relative_to(ROOT)),
                    "size_bytes": p.stat().st_size
                })

inventory = pd.DataFrame(inventory_rows)
inventory_csv = OUT / "week4_5_deliverable_inventory.csv"
inventory.to_csv(inventory_csv, index=False)

outline_path = FINAL / "Week4_5_Final_Report_Outline.md"

with open(outline_path, "w") as f:
    f.write("# Week 4–5 Final Report Outline\n\n")

    f.write("## Project focus\n\n")
    f.write(
        "The Week 4–5 work focuses on behaviour representation and feature engineering for pig behaviour analysis. "
        "The goal is to compare and prototype representations before final classifier training.\n\n"
    )

    f.write("## Report section plan\n\n")
    f.write(sections_df.to_markdown(index=False))
    f.write("\n\n")

    f.write("## Completed deliverable categories\n\n")
    f.write("- Trajectory feature extraction and visualization\n")
    f.write("- ROI/resource-based feature extraction and visualization\n")
    f.write("- Edinburgh public dataset exploration and annotation inspection\n")
    f.write("- Edinburgh vs Unibo annotation format comparison\n")
    f.write("- Behaviour representation survey\n")
    f.write("- Skeleton framework review\n")
    f.write("- Mesh-based representation review\n")
    f.write("- Embedding-based representation review\n")
    f.write("- Recommendation ranking for next project phase\n\n")

    f.write("## Main recommendation\n\n")
    f.write(
        "The immediate next phase should prioritize trajectory, group-spatial, and ROI/resource-based features because they are already "
        "available from the current tracking outputs and are interpretable. Skeleton/keypoint features should be the next refinement layer, "
        "especially for snout-to-resource distance and head/body orientation. Embeddings and mesh representations should remain future "
        "extensions unless additional labelled data and validation time are available.\n\n"
    )

    f.write("## Deliverable inventory preview\n\n")
    if len(inventory):
        f.write(inventory.head(80).to_markdown(index=False))
    else:
        f.write("No deliverable files found.")
    f.write("\n")

print("Saved:")
print(sections_csv)
print(inventory_csv)
print(outline_path)

print()
print("=== Section plan ===")
print(sections_df.to_string(index=False))

print()
print("=== Inventory count by category ===")
if len(inventory):
    print(inventory.groupby("category").size().reset_index(name="file_count").to_string(index=False))
else:
    print("No files found.")
