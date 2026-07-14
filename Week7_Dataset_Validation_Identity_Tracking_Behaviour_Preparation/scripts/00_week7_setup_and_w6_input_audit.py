from pathlib import Path
from datetime import datetime
import csv
import os
import pandas as pd


ROOT = Path.home() / "PigBench"

W6 = ROOT / "Week6_Unibo_Dataset_Validation"
W7 = ROOT / "Week7_Dataset_Validation_Identity_Tracking_Behaviour_Preparation"

RAW_UNIBO = Path("/work/pig/datasets/Unibo")

DIRS = [
    "inputs",
    "outputs",
    "outputs/dataset_statistics",
    "outputs/roi_and_crate_mapping",
    "outputs/colour_identity",
    "outputs/tracking",
    "outputs/identity_fusion",
    "outputs/visualizations",
    "outputs/external_dataset_validation",
    "outputs/split_protocols",
    "outputs/clip_level_representation",
    "outputs/scene_change_detection",
    "scripts",
    "notes",
    "shared_tracker",
    "final_delivery_package",
]

for d in DIRS:
    (W7 / d).mkdir(parents=True, exist_ok=True)


def safe_to_csv(df, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(
        path,
        index=False,
        quoting=csv.QUOTE_ALL,
        escapechar="\\",
        lineterminator="\n",
    )


def file_size_mb(path):
    path = Path(path)
    if not path.exists() or not path.is_file():
        return ""
    return round(path.stat().st_size / (1024 * 1024), 3)


def csv_row_count(path):
    path = Path(path)
    if not path.exists() or path.suffix.lower() != ".csv":
        return ""
    try:
        return len(pd.read_csv(path))
    except Exception:
        return "read_error"


def dir_file_count(path, pattern="*"):
    path = Path(path)
    if not path.exists() or not path.is_dir():
        return ""
    return len([p for p in path.glob(pattern) if p.is_file()])


def make_symlink(target, link):
    target = Path(target)
    link = Path(link)

    if link.exists() or link.is_symlink():
        return "already_exists"

    if not target.exists():
        return "target_missing"

    try:
        link.symlink_to(target)
        return "created"
    except Exception as e:
        return f"failed: {type(e).__name__}: {e}"


# ---------------------------------------------------------------------
# Optional symlinks to avoid copying large data.
# ---------------------------------------------------------------------
symlink_rows = []

symlink_rows.append({
    "link": str(W7 / "inputs" / "week6_project"),
    "target": str(W6),
    "status": make_symlink(W6, W7 / "inputs" / "week6_project"),
})

symlink_rows.append({
    "link": str(W7 / "inputs" / "raw_unibo_dataset"),
    "target": str(RAW_UNIBO),
    "status": make_symlink(RAW_UNIBO, W7 / "inputs" / "raw_unibo_dataset"),
})

safe_to_csv(
    pd.DataFrame(symlink_rows),
    W7 / "outputs" / "dataset_statistics" / "week7_input_symlink_status.csv",
)


# ---------------------------------------------------------------------
# Week 7 master task tracker.
# ---------------------------------------------------------------------
tasks = [
    {
        "stage": "0",
        "task_name": "Week 7 project setup",
        "objective": "Create the Week 7 folder structure, trackers, notes, and audit framework.",
        "primary_output_folder": "outputs/dataset_statistics",
        "status": "done",
        "priority": "critical",
    },
    {
        "stage": "1",
        "task_name": "Week 6 input ingestion and audit",
        "objective": "Verify that all Week 6 final outputs required for Week 7 exist and are readable.",
        "primary_output_folder": "outputs/dataset_statistics",
        "status": "done",
        "priority": "critical",
    },
    {
        "stage": "2",
        "task_name": "Region of Interest and crate mapping",
        "objective": "Define the annotated pen or crate and flag detections inside or outside the ground-truth region.",
        "primary_output_folder": "outputs/roi_and_crate_mapping",
        "status": "not_started",
        "priority": "critical",
    },
    {
        "stage": "3",
        "task_name": "Ground truth dataset statistical analysis",
        "objective": "Generate complete dataset statistics, behaviour distributions, crate or pen statistics, duration summaries, and class imbalance analysis.",
        "primary_output_folder": "outputs/dataset_statistics",
        "status": "not_started",
        "priority": "critical",
    },
    {
        "stage": "4",
        "task_name": "Colour marker rule definition",
        "objective": "Define robust colour matching rules using colour spaces, normalization, segmentation masks, and expected crate-level colour sets.",
        "primary_output_folder": "outputs/colour_identity",
        "status": "not_started",
        "priority": "critical",
    },
    {
        "stage": "5",
        "task_name": "Scanpoint colour assignment baseline",
        "objective": "Apply colour rules to scanpoint detections and compare assignments with manual colour labels.",
        "primary_output_folder": "outputs/colour_identity",
        "status": "not_started",
        "priority": "critical",
    },
    {
        "stage": "6",
        "task_name": "Long-term tracking setup",
        "objective": "Run tracking on longer video sequences, measure track continuity, gaps, and identity-switch candidates.",
        "primary_output_folder": "outputs/tracking",
        "status": "not_started",
        "priority": "critical",
    },
    {
        "stage": "7",
        "task_name": "Track ID and colour ID fusion",
        "objective": "Use temporal colour evidence and tracking continuity to assign stable colour identities to tracks.",
        "primary_output_folder": "outputs/identity_fusion",
        "status": "not_started",
        "priority": "critical",
    },
    {
        "stage": "8",
        "task_name": "Improved visualization interface",
        "objective": "Show bounding boxes, track IDs, colour identities, behaviour labels, timestamps, and Region of Interest overlays.",
        "primary_output_folder": "outputs/visualizations",
        "status": "not_started",
        "priority": "high",
    },
    {
        "stage": "9",
        "task_name": "Dataset splitting strategy analysis",
        "objective": "Compare chronological, behaviour-balanced, recording-session, crate or pen, and hybrid split protocols.",
        "primary_output_folder": "outputs/split_protocols",
        "status": "not_started",
        "priority": "high",
    },
    {
        "stage": "10",
        "task_name": "External dataset validation",
        "objective": "Compare our behaviour taxonomy with public datasets and evaluate transfer-learning opportunities.",
        "primary_output_folder": "outputs/external_dataset_validation",
        "status": "not_started",
        "priority": "high",
    },
    {
        "stage": "11",
        "task_name": "Behaviour feature representation analysis",
        "objective": "Compare trajectory, Region of Interest, detector embedding, colour descriptor, and segmentation descriptor relevance.",
        "primary_output_folder": "outputs/clip_level_representation",
        "status": "not_started",
        "priority": "high",
    },
    {
        "stage": "12",
        "task_name": "Clip-level behaviour representation",
        "objective": "Prepare temporal representations aligned with approximately ten-second observation windows.",
        "primary_output_folder": "outputs/clip_level_representation",
        "status": "not_started",
        "priority": "high",
    },
    {
        "stage": "13",
        "task_name": "Scene-change detection",
        "objective": "Detect camera interruptions, recording anomalies, and major environmental changes in long videos.",
        "primary_output_folder": "outputs/scene_change_detection",
        "status": "not_started",
        "priority": "medium",
    },
    {
        "stage": "14",
        "task_name": "Week 7 compliance and quality audit",
        "objective": "Map every task and deliverable to evidence files and identify remaining gaps.",
        "primary_output_folder": "outputs/dataset_statistics",
        "status": "not_started",
        "priority": "critical",
    },
    {
        "stage": "15",
        "task_name": "Week 7 final package and presentation update",
        "objective": "Create final Week 7 delivery package with manifest, checksums, notes, scripts, visuals, and presentation-ready outputs.",
        "primary_output_folder": "final_delivery_package",
        "status": "not_started",
        "priority": "critical",
    },
]

task_tracker = pd.DataFrame(tasks)
safe_to_csv(task_tracker, W7 / "outputs" / "dataset_statistics" / "week7_master_task_tracker.csv")


# ---------------------------------------------------------------------
# Assignment task and deliverable mapping.
# ---------------------------------------------------------------------
assignment_tasks = [
    {
        "assignment_task": "Task 1",
        "title": "Improve Pig Identity Assignment",
        "key_question": "Can each tracked pig be assigned one stable colour identity?",
        "planned_evidence": "colour rules, scanpoint colour assignment, track-level colour fusion",
        "week7_stage": "4,5,7",
        "status": "not_started",
    },
    {
        "assignment_task": "Task 2",
        "title": "Long-Term Tracking",
        "key_question": "Can identities be maintained over longer video sequences with fewer switches?",
        "planned_evidence": "tracking output, track statistics, smoothed trajectories, switch candidates",
        "week7_stage": "6,7",
        "status": "not_started",
    },
    {
        "assignment_task": "Task 3",
        "title": "Improve Ground Truth Visualization",
        "key_question": "Can the viewer show Region of Interest, track ID, colour identity, behaviour label, frame number, and timestamp?",
        "planned_evidence": "enhanced viewer, overlay videos, contact sheets, visualization library audit",
        "week7_stage": "2,8",
        "status": "not_started",
    },
    {
        "assignment_task": "Task 4",
        "title": "Ground Truth Dataset Analysis",
        "key_question": "What are the dataset statistics, behaviour counts, class imbalance, crate counts, and session coverage?",
        "planned_evidence": "statistical report, charts, crate/video/session summaries",
        "week7_stage": "3",
        "status": "not_started",
    },
    {
        "assignment_task": "Task 5",
        "title": "Dataset Splitting Strategy",
        "key_question": "Which train, validation, and test protocol is most reliable and least biased?",
        "planned_evidence": "split comparison tables and recommended protocol",
        "week7_stage": "9",
        "status": "not_started",
    },
    {
        "assignment_task": "Task 6",
        "title": "External Dataset Validation",
        "key_question": "How compatible are our behaviour labels with public pig behaviour datasets?",
        "planned_evidence": "dataset inventory, behaviour label mapping, protocol comparison",
        "week7_stage": "10",
        "status": "not_started",
    },
    {
        "assignment_task": "Task 7",
        "title": "Behaviour Representation",
        "key_question": "Which feature families are most informative for each behaviour?",
        "planned_evidence": "feature family relevance matrix and behaviour requirement table",
        "week7_stage": "11",
        "status": "not_started",
    },
    {
        "assignment_task": "Task 8",
        "title": "Clip-Level Behaviour Representation",
        "key_question": "How should ten-second observation windows be represented?",
        "planned_evidence": "clip window index, temporal feature design, frame-versus-clip analysis",
        "week7_stage": "12",
        "status": "not_started",
    },
    {
        "assignment_task": "Task 9",
        "title": "Scene-change detection",
        "key_question": "Can recording anomalies or major scene changes be detected automatically?",
        "planned_evidence": "scene-change candidate table, summary, visual examples",
        "week7_stage": "13",
        "status": "not_started",
    },
]

safe_to_csv(
    pd.DataFrame(assignment_tasks),
    W7 / "outputs" / "dataset_statistics" / "week7_assignment_task_mapping.csv",
)

deliverables = [
    {
        "deliverable": "Deliverable 1",
        "description": "Improved identity tracking and colour assignment",
        "planned_folder": "outputs/colour_identity and outputs/identity_fusion",
        "status": "not_started",
    },
    {
        "deliverable": "Deliverable 2",
        "description": "Updated visualization interface",
        "planned_folder": "outputs/visualizations",
        "status": "not_started",
    },
    {
        "deliverable": "Deliverable 3",
        "description": "Demonstration videos showing bounding boxes, colour IDs, and behaviour labels",
        "planned_folder": "outputs/visualizations",
        "status": "not_started",
    },
    {
        "deliverable": "Deliverable 4",
        "description": "Statistical analysis of the Unibo dataset",
        "planned_folder": "outputs/dataset_statistics",
        "status": "not_started",
    },
    {
        "deliverable": "Deliverable 5",
        "description": "Proposed train, validation, and test protocol",
        "planned_folder": "outputs/split_protocols",
        "status": "not_started",
    },
    {
        "deliverable": "Deliverable 6",
        "description": "Comparison with public behaviour datasets",
        "planned_folder": "outputs/external_dataset_validation",
        "status": "not_started",
    },
    {
        "deliverable": "Deliverable 7",
        "description": "Summary of clip-level behaviour representation ideas",
        "planned_folder": "outputs/clip_level_representation",
        "status": "not_started",
    },
    {
        "deliverable": "Deliverable 8",
        "description": "Updated shared Excel file documenting experiments, results, and issues",
        "planned_folder": "shared_tracker",
        "status": "not_started",
    },
]

safe_to_csv(
    pd.DataFrame(deliverables),
    W7 / "outputs" / "dataset_statistics" / "week7_deliverable_tracker.csv",
)


# ---------------------------------------------------------------------
# Week 6 input inventory for Week 7.
# ---------------------------------------------------------------------
inputs = [
    {
        "input_name": "Week 6 final package v3 zip",
        "path": W6 / "final_delivery_package_v3" / "Week6_Unibo_Dataset_Validation_Final_Package_v3.zip",
        "type": "zip",
        "used_for_week7_stage": "all reference",
        "critical": True,
    },
    {
        "input_name": "Week 6 final package v3 directory",
        "path": W6 / "final_delivery_package_v3" / "Week6_Unibo_Dataset_Validation_Final_Package_v3",
        "type": "directory",
        "used_for_week7_stage": "all reference",
        "critical": True,
    },
    {
        "input_name": "Raw Unibo dataset folder",
        "path": RAW_UNIBO,
        "type": "directory",
        "used_for_week7_stage": "tracking, statistics, scene-change detection",
        "critical": True,
    },
    {
        "input_name": "Unified ground truth CSV",
        "path": W6 / "outputs" / "unified_ground_truth" / "week6_unified_ground_truth_v2_with_recommended_split_v2.csv",
        "type": "csv",
        "used_for_week7_stage": "statistics, split, behaviour representation",
        "critical": True,
    },
    {
        "input_name": "Unified ground truth JSON",
        "path": W6 / "outputs" / "unified_ground_truth" / "week6_unified_ground_truth_v2_with_recommended_split_v2.json",
        "type": "json",
        "used_for_week7_stage": "visualization",
        "critical": True,
    },
    {
        "input_name": "Ground truth schema",
        "path": W6 / "outputs" / "unified_ground_truth" / "week6_unified_ground_truth_v2_with_recommended_split_v2_schema.json",
        "type": "json",
        "used_for_week7_stage": "documentation",
        "critical": False,
    },
    {
        "input_name": "Nested viewer JSON",
        "path": W6 / "outputs" / "unified_ground_truth" / "week6_scanpoint_annotations_nested_for_viewer.json",
        "type": "json",
        "used_for_week7_stage": "visualization",
        "critical": True,
    },
    {
        "input_name": "Scanpoint frame index",
        "path": W6 / "outputs" / "unified_ground_truth" / "week6_scanpoint_frame_index.csv",
        "type": "csv",
        "used_for_week7_stage": "visualization, ROI, colour assignment",
        "critical": True,
    },
    {
        "input_name": "Scanpoint frames directory",
        "path": W6 / "outputs" / "visual_label_check" / "all_scanpoint_frames",
        "type": "directory_jpg",
        "used_for_week7_stage": "ROI, colour assignment, visualization",
        "critical": True,
    },
    {
        "input_name": "YOLOv8-s detections with quality flags",
        "path": W6 / "outputs" / "feature_extractors" / "week6_yolov8s_all_scanpoint_detections_with_qc_flags.csv",
        "type": "csv",
        "used_for_week7_stage": "ROI, colour assignment, tracking bootstrap, segmentation",
        "critical": True,
    },
    {
        "input_name": "Detector-backed crop embeddings CSV",
        "path": W6 / "outputs" / "feature_extractors" / "week6_detector_backbone_crop_embeddings_896.csv",
        "type": "csv",
        "used_for_week7_stage": "feature representation",
        "critical": True,
    },
    {
        "input_name": "Detector-backed crop embeddings NPY",
        "path": W6 / "outputs" / "feature_extractors" / "week6_detector_backbone_crop_embeddings_896.npy",
        "type": "npy",
        "used_for_week7_stage": "feature representation",
        "critical": False,
    },
    {
        "input_name": "Classical segmentation features",
        "path": W6 / "outputs" / "feature_extractors" / "week6_preliminary_bbox_guided_segmentation_features.csv",
        "type": "csv",
        "used_for_week7_stage": "colour marker extraction, feature representation",
        "critical": True,
    },
    {
        "input_name": "Segment Anything Model segmentation features",
        "path": W6 / "outputs" / "feature_extractors" / "week6_sam_box_prompt_segmentation_features.csv",
        "type": "csv",
        "used_for_week7_stage": "colour marker extraction, feature representation",
        "critical": True,
    },
    {
        "input_name": "Segment Anything Model frame summary",
        "path": W6 / "outputs" / "feature_extractors" / "week6_sam_box_prompt_segmentation_frame_summary.csv",
        "type": "csv",
        "used_for_week7_stage": "visualization",
        "critical": False,
    },
    {
        "input_name": "Segment Anything Model masks directory",
        "path": W6 / "outputs" / "visual_label_check" / "sam_box_prompt_full_segmentation" / "masks",
        "type": "directory_png",
        "used_for_week7_stage": "colour marker extraction",
        "critical": True,
    },
    {
        "input_name": "Segment Anything Model overlays directory",
        "path": W6 / "outputs" / "visual_label_check" / "sam_box_prompt_full_segmentation" / "frame_overlays",
        "type": "directory_jpg",
        "used_for_week7_stage": "visualization",
        "critical": True,
    },
    {
        "input_name": "Week 6 presentation v2",
        "path": W6 / "final_presentation" / "Week6_Unibo_Dataset_Validation_Presentation_v2_with_Segment_Anything_Model.pdf",
        "type": "pdf",
        "used_for_week7_stage": "presentation reference",
        "critical": False,
    },
]

inventory_rows = []

for item in inputs:
    path = Path(item["path"])
    exists = path.exists()

    row = {
        "input_name": item["input_name"],
        "path": str(path),
        "type": item["type"],
        "used_for_week7_stage": item["used_for_week7_stage"],
        "critical": item["critical"],
        "exists": exists,
        "size_mb": file_size_mb(path),
        "csv_rows": csv_row_count(path),
        "directory_file_count": "",
        "status": "PASS" if exists else ("FAIL" if item["critical"] else "WARN"),
    }

    if exists and path.is_dir():
        if item["type"] == "directory_jpg":
            row["directory_file_count"] = dir_file_count(path, "*.jpg")
        elif item["type"] == "directory_png":
            row["directory_file_count"] = dir_file_count(path, "*.png")
        else:
            row["directory_file_count"] = dir_file_count(path)

    inventory_rows.append(row)

inventory = pd.DataFrame(inventory_rows)
inventory_path = W7 / "outputs" / "dataset_statistics" / "week7_w6_input_inventory.csv"
safe_to_csv(inventory, inventory_path)

critical_missing = inventory[(inventory["critical"] == True) & (inventory["exists"] == False)]
warnings = inventory[inventory["status"] == "WARN"]

summary = pd.DataFrame([
    {
        "metric": "generated_at",
        "value": datetime.now().isoformat(timespec="seconds"),
    },
    {
        "metric": "week7_root",
        "value": str(W7),
    },
    {
        "metric": "input_items_checked",
        "value": len(inventory),
    },
    {
        "metric": "critical_missing_count",
        "value": len(critical_missing),
    },
    {
        "metric": "warning_count",
        "value": len(warnings),
    },
    {
        "metric": "scanpoint_frame_count",
        "value": inventory.loc[inventory["input_name"] == "Scanpoint frames directory", "directory_file_count"].iloc[0],
    },
    {
        "metric": "sam_mask_count",
        "value": inventory.loc[inventory["input_name"] == "Segment Anything Model masks directory", "directory_file_count"].iloc[0],
    },
    {
        "metric": "sam_overlay_count",
        "value": inventory.loc[inventory["input_name"] == "Segment Anything Model overlays directory", "directory_file_count"].iloc[0],
    },
])

summary_path = W7 / "outputs" / "dataset_statistics" / "week7_setup_audit_summary.csv"
safe_to_csv(summary, summary_path)

gap_path = W7 / "outputs" / "dataset_statistics" / "week7_initial_gap_list.csv"
safe_to_csv(
    inventory[inventory["status"].isin(["FAIL", "WARN"])],
    gap_path,
)


# ---------------------------------------------------------------------
# README and notes.
# ---------------------------------------------------------------------
readme_path = W7 / "README.md"
readme_path.write_text(
    "# Week 7 Dataset Validation, Identity Tracking and Behaviour Preparation\n\n"
    "## Objective\n\n"
    "The main goal of Week 7 is to improve the reliability of the data pipeline before behaviour classification. "
    "The critical bottleneck is the association between pig, colour marker, stable identity, and behaviour label.\n\n"
    "## Main workflow\n\n"
    "1. Week 6 input audit.\n"
    "2. Region of Interest and crate mapping.\n"
    "3. Ground truth dataset statistical analysis.\n"
    "4. Colour marker rule definition.\n"
    "5. Scanpoint colour assignment baseline.\n"
    "6. Long-term tracking.\n"
    "7. Track ID and colour ID fusion.\n"
    "8. Improved visualization interface.\n"
    "9. Dataset splitting strategy analysis.\n"
    "10. External dataset validation.\n"
    "11. Behaviour feature representation analysis.\n"
    "12. Clip-level behaviour representation.\n"
    "13. Scene-change detection.\n"
    "14. Compliance and quality audit.\n"
    "15. Final package and presentation update.\n\n"
    "## Rule\n\n"
    "No major limitation should be silently left unresolved. If data, package, model weights, or human confirmation are needed, they must be explicitly requested and tracked.\n"
)

note_path = W7 / "notes" / "week7_setup_and_input_audit_notes.md"
with open(note_path, "w") as f:
    f.write("# Week 7 Setup and Week 6 Input Audit\n\n")

    f.write("## Purpose\n\n")
    f.write(
        "This step initializes the Week 7 project and verifies that the Week 6 outputs needed for identity assignment, "
        "colour matching, tracking, visualization, dataset statistics, split strategy, and clip-level representation are available.\n\n"
    )

    f.write("## Audit summary\n\n")
    f.write(summary.to_markdown(index=False))
    f.write("\n\n")

    f.write("## Critical missing inputs\n\n")
    if len(critical_missing):
        f.write(critical_missing.to_markdown(index=False))
    else:
        f.write("No critical Week 6 input is missing.")
    f.write("\n\n")

    f.write("## Warnings\n\n")
    if len(warnings):
        f.write(warnings.to_markdown(index=False))
    else:
        f.write("No warnings.")
    f.write("\n\n")

    f.write("## Next step\n\n")
    f.write(
        "If no critical input is missing, proceed to Region of Interest and crate mapping, followed by full ground truth dataset statistics.\n"
    )

print("Week 7 setup complete.")
print()
print("Week 7 root:")
print(W7)
print()
print("=== Setup audit summary ===")
print(summary.to_string(index=False))
print()
print("=== Critical missing inputs ===")
if len(critical_missing):
    print(critical_missing.to_string(index=False))
else:
    print("None")
print()
print("=== Warnings ===")
if len(warnings):
    print(warnings.to_string(index=False))
else:
    print("None")
print()
print("Saved:")
print(inventory_path)
print(summary_path)
print(gap_path)
print(note_path)
