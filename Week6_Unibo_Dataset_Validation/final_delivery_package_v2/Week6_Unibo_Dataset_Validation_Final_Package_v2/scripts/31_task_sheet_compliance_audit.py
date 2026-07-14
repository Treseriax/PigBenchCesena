from pathlib import Path
import csv
import pandas as pd


ROOT = Path.home() / "PigBench"
W6 = ROOT / "Week6_Unibo_Dataset_Validation"

OUT = W6 / "outputs"
STATS = OUT / "dataset_statistics"
GT = OUT / "unified_ground_truth"
FEAT = OUT / "feature_extractors"
VIS = OUT / "visual_label_check"
NOTES = W6 / "notes"
SCRIPTS = W6 / "scripts"

STATS.mkdir(parents=True, exist_ok=True)
NOTES.mkdir(parents=True, exist_ok=True)


def safe_to_csv(df, path):
    df.to_csv(
        path,
        index=False,
        quoting=csv.QUOTE_ALL,
        escapechar="\\",
        lineterminator="\n"
    )


def exists(path):
    return Path(path).exists()


def rel(path):
    try:
        return str(Path(path).relative_to(W6))
    except Exception:
        return str(path)


def count_csv(path):
    path = Path(path)
    if not path.exists():
        return ""
    try:
        return len(pd.read_csv(path))
    except Exception:
        return "read_error"


def status_from_requirements(required_paths):
    missing = [p for p in required_paths if not Path(p).exists()]
    if len(missing) == 0:
        return "done"
    if len(missing) < len(required_paths):
        return "partial"
    return "missing"


def evidence(paths):
    out = []
    for p in paths:
        p = Path(p)
        tag = "exists" if p.exists() else "missing"
        rows = count_csv(p) if p.suffix.lower() == ".csv" and p.exists() else ""
        if rows != "":
            out.append(f"{tag}: {rel(p)} ({rows} rows)")
        else:
            out.append(f"{tag}: {rel(p)}")
    return " | ".join(out)


# Core files
dataset_overview = STATS / "week6_final_dataset_overview_metrics.csv"
behaviour_dist = STATS / "week6_final_behaviour_distribution.csv"
rare_classes = STATS / "week6_final_rare_behaviour_classes.csv"
hourly_dist = STATS / "week6_final_hourly_annotation_distribution.csv"
colour_dist = STATS / "week6_final_colour_distribution.csv"

gt_recommended_csv = GT / "week6_unified_ground_truth_v2_with_recommended_split_v2.csv"
gt_recommended_json = GT / "week6_unified_ground_truth_v2_with_recommended_split_v2.json"
gt_v2_json = GT / "week6_unified_ground_truth_v2_with_recovered_videos.json"

frame_index = GT / "week6_scanpoint_frame_index.csv"
frame_labels = GT / "week6_scanpoint_frame_labels_long.csv"

split_summary = STATS / "week6_recommended_split_v2_summary.csv"
split_video = STATS / "week6_recommended_split_v2_video_level.csv"
split_coverage = STATS / "week6_recommended_split_v2_behaviour_coverage.csv"

marker_video = VIS / "marker_bbox_overlay_v3_video/week6_marker_bbox_overlay_v3_slideshow.mp4"
bbox_warning_sheet = VIS / "bbox_count_warning_qc/all_bbox_count_warning_frames_contact_sheet.jpg"
bbox_label_overlay_dir = VIS / "bbox_label_overlay_v2"
marker_overlay_dir = VIS / "marker_bbox_overlay_v3"

det_qc = FEAT / "week6_yolov8s_all_scanpoint_detections_with_qc_flags.csv"
det_conservative = FEAT / "week6_yolov8s_conservative_detections_score_ge_0_50.csv"
marker_features = FEAT / "week6_crop_colour_marker_features.csv"
candidate_assignments = FEAT / "week6_candidate_bbox_to_colour_assignments.csv"
feature_comparison = FEAT / "week6_feature_extractor_comparison_final_clean_v3.csv"
feature_recommendations = FEAT / "week6_feature_extractor_recommendations_final_clean_v3.csv"

postfix_audit = STATS / "week6_postfix_final_consistency_audit.csv"

# Search for possible interface / segmentation / Excel tracker files.
interface_candidates = []
for pattern in ["*viewer*.py", "*interface*.py", "*streamlit*.py", "*app*.py", "*.html"]:
    interface_candidates.extend(list(W6.rglob(pattern)))

segmentation_candidates = []
for pattern in ["*segment*.csv", "*segment*.json", "*segmentation*.csv", "*segmentation*.json", "*mask*.csv", "*mask*.json", "*seg*.jpg", "*segmentation*.jpg", "*mask*.jpg"]:
    segmentation_candidates.extend(list(OUT.rglob(pattern)))

excel_tracker_candidates = []
for pattern in ["*tracker*.xlsx", "*task*.xlsx", "*experiment*.xlsx", "*shared*.xlsx", "*Week6*.xlsx"]:
    excel_tracker_candidates.extend(list(W6.rglob(pattern)))

# Avoid counting unrelated scripts as a real interface if they are just previous pipeline scripts.
interface_candidates = sorted(set(interface_candidates))
segmentation_candidates = sorted(set(segmentation_candidates))
excel_tracker_candidates = sorted(set(excel_tracker_candidates))

rows = []

def add(task_id, requirement, required_paths, current_assessment, next_action, priority="core"):
    st = status_from_requirements(required_paths)
    rows.append({
        "task_id": task_id,
        "priority": priority,
        "requirement": requirement,
        "status": st,
        "current_assessment": current_assessment,
        "evidence": evidence(required_paths),
        "next_action": next_action,
    })


# Task 1
add(
    "Task 1",
    "Unibo dataset analysis: videos, cameras, pens/crates, scan windows, annotated frames, pigs, behaviour distribution, rare/missing labels.",
    [dataset_overview, behaviour_dist, rare_classes, hourly_dist, colour_dist],
    "Dataset statistics are mostly complete. Need one extra dedicated crate/pen/camera summary table to make the task sheet mapping explicit.",
    "Create final crate/pen/camera/video summary table and add it to v2 package.",
)

# Task 2
task2_paths = [gt_recommended_csv, frame_index, frame_labels]
task2_status = status_from_requirements(task2_paths)

rows.append({
    "task_id": "Task 2",
    "priority": "core",
    "requirement": "Extract labels from all GT files and create unified table/JSON with video_id, crate_id, camera_id, frame_index, timestamp, pig_id/colour_id, bbox, behaviour_label, label_source.",
    "status": "partial" if not gt_recommended_json.exists() else "done",
    "current_assessment": (
        "Recommended CSV exists and preserves 432 manual labels. Recommended JSON is not yet generated. "
        "BBox fields in GT are not manual bbox ground truth; detector bbox table exists separately."
    ),
    "evidence": evidence([gt_recommended_csv, gt_recommended_json, gt_v2_json, det_qc]),
    "next_action": "Generate recommended split v2 JSON and schema note; keep detector bbox table separate but linkable.",
})

# Task 3
add(
    "Task 3",
    "Dataset splitting protocol that avoids leakage and represents behaviours when possible.",
    [split_summary, split_video, split_coverage, postfix_audit],
    "Recommended split v2 is complete: train/val/test at video-hour level, no leakage, improved behaviour coverage.",
    "No major action. Include split protocol in final report/package v2.",
)

# Task 4
rows.append({
    "task_id": "Task 4",
    "priority": "core",
    "requirement": "Visualization interface that loads Unibo video and associated JSON annotation; displays bbox, pig colour/ID, behaviour label, timestamp/frame.",
    "status": "partial" if marker_video.exists() else "missing",
    "current_assessment": (
        "Overlay videos and images exist, but a minimal reusable interface/script that loads video + JSON annotation is not yet clearly packaged."
    ),
    "evidence": evidence([marker_video, bbox_warning_sheet]) + " | interface_candidates=" + ", ".join(rel(p) for p in interface_candidates[:10]),
    "next_action": "Create minimal Streamlit or HTML/OpenCV viewer using recommended JSON and scanpoint frames.",
})

# Task 5
rows.append({
    "task_id": "Task 5",
    "priority": "core",
    "requirement": "Test different feature extractors: trajectory, ROI/resource, group-spatial, YOLO detector embeddings, crop-based embeddings; optional DINOv2/VideoMAE/keypoints.",
    "status": "partial",
    "current_assessment": (
        "Implemented: bbox geometry/QC, conservative bbox subset, crop colour-marker features, candidate colour association, comparison table. "
        "Missing or only conceptual: trajectory features, ROI/resource features, group-spatial features, YOLO detector embeddings, crop-based embeddings."
    ),
    "evidence": evidence([det_qc, det_conservative, marker_features, candidate_assignments, feature_comparison, feature_recommendations]),
    "next_action": "Implement lightweight bbox geometry + group-spatial + ROI/resource + crop descriptor baseline tables. Document trajectory limitations due unresolved identity.",
})

# Task 6
rows.append({
    "task_id": "Task 6",
    "priority": "core",
    "requirement": "Test at least one segmentation approach and evaluate whether segmentation improves shape/posture/colour-marker/ROI/contact estimation.",
    "status": "missing" if len(segmentation_candidates) == 0 else "partial",
    "current_assessment": (
        "No clear preliminary segmentation output found in Week 6 outputs. Need at least a baseline segmentation test or documented fallback."
    ),
    "evidence": "segmentation_candidates=" + (", ".join(rel(p) for p in segmentation_candidates[:20]) if segmentation_candidates else "none found"),
    "next_action": "Audit available segmentation models. If no YOLO-seg/SAM is installed, create bbox-guided preliminary mask baseline and clearly label it as preliminary.",
})

# Deliverables
deliverables = [
    ("Deliverable 1", "Unibo dataset statistics report.", [dataset_overview, behaviour_dist, rare_classes], "done", "Already covered by dataset statistics and final outline.", "No major action."),
    ("Deliverable 2", "Unified GT extraction table/JSON.", [gt_recommended_csv, gt_recommended_json], "partial", "CSV exists; recommended JSON missing.", "Generate recommended JSON."),
    ("Deliverable 3", "Proposed train/test split protocol.", [split_summary, split_video, split_coverage], "done", "Recommended split v2 exists.", "No major action."),
    ("Deliverable 4", "Visualization interface demo.", [marker_video], "partial", "Video demo exists; reusable interface file not yet packaged.", "Create minimal viewer demo."),
    ("Deliverable 5", "Screenshots/videos showing bbox + pig colour + behaviour label.", [marker_video, bbox_warning_sheet], "done", "Visualization video and QC contact sheets exist.", "No major action."),
    ("Deliverable 6", "Feature extractor comparison.", [feature_comparison, feature_recommendations], "done", "Comparison table exists, but implementations can be expanded.", "Add extra lightweight feature tables."),
    ("Deliverable 7", "Preliminary segmentation results.", [], "missing", "No clear segmentation output yet.", "Add segmentation baseline/test."),
    ("Deliverable 8", "Updated shared Excel file with tasks, experiments, outputs, and issues.", [], "missing", "No Excel tracker found.", "Generate Week6 task/experiment tracker .xlsx."),
]

for did, req, paths, manual_status, assessment, action in deliverables:
    # preserve manual status for missing empty-path deliverables
    rows.append({
        "task_id": did,
        "priority": "deliverable",
        "requirement": req,
        "status": manual_status if len(paths) == 0 else ("done" if all(Path(p).exists() for p in paths) else manual_status),
        "current_assessment": assessment,
        "evidence": evidence(paths) if paths else "no required output found yet",
        "next_action": action,
    })

matrix = pd.DataFrame(rows)

# Sort status severity
status_order = {"missing": 0, "partial": 1, "done": 2}
matrix["status_rank"] = matrix["status"].map(status_order).fillna(1)
matrix = matrix.sort_values(["status_rank", "priority", "task_id"]).drop(columns=["status_rank"])

matrix_path = STATS / "week6_task_sheet_compliance_matrix.csv"
safe_to_csv(matrix, matrix_path)

gaps = matrix[matrix["status"].isin(["missing", "partial"])].copy()
gaps_path = STATS / "week6_task_sheet_gap_list.csv"
safe_to_csv(gaps, gaps_path)

summary = (
    matrix.groupby("status")
    .size()
    .reset_index(name="count")
    .sort_values("status")
)

summary_path = STATS / "week6_task_sheet_compliance_summary.csv"
safe_to_csv(summary, summary_path)

note_path = NOTES / "week6_task_sheet_compliance_audit_notes.md"

with open(note_path, "w") as f:
    f.write("# Week 6 Task Sheet Compliance Audit\n\n")

    f.write("## Purpose\n\n")
    f.write(
        "This audit maps the Week 6 assignment requirements to the current pipeline outputs. "
        "It is used to decide what still needs to be implemented before creating the final package v2.\n\n"
    )

    f.write("## Compliance summary\n\n")
    f.write(summary.to_markdown(index=False))
    f.write("\n\n")

    f.write("## Compliance matrix\n\n")
    f.write(matrix.to_markdown(index=False))
    f.write("\n\n")

    f.write("## Gaps to close before final package v2\n\n")
    if len(gaps):
        f.write(gaps[["task_id", "status", "requirement", "next_action"]].to_markdown(index=False))
    else:
        f.write("No gaps found.")
    f.write("\n\n")

    f.write("## Recommended next steps\n\n")
    f.write(
        "1. Generate recommended GT JSON and schema note.\n"
        "2. Create explicit crate/pen/camera summary table.\n"
        "3. Create minimal visualization interface demo.\n"
        "4. Add lightweight feature extractor test tables: bbox geometry, ROI/resource, group-spatial, crop descriptors.\n"
        "5. Add preliminary segmentation baseline or documented segmentation model audit.\n"
        "6. Generate updated shared Excel tracker.\n"
        "7. Build final package v2 and rerun zip/checksum validation.\n"
    )

print("Saved:")
print(matrix_path)
print(gaps_path)
print(summary_path)
print(note_path)

print()
print("=== Compliance summary ===")
print(summary.to_string(index=False))

print()
print("=== Gaps ===")
print(gaps[["task_id", "status", "requirement", "next_action"]].to_string(index=False) if len(gaps) else "No gaps.")
