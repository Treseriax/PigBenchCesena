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
INTERFACE = W6 / "interface_demo"
TRACKER = W6 / "shared_tracker"

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


def rel(path):
    try:
        return str(Path(path).relative_to(W6))
    except Exception:
        return str(path)


def exists(path):
    return Path(path).exists()


def count_csv(path):
    path = Path(path)
    if not path.exists() or path.suffix.lower() != ".csv":
        return ""
    try:
        return len(pd.read_csv(path))
    except Exception:
        return "read_error"


def evidence(paths):
    items = []
    for p in paths:
        p = Path(p)
        if p.exists():
            rows = count_csv(p)
            if rows != "":
                items.append(f"exists: {rel(p)} ({rows} rows)")
            else:
                items.append(f"exists: {rel(p)}")
        else:
            items.append(f"missing: {rel(p)}")
    return " | ".join(items)


def status_from_files(paths):
    if all(Path(p).exists() for p in paths):
        return "done"
    if any(Path(p).exists() for p in paths):
        return "partial"
    return "missing"


def xlsx_readable(path):
    path = Path(path)
    if not path.exists():
        return False, []
    try:
        xf = pd.ExcelFile(path)
        return True, xf.sheet_names
    except Exception:
        return False, []


# ------------------------------------------------------------
# Optional note cleanup before final package v2
# ------------------------------------------------------------
cleanup_replacements = {
    "andan explicit": "and an explicit",
    "and an explicit": "and an explicit",
    "requiredoutput": "required output",
    "notmanual": "not manual",
    "tocandidate": "to candidate",
    "posturerepresentation": "posture representation",
    "recommendedfile": "recommended file",
}

cleaned_notes = []

for note in NOTES.glob("*.md"):
    text = note.read_text(errors="ignore")
    original = text

    for old, new in cleanup_replacements.items():
        text = text.replace(old, new)

    if text != original:
        note.write_text(text)
        cleaned_notes.append(rel(note))


# ------------------------------------------------------------
# Required files
# ------------------------------------------------------------
files = {
    "dataset_overview": STATS / "week6_final_dataset_overview_metrics.csv",
    "behaviour_distribution": STATS / "week6_final_behaviour_distribution.csv",
    "rare_classes": STATS / "week6_final_rare_behaviour_classes.csv",
    "hourly_distribution": STATS / "week6_final_hourly_annotation_distribution.csv",
    "colour_distribution": STATS / "week6_final_colour_distribution.csv",
    "camera_pen_crate_summary": STATS / "week6_camera_pen_crate_video_summary.csv",

    "gt_csv": GT / "week6_unified_ground_truth_v2_with_recommended_split_v2.csv",
    "gt_json": GT / "week6_unified_ground_truth_v2_with_recommended_split_v2.json",
    "gt_schema": GT / "week6_unified_ground_truth_v2_with_recommended_split_v2_schema.json",
    "nested_viewer_json": GT / "week6_scanpoint_annotations_nested_for_viewer.json",
    "frame_index": GT / "week6_scanpoint_frame_index.csv",
    "frame_labels": GT / "week6_scanpoint_frame_labels_long.csv",

    "split_summary": STATS / "week6_recommended_split_v2_summary.csv",
    "split_video": STATS / "week6_recommended_split_v2_video_level.csv",
    "split_coverage": STATS / "week6_recommended_split_v2_behaviour_coverage.csv",
    "postfix_audit": STATS / "week6_postfix_final_consistency_audit.csv",

    "streamlit_app": INTERFACE / "week6_visualization_streamlit_app.py",
    "static_html_viewer": INTERFACE / "week6_static_visualization_viewer.html",
    "interface_index": INTERFACE / "week6_visualization_interface_index.csv",
    "marker_slideshow": VIS / "marker_bbox_overlay_v3_video/week6_marker_bbox_overlay_v3_slideshow.mp4",
    "bbox_contact_sheet": VIS / "bbox_count_warning_qc/all_bbox_count_warning_frames_contact_sheet.jpg",

    "det_qc": FEAT / "week6_yolov8s_all_scanpoint_detections_with_qc_flags.csv",
    "det_conservative": FEAT / "week6_yolov8s_conservative_detections_score_ge_0_50.csv",
    "marker_features": FEAT / "week6_crop_colour_marker_features.csv",
    "candidate_assignments": FEAT / "week6_candidate_bbox_to_colour_assignments.csv",
    "feature_comparison": FEAT / "week6_feature_extractor_comparison_final_clean_v3.csv",
    "feature_recommendations": FEAT / "week6_feature_extractor_recommendations_final_clean_v3.csv",
    "bbox_geometry": FEAT / "week6_bbox_geometry_features.csv",
    "group_spatial": FEAT / "week6_group_spatial_features_per_frame.csv",
    "roi_proxy": FEAT / "week6_coarse_roi_resource_proxy_features.csv",
    "crop_descriptor": FEAT / "week6_crop_descriptor_baseline_features.csv",
    "crop_descriptor_marker": FEAT / "week6_crop_descriptor_plus_marker_features.csv",
    "trajectory_report": FEAT / "week6_trajectory_feature_feasibility_report.csv",
    "lightweight_feature_summary": STATS / "week6_lightweight_feature_extractor_test_summary.csv",

    "seg_resource_audit": STATS / "week6_segmentation_resource_audit.csv",
    "seg_features": FEAT / "week6_preliminary_bbox_guided_segmentation_features.csv",
    "seg_features_json": FEAT / "week6_preliminary_bbox_guided_segmentation_features.json",
    "seg_frame_summary": FEAT / "week6_preliminary_segmentation_frame_summary.csv",
    "seg_quality": STATS / "week6_preliminary_segmentation_quality_summary.csv",
    "seg_status": STATS / "week6_preliminary_segmentation_status_summary.csv",
    "seg_contact_sheet": VIS / "week6_preliminary_segmentation_contact_sheet.jpg",

    "excel_tracker": TRACKER / "Week6_Unibo_Dataset_Validation_Shared_Task_Tracker.xlsx",
}

excel_ok, excel_sheets = xlsx_readable(files["excel_tracker"])

rows = []

def add(task_id, priority, requirement, required_keys, assessment, next_action):
    paths = [files[k] for k in required_keys]
    st = status_from_files(paths)

    rows.append({
        "task_id": task_id,
        "priority": priority,
        "requirement": requirement,
        "status": st,
        "current_assessment": assessment,
        "evidence": evidence(paths),
        "next_action": next_action if st != "done" else "No blocking action before final package v2.",
    })


add(
    "Task 1",
    "core",
    "Unibo dataset analysis: videos, cameras, pens/crates, scan windows, annotated frames, pigs, behaviour distribution, rare/missing labels.",
    [
        "dataset_overview",
        "behaviour_distribution",
        "rare_classes",
        "hourly_distribution",
        "colour_distribution",
        "camera_pen_crate_summary",
    ],
    "Dataset statistics and explicit camera/pen/crate/video summary are available.",
    "Create missing dataset statistics outputs.",
)

add(
    "Task 2",
    "core",
    "Extract labels from GT files and create unified table/JSON with video_id, crate_id, camera_id, frame_index, timestamp, pig_id/colour_id, bbox, behaviour_label, label_source.",
    [
        "gt_csv",
        "gt_json",
        "gt_schema",
        "nested_viewer_json",
        "frame_index",
        "frame_labels",
        "det_qc",
    ],
    "Recommended GT is available in CSV/JSON/schema formats; nested viewer JSON links frames, manual labels, and detector bboxes.",
    "Generate missing GT JSON/schema/viewer outputs.",
)

add(
    "Task 3",
    "core",
    "Dataset split protocol avoiding leakage and preserving behaviour coverage where possible.",
    [
        "split_summary",
        "split_video",
        "split_coverage",
        "postfix_audit",
    ],
    "Recommended split v2 is available with video-hour no-leakage protocol and final audit.",
    "Regenerate split protocol outputs.",
)

add(
    "Task 4",
    "core",
    "Visualization interface that loads video/frames and associated JSON annotation; displays bbox, pig colour/ID, behaviour label, timestamp/frame.",
    [
        "nested_viewer_json",
        "streamlit_app",
        "static_html_viewer",
        "interface_index",
        "marker_slideshow",
        "bbox_contact_sheet",
    ],
    "Streamlit app, static HTML viewer, index CSV, nested JSON, and overlay/demo visualizations are available.",
    "Create missing interface artifacts.",
)

add(
    "Task 5",
    "core",
    "Test feature extractors: trajectory, ROI/resource, group-spatial, detector/crop embeddings or descriptors, and feature comparison.",
    [
        "det_qc",
        "det_conservative",
        "marker_features",
        "candidate_assignments",
        "feature_comparison",
        "feature_recommendations",
        "bbox_geometry",
        "group_spatial",
        "roi_proxy",
        "crop_descriptor",
        "crop_descriptor_marker",
        "trajectory_report",
        "lightweight_feature_summary",
    ],
    "Feature extractor tests now include bbox geometry, group-spatial, ROI proxy, crop descriptors, marker features, conservative subset, and trajectory feasibility.",
    "Implement missing feature extractor outputs.",
)

add(
    "Task 6",
    "core",
    "Test at least one segmentation approach and evaluate preliminary shape/posture/colour-marker/ROI/contact features.",
    [
        "seg_resource_audit",
        "seg_features",
        "seg_features_json",
        "seg_frame_summary",
        "seg_quality",
        "seg_status",
        "seg_contact_sheet",
    ],
    "Preliminary bbox-guided GrabCut/Otsu segmentation baseline is available with features, JSON, summaries, and contact sheet.",
    "Generate preliminary segmentation outputs.",
)

add(
    "Deliverable 1",
    "deliverable",
    "Unibo dataset statistics report.",
    [
        "dataset_overview",
        "behaviour_distribution",
        "rare_classes",
        "camera_pen_crate_summary",
    ],
    "Dataset statistics outputs are available.",
    "Generate missing statistics report outputs.",
)

add(
    "Deliverable 2",
    "deliverable",
    "Unified GT extraction table/JSON.",
    [
        "gt_csv",
        "gt_json",
        "gt_schema",
        "nested_viewer_json",
    ],
    "GT CSV, JSON, schema, and nested viewer JSON are available.",
    "Generate missing GT outputs.",
)

add(
    "Deliverable 3",
    "deliverable",
    "Proposed train/test split protocol.",
    [
        "split_summary",
        "split_video",
        "split_coverage",
    ],
    "Recommended split v2 outputs are available.",
    "Generate missing split outputs.",
)

add(
    "Deliverable 4",
    "deliverable",
    "Visualization interface demo.",
    [
        "streamlit_app",
        "static_html_viewer",
        "interface_index",
        "nested_viewer_json",
    ],
    "Reusable Streamlit and static HTML viewer artifacts are available.",
    "Create missing viewer outputs.",
)

add(
    "Deliverable 5",
    "deliverable",
    "Screenshots/videos showing bbox + pig colour + behaviour label.",
    [
        "marker_slideshow",
        "bbox_contact_sheet",
    ],
    "Visualization slideshow and QC contact sheet are available.",
    "Create missing visualization outputs.",
)

add(
    "Deliverable 6",
    "deliverable",
    "Feature extractor comparison.",
    [
        "feature_comparison",
        "feature_recommendations",
        "lightweight_feature_summary",
    ],
    "Feature extractor comparison and expanded lightweight feature summary are available.",
    "Create missing feature comparison outputs.",
)

add(
    "Deliverable 7",
    "deliverable",
    "Preliminary segmentation results.",
    [
        "seg_features",
        "seg_features_json",
        "seg_quality",
        "seg_contact_sheet",
    ],
    "Preliminary segmentation results are available.",
    "Create missing segmentation outputs.",
)

# Deliverable 8 has extra xlsx readability condition.
tracker_status = "done" if files["excel_tracker"].exists() and excel_ok else ("partial" if files["excel_tracker"].exists() else "missing")

rows.append({
    "task_id": "Deliverable 8",
    "priority": "deliverable",
    "requirement": "Updated shared Excel file with tasks, experiments, outputs, and issues.",
    "status": tracker_status,
    "current_assessment": f"Shared Excel tracker exists and readable={excel_ok}. Sheets={excel_sheets}",
    "evidence": evidence([files["excel_tracker"]]),
    "next_action": "No blocking action before final package v2." if tracker_status == "done" else "Regenerate or repair Excel tracker.",
})

matrix = pd.DataFrame(rows)

status_order = {"missing": 0, "partial": 1, "done": 2}
matrix["status_rank"] = matrix["status"].map(status_order).fillna(1)
matrix = matrix.sort_values(["status_rank", "priority", "task_id"]).drop(columns=["status_rank"])

summary = (
    matrix.groupby("status")
    .size()
    .reset_index(name="count")
    .sort_values("status")
)

gaps = matrix[matrix["status"].isin(["missing", "partial"])].copy()

matrix_path = STATS / "week6_task_sheet_compliance_matrix_v2.csv"
summary_path = STATS / "week6_task_sheet_compliance_summary_v2.csv"
gaps_path = STATS / "week6_task_sheet_gap_list_v2.csv"

safe_to_csv(matrix, matrix_path)
safe_to_csv(summary, summary_path)
safe_to_csv(gaps, gaps_path)

# Detailed file verification table.
verification_rows = []

for key, path in files.items():
    row = {
        "key": key,
        "path": rel(path),
        "exists": path.exists(),
        "size_mb": round(path.stat().st_size / (1024 * 1024), 3) if path.exists() else "",
        "csv_rows": count_csv(path) if path.suffix.lower() == ".csv" and path.exists() else "",
    }

    if key == "excel_tracker":
        row["xlsx_readable"] = excel_ok
        row["xlsx_sheets"] = "; ".join(excel_sheets)
    else:
        row["xlsx_readable"] = ""
        row["xlsx_sheets"] = ""

    verification_rows.append(row)

verification = pd.DataFrame(verification_rows)
verification_path = STATS / "week6_final_v2_file_verification.csv"
safe_to_csv(verification, verification_path)

note_path = NOTES / "week6_final_task_sheet_compliance_audit_v2_notes.md"

with open(note_path, "w") as f:
    f.write("# Week 6 Final Task Sheet Compliance Audit v2\n\n")

    f.write("## Purpose\n\n")
    f.write(
        "This final compliance audit verifies that all task-sheet requirements and deliverables are covered before building final package v2.\n\n"
    )

    f.write("## Compliance summary\n\n")
    f.write(summary.to_markdown(index=False))
    f.write("\n\n")

    f.write("## Remaining gaps\n\n")
    if len(gaps):
        f.write(gaps[["task_id", "status", "requirement", "next_action"]].to_markdown(index=False))
    else:
        f.write("No missing or partial task-sheet items remain.")
    f.write("\n\n")

    f.write("## Compliance matrix\n\n")
    f.write(matrix.to_markdown(index=False))
    f.write("\n\n")

    f.write("## Excel tracker validation\n\n")
    f.write(f"- Excel tracker path: `{files['excel_tracker']}`\n")
    f.write(f"- Readable by pandas: `{excel_ok}`\n")
    f.write(f"- Sheets: `{'; '.join(excel_sheets)}`\n\n")

    f.write("## Cleaned notes\n\n")
    if cleaned_notes:
        for n in cleaned_notes:
            f.write(f"- `{n}`\n")
    else:
        f.write("No note typo cleanup was needed.")
    f.write("\n\n")

    f.write("## Interpretation\n\n")
    if len(gaps):
        f.write("Some items still require attention before final package v2.\n")
    else:
        f.write(
            "All required task-sheet items are now marked done. "
            "The project is ready for final package v2 construction and zip/checksum validation.\n"
        )

print("Saved:")
print(matrix_path)
print(summary_path)
print(gaps_path)
print(verification_path)
print(note_path)

print()
print("=== Compliance summary v2 ===")
print(summary.to_string(index=False))

print()
print("=== Remaining gaps v2 ===")
print(gaps[["task_id", "status", "requirement", "next_action"]].to_string(index=False) if len(gaps) else "None")

print()
print("=== Excel tracker readable ===")
print(excel_ok)
print(excel_sheets)

print()
print("=== Cleaned notes ===")
print(cleaned_notes if cleaned_notes else "None")
