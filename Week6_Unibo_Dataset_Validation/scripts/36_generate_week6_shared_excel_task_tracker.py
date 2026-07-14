from pathlib import Path
import csv
import pandas as pd
from datetime import datetime


ROOT = Path.home() / "PigBench"
W6 = ROOT / "Week6_Unibo_Dataset_Validation"

STATS = W6 / "outputs/dataset_statistics"
GT = W6 / "outputs/unified_ground_truth"
FEAT = W6 / "outputs/feature_extractors"
VIS = W6 / "outputs/visual_label_check"
NOTES = W6 / "notes"
TRACKER_DIR = W6 / "shared_tracker"

TRACKER_DIR.mkdir(parents=True, exist_ok=True)
STATS.mkdir(parents=True, exist_ok=True)
NOTES.mkdir(parents=True, exist_ok=True)


def read_csv(path):
    path = Path(path)
    if not path.exists():
        return pd.DataFrame()
    return pd.read_csv(path)


def exists(path):
    return Path(path).exists()


def rel(path):
    try:
        return str(Path(path).relative_to(W6))
    except Exception:
        return str(path)


def safe_to_csv(df, path):
    df.to_csv(
        path,
        index=False,
        quoting=csv.QUOTE_ALL,
        escapechar="\\",
        lineterminator="\n"
    )


# Input tables.
compliance = read_csv(STATS / "week6_task_sheet_compliance_matrix.csv")
deliverable_inventory = read_csv(STATS / "week6_final_deliverable_inventory.csv")
feature_summary = read_csv(STATS / "week6_lightweight_feature_extractor_test_summary.csv")
seg_quality = read_csv(STATS / "week6_preliminary_segmentation_quality_summary.csv")
split_summary = read_csv(STATS / "week6_recommended_split_v2_summary.csv")
dataset_overview = read_csv(STATS / "week6_final_dataset_overview_metrics.csv")

# ------------------------------------------------------------
# Overview sheet
# ------------------------------------------------------------
overview_rows = [
    {"metric": "project", "value": "Week 6 Unibo Dataset Validation"},
    {"metric": "generated_at", "value": datetime.now().isoformat(timespec="seconds")},
    {"metric": "final_manual_labels", "value": 432},
    {"metric": "scanpoint_frames", "value": 72},
    {"metric": "manual_labels_per_frame", "value": 6},
    {"metric": "yolov8s_primary_bboxes", "value": 540},
    {"metric": "medium_high_marker_candidates", "value": 196},
    {"metric": "recommended_split_v2_train_labels", "value": 288},
    {"metric": "recommended_split_v2_val_labels", "value": 72},
    {"metric": "recommended_split_v2_test_labels", "value": 72},
    {"metric": "preliminary_segmentation_records", "value": 540},
    {"metric": "final_package_v1_missing_files", "value": 0},
    {"metric": "current_goal", "value": "Close all task-sheet gaps and build final package v2."},
]

overview = pd.DataFrame(overview_rows)

# ------------------------------------------------------------
# Task compliance sheet
# Update statuses based on newly completed steps.
# ------------------------------------------------------------
if compliance.empty:
    compliance = pd.DataFrame(columns=[
        "task_id", "priority", "requirement", "status",
        "current_assessment", "evidence", "next_action"
    ])

updated = compliance.copy()

def set_status_contains(task_contains, status, assessment_append, next_action):
    mask = updated["task_id"].astype(str).str.contains(task_contains, case=False, regex=False)
    if mask.any():
        updated.loc[mask, "status"] = status
        updated.loc[mask, "current_assessment"] = (
            updated.loc[mask, "current_assessment"].astype(str)
            + " " + assessment_append
        )
        updated.loc[mask, "next_action"] = next_action

set_status_contains(
    "Task 2",
    "done",
    "Recommended GT JSON, schema JSON, nested viewer JSON, and camera/pen/crate/video summary have been generated.",
    "Include new JSON/schema outputs in final package v2."
)

set_status_contains(
    "Deliverable 2",
    "done",
    "Recommended GT JSON has been generated.",
    "Include CSV and JSON in final package v2."
)

set_status_contains(
    "Task 4",
    "done",
    "Streamlit app and static HTML visualization viewer have been generated.",
    "Include interface_demo directory in final package v2."
)

set_status_contains(
    "Deliverable 4",
    "done",
    "Reusable Streamlit and static HTML viewer artifacts have been generated.",
    "Include interface demo in final package v2."
)

set_status_contains(
    "Task 5",
    "done",
    "Additional lightweight feature extractor tables have been generated: bbox geometry, group-spatial, coarse ROI/resource proxy, crop descriptors, and trajectory feasibility.",
    "Include lightweight feature extractor outputs in final package v2."
)

set_status_contains(
    "Task 6",
    "done",
    "Preliminary bbox-guided segmentation baseline has been implemented with 540 segmentation records and 72 overlay images.",
    "Include preliminary segmentation outputs in final package v2."
)

set_status_contains(
    "Deliverable 7",
    "done",
    "Preliminary segmentation CSV/JSON, quality summaries, overlays, and contact sheet have been generated.",
    "Include preliminary segmentation outputs in final package v2."
)

set_status_contains(
    "Deliverable 8",
    "done",
    "Updated shared Excel/task tracker has been generated.",
    "Include tracker XLSX in final package v2."
)

# ------------------------------------------------------------
# Experiments sheet
# ------------------------------------------------------------
experiments_rows = [
    {
        "experiment_id": "E01",
        "experiment": "Raw Unibo video inventory",
        "status": "completed",
        "primary_output": "outputs/dataset_statistics/week6_final_dataset_overview_metrics.csv",
        "records_or_count": "84 raw work videos",
        "notes": "Raw videos and naming/camera metadata analyzed.",
    },
    {
        "experiment_id": "E02",
        "experiment": "Excel annotation parsing",
        "status": "completed",
        "primary_output": "outputs/unified_ground_truth/week6_unified_ground_truth_v2_with_recommended_split_v2.csv",
        "records_or_count": "432 labels",
        "notes": "Manual scan-sampling labels parsed and linked to videos.",
    },
    {
        "experiment_id": "E03",
        "experiment": "Recommended GT JSON/schema",
        "status": "completed",
        "primary_output": "outputs/unified_ground_truth/week6_unified_ground_truth_v2_with_recommended_split_v2.json",
        "records_or_count": "432 records",
        "notes": "JSON, schema, nested viewer JSON, and metadata summary generated.",
    },
    {
        "experiment_id": "E04",
        "experiment": "Scanpoint frame extraction",
        "status": "completed",
        "primary_output": "outputs/unified_ground_truth/week6_scanpoint_frame_index.csv",
        "records_or_count": "72 frames",
        "notes": "Each scanpoint frame links to six manual labels.",
    },
    {
        "experiment_id": "E05",
        "experiment": "YOLOv8-s bbox detection",
        "status": "completed",
        "primary_output": "outputs/feature_extractors/week6_yolov8s_all_scanpoint_detections_with_qc_flags.csv",
        "records_or_count": "540 bboxes",
        "notes": "Automatic detector bboxes with QC flags.",
    },
    {
        "experiment_id": "E06",
        "experiment": "Crop colour-marker features",
        "status": "completed",
        "primary_output": "outputs/feature_extractors/week6_crop_colour_marker_features.csv",
        "records_or_count": "540 crops, 196 medium/high candidates",
        "notes": "Candidate-level identity/colour evidence only.",
    },
    {
        "experiment_id": "E07",
        "experiment": "Recommended split v2",
        "status": "completed",
        "primary_output": "outputs/dataset_statistics/week6_recommended_split_v2_summary.csv",
        "records_or_count": "288/72/72 labels",
        "notes": "Video-hour no-leakage split.",
    },
    {
        "experiment_id": "E08",
        "experiment": "Visualization interface demo",
        "status": "completed",
        "primary_output": "interface_demo/week6_visualization_streamlit_app.py",
        "records_or_count": "72 frames / 432 labels / 540 bboxes",
        "notes": "Streamlit and static HTML viewer generated.",
    },
    {
        "experiment_id": "E09",
        "experiment": "Lightweight feature extractor tests",
        "status": "completed",
        "primary_output": "outputs/dataset_statistics/week6_lightweight_feature_extractor_test_summary.csv",
        "records_or_count": "bbox/group/ROI/crop/trajectory tables",
        "notes": "Task 5 expanded beyond detector/marker features.",
    },
    {
        "experiment_id": "E10",
        "experiment": "Preliminary segmentation baseline",
        "status": "completed",
        "primary_output": "outputs/feature_extractors/week6_preliminary_bbox_guided_segmentation_features.csv",
        "records_or_count": "540 segmentation records",
        "notes": "GrabCut + Otsu fallback; not manual segmentation GT.",
    },
    {
        "experiment_id": "E11",
        "experiment": "Final package v1",
        "status": "completed",
        "primary_output": "final_delivery_package/Week6_Unibo_Dataset_Validation_Final_Package.zip",
        "records_or_count": "29 files, 0 missing",
        "notes": "Zip and checksum verified; v2 will include additional task-sheet outputs.",
    },
]

experiments = pd.DataFrame(experiments_rows)

# ------------------------------------------------------------
# Key outputs sheet
# ------------------------------------------------------------
key_output_paths = [
    "outputs/unified_ground_truth/week6_unified_ground_truth_v2_with_recommended_split_v2.csv",
    "outputs/unified_ground_truth/week6_unified_ground_truth_v2_with_recommended_split_v2.json",
    "outputs/unified_ground_truth/week6_unified_ground_truth_v2_with_recommended_split_v2_schema.json",
    "outputs/unified_ground_truth/week6_scanpoint_annotations_nested_for_viewer.json",
    "outputs/dataset_statistics/week6_camera_pen_crate_video_summary.csv",
    "interface_demo/week6_visualization_streamlit_app.py",
    "interface_demo/week6_static_visualization_viewer.html",
    "outputs/feature_extractors/week6_bbox_geometry_features.csv",
    "outputs/feature_extractors/week6_group_spatial_features_per_frame.csv",
    "outputs/feature_extractors/week6_coarse_roi_resource_proxy_features.csv",
    "outputs/feature_extractors/week6_crop_descriptor_baseline_features.csv",
    "outputs/feature_extractors/week6_crop_descriptor_plus_marker_features.csv",
    "outputs/feature_extractors/week6_trajectory_feature_feasibility_report.csv",
    "outputs/feature_extractors/week6_preliminary_bbox_guided_segmentation_features.csv",
    "outputs/feature_extractors/week6_preliminary_bbox_guided_segmentation_features.json",
    "outputs/feature_extractors/week6_preliminary_segmentation_frame_summary.csv",
    "outputs/dataset_statistics/week6_preliminary_segmentation_quality_summary.csv",
    "outputs/dataset_statistics/week6_preliminary_segmentation_status_summary.csv",
    "outputs/visual_label_check/week6_preliminary_segmentation_contact_sheet.jpg",
    "outputs/visual_label_check/marker_bbox_overlay_v3_video/week6_marker_bbox_overlay_v3_slideshow.mp4",
]

key_outputs_rows = []

for p in key_output_paths:
    full = W6 / p
    key_outputs_rows.append({
        "path": p,
        "exists": full.exists(),
        "size_mb": round(full.stat().st_size / (1024 * 1024), 3) if full.exists() else "",
        "type": full.suffix.lower().replace(".", ""),
        "package_v2_include": "yes",
    })

key_outputs = pd.DataFrame(key_outputs_rows)

# ------------------------------------------------------------
# Issues and limitations sheet
# ------------------------------------------------------------
issues_rows = [
    {
        "issue_id": "L01",
        "category": "annotation_scope",
        "severity": "medium",
        "status": "documented",
        "issue": "Labels come from one Excel sheet/day/camera-pen context.",
        "mitigation": "Clearly report scope and avoid over-generalizing results.",
    },
    {
        "issue_id": "L02",
        "category": "video_linkage",
        "severity": "medium",
        "status": "documented",
        "issue": "15:00-19:00 c-token video mappings are candidate recovered mappings.",
        "mitigation": "Keep video_mapping_confidence as medium_needs_visual_confirmation.",
    },
    {
        "issue_id": "L03",
        "category": "bbox_ground_truth",
        "severity": "medium",
        "status": "documented",
        "issue": "Bboxes are automatic detector outputs, not manual bbox annotations.",
        "mitigation": "Keep detector bboxes in feature extractor tables and separate from manual GT labels.",
    },
    {
        "issue_id": "L04",
        "category": "identity",
        "severity": "high",
        "status": "documented",
        "issue": "BBox-to-colour identity is candidate-level only.",
        "mitigation": "Use marker assignments for QC/candidate analysis, not final identity-resolved GT.",
    },
    {
        "issue_id": "L05",
        "category": "marker_colour",
        "severity": "medium",
        "status": "documented",
        "issue": "Red markers are ambiguous between red_neck and red_tail.",
        "mitigation": "Do not claim final red_neck/red_tail assignment from red marker alone.",
    },
    {
        "issue_id": "L06",
        "category": "segmentation",
        "severity": "medium",
        "status": "documented",
        "issue": "Preliminary segmentation masks are automatic and not manual segmentation GT.",
        "mitigation": "Use as exploratory shape/posture features only; validate manually before label claims.",
    },
    {
        "issue_id": "L07",
        "category": "trajectory",
        "severity": "high",
        "status": "documented",
        "issue": "Stable identity-resolved tracking is not validated.",
        "mitigation": "Do not claim final per-pig trajectory features; document feasibility only.",
    },
    {
        "issue_id": "L08",
        "category": "class_balance",
        "severity": "medium",
        "status": "documented",
        "issue": "Rare behaviour classes remain limited.",
        "mitigation": "Use careful evaluation, avoid overclaiming performance on rare classes.",
    },
]

issues = pd.DataFrame(issues_rows)

# ------------------------------------------------------------
# Next steps sheet
# ------------------------------------------------------------
next_steps_rows = [
    {
        "step_id": "N01",
        "priority": "high",
        "status": "next",
        "task": "Re-run compliance audit after new outputs.",
        "expected_output": "updated compliance matrix with all required deliverables done.",
    },
    {
        "step_id": "N02",
        "priority": "high",
        "status": "next",
        "task": "Build final package v2 including JSON, interface, extra features, segmentation, tracker.",
        "expected_output": "Final_Package_v2.zip",
    },
    {
        "step_id": "N03",
        "priority": "high",
        "status": "next",
        "task": "Run zip integrity and checksum verification for final package v2.",
        "expected_output": "No errors and all checksums OK.",
    },
    {
        "step_id": "N04",
        "priority": "medium",
        "status": "future",
        "task": "Manually inspect segmentation overlays and mark low-quality masks.",
        "expected_output": "segmentation visual QC table.",
    },
    {
        "step_id": "N05",
        "priority": "medium",
        "status": "future",
        "task": "Add validated tracker/identity association if needed.",
        "expected_output": "identity-resolved trajectory features.",
    },
]

next_steps = pd.DataFrame(next_steps_rows)

# ------------------------------------------------------------
# Write CSV companion files
# ------------------------------------------------------------
safe_to_csv(overview, STATS / "week6_shared_tracker_overview.csv")
safe_to_csv(updated, STATS / "week6_shared_tracker_task_compliance_updated.csv")
safe_to_csv(experiments, STATS / "week6_shared_tracker_experiments.csv")
safe_to_csv(key_outputs, STATS / "week6_shared_tracker_key_outputs.csv")
safe_to_csv(issues, STATS / "week6_shared_tracker_issues_limitations.csv")
safe_to_csv(next_steps, STATS / "week6_shared_tracker_next_steps.csv")

# ------------------------------------------------------------
# Write XLSX
# ------------------------------------------------------------
xlsx_path = TRACKER_DIR / "Week6_Unibo_Dataset_Validation_Shared_Task_Tracker.xlsx"

sheets = {
    "Overview": overview,
    "Task Compliance": updated,
    "Experiments": experiments,
    "Key Outputs": key_outputs,
    "Issues Limitations": issues,
    "Next Steps": next_steps,
}

# Try xlsxwriter first for formatting. Pandas will raise if unavailable.
try:
    writer = pd.ExcelWriter(xlsx_path, engine="xlsxwriter")
    engine = "xlsxwriter"

    with writer:
        workbook = writer.book

        title_fmt = workbook.add_format({
            "bold": True,
            "font_size": 14,
            "font_color": "white",
            "bg_color": "#1F4E78",
            "align": "center",
            "valign": "vcenter",
        })

        header_fmt = workbook.add_format({
            "bold": True,
            "font_color": "white",
            "bg_color": "#5B9BD5",
            "border": 1,
            "text_wrap": True,
            "valign": "top",
        })

        body_fmt = workbook.add_format({
            "border": 1,
            "valign": "top",
            "text_wrap": True,
        })

        green_fmt = workbook.add_format({"bg_color": "#C6EFCE", "font_color": "#006100"})
        yellow_fmt = workbook.add_format({"bg_color": "#FFEB9C", "font_color": "#9C6500"})
        red_fmt = workbook.add_format({"bg_color": "#FFC7CE", "font_color": "#9C0006"})

        for sheet_name, df in sheets.items():
            df.to_excel(writer, sheet_name=sheet_name, index=False, startrow=2)
            ws = writer.sheets[sheet_name]

            ws.merge_range(0, 0, 0, max(len(df.columns) - 1, 0), f"Week 6 - {sheet_name}", title_fmt)

            for col_idx, col_name in enumerate(df.columns):
                ws.write(2, col_idx, col_name, header_fmt)

                series = df[col_name].astype(str)
                max_len = max([len(str(col_name))] + [len(x) for x in series.head(200)])
                width = min(max(max_len + 2, 12), 42)
                ws.set_column(col_idx, col_idx, width, body_fmt)

            ws.freeze_panes(3, 0)
            ws.autofilter(2, 0, max(len(df) + 2, 2), max(len(df.columns) - 1, 0))

            # Conditional formatting for status-like columns.
            for col_idx, col_name in enumerate(df.columns):
                if col_name.lower() in ["status", "exists", "package_v2_include"]:
                    first_row = 3
                    last_row = max(len(df) + 2, 3)
                    ws.conditional_format(first_row, col_idx, last_row, col_idx, {
                        "type": "text",
                        "criteria": "containing",
                        "value": "done",
                        "format": green_fmt,
                    })
                    ws.conditional_format(first_row, col_idx, last_row, col_idx, {
                        "type": "text",
                        "criteria": "containing",
                        "value": "completed",
                        "format": green_fmt,
                    })
                    ws.conditional_format(first_row, col_idx, last_row, col_idx, {
                        "type": "text",
                        "criteria": "containing",
                        "value": "True",
                        "format": green_fmt,
                    })
                    ws.conditional_format(first_row, col_idx, last_row, col_idx, {
                        "type": "text",
                        "criteria": "containing",
                        "value": "partial",
                        "format": yellow_fmt,
                    })
                    ws.conditional_format(first_row, col_idx, last_row, col_idx, {
                        "type": "text",
                        "criteria": "containing",
                        "value": "missing",
                        "format": red_fmt,
                    })

except Exception as e:
    # Fallback: simple Excel file.
    engine = f"fallback_auto ({type(e).__name__}: {e})"
    with pd.ExcelWriter(xlsx_path) as writer:
        for sheet_name, df in sheets.items():
            df.to_excel(writer, sheet_name=sheet_name, index=False)

# ------------------------------------------------------------
# Summary note
# ------------------------------------------------------------
summary = pd.DataFrame([
    {
        "artifact": "shared_excel_tracker",
        "path": rel(xlsx_path),
        "exists": xlsx_path.exists(),
        "size_mb": round(xlsx_path.stat().st_size / (1024 * 1024), 3) if xlsx_path.exists() else "",
        "engine": engine,
    },
    {
        "artifact": "overview_csv",
        "path": "outputs/dataset_statistics/week6_shared_tracker_overview.csv",
        "exists": exists(STATS / "week6_shared_tracker_overview.csv"),
        "size_mb": round((STATS / "week6_shared_tracker_overview.csv").stat().st_size / (1024 * 1024), 3),
        "engine": "csv",
    },
    {
        "artifact": "updated_compliance_csv",
        "path": "outputs/dataset_statistics/week6_shared_tracker_task_compliance_updated.csv",
        "exists": exists(STATS / "week6_shared_tracker_task_compliance_updated.csv"),
        "size_mb": round((STATS / "week6_shared_tracker_task_compliance_updated.csv").stat().st_size / (1024 * 1024), 3),
        "engine": "csv",
    },
])

summary_path = STATS / "week6_shared_excel_tracker_summary.csv"
safe_to_csv(summary, summary_path)

note_path = NOTES / "week6_shared_excel_task_tracker_notes.md"

with open(note_path, "w") as f:
    f.write("# Week 6 Shared Excel Task Tracker\n\n")

    f.write("## Purpose\n\n")
    f.write(
        "This step addresses the updated shared Excel/task tracker deliverable. "
        "The workbook summarizes task compliance, experiments, key outputs, issues/limitations, and next steps.\n\n"
    )

    f.write("## Workbook\n\n")
    f.write(f"- XLSX: `{xlsx_path}`\n")
    f.write(f"- Generation engine: `{engine}`\n\n")

    f.write("## Sheets\n\n")
    for sheet_name, df in sheets.items():
        f.write(f"- `{sheet_name}`: {len(df)} rows, {len(df.columns)} columns\n")

    f.write("\n## Summary artifacts\n\n")
    f.write(summary.to_markdown(index=False))
    f.write("\n\n")

    f.write("## Interpretation\n\n")
    f.write(
        "The tracker can be used as the shared experiment/task status file for Week 6. "
        "It documents completed work, remaining/future issues, and the expected final package v2 steps.\n"
    )

print("Saved:")
print(xlsx_path)
print(summary_path)
print(note_path)

print()
print("=== Tracker summary ===")
print(summary.to_string(index=False))

print()
print("=== Sheets ===")
for sheet_name, df in sheets.items():
    print(f"{sheet_name}: {len(df)} rows x {len(df.columns)} columns")
