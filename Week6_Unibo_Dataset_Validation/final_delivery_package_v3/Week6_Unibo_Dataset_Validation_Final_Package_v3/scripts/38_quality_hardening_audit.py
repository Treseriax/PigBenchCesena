from pathlib import Path
import csv
import pandas as pd
import re


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


def exists(path):
    return Path(path).exists()


def csv_rows(path):
    path = Path(path)
    if not path.exists() or path.suffix.lower() != ".csv":
        return None
    try:
        return len(pd.read_csv(path))
    except Exception:
        return "read_error"


def rel(path):
    try:
        return str(Path(path).relative_to(W6))
    except Exception:
        return str(path)


checks = []

def add_check(area, check_name, status, severity, evidence, recommendation):
    checks.append({
        "area": area,
        "check_name": check_name,
        "status": status,
        "severity": severity,
        "evidence": evidence,
        "recommendation": recommendation,
    })


# ------------------------------------------------------------
# Core output existence and counts
# ------------------------------------------------------------
expected_counts = [
    ("GT", "recommended_gt_csv_432", GT / "week6_unified_ground_truth_v2_with_recommended_split_v2.csv", 432),
    ("GT", "frame_index_72", GT / "week6_scanpoint_frame_index.csv", 72),
    ("GT", "frame_labels_432", GT / "week6_scanpoint_frame_labels_long.csv", 432),
    ("Detection", "det_qc_540", FEAT / "week6_yolov8s_all_scanpoint_detections_with_qc_flags.csv", 540),
    ("Feature", "bbox_geometry_540", FEAT / "week6_bbox_geometry_features.csv", 540),
    ("Feature", "group_spatial_72", FEAT / "week6_group_spatial_features_per_frame.csv", 72),
    ("Feature", "crop_descriptor_540", FEAT / "week6_crop_descriptor_baseline_features.csv", 540),
    ("Segmentation", "segmentation_features_540", FEAT / "week6_preliminary_bbox_guided_segmentation_features.csv", 540),
    ("Segmentation", "segmentation_frame_summary_72", FEAT / "week6_preliminary_segmentation_frame_summary.csv", 72),
]

for area, name, path, expected in expected_counts:
    rows = csv_rows(path)
    status = "PASS" if rows == expected else "FAIL"
    add_check(
        area,
        name,
        status,
        "high" if status == "FAIL" else "none",
        f"{rel(path)} rows={rows}, expected={expected}",
        "Regenerate or inspect this table." if status == "FAIL" else "No action.",
    )


# ------------------------------------------------------------
# Interface artifacts
# ------------------------------------------------------------
interface_files = [
    INTERFACE / "week6_visualization_streamlit_app.py",
    INTERFACE / "week6_static_visualization_viewer.html",
    INTERFACE / "week6_visualization_interface_index.csv",
]

for p in interface_files:
    add_check(
        "Interface",
        f"exists_{p.name}",
        "PASS" if p.exists() else "FAIL",
        "high" if not p.exists() else "none",
        rel(p),
        "Regenerate interface demo." if not p.exists() else "No action.",
    )


# ------------------------------------------------------------
# Segmentation quality
# ------------------------------------------------------------
seg_quality_path = STATS / "week6_preliminary_segmentation_quality_summary.csv"
seg_status_path = STATS / "week6_preliminary_segmentation_status_summary.csv"
seg_contact = VIS / "week6_preliminary_segmentation_contact_sheet.jpg"
seg_overlay_dir = VIS / "preliminary_segmentation_baseline"

if seg_quality_path.exists():
    q = pd.read_csv(seg_quality_path)
    val = q[q["metric"] == "success_rate"]["value"]
    success_rate = float(val.iloc[0]) if len(val) else None

    add_check(
        "Segmentation",
        "segmentation_success_rate_recorded",
        "PASS" if success_rate is not None else "FAIL",
        "medium" if success_rate is None else "none",
        f"success_rate={success_rate}",
        "Inspect segmentation quality summary." if success_rate is None else "No action.",
    )

    add_check(
        "Segmentation",
        "segmentation_visual_qc_needed",
        "WARN",
        "medium",
        "Automatic masks exist, but visual/manual quality scoring has not been recorded yet.",
        "Create segmentation visual QC sampling table with selected overlay frames.",
    )
else:
    add_check(
        "Segmentation",
        "segmentation_quality_summary_exists",
        "FAIL",
        "high",
        rel(seg_quality_path),
        "Regenerate segmentation baseline.",
    )

overlay_count = len(list(seg_overlay_dir.glob("*_preliminary_segmentation_overlay.jpg"))) if seg_overlay_dir.exists() else 0

add_check(
    "Segmentation",
    "segmentation_overlay_count_72",
    "PASS" if overlay_count == 72 else "FAIL",
    "high" if overlay_count != 72 else "none",
    f"overlay_count={overlay_count}",
    "Regenerate segmentation overlays." if overlay_count != 72 else "No action.",
)

add_check(
    "Segmentation",
    "segmentation_contact_sheet_exists",
    "PASS" if seg_contact.exists() else "FAIL",
    "medium" if not seg_contact.exists() else "none",
    rel(seg_contact),
    "Regenerate contact sheet." if not seg_contact.exists() else "No action.",
)


# ------------------------------------------------------------
# Learned embedding gap
# ------------------------------------------------------------
embedding_candidates = list(FEAT.glob("*embedding*.csv")) + list(FEAT.glob("*embedding*.json"))

add_check(
    "Feature",
    "learned_crop_embedding_output_exists",
    "PASS" if embedding_candidates else "WARN",
    "medium" if not embedding_candidates else "none",
    ", ".join(rel(p) for p in embedding_candidates) if embedding_candidates else "No learned embedding table found.",
    "Add a lightweight learned crop embedding baseline if torchvision/torch are available.",
)


# ------------------------------------------------------------
# Trajectory limitation honesty
# ------------------------------------------------------------
trajectory_path = FEAT / "week6_trajectory_feature_feasibility_report.csv"

if trajectory_path.exists():
    tr = pd.read_csv(trajectory_path)
    honest = tr.astype(str).apply(lambda col: col.str.contains("not_final_identity_resolved|candidate_only|not validated", case=False, regex=True)).any().any()

    add_check(
        "Trajectory",
        "trajectory_not_overclaimed",
        "PASS" if honest else "WARN",
        "medium" if not honest else "none",
        rel(trajectory_path),
        "Ensure trajectory is described as feasibility only unless identity tracking is validated.",
    )
else:
    add_check(
        "Trajectory",
        "trajectory_feasibility_report_exists",
        "FAIL",
        "medium",
        rel(trajectory_path),
        "Generate trajectory feasibility report.",
    )


# ------------------------------------------------------------
# Excel tracker readability
# ------------------------------------------------------------
xlsx_path = TRACKER / "Week6_Unibo_Dataset_Validation_Shared_Task_Tracker.xlsx"

try:
    xf = pd.ExcelFile(xlsx_path)
    sheets = xf.sheet_names
    excel_ok = True
except Exception as e:
    sheets = []
    excel_ok = False

add_check(
    "Tracker",
    "excel_tracker_readable",
    "PASS" if excel_ok else "FAIL",
    "high" if not excel_ok else "none",
    f"{rel(xlsx_path)} sheets={sheets}",
    "Regenerate tracker." if not excel_ok else "No action.",
)


# ------------------------------------------------------------
# Text typo scan in notes
# ------------------------------------------------------------
bad_terms = [
    "tocandidate",
    "posturerepresentation",
    "recommendedfile",
    "requiredoutput",
    "notmanual",
    "andan",
    "conservativesubset",
    "exists:interface",
]

bad_hits = []

for note in NOTES.glob("*.md"):
    text = note.read_text(errors="ignore")
    for term in bad_terms:
        if term in text:
            bad_hits.append({
                "file": rel(note),
                "bad_term": term,
            })

add_check(
    "TextQuality",
    "known_bad_terms_in_notes",
    "PASS" if not bad_hits else "WARN",
    "low" if bad_hits else "none",
    f"bad_hits={bad_hits[:20]}",
    "Patch known typo artifacts before final package." if bad_hits else "No action.",
)


# ------------------------------------------------------------
# Final risk list
# ------------------------------------------------------------
audit = pd.DataFrame(checks)

audit_path = STATS / "week6_quality_hardening_audit.csv"
safe_to_csv(audit, audit_path)

risk = audit[audit["status"].isin(["FAIL", "WARN"])].copy()
risk_path = STATS / "week6_quality_hardening_risk_list.csv"
safe_to_csv(risk, risk_path)

summary = (
    audit.groupby(["status", "severity"])
    .size()
    .reset_index(name="count")
    .sort_values(["status", "severity"])
)

summary_path = STATS / "week6_quality_hardening_summary.csv"
safe_to_csv(summary, summary_path)

bad_hits_df = pd.DataFrame(bad_hits)
bad_hits_path = STATS / "week6_note_typo_bad_term_hits.csv"
safe_to_csv(bad_hits_df, bad_hits_path)

note_path = NOTES / "week6_quality_hardening_audit_notes.md"

with open(note_path, "w") as f:
    f.write("# Week 6 Quality Hardening Audit\n\n")

    f.write("## Purpose\n\n")
    f.write(
        "The task-sheet compliance audit confirms that required outputs exist. "
        "This quality-hardening audit checks whether any technically complete item still needs extra validation before final package v2.\n\n"
    )

    f.write("## Summary\n\n")
    f.write(summary.to_markdown(index=False))
    f.write("\n\n")

    f.write("## Remaining risks / warnings\n\n")
    if len(risk):
        f.write(risk[["area", "check_name", "status", "severity", "evidence", "recommendation"]].to_markdown(index=False))
    else:
        f.write("No FAIL or WARN items found.")
    f.write("\n\n")

    f.write("## Interpretation\n\n")
    if len(risk):
        f.write(
            "The project is task-sheet complete, but the listed quality warnings should be resolved or explicitly validated before final package v2.\n"
        )
    else:
        f.write(
            "No additional quality risks were found. The project is ready for final package v2.\n"
        )

print("Saved:")
print(audit_path)
print(risk_path)
print(summary_path)
print(bad_hits_path)
print(note_path)

print()
print("=== Quality hardening summary ===")
print(summary.to_string(index=False))

print()
print("=== Risks / warnings ===")
print(risk[["area", "check_name", "status", "severity", "evidence", "recommendation"]].to_string(index=False) if len(risk) else "None")
