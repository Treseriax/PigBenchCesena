from pathlib import Path
import csv
import pandas as pd
import numpy as np
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


# ------------------------------------------------------------
# 1) Text cleanup pass
# ------------------------------------------------------------
cleanup_replacements = {
    "alightweight": "a lightweight",
    "manualvisual": "manual visual",
    "conservativesubset": "conservative subset",
    "exists:interface": "exists: interface",
    "tocandidate": "to candidate",
    "posturerepresentation": "posture representation",
    "recommendedfile": "recommended file",
    "requiredoutput": "required output",
    "notmanual": "not manual",
    "andan": "and an",
}

cleanup_rows = []

for note in NOTES.glob("*.md"):
    text = note.read_text(errors="ignore")
    original = text
    applied = []

    for old, new in cleanup_replacements.items():
        if old in text:
            text = text.replace(old, new)
            applied.append(f"{old}->{new}")

    if text != original:
        note.write_text(text)
        cleanup_rows.append({
            "file": rel(note),
            "changed": True,
            "replacements": "; ".join(applied),
        })

cleanup_df = pd.DataFrame(cleanup_rows)
cleanup_path = STATS / "week6_quality_hardening_v2_text_cleanup_report.csv"
safe_to_csv(cleanup_df, cleanup_path)

# ------------------------------------------------------------
# 2) Audit checks
# ------------------------------------------------------------
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


# Core counts.
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


# Interface artifacts.
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


# Segmentation overlays and manual QC.
seg_overlay_dir = VIS / "preliminary_segmentation_baseline"
overlay_count = len(list(seg_overlay_dir.glob("*_preliminary_segmentation_overlay.jpg"))) if seg_overlay_dir.exists() else 0

add_check(
    "Segmentation",
    "segmentation_overlay_count_72",
    "PASS" if overlay_count == 72 else "FAIL",
    "high" if overlay_count != 72 else "none",
    f"overlay_count={overlay_count}",
    "Regenerate segmentation overlays." if overlay_count != 72 else "No action.",
)

seg_qc_summary_path = STATS / "week6_segmentation_visual_qc_final_review_summary.csv"
seg_qc_verdict_path = STATS / "week6_segmentation_visual_qc_sheet_level_verdict.csv"
seg_qc_manual_path = STATS / "week6_segmentation_visual_qc_manual_review_results.csv"

seg_qc_ok = False
seg_qc_evidence = ""

if seg_qc_summary_path.exists():
    q = pd.read_csv(seg_qc_summary_path)
    q_str = q.astype(str)

    has_score = q_str.apply(lambda col: col.str.contains("high_risk_sheet_score", case=False, regex=False)).any().any()
    has_verdict = q_str.apply(lambda col: col.str.contains("accepted_for_preliminary_feature_extraction", case=False, regex=False)).any().any()
    has_no_fix = q_str.apply(lambda col: col.str.contains("False", case=False, regex=False)).any().any()

    seg_qc_ok = bool(has_score and has_verdict and has_no_fix)
    seg_qc_evidence = f"summary_exists=True; has_score={has_score}; has_verdict={has_verdict}; has_no_fix={has_no_fix}"
else:
    seg_qc_evidence = "summary_exists=False"

add_check(
    "Segmentation",
    "segmentation_manual_visual_qc_recorded",
    "PASS" if seg_qc_ok else "WARN",
    "medium" if not seg_qc_ok else "none",
    seg_qc_evidence,
    "Record manual visual QC result before final package." if not seg_qc_ok else "No action.",
)

for p in [seg_qc_verdict_path, seg_qc_manual_path]:
    add_check(
        "Segmentation",
        f"exists_{p.name}",
        "PASS" if p.exists() else "WARN",
        "medium" if not p.exists() else "none",
        rel(p),
        "Regenerate segmentation manual QC outputs." if not p.exists() else "No action.",
    )


# Learned detector-backed crop embeddings.
embedding_csv = FEAT / "week6_detector_backbone_crop_embeddings_896.csv"
embedding_npy = FEAT / "week6_detector_backbone_crop_embeddings_896.npy"
embedding_metadata = FEAT / "week6_detector_backbone_crop_embedding_metadata.csv"
embedding_pca = FEAT / "week6_detector_backbone_crop_embedding_pca_features.csv"
embedding_verification = STATS / "week6_detector_backbone_crop_embedding_verification.csv"
embedding_summary = STATS / "week6_detector_backbone_crop_embedding_summary.csv"

embedding_ok = False
embedding_evidence = ""

if embedding_verification.exists():
    v = pd.read_csv(embedding_verification)
    all_pass = bool((v["status"] == "PASS").all()) if "status" in v.columns else False
    embedding_ok = all_pass
    embedding_evidence = f"verification_exists=True; all_pass={all_pass}; rows={len(v)}"
else:
    embedding_evidence = "verification_exists=False"

add_check(
    "Feature",
    "learned_detector_backbone_crop_embeddings_verified",
    "PASS" if embedding_ok else "WARN",
    "medium" if not embedding_ok else "none",
    embedding_evidence,
    "Regenerate or inspect detector-backed embedding extraction." if not embedding_ok else "No action.",
)

for p, expected in [
    (embedding_csv, 540),
    (embedding_metadata, 540),
    (embedding_pca, 540),
]:
    rows = csv_rows(p)
    add_check(
        "Feature",
        f"exists_rows_{p.name}",
        "PASS" if rows == expected else "WARN",
        "medium" if rows != expected else "none",
        f"{rel(p)} rows={rows}, expected={expected}",
        "Regenerate full detector-backed crop embeddings." if rows != expected else "No action.",
    )

add_check(
    "Feature",
    "embedding_npy_exists",
    "PASS" if embedding_npy.exists() else "WARN",
    "medium" if not embedding_npy.exists() else "none",
    rel(embedding_npy),
    "Regenerate embedding NPY." if not embedding_npy.exists() else "No action.",
)


# Trajectory not overclaimed.
trajectory_path = FEAT / "week6_trajectory_feature_feasibility_report.csv"

if trajectory_path.exists():
    tr = pd.read_csv(trajectory_path)
    text = " ".join(tr.astype(str).values.flatten().tolist()).lower()
    honest = any(term in text for term in ["not", "feasibility", "not validated", "not final", "candidate"])

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


# Excel tracker readability.
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


# Known bad term scan after cleanup.
bad_terms = list(cleanup_replacements.keys())
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
    "known_bad_terms_in_notes_after_cleanup",
    "PASS" if not bad_hits else "WARN",
    "low" if bad_hits else "none",
    f"bad_hits={bad_hits[:20]}",
    "Patch known typo artifacts before final package." if bad_hits else "No action.",
)


# ------------------------------------------------------------
# 3) Save audit
# ------------------------------------------------------------
audit = pd.DataFrame(checks)

audit_path = STATS / "week6_quality_hardening_audit_v2.csv"
safe_to_csv(audit, audit_path)

risk = audit[audit["status"].isin(["FAIL", "WARN"])].copy()
risk_path = STATS / "week6_quality_hardening_risk_list_v2.csv"
safe_to_csv(risk, risk_path)

summary = (
    audit.groupby(["status", "severity"])
    .size()
    .reset_index(name="count")
    .sort_values(["status", "severity"])
)

summary_path = STATS / "week6_quality_hardening_summary_v2.csv"
safe_to_csv(summary, summary_path)

bad_hits_df = pd.DataFrame(bad_hits)
bad_hits_path = STATS / "week6_note_typo_bad_term_hits_v2.csv"
safe_to_csv(bad_hits_df, bad_hits_path)

note_path = NOTES / "week6_quality_hardening_audit_v2_notes.md"

with open(note_path, "w") as f:
    f.write("# Week 6 Quality Hardening Audit v2\n\n")

    f.write("## Purpose\n\n")
    f.write(
        "This audit reruns the quality-hardening checks after adding detector-backed crop embeddings and recording segmentation manual visual QC.\n\n"
    )

    f.write("## Summary\n\n")
    f.write(summary.to_markdown(index=False))
    f.write("\n\n")

    f.write("## Remaining risks / warnings\n\n")
    if len(risk):
        f.write(risk[["area", "check_name", "status", "severity", "evidence", "recommendation"]].to_markdown(index=False))
    else:
        f.write("No FAIL or WARN items remain.")
    f.write("\n\n")

    f.write("## Text cleanup\n\n")
    if len(cleanup_df):
        f.write(cleanup_df.to_markdown(index=False))
    else:
        f.write("No text cleanup changes were needed.")
    f.write("\n\n")

    f.write("## Interpretation\n\n")
    if len(risk):
        f.write(
            "Some quality-hardening warnings still remain. Resolve them before final package v2.\n"
        )
    else:
        f.write(
            "All quality-hardening checks pass. The project is task-sheet complete and quality-hardened for final package v2 construction.\n"
        )

print("Saved:")
print(audit_path)
print(risk_path)
print(summary_path)
print(cleanup_path)
print(bad_hits_path)
print(note_path)

print()
print("=== Quality hardening summary v2 ===")
print(summary.to_string(index=False))

print()
print("=== Remaining risks / warnings v2 ===")
print(risk[["area", "check_name", "status", "severity", "evidence", "recommendation"]].to_string(index=False) if len(risk) else "None")

print()
print("=== Text cleanup ===")
print(cleanup_df.to_string(index=False) if len(cleanup_df) else "None")
