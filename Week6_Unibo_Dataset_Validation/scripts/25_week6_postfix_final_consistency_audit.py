from pathlib import Path
import csv
import pandas as pd


PROJECT_ROOT = Path.home() / "PigBench"
W6 = PROJECT_ROOT / "Week6_Unibo_Dataset_Validation"

OUT_GT = W6 / "outputs/unified_ground_truth"
OUT_STATS = W6 / "outputs/dataset_statistics"
OUT_FEAT = W6 / "outputs/feature_extractors"
NOTES = W6 / "notes"

GT_RECOMMENDED = OUT_GT / "week6_unified_ground_truth_v2_with_recommended_split_v2.csv"
FRAME_INDEX = OUT_GT / "week6_scanpoint_frame_index.csv"
FRAME_LABELS = OUT_GT / "week6_scanpoint_frame_labels_long.csv"

DETECTIONS_PRIMARY = OUT_FEAT / "week6_yolov8s_all_scanpoint_detections.csv"
DETECTIONS_QC = OUT_FEAT / "week6_yolov8s_all_scanpoint_detections_with_qc_flags.csv"
DETECTIONS_050 = OUT_FEAT / "week6_yolov8s_conservative_detections_score_ge_0_50.csv"
DETECTIONS_070 = OUT_FEAT / "week6_yolov8s_conservative_detections_score_ge_0_70.csv"

MARKERS = OUT_FEAT / "week6_crop_colour_marker_features.csv"
CANDIDATES = OUT_FEAT / "week6_candidate_bbox_to_colour_assignments.csv"

SPLIT_COVERAGE = OUT_STATS / "week6_recommended_split_v2_behaviour_coverage.csv"
SPLIT_LEAKAGE = OUT_STATS / "week6_recommended_split_v2_leakage_check.csv"
DETECTION_QC_SUMMARY = OUT_STATS / "week6_detection_qc_conservative_subset_summary.csv"

OUT_STATS.mkdir(parents=True, exist_ok=True)
NOTES.mkdir(parents=True, exist_ok=True)


def safe_to_csv(df, path):
    df.to_csv(
        path,
        index=False,
        quoting=csv.QUOTE_ALL,
        escapechar="\\",
        lineterminator="\n"
    )


def read_required(path):
    if not path.exists():
        raise FileNotFoundError(path)
    return pd.read_csv(path)


def read_optional(path):
    if path.exists():
        return pd.read_csv(path)
    return pd.DataFrame()


def status(condition):
    return "PASS" if condition else "FAIL"


gt = read_required(GT_RECOMMENDED)
frames = read_required(FRAME_INDEX)
frame_labels = read_required(FRAME_LABELS)

det_primary = read_required(DETECTIONS_PRIMARY)
det_qc = read_required(DETECTIONS_QC)
det_050 = read_required(DETECTIONS_050)
det_070 = read_required(DETECTIONS_070)

markers = read_required(MARKERS)
candidates = read_required(CANDIDATES)

split_coverage = read_optional(SPLIT_COVERAGE)
split_leakage = read_optional(SPLIT_LEAKAGE)
det_qc_summary = read_optional(DETECTION_QC_SUMMARY)

checks = []

checks.append({
    "check_name": "recommended_gt_exists_and_has_432_records",
    "status": status(len(gt) == 432),
    "observed": len(gt),
    "expected": 432,
    "severity": "critical",
    "note": "Recommended split v2 GT must preserve all manual Excel labels.",
})

checks.append({
    "check_name": "recommended_split_v2_column_exists",
    "status": status("recommended_split_v2" in gt.columns),
    "observed": "recommended_split_v2" in gt.columns,
    "expected": True,
    "severity": "critical",
    "note": "Recommended split v2 file should contain recommended_split_v2 column.",
})

if "recommended_split_v2" in gt.columns:
    split_counts = gt["recommended_split_v2"].value_counts().to_dict()
    checks.append({
        "check_name": "recommended_split_v2_counts_are_288_72_72",
        "status": status(
            split_counts.get("train", 0) == 288
            and split_counts.get("val", 0) == 72
            and split_counts.get("test", 0) == 72
        ),
        "observed": str(split_counts),
        "expected": "{'train': 288, 'val': 72, 'test': 72}",
        "severity": "critical",
        "note": "Recommended split should be 8/2/2 video-hour units.",
    })

    leakage = (
        gt.groupby("split_unit_id")["recommended_split_v2"]
        .nunique()
        .reset_index(name="num_splits")
    )
    leakage = leakage[leakage["num_splits"] > 1]

    checks.append({
        "check_name": "recommended_split_v2_has_no_video_hour_leakage",
        "status": status(len(leakage) == 0),
        "observed": len(leakage),
        "expected": 0,
        "severity": "critical",
        "note": "No hourly video unit should appear in multiple splits.",
    })

    train_beh = gt[gt["recommended_split_v2"] == "train"]["behaviour_code"].nunique()
    val_beh = gt[gt["recommended_split_v2"] == "val"]["behaviour_code"].nunique()
    test_beh = gt[gt["recommended_split_v2"] == "test"]["behaviour_code"].nunique()

    checks.append({
        "check_name": "recommended_split_v2_behaviour_coverage_expected",
        "status": status(train_beh == 11 and val_beh == 10 and test_beh == 10),
        "observed": f"train={train_beh}, val={val_beh}, test={test_beh}",
        "expected": "train=11, val=10, test=10",
        "severity": "important",
        "note": "Optimized split should improve val/test coverage while keeping full train coverage.",
    })

checks.append({
    "check_name": "scanpoint_frames_still_72",
    "status": status(len(frames) == 72),
    "observed": len(frames),
    "expected": 72,
    "severity": "critical",
    "note": "Scanpoint frame index should remain unchanged.",
})

checks.append({
    "check_name": "frame_labels_still_432",
    "status": status(len(frame_labels) == 432),
    "observed": len(frame_labels),
    "expected": 432,
    "severity": "critical",
    "note": "Frame-label long table should keep six labels per scanpoint frame.",
})

bad_label_counts = (
    frame_labels.groupby("scan_frame_id")
    .size()
    .reset_index(name="label_count")
)
bad_label_counts = bad_label_counts[bad_label_counts["label_count"] != 6]

checks.append({
    "check_name": "six_labels_per_scan_frame_still_true",
    "status": status(len(bad_label_counts) == 0),
    "observed": len(bad_label_counts),
    "expected": 0,
    "severity": "critical",
    "note": "Every scan frame should still have six manual labels.",
})

checks.append({
    "check_name": "primary_detection_count_still_540",
    "status": status(len(det_primary) == 540),
    "observed": len(det_primary),
    "expected": 540,
    "severity": "important",
    "note": "Primary detection table should remain unchanged.",
})

checks.append({
    "check_name": "qc_detection_count_matches_primary",
    "status": status(len(det_qc) == len(det_primary)),
    "observed": len(det_qc),
    "expected": len(det_primary),
    "severity": "critical",
    "note": "QC flag table should not add/drop detections.",
})

required_qc_cols = [
    "bbox_count_issue_type",
    "detection_score_band",
    "recommended_use",
    "conservative_keep_score_ge_0_50",
    "conservative_keep_score_ge_0_70",
]

missing_qc_cols = [c for c in required_qc_cols if c not in det_qc.columns]

checks.append({
    "check_name": "qc_detection_columns_exist",
    "status": status(len(missing_qc_cols) == 0),
    "observed": ",".join(missing_qc_cols),
    "expected": "no missing QC columns",
    "severity": "critical",
    "note": "QC flag columns are needed for later feature extraction decisions.",
})

if "bbox_count_issue_type" in det_qc.columns:
    issue_frame_counts = (
        det_qc.groupby("bbox_count_issue_type")["scan_frame_id"]
        .nunique()
        .to_dict()
    )

    checks.append({
        "check_name": "qc_warning_frame_counts_expected",
        "status": status(
            issue_frame_counts.get("low_bbox_count_warning", 0) == 10
            and issue_frame_counts.get("high_bbox_count_warning", 0) == 4
        ),
        "observed": str(issue_frame_counts),
        "expected": "10 low warning frames, 4 high warning frames",
        "severity": "important",
        "note": "QC flag table should preserve the warning frame audit result.",
    })

checks.append({
    "check_name": "conservative_050_covers_all_72_frames",
    "status": status(det_050["scan_frame_id"].nunique() == 72),
    "observed": det_050["scan_frame_id"].nunique(),
    "expected": 72,
    "severity": "important",
    "note": "Score >=0.50 subset should still cover every scanpoint frame.",
})

checks.append({
    "check_name": "conservative_070_covers_all_72_frames",
    "status": status(det_070["scan_frame_id"].nunique() == 72),
    "observed": det_070["scan_frame_id"].nunique(),
    "expected": 72,
    "severity": "warning",
    "note": "Score >=0.70 subset coverage is useful but strict; missing frames would be a warning, not critical.",
})

checks.append({
    "check_name": "marker_features_match_primary_detections",
    "status": status(len(markers) == len(det_primary)),
    "observed": len(markers),
    "expected": len(det_primary),
    "severity": "critical",
    "note": "Each primary detection should still have one marker feature row.",
})

medium_high = markers[markers["marker_confidence"].isin(["medium", "high"])]

checks.append({
    "check_name": "candidate_bbox_colour_assignments_match_medium_high_markers",
    "status": status(len(candidates) == len(medium_high)),
    "observed": len(candidates),
    "expected": len(medium_high),
    "severity": "important",
    "note": "Candidate colour assignments should equal high/medium marker detections.",
})

checks_df = pd.DataFrame(checks)

audit_path = OUT_STATS / "week6_postfix_final_consistency_audit.csv"
safe_to_csv(checks_df, audit_path)

# Useful summaries
split_final_summary = (
    gt.groupby("recommended_split_v2")
    .agg(
        labels=("record_id", "count"),
        video_units=("split_unit_id", "nunique"),
        behaviours=("behaviour_code", "nunique"),
        colours=("colour_id", "nunique"),
        direct_tlc_records=("video_match_status", lambda s: int((s == "matched_tlc_hour_video").sum())),
        candidate_ctoken_records=("video_match_status", lambda s: int((s == "candidate_recovered_ctoken_video").sum())),
    )
    .reset_index()
    .rename(columns={"recommended_split_v2": "split"})
)

split_final_summary_path = OUT_STATS / "week6_postfix_final_split_summary.csv"
safe_to_csv(split_final_summary, split_final_summary_path)

detection_final_summary = pd.DataFrame([
    {
        "table": "primary_score_ge_0_25",
        "detections": len(det_primary),
        "frames": det_primary["scan_frame_id"].nunique(),
        "mean_bboxes_per_frame": round(det_primary.groupby("scan_frame_id").size().mean(), 3),
    },
    {
        "table": "qc_flagged_primary",
        "detections": len(det_qc),
        "frames": det_qc["scan_frame_id"].nunique(),
        "mean_bboxes_per_frame": round(det_qc.groupby("scan_frame_id").size().mean(), 3),
    },
    {
        "table": "conservative_score_ge_0_50",
        "detections": len(det_050),
        "frames": det_050["scan_frame_id"].nunique(),
        "mean_bboxes_per_frame": round(det_050.groupby("scan_frame_id").size().mean(), 3),
    },
    {
        "table": "conservative_score_ge_0_70",
        "detections": len(det_070),
        "frames": det_070["scan_frame_id"].nunique(),
        "mean_bboxes_per_frame": round(det_070.groupby("scan_frame_id").size().mean(), 3),
    },
])

detection_final_summary_path = OUT_STATS / "week6_postfix_final_detection_summary.csv"
safe_to_csv(detection_final_summary, detection_final_summary_path)

fail_count = int((checks_df["status"] == "FAIL").sum())
warning_fail_count = int(((checks_df["status"] == "FAIL") & (checks_df["severity"] == "warning")).sum())

note_path = NOTES / "week6_postfix_final_consistency_audit_notes.md"

with open(note_path, "w") as f:
    f.write("# Week 6 Post-fix Final Consistency Audit\n\n")

    f.write("## Purpose\n\n")
    f.write(
        "This audit verifies the pipeline after adopting recommended split v2 and adding detection QC flags/conservative subsets. "
        "It is intended as the final checkpoint before moving to feature extractor comparison.\n\n"
    )

    f.write("## Result summary\n\n")
    f.write(f"- Failed checks: `{fail_count}`\n")
    f.write(f"- Total checks: `{len(checks_df)}`\n\n")

    f.write("## Checks\n\n")
    f.write(checks_df.to_markdown(index=False))
    f.write("\n\n")

    f.write("## Final split summary\n\n")
    f.write(split_final_summary.to_markdown(index=False))
    f.write("\n\n")

    f.write("## Final detection summary\n\n")
    f.write(detection_final_summary.to_markdown(index=False))
    f.write("\n\n")

    f.write("## Interpretation\n\n")
    if fail_count == 0:
        f.write(
            "All post-fix consistency checks passed. The Week 6 pipeline is ready for the next stage: feature extractor comparison and final report packaging. "
            "The primary detector table remains unchanged, while QC flags and conservative subsets are available for safer downstream analysis.\n"
        )
    else:
        f.write(
            "Some post-fix checks failed. Review the failed rows before continuing to feature extractor comparison.\n"
        )

print("Saved:")
print(audit_path)
print(split_final_summary_path)
print(detection_final_summary_path)
print(note_path)

print()
print("=== Post-fix audit checks ===")
print(checks_df.to_string(index=False))

print()
print("=== Final split summary ===")
print(split_final_summary.to_string(index=False))

print()
print("=== Final detection summary ===")
print(detection_final_summary.to_string(index=False))
