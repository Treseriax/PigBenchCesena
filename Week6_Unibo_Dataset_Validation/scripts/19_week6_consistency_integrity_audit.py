from pathlib import Path
import csv
import pandas as pd


PROJECT_ROOT = Path.home() / "PigBench"
W6 = PROJECT_ROOT / "Week6_Unibo_Dataset_Validation"

OUT_GT = W6 / "outputs/unified_ground_truth"
OUT_STATS = W6 / "outputs/dataset_statistics"
OUT_FEAT = W6 / "outputs/feature_extractors"
OUT_VIS = W6 / "outputs/visual_label_check"
NOTES = W6 / "notes"

GT_SPLIT = OUT_GT / "week6_unified_ground_truth_v2_with_split.csv"
GT_V2 = OUT_GT / "week6_unified_ground_truth_v2_with_recovered_videos.csv"
FRAME_INDEX = OUT_GT / "week6_scanpoint_frame_index.csv"
FRAME_LABELS = OUT_GT / "week6_scanpoint_frame_labels_long.csv"
DETECTIONS = OUT_FEAT / "week6_yolov8s_all_scanpoint_detections.csv"
MARKERS = OUT_FEAT / "week6_crop_colour_marker_features.csv"
CANDIDATES = OUT_FEAT / "week6_candidate_bbox_to_colour_assignments.csv"

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


def read_csv_required(path):
    if not path.exists():
        raise FileNotFoundError(path)
    return pd.read_csv(path)


def pass_fail(condition):
    return "PASS" if condition else "FAIL"


gt = read_csv_required(GT_SPLIT if GT_SPLIT.exists() else GT_V2)
frames = read_csv_required(FRAME_INDEX)
labels = read_csv_required(FRAME_LABELS)
dets = read_csv_required(DETECTIONS)
markers = read_csv_required(MARKERS)
candidates = read_csv_required(CANDIDATES)

checks = []

# -----------------------------
# Basic expected counts
# -----------------------------
checks.append({
    "check_name": "gt_total_records_is_432",
    "status": pass_fail(len(gt) == 432),
    "observed": len(gt),
    "expected": 432,
    "severity": "critical",
    "note": "Unified GT should preserve all Excel scan-sampling labels."
})

checks.append({
    "check_name": "scanpoint_frames_is_72",
    "status": pass_fail(len(frames) == 72),
    "observed": len(frames),
    "expected": 72,
    "severity": "critical",
    "note": "12 hours x 6 scan points per hour."
})

checks.append({
    "check_name": "frame_labels_is_432",
    "status": pass_fail(len(labels) == 432),
    "observed": len(labels),
    "expected": 432,
    "severity": "critical",
    "note": "Each of 72 frames should have six manual labels."
})

checks.append({
    "check_name": "detections_exist",
    "status": pass_fail(len(dets) > 0),
    "observed": len(dets),
    "expected": ">0",
    "severity": "critical",
    "note": "YOLOv8-s detection output should not be empty."
})

checks.append({
    "check_name": "marker_features_match_detection_count",
    "status": pass_fail(len(markers) == len(dets)),
    "observed": len(markers),
    "expected": len(dets),
    "severity": "critical",
    "note": "Each detection bbox should have one crop-marker feature row."
})

# -----------------------------
# Duplicate / uniqueness checks
# -----------------------------
record_dup = int(gt["record_id"].duplicated().sum()) if "record_id" in gt.columns else -1
checks.append({
    "check_name": "no_duplicate_record_id",
    "status": pass_fail(record_dup == 0),
    "observed": record_dup,
    "expected": 0,
    "severity": "critical",
    "note": "record_id should be unique."
})

scan_dup = int(frames["scan_frame_id"].duplicated().sum()) if "scan_frame_id" in frames.columns else -1
checks.append({
    "check_name": "no_duplicate_scan_frame_id",
    "status": pass_fail(scan_dup == 0),
    "observed": scan_dup,
    "expected": 0,
    "severity": "critical",
    "note": "scan_frame_id should be unique."
})

# -----------------------------
# Six labels per frame
# -----------------------------
label_counts = (
    labels.groupby("scan_frame_id")
    .size()
    .reset_index(name="manual_label_count")
)

bad_label_counts = label_counts[label_counts["manual_label_count"] != 6].copy()

checks.append({
    "check_name": "six_manual_labels_per_scan_frame",
    "status": pass_fail(len(bad_label_counts) == 0 and len(label_counts) == 72),
    "observed": f"{len(bad_label_counts)} bad frames; {len(label_counts)} labelled frames",
    "expected": "0 bad frames; 72 labelled frames",
    "severity": "critical",
    "note": "Each scan frame should contain six colour-ID behaviour labels."
})

# -----------------------------
# File path existence checks
# -----------------------------
frames["frame_image_exists"] = frames["frame_image_path"].apply(lambda p: Path(str(p)).exists())
missing_frame_images = frames[~frames["frame_image_exists"]].copy()

checks.append({
    "check_name": "all_frame_images_exist",
    "status": pass_fail(len(missing_frame_images) == 0),
    "observed": len(missing_frame_images),
    "expected": 0,
    "severity": "critical",
    "note": "All extracted scanpoint frame image paths should exist."
})

gt["video_path_exists"] = gt["video_path"].apply(lambda p: Path(str(p)).exists())
missing_video_paths = gt[~gt["video_path_exists"]].copy()

checks.append({
    "check_name": "all_gt_video_paths_exist",
    "status": pass_fail(len(missing_video_paths) == 0),
    "observed": len(missing_video_paths),
    "expected": 0,
    "severity": "critical",
    "note": "All GT video paths should exist after v2 recovery."
})

# -----------------------------
# Split checks
# -----------------------------
if "split" in gt.columns:
    split_missing = int(gt["split"].isna().sum() + (gt["split"].astype(str).str.strip() == "").sum())
    checks.append({
        "check_name": "all_gt_rows_have_split",
        "status": pass_fail(split_missing == 0),
        "observed": split_missing,
        "expected": 0,
        "severity": "critical",
        "note": "Every GT row should have train/val/test split."
    })

    video_split_counts = (
        gt.groupby("video_id")["split"]
        .nunique()
        .reset_index(name="num_splits")
    )
    leakage_videos = video_split_counts[video_split_counts["num_splits"] > 1].copy()

    checks.append({
        "check_name": "no_video_id_split_leakage",
        "status": pass_fail(len(leakage_videos) == 0),
        "observed": len(leakage_videos),
        "expected": 0,
        "severity": "critical",
        "note": "A video/hour unit must not appear in multiple splits."
    })

    split_summary = (
        gt.groupby("split")
        .agg(
            rows=("record_id", "count"),
            videos=("video_id", "nunique"),
            colours=("colour_id", "nunique"),
            behaviours=("behaviour_code", "nunique"),
        )
        .reset_index()
    )
else:
    split_summary = pd.DataFrame()
    leakage_videos = pd.DataFrame()
    checks.append({
        "check_name": "split_column_exists",
        "status": "FAIL",
        "observed": "missing",
        "expected": "split column",
        "severity": "critical",
        "note": "Split protocol file may not have been generated."
    })

# Behaviour coverage per split.
if "split" in gt.columns:
    split_behaviour = (
        gt.groupby(["split", "behaviour_code"])
        .size()
        .reset_index(name="count")
    )

    all_behaviours = set(gt["behaviour_code"].dropna().astype(str))
    split_coverage_rows = []

    for split_name, group in gt.groupby("split"):
        present = set(group["behaviour_code"].dropna().astype(str))
        missing = sorted(list(all_behaviours - present))
        split_coverage_rows.append({
            "split": split_name,
            "present_behaviour_count": len(present),
            "total_behaviour_count": len(all_behaviours),
            "missing_behaviours": ",".join(missing),
            "missing_count": len(missing),
        })

    split_coverage = pd.DataFrame(split_coverage_rows)
else:
    split_behaviour = pd.DataFrame()
    split_coverage = pd.DataFrame()

# This is warning, not hard fail, because rare classes may be absent from val/test.
if len(split_coverage):
    max_missing = int(split_coverage["missing_count"].max())
    checks.append({
        "check_name": "behaviour_coverage_per_split_warning",
        "status": "PASS_WITH_WARNINGS" if max_missing > 0 else "PASS",
        "observed": f"max missing behaviours in a split = {max_missing}",
        "expected": "0 ideal, but rare classes may be absent",
        "severity": "warning",
        "note": "Rare behaviours may not appear in every split because dataset is small."
    })

# -----------------------------
# Detection-frame alignment
# -----------------------------
det_frame_ids = set(dets["scan_frame_id"].astype(str))
frame_ids = set(frames["scan_frame_id"].astype(str))

missing_detection_frames = sorted(list(frame_ids - det_frame_ids))
extra_detection_frames = sorted(list(det_frame_ids - frame_ids))

checks.append({
    "check_name": "detections_cover_all_scan_frames",
    "status": pass_fail(len(missing_detection_frames) == 0),
    "observed": len(missing_detection_frames),
    "expected": 0,
    "severity": "critical",
    "note": "Each extracted frame should have detector rows. If a frame has zero detections, it should still be documented separately."
})

checks.append({
    "check_name": "no_extra_detection_scan_frame_ids",
    "status": pass_fail(len(extra_detection_frames) == 0),
    "observed": len(extra_detection_frames),
    "expected": 0,
    "severity": "critical",
    "note": "Detection scan_frame_id values should all exist in frame index."
})

det_per_frame = (
    dets.groupby("scan_frame_id")
    .size()
    .reset_index(name="bbox_count")
)

low_bbox_frames = det_per_frame[det_per_frame["bbox_count"] < 6].copy()
high_bbox_frames = det_per_frame[det_per_frame["bbox_count"] > 10].copy()

checks.append({
    "check_name": "bbox_count_reasonable_warning",
    "status": "PASS_WITH_WARNINGS" if (len(low_bbox_frames) > 0 or len(high_bbox_frames) > 0) else "PASS",
    "observed": f"{len(low_bbox_frames)} frames <6 boxes; {len(high_bbox_frames)} frames >10 boxes",
    "expected": "around 6 pigs, but occlusion/duplicate detections possible",
    "severity": "warning",
    "note": "Detector bboxes are automatic, so counts can differ from six manual pig labels."
})

# -----------------------------
# Candidate c-token confidence check
# -----------------------------
if "video_match_status" in gt.columns and "video_mapping_confidence" in gt.columns:
    ctoken = gt[gt["video_match_status"] == "candidate_recovered_ctoken_video"].copy()
    bad_ctoken_conf = ctoken[ctoken["video_mapping_confidence"] != "medium_needs_visual_confirmation"].copy()

    checks.append({
        "check_name": "candidate_ctoken_rows_have_medium_confidence",
        "status": pass_fail(len(bad_ctoken_conf) == 0),
        "observed": len(bad_ctoken_conf),
        "expected": 0,
        "severity": "important",
        "note": "Recovered c-token videos should remain marked as candidate/medium confidence."
    })

# -----------------------------
# Marker candidate checks
# -----------------------------
medium_high = markers[markers["marker_confidence"].isin(["medium", "high"])].copy()
checks.append({
    "check_name": "candidate_assignment_count_matches_medium_high_markers",
    "status": pass_fail(len(candidates) == len(medium_high)),
    "observed": len(candidates),
    "expected": len(medium_high),
    "severity": "important",
    "note": "Candidate bbox-to-colour assignment table should contain medium/high marker rows."
})

# -----------------------------
# Save outputs
# -----------------------------
checks_df = pd.DataFrame(checks)

audit_path = OUT_STATS / "week6_consistency_integrity_audit.csv"
safe_to_csv(checks_df, audit_path)

bad_label_counts_path = OUT_STATS / "week6_integrity_bad_label_counts_per_frame.csv"
safe_to_csv(bad_label_counts, bad_label_counts_path)

missing_frames_path = OUT_STATS / "week6_integrity_missing_frame_images.csv"
safe_to_csv(missing_frame_images, missing_frames_path)

missing_videos_path = OUT_STATS / "week6_integrity_missing_video_paths.csv"
safe_to_csv(missing_video_paths, missing_videos_path)

split_summary_path = OUT_STATS / "week6_integrity_split_summary.csv"
safe_to_csv(split_summary, split_summary_path)

split_coverage_path = OUT_STATS / "week6_integrity_split_behaviour_coverage.csv"
safe_to_csv(split_coverage, split_coverage_path)

low_bbox_path = OUT_STATS / "week6_integrity_low_bbox_frames.csv"
safe_to_csv(low_bbox_frames, low_bbox_path)

high_bbox_path = OUT_STATS / "week6_integrity_high_bbox_frames.csv"
safe_to_csv(high_bbox_frames, high_bbox_path)

note_path = NOTES / "week6_consistency_integrity_audit_notes.md"

fail_count = int((checks_df["status"] == "FAIL").sum())
warn_count = int((checks_df["status"] == "PASS_WITH_WARNINGS").sum())

with open(note_path, "w") as f:
    f.write("# Week 6 Consistency and Integrity Audit\n\n")

    f.write("## Purpose\n\n")
    f.write(
        "This audit checks the integrity of the Week 6 pipeline before moving to additional feature extractors. "
        "It validates GT counts, scanpoint frames, label-frame alignment, video path existence, split leakage, detection alignment, and marker candidate consistency.\n\n"
    )

    f.write("## Audit result summary\n\n")
    f.write(f"- Failed checks: `{fail_count}`\n")
    f.write(f"- Warning checks: `{warn_count}`\n")
    f.write(f"- Total checks: `{len(checks_df)}`\n\n")

    f.write("## Checks\n\n")
    f.write(checks_df.to_markdown(index=False))
    f.write("\n\n")

    f.write("## Split summary\n\n")
    f.write(split_summary.to_markdown(index=False) if len(split_summary) else "No split summary available.")
    f.write("\n\n")

    f.write("## Split behaviour coverage\n\n")
    f.write(split_coverage.to_markdown(index=False) if len(split_coverage) else "No split coverage available.")
    f.write("\n\n")

    f.write("## Low bbox-count frames\n\n")
    f.write(low_bbox_frames.to_markdown(index=False) if len(low_bbox_frames) else "No frames with fewer than six bboxes.")
    f.write("\n\n")

    f.write("## High bbox-count frames\n\n")
    f.write(high_bbox_frames.to_markdown(index=False) if len(high_bbox_frames) else "No frames with more than ten bboxes.")
    f.write("\n\n")

    f.write("## Interpretation\n\n")
    if fail_count == 0:
        f.write(
            "No critical integrity failures were found. Warning-level issues should still be reviewed, especially split behaviour coverage and frames with unusually low/high bbox counts. "
            "If only warnings remain, the pipeline can continue to feature extractor comparison.\n"
        )
    else:
        f.write(
            "One or more critical checks failed. These should be corrected before continuing to feature extractor comparison or final packaging.\n"
        )

print("Saved:")
print(audit_path)
print(bad_label_counts_path)
print(missing_frames_path)
print(missing_videos_path)
print(split_summary_path)
print(split_coverage_path)
print(low_bbox_path)
print(high_bbox_path)
print(note_path)

print()
print("=== Integrity audit checks ===")
print(checks_df.to_string(index=False))

print()
print("=== Split behaviour coverage ===")
print(split_coverage.to_string(index=False) if len(split_coverage) else "No split coverage.")

print()
print("=== Low bbox frames (<6) ===")
print(low_bbox_frames.to_string(index=False) if len(low_bbox_frames) else "None")

print()
print("=== High bbox frames (>10) ===")
print(high_bbox_frames.to_string(index=False) if len(high_bbox_frames) else "None")
