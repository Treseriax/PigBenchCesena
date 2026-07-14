from pathlib import Path
from datetime import datetime
import csv
import math
import re

import pandas as pd
import numpy as np


ROOT = Path.home() / "PigBench"
W7 = ROOT / "Week7_Dataset_Validation_Identity_Tracking_Behaviour_Preparation"

V27A_ROOT = W7 / "outputs" / "detector_tracker_preflight_v27a"
V27B_ROOT = W7 / "outputs" / "detector_tracker_dryrun_v27b"

SELECTED = V27A_ROOT / "week7_detector_tracker_preflight_v27a_selected_dryrun_clips.csv"

DETECTIONS = V27B_ROOT / "week7_detector_tracker_dryrun_v27b_detections.csv"
CLIP_SUMMARY = V27B_ROOT / "week7_detector_tracker_dryrun_v27b_clip_summary.csv"
FRAME_SUMMARY = V27B_ROOT / "week7_detector_tracker_dryrun_v27b_frame_summary.csv"
CONTACT_INDEX = V27B_ROOT / "week7_detector_tracker_dryrun_v27b_contact_sheet_index.csv"
V27B_ISSUES = V27B_ROOT / "week7_detector_tracker_dryrun_v27b_issues.csv"
V27B_SUMMARY = V27B_ROOT / "week7_detector_tracker_dryrun_v27b_summary.csv"

BOX_LEVEL = W7 / "outputs" / "behaviour_label_fusion_v18c_verified_crosswalk" / "week7_behaviour_label_fusion_v18c_box_level_dataset.csv"

OUT_ROOT = W7 / "outputs" / "detector_tracker_quality_audit_v27c"
OUT_ROOT.mkdir(parents=True, exist_ok=True)

OUT_ENRICHED_DETECTIONS = OUT_ROOT / "week7_detector_tracker_quality_audit_v27c_enriched_detections.csv"
OUT_FRAME_QUALITY = OUT_ROOT / "week7_detector_tracker_quality_audit_v27c_frame_quality.csv"
OUT_CLIP_QUALITY = OUT_ROOT / "week7_detector_tracker_quality_audit_v27c_clip_quality.csv"
OUT_VISUAL_REVIEW_TEMPLATE = OUT_ROOT / "week7_detector_tracker_quality_audit_v27c_visual_review_template.csv"
OUT_DECISION = OUT_ROOT / "week7_detector_tracker_quality_audit_v27c_decision_summary.csv"
OUT_LIMITATIONS = OUT_ROOT / "week7_detector_tracker_quality_audit_v27c_limitations.md"
OUT_ISSUES = OUT_ROOT / "week7_detector_tracker_quality_audit_v27c_issues.csv"
OUT_README = OUT_ROOT / "README_detector_tracker_quality_audit_v27c.md"
OUT_NOTE = W7 / "notes" / "week7_detector_tracker_quality_audit_v27c_notes.md"

VALID_COLOURS = {"blue", "green", "cyan", "red", "pink", "purple"}


def safe_to_csv(df, path):
    df.to_csv(
        path,
        index=False,
        quoting=csv.QUOTE_ALL,
        escapechar="\\",
        lineterminator="\n",
    )


def clean(v):
    if pd.isna(v):
        return ""
    return str(v).strip()


def to_num(v, default=np.nan):
    try:
        x = pd.to_numeric(pd.Series([v]), errors="coerce").iloc[0]
        if pd.isna(x):
            return default
        return float(x)
    except Exception:
        return default


def bbox_from_row(row):
    direct_sets = [
        ("x1", "y1", "x2", "y2"),
        ("final_x1", "final_y1", "final_x2", "final_y2"),
        ("bbox_x1", "bbox_y1", "bbox_x2", "bbox_y2"),
        ("xmin", "ymin", "xmax", "ymax"),
    ]

    for cols in direct_sets:
        if all(c in row.index for c in cols):
            vals = [to_num(row[c]) for c in cols]
            if all(not pd.isna(v) for v in vals):
                return vals

    for c in ["bbox", "final_bbox", "box"]:
        if c in row.index:
            nums = re.findall(r"-?\d+(?:\.\d+)?", str(row[c]))
            if len(nums) >= 4:
                return [float(x) for x in nums[:4]]

    return None


def build_scanpoint_roi(box_df, clip_summary):
    """
    This is not a final pen polygon.
    It is a scanpoint-centered target-pig ROI proxy derived from final corrected target-pig boxes.
    Purpose: detect obvious target-pen leakage in v27b dry-run.
    """
    rows = []

    width_map = dict(zip(clip_summary["scan_frame_id"].astype(str), clip_summary["width"]))
    height_map = dict(zip(clip_summary["scan_frame_id"].astype(str), clip_summary["height"]))

    for sid, g in box_df.groupby("scan_frame_id"):
        boxes = []

        for _, r in g.iterrows():
            b = bbox_from_row(r)
            if b is None:
                continue

            x1, y1, x2, y2 = b
            if x2 <= x1 or y2 <= y1:
                continue

            boxes.append(b)

        if not boxes:
            continue

        arr = np.array(boxes, dtype=float)

        x1 = float(arr[:, 0].min())
        y1 = float(arr[:, 1].min())
        x2 = float(arr[:, 2].max())
        y2 = float(arr[:, 3].max())

        w = float(width_map.get(str(sid), 704))
        h = float(height_map.get(str(sid), 576))

        bw = x2 - x1
        bh = y2 - y1

        margin_x = max(25.0, bw * 0.10)
        margin_y = max(25.0, bh * 0.10)

        rx1 = max(0.0, x1 - margin_x)
        ry1 = max(0.0, y1 - margin_y)
        rx2 = min(w, x2 + margin_x)
        ry2 = min(h, y2 + margin_y)

        rows.append({
            "scan_frame_id": sid,
            "roi_proxy_x1": round(rx1, 4),
            "roi_proxy_y1": round(ry1, 4),
            "roi_proxy_x2": round(rx2, 4),
            "roi_proxy_y2": round(ry2, 4),
            "roi_source_box_count": len(boxes),
            "roi_note": "expanded envelope of final corrected target-pig boxes; proxy only, not final pen polygon",
        })

    return pd.DataFrame(rows)


def classify_over_detection(mean_det, expected_pigs):
    if expected_pigs <= 0:
        return "not_applicable_no_expected_pigs"

    ratio = mean_det / expected_pigs

    if ratio < 0.60:
        return "possible_under_detection"
    if ratio > 1.50:
        return "possible_over_detection_or_extra_pen_pigs"
    if ratio > 1.20:
        return "mild_over_detection"
    return "reasonable_detection_count"


def classify_fragmentation(unique_tracks, expected_pigs):
    if expected_pigs <= 0:
        return "not_applicable_no_expected_pigs"

    ratio = unique_tracks / expected_pigs

    if ratio >= 4.0:
        return "high_fragmentation"
    if ratio >= 2.5:
        return "moderate_fragmentation"
    if ratio >= 1.5:
        return "mild_fragmentation"
    return "low_fragmentation"


def classify_roi_leakage(outside_ratio):
    if pd.isna(outside_ratio):
        return "not_available"

    if outside_ratio >= 0.25:
        return "high_roi_leakage"
    if outside_ratio >= 0.10:
        return "moderate_roi_leakage"
    if outside_ratio > 0:
        return "low_roi_leakage"
    return "no_roi_leakage_detected"


def recommended_action(det_status, frag_status, roi_status):
    actions = []

    if "over_detection" in det_status or roi_status in {"moderate_roi_leakage", "high_roi_leakage"}:
        actions.append("apply_target_pen_roi_filtering")

    if frag_status in {"moderate_fragmentation", "high_fragmentation"}:
        actions.append("do_not_use_simple_iou_as_final_tracker")
        actions.append("use_improved_tracker_or_dense_frame_tracking")

    if frag_status in {"low_fragmentation", "mild_fragmentation"}:
        actions.append("simple_iou_acceptable_only_as_baseline")

    if not actions:
        actions.append("continue_with_caution")

    return " | ".join(actions)


issues = []

required_files = [SELECTED, DETECTIONS, CLIP_SUMMARY, FRAME_SUMMARY, CONTACT_INDEX, V27B_SUMMARY, BOX_LEVEL]
for p in required_files:
    if not p.exists():
        issues.append({
            "issue_type": "missing_required_file",
            "issue_detail": str(p),
        })

if issues:
    issues_df = pd.DataFrame(issues)
    safe_to_csv(issues_df, OUT_ISSUES)
    print("Missing required files; cannot continue.")
    print(issues_df.to_string(index=False))
    raise SystemExit(1)

selected = pd.read_csv(SELECTED)
det = pd.read_csv(DETECTIONS)
clip_summary = pd.read_csv(CLIP_SUMMARY)
frame_summary = pd.read_csv(FRAME_SUMMARY)
contact = pd.read_csv(CONTACT_INDEX)
v27b_summary = pd.read_csv(V27B_SUMMARY)
box = pd.read_csv(BOX_LEVEL)

for df in [selected, det, clip_summary, frame_summary, contact, box]:
    if "scan_frame_id" in df.columns:
        df["scan_frame_id"] = df["scan_frame_id"].fillna("").astype(str).str.strip()
    if "clip_id" in df.columns:
        df["clip_id"] = df["clip_id"].fillna("").astype(str).str.strip()

# Build ROI proxy.
roi = build_scanpoint_roi(box, clip_summary)

# Enrich detections with ROI proxy.
det = det.merge(roi, on="scan_frame_id", how="left")

for c in ["x1", "y1", "x2", "y2", "score"]:
    det[c] = pd.to_numeric(det[c], errors="coerce")

det["center_x"] = (det["x1"] + det["x2"]) / 2.0
det["center_y"] = (det["y1"] + det["y2"]) / 2.0

det["has_roi_proxy"] = det["roi_proxy_x1"].notna()

det["center_inside_roi_proxy"] = (
    det["has_roi_proxy"]
    & (det["center_x"] >= det["roi_proxy_x1"])
    & (det["center_x"] <= det["roi_proxy_x2"])
    & (det["center_y"] >= det["roi_proxy_y1"])
    & (det["center_y"] <= det["roi_proxy_y2"])
)

det["roi_proxy_status"] = np.where(
    ~det["has_roi_proxy"],
    "roi_proxy_missing",
    np.where(det["center_inside_roi_proxy"], "inside_roi_proxy", "outside_roi_proxy")
)

safe_to_csv(det, OUT_ENRICHED_DETECTIONS)

# Frame-level quality.
frame_quality_rows = []

if len(det):
    for (clip_id, frame_idx), g in det.groupby(["clip_id", "sampled_frame_index"]):
        outside = int((g["roi_proxy_status"] == "outside_roi_proxy").sum())
        total = int(len(g))

        frame_quality_rows.append({
            "clip_id": clip_id,
            "scan_frame_id": clean(g["scan_frame_id"].iloc[0]),
            "sampled_frame_index": int(frame_idx),
            "detections_in_frame": total,
            "unique_track_ids_in_frame": int(g["track_id"].nunique()),
            "outside_roi_proxy_detections": outside,
            "outside_roi_proxy_ratio": round(outside / total, 4) if total else 0,
            "mean_score": round(float(g["score"].mean()), 6) if total else "",
            "min_score": round(float(g["score"].min()), 6) if total else "",
            "max_score": round(float(g["score"].max()), 6) if total else "",
        })

frame_quality = pd.DataFrame(frame_quality_rows)
safe_to_csv(frame_quality, OUT_FRAME_QUALITY)

# Clip-level quality.
selected_small = selected[[
    "scan_frame_id",
    "clip_path",
    "selection_reason",
    "behaviour_codes_present",
    "pig_rows",
    "pig_rows_numeric",
]].copy()

clip_q = clip_summary.merge(
    selected_small,
    on=["scan_frame_id", "clip_path", "selection_reason", "behaviour_codes_present"],
    how="left",
)

contact_small = contact[["clip_id", "contact_sheet_path"]].copy() if "contact_sheet_path" in contact.columns else pd.DataFrame()
if len(contact_small):
    clip_q = clip_q.merge(contact_small, on="clip_id", how="left")
else:
    clip_q["contact_sheet_path"] = ""

clip_metrics = []

for _, r in clip_q.iterrows():
    clip_id = clean(r["clip_id"])
    expected_pigs = int(to_num(r.get("pig_rows_numeric", r.get("pig_rows", 0)), 0))
    mean_det = float(to_num(r.get("mean_detections_per_sampled_frame", 0), 0))
    unique_tracks = int(to_num(r.get("unique_track_ids", 0), 0))
    detections_total = int(to_num(r.get("detections_total", 0), 0))

    gdet = det[det["clip_id"] == clip_id]

    outside_count = int((gdet["roi_proxy_status"] == "outside_roi_proxy").sum()) if len(gdet) else 0
    inside_count = int((gdet["roi_proxy_status"] == "inside_roi_proxy").sum()) if len(gdet) else 0
    roi_missing_count = int((gdet["roi_proxy_status"] == "roi_proxy_missing").sum()) if len(gdet) else 0

    outside_ratio = outside_count / detections_total if detections_total else np.nan

    frag_ratio = unique_tracks / expected_pigs if expected_pigs > 0 else np.nan
    over_det_ratio = mean_det / expected_pigs if expected_pigs > 0 else np.nan

    det_status = classify_over_detection(mean_det, expected_pigs)
    frag_status = classify_fragmentation(unique_tracks, expected_pigs)
    roi_status = classify_roi_leakage(outside_ratio)

    score_mean = round(float(gdet["score"].mean()), 6) if len(gdet) else ""
    score_median = round(float(gdet["score"].median()), 6) if len(gdet) else ""
    score_p10 = round(float(gdet["score"].quantile(0.10)), 6) if len(gdet) else ""
    score_p90 = round(float(gdet["score"].quantile(0.90)), 6) if len(gdet) else ""

    clip_metrics.append({
        "clip_id": clip_id,
        "scan_frame_id": clean(r["scan_frame_id"]),
        "video_id": clean(r.get("video_id", "")),
        "selection_reason": clean(r.get("selection_reason", "")),
        "behaviour_codes_present": clean(r.get("behaviour_codes_present", "")),
        "expected_target_pig_rows": expected_pigs,
        "sampled_frames_processed": int(to_num(r.get("sampled_frames_processed", 0), 0)),
        "detections_total": detections_total,
        "mean_detections_per_sampled_frame": round(mean_det, 4),
        "unique_track_ids": unique_tracks,
        "fragmentation_ratio_unique_tracks_over_expected_pigs": round(frag_ratio, 4) if not pd.isna(frag_ratio) else "",
        "over_detection_ratio_mean_det_over_expected_pigs": round(over_det_ratio, 4) if not pd.isna(over_det_ratio) else "",
        "inside_roi_proxy_detections": inside_count,
        "outside_roi_proxy_detections": outside_count,
        "roi_proxy_missing_detections": roi_missing_count,
        "outside_roi_proxy_ratio": round(outside_ratio, 4) if not pd.isna(outside_ratio) else "",
        "score_mean": score_mean,
        "score_median": score_median,
        "score_p10": score_p10,
        "score_p90": score_p90,
        "detection_count_status": det_status,
        "tracking_fragmentation_status": frag_status,
        "roi_leakage_status": roi_status,
        "automatic_recommended_action": recommended_action(det_status, frag_status, roi_status),
        "contact_sheet_path": clean(r.get("contact_sheet_path", "")),
    })

clip_quality = pd.DataFrame(clip_metrics)
safe_to_csv(clip_quality, OUT_CLIP_QUALITY)

# Manual visual review template.
visual_rows = []

for _, r in clip_quality.iterrows():
    visual_rows.append({
        "clip_id": r["clip_id"],
        "scan_frame_id": r["scan_frame_id"],
        "selection_reason": r["selection_reason"],
        "contact_sheet_path": r["contact_sheet_path"],
        "automatic_detection_count_status": r["detection_count_status"],
        "automatic_tracking_fragmentation_status": r["tracking_fragmentation_status"],
        "automatic_roi_leakage_status": r["roi_leakage_status"],
        "automatic_recommended_action": r["automatic_recommended_action"],
        "manual_detector_quality": "pending_review",
        "manual_false_positive_level": "pending_review",
        "manual_roi_leakage_level": "pending_review",
        "manual_track_fragmentation_level": "pending_review",
        "manual_occlusion_difficulty": "pending_review",
        "manual_decision": "pending_review",
        "manual_notes": "",
    })

visual_template = pd.DataFrame(visual_rows)
safe_to_csv(visual_template, OUT_VISUAL_REVIEW_TEMPLATE)

# Global decision.
v27b_issue_count = int(to_num(v27b_summary.iloc[0].get("issue_count", 0), 0)) if len(v27b_summary) else 999
processed_clips = int(to_num(v27b_summary.iloc[0].get("processed_clips", 0), 0)) if len(v27b_summary) else 0
detections_total = int(to_num(v27b_summary.iloc[0].get("detections_total", 0), 0)) if len(v27b_summary) else 0

clips_high_frag = int(clip_quality["tracking_fragmentation_status"].eq("high_fragmentation").sum()) if len(clip_quality) else 0
clips_mod_or_high_frag = int(clip_quality["tracking_fragmentation_status"].isin(["moderate_fragmentation", "high_fragmentation"]).sum()) if len(clip_quality) else 0
clips_mod_or_high_roi = int(clip_quality["roi_leakage_status"].isin(["moderate_roi_leakage", "high_roi_leakage"]).sum()) if len(clip_quality) else 0
clips_overdet = int(clip_quality["detection_count_status"].isin(["possible_over_detection_or_extra_pen_pigs", "mild_over_detection"]).sum()) if len(clip_quality) else 0

detector_dryrun_pass = bool(v27b_issue_count == 0 and processed_clips == 5 and detections_total > 0)
simple_iou_final_pass = bool(clips_mod_or_high_frag == 0 and clips_mod_or_high_roi == 0)
ready_for_v28_roi_filtered = bool(detector_dryrun_pass)

decision = pd.DataFrame([{
    "v27c_decision": "detector_pass_tracker_needs_refinement",
    "detector_dryrun_pass": detector_dryrun_pass,
    "simple_iou_tracker_final_pass": simple_iou_final_pass,
    "processed_clips": processed_clips,
    "detections_total": detections_total,
    "clips_with_moderate_or_high_fragmentation": clips_mod_or_high_frag,
    "clips_with_high_fragmentation": clips_high_frag,
    "clips_with_moderate_or_high_roi_leakage": clips_mod_or_high_roi,
    "clips_with_over_detection_warning": clips_overdet,
    "recommended_v28_step": "ROI-filtered detector/tracker dry-run before full 72-clip tracking",
    "recommended_v28_detector_policy": "keep YOLOv8-s; apply target-pen ROI filtering; evaluate thresholds 0.25, 0.35, 0.45, 0.55",
    "recommended_v28_tracking_policy": "simple IoU only as baseline; prefer dense frame tracking and/or ByteTrack/BoT-SORT style association after ROI filtering",
    "ready_for_v28_roi_filtered_dryrun": ready_for_v28_roi_filtered,
    "generated_at": datetime.now().isoformat(timespec="seconds"),
}])

safe_to_csv(decision, OUT_DECISION)

limitations_text = """# Week 7 v27c Limitations and Professional Decision

## What v27b proves

The v27b dry-run proves that the YOLOv8-s detector can run on the extracted 10-second clips and produce pig detections without runtime errors.

## What v27b does not prove

v27b does not prove final identity tracking quality. The simple IoU tracker used in v27b is a baseline only.

## Main observed limitations

1. **Target-pen leakage**
   The detector can detect pigs outside the labelled target pen or in neighbouring pen areas. This is expected for a generic pig detector but problematic for target-pen behaviour fusion.

2. **Track fragmentation**
   The simple IoU tracker can split the same pig into multiple track IDs, especially in crowded or occluded cases.

3. **Sampling gap**
   v27b sampled every 10th frame. This is useful for fast QA, but not ideal for stable final tracking.

4. **ROI proxy limitation**
   The automatic ROI proxy in v27c is derived from final scanpoint boxes. It is useful for leakage diagnostics but should not be treated as a final pen polygon.

## Professional decision

Proceed to v28 only as an ROI-filtered dry-run, not full final tracking.

v28 should:
- keep YOLOv8-s as detector,
- add target-pen ROI filtering,
- sweep confidence thresholds,
- run denser frame tracking,
- evaluate whether simple IoU is enough after filtering,
- avoid claiming stable identity tracking until fragmentation is reduced.
"""

OUT_LIMITATIONS.write_text(limitations_text)

issues_df = pd.DataFrame(issues, columns=["issue_type", "issue_detail"])
safe_to_csv(issues_df, OUT_ISSUES)

OUT_README.write_text(
    "# Week 7 Detector / Tracker Quality Audit v27c\n\n"
    "## Purpose\n\n"
    "This step turns the v27b detector/tracker dry-run into a professional quality decision. "
    "It checks detection count, track fragmentation, and target-pen ROI leakage using an automatic scanpoint ROI proxy.\n\n"
    "## Outputs\n\n"
    "- `week7_detector_tracker_quality_audit_v27c_enriched_detections.csv`\n"
    "- `week7_detector_tracker_quality_audit_v27c_frame_quality.csv`\n"
    "- `week7_detector_tracker_quality_audit_v27c_clip_quality.csv`\n"
    "- `week7_detector_tracker_quality_audit_v27c_visual_review_template.csv`\n"
    "- `week7_detector_tracker_quality_audit_v27c_decision_summary.csv`\n"
    "- `week7_detector_tracker_quality_audit_v27c_limitations.md`\n\n"
    "## Decision\n\n"
    "The detector passes the dry-run. The simple IoU tracker is not accepted as final tracking. "
    "The next step should be ROI-filtered tracking dry-run.\n"
)

OUT_NOTE.write_text(
    "# Week 7 Detector / Tracker Quality Audit v27c\n\n"
    "## Purpose\n\n"
    "This step documents v27b dry-run quality and decides whether full tracking is justified.\n\n"
    "## Summary\n\n"
    f"- Detector dry-run pass: `{bool(decision.iloc[0]['detector_dryrun_pass'])}`\n"
    f"- Simple IoU tracker final pass: `{bool(decision.iloc[0]['simple_iou_tracker_final_pass'])}`\n"
    f"- Clips with moderate/high fragmentation: `{int(decision.iloc[0]['clips_with_moderate_or_high_fragmentation'])}`\n"
    f"- Clips with moderate/high ROI leakage: `{int(decision.iloc[0]['clips_with_moderate_or_high_roi_leakage'])}`\n"
    f"- Recommended next step: `{decision.iloc[0]['recommended_v28_step']}`\n"
    f"- Ready for v28 ROI-filtered dry-run: `{bool(decision.iloc[0]['ready_for_v28_roi_filtered_dryrun'])}`\n\n"
    "## Outputs\n\n"
    f"- Decision summary: `{OUT_DECISION}`\n"
    f"- Clip quality: `{OUT_CLIP_QUALITY}`\n"
    f"- Frame quality: `{OUT_FRAME_QUALITY}`\n"
    f"- Visual review template: `{OUT_VISUAL_REVIEW_TEMPLATE}`\n"
    f"- Limitations: `{OUT_LIMITATIONS}`\n"
    f"- Issues: `{OUT_ISSUES}`\n"
)

print("Saved:")
print(OUT_ENRICHED_DETECTIONS)
print(OUT_FRAME_QUALITY)
print(OUT_CLIP_QUALITY)
print(OUT_VISUAL_REVIEW_TEMPLATE)
print(OUT_DECISION)
print(OUT_LIMITATIONS)
print(OUT_ISSUES)
print(OUT_README)
print(OUT_NOTE)

print()
print("=== v27c decision ===")
print(decision.to_string(index=False))

print()
print("=== v27c clip quality ===")
print(clip_quality.to_string(index=False))

print()
print("=== v27c issues ===")
if len(issues_df):
    print(issues_df.to_string(index=False))
else:
    print("No issues found.")
