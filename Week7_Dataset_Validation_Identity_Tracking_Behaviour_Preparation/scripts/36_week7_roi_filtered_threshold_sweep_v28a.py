from pathlib import Path
from datetime import datetime
import csv
import re
import math

import pandas as pd
import numpy as np
from PIL import Image, ImageDraw, ImageFont


ROOT = Path.home() / "PigBench"
W7 = ROOT / "Week7_Dataset_Validation_Identity_Tracking_Behaviour_Preparation"

V27C_ROOT = W7 / "outputs" / "detector_tracker_quality_audit_v27c"
V27B_ROOT = W7 / "outputs" / "detector_tracker_dryrun_v27b"
V25_ROOT = W7 / "outputs" / "tracking_temporal_preparation_v25"

ENRICHED_DET = V27C_ROOT / "week7_detector_tracker_quality_audit_v27c_enriched_detections.csv"
CLIP_QUALITY = V27C_ROOT / "week7_detector_tracker_quality_audit_v27c_clip_quality.csv"
FRAME_SUMMARY = V27B_ROOT / "week7_detector_tracker_dryrun_v27b_frame_summary.csv"
BOX_LEVEL = W7 / "outputs" / "behaviour_label_fusion_v18c_verified_crosswalk" / "week7_behaviour_label_fusion_v18c_box_level_dataset.csv"
SCANPOINT_TIMELINE = V25_ROOT / "week7_tracking_temporal_v25_scanpoint_timeline.csv"

OUT_ROOT = W7 / "outputs" / "roi_filtered_threshold_sweep_v28a"
CONTACT_ROOT = OUT_ROOT / "roi_overlay_contact_sheets"
OUT_ROOT.mkdir(parents=True, exist_ok=True)
CONTACT_ROOT.mkdir(parents=True, exist_ok=True)

OUT_ROIS = OUT_ROOT / "week7_roi_filtered_threshold_sweep_v28a_roi_candidates.csv"
OUT_DET_ROI = OUT_ROOT / "week7_roi_filtered_threshold_sweep_v28a_detections_with_roi_candidates.csv"
OUT_BY_CLIP = OUT_ROOT / "week7_roi_filtered_threshold_sweep_v28a_by_clip.csv"
OUT_GLOBAL = OUT_ROOT / "week7_roi_filtered_threshold_sweep_v28a_global_summary.csv"
OUT_RECOMMENDATION = OUT_ROOT / "week7_roi_filtered_threshold_sweep_v28a_recommended_strategy.csv"
OUT_CONTACT_INDEX = OUT_ROOT / "week7_roi_filtered_threshold_sweep_v28a_roi_overlay_contact_sheet_index.csv"
OUT_LIMITATIONS = OUT_ROOT / "week7_roi_filtered_threshold_sweep_v28a_limitations.md"
OUT_ISSUES = OUT_ROOT / "week7_roi_filtered_threshold_sweep_v28a_issues.csv"
OUT_README = OUT_ROOT / "README_roi_filtered_threshold_sweep_v28a.md"
OUT_NOTE = W7 / "notes" / "week7_roi_filtered_threshold_sweep_v28a_notes.md"

THRESHOLDS = [0.25, 0.35, 0.45, 0.55]


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


def load_font(size=14):
    try:
        return ImageFont.truetype("DejaVuSans.ttf", size)
    except Exception:
        return ImageFont.load_default()


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


def envelope_from_boxes(boxes, width, height, margin_ratio=0.20, min_margin=50):
    if not boxes:
        return None

    arr = np.array(boxes, dtype=float)

    x1 = float(arr[:, 0].min())
    y1 = float(arr[:, 1].min())
    x2 = float(arr[:, 2].max())
    y2 = float(arr[:, 3].max())

    bw = max(1.0, x2 - x1)
    bh = max(1.0, y2 - y1)

    mx = max(float(min_margin), bw * float(margin_ratio))
    my = max(float(min_margin), bh * float(margin_ratio))

    return {
        "x1": round(max(0.0, x1 - mx), 4),
        "y1": round(max(0.0, y1 - my), 4),
        "x2": round(min(float(width), x2 + mx), 4),
        "y2": round(min(float(height), y2 + my), 4),
        "source_box_count": len(boxes),
    }


def classify_count(mean_kept, expected):
    if expected <= 0:
        return "not_applicable"

    ratio = mean_kept / expected

    if ratio < 0.50:
        return "too_few_after_filtering"
    if ratio < 0.75:
        return "mild_under_detection_after_filtering"
    if ratio > 1.50:
        return "too_many_after_filtering"
    if ratio > 1.25:
        return "mild_over_detection_after_filtering"
    return "reasonable_after_filtering"


def classify_fragmentation(unique_tracks, expected):
    if expected <= 0:
        return "not_applicable"

    ratio = unique_tracks / expected

    if ratio >= 4.0:
        return "high_fragmentation"
    if ratio >= 2.5:
        return "moderate_fragmentation"
    if ratio >= 1.5:
        return "mild_fragmentation"
    return "low_fragmentation"


def inside_roi(df, roi):
    return (
        (df["center_x"] >= roi["x1"])
        & (df["center_x"] <= roi["x2"])
        & (df["center_y"] >= roi["y1"])
        & (df["center_y"] <= roi["y2"])
    )


def draw_roi_overlay(base_path, rois_for_scanframe, out_path, title):
    font = load_font(12)
    title_font = load_font(15)

    try:
        img = Image.open(base_path).convert("RGB")
    except Exception:
        return False

    draw = ImageDraw.Draw(img)

    palette = {
        "scanpoint_loose": (255, 180, 0),
        "scanpoint_wide": (255, 60, 60),
        "video_loose": (0, 130, 255),
    }

    # Title.
    draw.rectangle((0, 0, img.width, 44), fill=(255, 255, 255))
    draw.text((8, 8), title, fill=(0, 0, 0), font=title_font)

    y = 48

    for _, r in rois_for_scanframe.iterrows():
        roi_type = clean(r["roi_type"])
        color = palette.get(roi_type, (0, 0, 0))

        x1, y1, x2, y2 = [int(round(float(r[c]))) for c in ["roi_x1", "roi_y1", "roi_x2", "roi_y2"]]

        for k in range(3):
            draw.rectangle((x1 - k, y1 - k, x2 + k, y2 + k), outline=color)

        label = f"{roi_type}"
        draw.rectangle((8, y, 260, y + 18), fill=(255, 255, 255))
        draw.text((12, y + 2), label, fill=color, font=font)
        y += 20

    img.save(out_path, quality=95)
    return True


issues = []

required = [ENRICHED_DET, CLIP_QUALITY, FRAME_SUMMARY, BOX_LEVEL, SCANPOINT_TIMELINE]
for p in required:
    if not p.exists():
        issues.append({
            "issue_type": "missing_required_file",
            "issue_detail": str(p),
        })

if issues:
    issues_df = pd.DataFrame(issues)
    safe_to_csv(issues_df, OUT_ISSUES)
    print("Missing required files.")
    print(issues_df.to_string(index=False))
    raise SystemExit(1)

det = pd.read_csv(ENRICHED_DET)
clip_q = pd.read_csv(CLIP_QUALITY)
frame_summary = pd.read_csv(FRAME_SUMMARY)
box = pd.read_csv(BOX_LEVEL)
timeline = pd.read_csv(SCANPOINT_TIMELINE)

for df in [det, clip_q, frame_summary, box, timeline]:
    if "scan_frame_id" in df.columns:
        df["scan_frame_id"] = df["scan_frame_id"].fillna("").astype(str).str.strip()
    if "clip_id" in df.columns:
        df["clip_id"] = df["clip_id"].fillna("").astype(str).str.strip()

for c in ["x1", "y1", "x2", "y2", "score"]:
    det[c] = pd.to_numeric(det[c], errors="coerce")

det["center_x"] = (det["x1"] + det["x2"]) / 2.0
det["center_y"] = (det["y1"] + det["y2"]) / 2.0

# scan_frame_id -> video_id mapping.
scan_to_video = {}
if "video_id" in timeline.columns:
    scan_to_video = dict(zip(timeline["scan_frame_id"].astype(str), timeline["video_id"].astype(str)))

box["video_id_for_roi"] = box["scan_frame_id"].map(scan_to_video).fillna("")

# Dimensions from v27b clip quality.
dim_map = {}
for _, r in clip_q.iterrows():
    # If width/height absent in v27c clip quality, default later.
    dim_map[clean(r["scan_frame_id"])] = {
        "width": int(to_num(r.get("width", 704), 704)),
        "height": int(to_num(r.get("height", 576), 576)),
        "video_id": clean(r.get("video_id", scan_to_video.get(clean(r["scan_frame_id"]), ""))),
    }

selected_scanframes = sorted(clip_q["scan_frame_id"].dropna().astype(str).unique().tolist())

roi_rows = []

for sid in selected_scanframes:
    dims = dim_map.get(sid, {"width": 704, "height": 576, "video_id": scan_to_video.get(sid, "")})
    width = dims["width"]
    height = dims["height"]
    video_id = dims["video_id"]

    # Boxes for exact scanpoint.
    scan_boxes = []
    scan_g = box[box["scan_frame_id"] == sid]

    for _, r in scan_g.iterrows():
        b = bbox_from_row(r)
        if b is not None:
            scan_boxes.append(b)

    # Boxes for same annotated video.
    video_boxes = []
    if video_id:
        vg = box[box["video_id_for_roi"].astype(str) == str(video_id)]
        for _, r in vg.iterrows():
            b = bbox_from_row(r)
            if b is not None:
                video_boxes.append(b)

    candidates = [
        ("scanpoint_loose", scan_boxes, 0.25, 60, "expanded envelope of final boxes for this scanpoint"),
        ("scanpoint_wide", scan_boxes, 0.45, 95, "wide envelope of final boxes for this scanpoint; safer for movement"),
        ("video_loose", video_boxes, 0.20, 70, "expanded envelope of all final boxes in the same annotated video"),
    ]

    for roi_type, boxes, margin_ratio, min_margin, note in candidates:
        env = envelope_from_boxes(boxes, width, height, margin_ratio=margin_ratio, min_margin=min_margin)

        if env is None:
            issues.append({
                "issue_type": "roi_candidate_failed",
                "issue_detail": f"{sid} {roi_type}: no boxes available",
            })
            continue

        roi_rows.append({
            "scan_frame_id": sid,
            "video_id": video_id,
            "roi_type": roi_type,
            "roi_x1": env["x1"],
            "roi_y1": env["y1"],
            "roi_x2": env["x2"],
            "roi_y2": env["y2"],
            "roi_width": round(env["x2"] - env["x1"], 4),
            "roi_height": round(env["y2"] - env["y1"], 4),
            "roi_source_box_count": env["source_box_count"],
            "image_width": width,
            "image_height": height,
            "roi_note": note,
        })

roi_df = pd.DataFrame(roi_rows)
safe_to_csv(roi_df, OUT_ROIS)

# Enrich detections with candidate ROI membership.
det_roi_rows = []

for _, d in det.iterrows():
    sid = clean(d["scan_frame_id"])
    base = d.to_dict()

    for _, roi in roi_df[roi_df["scan_frame_id"] == sid].iterrows():
        inside = (
            float(d["center_x"]) >= float(roi["roi_x1"])
            and float(d["center_x"]) <= float(roi["roi_x2"])
            and float(d["center_y"]) >= float(roi["roi_y1"])
            and float(d["center_y"]) <= float(roi["roi_y2"])
        )

        rr = base.copy()
        rr.update({
            "roi_type_candidate": roi["roi_type"],
            "inside_roi_candidate": bool(inside),
            "roi_candidate_x1": roi["roi_x1"],
            "roi_candidate_y1": roi["roi_y1"],
            "roi_candidate_x2": roi["roi_x2"],
            "roi_candidate_y2": roi["roi_y2"],
        })

        det_roi_rows.append(rr)

det_roi = pd.DataFrame(det_roi_rows)
safe_to_csv(det_roi, OUT_DET_ROI)

# Threshold sweep by clip and ROI type.
by_clip_rows = []

for _, cq in clip_q.iterrows():
    clip_id = clean(cq["clip_id"])
    sid = clean(cq["scan_frame_id"])
    expected = int(to_num(cq.get("expected_target_pig_rows", 0), 0))
    sampled_frames = int(to_num(cq.get("sampled_frames_processed", 0), 0))

    for roi_type in sorted(roi_df[roi_df["scan_frame_id"] == sid]["roi_type"].unique().tolist()):
        g = det_roi[(det_roi["clip_id"] == clip_id) & (det_roi["roi_type_candidate"] == roi_type)].copy()

        for th in THRESHOLDS:
            above = g[g["score"] >= th]
            kept = above[above["inside_roi_candidate"] == True]
            removed = above[above["inside_roi_candidate"] == False]

            kept_total = int(len(kept))
            removed_total = int(len(removed))
            above_total = int(len(above))

            mean_kept = kept_total / sampled_frames if sampled_frames else 0.0
            unique_tracks_kept = int(kept["track_id"].nunique()) if kept_total else 0

            count_status = classify_count(mean_kept, expected)
            frag_status = classify_fragmentation(unique_tracks_kept, expected)

            count_error_ratio = abs(mean_kept - expected) / expected if expected > 0 else np.nan
            fragmentation_ratio = unique_tracks_kept / expected if expected > 0 else np.nan
            removed_ratio_of_above = removed_total / above_total if above_total else 0.0

            by_clip_rows.append({
                "clip_id": clip_id,
                "scan_frame_id": sid,
                "video_id": clean(cq.get("video_id", "")),
                "selection_reason": clean(cq.get("selection_reason", "")),
                "behaviour_codes_present": clean(cq.get("behaviour_codes_present", "")),
                "roi_type": roi_type,
                "confidence_threshold": th,
                "expected_target_pig_rows": expected,
                "sampled_frames_processed": sampled_frames,
                "detections_above_threshold_total": above_total,
                "kept_inside_roi_total": kept_total,
                "removed_outside_roi_total": removed_total,
                "removed_outside_ratio_of_above_threshold": round(removed_ratio_of_above, 4),
                "mean_kept_inside_roi_per_frame": round(mean_kept, 4),
                "unique_track_ids_kept": unique_tracks_kept,
                "count_error_ratio": round(count_error_ratio, 4) if not pd.isna(count_error_ratio) else "",
                "fragmentation_ratio_after_filter_proxy": round(fragmentation_ratio, 4) if not pd.isna(fragmentation_ratio) else "",
                "count_status_after_filtering": count_status,
                "fragmentation_status_after_filtering_proxy": frag_status,
                "kept_score_mean": round(float(kept["score"].mean()), 6) if kept_total else "",
                "kept_score_median": round(float(kept["score"].median()), 6) if kept_total else "",
            })

by_clip = pd.DataFrame(by_clip_rows)
safe_to_csv(by_clip, OUT_BY_CLIP)

# Global summary and scoring.
global_rows = []

for (roi_type, th), g in by_clip.groupby(["roi_type", "confidence_threshold"]):
    expected_valid = g[g["expected_target_pig_rows"] > 0].copy()

    avg_count_error = float(expected_valid["count_error_ratio"].replace("", np.nan).astype(float).mean()) if len(expected_valid) else np.nan
    avg_frag = float(expected_valid["fragmentation_ratio_after_filter_proxy"].replace("", np.nan).astype(float).mean()) if len(expected_valid) else np.nan

    too_few = int(g["count_status_after_filtering"].isin(["too_few_after_filtering", "mild_under_detection_after_filtering"]).sum())
    too_many = int(g["count_status_after_filtering"].isin(["too_many_after_filtering", "mild_over_detection_after_filtering"]).sum())
    reasonable = int(g["count_status_after_filtering"].eq("reasonable_after_filtering").sum())

    moderate_high_frag = int(g["fragmentation_status_after_filtering_proxy"].isin(["moderate_fragmentation", "high_fragmentation"]).sum())
    high_frag = int(g["fragmentation_status_after_filtering_proxy"].eq("high_fragmentation").sum())

    kept_total = int(g["kept_inside_roi_total"].sum())
    removed_total = int(g["removed_outside_roi_total"].sum())
    above_total = int(g["detections_above_threshold_total"].sum())

    removed_ratio = removed_total / above_total if above_total else 0.0

    # Lower score is better.
    # Count matching is prioritized. Fragmentation proxy matters, but v28b will re-track after filtering.
    quality_score = (
        (avg_count_error if not pd.isna(avg_count_error) else 99)
        + 0.18 * too_few
        + 0.14 * too_many
        + 0.08 * moderate_high_frag
        + 0.04 * high_frag
    )

    global_rows.append({
        "roi_type": roi_type,
        "confidence_threshold": th,
        "clips_evaluated": int(len(g)),
        "reasonable_count_clips": reasonable,
        "under_count_warning_clips": too_few,
        "over_count_warning_clips": too_many,
        "moderate_or_high_fragmentation_proxy_clips": moderate_high_frag,
        "high_fragmentation_proxy_clips": high_frag,
        "detections_above_threshold_total": above_total,
        "kept_inside_roi_total": kept_total,
        "removed_outside_roi_total": removed_total,
        "removed_outside_ratio_of_above_threshold": round(removed_ratio, 4),
        "avg_count_error_ratio": round(avg_count_error, 4) if not pd.isna(avg_count_error) else "",
        "avg_fragmentation_ratio_proxy": round(avg_frag, 4) if not pd.isna(avg_frag) else "",
        "quality_score_lower_is_better": round(float(quality_score), 6),
    })

global_df = pd.DataFrame(global_rows).sort_values(
    ["quality_score_lower_is_better", "confidence_threshold"],
    ascending=[True, True],
)
safe_to_csv(global_df, OUT_GLOBAL)

best = global_df.iloc[0].to_dict() if len(global_df) else {}

recommendation = pd.DataFrame([{
    "v28a_decision": "roi_filtering_required_before_tracking",
    "recommended_roi_type": best.get("roi_type", ""),
    "recommended_confidence_threshold": best.get("confidence_threshold", ""),
    "recommended_strategy_note": "Use this as v28b dry-run starting point only; not final tracking yet.",
    "detector_policy": "Keep YOLOv8-s. Apply target-pen ROI filtering before tracker association.",
    "tracking_policy": "Re-run tracking after ROI filtering with denser sampling. Simple IoU remains baseline only.",
    "why_not_full_tracking_yet": "v27c showed ROI leakage and fragmentation; v28a only selects filtering strategy.",
    "best_quality_score_lower_is_better": best.get("quality_score_lower_is_better", ""),
    "best_reasonable_count_clips": best.get("reasonable_count_clips", ""),
    "best_under_count_warning_clips": best.get("under_count_warning_clips", ""),
    "best_over_count_warning_clips": best.get("over_count_warning_clips", ""),
    "best_removed_outside_ratio": best.get("removed_outside_ratio_of_above_threshold", ""),
    "ready_for_v28b_roi_filtered_tracking_dryrun": bool(len(global_df) > 0),
    "generated_at": datetime.now().isoformat(timespec="seconds"),
}])

safe_to_csv(recommendation, OUT_RECOMMENDATION)

# ROI overlay contact sheets.
contact_rows = []

for sid in selected_scanframes:
    fs = frame_summary[frame_summary["scan_frame_id"] == sid].copy()

    if len(fs) == 0:
        continue

    # Choose middle sampled frame for ROI visual.
    fs["dist_to_middle"] = (pd.to_numeric(fs["sampled_frame_index"], errors="coerce") - 120).abs()
    fs = fs.sort_values("dist_to_middle")

    base_path = clean(fs.iloc[0]["annotated_frame_path"])
    clip_id = clean(fs.iloc[0]["clip_id"])
    rois = roi_df[roi_df["scan_frame_id"] == sid].copy()

    out_path = CONTACT_ROOT / f"{sid}_roi_candidates_overlay_v28a.jpg"
    made = draw_roi_overlay(
        base_path,
        rois,
        out_path,
        f"{sid} ROI candidates on v27b annotated frame",
    )

    if made:
        contact_rows.append({
            "scan_frame_id": sid,
            "clip_id": clip_id,
            "source_annotated_frame_path": base_path,
            "roi_overlay_path": str(out_path),
            "roi_count": int(len(rois)),
        })

contact_index = pd.DataFrame(contact_rows)
safe_to_csv(contact_index, OUT_CONTACT_INDEX)

limitations = """# Week 7 v28a Limitations

## What this step does

v28a evaluates candidate ROI filters and confidence thresholds using the detections produced by v27b.

## What this step does not do

v28a does not produce final tracking. It does not claim stable pig identity over time.

## Important caveats

1. The ROI candidates are proxy ROIs derived from final corrected target-pig boxes.
2. The scanpoint-level ROI can be too tight when pigs move during a 10-second clip.
3. The video-level ROI is safer for motion but may still require visual validation.
4. Fragmentation metrics are proxy metrics because the track IDs were generated before ROI filtering.
5. v28b must re-run association after ROI filtering to fairly evaluate tracking stability.

## Professional decision

Use v28a to choose an initial ROI + threshold strategy, then run v28b as ROI-filtered tracking dry-run.
"""

OUT_LIMITATIONS.write_text(limitations)

issues_df = pd.DataFrame(issues, columns=["issue_type", "issue_detail"])
safe_to_csv(issues_df, OUT_ISSUES)

OUT_README.write_text(
    "# Week 7 ROI-filtered Threshold Sweep v28a\n\n"
    "## Purpose\n\n"
    "This step evaluates candidate target-pen ROI filters and confidence thresholds before re-running tracking.\n\n"
    "## Outputs\n\n"
    "- `week7_roi_filtered_threshold_sweep_v28a_roi_candidates.csv`\n"
    "- `week7_roi_filtered_threshold_sweep_v28a_detections_with_roi_candidates.csv`\n"
    "- `week7_roi_filtered_threshold_sweep_v28a_by_clip.csv`\n"
    "- `week7_roi_filtered_threshold_sweep_v28a_global_summary.csv`\n"
    "- `week7_roi_filtered_threshold_sweep_v28a_recommended_strategy.csv`\n"
    "- `roi_overlay_contact_sheets/`\n"
)

OUT_NOTE.write_text(
    "# Week 7 ROI-filtered Threshold Sweep v28a\n\n"
    "## Purpose\n\n"
    "This step selects an initial ROI + confidence threshold strategy before ROI-filtered tracking.\n\n"
    "## Summary\n\n"
    f"- ROI candidates generated: `{int(len(roi_df))}`\n"
    f"- Detection-ROI rows: `{int(len(det_roi))}`\n"
    f"- Threshold combinations evaluated: `{int(len(global_df))}`\n"
    f"- Recommended ROI type: `{recommendation.iloc[0]['recommended_roi_type']}`\n"
    f"- Recommended confidence threshold: `{recommendation.iloc[0]['recommended_confidence_threshold']}`\n"
    f"- Ready for v28b ROI-filtered tracking dry-run: `{bool(recommendation.iloc[0]['ready_for_v28b_roi_filtered_tracking_dryrun'])}`\n\n"
    "## Outputs\n\n"
    f"- Recommendation: `{OUT_RECOMMENDATION}`\n"
    f"- Global summary: `{OUT_GLOBAL}`\n"
    f"- By-clip summary: `{OUT_BY_CLIP}`\n"
    f"- ROI candidates: `{OUT_ROIS}`\n"
    f"- ROI overlays: `{CONTACT_ROOT}`\n"
    f"- Limitations: `{OUT_LIMITATIONS}`\n"
    f"- Issues: `{OUT_ISSUES}`\n"
)

print("Saved:")
print(OUT_ROIS)
print(OUT_DET_ROI)
print(OUT_BY_CLIP)
print(OUT_GLOBAL)
print(OUT_RECOMMENDATION)
print(OUT_CONTACT_INDEX)
print(OUT_LIMITATIONS)
print(OUT_ISSUES)
print(OUT_README)
print(OUT_NOTE)

print()
print("=== v28a recommendation ===")
print(recommendation.to_string(index=False))

print()
print("=== v28a global summary ===")
print(global_df.to_string(index=False))

print()
print("=== v28a issues ===")
if len(issues_df):
    print(issues_df.to_string(index=False))
else:
    print("No issues found.")
