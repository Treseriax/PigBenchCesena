from pathlib import Path
from datetime import datetime
import csv
import importlib.util
import shutil
import pandas as pd
import numpy as np


ROOT = Path.home() / "PigBench"
W7 = ROOT / "Week7_Dataset_Validation_Identity_Tracking_Behaviour_Preparation"

V27B_SUMMARY = W7 / "outputs" / "detector_tracker_dryrun_v27b" / "week7_detector_tracker_dryrun_v27b_summary.csv"
V27C_DECISION = W7 / "outputs" / "detector_tracker_quality_audit_v27c" / "week7_detector_tracker_quality_audit_v27c_decision_summary.csv"
V28A_RECOMMENDATION = W7 / "outputs" / "roi_filtered_threshold_sweep_v28a" / "week7_roi_filtered_threshold_sweep_v28a_recommended_strategy.csv"
V28B_SUMMARY = W7 / "outputs" / "target_pen_polygon_roi_v28b" / "week7_target_pen_polygon_roi_v28b_summary.csv"
V28C_DECISION = W7 / "outputs" / "polygon_filtered_retracking_v28c" / "week7_polygon_filtered_retracking_v28c_decision_summary.csv"
V28C_CLIP = W7 / "outputs" / "polygon_filtered_retracking_v28c" / "week7_polygon_filtered_retracking_v28c_clip_summary.csv"
V28D_DECISION_FIXED = W7 / "outputs" / "dense_polygon_filtered_tracking_v28d" / "week7_dense_polygon_filtered_tracking_v28d_decision_summary_fixed.csv"
V28D_CLIP = W7 / "outputs" / "dense_polygon_filtered_tracking_v28d" / "week7_dense_polygon_filtered_tracking_v28d_clip_summary.csv"
V28D_COMPARISON = W7 / "outputs" / "dense_polygon_filtered_tracking_v28d" / "week7_dense_polygon_filtered_tracking_v28d_compare_to_v28c.csv"

V18C_FUSION = W7 / "outputs" / "behaviour_label_fusion_v18c_verified_crosswalk" / "week7_behaviour_label_fusion_v18c_box_level_dataset.csv"
V17_IDENTITY = W7 / "outputs" / "colour_identity" / "final_colour_identity_v17_fixed" / "week7_final_colour_identity_v17_fixed_locked_assignments.csv"

OUT_ROOT = W7 / "outputs" / "tracker_strategy_decision_v29a"
OUT_ROOT.mkdir(parents=True, exist_ok=True)

OUT_EVIDENCE = OUT_ROOT / "week7_tracker_strategy_decision_v29a_evidence_summary.csv"
OUT_FEASIBILITY = OUT_ROOT / "week7_tracker_strategy_decision_v29a_tracker_feasibility.csv"
OUT_OPTIONS = OUT_ROOT / "week7_tracker_strategy_decision_v29a_strategy_options.csv"
OUT_RECOMMENDATION = OUT_ROOT / "week7_tracker_strategy_decision_v29a_recommendation.csv"
OUT_REPORT = OUT_ROOT / "week7_tracker_strategy_decision_v29a_report.md"
OUT_ISSUES = OUT_ROOT / "week7_tracker_strategy_decision_v29a_issues.csv"
OUT_README = OUT_ROOT / "README_tracker_strategy_decision_v29a.md"
OUT_NOTE = W7 / "notes" / "week7_tracker_strategy_decision_v29a_notes.md"


def safe_to_csv(df, path):
    df.to_csv(
        path,
        index=False,
        quoting=csv.QUOTE_ALL,
        escapechar="\\",
        lineterminator="\n",
    )


def read_first(path):
    p = Path(path)
    if not p.exists():
        return {}
    try:
        df = pd.read_csv(p)
        if len(df):
            return df.iloc[0].to_dict()
    except Exception:
        pass
    return {}


def read_df(path):
    p = Path(path)
    if not p.exists():
        return pd.DataFrame()
    try:
        return pd.read_csv(p)
    except Exception:
        return pd.DataFrame()


def clean(v):
    if pd.isna(v):
        return ""
    return str(v).strip()


def to_float(v, default=np.nan):
    try:
        x = pd.to_numeric(pd.Series([v]), errors="coerce").iloc[0]
        if pd.isna(x):
            return default
        return float(x)
    except Exception:
        return default


def module_available(name):
    try:
        return importlib.util.find_spec(name) is not None
    except Exception:
        return False


def command_available(name):
    return shutil.which(name) is not None


issues = []

required = [
    V27B_SUMMARY,
    V27C_DECISION,
    V28A_RECOMMENDATION,
    V28B_SUMMARY,
    V28C_DECISION,
    V28C_CLIP,
    V28D_DECISION_FIXED,
    V28D_CLIP,
    V28D_COMPARISON,
]

for p in required:
    if not Path(p).exists():
        issues.append({
            "issue_type": "missing_required_file",
            "issue_detail": str(p),
        })

v27b = read_first(V27B_SUMMARY)
v27c = read_first(V27C_DECISION)
v28a = read_first(V28A_RECOMMENDATION)
v28b = read_first(V28B_SUMMARY)
v28c = read_first(V28C_DECISION)
v28d = read_first(V28D_DECISION_FIXED)

v28c_clip = read_df(V28C_CLIP)
v28d_clip = read_df(V28D_CLIP)
v28d_cmp = read_df(V28D_COMPARISON)

# Evidence summary.
evidence_rows = [
    {
        "stage": "v27b_detector_tracker_dryrun",
        "main_result": "YOLOv8-s detector works; simple IoU baseline generated outputs.",
        "processed_clips": v27b.get("processed_clips", ""),
        "sampled_frames": v27b.get("sampled_frames_processed", ""),
        "detections": v27b.get("detections_total", ""),
        "issues": v27b.get("issue_count", ""),
        "decision": "detector usable; simple IoU only baseline",
    },
    {
        "stage": "v27c_quality_audit",
        "main_result": "Detected ROI leakage and track fragmentation.",
        "processed_clips": v27c.get("processed_clips", ""),
        "sampled_frames": "",
        "detections": v27c.get("detections_total", ""),
        "issues": "",
        "decision": v27c.get("v27c_decision", ""),
    },
    {
        "stage": "v28a_rectangle_roi_sweep",
        "main_result": "Rectangle ROI useful but not final; recommended scanpoint_wide + 0.25 only as starting point.",
        "processed_clips": "",
        "sampled_frames": "",
        "detections": "",
        "issues": "",
        "decision": v28a.get("v28a_decision", ""),
    },
    {
        "stage": "v28b_manual_polygon_roi",
        "main_result": "Manual target-pen polygon ROIs confirmed.",
        "processed_clips": v28b.get("total_polygon_items", ""),
        "sampled_frames": "",
        "detections": "",
        "issues": v28b.get("issue_count", ""),
        "decision": "manual polygon ROI ready",
    },
    {
        "stage": "v28c_polygon_filtered_sparse_retracking",
        "main_result": "Polygon filtering reduced leakage but sparse simple IoU still fragmented crowded clips.",
        "processed_clips": v28c.get("processed_clips", ""),
        "sampled_frames": v28c.get("sampled_frames_processed", ""),
        "detections": v28c.get("kept_detections_total", ""),
        "issues": v28c.get("issue_count", ""),
        "decision": v28c.get("v28c_decision", ""),
    },
    {
        "stage": "v28d_dense_polygon_filtered_tracking",
        "main_result": "Dense sampling did not solve 0033/0048 fragmentation; issue is tracker association, not only sampling gap.",
        "processed_clips": v28d.get("processed_clips", ""),
        "sampled_frames": v28d.get("sampled_frames_processed", ""),
        "detections": v28d.get("kept_detections_total", ""),
        "issues": v28d.get("issue_count", ""),
        "decision": v28d.get("v28d_decision", ""),
    },
]

evidence = pd.DataFrame(evidence_rows)
safe_to_csv(evidence, OUT_EVIDENCE)

# Feasibility audit.
feasibility_rows = []

for module in ["boxmot", "ultralytics", "mmdet", "mmyolo", "mmengine", "cv2", "torch", "numpy", "pandas"]:
    feasibility_rows.append({
        "type": "python_module",
        "name": module,
        "available": module_available(module),
        "role": {
            "boxmot": "BoT-SORT / ByteTrack style tracker package if installed",
            "ultralytics": "alternative YOLO + tracker interface if compatible",
            "mmdet": "current detector inference framework",
            "mmyolo": "current YOLOv8 config support",
            "mmengine": "current model/runtime support",
            "cv2": "video and frame IO",
            "torch": "model runtime",
            "numpy": "array processing",
            "pandas": "CSV pipeline",
        }.get(module, ""),
    })

for cmd in ["ffmpeg", "ffprobe", "python", "nvidia-smi"]:
    feasibility_rows.append({
        "type": "command",
        "name": cmd,
        "available": command_available(cmd),
        "role": {
            "ffmpeg": "clip extraction / video encoding",
            "ffprobe": "video metadata",
            "python": "pipeline runtime",
            "nvidia-smi": "GPU diagnostics; GPU currently not required",
        }.get(cmd, ""),
    })

feasibility = pd.DataFrame(feasibility_rows)
safe_to_csv(feasibility, OUT_FEASIBILITY)

boxmot_available = bool(feasibility[(feasibility["name"] == "boxmot") & (feasibility["available"] == True)].shape[0])
ultralytics_available = bool(feasibility[(feasibility["name"] == "ultralytics") & (feasibility["available"] == True)].shape[0])

# Quantify core v28d problem.
dense_frag_problem_clips = []
dense_good_clips = []
ambiguous_metric_clips = []

if len(v28d_clip):
    for _, r in v28d_clip.iterrows():
        sid = clean(r.get("scan_frame_id", ""))
        frag = clean(r.get("fragmentation_status", ""))
        count = clean(r.get("count_status", ""))
        expected = to_float(r.get("expected_target_pig_rows", np.nan))
        mean_kept = to_float(r.get("mean_kept_detections_per_frame", np.nan))
        unique_tracks = to_float(r.get("unique_dense_track_ids", np.nan))

        if frag in ["moderate_fragmentation_dense", "high_fragmentation_dense"]:
            dense_frag_problem_clips.append(sid)

        if frag == "low_fragmentation_dense" and count == "reasonable_dense":
            dense_good_clips.append(sid)

        # This is our known caveat: expected rows may not represent all visible pigs in polygon.
        if expected <= 2 and mean_kept > expected * 1.5:
            ambiguous_metric_clips.append(sid)

# Strategy options.
options = pd.DataFrame([
    {
        "strategy_id": "A",
        "strategy_name": "Keep simple IoU baseline only",
        "description": "Continue with the current polygon-filtered dense simple IoU tracker.",
        "advantages": "Already implemented; transparent; reproducible; easy to debug.",
        "risks": "Fragmentation persists in crowded/occluded clips; not acceptable as final identity tracking.",
        "evidence": "v28d still has moderate/high fragmentation clips after dense sampling.",
        "implementation_complexity": "low",
        "recommended": False,
        "reason": "Useful as baseline only, not as final strategy.",
    },
    {
        "strategy_id": "B",
        "strategy_name": "Use ByteTrack / BoT-SORT style tracker",
        "description": "Use a stronger tracking-by-detection association method after polygon filtering.",
        "advantages": "Designed for detection-based MOT; can reduce ID fragmentation and missed detections.",
        "risks": "May require dependency setup; still not marker-aware; can still confuse occluded pigs.",
        "evidence": f"boxmot_available={boxmot_available}; ultralytics_available={ultralytics_available}.",
        "implementation_complexity": "medium",
        "recommended": bool(boxmot_available or ultralytics_available),
        "reason": "Good engineering candidate if dependency path is stable.",
    },
    {
        "strategy_id": "C",
        "strategy_name": "Colour-constrained tracklet identity linking",
        "description": "Treat tracker outputs as tracklets, then assign visual marker colour / behaviour pig ID to each tracklet using existing colour evidence.",
        "advantages": "Problem-specific; uses the strongest supervision signal: pig marker colour; can repair fragmented track IDs.",
        "risks": "Requires reliable colour evidence per tracklet; may fail under occlusion, blur, or missing marker visibility.",
        "evidence": "Dataset already has final visual colour identities and colour→behaviour pig ID crosswalk from v17/v18c.",
        "implementation_complexity": "medium",
        "recommended": True,
        "reason": "Best fit for this dataset because marker colours are part of the annotation design.",
    },
    {
        "strategy_id": "D",
        "strategy_name": "Hybrid improved tracker + colour-constrained linking",
        "description": "Run polygon-filtered stronger tracking, then correct/link tracklets using marker colour constraints.",
        "advantages": "Combines MOT association with dataset-specific colour identity evidence.",
        "risks": "Most complex; should be built after C prototype validates colour-linking value.",
        "evidence": "v28d proves IoU alone is insufficient; v17/v18c prove colour identity layer exists.",
        "implementation_complexity": "high",
        "recommended": True,
        "reason": "Most professional final direction, but should be staged after v29b/v29c prototypes.",
    },
])

safe_to_csv(options, OUT_OPTIONS)

# Recommendation.
if boxmot_available or ultralytics_available:
    recommended_next = "v29b ByteTrack/BoT-SORT feasibility dry-run plus v29c colour-constrained linking prototype"
else:
    recommended_next = "v29b colour-constrained tracklet identity linking prototype first; revisit ByteTrack/BoT-SORT after dependency audit"

recommendation = pd.DataFrame([{
    "v29a_decision": "simple_iou_baseline_not_final_use_colour_constrained_linking_next",
    "simple_iou_final_pass": False,
    "detector_pass": True,
    "polygon_roi_pass": True,
    "dense_sampling_solved_fragmentation": False,
    "fragmentation_problem_clips": " | ".join(dense_frag_problem_clips),
    "good_baseline_clips": " | ".join(dense_good_clips),
    "ambiguous_metric_clips": " | ".join(ambiguous_metric_clips),
    "boxmot_available": boxmot_available,
    "ultralytics_available": ultralytics_available,
    "recommended_primary_next_step": "colour-constrained tracklet identity linking prototype",
    "recommended_secondary_next_step": "stronger MOT tracker feasibility if dependency path is stable",
    "recommended_next_script": recommended_next,
    "why": "v28d showed fragmentation persists after denser sampling; marker colour is a dataset-specific identity signal and should be used to link/repair tracklets.",
    "ready_for_v29b": bool(len(v28d_clip) > 0 and int(to_float(v28d.get("issue_count", 999), 999)) == 0),
    "generated_at": datetime.now().isoformat(timespec="seconds"),
}])

safe_to_csv(recommendation, OUT_RECOMMENDATION)

# Markdown report.
report = f"""# Week 7 v29a Tracker Strategy Decision Report

## Purpose

This report decides the next tracking strategy after v27b–v28d dry-runs.

The goal is not to claim final tracking. The goal is to decide what strategy is technically justified by the evidence.

## Evidence summary

### v27b — Detector / simple IoU dry-run

- Processed clips: `{v27b.get("processed_clips", "")}`
- Sampled frames: `{v27b.get("sampled_frames_processed", "")}`
- Detections: `{v27b.get("detections_total", "")}`
- Issues: `{v27b.get("issue_count", "")}`

Decision: detector is usable, simple IoU is baseline only.

### v27c — Quality audit

- Decision: `{v27c.get("v27c_decision", "")}`
- Moderate/high fragmentation clips: `{v27c.get("clips_with_moderate_or_high_fragmentation", "")}`
- Moderate/high ROI leakage clips: `{v27c.get("clips_with_moderate_or_high_roi_leakage", "")}`

Decision: do not proceed to full tracking without ROI filtering.

### v28a — Rectangle ROI sweep

- Recommended ROI type: `{v28a.get("recommended_roi_type", "")}`
- Recommended confidence threshold: `{v28a.get("recommended_confidence_threshold", "")}`

Decision: rectangle ROI is useful diagnostically but not final.

### v28b — Manual polygon ROI

- Total polygon items: `{v28b.get("total_polygon_items", "")}`
- Confirmed polygon items: `{v28b.get("confirmed_polygon_items", "")}`
- Issues: `{v28b.get("issue_count", "")}`

Decision: manual polygon ROI is accepted for dry-run filtering.

### v28c — Sparse polygon-filtered re-tracking

- Kept detections: `{v28c.get("kept_detections_total", "")}`
- Removed detections: `{v28c.get("removed_detections_total", "")}`
- Moderate/high fragmentation after polygon: `{v28c.get("clips_with_moderate_or_high_fragmentation_after_polygon", "")}`

Decision: polygon filtering helps, but tracking still requires review.

### v28d — Dense polygon-filtered tracking

- Sampled frames: `{v28d.get("sampled_frames_processed", "")}`
- Kept detections: `{v28d.get("kept_detections_total", "")}`
- Removed detections: `{v28d.get("removed_detections_total", "")}`
- Moderate/high fragmentation dense clips: `{v28d.get("clips_with_moderate_or_high_fragmentation_dense", "")}`
- High fragmentation dense clips: `{v28d.get("clips_with_high_fragmentation_dense", "")}`

Decision: denser sampling did not solve fragmentation. The issue is not only sampling gap; it is tracker association under occlusion/crowding.

## Clip-level interpretation

Good baseline clips:
- `{ " | ".join(dense_good_clips) if dense_good_clips else "None" }`

Fragmentation problem clips:
- `{ " | ".join(dense_frag_problem_clips) if dense_frag_problem_clips else "None" }`

Ambiguous metric clips:
- `{ " | ".join(ambiguous_metric_clips) if ambiguous_metric_clips else "None" }`

## Strategy decision

### Rejected as final

Simple IoU tracking is rejected as the final identity tracker. It remains useful as a transparent baseline.

### Recommended next

Primary next step:

**Colour-constrained tracklet identity linking prototype**

Reason:

The dataset has visual marker colour identities. The final identity should not rely only on tracker ID. Instead, tracklets should be linked to marker colour and then to behaviour pig ID.

### Secondary next step

ByteTrack / BoT-SORT style tracking should be evaluated if dependency setup is stable.

Current feasibility:
- boxmot available: `{boxmot_available}`
- ultralytics available: `{ultralytics_available}`

## Proposed v29 roadmap

1. **v29b — Colour-constrained tracklet identity linking prototype**
   - Input: v28d dense polygon tracks.
   - Output: tracklet-level colour/behaviour identity candidates.
   - Goal: test whether marker colour can repair fragmented track IDs.

2. **v29c — Stronger tracker feasibility**
   - Evaluate ByteTrack/BoT-SORT only if dependencies are stable.

3. **v29d — Hybrid strategy**
   - Improved tracker + colour-constrained linking.

## Final decision

`{recommendation.iloc[0]["v29a_decision"]}`
"""

OUT_REPORT.write_text(report)

issues_df = pd.DataFrame(issues, columns=["issue_type", "issue_detail"])
safe_to_csv(issues_df, OUT_ISSUES)

OUT_README.write_text(
    "# Week 7 Tracker Strategy Decision v29a\n\n"
    "## Purpose\n\n"
    "This step consolidates v27b-v28d evidence and decides the next tracking strategy.\n\n"
    "## Main decision\n\n"
    "Simple IoU is not accepted as final tracking. The recommended next step is colour-constrained tracklet identity linking.\n\n"
    "## Outputs\n\n"
    "- `week7_tracker_strategy_decision_v29a_evidence_summary.csv`\n"
    "- `week7_tracker_strategy_decision_v29a_tracker_feasibility.csv`\n"
    "- `week7_tracker_strategy_decision_v29a_strategy_options.csv`\n"
    "- `week7_tracker_strategy_decision_v29a_recommendation.csv`\n"
    "- `week7_tracker_strategy_decision_v29a_report.md`\n"
)

OUT_NOTE.write_text(
    "# Week 7 Tracker Strategy Decision v29a\n\n"
    "## Purpose\n\n"
    "This step decides the next tracker/identity strategy after v28d.\n\n"
    "## Summary\n\n"
    f"- Simple IoU final pass: `False`\n"
    f"- Detector pass: `True`\n"
    f"- Polygon ROI pass: `True`\n"
    f"- Dense sampling solved fragmentation: `False`\n"
    f"- Fragmentation problem clips: `{recommendation.iloc[0]['fragmentation_problem_clips']}`\n"
    f"- Ambiguous metric clips: `{recommendation.iloc[0]['ambiguous_metric_clips']}`\n"
    f"- boxmot available: `{boxmot_available}`\n"
    f"- ultralytics available: `{ultralytics_available}`\n"
    f"- Recommended primary next step: `{recommendation.iloc[0]['recommended_primary_next_step']}`\n"
    f"- Ready for v29b: `{bool(recommendation.iloc[0]['ready_for_v29b'])}`\n\n"
    "## Outputs\n\n"
    f"- Recommendation: `{OUT_RECOMMENDATION}`\n"
    f"- Strategy options: `{OUT_OPTIONS}`\n"
    f"- Feasibility: `{OUT_FEASIBILITY}`\n"
    f"- Report: `{OUT_REPORT}`\n"
    f"- Issues: `{OUT_ISSUES}`\n"
)

print("Saved:")
print(OUT_EVIDENCE)
print(OUT_FEASIBILITY)
print(OUT_OPTIONS)
print(OUT_RECOMMENDATION)
print(OUT_REPORT)
print(OUT_ISSUES)
print(OUT_README)
print(OUT_NOTE)

print()
print("=== v29a recommendation ===")
print(recommendation.to_string(index=False))

print()
print("=== v29a strategy options ===")
print(options.to_string(index=False))

print()
print("=== v29a feasibility ===")
print(feasibility.to_string(index=False))

print()
print("=== v29a issues ===")
if len(issues_df):
    print(issues_df.to_string(index=False))
else:
    print("No issues found.")
