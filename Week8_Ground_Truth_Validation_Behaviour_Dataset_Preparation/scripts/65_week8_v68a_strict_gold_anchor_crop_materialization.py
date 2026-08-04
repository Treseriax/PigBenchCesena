from pathlib import Path
from datetime import datetime
import csv
import json
import hashlib
import re
import zipfile
import pandas as pd
import numpy as np


ROOT = Path.home() / "PigBench"
W8 = ROOT / "Week8_Ground_Truth_Validation_Behaviour_Dataset_Preparation"

V67 = W8 / "outputs" / "v67_week8_report_ready_package" / "Week8_Report_Ready_GT_v2_Package"
V67C = W8 / "outputs" / "v67c_dataset_card_gt_documentation"

STRICT = V67 / "data" / "week8_final_gt_v2_strict_gold_objects_for_classification.csv"
V66A_OBJECTS = V67 / "data" / "week8_v66a_final_gt_v2_object_table.csv"
V67C_DECISION = V67C / "week8_v67c_decision_summary.csv"

OUT = W8 / "outputs" / "v68a_strict_gold_anchor_crop_materialization"
DATASET = OUT / "Week8_StrictGold_AnchorFrame_Crop_Dataset"
CROPS_TIGHT = DATASET / "crops_tight"
CROPS_CONTEXT = DATASET / "crops_context10"
ANCHOR_FRAMES = DATASET / "anchor_frames"
METADATA_DIR = DATASET / "metadata"
AUDIT_DIR = DATASET / "audit"
NOTES = W8 / "notes"
REPORTS = W8 / "reports"
PROGRESS = W8 / "progress"

for p in [OUT, DATASET, CROPS_TIGHT, CROPS_CONTEXT, ANCHOR_FRAMES, METADATA_DIR, AUDIT_DIR, NOTES, REPORTS, PROGRESS]:
    p.mkdir(parents=True, exist_ok=True)

OUT_METADATA = METADATA_DIR / "week8_v68a_strict_gold_anchor_crop_metadata.csv"
OUT_FAILED = AUDIT_DIR / "week8_v68a_failed_crop_rows.csv"
OUT_SCANFRAME = AUDIT_DIR / "week8_v68a_scanframe_anchor_frame_summary.csv"
OUT_BEHAVIOUR_DIST = AUDIT_DIR / "week8_v68a_behaviour_distribution.csv"
OUT_SPLIT_DIST = AUDIT_DIR / "week8_v68a_split_distribution.csv"
OUT_QA = AUDIT_DIR / "week8_v68a_crop_materialization_quality_checks.csv"
OUT_MANIFEST = DATASET / "week8_v68a_crop_dataset_manifest.json"
OUT_README = DATASET / "README_Week8_StrictGold_AnchorFrame_Crop_Dataset.md"
OUT_DECISION = OUT / "week8_v68a_decision_summary.csv"
OUT_ISSUES = OUT / "week8_v68a_issues.csv"
OUT_ZIP = OUT / "Week8_StrictGold_AnchorFrame_Crop_Dataset.zip"
OUT_SHA256 = OUT / "Week8_StrictGold_AnchorFrame_Crop_Dataset.sha256"
OUT_NOTE = NOTES / "week8_v68a_strict_gold_anchor_crop_materialization_notes.md"
OUT_REPORT = REPORTS / "week8_v68a_strict_gold_anchor_crop_materialization_report.md"
OUT_PROGRESS = PROGRESS / "week8_experiment_progress_log.csv"


def safe_to_csv(df, path):
    df.to_csv(path, index=False, quoting=csv.QUOTE_ALL, escapechar="\\", lineterminator="\n")


def clean(x):
    if pd.isna(x):
        return ""
    s = str(x).strip()
    if s.lower() in ["nan", "none", "null"]:
        return ""
    return s


def read_csv_clean(path):
    df = pd.read_csv(path).fillna("")
    for c in df.columns:
        if df[c].dtype == object:
            df[c] = df[c].map(clean)
    return df


def bool_true(value):
    return str(value).strip().lower() == "true"


def sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def safe_filename(s):
    s = clean(s)
    s = re.sub(r"[^A-Za-z0-9_.-]+", "_", s)
    s = re.sub(r"_+", "_", s)
    return s.strip("_")


def parse_clip_times_from_path(path_str):
    name = Path(path_str).name
    m = re.search(r"centerf_(\d+).*?__(\d+)ms_(\d+)ms", name)
    if not m:
        return None
    return {
        "center_frame_source": int(m.group(1)),
        "clip_start_ms": int(m.group(2)),
        "clip_end_ms": int(m.group(3)),
    }


def estimate_source_fps_by_video(rows):
    fps_by_video = {}
    for video_id, g in rows.groupby("video_id"):
        vals = []
        for _, r in g.iterrows():
            parsed = parse_clip_times_from_path(r.get("clip_path", ""))
            if not parsed:
                continue
            center_frame = parsed["center_frame_source"]
            start_ms = parsed["clip_start_ms"]
            end_ms = parsed["clip_end_ms"]
            midpoint_sec = ((start_ms + end_ms) / 2.0) / 1000.0
            if center_frame > 0 and midpoint_sec > 0:
                vals.append(center_frame / midpoint_sec)
        if vals:
            fps_by_video[video_id] = float(np.median(vals))
    return fps_by_video


def compute_anchor_frame_idx(row, frame_count, output_fps, source_fps_by_video):
    parsed = parse_clip_times_from_path(row.get("clip_path", ""))
    if not parsed:
        return max(min(frame_count // 2, frame_count - 1), 0), "fallback_midpoint_no_filename_parse"

    center_frame = parsed["center_frame_source"]
    start_sec = parsed["clip_start_ms"] / 1000.0
    end_sec = parsed["clip_end_ms"] / 1000.0
    duration_sec = max(end_sec - start_sec, 0.001)

    video_id = row.get("video_id", "")
    source_fps = source_fps_by_video.get(video_id)

    if source_fps is None or source_fps <= 0:
        if center_frame == 0:
            center_sec = 0.0
            source_fps = 0.0
        else:
            midpoint_sec = (start_sec + end_sec) / 2.0
            source_fps = center_frame / midpoint_sec if midpoint_sec > 0 else 0.0
            center_sec = center_frame / source_fps if source_fps > 0 else (start_sec + duration_sec / 2.0)
            return int(np.clip(round((center_sec - start_sec) * output_fps), 0, frame_count - 1)), "estimated_from_filename_midpoint"
    else:
        center_sec = center_frame / source_fps

    anchor_rel_sec = center_sec - start_sec
    anchor_idx = int(round(anchor_rel_sec * output_fps))
    anchor_idx = int(np.clip(anchor_idx, 0, frame_count - 1))

    if center_frame == 0:
        method = "source_center_frame_zero_anchor_start"
    else:
        method = "source_frame_minus_clip_start"

    return anchor_idx, method


def clamp_bbox(x1, y1, x2, y2, w, h):
    x1 = int(np.floor(max(0, min(w - 1, x1))))
    y1 = int(np.floor(max(0, min(h - 1, y1))))
    x2 = int(np.ceil(max(0, min(w, x2))))
    y2 = int(np.ceil(max(0, min(h, y2))))
    return x1, y1, x2, y2


def expand_bbox(x1, y1, x2, y2, w, h, pad_ratio=0.10):
    bw = x2 - x1
    bh = y2 - y1
    pad_x = bw * pad_ratio
    pad_y = bh * pad_ratio
    return clamp_bbox(x1 - pad_x, y1 - pad_y, x2 + pad_x, y2 + pad_y, w, h)


issues = []

for p in [STRICT, V66A_OBJECTS, V67C_DECISION]:
    if not p.exists():
        issues.append({
            "item": str(p),
            "issue_type": "hard_missing_required_input",
            "issue_detail": "Required input missing for v68a crop materialization.",
            "severity": "hard",
        })

if issues:
    issues_df = pd.DataFrame(issues)
    safe_to_csv(issues_df, OUT_ISSUES)
    decision = pd.DataFrame([{
        "v68a_decision": "strict_gold_anchor_crop_materialization_blocked",
        "hard_issue_count": int((issues_df["severity"] == "hard").sum()),
        "ready_for_v68b_crop_qa_gallery": False,
        "generated_at": datetime.now().isoformat(timespec="seconds"),
    }])
    safe_to_csv(decision, OUT_DECISION)
    print(decision.to_string(index=False))
    raise SystemExit(1)


v67c_decision = read_csv_clean(V67C_DECISION)
if len(v67c_decision) == 0 or not bool_true(v67c_decision.iloc[0].get("ready_for_v68a_crop_dataset_materialization", "")):
    issues.append({
        "item": str(V67C_DECISION),
        "issue_type": "hard_v67c_not_ready",
        "issue_detail": "v67c must be ready before v68a crop materialization.",
        "severity": "hard",
    })

strict = read_csv_clean(STRICT)
objects = read_csv_clean(V66A_OBJECTS)

# Use v66a object table because it contains clip_path and propagation metadata.
cols_from_objects = [
    "canonical_gt_object_id",
    "clip_path",
    "source_video_path",
    "generated_frame_count",
    "fps_used",
    "start_sec",
    "end_sec",
    "duration_sec",
    "final_gt_v2_category_v66a",
    "classification_gate_v66a",
    "recommended_split",
    "bbox_propagation_status",
]

merge_cols = [c for c in cols_from_objects if c in objects.columns]

df = strict.merge(
    objects[merge_cols].drop_duplicates("canonical_gt_object_id"),
    on="canonical_gt_object_id",
    how="left",
    suffixes=("", "_v66a"),
)

# Keep only strict rows by rule.
strict_rule = (
    (df["manual_gt_v2_status"] == "gold_usable")
    & (df["manual_classification_use"] == "use_for_classification")
    & (df["manual_bbox_status"] == "bbox_ok")
    & (df["manual_identity_status"] == "identity_confirmed")
    & (df["manual_assigned_candidate_box_id"].astype(str).str.strip() != "")
)

df = df[strict_rule].copy()

# Safety.
if len(df) != 372:
    issues.append({
        "item": "strict_gold_input_rows",
        "issue_type": "hard_unexpected_strict_gold_count",
        "issue_detail": f"Expected 372 strict gold rows after strict rule, found {len(df)}.",
        "severity": "hard",
    })

try:
    import cv2
except Exception as e:
    issues.append({
        "item": "opencv",
        "issue_type": "hard_opencv_import_failed",
        "issue_detail": str(e),
        "severity": "hard",
    })

if any(i["severity"] == "hard" for i in issues):
    issues_df = pd.DataFrame(issues)
    safe_to_csv(issues_df, OUT_ISSUES)
    decision = pd.DataFrame([{
        "v68a_decision": "strict_gold_anchor_crop_materialization_blocked",
        "strict_gold_input_rows": int(len(df)),
        "hard_issue_count": int((issues_df["severity"] == "hard").sum()),
        "ready_for_v68b_crop_qa_gallery": False,
        "generated_at": datetime.now().isoformat(timespec="seconds"),
    }])
    safe_to_csv(decision, OUT_DECISION)
    print(decision.to_string(index=False))
    print(issues_df.to_string(index=False))
    raise SystemExit(1)


source_fps_by_video = estimate_source_fps_by_video(df)

metadata_rows = []
failed_rows = []
scanframe_rows = []

anchor_cache = {}

for scan_id, g in df.groupby("scan_frame_id"):
    g = g.copy()
    clip_path = clean(g["clip_path"].iloc[0]) if "clip_path" in g.columns else ""

    if not clip_path or not Path(clip_path).exists():
        failed_rows.append({
            "scan_frame_id": scan_id,
            "canonical_gt_object_id": "",
            "failure_type": "clip_path_missing_or_not_found",
            "failure_detail": clip_path,
        })
        scanframe_rows.append({
            "scan_frame_id": scan_id,
            "clip_path": clip_path,
            "frame_read_status": "failed_clip_missing",
            "anchor_frame_idx": "",
            "object_rows": len(g),
            "successful_crops": 0,
            "failed_crops": len(g),
        })
        continue

    cap = cv2.VideoCapture(clip_path)
    if not cap.isOpened():
        failed_rows.append({
            "scan_frame_id": scan_id,
            "canonical_gt_object_id": "",
            "failure_type": "video_open_failed",
            "failure_detail": clip_path,
        })
        scanframe_rows.append({
            "scan_frame_id": scan_id,
            "clip_path": clip_path,
            "frame_read_status": "failed_video_open",
            "anchor_frame_idx": "",
            "object_rows": len(g),
            "successful_crops": 0,
            "failed_crops": len(g),
        })
        continue

    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    output_fps = float(cap.get(cv2.CAP_PROP_FPS))
    if frame_count <= 0:
        frame_count = int(float(g["generated_frame_count"].iloc[0])) if "generated_frame_count" in g.columns and clean(g["generated_frame_count"].iloc[0]) else 0
    if output_fps <= 0:
        output_fps = 25.0

    anchor_idx, anchor_method = compute_anchor_frame_idx(g.iloc[0].to_dict(), frame_count, output_fps, source_fps_by_video)

    cap.set(cv2.CAP_PROP_POS_FRAMES, anchor_idx)
    ok, frame = cap.read()

    if not ok or frame is None:
        fallback_idx = max(min(frame_count // 2, frame_count - 1), 0)
        cap.set(cv2.CAP_PROP_POS_FRAMES, fallback_idx)
        ok, frame = cap.read()
        anchor_idx = fallback_idx
        anchor_method = "fallback_midpoint_read_after_anchor_failed"

    cap.release()

    if not ok or frame is None:
        failed_rows.append({
            "scan_frame_id": scan_id,
            "canonical_gt_object_id": "",
            "failure_type": "anchor_frame_read_failed",
            "failure_detail": clip_path,
        })
        scanframe_rows.append({
            "scan_frame_id": scan_id,
            "clip_path": clip_path,
            "frame_read_status": "failed_frame_read",
            "anchor_frame_idx": anchor_idx,
            "object_rows": len(g),
            "successful_crops": 0,
            "failed_crops": len(g),
        })
        continue

    h, w = frame.shape[:2]

    anchor_frame_path = ANCHOR_FRAMES / f"{safe_filename(scan_id)}__anchor_{anchor_idx:06d}.jpg"
    cv2.imwrite(str(anchor_frame_path), frame, [int(cv2.IMWRITE_JPEG_QUALITY), 95])

    successful = 0
    failed = 0

    for _, r in g.iterrows():
        obj_id = clean(r["canonical_gt_object_id"])
        behaviour = safe_filename(r["behaviour_code"])
        split = clean(r.get("recommended_split", ""))
        if split not in ["train", "val", "test"]:
            split = "unspecified"

        try:
            x1 = float(r["manual_bbox_x1"])
            y1 = float(r["manual_bbox_y1"])
            x2 = float(r["manual_bbox_x2"])
            y2 = float(r["manual_bbox_y2"])
        except Exception as e:
            failed += 1
            failed_rows.append({
                "scan_frame_id": scan_id,
                "canonical_gt_object_id": obj_id,
                "failure_type": "bbox_parse_failed",
                "failure_detail": str(e),
            })
            continue

        x1c, y1c, x2c, y2c = clamp_bbox(x1, y1, x2, y2, w, h)
        if x2c <= x1c or y2c <= y1c:
            failed += 1
            failed_rows.append({
                "scan_frame_id": scan_id,
                "canonical_gt_object_id": obj_id,
                "failure_type": "invalid_clamped_bbox",
                "failure_detail": f"{x1},{y1},{x2},{y2} -> {x1c},{y1c},{x2c},{y2c}",
            })
            continue

        x1p, y1p, x2p, y2p = expand_bbox(x1c, y1c, x2c, y2c, w, h, 0.10)

        tight_crop = frame[y1c:y2c, x1c:x2c]
        context_crop = frame[y1p:y2p, x1p:x2p]

        if tight_crop.size == 0 or context_crop.size == 0:
            failed += 1
            failed_rows.append({
                "scan_frame_id": scan_id,
                "canonical_gt_object_id": obj_id,
                "failure_type": "empty_crop",
                "failure_detail": f"tight={tight_crop.shape if tight_crop is not None else None}, context={context_crop.shape if context_crop is not None else None}",
            })
            continue

        fname = f"{safe_filename(obj_id)}__{behaviour}.jpg"

        tight_dir = CROPS_TIGHT / split / behaviour
        context_dir = CROPS_CONTEXT / split / behaviour
        tight_dir.mkdir(parents=True, exist_ok=True)
        context_dir.mkdir(parents=True, exist_ok=True)

        tight_path = tight_dir / fname
        context_path = context_dir / fname

        cv2.imwrite(str(tight_path), tight_crop, [int(cv2.IMWRITE_JPEG_QUALITY), 95])
        cv2.imwrite(str(context_path), context_crop, [int(cv2.IMWRITE_JPEG_QUALITY), 95])

        successful += 1

        metadata_rows.append({
            "dataset_version": "week8_v68a_strict_gold_anchor_frame_crops",
            "canonical_gt_object_id": obj_id,
            "scan_frame_id": scan_id,
            "video_id": clean(r["video_id"]),
            "clip_path": clip_path,
            "anchor_frame_path": str(anchor_frame_path),
            "anchor_frame_idx": anchor_idx,
            "anchor_frame_method": anchor_method,
            "video_frame_count": frame_count,
            "video_fps": output_fps,
            "image_width": w,
            "image_height": h,
            "canonical_colour_label_norm": clean(r["canonical_colour_label_norm"]),
            "behaviour_code": clean(r["behaviour_code"]),
            "recommended_split": split,
            "manual_assigned_candidate_box_id": clean(r["manual_assigned_candidate_box_id"]),
            "manual_bbox_x1": x1,
            "manual_bbox_y1": y1,
            "manual_bbox_x2": x2,
            "manual_bbox_y2": y2,
            "clamped_bbox_x1": x1c,
            "clamped_bbox_y1": y1c,
            "clamped_bbox_x2": x2c,
            "clamped_bbox_y2": y2c,
            "context10_bbox_x1": x1p,
            "context10_bbox_y1": y1p,
            "context10_bbox_x2": x2p,
            "context10_bbox_y2": y2p,
            "tight_crop_width": x2c - x1c,
            "tight_crop_height": y2c - y1c,
            "context_crop_width": x2p - x1p,
            "context_crop_height": y2p - y1p,
            "tight_crop_path": str(tight_path),
            "context10_crop_path": str(context_path),
            "gt_source": "manual_gt_v2_strict_gold",
            "tracking_used": "no",
            "claim_boundary": "anchor_frame_crop_dataset_for_baseline_classification_only",
        })

    scanframe_rows.append({
        "scan_frame_id": scan_id,
        "video_id": clean(g["video_id"].iloc[0]),
        "clip_path": clip_path,
        "frame_read_status": "ok",
        "anchor_frame_idx": anchor_idx,
        "anchor_frame_method": anchor_method,
        "video_frame_count": frame_count,
        "video_fps": output_fps,
        "object_rows": len(g),
        "successful_crops": successful,
        "failed_crops": failed,
        "anchor_frame_path": str(anchor_frame_path),
    })

metadata = pd.DataFrame(metadata_rows)
failed = pd.DataFrame(failed_rows, columns=["scan_frame_id", "canonical_gt_object_id", "failure_type", "failure_detail"])
scanframe_summary = pd.DataFrame(scanframe_rows)

safe_to_csv(metadata, OUT_METADATA)
safe_to_csv(failed, OUT_FAILED)
safe_to_csv(scanframe_summary, OUT_SCANFRAME)

if len(metadata):
    behaviour_dist = (
        metadata.groupby(["behaviour_code"])
        .size()
        .reset_index(name="crop_count")
        .sort_values("crop_count", ascending=False)
    )

    split_dist = (
        metadata.groupby(["recommended_split", "behaviour_code"])
        .size()
        .reset_index(name="crop_count")
        .sort_values(["recommended_split", "behaviour_code"])
    )
else:
    behaviour_dist = pd.DataFrame(columns=["behaviour_code", "crop_count"])
    split_dist = pd.DataFrame(columns=["recommended_split", "behaviour_code", "crop_count"])

safe_to_csv(behaviour_dist, OUT_BEHAVIOUR_DIST)
safe_to_csv(split_dist, OUT_SPLIT_DIST)

qa_rows = []


def add_qa(name, expected, actual, passed, severity, detail):
    qa_rows.append({
        "check_name": name,
        "expected": str(expected),
        "actual": str(actual),
        "passed": bool(passed),
        "severity": severity,
        "detail": detail,
    })


add_qa("strict_gold_input_rows", 372, len(df), len(df) == 372, "hard", "Input strict gold rows should be 372.")
add_qa("metadata_rows", 372, len(metadata), len(metadata) == 372, "hard", "One successful crop metadata row per strict object is expected.")
add_qa("failed_crop_rows", 0, len(failed), len(failed) == 0, "hard", "No crop extraction failure should occur.")
add_qa("scanframes_with_crops", 70, metadata["scan_frame_id"].nunique() if len(metadata) else 0, metadata["scan_frame_id"].nunique() == 70 if len(metadata) else False, "hard", "Strict gold objects should come from 70 scanframes.")
add_qa("behaviour_classes", 11, metadata["behaviour_code"].nunique() if len(metadata) else 0, metadata["behaviour_code"].nunique() == 11 if len(metadata) else False, "hard", "Strict crop dataset should preserve 11 behaviour classes.")
add_qa("tight_crop_files_exist", len(metadata), int(metadata["tight_crop_path"].map(lambda x: Path(x).exists()).sum()) if len(metadata) else 0, int(metadata["tight_crop_path"].map(lambda x: Path(x).exists()).sum()) == len(metadata) if len(metadata) else False, "hard", "All tight crop files should exist.")
add_qa("context_crop_files_exist", len(metadata), int(metadata["context10_crop_path"].map(lambda x: Path(x).exists()).sum()) if len(metadata) else 0, int(metadata["context10_crop_path"].map(lambda x: Path(x).exists()).sum()) == len(metadata) if len(metadata) else False, "hard", "All context crop files should exist.")
add_qa("train_val_test_only", "train/val/test", ";".join(sorted(metadata["recommended_split"].unique())) if len(metadata) else "", set(metadata["recommended_split"].unique()) <= {"train", "val", "test"} if len(metadata) else False, "hard", "All crops should have train/val/test split.")
add_qa("tracking_not_used", "no", ";".join(sorted(metadata["tracking_used"].unique())) if len(metadata) else "", set(metadata["tracking_used"].unique()) == {"no"} if len(metadata) else False, "hard", "Crop dataset should not use tracking-derived labels.")

qa = pd.DataFrame(qa_rows)
safe_to_csv(qa, OUT_QA)

hard_quality_failures = int(((qa["severity"] == "hard") & (~qa["passed"])).sum())
if hard_quality_failures:
    issues.append({
        "item": "v68a_quality_checks",
        "issue_type": "hard_crop_materialization_quality_check_failed",
        "issue_detail": f"{hard_quality_failures} hard crop materialization checks failed.",
        "severity": "hard",
    })

issues_df = pd.DataFrame(issues, columns=["item", "issue_type", "issue_detail", "severity"])
safe_to_csv(issues_df, OUT_ISSUES)

hard_issue_count = int((issues_df["severity"] == "hard").sum()) if len(issues_df) else 0
warning_count = int((issues_df["severity"] == "warning").sum()) if len(issues_df) else 0
info_count = int((issues_df["severity"] == "info").sum()) if len(issues_df) else 0

manifest = {
    "dataset_version": "week8_v68a_strict_gold_anchor_frame_crops",
    "created_at": datetime.now().isoformat(timespec="seconds"),
    "source_gt": str(STRICT),
    "strict_gold_objects": int(len(df)),
    "successful_crop_rows": int(len(metadata)),
    "failed_crop_rows": int(len(failed)),
    "scanframes_with_crops": int(metadata["scan_frame_id"].nunique()) if len(metadata) else 0,
    "behaviour_classes": int(metadata["behaviour_code"].nunique()) if len(metadata) else 0,
    "crop_types": ["tight", "context10"],
    "tracking_used": False,
    "usage": "baseline/proof-of-concept classification only",
    "claim_boundary": "not production-grade classification; not tracking GT; not full raw Unibo archive",
    "files": {
        "metadata": str(OUT_METADATA),
        "failed_rows": str(OUT_FAILED),
        "scanframe_summary": str(OUT_SCANFRAME),
        "behaviour_distribution": str(OUT_BEHAVIOUR_DIST),
        "split_distribution": str(OUT_SPLIT_DIST),
    },
}

OUT_MANIFEST.write_text(json.dumps(manifest, indent=2, ensure_ascii=False))

readme = f"""# Week8 Strict-Gold Anchor-Frame Crop Dataset

## Summary

This dataset contains anchor-frame crops generated only from strict gold GT v2 objects.

## Key Numbers

- Strict gold input objects: {len(df)}
- Successful crop rows: {len(metadata)}
- Failed crop rows: {len(failed)}
- Scanframes with strict crops: {metadata["scan_frame_id"].nunique() if len(metadata) else 0}
- Behaviour classes: {metadata["behaviour_code"].nunique() if len(metadata) else 0}

## Crop Types

- crops_tight: exact manual GT v2 bbox crop
- crops_context10: bbox crop with 10 percent context padding

## Source of Truth

Manual GT v2 is the source of truth. Tracking is not used for crop labels.

## Usage

Use this dataset only for baseline/proof-of-concept classification.

Do not claim production-grade classification.

Do not mix caution or nonusable rows into training/evaluation.
"""

OUT_README.write_text(readme)

if OUT_ZIP.exists():
    OUT_ZIP.unlink()

with zipfile.ZipFile(OUT_ZIP, "w", compression=zipfile.ZIP_DEFLATED) as z:
    for p in sorted(DATASET.rglob("*")):
        if p.is_file():
            z.write(p, p.relative_to(OUT))

zip_hash = sha256_file(OUT_ZIP)
OUT_SHA256.write_text(f"{zip_hash}  {OUT_ZIP.name}\n")

decision = pd.DataFrame([{
    "v68a_decision": "strict_gold_anchor_crop_materialization_completed" if hard_issue_count == 0 else "strict_gold_anchor_crop_materialization_has_blocking_issues",
    "strict_gold_input_rows": int(len(df)),
    "successful_crop_rows": int(len(metadata)),
    "failed_crop_rows": int(len(failed)),
    "scanframes_with_crops": int(metadata["scan_frame_id"].nunique()) if len(metadata) else 0,
    "behaviour_classes": int(metadata["behaviour_code"].nunique()) if len(metadata) else 0,
    "tight_crop_count": int(metadata["tight_crop_path"].map(lambda x: Path(x).exists()).sum()) if len(metadata) else 0,
    "context10_crop_count": int(metadata["context10_crop_path"].map(lambda x: Path(x).exists()).sum()) if len(metadata) else 0,
    "dataset_dir": str(DATASET),
    "zip_path": str(OUT_ZIP),
    "zip_sha256": zip_hash,
    "hard_quality_failures": hard_quality_failures,
    "hard_issue_count": hard_issue_count,
    "warning_count": warning_count,
    "info_count": info_count,
    "issue_count": int(len(issues_df)),
    "ready_for_v68b_crop_qa_gallery": bool(hard_issue_count == 0),
    "generated_at": datetime.now().isoformat(timespec="seconds"),
}])

safe_to_csv(decision, OUT_DECISION)

OUT_NOTE.write_text(
    "# Week 8 v68a Strict-Gold Anchor-Frame Crop Materialization\n\n"
    f"- v68a decision: {decision.iloc[0]['v68a_decision']}\n"
    f"- Strict gold input rows: {len(df)}\n"
    f"- Successful crop rows: {len(metadata)}\n"
    f"- Failed crop rows: {len(failed)}\n"
    f"- Scanframes with crops: {metadata['scan_frame_id'].nunique() if len(metadata) else 0}\n"
    f"- Behaviour classes: {metadata['behaviour_code'].nunique() if len(metadata) else 0}\n"
    f"- Tight crop count: {decision.iloc[0]['tight_crop_count']}\n"
    f"- Context10 crop count: {decision.iloc[0]['context10_crop_count']}\n"
    f"- Hard issues: {hard_issue_count}\n"
    f"- Dataset: {DATASET}\n"
    f"- ZIP: {OUT_ZIP}\n"
    f"- SHA256: {zip_hash}\n"
    f"- Ready for v68b crop QA gallery: {bool(hard_issue_count == 0)}\n\n"
    "Only strict gold rows are materialized. Tracking is not used for labels.\n"
)

OUT_REPORT.write_text(
    "# Week 8 v68a Strict-Gold Anchor-Frame Crop Materialization Report\n\n"
    f"Decision: {decision.iloc[0]['v68a_decision']}\n\n"
    f"Dataset directory: {DATASET}\n\n"
    f"ZIP: {OUT_ZIP}\n\n"
    f"SHA256: {zip_hash}\n"
)

progress_row = pd.DataFrame([{
    "date": datetime.now().date().isoformat(),
    "stage": "v68a",
    "task_name": "Strict-gold anchor-frame crop dataset materialization",
    "status": "PASS" if hard_issue_count == 0 else "NEEDS_FIX",
    "input_summary": str(STRICT),
    "output_summary": str(DATASET),
    "hard_issues": hard_issue_count,
    "warnings": warning_count,
    "next_action": "Create v68b crop QA gallery and visual inspection package." if hard_issue_count == 0 else "Fix crop materialization issues.",
}])

if OUT_PROGRESS.exists():
    old = pd.read_csv(OUT_PROGRESS)
    progress = pd.concat([old, progress_row], ignore_index=True)
else:
    progress = progress_row

safe_to_csv(progress, OUT_PROGRESS)

print("Saved:")
print(OUT_METADATA)
print(OUT_FAILED)
print(OUT_SCANFRAME)
print(OUT_BEHAVIOUR_DIST)
print(OUT_SPLIT_DIST)
print(OUT_QA)
print(OUT_MANIFEST)
print(OUT_ZIP)
print(OUT_SHA256)
print(OUT_DECISION)
print(OUT_NOTE)

print()
print("=== v68a decision ===")
print(decision.to_string(index=False))

print()
print("=== QA ===")
print(qa.to_string(index=False))

print()
print("=== behaviour distribution ===")
print(behaviour_dist.to_string(index=False))

print()
print("=== split distribution ===")
print(split_dist.to_string(index=False))

print()
print("=== issues ===")
if len(issues_df):
    print(issues_df.to_string(index=False))
else:
    print("No issues found.")
