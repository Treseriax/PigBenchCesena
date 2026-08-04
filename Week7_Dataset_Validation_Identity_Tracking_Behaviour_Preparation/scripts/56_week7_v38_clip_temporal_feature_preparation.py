from pathlib import Path
from datetime import datetime
import csv
import json
import math
import pandas as pd
import numpy as np

try:
    import cv2
    CV2_AVAILABLE = True
except Exception:
    CV2_AVAILABLE = False
    cv2 = None


ROOT = Path.home() / "PigBench"
W7 = ROOT / "Week7_Dataset_Validation_Identity_Tracking_Behaviour_Preparation"

V34B = W7 / "outputs" / "week7_baseline_behaviour_representation_dataset_v34b_split_fixed"
CLIP_INDEX_IN = V34B / "week7_v34b_clip_level_multilabel_behaviour_index_split_fixed.csv"

V37 = W7 / "outputs" / "week7_crop_baseline_classifier_dryrun_v37"
V37_CONFIG_IN = V37 / "week7_v37_recommended_v38_clip_feature_preparation_config.json"

OUT_ROOT = W7 / "outputs" / "week7_clip_temporal_feature_preparation_v38"
OUT_ROOT.mkdir(parents=True, exist_ok=True)

OUT_CLIP_AUDIT = OUT_ROOT / "week7_v38_clip_loading_sampling_audit.csv"
OUT_FRAME_FEATURES = OUT_ROOT / "week7_v38_sampled_frame_feature_rows.csv"
OUT_CLIP_FEATURES = OUT_ROOT / "week7_v38_clip_temporal_lightweight_features.csv"
OUT_TARGETS = OUT_ROOT / "week7_v38_clip_multilabel_targets.csv"
OUT_MATRIX_RAW = OUT_ROOT / "week7_v38_clip_feature_matrix_raw.csv"
OUT_MATRIX_STD = OUT_ROOT / "week7_v38_clip_feature_matrix_standardized_train_stats.csv"
OUT_TRAIN_STATS = OUT_ROOT / "week7_v38_train_standardization_stats.csv"
OUT_LABEL_SUMMARY = OUT_ROOT / "week7_v38_multilabel_summary_by_split.csv"
OUT_FEATURE_SUMMARY = OUT_ROOT / "week7_v38_clip_feature_summary.csv"
OUT_V39_CONFIG = OUT_ROOT / "week7_v38_recommended_v39_clip_temporal_baseline_config.json"
OUT_REPORT = OUT_ROOT / "week7_v38_clip_temporal_feature_preparation_report.md"
OUT_DECISION = OUT_ROOT / "week7_v38_clip_temporal_feature_preparation_decision_summary.csv"
OUT_ISSUES = OUT_ROOT / "week7_v38_clip_temporal_feature_preparation_issues.csv"
OUT_NOTE = W7 / "notes" / "week7_v38_clip_temporal_feature_preparation_notes.md"


N_SAMPLE_FRAMES = 8
RESIZE_WIDTH_FOR_MOTION = 160


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


def read_df(path):
    path = Path(path)
    if not path.exists():
        return pd.DataFrame()
    try:
        return pd.read_csv(path)
    except Exception:
        return pd.DataFrame()


def read_json(path):
    path = Path(path)
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text())
    except Exception:
        return {}


def resolve_path(value):
    s = clean(value)
    if not s:
        return ""

    p = Path(s)
    if p.is_absolute():
        return str(p)

    candidates = [
        ROOT / p,
        W7 / p,
        Path.cwd() / p,
    ]

    for c in candidates:
        if c.exists():
            return str(c)

    return str(ROOT / p)


def md_table(rows, columns):
    if not rows:
        return "_No rows available._"

    out = []
    out.append("| " + " | ".join(columns) + " |")
    out.append("| " + " | ".join(["---"] * len(columns)) + " |")

    for r in rows:
        vals = []
        for c in columns:
            x = str(r.get(c, ""))
            x = x.replace("\n", " ").replace("|", "/")
            vals.append(x)
        out.append("| " + " | ".join(vals) + " |")

    return "\n".join(out)


def frame_features_rgb(img_rgb):
    arr = img_rgb.astype(np.float32)
    h, w = arr.shape[:2]

    r = arr[:, :, 0]
    g = arr[:, :, 1]
    b = arr[:, :, 2]

    gray = 0.299 * r + 0.587 * g + 0.114 * b

    feats = {
        "frame_width": float(w),
        "frame_height": float(h),
        "r_mean": float(np.mean(r)),
        "g_mean": float(np.mean(g)),
        "b_mean": float(np.mean(b)),
        "r_std": float(np.std(r)),
        "g_std": float(np.std(g)),
        "b_std": float(np.std(b)),
        "gray_mean": float(np.mean(gray)),
        "gray_std": float(np.std(gray)),
    }

    eps = 1e-6
    total = r + g + b + eps

    feats["r_ratio_mean"] = float(np.mean(r / total))
    feats["g_ratio_mean"] = float(np.mean(g / total))
    feats["b_ratio_mean"] = float(np.mean(b / total))

    if CV2_AVAILABLE:
        gray_u8 = np.clip(gray, 0, 255).astype(np.uint8)
        edges = cv2.Canny(gray_u8, 80, 160)
        feats["edge_density"] = float(np.mean(edges > 0))
        lap = cv2.Laplacian(gray_u8, cv2.CV_64F)
        feats["laplacian_variance"] = float(lap.var())
    else:
        feats["edge_density"] = np.nan
        feats["laplacian_variance"] = np.nan

    return feats


def resize_gray_for_motion(img_rgb, width=160):
    arr = img_rgb.astype(np.uint8)

    if not CV2_AVAILABLE:
        gray = 0.299 * arr[:, :, 0] + 0.587 * arr[:, :, 1] + 0.114 * arr[:, :, 2]
        return gray.astype(np.float32)

    gray = cv2.cvtColor(arr, cv2.COLOR_RGB2GRAY)
    h, w = gray.shape[:2]

    if w <= 0 or h <= 0:
        return gray.astype(np.float32)

    new_h = max(1, int(h * (width / w)))
    resized = cv2.resize(gray, (width, new_h), interpolation=cv2.INTER_AREA)
    return resized.astype(np.float32)


def read_sampled_frames(video_path, n_samples):
    if not CV2_AVAILABLE:
        raise RuntimeError("OpenCV is required for clip-level feature preparation.")

    cap = cv2.VideoCapture(str(video_path))

    if not cap.isOpened():
        raise RuntimeError("cv2.VideoCapture could not open clip.")

    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = float(cap.get(cv2.CAP_PROP_FPS))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    if frame_count <= 0:
        raise RuntimeError("Clip has no readable frames.")

    if n_samples >= frame_count:
        indices = list(range(frame_count))
    else:
        indices = np.linspace(0, frame_count - 1, n_samples).round().astype(int).tolist()

    frames = []

    for idx in indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(idx))
        ok, frame_bgr = cap.read()

        if not ok or frame_bgr is None:
            continue

        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        frames.append((int(idx), frame_rgb))

    cap.release()

    return {
        "frame_count": frame_count,
        "fps": fps,
        "width": width,
        "height": height,
        "sample_indices": indices,
        "frames": frames,
    }


def aggregate_clip_features(frame_feature_rows, motion_values):
    feats = {}

    feature_names = [
        "r_mean", "g_mean", "b_mean",
        "r_std", "g_std", "b_std",
        "gray_mean", "gray_std",
        "r_ratio_mean", "g_ratio_mean", "b_ratio_mean",
        "edge_density", "laplacian_variance",
    ]

    for name in feature_names:
        vals = np.array([r[name] for r in frame_feature_rows if name in r and pd.notna(r[name])], dtype=float)

        if len(vals):
            feats[f"{name}_clip_mean"] = float(np.mean(vals))
            feats[f"{name}_clip_std"] = float(np.std(vals))
            feats[f"{name}_clip_min"] = float(np.min(vals))
            feats[f"{name}_clip_max"] = float(np.max(vals))
        else:
            feats[f"{name}_clip_mean"] = np.nan
            feats[f"{name}_clip_std"] = np.nan
            feats[f"{name}_clip_min"] = np.nan
            feats[f"{name}_clip_max"] = np.nan

    gray_means = np.array([r["gray_mean"] for r in frame_feature_rows if "gray_mean" in r], dtype=float)

    if len(gray_means) >= 2:
        deltas = np.diff(gray_means)
        feats["gray_mean_delta_mean"] = float(np.mean(deltas))
        feats["gray_mean_delta_abs_mean"] = float(np.mean(np.abs(deltas)))
        feats["gray_mean_delta_std"] = float(np.std(deltas))
    else:
        feats["gray_mean_delta_mean"] = np.nan
        feats["gray_mean_delta_abs_mean"] = np.nan
        feats["gray_mean_delta_std"] = np.nan

    motion = np.array(motion_values, dtype=float)

    if len(motion):
        feats["motion_absdiff_mean"] = float(np.mean(motion))
        feats["motion_absdiff_std"] = float(np.std(motion))
        feats["motion_absdiff_min"] = float(np.min(motion))
        feats["motion_absdiff_max"] = float(np.max(motion))
    else:
        feats["motion_absdiff_mean"] = np.nan
        feats["motion_absdiff_std"] = np.nan
        feats["motion_absdiff_min"] = np.nan
        feats["motion_absdiff_max"] = np.nan

    return feats


issues = []

if not CV2_AVAILABLE:
    issues.append({
        "item": "opencv",
        "issue_type": "hard_cv2_not_available",
        "issue_detail": "OpenCV is required for video clip feature preparation.",
    })

clip_index = read_df(CLIP_INDEX_IN)
v37_config = read_json(V37_CONFIG_IN)

if len(clip_index) == 0:
    issues.append({
        "item": "clip_index",
        "issue_type": "hard_missing_or_empty_input",
        "issue_detail": str(CLIP_INDEX_IN),
    })

if "clip_path" not in clip_index.columns:
    issues.append({
        "item": "clip_path",
        "issue_type": "hard_missing_clip_path_column",
        "issue_detail": "clip_path column missing from v34b clip index.",
    })

if "scan_frame_id" not in clip_index.columns:
    issues.append({
        "item": "scan_frame_id",
        "issue_type": "hard_missing_scan_frame_id",
        "issue_detail": "scan_frame_id column missing from v34b clip index.",
    })

if "split" not in clip_index.columns:
    issues.append({
        "item": "split",
        "issue_type": "hard_missing_split_column",
        "issue_detail": "split column missing from v34b clip index.",
    })

label_cols = [c for c in clip_index.columns if c.startswith("label__")]

if not label_cols:
    issues.append({
        "item": "label_columns",
        "issue_type": "hard_missing_multilabel_columns",
        "issue_detail": "No label__ columns found in clip index.",
    })

if len(issues):
    issues_df = pd.DataFrame(issues, columns=["item", "issue_type", "issue_detail"])
    safe_to_csv(issues_df, OUT_ISSUES)
    print("Hard input issues:")
    print(issues_df.to_string(index=False))
    raise SystemExit(1)

clip_index = clip_index.copy().reset_index(drop=True)
clip_index["scan_frame_id"] = clip_index["scan_frame_id"].fillna("").astype(str).str.strip()
clip_index["split"] = clip_index["split"].fillna("").astype(str).str.strip()
clip_index["resolved_clip_path"] = clip_index["clip_path"].map(resolve_path)

audit_rows = []
frame_rows = []
clip_feature_rows = []
target_rows = []

for idx, r in clip_index.iterrows():
    scan_id = clean(r.get("scan_frame_id", ""))
    split = clean(r.get("split", ""))
    raw_clip_path = clean(r.get("clip_path", ""))
    resolved_clip_path = clean(r.get("resolved_clip_path", ""))

    audit = {
        "v38_clip_row_id": f"v38_clip_{idx:04d}",
        "source_row_index": int(idx),
        "scan_frame_id": scan_id,
        "split": split,
        "raw_clip_path": raw_clip_path,
        "resolved_clip_path": resolved_clip_path,
        "clip_path_exists": bool(resolved_clip_path and Path(resolved_clip_path).exists()),
        "clip_open_ok": False,
        "frame_count": "",
        "fps": "",
        "width": "",
        "height": "",
        "requested_sample_frames": int(N_SAMPLE_FRAMES),
        "sampled_frame_count": 0,
        "sampling_error": "",
    }

    target = {
        "v38_clip_row_id": f"v38_clip_{idx:04d}",
        "scan_frame_id": scan_id,
        "split": split,
        "pig_level_rows": clean(r.get("pig_level_rows", "")),
        "pig_count": clean(r.get("pig_count", "")),
        "behaviour_count": clean(r.get("behaviour_count", "")),
        "behaviour_set": clean(r.get("behaviour_set", "")),
    }

    for c in label_cols:
        target[c] = int(float(r.get(c, 0) or 0))

    target_rows.append(target)

    try:
        sample = read_sampled_frames(resolved_clip_path, N_SAMPLE_FRAMES)

        audit["clip_open_ok"] = True
        audit["frame_count"] = int(sample["frame_count"])
        audit["fps"] = float(sample["fps"])
        audit["width"] = int(sample["width"])
        audit["height"] = int(sample["height"])
        audit["sampled_frame_count"] = int(len(sample["frames"]))

        frame_feature_list = []
        motion_values = []
        prev_motion_gray = None

        for sample_i, (frame_idx, frame_rgb) in enumerate(sample["frames"]):
            feats = frame_features_rgb(frame_rgb)
            feats.update({
                "v38_clip_row_id": f"v38_clip_{idx:04d}",
                "scan_frame_id": scan_id,
                "split": split,
                "sample_order": int(sample_i),
                "frame_index": int(frame_idx),
                "clip_frame_count": int(sample["frame_count"]),
                "clip_fps": float(sample["fps"]),
            })

            frame_rows.append(feats)
            frame_feature_list.append(feats)

            motion_gray = resize_gray_for_motion(frame_rgb, RESIZE_WIDTH_FOR_MOTION)

            if prev_motion_gray is not None and motion_gray.shape == prev_motion_gray.shape:
                motion_values.append(float(np.mean(np.abs(motion_gray - prev_motion_gray))))

            prev_motion_gray = motion_gray

        clip_feats = aggregate_clip_features(frame_feature_list, motion_values)

        clip_feats.update({
            "v38_clip_row_id": f"v38_clip_{idx:04d}",
            "scan_frame_id": scan_id,
            "split": split,
            "raw_clip_path": raw_clip_path,
            "resolved_clip_path": resolved_clip_path,
            "frame_count": int(sample["frame_count"]),
            "fps": float(sample["fps"]),
            "video_width": int(sample["width"]),
            "video_height": int(sample["height"]),
            "sampled_frame_count": int(len(sample["frames"])),
            "pig_level_rows": clean(r.get("pig_level_rows", "")),
            "pig_count": clean(r.get("pig_count", "")),
            "behaviour_count": clean(r.get("behaviour_count", "")),
            "behaviour_set": clean(r.get("behaviour_set", "")),
        })

        for c in label_cols:
            clip_feats[c] = int(float(r.get(c, 0) or 0))

        clip_feature_rows.append(clip_feats)

    except Exception as e:
        audit["sampling_error"] = str(e)

        clip_feats = {
            "v38_clip_row_id": f"v38_clip_{idx:04d}",
            "scan_frame_id": scan_id,
            "split": split,
            "raw_clip_path": raw_clip_path,
            "resolved_clip_path": resolved_clip_path,
            "frame_count": np.nan,
            "fps": np.nan,
            "video_width": np.nan,
            "video_height": np.nan,
            "sampled_frame_count": 0,
            "pig_level_rows": clean(r.get("pig_level_rows", "")),
            "pig_count": clean(r.get("pig_count", "")),
            "behaviour_count": clean(r.get("behaviour_count", "")),
            "behaviour_set": clean(r.get("behaviour_set", "")),
        }

        for c in label_cols:
            clip_feats[c] = int(float(r.get(c, 0) or 0))

        clip_feature_rows.append(clip_feats)

    audit_rows.append(audit)

clip_audit = pd.DataFrame(audit_rows)
frame_features = pd.DataFrame(frame_rows)
clip_features = pd.DataFrame(clip_feature_rows)
targets = pd.DataFrame(target_rows)

safe_to_csv(clip_audit, OUT_CLIP_AUDIT)
safe_to_csv(frame_features, OUT_FRAME_FEATURES)
safe_to_csv(clip_features, OUT_CLIP_FEATURES)
safe_to_csv(targets, OUT_TARGETS)

metadata_cols = [
    "v38_clip_row_id",
    "scan_frame_id",
    "split",
    "raw_clip_path",
    "resolved_clip_path",
    "frame_count",
    "fps",
    "video_width",
    "video_height",
    "sampled_frame_count",
    "pig_level_rows",
    "pig_count",
    "behaviour_count",
    "behaviour_set",
] + label_cols

non_feature_cols = set(metadata_cols)
feature_cols = [c for c in clip_features.columns if c not in non_feature_cols]

for c in feature_cols:
    clip_features[c] = pd.to_numeric(clip_features[c], errors="coerce")

matrix_raw = clip_features[["v38_clip_row_id"] + feature_cols].copy()
safe_to_csv(matrix_raw, OUT_MATRIX_RAW)

# Train standardization only on rows with split=train and all numeric feature columns.
train_mask = clip_features["split"].astype(str) == "train"
train_features = clip_features.loc[train_mask, feature_cols].copy()

stats_rows = []

for c in feature_cols:
    mean = float(train_features[c].mean())
    std = float(train_features[c].std(ddof=0))

    if not np.isfinite(mean):
        mean = 0.0

    if not np.isfinite(std) or std == 0:
        std = 1.0

    stats_rows.append({
        "feature_name": c,
        "train_mean": mean,
        "train_std": std,
    })

stats = pd.DataFrame(stats_rows)
safe_to_csv(stats, OUT_TRAIN_STATS)

standardized = clip_features[["v38_clip_row_id"]].copy()

for _, s in stats.iterrows():
    c = s["feature_name"]
    standardized[c] = (clip_features[c] - float(s["train_mean"])) / float(s["train_std"])

safe_to_csv(standardized, OUT_MATRIX_STD)

# Label summary by split.
summary_rows = []

for split_name, g in clip_features.groupby("split", dropna=False):
    split_name = clean(split_name) or "unmapped"

    row = {
        "split": split_name,
        "clip_rows": int(len(g)),
        "model_ready_clip_rows": int((g["split"].astype(str).isin(["train", "val", "test"])).sum()),
    }

    for c in label_cols:
        row[c] = int(pd.to_numeric(g[c], errors="coerce").fillna(0).sum())

    summary_rows.append(row)

label_summary = pd.DataFrame(summary_rows)
safe_to_csv(label_summary, OUT_LABEL_SUMMARY)

# Feature summary.
feature_summary_rows = []

summary_feature_subset = [
    "gray_mean_clip_mean",
    "gray_std_clip_mean",
    "edge_density_clip_mean",
    "laplacian_variance_clip_mean",
    "motion_absdiff_mean",
    "motion_absdiff_std",
]

summary_feature_subset = [c for c in summary_feature_subset if c in clip_features.columns]

for split_name, g in clip_features.groupby("split", dropna=False):
    row = {
        "split": clean(split_name) or "unmapped",
        "clip_rows": int(len(g)),
    }

    for c in summary_feature_subset:
        vals = pd.to_numeric(g[c], errors="coerce")
        row[f"{c}_mean"] = float(vals.mean())
        row[f"{c}_std"] = float(vals.std(ddof=0))

    feature_summary_rows.append(row)

feature_summary = pd.DataFrame(feature_summary_rows)
safe_to_csv(feature_summary, OUT_FEATURE_SUMMARY)

# Decision and issues.
clips_total = int(len(clip_audit))
clips_exist = int(clip_audit["clip_path_exists"].sum()) if clips_total else 0
clips_open_ok = int(clip_audit["clip_open_ok"].sum()) if clips_total else 0
clips_open_ok_ratio = clips_open_ok / clips_total if clips_total else 0.0

sampled_frame_total = int(len(frame_features))
feature_rows_total = int(len(clip_features))

model_ready = clip_features[clip_features["split"].astype(str).isin(["train", "val", "test"])].copy()
model_ready_rows = int(len(model_ready))
unmapped_rows = int((clip_features["split"].astype(str).str.len() == 0).sum())

nan_cells = int(clip_features[feature_cols].isna().sum().sum()) if feature_cols else 0
std_nan_cells = int(standardized[feature_cols].isna().sum().sum()) if feature_cols else 0

hard_issues = []
warnings = []

if clips_open_ok_ratio < 0.95:
    hard_issues.append({
        "item": "clip_open_ok_ratio",
        "issue_type": "hard_low_clip_open_success",
        "issue_detail": f"clip_open_ok_ratio={clips_open_ok_ratio:.4f}",
    })

if sampled_frame_total == 0:
    hard_issues.append({
        "item": "sampled_frames",
        "issue_type": "hard_no_sampled_frames_created",
        "issue_detail": "No sampled frame features were created.",
    })

if model_ready_rows == 0:
    hard_issues.append({
        "item": "model_ready_clip_rows",
        "issue_type": "hard_no_model_ready_clip_rows",
        "issue_detail": "No train/val/test clip rows available.",
    })

if not set(["train", "val", "test"]).issubset(set(model_ready["split"].dropna().astype(str).unique().tolist())):
    hard_issues.append({
        "item": "split_values",
        "issue_type": "hard_missing_train_val_test",
        "issue_detail": "Model-ready clip rows do not include all train/val/test splits.",
    })

if nan_cells > 0:
    warnings.append({
        "item": "clip_feature_nan_cells",
        "issue_type": "warning_clip_feature_nan_values",
        "issue_detail": f"{nan_cells} NaN cells found in raw clip feature table.",
    })

if std_nan_cells > 0:
    warnings.append({
        "item": "standardized_feature_nan_cells",
        "issue_type": "warning_standardized_feature_nan_values",
        "issue_detail": f"{std_nan_cells} NaN cells found in standardized clip feature matrix.",
    })

if unmapped_rows > 0:
    warnings.append({
        "item": "unmapped_clip_rows",
        "issue_type": "warning_unmapped_clip_rows",
        "issue_detail": f"{unmapped_rows} clip rows do not have train/val/test split. Usually these are unlabelled/non-training clips.",
    })

all_issues = hard_issues + warnings
issues_df = pd.DataFrame(all_issues, columns=["item", "issue_type", "issue_detail"])
safe_to_csv(issues_df, OUT_ISSUES)

ready_for_v39 = (
    len(hard_issues) == 0
    and feature_rows_total > 0
    and model_ready_rows > 0
)

v39_config = {
    "stage": "v38_clip_temporal_feature_preparation",
    "claim_scope": "clip-level temporal representation preparation; not final behaviour classifier",
    "clip_features": str(OUT_CLIP_FEATURES),
    "feature_matrix_raw": str(OUT_MATRIX_RAW),
    "feature_matrix_standardized": str(OUT_MATRIX_STD),
    "targets": str(OUT_TARGETS),
    "train_standardization_stats": str(OUT_TRAIN_STATS),
    "label_columns": label_cols,
    "feature_columns": feature_cols,
    "split_column": "split",
    "id_column": "v38_clip_row_id",
    "recommended_v39_task": "clip_level_multilabel_baseline_dryrun_or_temporal_feature_audit",
    "ready_for_v39_clip_temporal_baseline": bool(ready_for_v39),
}

OUT_V39_CONFIG.write_text(json.dumps(v39_config, indent=2))

decision = pd.DataFrame([{
    "v38_decision": "clip_temporal_feature_preparation_completed" if ready_for_v39 else "clip_temporal_feature_preparation_issues_found",
    "clips_total": int(clips_total),
    "clips_exist": int(clips_exist),
    "clips_open_ok": int(clips_open_ok),
    "clips_open_ok_ratio": round(clips_open_ok_ratio, 4),
    "sampled_frame_rows": int(sampled_frame_total),
    "clip_feature_rows": int(feature_rows_total),
    "model_ready_clip_rows": int(model_ready_rows),
    "unmapped_clip_rows": int(unmapped_rows),
    "label_column_count": int(len(label_cols)),
    "feature_column_count": int(len(feature_cols)),
    "raw_feature_nan_cells": int(nan_cells),
    "standardized_feature_nan_cells": int(std_nan_cells),
    "hard_issue_count": int(len(hard_issues)),
    "warning_count": int(len(warnings)),
    "issue_count": int(len(issues_df)),
    "ready_for_v39_clip_temporal_baseline": bool(ready_for_v39),
    "generated_at": datetime.now().isoformat(timespec="seconds"),
}])

safe_to_csv(decision, OUT_DECISION)

label_table = md_table(
    label_summary.to_dict("records"),
    ["split", "clip_rows", "model_ready_clip_rows"] + label_cols,
)

feature_summary_table = md_table(
    feature_summary.to_dict("records"),
    list(feature_summary.columns),
)

report = f"""# Week 7 v38 Clip Temporal Feature Preparation Report

## Purpose

This step prepares clip-level temporal/context features from the 10-second clips.

No model is trained in this step. The purpose is to move beyond crop-only single-frame features and create a temporal representation suitable for later clip-level or multi-label baseline experiments.

## Inputs

- Clip index: `{CLIP_INDEX_IN}`
- v37 recommendation config: `{V37_CONFIG_IN}`

## Sampling policy

- Sampled frames per clip: `{N_SAMPLE_FRAMES}`
- Motion proxy: mean absolute difference between consecutive sampled grayscale frames.
- Feature type: lightweight colour, brightness, texture and temporal-difference statistics.

## Results

- Total clips: `{clips_total}`
- Clips opened OK: `{clips_open_ok}` / `{clips_total}`
- Clip open OK ratio: `{clips_open_ok_ratio:.4f}`
- Sampled frame feature rows: `{sampled_frame_total}`
- Clip feature rows: `{feature_rows_total}`
- Model-ready clip rows: `{model_ready_rows}`
- Unmapped clip rows: `{unmapped_rows}`
- Label columns: `{len(label_cols)}`
- Feature columns: `{len(feature_cols)}`
- Ready for v39: `{ready_for_v39}`

## Label summary by split

{label_table}

## Feature summary by split

{feature_summary_table}

## Outputs

- Clip loading/sampling audit: `{OUT_CLIP_AUDIT}`
- Sampled frame features: `{OUT_FRAME_FEATURES}`
- Clip temporal features: `{OUT_CLIP_FEATURES}`
- Multi-label targets: `{OUT_TARGETS}`
- Raw feature matrix: `{OUT_MATRIX_RAW}`
- Standardized feature matrix: `{OUT_MATRIX_STD}`
- Train standardization stats: `{OUT_TRAIN_STATS}`
- v39 config: `{OUT_V39_CONFIG}`

## Interpretation

The v38 feature matrix is more appropriate than crop-only features because it includes temporal context and motion proxies. However, it is still a lightweight representation and should not be overclaimed as a final behaviour model.

## Recommended next step

Proceed to v39: clip-level multi-label baseline dry-run or temporal feature audit.
"""

OUT_REPORT.write_text(report)

OUT_NOTE.write_text(
    "# Week 7 v38 Clip Temporal Feature Preparation\n\n"
    "## Summary\n\n"
    f"- Total clips: `{clips_total}`\n"
    f"- Clips opened OK: `{clips_open_ok}` / `{clips_total}`\n"
    f"- Clip open OK ratio: `{clips_open_ok_ratio:.4f}`\n"
    f"- Sampled frame rows: `{sampled_frame_total}`\n"
    f"- Clip feature rows: `{feature_rows_total}`\n"
    f"- Model-ready clip rows: `{model_ready_rows}`\n"
    f"- Unmapped clip rows: `{unmapped_rows}`\n"
    f"- Label columns: `{len(label_cols)}`\n"
    f"- Feature columns: `{len(feature_cols)}`\n"
    f"- Raw feature NaN cells: `{nan_cells}`\n"
    f"- Standardized feature NaN cells: `{std_nan_cells}`\n"
    f"- Hard issue count: `{len(hard_issues)}`\n"
    f"- Warning count: `{len(warnings)}`\n"
    f"- Ready for v39 clip temporal baseline: `{ready_for_v39}`\n\n"
    "## Outputs\n\n"
    f"- Clip audit: `{OUT_CLIP_AUDIT}`\n"
    f"- Sampled frame features: `{OUT_FRAME_FEATURES}`\n"
    f"- Clip temporal features: `{OUT_CLIP_FEATURES}`\n"
    f"- Targets: `{OUT_TARGETS}`\n"
    f"- Raw matrix: `{OUT_MATRIX_RAW}`\n"
    f"- Standardized matrix: `{OUT_MATRIX_STD}`\n"
    f"- Train stats: `{OUT_TRAIN_STATS}`\n"
    f"- Label summary: `{OUT_LABEL_SUMMARY}`\n"
    f"- Feature summary: `{OUT_FEATURE_SUMMARY}`\n"
    f"- v39 config: `{OUT_V39_CONFIG}`\n"
    f"- Report: `{OUT_REPORT}`\n"
    f"- Decision: `{OUT_DECISION}`\n"
    f"- Issues: `{OUT_ISSUES}`\n"
)

print("Saved:")
print(OUT_CLIP_AUDIT)
print(OUT_FRAME_FEATURES)
print(OUT_CLIP_FEATURES)
print(OUT_TARGETS)
print(OUT_MATRIX_RAW)
print(OUT_MATRIX_STD)
print(OUT_TRAIN_STATS)
print(OUT_LABEL_SUMMARY)
print(OUT_FEATURE_SUMMARY)
print(OUT_V39_CONFIG)
print(OUT_REPORT)
print(OUT_DECISION)
print(OUT_ISSUES)
print(OUT_NOTE)

print()
print("=== v38 decision ===")
print(decision.to_string(index=False))

print()
print("=== v38 issues ===")
if len(issues_df):
    print(issues_df.to_string(index=False))
else:
    print("No issues found.")
