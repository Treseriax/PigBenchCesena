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

try:
    from PIL import Image
    PIL_AVAILABLE = True
except Exception:
    PIL_AVAILABLE = False
    Image = None


ROOT = Path.home() / "PigBench"
W7 = ROOT / "Week7_Dataset_Validation_Identity_Tracking_Behaviour_Preparation"

V35 = W7 / "outputs" / "week7_baseline_feature_experiment_readiness_v35"
CONFIG_PATH = V35 / "experiment_configs" / "v35_crop_single_frame_baseline_ready_classes.json"

OUT_ROOT = W7 / "outputs" / "week7_crop_feature_extraction_preflight_v36a"
OUT_ROOT.mkdir(parents=True, exist_ok=True)

OUT_LOADING_AUDIT = OUT_ROOT / "week7_v36a_crop_image_loading_audit.csv"
OUT_BASELINE_SUBSET = OUT_ROOT / "week7_v36a_crop_baseline_ready_subset.csv"
OUT_CLASS_SPLIT_COUNTS = OUT_ROOT / "week7_v36a_class_split_counts.csv"
OUT_FEATURE_SCHEMA = OUT_ROOT / "week7_v36a_lightweight_feature_schema.csv"
OUT_DRYRUN_FEATURES = OUT_ROOT / "week7_v36a_dryrun_lightweight_crop_features.csv"
OUT_EXPERIMENT_CONFIG = OUT_ROOT / "week7_v36a_crop_feature_extraction_config.json"
OUT_REPORT = OUT_ROOT / "week7_v36a_crop_feature_extraction_preflight_report.md"
OUT_DECISION = OUT_ROOT / "week7_v36a_crop_feature_extraction_preflight_decision_summary.csv"
OUT_ISSUES = OUT_ROOT / "week7_v36a_crop_feature_extraction_preflight_issues.csv"
OUT_NOTE = W7 / "notes" / "week7_v36a_crop_feature_extraction_preflight_notes.md"


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


def load_image_rgb(path):
    p = Path(path)

    if CV2_AVAILABLE:
        img = cv2.imread(str(p), cv2.IMREAD_COLOR)
        if img is None:
            raise ValueError("cv2.imread returned None")
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        return img

    if PIL_AVAILABLE:
        img = Image.open(p).convert("RGB")
        return np.array(img)

    raise RuntimeError("Neither cv2 nor PIL is available for image loading.")


def extract_lightweight_features(img_rgb):
    arr = img_rgb.astype(np.float32)
    h, w = arr.shape[:2]

    if h == 0 or w == 0:
        raise ValueError("empty image array")

    features = {
        "image_width": int(w),
        "image_height": int(h),
        "aspect_ratio": float(w / h) if h else "",
        "area_pixels": int(w * h),
    }

    channels = {
        "r": arr[:, :, 0],
        "g": arr[:, :, 1],
        "b": arr[:, :, 2],
    }

    for name, ch in channels.items():
        features[f"{name}_mean"] = float(np.mean(ch))
        features[f"{name}_std"] = float(np.std(ch))
        features[f"{name}_min"] = float(np.min(ch))
        features[f"{name}_max"] = float(np.max(ch))

    gray = 0.299 * arr[:, :, 0] + 0.587 * arr[:, :, 1] + 0.114 * arr[:, :, 2]
    features["gray_mean"] = float(np.mean(gray))
    features["gray_std"] = float(np.std(gray))

    # Simple colour dominance ratios.
    eps = 1e-6
    total = arr[:, :, 0] + arr[:, :, 1] + arr[:, :, 2] + eps
    features["r_ratio_mean"] = float(np.mean(arr[:, :, 0] / total))
    features["g_ratio_mean"] = float(np.mean(arr[:, :, 1] / total))
    features["b_ratio_mean"] = float(np.mean(arr[:, :, 2] / total))

    # Very lightweight edge/texture proxy.
    if CV2_AVAILABLE:
        gray_u8 = np.clip(gray, 0, 255).astype(np.uint8)
        edges = cv2.Canny(gray_u8, 80, 160)
        features["edge_density"] = float(np.mean(edges > 0))
    else:
        features["edge_density"] = ""

    return features


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


issues = []

config = read_json(CONFIG_PATH)
if not config:
    issues.append({
        "item": "v35_crop_config",
        "issue_type": "missing_or_unreadable_config",
        "issue_detail": str(CONFIG_PATH),
    })

dataset_index = Path(config.get("dataset_index", "")) if config else Path("")
feature_input_column = config.get("feature_input_column", "crop_path") if config else "crop_path"
label_column = config.get("label_column", "behaviour_code") if config else "behaviour_code"
split_column = config.get("split_column", "split") if config else "split"
eligible_classes = config.get("eligible_classes", []) if config else []

crop = read_df(dataset_index)

if len(crop) == 0:
    issues.append({
        "item": "crop_dataset_index",
        "issue_type": "missing_or_empty_dataset_index",
        "issue_detail": str(dataset_index),
    })

required_cols = [feature_input_column, label_column, split_column]
for c in required_cols:
    if c not in crop.columns:
        issues.append({
            "item": c,
            "issue_type": "missing_required_column",
            "issue_detail": f"Column `{c}` not found in crop dataset index.",
        })

if not eligible_classes:
    issues.append({
        "item": "eligible_classes",
        "issue_type": "missing_eligible_classes",
        "issue_detail": "No eligible baseline classes found in v35 config.",
    })

if len(issues):
    issues_df = pd.DataFrame(issues, columns=["item", "issue_type", "issue_detail"])
    safe_to_csv(issues_df, OUT_ISSUES)
    print("Hard issues found before processing:")
    print(issues_df.to_string(index=False))
    raise SystemExit(1)

# ---------------------------------------------------------------------
# Image loading audit for all crop rows.
# ---------------------------------------------------------------------

audit_rows = []

for idx, r in crop.iterrows():
    raw_path = clean(r.get(feature_input_column, ""))
    resolved_path = resolve_path(raw_path)
    exists = bool(resolved_path and Path(resolved_path).exists())

    row = {
        "row_index": int(idx),
        "sample_id": clean(r.get("v34_crop_sample_id", "")),
        "scan_frame_id": clean(r.get("scan_frame_id", "")),
        "split": clean(r.get(split_column, "")),
        "behaviour_code": clean(r.get(label_column, "")),
        "raw_crop_path": raw_path,
        "resolved_crop_path": resolved_path,
        "path_exists": bool(exists),
        "image_load_ok": False,
        "image_width": "",
        "image_height": "",
        "load_error": "",
    }

    if exists:
        try:
            img = load_image_rgb(resolved_path)
            h, w = img.shape[:2]
            row["image_load_ok"] = True
            row["image_width"] = int(w)
            row["image_height"] = int(h)
        except Exception as e:
            row["load_error"] = str(e)

    audit_rows.append(row)

loading_audit = pd.DataFrame(audit_rows)
safe_to_csv(loading_audit, OUT_LOADING_AUDIT)

path_exists_count = int(loading_audit["path_exists"].sum())
load_ok_count = int(loading_audit["image_load_ok"].sum())
total_crop_rows = int(len(loading_audit))
load_ok_ratio = load_ok_count / total_crop_rows if total_crop_rows else 0.0

# ---------------------------------------------------------------------
# Baseline-ready subset.
# ---------------------------------------------------------------------

baseline_subset = crop[crop[label_column].astype(str).isin(eligible_classes)].copy()
baseline_subset = baseline_subset.reset_index(drop=True)
baseline_subset["v36a_baseline_sample_id"] = ["v36a_base_%04d" % i for i in range(len(baseline_subset))]
safe_to_csv(baseline_subset, OUT_BASELINE_SUBSET)

# ---------------------------------------------------------------------
# Class/split counts.
# ---------------------------------------------------------------------

count_rows = []

for cls in sorted(crop[label_column].dropna().astype(str).unique().tolist()):
    g = crop[crop[label_column].astype(str) == cls]

    row = {
        "behaviour_code": cls,
        "total_rows": int(len(g)),
        "eligible_for_main_crop_baseline": bool(cls in eligible_classes),
    }

    for split in ["train", "val", "test"]:
        row[f"{split}_rows"] = int((g[split_column].astype(str) == split).sum())

    count_rows.append(row)

class_split_counts = pd.DataFrame(count_rows).sort_values(
    ["eligible_for_main_crop_baseline", "total_rows", "behaviour_code"],
    ascending=[False, False, True],
)
safe_to_csv(class_split_counts, OUT_CLASS_SPLIT_COUNTS)

# ---------------------------------------------------------------------
# Dry-run lightweight features.
# ---------------------------------------------------------------------

# Balanced dry-run sample: up to 10 per eligible class per split.
dryrun_parts = []

for cls in sorted(eligible_classes):
    for split in ["train", "val", "test"]:
        g = baseline_subset[
            (baseline_subset[label_column].astype(str) == cls)
            & (baseline_subset[split_column].astype(str) == split)
        ].copy()

        if len(g):
            dryrun_parts.append(g.head(10))

if dryrun_parts:
    dryrun_sample = pd.concat(dryrun_parts, ignore_index=True)
else:
    dryrun_sample = baseline_subset.head(60).copy()

feature_rows = []

for idx, r in dryrun_sample.iterrows():
    raw_path = clean(r.get(feature_input_column, ""))
    resolved_path = resolve_path(raw_path)

    base = {
        "dryrun_row_index": int(idx),
        "v36a_baseline_sample_id": clean(r.get("v36a_baseline_sample_id", "")),
        "scan_frame_id": clean(r.get("scan_frame_id", "")),
        "split": clean(r.get(split_column, "")),
        "behaviour_code": clean(r.get(label_column, "")),
        "raw_crop_path": raw_path,
        "resolved_crop_path": resolved_path,
        "feature_extraction_ok": False,
        "feature_error": "",
    }

    try:
        img = load_image_rgb(resolved_path)
        feats = extract_lightweight_features(img)
        base.update(feats)
        base["feature_extraction_ok"] = True
    except Exception as e:
        base["feature_error"] = str(e)

    feature_rows.append(base)

dryrun_features = pd.DataFrame(feature_rows)
safe_to_csv(dryrun_features, OUT_DRYRUN_FEATURES)

features_ok_count = int(dryrun_features["feature_extraction_ok"].sum()) if len(dryrun_features) else 0
features_total = int(len(dryrun_features))
features_ok_ratio = features_ok_count / features_total if features_total else 0.0

# ---------------------------------------------------------------------
# Feature schema.
# ---------------------------------------------------------------------

feature_schema_rows = [
    {"feature_name": "image_width", "feature_group": "geometry", "description": "Crop image width in pixels."},
    {"feature_name": "image_height", "feature_group": "geometry", "description": "Crop image height in pixels."},
    {"feature_name": "aspect_ratio", "feature_group": "geometry", "description": "Width divided by height."},
    {"feature_name": "area_pixels", "feature_group": "geometry", "description": "Crop pixel area."},
    {"feature_name": "r_mean/g_mean/b_mean", "feature_group": "colour", "description": "Mean RGB channel intensity."},
    {"feature_name": "r_std/g_std/b_std", "feature_group": "colour", "description": "RGB channel standard deviation."},
    {"feature_name": "gray_mean", "feature_group": "brightness", "description": "Mean grayscale brightness."},
    {"feature_name": "gray_std", "feature_group": "contrast", "description": "Grayscale contrast proxy."},
    {"feature_name": "r_ratio_mean/g_ratio_mean/b_ratio_mean", "feature_group": "relative_colour", "description": "Mean relative RGB channel ratio."},
    {"feature_name": "edge_density", "feature_group": "texture", "description": "Canny edge density proxy if OpenCV is available."},
]

feature_schema = pd.DataFrame(feature_schema_rows)
safe_to_csv(feature_schema, OUT_FEATURE_SCHEMA)

# ---------------------------------------------------------------------
# Experiment config for v36b.
# ---------------------------------------------------------------------

v36a_config = {
    "stage": "v36a_crop_feature_extraction_preflight",
    "dataset_index": str(dataset_index),
    "baseline_subset_index": str(OUT_BASELINE_SUBSET),
    "feature_input_column": feature_input_column,
    "label_column": label_column,
    "split_column": split_column,
    "eligible_classes": eligible_classes,
    "dryrun_feature_output": str(OUT_DRYRUN_FEATURES),
    "feature_schema": str(OUT_FEATURE_SCHEMA),
    "recommended_v36b": {
        "task": "full_crop_lightweight_feature_extraction_or_deep_embedding_extraction",
        "input": str(OUT_BASELINE_SUBSET),
        "safe_first_option": "extract lightweight features for all baseline subset rows",
        "stronger_option": "extract CNN embeddings if model/dependencies are stable",
    },
    "claim_scope": "preflight only; no model training and no behaviour classifier validation",
}

OUT_EXPERIMENT_CONFIG.write_text(json.dumps(v36a_config, indent=2))

# ---------------------------------------------------------------------
# Decision, report, note.
# ---------------------------------------------------------------------

hard_issues = []
warnings = []

if load_ok_ratio < 0.95:
    hard_issues.append({
        "item": "image_loading",
        "issue_type": "low_image_load_success",
        "issue_detail": f"load_ok_ratio={load_ok_ratio:.4f}",
    })

if features_ok_ratio < 0.95:
    hard_issues.append({
        "item": "dryrun_feature_extraction",
        "issue_type": "low_feature_extraction_success",
        "issue_detail": f"features_ok_ratio={features_ok_ratio:.4f}",
    })

if len(baseline_subset) == 0:
    hard_issues.append({
        "item": "baseline_subset",
        "issue_type": "empty_baseline_subset",
        "issue_detail": "No rows matched eligible classes.",
    })

issues.extend(hard_issues)
issues.extend(warnings)

issues_df = pd.DataFrame(issues, columns=["item", "issue_type", "issue_detail"])
safe_to_csv(issues_df, OUT_ISSUES)

ready_for_v36b = (
    len(hard_issues) == 0
    and total_crop_rows > 0
    and len(baseline_subset) > 0
    and features_total > 0
)

decision = pd.DataFrame([{
    "v36a_decision": "crop_feature_extraction_preflight_passed" if ready_for_v36b else "crop_feature_extraction_preflight_issues_found",
    "cv2_available": bool(CV2_AVAILABLE),
    "pil_available": bool(PIL_AVAILABLE),
    "total_crop_rows": int(total_crop_rows),
    "crop_paths_exist": int(path_exists_count),
    "crop_images_load_ok": int(load_ok_count),
    "crop_image_load_ok_ratio": round(load_ok_ratio, 4),
    "baseline_subset_rows": int(len(baseline_subset)),
    "eligible_classes": " | ".join(eligible_classes),
    "dryrun_feature_rows": int(features_total),
    "dryrun_features_ok": int(features_ok_count),
    "dryrun_features_ok_ratio": round(features_ok_ratio, 4),
    "hard_issue_count": int(len(hard_issues)),
    "warning_count": int(len(warnings)),
    "issue_count": int(len(issues_df)),
    "ready_for_v36b_full_crop_feature_extraction": bool(ready_for_v36b),
    "generated_at": datetime.now().isoformat(timespec="seconds"),
}])

safe_to_csv(decision, OUT_DECISION)

class_table = md_table(
    class_split_counts.to_dict("records"),
    ["behaviour_code", "total_rows", "train_rows", "val_rows", "test_rows", "eligible_for_main_crop_baseline"],
)

readiness_rows = [
    {
        "check": "crop paths exist",
        "value": f"{path_exists_count}/{total_crop_rows}",
        "pass": path_exists_count == total_crop_rows,
    },
    {
        "check": "crop images load",
        "value": f"{load_ok_count}/{total_crop_rows}",
        "pass": load_ok_ratio >= 0.95,
    },
    {
        "check": "baseline subset non-empty",
        "value": str(len(baseline_subset)),
        "pass": len(baseline_subset) > 0,
    },
    {
        "check": "dryrun features extracted",
        "value": f"{features_ok_count}/{features_total}",
        "pass": features_ok_ratio >= 0.95,
    },
]

readiness_table = md_table(readiness_rows, ["check", "value", "pass"])

report = f"""# Week 7 v36a Crop Feature Extraction Preflight Report

## Purpose

This step checks whether the crop-level behaviour baseline dataset is ready for feature extraction.

It does not train a model. It verifies image loading, baseline class filtering and lightweight feature extraction on a balanced dry-run sample.

## Inputs

- v35 crop baseline config: `{CONFIG_PATH}`
- crop dataset index: `{dataset_index}`
- feature input column: `{feature_input_column}`
- label column: `{label_column}`
- split column: `{split_column}`

## Environment

- OpenCV available: `{CV2_AVAILABLE}`
- PIL available: `{PIL_AVAILABLE}`

## Readiness checks

{readiness_table}

## Class / split counts

{class_table}

## Outputs

- Image loading audit: `{OUT_LOADING_AUDIT}`
- Baseline-ready subset: `{OUT_BASELINE_SUBSET}`
- Class split counts: `{OUT_CLASS_SPLIT_COUNTS}`
- Lightweight feature schema: `{OUT_FEATURE_SCHEMA}`
- Dry-run features: `{OUT_DRYRUN_FEATURES}`
- v36b config: `{OUT_EXPERIMENT_CONFIG}`

## Interpretation

The crop-level feature extraction path is considered ready only if crop paths exist, images can be loaded, the baseline subset is non-empty and the dry-run feature extraction succeeds.

## Recommended next step

If this preflight passes, proceed to v36b: full crop-level feature extraction for the baseline-ready classes.

The safest first v36b option is lightweight full-crop feature extraction. A stronger future option is CNN embedding extraction, but that should be introduced only after the lightweight full pass is stable.
"""

OUT_REPORT.write_text(report)

OUT_NOTE.write_text(
    "# Week 7 v36a Crop Feature Extraction Preflight\n\n"
    "## Summary\n\n"
    f"- OpenCV available: `{CV2_AVAILABLE}`\n"
    f"- PIL available: `{PIL_AVAILABLE}`\n"
    f"- Total crop rows: `{total_crop_rows}`\n"
    f"- Crop paths exist: `{path_exists_count}` / `{total_crop_rows}`\n"
    f"- Crop images load OK: `{load_ok_count}` / `{total_crop_rows}`\n"
    f"- Crop image load OK ratio: `{load_ok_ratio:.4f}`\n"
    f"- Baseline subset rows: `{len(baseline_subset)}`\n"
    f"- Eligible classes: `{', '.join(eligible_classes)}`\n"
    f"- Dry-run feature rows: `{features_total}`\n"
    f"- Dry-run features OK: `{features_ok_count}` / `{features_total}`\n"
    f"- Dry-run features OK ratio: `{features_ok_ratio:.4f}`\n"
    f"- Hard issue count: `{len(hard_issues)}`\n"
    f"- Issue count: `{len(issues_df)}`\n"
    f"- Ready for v36b full crop feature extraction: `{ready_for_v36b}`\n\n"
    "## Outputs\n\n"
    f"- Loading audit: `{OUT_LOADING_AUDIT}`\n"
    f"- Baseline subset: `{OUT_BASELINE_SUBSET}`\n"
    f"- Class split counts: `{OUT_CLASS_SPLIT_COUNTS}`\n"
    f"- Feature schema: `{OUT_FEATURE_SCHEMA}`\n"
    f"- Dry-run features: `{OUT_DRYRUN_FEATURES}`\n"
    f"- v36b config: `{OUT_EXPERIMENT_CONFIG}`\n"
    f"- Report: `{OUT_REPORT}`\n"
    f"- Decision: `{OUT_DECISION}`\n"
    f"- Issues: `{OUT_ISSUES}`\n"
)

print("Saved:")
print(OUT_LOADING_AUDIT)
print(OUT_BASELINE_SUBSET)
print(OUT_CLASS_SPLIT_COUNTS)
print(OUT_FEATURE_SCHEMA)
print(OUT_DRYRUN_FEATURES)
print(OUT_EXPERIMENT_CONFIG)
print(OUT_REPORT)
print(OUT_DECISION)
print(OUT_ISSUES)
print(OUT_NOTE)

print()
print("=== v36a decision ===")
print(decision.to_string(index=False))

print()
print("=== v36a issues ===")
if len(issues_df):
    print(issues_df.to_string(index=False))
else:
    print("No issues found.")
