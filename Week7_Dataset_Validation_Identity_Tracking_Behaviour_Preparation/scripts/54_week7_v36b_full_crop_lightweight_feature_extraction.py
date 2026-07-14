from pathlib import Path
from datetime import datetime
import csv
import json
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

V36A2 = W7 / "outputs" / "week7_crop_duplicate_leakage_integrity_audit_v36a2"
CONFIG_IN = V36A2 / "week7_v36a2_recommended_v36b_feature_extraction_config.json"

OUT_ROOT = W7 / "outputs" / "week7_full_crop_lightweight_feature_extraction_v36b"
OUT_ROOT.mkdir(parents=True, exist_ok=True)

OUT_FEATURES = OUT_ROOT / "week7_v36b_full_crop_lightweight_features.csv"
OUT_FEATURE_MATRIX_RAW = OUT_ROOT / "week7_v36b_feature_matrix_raw.csv"
OUT_FEATURE_MATRIX_STANDARDIZED = OUT_ROOT / "week7_v36b_feature_matrix_standardized_train_stats.csv"
OUT_METADATA = OUT_ROOT / "week7_v36b_feature_metadata.csv"
OUT_TRAIN_STATS = OUT_ROOT / "week7_v36b_train_standardization_stats.csv"
OUT_CLASS_SPLIT_COUNTS = OUT_ROOT / "week7_v36b_class_split_counts.csv"
OUT_FEATURE_SUMMARY = OUT_ROOT / "week7_v36b_feature_summary_by_class_split.csv"
OUT_MANIFEST = OUT_ROOT / "week7_v36b_feature_file_manifest.csv"
OUT_V37_CONFIG = OUT_ROOT / "week7_v36b_recommended_v37_baseline_classifier_config.json"
OUT_REPORT = OUT_ROOT / "week7_v36b_full_crop_lightweight_feature_extraction_report.md"
OUT_DECISION = OUT_ROOT / "week7_v36b_full_crop_lightweight_feature_extraction_decision_summary.csv"
OUT_ISSUES = OUT_ROOT / "week7_v36b_full_crop_lightweight_feature_extraction_issues.csv"
OUT_NOTE = W7 / "notes" / "week7_v36b_full_crop_lightweight_feature_extraction_notes.md"


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


def read_json(path):
    path = Path(path)
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text())
    except Exception:
        return {}


def read_df(path):
    path = Path(path)
    if not path.exists():
        return pd.DataFrame()
    try:
        return pd.read_csv(path)
    except Exception:
        return pd.DataFrame()


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
        return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    if PIL_AVAILABLE:
        return np.array(Image.open(p).convert("RGB"))

    raise RuntimeError("Neither cv2 nor PIL is available.")


def extract_lightweight_features(img_rgb):
    arr = img_rgb.astype(np.float32)
    h, w = arr.shape[:2]

    if h == 0 or w == 0:
        raise ValueError("empty image array")

    features = {
        "image_width": float(w),
        "image_height": float(h),
        "aspect_ratio": float(w / h),
        "area_pixels": float(w * h),
    }

    r = arr[:, :, 0]
    g = arr[:, :, 1]
    b = arr[:, :, 2]

    channels = {"r": r, "g": g, "b": b}

    for name, ch in channels.items():
        features[f"{name}_mean"] = float(np.mean(ch))
        features[f"{name}_std"] = float(np.std(ch))
        features[f"{name}_min"] = float(np.min(ch))
        features[f"{name}_max"] = float(np.max(ch))
        features[f"{name}_p10"] = float(np.percentile(ch, 10))
        features[f"{name}_p50"] = float(np.percentile(ch, 50))
        features[f"{name}_p90"] = float(np.percentile(ch, 90))

    gray = 0.299 * r + 0.587 * g + 0.114 * b

    features["gray_mean"] = float(np.mean(gray))
    features["gray_std"] = float(np.std(gray))
    features["gray_min"] = float(np.min(gray))
    features["gray_max"] = float(np.max(gray))
    features["gray_p10"] = float(np.percentile(gray, 10))
    features["gray_p50"] = float(np.percentile(gray, 50))
    features["gray_p90"] = float(np.percentile(gray, 90))

    eps = 1e-6
    total = r + g + b + eps

    rr = r / total
    gr = g / total
    br = b / total

    features["r_ratio_mean"] = float(np.mean(rr))
    features["g_ratio_mean"] = float(np.mean(gr))
    features["b_ratio_mean"] = float(np.mean(br))
    features["r_ratio_std"] = float(np.std(rr))
    features["g_ratio_std"] = float(np.std(gr))
    features["b_ratio_std"] = float(np.std(br))

    features["red_minus_green_mean"] = float(np.mean(r - g))
    features["red_minus_blue_mean"] = float(np.mean(r - b))
    features["green_minus_blue_mean"] = float(np.mean(g - b))

    if CV2_AVAILABLE:
        gray_u8 = np.clip(gray, 0, 255).astype(np.uint8)
        edges = cv2.Canny(gray_u8, 80, 160)
        features["edge_density"] = float(np.mean(edges > 0))

        lap = cv2.Laplacian(gray_u8, cv2.CV_64F)
        features["laplacian_variance"] = float(lap.var())
    else:
        features["edge_density"] = np.nan
        features["laplacian_variance"] = np.nan

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

config = read_json(CONFIG_IN)

if not config:
    issues.append({
        "item": "v36a2_config",
        "issue_type": "hard_missing_or_unreadable_config",
        "issue_detail": str(CONFIG_IN),
    })

input_path = Path(config.get("recommended_input_for_v36b", "")) if config else Path("")
label_col = config.get("label_column", "behaviour_code") if config else "behaviour_code"
split_col = config.get("split_column", "split") if config else "split"
feature_input_col = config.get("feature_input_column", "crop_path") if config else "crop_path"
resolved_path_col = config.get("resolved_path_column", "resolved_crop_path") if config else "resolved_crop_path"
eligible_classes = config.get("eligible_classes", []) if config else []

df = read_df(input_path)

if len(df) == 0:
    issues.append({
        "item": "v36b_input_index",
        "issue_type": "hard_missing_or_empty_input",
        "issue_detail": str(input_path),
    })

for col in [label_col, split_col, feature_input_col]:
    if col not in df.columns:
        issues.append({
            "item": col,
            "issue_type": "hard_missing_required_column",
            "issue_detail": f"Column `{col}` missing from input index.",
        })

if resolved_path_col not in df.columns:
    df[resolved_path_col] = df[feature_input_col].map(resolve_path) if feature_input_col in df.columns else ""

if len(issues):
    issues_df = pd.DataFrame(issues, columns=["item", "issue_type", "issue_detail"])
    safe_to_csv(issues_df, OUT_ISSUES)
    print("Hard input issue:")
    print(issues_df.to_string(index=False))
    raise SystemExit(1)

df = df.copy().reset_index(drop=True)
df[label_col] = df[label_col].fillna("").astype(str).str.strip()
df[split_col] = df[split_col].fillna("").astype(str).str.strip()
df[resolved_path_col] = df[resolved_path_col].fillna("").astype(str).str.strip()

if "scan_frame_id" in df.columns:
    df["scan_frame_id"] = df["scan_frame_id"].fillna("").astype(str).str.strip()

feature_rows = []

for idx, r in df.iterrows():
    raw_path = clean(r.get(feature_input_col, ""))
    resolved_path = clean(r.get(resolved_path_col, "")) or resolve_path(raw_path)

    base = {
        "v36b_feature_row_id": f"v36b_feat_{idx:04d}",
        "source_row_index": int(idx),
        "scan_frame_id": clean(r.get("scan_frame_id", "")),
        "split": clean(r.get(split_col, "")),
        "behaviour_code": clean(r.get(label_col, "")),
        "raw_crop_path": raw_path,
        "resolved_crop_path": resolved_path,
        "path_exists": bool(resolved_path and Path(resolved_path).exists()),
        "feature_extraction_ok": False,
        "feature_error": "",
    }

    for optional_col in [
        "behaviour_pig_id",
        "visual_marker_colour",
        "v34_crop_sample_id",
        "v36a_baseline_sample_id",
        "v36a2_dedup_sample_id",
    ]:
        if optional_col in df.columns:
            base[optional_col] = clean(r.get(optional_col, ""))

    try:
        img = load_image_rgb(resolved_path)
        feats = extract_lightweight_features(img)
        base.update(feats)
        base["feature_extraction_ok"] = True
    except Exception as e:
        base["feature_error"] = str(e)

    feature_rows.append(base)

features = pd.DataFrame(feature_rows)
safe_to_csv(features, OUT_FEATURES)

feature_cols = [
    c for c in features.columns
    if c not in [
        "v36b_feature_row_id",
        "source_row_index",
        "scan_frame_id",
        "split",
        "behaviour_code",
        "raw_crop_path",
        "resolved_crop_path",
        "path_exists",
        "feature_extraction_ok",
        "feature_error",
        "behaviour_pig_id",
        "visual_marker_colour",
        "v34_crop_sample_id",
        "v36a_baseline_sample_id",
        "v36a2_dedup_sample_id",
    ]
]

# Ensure numeric features.
for c in feature_cols:
    features[c] = pd.to_numeric(features[c], errors="coerce")

metadata_cols = [
    c for c in [
        "v36b_feature_row_id",
        "source_row_index",
        "scan_frame_id",
        "split",
        "behaviour_code",
        "behaviour_pig_id",
        "visual_marker_colour",
        "v34_crop_sample_id",
        "v36a_baseline_sample_id",
        "v36a2_dedup_sample_id",
        "raw_crop_path",
        "resolved_crop_path",
    ]
    if c in features.columns
]

metadata = features[metadata_cols].copy()
safe_to_csv(metadata, OUT_METADATA)

matrix_raw = features[["v36b_feature_row_id"] + feature_cols].copy()
safe_to_csv(matrix_raw, OUT_FEATURE_MATRIX_RAW)

# Train-based standardization.
train_mask = features["split"].astype(str) == "train"
train_features = features.loc[train_mask, feature_cols].copy()

stats_rows = []

for c in feature_cols:
    mean = float(train_features[c].mean())
    std = float(train_features[c].std(ddof=0))

    if not np.isfinite(std) or std == 0:
        std = 1.0

    stats_rows.append({
        "feature_name": c,
        "train_mean": mean,
        "train_std": std,
    })

stats = pd.DataFrame(stats_rows)
safe_to_csv(stats, OUT_TRAIN_STATS)

standardized = features[["v36b_feature_row_id"]].copy()

for _, s in stats.iterrows():
    c = s["feature_name"]
    standardized[c] = (features[c] - float(s["train_mean"])) / float(s["train_std"])

safe_to_csv(standardized, OUT_FEATURE_MATRIX_STANDARDIZED)

# Counts.
count_rows = []

for cls in sorted(features["behaviour_code"].dropna().astype(str).unique().tolist()):
    g = features[features["behaviour_code"] == cls]

    count_rows.append({
        "behaviour_code": cls,
        "total_rows": int(len(g)),
        "train_rows": int((g["split"] == "train").sum()),
        "val_rows": int((g["split"] == "val").sum()),
        "test_rows": int((g["split"] == "test").sum()),
    })

class_split_counts = pd.DataFrame(count_rows)
safe_to_csv(class_split_counts, OUT_CLASS_SPLIT_COUNTS)

# Feature summary by class/split.
summary_rows = []

summary_feature_subset = [
    "image_width",
    "image_height",
    "aspect_ratio",
    "area_pixels",
    "gray_mean",
    "gray_std",
    "edge_density",
    "laplacian_variance",
]

summary_feature_subset = [c for c in summary_feature_subset if c in features.columns]

for (cls, split), g in features.groupby(["behaviour_code", "split"]):
    row = {
        "behaviour_code": cls,
        "split": split,
        "row_count": int(len(g)),
    }

    for c in summary_feature_subset:
        row[f"{c}_mean"] = float(g[c].mean())
        row[f"{c}_std"] = float(g[c].std(ddof=0))

    summary_rows.append(row)

feature_summary = pd.DataFrame(summary_rows)
safe_to_csv(feature_summary, OUT_FEATURE_SUMMARY)

# Manifest.
manifest_rows = []

for path, kind in [
    (OUT_FEATURES, "full_features"),
    (OUT_FEATURE_MATRIX_RAW, "feature_matrix_raw"),
    (OUT_FEATURE_MATRIX_STANDARDIZED, "feature_matrix_standardized"),
    (OUT_METADATA, "metadata"),
    (OUT_TRAIN_STATS, "train_standardization_stats"),
    (OUT_CLASS_SPLIT_COUNTS, "class_split_counts"),
    (OUT_FEATURE_SUMMARY, "feature_summary"),
]:
    manifest_rows.append({
        "artifact_type": kind,
        "path": str(path),
        "exists": Path(path).exists(),
        "size_bytes": Path(path).stat().st_size if Path(path).exists() else "",
    })

manifest = pd.DataFrame(manifest_rows)
safe_to_csv(manifest, OUT_MANIFEST)

feature_ok_count = int(features["feature_extraction_ok"].sum())
total_rows = int(len(features))
feature_ok_ratio = feature_ok_count / total_rows if total_rows else 0.0
path_exists_count = int(features["path_exists"].sum())

nan_feature_cells = int(features[feature_cols].isna().sum().sum())
duplicate_paths_after_dedup = int(features["resolved_crop_path"].duplicated().sum())

hard_issues = []
warnings = []

if feature_ok_ratio < 0.95:
    hard_issues.append({
        "item": "feature_extraction_success",
        "issue_type": "hard_low_feature_extraction_success",
        "issue_detail": f"feature_ok_ratio={feature_ok_ratio:.4f}",
    })

if nan_feature_cells > 0:
    hard_issues.append({
        "item": "nan_feature_cells",
        "issue_type": "hard_nan_feature_values",
        "issue_detail": f"{nan_feature_cells} NaN cells found in feature matrix.",
    })

if duplicate_paths_after_dedup > 0:
    warnings.append({
        "item": "duplicate_paths_after_dedup",
        "issue_type": "warning_duplicate_paths_remaining",
        "issue_detail": f"{duplicate_paths_after_dedup} duplicated resolved paths remain.",
    })

if not set(["train", "val", "test"]).issubset(set(features["split"].unique())):
    hard_issues.append({
        "item": "split_values",
        "issue_type": "hard_missing_train_val_test_split",
        "issue_detail": "Feature table does not contain all train/val/test split values.",
    })

all_issues = hard_issues + warnings
issues_df = pd.DataFrame(all_issues, columns=["item", "issue_type", "issue_detail"])
safe_to_csv(issues_df, OUT_ISSUES)

ready_for_v37 = (
    len(hard_issues) == 0
    and total_rows > 0
    and feature_ok_ratio >= 0.95
    and nan_feature_cells == 0
)

v37_config = {
    "stage": "v36b_full_crop_lightweight_feature_extraction",
    "input_feature_table": str(OUT_FEATURES),
    "feature_matrix_raw": str(OUT_FEATURE_MATRIX_RAW),
    "feature_matrix_standardized": str(OUT_FEATURE_MATRIX_STANDARDIZED),
    "metadata": str(OUT_METADATA),
    "label_column": "behaviour_code",
    "split_column": "split",
    "feature_columns": feature_cols,
    "eligible_classes": eligible_classes,
    "recommended_baseline_task": "simple classical baseline on standardized lightweight crop features",
    "claim_scope": "baseline sanity check only; not final behaviour classifier",
    "ready_for_v37_baseline_classifier_dryrun": bool(ready_for_v37),
}

OUT_V37_CONFIG.write_text(json.dumps(v37_config, indent=2))

decision = pd.DataFrame([{
    "v36b_decision": "full_crop_lightweight_feature_extraction_completed" if ready_for_v37 else "full_crop_lightweight_feature_extraction_issues_found",
    "input_rows": int(len(df)),
    "feature_rows": int(total_rows),
    "feature_ok_count": int(feature_ok_count),
    "feature_ok_ratio": round(feature_ok_ratio, 4),
    "path_exists_count": int(path_exists_count),
    "feature_column_count": int(len(feature_cols)),
    "class_count": int(features["behaviour_code"].nunique()),
    "split_values": " | ".join(sorted(features["split"].dropna().astype(str).unique().tolist())),
    "nan_feature_cells": int(nan_feature_cells),
    "duplicate_paths_after_dedup": int(duplicate_paths_after_dedup),
    "hard_issue_count": int(len(hard_issues)),
    "warning_count": int(len(warnings)),
    "issue_count": int(len(issues_df)),
    "ready_for_v37_baseline_classifier_dryrun": bool(ready_for_v37),
    "generated_at": datetime.now().isoformat(timespec="seconds"),
}])

safe_to_csv(decision, OUT_DECISION)

count_table = md_table(
    class_split_counts.to_dict("records"),
    ["behaviour_code", "total_rows", "train_rows", "val_rows", "test_rows"],
)

report = f"""# Week 7 v36b Full Crop Lightweight Feature Extraction Report

## Purpose

This step extracts lightweight crop-level features for the deduplicated baseline-ready crop subset.

No model is trained in this step. The purpose is to create a clean feature matrix for later baseline sanity checks.

## Input

- v36a2 recommended input: `{input_path}`
- Rows: `{len(df)}`
- Label column: `{label_col}`
- Split column: `{split_col}`
- Feature input column: `{feature_input_col}`

## Environment

- OpenCV available: `{CV2_AVAILABLE}`
- PIL available: `{PIL_AVAILABLE}`

## Results

- Feature rows: `{total_rows}`
- Feature extraction OK: `{feature_ok_count}` / `{total_rows}`
- Feature OK ratio: `{feature_ok_ratio:.4f}`
- Feature columns: `{len(feature_cols)}`
- NaN feature cells: `{nan_feature_cells}`
- Duplicate paths after dedup: `{duplicate_paths_after_dedup}`
- Ready for v37 baseline classifier dry-run: `{ready_for_v37}`

## Class / split counts

{count_table}

## Outputs

- Full features: `{OUT_FEATURES}`
- Raw feature matrix: `{OUT_FEATURE_MATRIX_RAW}`
- Train-standardized feature matrix: `{OUT_FEATURE_MATRIX_STANDARDIZED}`
- Metadata: `{OUT_METADATA}`
- Train standardization stats: `{OUT_TRAIN_STATS}`
- Feature summary: `{OUT_FEATURE_SUMMARY}`
- Recommended v37 config: `{OUT_V37_CONFIG}`

## Interpretation

The feature matrix is suitable for a simple baseline sanity check if all features were extracted successfully, no NaN feature values exist and train/val/test splits are present.

## Next step

v37 should run a small baseline classifier dry-run using the standardized lightweight feature matrix. The result should be described only as a baseline sanity check, not as a final behaviour model.
"""

OUT_REPORT.write_text(report)

OUT_NOTE.write_text(
    "# Week 7 v36b Full Crop Lightweight Feature Extraction\n\n"
    "## Summary\n\n"
    f"- Input rows: `{len(df)}`\n"
    f"- Feature rows: `{total_rows}`\n"
    f"- Feature extraction OK: `{feature_ok_count}` / `{total_rows}`\n"
    f"- Feature OK ratio: `{feature_ok_ratio:.4f}`\n"
    f"- Feature column count: `{len(feature_cols)}`\n"
    f"- NaN feature cells: `{nan_feature_cells}`\n"
    f"- Duplicate paths after dedup: `{duplicate_paths_after_dedup}`\n"
    f"- Hard issue count: `{len(hard_issues)}`\n"
    f"- Warning count: `{len(warnings)}`\n"
    f"- Ready for v37 baseline classifier dry-run: `{ready_for_v37}`\n\n"
    "## Outputs\n\n"
    f"- Full features: `{OUT_FEATURES}`\n"
    f"- Raw feature matrix: `{OUT_FEATURE_MATRIX_RAW}`\n"
    f"- Standardized matrix: `{OUT_FEATURE_MATRIX_STANDARDIZED}`\n"
    f"- Metadata: `{OUT_METADATA}`\n"
    f"- Train stats: `{OUT_TRAIN_STATS}`\n"
    f"- Class split counts: `{OUT_CLASS_SPLIT_COUNTS}`\n"
    f"- Feature summary: `{OUT_FEATURE_SUMMARY}`\n"
    f"- v37 config: `{OUT_V37_CONFIG}`\n"
    f"- Report: `{OUT_REPORT}`\n"
    f"- Decision: `{OUT_DECISION}`\n"
    f"- Issues: `{OUT_ISSUES}`\n"
)

print("Saved:")
print(OUT_FEATURES)
print(OUT_FEATURE_MATRIX_RAW)
print(OUT_FEATURE_MATRIX_STANDARDIZED)
print(OUT_METADATA)
print(OUT_TRAIN_STATS)
print(OUT_CLASS_SPLIT_COUNTS)
print(OUT_FEATURE_SUMMARY)
print(OUT_MANIFEST)
print(OUT_V37_CONFIG)
print(OUT_REPORT)
print(OUT_DECISION)
print(OUT_ISSUES)
print(OUT_NOTE)

print()
print("=== v36b decision ===")
print(decision.to_string(index=False))

print()
print("=== v36b issues ===")
if len(issues_df):
    print(issues_df.to_string(index=False))
else:
    print("No issues found.")
