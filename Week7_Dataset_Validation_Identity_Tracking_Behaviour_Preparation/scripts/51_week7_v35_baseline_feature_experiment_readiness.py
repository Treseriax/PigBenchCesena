from pathlib import Path
from datetime import datetime
import csv
import json
import pandas as pd


ROOT = Path.home() / "PigBench"
W7 = ROOT / "Week7_Dataset_Validation_Identity_Tracking_Behaviour_Preparation"

V34B = W7 / "outputs" / "week7_baseline_behaviour_representation_dataset_v34b_split_fixed"
V34 = W7 / "outputs" / "week7_baseline_behaviour_representation_dataset_v34"

CROP_IN = V34B / "week7_v34b_crop_level_baseline_behaviour_index_split_fixed.csv"
CLIP_IN = V34B / "week7_v34b_clip_level_multilabel_behaviour_index_split_fixed.csv"
CLASS_SPLIT_IN = V34B / "week7_v34b_class_by_split_readiness.csv"
SPLIT_READINESS_IN = V34B / "week7_v34b_split_readiness_checks.csv"

IDENTITY_IN = V34 / "week7_v34_conservative_identity_subset_index.csv"

OUT_ROOT = W7 / "outputs" / "week7_baseline_feature_experiment_readiness_v35"
CONFIG_DIR = OUT_ROOT / "experiment_configs"
OUT_ROOT.mkdir(parents=True, exist_ok=True)
CONFIG_DIR.mkdir(parents=True, exist_ok=True)

OUT_CROP_PATH_READINESS = OUT_ROOT / "week7_v35_crop_path_readiness.csv"
OUT_CLIP_PATH_READINESS = OUT_ROOT / "week7_v35_clip_path_readiness.csv"
OUT_SPLIT_CLASS_POLICY = OUT_ROOT / "week7_v35_split_class_policy.csv"
OUT_EXPERIMENT_CANDIDATES = OUT_ROOT / "week7_v35_baseline_experiment_candidates.csv"
OUT_FEATURE_READINESS = OUT_ROOT / "week7_v35_feature_extraction_readiness.csv"
OUT_DATASET_CARD = OUT_ROOT / "week7_v35_dataset_card.md"
OUT_REPORT = OUT_ROOT / "week7_v35_baseline_feature_experiment_readiness_report.md"
OUT_DECISION = OUT_ROOT / "week7_v35_baseline_feature_experiment_readiness_decision_summary.csv"
OUT_ISSUES = OUT_ROOT / "week7_v35_baseline_feature_experiment_readiness_issues.csv"
OUT_NOTE = W7 / "notes" / "week7_v35_baseline_feature_experiment_readiness_notes.md"


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


def pick_col(df, candidates):
    for c in candidates:
        if c in df.columns:
            return c
    for c in df.columns:
        cl = c.lower()
        for cand in candidates:
            if cand.lower() in cl:
                return c
    return None


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


def path_exists(value):
    rp = resolve_path(value)
    return bool(rp and Path(rp).exists())


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

crop = read_df(CROP_IN)
clip = read_df(CLIP_IN)
class_split = read_df(CLASS_SPLIT_IN)
split_readiness = read_df(SPLIT_READINESS_IN)
identity_subset = read_df(IDENTITY_IN)

required_inputs = [
    ("crop_split_fixed_index", CROP_IN, crop),
    ("clip_split_fixed_index", CLIP_IN, clip),
    ("class_by_split_readiness", CLASS_SPLIT_IN, class_split),
    ("split_readiness", SPLIT_READINESS_IN, split_readiness),
]

for key, path, df in required_inputs:
    if not path.exists() or len(df) == 0:
        issues.append({
            "item": key,
            "issue_type": "hard_missing_or_empty_input",
            "issue_detail": str(path),
        })

if len(issues):
    issues_df = pd.DataFrame(issues, columns=["item", "issue_type", "issue_detail"])
    safe_to_csv(issues_df, OUT_ISSUES)
    print("Hard input issues found:")
    print(issues_df.to_string(index=False))
    raise SystemExit(1)

for df in [crop, clip, class_split, split_readiness, identity_subset]:
    if "scan_frame_id" in df.columns:
        df["scan_frame_id"] = df["scan_frame_id"].fillna("").astype(str).str.strip()

# ---------------------------------------------------------------------
# Column identification
# ---------------------------------------------------------------------

crop_path_col = pick_col(
    crop,
    ["crop_path", "crop_file", "crop_image_path", "image_crop_path", "path"],
)

clip_path_col = pick_col(
    clip,
    ["clip_path", "video_clip_path", "clip_file", "path"],
)

preview_path_col = pick_col(
    clip,
    ["preview_frame_path", "preview_path", "frame_path"],
)

behaviour_col = pick_col(crop, ["behaviour_code", "behaviour_label", "behaviour", "label"])
split_col = pick_col(crop, ["split", "primary_split", "set", "subset"])
pig_id_col = pick_col(crop, ["behaviour_pig_id", "pig_id"])
sample_id_col = pick_col(crop, ["v34_crop_sample_id", "sample_id", "id"])

if behaviour_col is None:
    issues.append({
        "item": "crop_behaviour_column",
        "issue_type": "hard_missing_column",
        "issue_detail": "Could not find behaviour column in crop index.",
    })

if split_col is None:
    issues.append({
        "item": "crop_split_column",
        "issue_type": "hard_missing_column",
        "issue_detail": "Could not find split column in crop index.",
    })

if crop_path_col is None:
    issues.append({
        "item": "crop_path_column",
        "issue_type": "warning_missing_crop_path_column",
        "issue_detail": "Crop index has no crop_path-like column. Feature extraction may need path repair.",
    })

if clip_path_col is None:
    issues.append({
        "item": "clip_path_column",
        "issue_type": "warning_missing_clip_path_column",
        "issue_detail": "Clip index has no clip_path-like column. Clip feature extraction may need path repair.",
    })

if any(i["issue_type"].startswith("hard_") for i in issues):
    issues_df = pd.DataFrame(issues, columns=["item", "issue_type", "issue_detail"])
    safe_to_csv(issues_df, OUT_ISSUES)
    print("Hard issues found:")
    print(issues_df.to_string(index=False))
    raise SystemExit(1)

# ---------------------------------------------------------------------
# Crop path readiness
# ---------------------------------------------------------------------

crop_ready_rows = []

for _, r in crop.iterrows():
    raw_path = clean(r.get(crop_path_col, "")) if crop_path_col else ""
    resolved = resolve_path(raw_path) if raw_path else ""
    exists = bool(resolved and Path(resolved).exists())

    crop_ready_rows.append({
        "sample_id": clean(r.get(sample_id_col, "")) if sample_id_col else "",
        "scan_frame_id": clean(r.get("scan_frame_id", "")),
        "split": clean(r.get(split_col, "")),
        "behaviour_code": clean(r.get(behaviour_col, "")),
        "behaviour_pig_id": clean(r.get(pig_id_col, "")) if pig_id_col else "",
        "raw_crop_path": raw_path,
        "resolved_crop_path": resolved,
        "crop_path_nonempty": bool(raw_path),
        "crop_path_exists": bool(exists),
    })

crop_path_readiness = pd.DataFrame(crop_ready_rows)
safe_to_csv(crop_path_readiness, OUT_CROP_PATH_READINESS)

# ---------------------------------------------------------------------
# Clip path readiness
# ---------------------------------------------------------------------

clip_ready_rows = []

for _, r in clip.iterrows():
    raw_clip = clean(r.get(clip_path_col, "")) if clip_path_col else ""
    resolved_clip = resolve_path(raw_clip) if raw_clip else ""
    clip_exists = bool(resolved_clip and Path(resolved_clip).exists())

    raw_preview = clean(r.get(preview_path_col, "")) if preview_path_col else ""
    resolved_preview = resolve_path(raw_preview) if raw_preview else ""
    preview_exists = bool(resolved_preview and Path(resolved_preview).exists())

    clip_ready_rows.append({
        "scan_frame_id": clean(r.get("scan_frame_id", "")),
        "split": clean(r.get("split", "")),
        "behaviour_set": clean(r.get("behaviour_set", "")),
        "pig_level_rows": clean(r.get("pig_level_rows", "")),
        "raw_clip_path": raw_clip,
        "resolved_clip_path": resolved_clip,
        "clip_path_nonempty": bool(raw_clip),
        "clip_path_exists": bool(clip_exists),
        "raw_preview_frame_path": raw_preview,
        "resolved_preview_frame_path": resolved_preview,
        "preview_path_nonempty": bool(raw_preview),
        "preview_path_exists": bool(preview_exists),
    })

clip_path_readiness = pd.DataFrame(clip_ready_rows)
safe_to_csv(clip_path_readiness, OUT_CLIP_PATH_READINESS)

# ---------------------------------------------------------------------
# Split/class policy
# ---------------------------------------------------------------------

policy_rows = []

for _, r in class_split.iterrows():
    behaviour = clean(r.get("behaviour_code", ""))
    total = int(float(r.get("total_crop_rows", 0) or 0))
    train = int(float(r.get("train_rows", 0) or 0))
    val = int(float(r.get("val_rows", 0) or 0))
    test = int(float(r.get("test_rows", 0) or 0))
    readiness = clean(r.get("readiness", ""))

    if readiness == "baseline_ready_all_splits":
        experiment_policy = "eligible_for_crop_baseline_classifier"
    elif readiness == "limited_use_with_caution":
        experiment_policy = "include_for_reporting_or_secondary_analysis_only"
    else:
        experiment_policy = "exclude_from_main_baseline_or_group_as_other_after_review"

    has_val = val > 0
    has_test = test > 0

    policy_rows.append({
        "behaviour_code": behaviour,
        "total_crop_rows": total,
        "train_rows": train,
        "val_rows": val,
        "test_rows": test,
        "has_validation_examples": bool(has_val),
        "has_test_examples": bool(has_test),
        "readiness": readiness,
        "experiment_policy": experiment_policy,
    })

split_class_policy = pd.DataFrame(policy_rows)
safe_to_csv(split_class_policy, OUT_SPLIT_CLASS_POLICY)

baseline_classes = sorted(
    split_class_policy[
        split_class_policy["experiment_policy"] == "eligible_for_crop_baseline_classifier"
    ]["behaviour_code"].tolist()
)

limited_classes = sorted(
    split_class_policy[
        split_class_policy["experiment_policy"] == "include_for_reporting_or_secondary_analysis_only"
    ]["behaviour_code"].tolist()
)

rare_classes = sorted(
    split_class_policy[
        split_class_policy["experiment_policy"] == "exclude_from_main_baseline_or_group_as_other_after_review"
    ]["behaviour_code"].tolist()
)

# ---------------------------------------------------------------------
# Feature extraction readiness summary
# ---------------------------------------------------------------------

crop_total = int(len(crop_path_readiness))
crop_path_nonempty = int(crop_path_readiness["crop_path_nonempty"].sum()) if len(crop_path_readiness) else 0
crop_path_exists_count = int(crop_path_readiness["crop_path_exists"].sum()) if len(crop_path_readiness) else 0
crop_path_exists_ratio = crop_path_exists_count / crop_total if crop_total else 0.0

clip_total = int(len(clip_path_readiness))
clip_path_nonempty = int(clip_path_readiness["clip_path_nonempty"].sum()) if len(clip_path_readiness) else 0
clip_path_exists_count = int(clip_path_readiness["clip_path_exists"].sum()) if len(clip_path_readiness) else 0
clip_path_exists_ratio = clip_path_exists_count / clip_total if clip_total else 0.0

preview_nonempty = int(clip_path_readiness["preview_path_nonempty"].sum()) if len(clip_path_readiness) else 0
preview_exists_count = int(clip_path_readiness["preview_path_exists"].sum()) if len(clip_path_readiness) else 0
preview_exists_ratio = preview_exists_count / clip_total if clip_total else 0.0

splits = sorted(crop[split_col].dropna().astype(str).unique().tolist())
has_train_val_test = set(["train", "val", "test"]).issubset(set(splits))

feature_rows = [
    {
        "readiness_area": "crop_level_feature_extraction",
        "total_items": crop_total,
        "path_column": crop_path_col or "",
        "nonempty_paths": crop_path_nonempty,
        "existing_paths": crop_path_exists_count,
        "existing_path_ratio": round(crop_path_exists_ratio, 4),
        "ready": bool(crop_total > 0 and crop_path_exists_ratio >= 0.95),
        "interpretation": "Ready if crop image paths are available and mostly exist.",
    },
    {
        "readiness_area": "clip_level_feature_extraction",
        "total_items": clip_total,
        "path_column": clip_path_col or "",
        "nonempty_paths": clip_path_nonempty,
        "existing_paths": clip_path_exists_count,
        "existing_path_ratio": round(clip_path_exists_ratio, 4),
        "ready": bool(clip_total > 0 and clip_path_exists_ratio >= 0.95),
        "interpretation": "Ready if 10-second clip paths are available and mostly exist.",
    },
    {
        "readiness_area": "preview_frame_visual_qa",
        "total_items": clip_total,
        "path_column": preview_path_col or "",
        "nonempty_paths": preview_nonempty,
        "existing_paths": preview_exists_count,
        "existing_path_ratio": round(preview_exists_ratio, 4),
        "ready": bool(clip_total > 0 and preview_exists_ratio >= 0.95),
        "interpretation": "Useful for QA; not mandatory for feature extraction if clip paths exist.",
    },
    {
        "readiness_area": "split_readiness",
        "total_items": crop_total,
        "path_column": split_col or "",
        "nonempty_paths": int(crop[split_col].fillna("").astype(str).str.len().gt(0).sum()) if split_col else 0,
        "existing_paths": "",
        "existing_path_ratio": "",
        "ready": bool(has_train_val_test),
        "interpretation": "Ready if train/val/test split values exist.",
    },
]

feature_readiness = pd.DataFrame(feature_rows)
safe_to_csv(feature_readiness, OUT_FEATURE_READINESS)

# ---------------------------------------------------------------------
# Experiment configs
# ---------------------------------------------------------------------

configs = []

crop_baseline_config = {
    "experiment_id": "v35_crop_single_frame_baseline_ready_classes",
    "experiment_type": "crop_level_single_frame_classification_baseline",
    "dataset_index": str(CROP_IN),
    "split_fixed_index": str(CROP_IN),
    "feature_input_column": crop_path_col,
    "label_column": behaviour_col,
    "split_column": split_col,
    "eligible_classes": baseline_classes,
    "limited_classes": limited_classes,
    "excluded_or_report_only_classes": rare_classes,
    "claim_scope": "debug/baseline only; not temporal behaviour understanding",
    "recommended_next_action": "extract crop visual embeddings or train a simple baseline classifier only on eligible classes",
}

clip_multilabel_config = {
    "experiment_id": "v35_clip_level_multilabel_context",
    "experiment_type": "clip_level_multilabel_representation",
    "dataset_index": str(CLIP_IN),
    "feature_input_column": clip_path_col,
    "label_columns_prefix": "label__",
    "split_column": "split",
    "claim_scope": "temporal context representation; not clean single-label classification",
    "recommended_next_action": "extract clip/frame embeddings and evaluate multi-label or multi-instance formulation",
}

identity_subset_config = {
    "experiment_id": "v35_conservative_identity_subset",
    "experiment_type": "identity_aware_tracklet_subset",
    "dataset_index": str(IDENTITY_IN),
    "accepted_identity_policy": "use conservative accepted identities only",
    "review_queue_policy": "do not use review queue as final labels",
    "claim_scope": "identity-aware candidate subset, not final production MOT",
    "recommended_next_action": "use for identity-aware temporal feature experiments after visual review",
}

configs.append(crop_baseline_config)
configs.append(clip_multilabel_config)
configs.append(identity_subset_config)

for cfg in configs:
    cfg_path = CONFIG_DIR / f"{cfg['experiment_id']}.json"
    cfg_path.write_text(json.dumps(cfg, indent=2))

experiment_candidates = pd.DataFrame([
    {
        "experiment_id": crop_baseline_config["experiment_id"],
        "experiment_type": crop_baseline_config["experiment_type"],
        "primary_input": crop_baseline_config["dataset_index"],
        "eligible_classes": " | ".join(baseline_classes),
        "limited_classes": " | ".join(limited_classes),
        "excluded_or_report_only_classes": " | ".join(rare_classes),
        "readiness": "ready" if crop_total > 0 and has_train_val_test else "not_ready",
        "next_action": crop_baseline_config["recommended_next_action"],
    },
    {
        "experiment_id": clip_multilabel_config["experiment_id"],
        "experiment_type": clip_multilabel_config["experiment_type"],
        "primary_input": clip_multilabel_config["dataset_index"],
        "eligible_classes": "multi_label_all_classes_with_policy",
        "limited_classes": "see class policy",
        "excluded_or_report_only_classes": "see class policy",
        "readiness": "ready" if clip_total > 0 else "not_ready",
        "next_action": clip_multilabel_config["recommended_next_action"],
    },
    {
        "experiment_id": identity_subset_config["experiment_id"],
        "experiment_type": identity_subset_config["experiment_type"],
        "primary_input": identity_subset_config["dataset_index"],
        "eligible_classes": "conservative_accepted_identity_tracklets",
        "limited_classes": "review_queue_candidates",
        "excluded_or_report_only_classes": "unknown_or_low_confidence_identity",
        "readiness": "ready" if len(identity_subset) > 0 else "not_ready",
        "next_action": identity_subset_config["recommended_next_action"],
    },
])

safe_to_csv(experiment_candidates, OUT_EXPERIMENT_CANDIDATES)

# ---------------------------------------------------------------------
# Issues, report, decision
# ---------------------------------------------------------------------

hard_issue_count = len([x for x in issues if str(x["issue_type"]).startswith("hard_")])
warning_count = len(issues) - hard_issue_count

# Do not require path_exists_ratio >= .95 to be ready for planning,
# but require it for immediate feature extraction.
ready_for_immediate_crop_feature_extraction = bool(crop_total > 0 and crop_path_exists_ratio >= 0.95)
ready_for_immediate_clip_feature_extraction = bool(clip_total > 0 and clip_path_exists_ratio >= 0.95)
ready_for_v36 = bool(hard_issue_count == 0 and crop_total > 0 and clip_total > 0 and has_train_val_test)

issues_df = pd.DataFrame(issues, columns=["item", "issue_type", "issue_detail"])
safe_to_csv(issues_df, OUT_ISSUES)

decision = pd.DataFrame([{
    "v35_decision": "baseline_feature_experiment_readiness_created",
    "crop_rows": int(crop_total),
    "clip_rows": int(clip_total),
    "identity_subset_rows": int(len(identity_subset)),
    "split_values": " | ".join(splits),
    "has_train_val_test": bool(has_train_val_test),
    "baseline_classes": " | ".join(baseline_classes),
    "limited_classes": " | ".join(limited_classes),
    "rare_or_report_only_classes": " | ".join(rare_classes),
    "crop_path_column": crop_path_col or "",
    "crop_path_exists_count": int(crop_path_exists_count),
    "crop_path_exists_ratio": round(crop_path_exists_ratio, 4),
    "clip_path_column": clip_path_col or "",
    "clip_path_exists_count": int(clip_path_exists_count),
    "clip_path_exists_ratio": round(clip_path_exists_ratio, 4),
    "preview_path_exists_count": int(preview_exists_count),
    "preview_path_exists_ratio": round(preview_exists_ratio, 4),
    "experiment_configs_created": int(len(configs)),
    "hard_issue_count": int(hard_issue_count),
    "warning_count": int(warning_count),
    "issue_count": int(len(issues_df)),
    "ready_for_immediate_crop_feature_extraction": bool(ready_for_immediate_crop_feature_extraction),
    "ready_for_immediate_clip_feature_extraction": bool(ready_for_immediate_clip_feature_extraction),
    "ready_for_v36_feature_extraction_or_baseline_config": bool(ready_for_v36),
    "generated_at": datetime.now().isoformat(timespec="seconds"),
}])

safe_to_csv(decision, OUT_DECISION)

policy_table = md_table(
    split_class_policy.to_dict("records"),
    [
        "behaviour_code",
        "total_crop_rows",
        "train_rows",
        "val_rows",
        "test_rows",
        "readiness",
        "experiment_policy",
    ],
)

experiment_table = md_table(
    experiment_candidates.to_dict("records"),
    [
        "experiment_id",
        "experiment_type",
        "readiness",
        "eligible_classes",
        "next_action",
    ],
)

readiness_table = md_table(
    feature_readiness.to_dict("records"),
    [
        "readiness_area",
        "total_items",
        "path_column",
        "nonempty_paths",
        "existing_paths",
        "existing_path_ratio",
        "ready",
    ],
)

dataset_card = f"""# Week 7 v35 Dataset Card

## Dataset purpose

This dataset card describes the baseline behaviour representation indices prepared after the Week 7 validation and identity-linking pipeline.

## Scope

The v35 outputs are intended for feature extraction and baseline experiment planning. They are not a final behaviour classifier and not final production multi-object tracking.

## Main indices

- Crop-level baseline index: `{CROP_IN}`
- Clip-level multi-label index: `{CLIP_IN}`
- Conservative identity subset index: `{IDENTITY_IN}`

## Class policy

- Baseline classes: `{', '.join(baseline_classes)}`
- Limited-use classes: `{', '.join(limited_classes)}`
- Rare/report-only classes: `{', '.join(rare_classes)}`

## Splits

Split values: `{', '.join(splits)}`

## Identity policy

Only conservative accepted identities should be used for identity-aware experiments. Review queue, low-confidence and unknown identities should not be treated as final labels.

## Recommended usage

1. Use crop-level baseline for simple visual baseline/debugging.
2. Use clip-level multi-label representation for temporal context.
3. Use conservative identity subset only after visual review.
4. Report rare classes separately.
5. Use per-class metrics.
"""

OUT_DATASET_CARD.write_text(dataset_card)

report = f"""# Week 7 v35 Baseline Feature / Experiment Readiness Report

## Purpose

This step checks whether the v34b split-fixed behaviour representation datasets are ready for baseline feature extraction or baseline experiment configuration.

It creates path readiness tables, class policies and baseline experiment configuration files.

## Inputs

- Crop-level split-fixed index: `{CROP_IN}`
- Clip-level split-fixed index: `{CLIP_IN}`
- Class-by-split readiness: `{CLASS_SPLIT_IN}`
- Conservative identity subset: `{IDENTITY_IN}`

## Dataset summary

- Crop rows: `{crop_total}`
- Clip rows: `{clip_total}`
- Conservative identity subset rows: `{len(identity_subset)}`
- Split values: `{', '.join(splits)}`
- Has train/val/test: `{has_train_val_test}`

## Path readiness

{readiness_table}

## Class policy

{policy_table}

## Experiment candidates

{experiment_table}

## Recommended next action

If crop and clip paths exist, v36 can perform actual feature extraction. If paths are incomplete, v36 should first repair path references or use the existing v22/v26 source paths directly.

## Important modelling policy

The crop baseline is allowed only as a simple single-frame baseline. The stronger direction is clip-level multi-label or multi-instance representation because many clips contain several pigs and several behaviours.

## Output claim

v35 creates baseline feature/experiment readiness. It does not train a model and does not validate a final behaviour classifier.
"""

OUT_REPORT.write_text(report)

OUT_NOTE.write_text(
    "# Week 7 v35 Baseline Feature / Experiment Readiness\n\n"
    "## Summary\n\n"
    f"- Crop rows: `{crop_total}`\n"
    f"- Clip rows: `{clip_total}`\n"
    f"- Conservative identity subset rows: `{len(identity_subset)}`\n"
    f"- Split values: `{', '.join(splits)}`\n"
    f"- Baseline classes: `{', '.join(baseline_classes)}`\n"
    f"- Limited classes: `{', '.join(limited_classes)}`\n"
    f"- Rare/report-only classes: `{', '.join(rare_classes)}`\n"
    f"- Crop path exists ratio: `{crop_path_exists_ratio:.4f}`\n"
    f"- Clip path exists ratio: `{clip_path_exists_ratio:.4f}`\n"
    f"- Experiment configs created: `{len(configs)}`\n"
    f"- Hard issue count: `{hard_issue_count}`\n"
    f"- Warning count: `{warning_count}`\n"
    f"- Ready for v36 feature extraction/config: `{ready_for_v36}`\n\n"
    "## Outputs\n\n"
    f"- Crop path readiness: `{OUT_CROP_PATH_READINESS}`\n"
    f"- Clip path readiness: `{OUT_CLIP_PATH_READINESS}`\n"
    f"- Class policy: `{OUT_SPLIT_CLASS_POLICY}`\n"
    f"- Experiment candidates: `{OUT_EXPERIMENT_CANDIDATES}`\n"
    f"- Feature readiness: `{OUT_FEATURE_READINESS}`\n"
    f"- Dataset card: `{OUT_DATASET_CARD}`\n"
    f"- Report: `{OUT_REPORT}`\n"
    f"- Decision: `{OUT_DECISION}`\n"
    f"- Issues: `{OUT_ISSUES}`\n"
)

print("Saved:")
print(OUT_CROP_PATH_READINESS)
print(OUT_CLIP_PATH_READINESS)
print(OUT_SPLIT_CLASS_POLICY)
print(OUT_EXPERIMENT_CANDIDATES)
print(OUT_FEATURE_READINESS)
print(OUT_DATASET_CARD)
print(OUT_REPORT)
print(OUT_DECISION)
print(OUT_ISSUES)
print(OUT_NOTE)
print(CONFIG_DIR)

print()
print("=== v35 decision ===")
print(decision.to_string(index=False))

print()
print("=== v35 issues ===")
if len(issues_df):
    print(issues_df.to_string(index=False))
else:
    print("No issues found.")
