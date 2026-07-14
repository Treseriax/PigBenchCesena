from pathlib import Path
from datetime import datetime
import csv
import pandas as pd
import numpy as np


ROOT = Path.home() / "PigBench"
W7 = ROOT / "Week7_Dataset_Validation_Identity_Tracking_Behaviour_Preparation"

OUT_ROOT = W7 / "outputs" / "week7_baseline_behaviour_representation_dataset_v34"
OUT_ROOT.mkdir(parents=True, exist_ok=True)

OUT_CROP_INDEX = OUT_ROOT / "week7_v34_crop_level_baseline_behaviour_index.csv"
OUT_CLIP_MULTI = OUT_ROOT / "week7_v34_clip_level_multilabel_behaviour_index.csv"
OUT_IDENTITY_SUBSET = OUT_ROOT / "week7_v34_conservative_identity_subset_index.csv"
OUT_CLASS_READINESS = OUT_ROOT / "week7_v34_behaviour_class_readiness.csv"
OUT_SPLIT_READINESS = OUT_ROOT / "week7_v34_split_readiness_checks.csv"
OUT_POLICY = OUT_ROOT / "week7_v34_modelling_policy.csv"
OUT_REPORT = OUT_ROOT / "week7_v34_baseline_behaviour_representation_report.md"
OUT_DECISION = OUT_ROOT / "week7_v34_baseline_behaviour_representation_decision_summary.csv"
OUT_ISSUES = OUT_ROOT / "week7_v34_baseline_behaviour_representation_issues.csv"
OUT_NOTE = W7 / "notes" / "week7_baseline_behaviour_representation_dataset_v34_notes.md"


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


def read_first(path):
    df = read_df(path)
    if len(df):
        return df.iloc[0].to_dict()
    return {}


def find_csv(root, required_terms, preferred_terms=None):
    root = Path(root)
    preferred_terms = preferred_terms or []
    if not root.exists():
        return None

    candidates = []
    for p in root.rglob("*.csv"):
        s = str(p).lower()
        if all(t.lower() in s for t in required_terms):
            candidates.append(p)

    if not candidates:
        return None

    def score(p):
        s = str(p).lower()
        val = 0
        for term in preferred_terms:
            if term.lower() in s:
                val += 10
        if "issues" in s:
            val -= 50
        if "manifest" in s:
            val -= 30
        if "summary" in s:
            val += 5
        if "dataset" in s:
            val += 5
        if "index" in s:
            val += 5
        return val

    return sorted(candidates, key=lambda p: (-score(p), str(p)))[0]


def pick_col(df, candidates):
    for c in candidates:
        if c in df.columns:
            return c
    return None


def bool_series(s):
    return s.fillna("").astype(str).str.lower().isin(["true", "1", "yes", "y", "accepted"])


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

# ---------------------------------------------------------------------
# Input discovery
# ---------------------------------------------------------------------

v18c_path = W7 / "outputs" / "behaviour_label_fusion_v18c_verified_crosswalk" / "week7_behaviour_label_fusion_v18c_box_level_dataset.csv"
if not v18c_path.exists():
    v18c_path = find_csv(W7 / "outputs", ["v18c"], ["box_level", "training", "dataset"])

v22_path = find_csv(W7 / "outputs", ["v22"], ["crop", "index", "dataset"])
v33_clip_path = W7 / "outputs" / "week7_behaviour_clip_representation_preparation_v33" / "week7_behaviour_clip_representation_v33_clip_level_index.csv"
v33_label_space_path = W7 / "outputs" / "week7_behaviour_clip_representation_preparation_v33" / "week7_behaviour_clip_representation_v33_behaviour_label_space.csv"
v29d_arbitration_path = W7 / "outputs" / "conservative_identity_arbitration_v29d" / "week7_conservative_identity_arbitration_v29d_tracklet_arbitration.csv"
v29d_review_path = W7 / "outputs" / "conservative_identity_arbitration_v29d" / "week7_conservative_identity_arbitration_v29d_review_queue.csv"
v32_decision_path = W7 / "outputs" / "week7_report_ready_methodology_results_v32" / "week7_report_ready_methodology_results_v32_decision_summary.csv"
v33_decision_path = W7 / "outputs" / "week7_behaviour_clip_representation_preparation_v33" / "week7_behaviour_clip_representation_v33_decision_summary.csv"

required = {
    "v18c_behaviour_fusion_dataset": v18c_path,
    "v33_clip_level_index": v33_clip_path,
    "v33_label_space": v33_label_space_path,
    "v29d_tracklet_arbitration": v29d_arbitration_path,
}

for key, path in required.items():
    if path is None or not Path(path).exists():
        issues.append({
            "item": key,
            "issue_type": "missing_required_file",
            "issue_detail": str(path),
        })

v18c = read_df(v18c_path) if v18c_path else pd.DataFrame()
v22 = read_df(v22_path) if v22_path else pd.DataFrame()
v33_clip = read_df(v33_clip_path)
v33_label_space = read_df(v33_label_space_path)
v29d_arb = read_df(v29d_arbitration_path)
v29d_review = read_df(v29d_review_path)
v32_decision = read_first(v32_decision_path)
v33_decision = read_first(v33_decision_path)

for key, df in [
    ("v18c_behaviour_fusion_dataset", v18c),
    ("v33_clip_level_index", v33_clip),
    ("v33_label_space", v33_label_space),
    ("v29d_tracklet_arbitration", v29d_arb),
]:
    if len(df) == 0:
        issues.append({
            "item": key,
            "issue_type": "empty_or_unreadable",
            "issue_detail": "No rows could be read.",
        })

if len(issues):
    issues_df = pd.DataFrame(issues, columns=["item", "issue_type", "issue_detail"])
    safe_to_csv(issues_df, OUT_ISSUES)
    print("Issues found before processing:")
    print(issues_df.to_string(index=False))
    raise SystemExit(1)

# ---------------------------------------------------------------------
# Column detection
# ---------------------------------------------------------------------

for df in [v18c, v22, v33_clip, v29d_arb, v29d_review]:
    if "scan_frame_id" in df.columns:
        df["scan_frame_id"] = df["scan_frame_id"].fillna("").astype(str).str.strip()

behaviour_col = pick_col(
    v18c,
    ["behaviour_code", "behaviour_code_label", "behaviour_label", "behaviour", "label", "action"],
)

pig_id_col = pick_col(
    v18c,
    ["behaviour_pig_id", "pig_id", "assigned_behaviour_pig_id", "visual_behaviour_pig_id"],
)

colour_col = pick_col(
    v18c,
    ["visual_marker_colour", "final_colour_identity_v17", "assigned_visual_colour", "final_colour", "colour"],
)

crop_path_col = pick_col(
    v18c,
    ["crop_path", "crop_file", "crop_image_path", "image_crop_path"],
)

split_col = pick_col(v18c, ["split", "primary_split", "subset", "set"])

if behaviour_col is None:
    issues.append({
        "item": "behaviour_column",
        "issue_type": "missing_column",
        "issue_detail": "Could not identify behaviour label column in v18c dataset.",
    })

if pig_id_col is None:
    issues.append({
        "item": "pig_id_column",
        "issue_type": "missing_column",
        "issue_detail": "Could not identify pig ID column in v18c dataset.",
    })

if "scan_frame_id" not in v18c.columns:
    issues.append({
        "item": "scan_frame_id",
        "issue_type": "missing_column",
        "issue_detail": "scan_frame_id missing from v18c dataset.",
    })

if len(issues):
    issues_df = pd.DataFrame(issues, columns=["item", "issue_type", "issue_detail"])
    safe_to_csv(issues_df, OUT_ISSUES)
    print("Issues found:")
    print(issues_df.to_string(index=False))
    raise SystemExit(1)

# ---------------------------------------------------------------------
# Crop-level baseline index
# ---------------------------------------------------------------------

bad_tokens = {"unknown", "not_visible", "uncertain", "unassigned", "nan", "none", ""}

crop_df = v18c.copy()
crop_df[behaviour_col] = crop_df[behaviour_col].fillna("").astype(str).str.strip()
crop_df[pig_id_col] = crop_df[pig_id_col].fillna("").astype(str).str.strip()

crop_df = crop_df[
    (crop_df["scan_frame_id"].astype(str).str.len() > 0)
    & (~crop_df[behaviour_col].str.lower().isin(bad_tokens))
    & (~crop_df[pig_id_col].str.lower().isin(bad_tokens))
].copy()

# Add stable row id.
crop_df = crop_df.reset_index(drop=True)
crop_df["v34_crop_sample_id"] = ["v34_crop_%04d" % i for i in range(len(crop_df))]

# If v18c does not have crop path, try to merge from v22 by common columns.
if crop_path_col is None and len(v22):
    v22_crop_col = pick_col(v22, ["crop_path", "crop_file", "crop_image_path", "image_crop_path"])
    if v22_crop_col:
        common_cols = [c for c in ["scan_frame_id", pig_id_col, behaviour_col] if c in crop_df.columns and c in v22.columns]
        if common_cols:
            tmp = v22[common_cols + [v22_crop_col]].copy()
            tmp = tmp.drop_duplicates(common_cols)
            crop_df = crop_df.merge(tmp, on=common_cols, how="left")
            crop_path_col = v22_crop_col

# Output standardized crop columns.
crop_out_cols = []
for c in [
    "v34_crop_sample_id",
    "scan_frame_id",
    behaviour_col,
    pig_id_col,
    colour_col,
    crop_path_col,
    split_col,
]:
    if c and c in crop_df.columns and c not in crop_out_cols:
        crop_out_cols.append(c)

# Add all useful geometry columns if present.
for c in ["x1", "y1", "x2", "y2", "bbox_x1", "bbox_y1", "bbox_x2", "bbox_y2", "box_x1", "box_y1", "box_x2", "box_y2"]:
    if c in crop_df.columns and c not in crop_out_cols:
        crop_out_cols.append(c)

crop_index = crop_df[crop_out_cols].copy()
crop_index = crop_index.rename(columns={
    behaviour_col: "behaviour_code",
    pig_id_col: "behaviour_pig_id",
    colour_col if colour_col else "": "visual_marker_colour",
    crop_path_col if crop_path_col else "": "crop_path",
    split_col if split_col else "": "split",
})

# Add baseline policy columns.
crop_index["v34_representation_type"] = "crop_level_single_frame_baseline"
crop_index["recommended_use"] = "debug_baseline_not_final_temporal_model"

safe_to_csv(crop_index, OUT_CROP_INDEX)

# ---------------------------------------------------------------------
# Clip-level multi-label index
# ---------------------------------------------------------------------

behaviours = sorted(crop_index["behaviour_code"].dropna().astype(str).unique().tolist())

clip_rows = []

for _, r in v33_clip.iterrows():
    scan_id = clean(r.get("scan_frame_id", ""))
    if not scan_id:
        continue

    g = crop_index[crop_index["scan_frame_id"] == scan_id].copy()
    labels = sorted(g["behaviour_code"].dropna().astype(str).unique().tolist())

    row = {
        "scan_frame_id": scan_id,
        "clip_path": clean(r.get("clip_path", "")),
        "preview_frame_path": clean(r.get("preview_frame_path", "")),
        "source_video": clean(r.get("source_video", "")),
        "start_sec": clean(r.get("start_sec", "")),
        "end_sec": clean(r.get("end_sec", "")),
        "split": clean(r.get("split", "")),
        "pig_level_rows": int(len(g)),
        "pig_count": int(g["behaviour_pig_id"].nunique()) if "behaviour_pig_id" in g.columns else 0,
        "behaviour_count": int(len(labels)),
        "behaviour_set": " | ".join(labels),
        "v34_representation_type": "clip_level_multilabel",
    }

    for b in behaviours:
        row[f"label__{b}"] = int(b in labels)

    if len(labels) == 0:
        row["recommended_use"] = "exclude_until_review"
    elif len(labels) == 1:
        row["recommended_use"] = "single_label_clip_baseline_possible"
    else:
        row["recommended_use"] = "multi_label_or_multi_instance_recommended"

    clip_rows.append(row)

clip_multi = pd.DataFrame(clip_rows)
safe_to_csv(clip_multi, OUT_CLIP_MULTI)

# ---------------------------------------------------------------------
# Conservative identity subset index
# ---------------------------------------------------------------------

arb = v29d_arb.copy()

accepted_col = pick_col(
    arb,
    ["conservative_identity_accepted", "accepted", "is_accepted", "final_identity_accepted"],
)

status_col = pick_col(
    arb,
    ["arbitration_status", "identity_status", "conservative_status", "status"],
)

if accepted_col:
    accepted_mask = bool_series(arb[accepted_col])
elif status_col:
    accepted_mask = arb[status_col].fillna("").astype(str).str.lower().str.contains("accepted")
else:
    accepted_mask = pd.Series([False] * len(arb), index=arb.index)
    issues.append({
        "item": "v29d_acceptance_column",
        "issue_type": "missing_column_warning",
        "issue_detail": "Could not identify accepted/status column; identity subset may be empty.",
    })

identity_subset = arb[accepted_mask].copy().reset_index(drop=True)
identity_subset["v34_identity_subset_sample_id"] = ["v34_identity_%04d" % i for i in range(len(identity_subset))]
identity_subset["v34_representation_type"] = "conservative_identity_tracklet_subset"
identity_subset["recommended_use"] = "identity_aware_behaviour_experiments_after_review"

safe_to_csv(identity_subset, OUT_IDENTITY_SUBSET)

# ---------------------------------------------------------------------
# Class readiness
# ---------------------------------------------------------------------

label_counts = crop_index["behaviour_code"].value_counts().to_dict()
clip_counts = {}
for b in behaviours:
    if f"label__{b}" in clip_multi.columns:
        clip_counts[b] = int(clip_multi[f"label__{b}"].sum())
    else:
        clip_counts[b] = 0

class_rows = []
for b in behaviours:
    row_count = int(label_counts.get(b, 0))
    clip_count = int(clip_counts.get(b, 0))

    if row_count < 10:
        readiness = "report_only_or_group_until_more_data"
    elif row_count < 25:
        readiness = "limited_use_with_caution"
    else:
        readiness = "baseline_ready"

    class_rows.append({
        "behaviour_code": b,
        "crop_level_rows": row_count,
        "clip_level_positive_clips": clip_count,
        "baseline_readiness": readiness,
        "recommended_metric_policy": "per_class_metrics_required",
    })

class_readiness = pd.DataFrame(class_rows).sort_values(
    ["crop_level_rows", "behaviour_code"],
    ascending=[False, True],
)

safe_to_csv(class_readiness, OUT_CLASS_READINESS)

# ---------------------------------------------------------------------
# Split readiness
# ---------------------------------------------------------------------

split_rows = []

if "split" in crop_index.columns and crop_index["split"].astype(str).str.len().sum() > 0:
    for split_name, g in crop_index.groupby("split"):
        split_name = clean(split_name)
        row = {
            "split": split_name,
            "crop_level_rows": int(len(g)),
            "unique_scan_frames": int(g["scan_frame_id"].nunique()),
            "unique_behaviours": int(g["behaviour_code"].nunique()),
            "behaviour_set": " | ".join(sorted(g["behaviour_code"].dropna().astype(str).unique().tolist())),
        }

        for b in behaviours:
            row[f"count__{b}"] = int((g["behaviour_code"] == b).sum())

        split_rows.append(row)
else:
    split_rows.append({
        "split": "unknown_or_not_available",
        "crop_level_rows": int(len(crop_index)),
        "unique_scan_frames": int(crop_index["scan_frame_id"].nunique()),
        "unique_behaviours": int(crop_index["behaviour_code"].nunique()),
        "behaviour_set": " | ".join(sorted(crop_index["behaviour_code"].dropna().astype(str).unique().tolist())),
    })
    issues.append({
        "item": "split_column",
        "issue_type": "split_not_available_warning",
        "issue_detail": "No split column found in crop-level index. Split readiness reported as unknown.",
    })

split_readiness = pd.DataFrame(split_rows)
safe_to_csv(split_readiness, OUT_SPLIT_READINESS)

# ---------------------------------------------------------------------
# Modelling policy
# ---------------------------------------------------------------------

policy_rows = [
    {
        "policy_area": "crop_level_baseline",
        "policy": "Use crop-level index only as a single-frame baseline. Do not claim temporal behaviour understanding from crop-only evidence.",
    },
    {
        "policy_area": "clip_level_multilabel",
        "policy": "Use clip-level multi-label index for temporal context. Do not force multi-pig clips into one clean single-label class.",
    },
    {
        "policy_area": "identity_subset",
        "policy": "Use conservative identity subset only. Review queue and low-confidence identities must not be treated as final labels.",
    },
    {
        "policy_area": "rare_classes",
        "policy": "Classes with fewer than 10 rows should be report-only or grouped until more data is available.",
    },
    {
        "policy_area": "evaluation",
        "policy": "Use per-class metrics, class distribution tables and confusion analysis. Overall accuracy alone is not acceptable.",
    },
    {
        "policy_area": "claim_scope",
        "policy": "v34 creates baseline representation datasets. It does not train or validate a final behaviour classifier.",
    },
]

policy = pd.DataFrame(policy_rows)
safe_to_csv(policy, OUT_POLICY)

# ---------------------------------------------------------------------
# Report and decision
# ---------------------------------------------------------------------

issues_df = pd.DataFrame(issues, columns=["item", "issue_type", "issue_detail"])
safe_to_csv(issues_df, OUT_ISSUES)

crop_rows = int(len(crop_index))
clip_rows_n = int(len(clip_multi))
identity_rows = int(len(identity_subset))
behaviour_class_count = int(len(behaviours))
rare_class_count = int((class_readiness["baseline_readiness"] == "report_only_or_group_until_more_data").sum())
baseline_ready_count = int((class_readiness["baseline_readiness"] == "baseline_ready").sum())

ready_for_v35 = (
    crop_rows > 0
    and clip_rows_n > 0
    and behaviour_class_count > 0
)

decision = pd.DataFrame([{
    "v34_decision": "baseline_behaviour_representation_dataset_created",
    "crop_level_rows": crop_rows,
    "clip_level_rows": clip_rows_n,
    "conservative_identity_subset_rows": identity_rows,
    "behaviour_classes": behaviour_class_count,
    "baseline_ready_classes": baseline_ready_count,
    "rare_or_report_only_classes": rare_class_count,
    "issue_count": int(len(issues_df)),
    "ready_for_v35_baseline_experiment_or_feature_extraction": bool(ready_for_v35),
    "generated_at": datetime.now().isoformat(timespec="seconds"),
}])

safe_to_csv(decision, OUT_DECISION)

class_table = md_table(
    class_readiness.to_dict("records"),
    ["behaviour_code", "crop_level_rows", "clip_level_positive_clips", "baseline_readiness", "recommended_metric_policy"],
)

split_table = md_table(
    split_readiness.to_dict("records"),
    ["split", "crop_level_rows", "unique_scan_frames", "unique_behaviours", "behaviour_set"],
)

report = f"""# Week 7 Baseline Behaviour Representation Dataset v34

## Purpose

This step creates the first downstream behaviour-representation dataset after the Week 7 validation and identity-linking pipeline.

The output is not a trained behaviour model. Instead, it creates structured dataset indices for baseline modelling and feature extraction.

## Created outputs

1. Crop-level single-frame baseline index.
2. Clip-level multi-label behaviour index.
3. Conservative identity subset index.
4. Behaviour class readiness table.
5. Split readiness checks.
6. Modelling policy table.

## Input sources

- Behaviour fusion dataset: `{v18c_path}`
- Optional crop dataset/index: `{v22_path}`
- v33 clip-level representation index: `{v33_clip_path}`
- v29d conservative arbitration: `{v29d_arbitration_path}`

## Dataset summary

- Crop-level baseline rows: `{crop_rows}`
- Clip-level rows: `{clip_rows_n}`
- Conservative identity subset rows: `{identity_rows}`
- Behaviour classes: `{behaviour_class_count}`
- Baseline-ready classes: `{baseline_ready_count}`
- Rare/report-only classes: `{rare_class_count}`
- Issue count: `{len(issues_df)}`

## Behaviour class readiness

{class_table}

## Split readiness

{split_table}

## Recommended usage

The crop-level index can be used as a simple baseline, but it should not be overclaimed as a temporal behaviour model. The clip-level index is more appropriate for temporal context, but many clips are multi-pig and multi-label. The conservative identity subset should be used only for identity-aware experiments where reliable identity assignment is required.

## Key modelling rule

Do not force every clip into one clean single-label class. Use either pig-level rows with identity constraints or clip-level multi-label / multi-instance formulations.

## Next step

The next step should be v35. Recommended options:

1. extract crop-level visual embeddings for the v34 crop index;
2. extract clip-level frame/temporal features for the v34 clip index;
3. prepare a simple baseline classifier only for baseline-ready classes;
4. create a rare-class reporting and grouping strategy.
"""

OUT_REPORT.write_text(report)

OUT_NOTE.write_text(
    "# Week 7 Baseline Behaviour Representation Dataset v34\n\n"
    "## Summary\n\n"
    f"- Crop-level rows: `{crop_rows}`\n"
    f"- Clip-level rows: `{clip_rows_n}`\n"
    f"- Conservative identity subset rows: `{identity_rows}`\n"
    f"- Behaviour classes: `{behaviour_class_count}`\n"
    f"- Baseline-ready classes: `{baseline_ready_count}`\n"
    f"- Rare/report-only classes: `{rare_class_count}`\n"
    f"- Issue count: `{len(issues_df)}`\n"
    f"- Ready for v35 baseline experiment or feature extraction: `{bool(ready_for_v35)}`\n\n"
    "## Outputs\n\n"
    f"- Crop-level index: `{OUT_CROP_INDEX}`\n"
    f"- Clip-level multi-label index: `{OUT_CLIP_MULTI}`\n"
    f"- Conservative identity subset: `{OUT_IDENTITY_SUBSET}`\n"
    f"- Class readiness: `{OUT_CLASS_READINESS}`\n"
    f"- Split readiness: `{OUT_SPLIT_READINESS}`\n"
    f"- Policy: `{OUT_POLICY}`\n"
    f"- Report: `{OUT_REPORT}`\n"
    f"- Decision: `{OUT_DECISION}`\n"
    f"- Issues: `{OUT_ISSUES}`\n"
)

print("Saved:")
print(OUT_CROP_INDEX)
print(OUT_CLIP_MULTI)
print(OUT_IDENTITY_SUBSET)
print(OUT_CLASS_READINESS)
print(OUT_SPLIT_READINESS)
print(OUT_POLICY)
print(OUT_REPORT)
print(OUT_DECISION)
print(OUT_ISSUES)
print(OUT_NOTE)

print()
print("=== v34 decision ===")
print(decision.to_string(index=False))

print()
print("=== v34 issues ===")
if len(issues_df):
    print(issues_df.to_string(index=False))
else:
    print("No issues found.")
