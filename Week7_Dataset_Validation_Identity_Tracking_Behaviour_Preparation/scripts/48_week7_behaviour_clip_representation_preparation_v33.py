from pathlib import Path
from datetime import datetime
import csv
import pandas as pd
import numpy as np


ROOT = Path.home() / "PigBench"
W7 = ROOT / "Week7_Dataset_Validation_Identity_Tracking_Behaviour_Preparation"

OUT_ROOT = W7 / "outputs" / "week7_behaviour_clip_representation_preparation_v33"
OUT_ROOT.mkdir(parents=True, exist_ok=True)

OUT_CLIP_INDEX = OUT_ROOT / "week7_behaviour_clip_representation_v33_clip_level_index.csv"
OUT_LABEL_SPACE = OUT_ROOT / "week7_behaviour_clip_representation_v33_behaviour_label_space.csv"
OUT_REPRESENTATION_DESIGN = OUT_ROOT / "week7_behaviour_clip_representation_v33_representation_design.csv"
OUT_MODELLING_POLICY = OUT_ROOT / "week7_behaviour_clip_representation_v33_modelling_policy.csv"
OUT_LIMITATIONS = OUT_ROOT / "week7_behaviour_clip_representation_v33_limitations.csv"
OUT_REPORT = OUT_ROOT / "week7_behaviour_clip_representation_v33_report.md"
OUT_DECISION = OUT_ROOT / "week7_behaviour_clip_representation_v33_decision_summary.csv"
OUT_ISSUES = OUT_ROOT / "week7_behaviour_clip_representation_v33_issues.csv"
OUT_NOTE = W7 / "notes" / "week7_behaviour_clip_representation_preparation_v33_notes.md"


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
            val -= 20
        if "summary" in s:
            val += 5
        if "dataset" in s:
            val += 5
        return val

    return sorted(candidates, key=lambda p: (-score(p), str(p)))[0]


def pick_col(df, candidates):
    for c in candidates:
        if c in df.columns:
            return c
    return None


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

v26_path = W7 / "outputs" / "clip_extraction_temporal_qa_v26" / "week7_clip_extraction_v26_index.csv"
if not v26_path.exists():
    found = find_csv(W7 / "outputs", ["v26", "clip"], ["index", "clip_extraction"])
    if found:
        v26_path = found

v18c_path = W7 / "outputs" / "behaviour_label_fusion_v18c_verified_crosswalk" / "week7_behaviour_label_fusion_v18c_box_level_dataset.csv"
if not v18c_path.exists():
    found = find_csv(W7 / "outputs", ["v18c"], ["box_level", "training", "dataset"])
    if found:
        v18c_path = found

v21_path = find_csv(W7 / "outputs", ["v21"], ["primary", "split", "dataset"])
v29e_decision_path = W7 / "outputs" / "final_identity_linking_report_package_v29e" / "week7_final_identity_linking_report_package_v29e_decision_summary.csv"
v31_decision_path = W7 / "outputs" / "week7_complete_final_audit_v31" / "week7_complete_final_audit_v31_decision_summary.csv"
v32_decision_path = W7 / "outputs" / "week7_report_ready_methodology_results_v32" / "week7_report_ready_methodology_results_v32_decision_summary.csv"

for key, path in [
    ("v26_clip_index", v26_path),
    ("v18c_behaviour_fusion_dataset", v18c_path),
]:
    if path is None or not Path(path).exists():
        issues.append({
            "item": key,
            "issue_type": "missing_required_file",
            "issue_detail": str(path),
        })

v26 = read_df(v26_path)
v18c = read_df(v18c_path)
v21 = read_df(v21_path) if v21_path else pd.DataFrame()
v29e = read_first(v29e_decision_path)
v31 = read_first(v31_decision_path)
v32 = read_first(v32_decision_path)

if len(v26) == 0:
    issues.append({
        "item": "v26_clip_index",
        "issue_type": "empty_or_unreadable",
        "issue_detail": str(v26_path),
    })

if len(v18c) == 0:
    issues.append({
        "item": "v18c_behaviour_fusion_dataset",
        "issue_type": "empty_or_unreadable",
        "issue_detail": str(v18c_path),
    })

if len(issues):
    issues_df = pd.DataFrame(issues, columns=["item", "issue_type", "issue_detail"])
    safe_to_csv(issues_df, OUT_ISSUES)
    print("Issues found before processing:")
    print(issues_df.to_string(index=False))
    raise SystemExit(1)

for df in [v26, v18c, v21]:
    if "scan_frame_id" in df.columns:
        df["scan_frame_id"] = df["scan_frame_id"].fillna("").astype(str).str.strip()

behaviour_col = pick_col(
    v18c,
    [
        "behaviour_code",
        "behaviour_code_label",
        "behaviour_label",
        "behaviour",
        "label",
        "action",
    ],
)

pig_id_col = pick_col(
    v18c,
    [
        "behaviour_pig_id",
        "pig_id",
        "assigned_behaviour_pig_id",
        "visual_behaviour_pig_id",
    ],
)

colour_col = pick_col(
    v18c,
    [
        "visual_marker_colour",
        "final_colour_identity_v17",
        "assigned_visual_colour",
        "final_colour",
        "colour",
    ],
)

status_col = pick_col(
    v18c,
    [
        "status",
        "fusion_status",
        "identity_status",
        "assignment_status",
    ],
)

if behaviour_col is None:
    issues.append({
        "item": "v18c_behaviour_column",
        "issue_type": "missing_column",
        "issue_detail": "Could not identify behaviour label column.",
    })

if pig_id_col is None:
    issues.append({
        "item": "v18c_pig_id_column",
        "issue_type": "missing_column",
        "issue_detail": "Could not identify pig ID column.",
    })

if len(issues):
    issues_df = pd.DataFrame(issues, columns=["item", "issue_type", "issue_detail"])
    safe_to_csv(issues_df, OUT_ISSUES)
    print("Issues found:")
    print(issues_df.to_string(index=False))
    raise SystemExit(1)

# Training-ready style filtering:
# Keep rows with a real behaviour label and a real pig ID.
label_df = v18c.copy()
label_df[behaviour_col] = label_df[behaviour_col].fillna("").astype(str).str.strip()
label_df[pig_id_col] = label_df[pig_id_col].fillna("").astype(str).str.strip()

label_df = label_df[
    (label_df["scan_frame_id"].astype(str).str.len() > 0)
    & (label_df[behaviour_col].astype(str).str.len() > 0)
    & (label_df[pig_id_col].astype(str).str.len() > 0)
].copy()

# Remove obvious non-training identities if they appear as text.
bad_tokens = {"unknown", "not_visible", "uncertain", "unassigned", "nan", "none"}
label_df = label_df[
    ~label_df[pig_id_col].str.lower().isin(bad_tokens)
    & ~label_df[behaviour_col].str.lower().isin(bad_tokens)
].copy()

overall_counts = label_df[behaviour_col].value_counts().to_dict()
total_label_rows = int(len(label_df))
behaviour_classes = sorted(overall_counts.keys())

rare_threshold = 10
rare_behaviours = sorted([k for k, v in overall_counts.items() if v < rare_threshold])

label_space_rows = []
for behaviour in behaviour_classes:
    count = int(overall_counts.get(behaviour, 0))
    clip_count = int(label_df[label_df[behaviour_col] == behaviour]["scan_frame_id"].nunique())
    percent = count / total_label_rows if total_label_rows else 0.0

    if count < rare_threshold:
        policy = "rare_class_report_only_or_grouped_until_more_data"
    elif count < 25:
        policy = "limited_class_use_with_caution"
    else:
        policy = "usable_for_baseline_experiments"

    label_space_rows.append({
        "behaviour_code": behaviour,
        "label_rows": count,
        "percent_of_label_rows": round(percent, 4),
        "clip_count": clip_count,
        "rare_class": bool(count < rare_threshold),
        "recommended_policy": policy,
    })

label_space = pd.DataFrame(label_space_rows).sort_values(
    ["label_rows", "behaviour_code"],
    ascending=[False, True],
)
safe_to_csv(label_space, OUT_LABEL_SPACE)

# Split mapping if available.
split_map = {}
if len(v21) and "scan_frame_id" in v21.columns:
    split_col = pick_col(v21, ["split", "primary_split", "set", "subset"])
    if split_col:
        tmp = v21[["scan_frame_id", split_col]].dropna().copy()
        tmp["scan_frame_id"] = tmp["scan_frame_id"].astype(str).str.strip()
        tmp[split_col] = tmp[split_col].astype(str).str.strip()
        split_map = dict(tmp.drop_duplicates("scan_frame_id").values)

# Clip-level representation index.
clip_rows = []

for _, r in v26.iterrows():
    scan_id = clean(r.get("scan_frame_id", ""))
    if not scan_id:
        continue

    g = label_df[label_df["scan_frame_id"] == scan_id].copy()

    behaviours = sorted(g[behaviour_col].dropna().astype(str).unique().tolist())
    pig_ids = sorted(g[pig_id_col].dropna().astype(str).unique().tolist())

    if len(g):
        counts = g[behaviour_col].value_counts()
        majority_behaviour = str(counts.index[0])
        majority_count = int(counts.iloc[0])
    else:
        majority_behaviour = ""
        majority_count = 0

    rare_present = sorted([b for b in behaviours if b in rare_behaviours])

    if len(behaviours) == 0:
        problem_type = "unlabelled_or_no_training_rows"
        recommended_target = "exclude_until_review"
    elif len(behaviours) == 1:
        problem_type = "single_behaviour_clip_or_scanpoint"
        recommended_target = "single_label_or_pig_level_rows"
    else:
        problem_type = "multi_pig_multi_behaviour_clip"
        recommended_target = "multi_label_clip_or_multi_instance_pig_level"

    if rare_present:
        rare_policy = "contains_rare_behaviour_review_or_report_carefully"
    else:
        rare_policy = "no_rare_behaviour_flag"

    row = {
        "scan_frame_id": scan_id,
        "clip_path": clean(r.get("clip_path", "")),
        "preview_frame_path": clean(r.get("preview_frame_path", "")),
        "source_video": clean(r.get("video_name", r.get("video_id", ""))),
        "start_sec": clean(r.get("start_sec", r.get("clip_start_sec", ""))),
        "end_sec": clean(r.get("end_sec", r.get("clip_end_sec", ""))),
        "center_sec": clean(r.get("center_sec", "")),
        "edge_adjusted": clean(r.get("edge_adjusted", "")),
        "split": split_map.get(scan_id, ""),
        "label_rows": int(len(g)),
        "pig_count": int(len(pig_ids)),
        "behaviour_count": int(len(behaviours)),
        "behaviour_set": " | ".join(behaviours),
        "pig_id_set": " | ".join(pig_ids),
        "majority_behaviour": majority_behaviour,
        "majority_behaviour_count": majority_count,
        "rare_behaviour_present": " | ".join(rare_present),
        "problem_type": problem_type,
        "recommended_target": recommended_target,
        "rare_policy": rare_policy,
    }

    clip_rows.append(row)

clip_index = pd.DataFrame(clip_rows)
safe_to_csv(clip_index, OUT_CLIP_INDEX)

# Representation design table.
representation_rows = [
    {
        "representation_id": "R1",
        "representation_name": "pig_crop_single_frame_baseline",
        "input_data": "v22 model-ready pig crops",
        "target": "pig-level behaviour label at scanpoint",
        "advantages": "simple baseline; uses existing crop dataset; easy to train and debug",
        "risks": "weak for behaviours requiring motion/context; rare classes underrepresented",
        "recommended_use": "baseline only",
    },
    {
        "representation_id": "R2",
        "representation_name": "clip_level_multi_label_context",
        "input_data": "v26 10-second clips",
        "target": "set of behaviours present around scanpoint",
        "advantages": "captures temporal context and multiple pigs",
        "risks": "clip may contain several behaviours and identities; requires careful label definition",
        "recommended_use": "main next representation candidate",
    },
    {
        "representation_id": "R3",
        "representation_name": "identity_aware_clip_tracklet_representation",
        "input_data": "v28d dense polygon tracklets + v29d conservative identities",
        "target": "identity-aware behaviour candidate per tracklet",
        "advantages": "uses marker-colour identity constraints; supports pig-specific temporal analysis",
        "risks": "only conservative identity candidates should be used; review queue cannot be treated as final",
        "recommended_use": "use for reviewed/conservative subset",
    },
    {
        "representation_id": "R4",
        "representation_name": "motion_feature_representation",
        "input_data": "tracklet trajectories from dense polygon-filtered tracking",
        "target": "hand-crafted temporal features for behaviour cues",
        "advantages": "captures displacement, speed, stillness and interaction cues",
        "risks": "tracking fragmentation affects feature quality",
        "recommended_use": "diagnostic and supplementary features",
    },
    {
        "representation_id": "R5",
        "representation_name": "hybrid_future_representation",
        "input_data": "clip frames + crops + motion + identity candidates",
        "target": "behaviour classification or retrieval-ready embeddings",
        "advantages": "most complete representation",
        "risks": "requires more engineering and careful validation",
        "recommended_use": "future phase after baseline",
    },
]

representation_design = pd.DataFrame(representation_rows)
safe_to_csv(representation_design, OUT_REPRESENTATION_DESIGN)

policy_rows = [
    {
        "policy_area": "problem_definition",
        "policy": "Do not treat every 10-second clip as a clean single-label sample. Many clips contain multiple pigs and potentially multiple behaviours.",
    },
    {
        "policy_area": "identity",
        "policy": "Use v29d conservative accepted identities as the safest identity subset. Review-required candidates must remain separate.",
    },
    {
        "policy_area": "rare_classes",
        "policy": "Rare behaviours should be reported carefully and should not be overclaimed as learnable without additional data.",
    },
    {
        "policy_area": "splitting",
        "policy": "Use the locked primary split for baseline experiments when available, and explicitly report any same-video leakage or temporal bias limitation.",
    },
    {
        "policy_area": "evaluation",
        "policy": "Use per-class metrics and confusion analysis. Do not rely only on overall accuracy because the dataset is imbalanced.",
    },
    {
        "policy_area": "output_claim",
        "policy": "The current output is behaviour-representation preparation, not final behaviour classification.",
    },
]

modelling_policy = pd.DataFrame(policy_rows)
safe_to_csv(modelling_policy, OUT_MODELLING_POLICY)

limitation_rows = [
    {
        "limitation": "small_dataset",
        "detail": "The number of labelled scanpoint examples is limited, and rare behaviour classes have very few rows.",
    },
    {
        "limitation": "multi_pig_scene",
        "detail": "A clip may contain multiple pigs, identities and behaviours, so clip-level labels must be interpreted carefully.",
    },
    {
        "limitation": "identity_uncertainty",
        "detail": "Only conservative accepted identities should be considered reliable without further review.",
    },
    {
        "limitation": "tracking_fragmentation",
        "detail": "Simple-IoU tracklets remain fragmented, especially in crowded and occluded clips.",
    },
    {
        "limitation": "temporal_context",
        "detail": "Single-frame crop labels may be insufficient for behaviours that require motion or interaction context.",
    },
]

limitations = pd.DataFrame(limitation_rows)
safe_to_csv(limitations, OUT_LIMITATIONS)

# Report
clip_table_rows = clip_index[
    [
        "scan_frame_id",
        "label_rows",
        "pig_count",
        "behaviour_count",
        "behaviour_set",
        "rare_behaviour_present",
        "problem_type",
        "recommended_target",
    ]
].head(20).to_dict("records")

label_table_rows = label_space.to_dict("records")

report = f"""# Week 7 Behaviour / Clip-Level Representation Preparation v33

## Purpose

This step prepares the next phase after Week 7 dataset validation and identity-linking. The goal is to define how the available 10-second clips, scanpoint labels, pig identities and conservative tracklet identities should be represented for behaviour modelling.

This step does not train a behaviour model. It creates a representation design, a clip-level index, a behaviour label-space summary and modelling policies.

## Inputs

- Clip index: `{v26_path}`
- Behaviour fusion dataset: `{v18c_path}`
- Optional split file: `{v21_path}`
- Final identity-linking decision: `{v29e_decision_path}`
- Final audit decision: `{v31_decision_path}`
- Report-ready text decision: `{v32_decision_path}`

## Key dataset facts

- Clip rows prepared: `{len(clip_index)}`
- Behaviour label rows used: `{len(label_df)}`
- Behaviour classes: `{len(behaviour_classes)}`
- Rare behaviours under threshold {rare_threshold}: `{', '.join(rare_behaviours) if rare_behaviours else 'none'}`
- Conservative accepted tracklets from v29e: `{clean(v29e.get('conservative_accepted_tracklets', 'not available'))}`
- Review queue tracklets from v29e: `{clean(v29e.get('review_queue_tracklets', 'not available'))}`
- v31 audit pass: `{clean(v31.get('audit_pass', 'not available'))}`
- v32 report-ready: `{clean(v32.get('report_ready', 'not available'))}`

## Behaviour label space

{md_table(label_table_rows, ['behaviour_code', 'label_rows', 'percent_of_label_rows', 'clip_count', 'rare_class', 'recommended_policy'])}

## Clip-level representation preview

The following table shows the first 20 prepared clip-level rows.

{md_table(clip_table_rows, ['scan_frame_id', 'label_rows', 'pig_count', 'behaviour_count', 'behaviour_set', 'rare_behaviour_present', 'problem_type', 'recommended_target'])}

## Recommended representation strategy

The recommended next representation is not a single forced target. The safest professional design is staged:

1. Pig-crop single-frame baseline for simple debugging.
2. Clip-level multi-label representation for 10-second temporal context.
3. Identity-aware tracklet representation only for conservative accepted identities.
4. Motion-feature representation as supplementary temporal evidence.
5. Hybrid clip/crop/motion/identity representation as a future phase.

## Main modelling decision

The project should not treat all clips as clean single-label samples. Many clips contain several pigs and potentially several behaviour labels. Therefore, the next behaviour-modelling phase should support either:

- pig-level samples with identity constraints; or
- clip-level multi-label / multi-instance representation.

## Limitations

1. This is representation preparation, not final behaviour classification.
2. Rare behaviour classes should be reported carefully.
3. Review-required identities should not be treated as final labels.
4. Tracking fragmentation can affect tracklet-level temporal features.
5. Single-frame crop-only evidence may be weak for temporal behaviours.

## Next step

The next step should be v34: build a baseline behaviour representation dataset.

Recommended v34 output:

- crop-level baseline index,
- clip-level multi-label index,
- conservative identity subset index,
- rare-class handling policy,
- train/validation/test readiness checks.
"""

OUT_REPORT.write_text(report)

issues_df = pd.DataFrame(issues, columns=["item", "issue_type", "issue_detail"])
safe_to_csv(issues_df, OUT_ISSUES)

decision = pd.DataFrame([{
    "v33_decision": "behaviour_clip_representation_preparation_created",
    "clip_rows_prepared": int(len(clip_index)),
    "behaviour_label_rows_used": int(len(label_df)),
    "behaviour_classes": int(len(behaviour_classes)),
    "rare_threshold": int(rare_threshold),
    "rare_behaviours": " | ".join(rare_behaviours),
    "v31_audit_pass": clean(v31.get("audit_pass", "")),
    "v32_report_ready": clean(v32.get("report_ready", "")),
    "representation_designs_created": int(len(representation_design)),
    "modelling_policies_created": int(len(modelling_policy)),
    "limitations_created": int(len(limitations)),
    "issue_count": int(len(issues_df)),
    "ready_for_v34_baseline_behaviour_representation_dataset": True,
    "generated_at": datetime.now().isoformat(timespec="seconds"),
}])

safe_to_csv(decision, OUT_DECISION)

OUT_NOTE.write_text(
    "# Week 7 Behaviour / Clip-Level Representation Preparation v33\n\n"
    "## Summary\n\n"
    f"- Clip rows prepared: `{len(clip_index)}`\n"
    f"- Behaviour label rows used: `{len(label_df)}`\n"
    f"- Behaviour classes: `{len(behaviour_classes)}`\n"
    f"- Rare behaviours: `{', '.join(rare_behaviours) if rare_behaviours else 'none'}`\n"
    f"- Representation designs created: `{len(representation_design)}`\n"
    f"- Modelling policies created: `{len(modelling_policy)}`\n"
    f"- Issue count: `{len(issues_df)}`\n"
    f"- Ready for v34 baseline behaviour representation dataset: `True`\n\n"
    "## Outputs\n\n"
    f"- Clip-level index: `{OUT_CLIP_INDEX}`\n"
    f"- Behaviour label space: `{OUT_LABEL_SPACE}`\n"
    f"- Representation design: `{OUT_REPRESENTATION_DESIGN}`\n"
    f"- Modelling policy: `{OUT_MODELLING_POLICY}`\n"
    f"- Limitations: `{OUT_LIMITATIONS}`\n"
    f"- Report: `{OUT_REPORT}`\n"
    f"- Decision summary: `{OUT_DECISION}`\n"
    f"- Issues: `{OUT_ISSUES}`\n"
)

print("Saved:")
print(OUT_CLIP_INDEX)
print(OUT_LABEL_SPACE)
print(OUT_REPRESENTATION_DESIGN)
print(OUT_MODELLING_POLICY)
print(OUT_LIMITATIONS)
print(OUT_REPORT)
print(OUT_DECISION)
print(OUT_ISSUES)
print(OUT_NOTE)

print()
print("=== v33 decision ===")
print(decision.to_string(index=False))

print()
print("=== v33 issues ===")
if len(issues_df):
    print(issues_df.to_string(index=False))
else:
    print("No issues found.")
