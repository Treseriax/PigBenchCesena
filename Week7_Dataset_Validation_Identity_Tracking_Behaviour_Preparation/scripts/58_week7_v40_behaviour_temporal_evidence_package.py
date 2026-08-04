from pathlib import Path
from datetime import datetime
import csv
import json
import shutil
import hashlib
import zipfile
import pandas as pd


ROOT = Path.home() / "PigBench"
W7 = ROOT / "Week7_Dataset_Validation_Identity_Tracking_Behaviour_Preparation"

OUT_ROOT = W7 / "outputs" / "week7_behaviour_temporal_evidence_package_v40"
PACKAGE_DIR = OUT_ROOT / "Week7_Behaviour_Temporal_Evidence_Package_v40"
OUT_ROOT.mkdir(parents=True, exist_ok=True)
PACKAGE_DIR.mkdir(parents=True, exist_ok=True)

# Input roots
V34B = W7 / "outputs" / "week7_baseline_behaviour_representation_dataset_v34b_split_fixed"
V35 = W7 / "outputs" / "week7_baseline_feature_experiment_readiness_v35"
V36B = W7 / "outputs" / "week7_full_crop_lightweight_feature_extraction_v36b"
V37 = W7 / "outputs" / "week7_crop_baseline_classifier_dryrun_v37"
V38 = W7 / "outputs" / "week7_clip_temporal_feature_preparation_v38"
V39 = W7 / "outputs" / "week7_clip_multilabel_temporal_baseline_dryrun_v39"

OUT_KEY_METRICS = OUT_ROOT / "week7_v40_behaviour_temporal_evidence_key_metrics.csv"
OUT_LIMITATIONS = OUT_ROOT / "week7_v40_behaviour_temporal_evidence_limitations.csv"
OUT_RECOMMENDATIONS = OUT_ROOT / "week7_v40_behaviour_temporal_evidence_recommendations.csv"
OUT_MANIFEST = OUT_ROOT / "week7_v40_behaviour_temporal_evidence_manifest.csv"
OUT_REPORT = OUT_ROOT / "Week7_Behaviour_Temporal_Evidence_Report_v40.md"
OUT_DECISION = OUT_ROOT / "week7_v40_behaviour_temporal_evidence_decision_summary.csv"
OUT_ISSUES = OUT_ROOT / "week7_v40_behaviour_temporal_evidence_issues.csv"
OUT_ZIP = OUT_ROOT / "Week7_Behaviour_Temporal_Evidence_Package_v40.zip"
OUT_NOTE = W7 / "notes" / "week7_v40_behaviour_temporal_evidence_package_notes.md"


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


def first_row(path):
    df = read_df(path)
    if len(df):
        return df.iloc[0].to_dict()
    return {}


def get_metric(df, model_name, split, metric, default=""):
    if df is None or len(df) == 0:
        return default
    sub = df[(df["model_name"] == model_name) & (df["split"] == split)]
    if len(sub) == 0 or metric not in sub.columns:
        return default
    return clean(sub.iloc[0][metric])


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


def sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


issues = []
manifest_rows = []


def copy_artifact(src, rel_dst, artifact_group, required=True, description=""):
    src = Path(src)
    dst = PACKAGE_DIR / rel_dst
    dst.parent.mkdir(parents=True, exist_ok=True)

    exists = src.exists()

    if exists:
        shutil.copy2(src, dst)
        size = dst.stat().st_size
        sha = sha256_file(dst)
    else:
        size = ""
        sha = ""
        if required:
            issues.append({
                "item": str(src),
                "issue_type": "hard_missing_required_artifact",
                "issue_detail": description or str(rel_dst),
            })
        else:
            issues.append({
                "item": str(src),
                "issue_type": "warning_missing_optional_artifact",
                "issue_detail": description or str(rel_dst),
            })

    manifest_rows.append({
        "artifact_group": artifact_group,
        "source_path": str(src),
        "package_relative_path": str(rel_dst),
        "required": bool(required),
        "exists": bool(exists),
        "size_bytes": size,
        "sha256": sha,
        "description": description,
    })


# ---------------------------------------------------------------------
# Read key inputs
# ---------------------------------------------------------------------

v36b_decision_path = V36B / "week7_v36b_full_crop_lightweight_feature_extraction_decision_summary.csv"
v37_decision_path = V37 / "week7_v37_crop_baseline_classifier_dryrun_decision_summary.csv"
v37_metrics_path = V37 / "week7_v37_crop_baseline_metrics.csv"
v37_issues_path = V37 / "week7_v37_crop_baseline_classifier_dryrun_issues.csv"

v38_decision_path = V38 / "week7_v38_clip_temporal_feature_preparation_decision_summary.csv"
v38_label_summary_path = V38 / "week7_v38_multilabel_summary_by_split.csv"
v38_feature_summary_path = V38 / "week7_v38_clip_feature_summary.csv"
v38_issues_path = V38 / "week7_v38_clip_temporal_feature_preparation_issues.csv"

v39_decision_path = V39 / "week7_v39_clip_multilabel_temporal_baseline_decision_summary.csv"
v39_metrics_path = V39 / "week7_v39_clip_multilabel_metrics.csv"
v39_per_label_path = V39 / "week7_v39_clip_multilabel_per_label_metrics.csv"
v39_data_audit_path = V39 / "week7_v39_clip_multilabel_data_audit.csv"
v39_issues_path = V39 / "week7_v39_clip_multilabel_temporal_baseline_issues.csv"

v34b_split_readiness_path = V34B / "week7_v34b_split_readiness_checks.csv"
v34b_class_split_path = V34B / "week7_v34b_class_by_split_readiness.csv"
v35_dataset_card_path = V35 / "week7_v35_dataset_card.md"

v36b_decision = first_row(v36b_decision_path)
v37_decision = first_row(v37_decision_path)
v38_decision = first_row(v38_decision_path)
v39_decision = first_row(v39_decision_path)

v37_metrics = read_df(v37_metrics_path)
v39_metrics = read_df(v39_metrics_path)
v39_per_label = read_df(v39_per_label_path)
v39_data_audit = read_df(v39_data_audit_path)
v38_label_summary = read_df(v38_label_summary_path)

# ---------------------------------------------------------------------
# Copy artifacts into package
# ---------------------------------------------------------------------

copy_items = [
    # Dataset readiness
    (v34b_split_readiness_path, "01_dataset_readiness/week7_v34b_split_readiness_checks.csv", "dataset_readiness", True, "Split readiness after v34b."),
    (v34b_class_split_path, "01_dataset_readiness/week7_v34b_class_by_split_readiness.csv", "dataset_readiness", True, "Class-by-split readiness after v34b."),
    (v35_dataset_card_path, "01_dataset_readiness/week7_v35_dataset_card.md", "dataset_readiness", False, "Dataset card from v35."),

    # Crop features and crop-only baseline
    (v36b_decision_path, "02_crop_baseline/week7_v36b_feature_extraction_decision.csv", "crop_baseline", True, "Full crop lightweight feature extraction decision."),
    (V36B / "week7_v36b_class_split_counts.csv", "02_crop_baseline/week7_v36b_class_split_counts.csv", "crop_baseline", True, "Crop feature class split counts."),
    (V36B / "week7_v36b_feature_summary_by_class_split.csv", "02_crop_baseline/week7_v36b_feature_summary_by_class_split.csv", "crop_baseline", True, "Crop feature summary by class/split."),
    (v37_decision_path, "02_crop_baseline/week7_v37_crop_baseline_decision.csv", "crop_baseline", True, "Crop baseline classifier dry-run decision."),
    (v37_metrics_path, "02_crop_baseline/week7_v37_crop_baseline_metrics.csv", "crop_baseline", True, "Crop baseline metrics."),
    (V37 / "week7_v37_crop_baseline_per_class_metrics.csv", "02_crop_baseline/week7_v37_crop_baseline_per_class_metrics.csv", "crop_baseline", True, "Crop baseline per-class metrics."),
    (V37 / "week7_v37_crop_baseline_confusion_matrices.csv", "02_crop_baseline/week7_v37_crop_baseline_confusion_matrices.csv", "crop_baseline", True, "Crop baseline confusion matrices."),
    (v37_issues_path, "02_crop_baseline/week7_v37_crop_baseline_issues.csv", "crop_baseline", True, "Crop baseline warnings/issues."),

    # Clip temporal features
    (v38_decision_path, "03_clip_temporal_features/week7_v38_clip_temporal_feature_decision.csv", "clip_temporal_features", True, "Clip temporal feature preparation decision."),
    (v38_label_summary_path, "03_clip_temporal_features/week7_v38_multilabel_summary_by_split.csv", "clip_temporal_features", True, "Multi-label summary by split."),
    (v38_feature_summary_path, "03_clip_temporal_features/week7_v38_clip_feature_summary.csv", "clip_temporal_features", True, "Clip feature summary by split."),
    (V38 / "week7_v38_clip_loading_sampling_audit.csv", "03_clip_temporal_features/week7_v38_clip_loading_sampling_audit.csv", "clip_temporal_features", True, "Clip loading and frame sampling audit."),
    (v38_issues_path, "03_clip_temporal_features/week7_v38_clip_temporal_feature_issues.csv", "clip_temporal_features", True, "Clip temporal feature warnings/issues."),

    # Clip multi-label baseline
    (v39_decision_path, "04_clip_multilabel_baseline/week7_v39_clip_multilabel_decision.csv", "clip_multilabel_baseline", True, "Clip multi-label baseline decision."),
    (v39_metrics_path, "04_clip_multilabel_baseline/week7_v39_clip_multilabel_metrics.csv", "clip_multilabel_baseline", True, "Clip multi-label baseline metrics."),
    (v39_per_label_path, "04_clip_multilabel_baseline/week7_v39_clip_multilabel_per_label_metrics.csv", "clip_multilabel_baseline", True, "Clip multi-label per-label metrics."),
    (V39 / "week7_v39_clip_multilabel_label_confusion.csv", "04_clip_multilabel_baseline/week7_v39_clip_multilabel_label_confusion.csv", "clip_multilabel_baseline", True, "Clip multi-label label confusion counts."),
    (v39_data_audit_path, "04_clip_multilabel_baseline/week7_v39_clip_multilabel_data_audit.csv", "clip_multilabel_baseline", True, "Clip multi-label data audit."),
    (v39_issues_path, "04_clip_multilabel_baseline/week7_v39_clip_multilabel_issues.csv", "clip_multilabel_baseline", True, "Clip multi-label warnings/issues."),
]

for item in copy_items:
    copy_artifact(*item)

# ---------------------------------------------------------------------
# Key metrics
# ---------------------------------------------------------------------

crop_test_acc = get_metric(v37_metrics, "nearest_centroid_lightweight_features", "test", "accuracy", "")
crop_test_macro_f1 = get_metric(v37_metrics, "nearest_centroid_lightweight_features", "test", "macro_f1", "")
crop_test_weighted_f1 = get_metric(v37_metrics, "nearest_centroid_lightweight_features", "test", "weighted_f1", "")

clip_test_micro_f1 = get_metric(v39_metrics, "one_vs_rest_nearest_centroid_multilabel", "test", "micro_f1", "")
clip_test_macro_f1 = get_metric(v39_metrics, "one_vs_rest_nearest_centroid_multilabel", "test", "macro_f1", "")
clip_test_sample_f1 = get_metric(v39_metrics, "one_vs_rest_nearest_centroid_multilabel", "test", "sample_f1", "")
clip_test_hamming_loss = get_metric(v39_metrics, "one_vs_rest_nearest_centroid_multilabel", "test", "hamming_loss", "")

key_metrics_rows = [
    {
        "section": "crop_feature_extraction",
        "metric": "deduplicated_crop_feature_rows",
        "value": clean(v36b_decision.get("feature_rows", "")),
        "interpretation": "Deduplicated crop-level samples used for crop-only baseline.",
    },
    {
        "section": "crop_feature_extraction",
        "metric": "crop_feature_columns",
        "value": clean(v36b_decision.get("feature_column_count", "")),
        "interpretation": "Lightweight crop features extracted.",
    },
    {
        "section": "crop_only_baseline",
        "metric": "nearest_centroid_test_accuracy",
        "value": crop_test_acc,
        "interpretation": "Single-frame crop-only baseline sanity-check accuracy.",
    },
    {
        "section": "crop_only_baseline",
        "metric": "nearest_centroid_test_macro_f1",
        "value": crop_test_macro_f1,
        "interpretation": "Crop-only macro-F1; expected to be weak because behaviour needs temporal context.",
    },
    {
        "section": "clip_temporal_features",
        "metric": "clip_feature_rows",
        "value": clean(v38_decision.get("clip_feature_rows", "")),
        "interpretation": "Clip-level temporal feature rows.",
    },
    {
        "section": "clip_temporal_features",
        "metric": "sampled_frame_rows",
        "value": clean(v38_decision.get("sampled_frame_rows", "")),
        "interpretation": "Sampled frames used to construct clip-level temporal features.",
    },
    {
        "section": "clip_temporal_features",
        "metric": "clip_feature_columns",
        "value": clean(v38_decision.get("feature_column_count", "")),
        "interpretation": "Lightweight temporal/context features extracted per clip.",
    },
    {
        "section": "clip_multilabel_baseline",
        "metric": "model_ready_clip_rows",
        "value": clean(v39_decision.get("model_ready_clip_rows", "")),
        "interpretation": "Train/val/test clips used for clip-level multi-label baseline sanity check.",
    },
    {
        "section": "clip_multilabel_baseline",
        "metric": "centroid_test_micro_f1",
        "value": clip_test_micro_f1,
        "interpretation": "Clip-level multi-label micro-F1; sanity-check metric only.",
    },
    {
        "section": "clip_multilabel_baseline",
        "metric": "centroid_test_macro_f1",
        "value": clip_test_macro_f1,
        "interpretation": "Clip-level multi-label macro-F1; unstable due rare labels and small test set.",
    },
    {
        "section": "clip_multilabel_baseline",
        "metric": "centroid_test_sample_f1",
        "value": clip_test_sample_f1,
        "interpretation": "Average sample-level F1 for clip-level multi-label baseline.",
    },
    {
        "section": "clip_multilabel_baseline",
        "metric": "centroid_test_hamming_loss",
        "value": clip_test_hamming_loss,
        "interpretation": "Label-wise error rate for clip-level multi-label baseline.",
    },
]

key_metrics = pd.DataFrame(key_metrics_rows)
safe_to_csv(key_metrics, OUT_KEY_METRICS)

# ---------------------------------------------------------------------
# Limitations and recommendations
# ---------------------------------------------------------------------

limitations = pd.DataFrame([
    {
        "limitation": "not_final_classifier",
        "detail": "v37 and v39 are sanity-check baselines only. They should not be reported as final behaviour classifier performance.",
        "severity": "high",
    },
    {
        "limitation": "small_test_split",
        "detail": "The clip-level test split has only 11 clips, so metrics are unstable.",
        "severity": "high",
    },
    {
        "limitation": "rare_labels",
        "detail": "Several labels have fewer than 5 positive training clips, including BE, DE, IA, NU and SI.",
        "severity": "high",
    },
    {
        "limitation": "crop_only_insufficiency",
        "detail": "Crop-only single-frame lightweight features are insufficient for reliable behaviour modelling.",
        "severity": "medium",
    },
    {
        "limitation": "lightweight_temporal_features",
        "detail": "v38/v39 use lightweight temporal statistics, not deep spatiotemporal embeddings.",
        "severity": "medium",
    },
    {
        "limitation": "multi_pig_multilabel_scene",
        "detail": "Each 10-second clip may contain multiple pigs and multiple behaviours, so single-label classification is not appropriate.",
        "severity": "high",
    },
    {
        "limitation": "identity_uncertainty",
        "detail": "Identity-aware temporal modelling should use only conservative accepted identities and preserve review-required candidates separately.",
        "severity": "medium",
    },
])

safe_to_csv(limitations, OUT_LIMITATIONS)

recommendations = pd.DataFrame([
    {
        "recommendation": "use_clip_level_temporal_context",
        "detail": "Future behaviour modelling should prioritize clip-level temporal or multi-instance representations over crop-only single-frame baselines.",
        "priority": "high",
    },
    {
        "recommendation": "increase_annotation_volume",
        "detail": "Collect more labelled clips, especially for rare behaviours BE, DE, IA, NU and SI.",
        "priority": "high",
    },
    {
        "recommendation": "use_multilabel_metrics",
        "detail": "Report hamming loss, micro/macro/sample F1 and per-label metrics rather than only subset accuracy.",
        "priority": "high",
    },
    {
        "recommendation": "keep_baselines_as_sanity_checks",
        "detail": "Use crop-only and lightweight temporal baselines as sanity checks, not as final model claims.",
        "priority": "high",
    },
    {
        "recommendation": "consider_deep_temporal_embeddings",
        "detail": "After the lightweight pipeline is stable, evaluate frame/clip embeddings from video backbones or CNN encoders.",
        "priority": "medium",
    },
    {
        "recommendation": "integrate_identity_when_safe",
        "detail": "Use conservative identity-linked tracklets for identity-aware temporal analysis after review.",
        "priority": "medium",
    },
    {
        "recommendation": "prepare_report_ready_figures",
        "detail": "Create visual summary tables/plots for class imbalance, crop vs temporal baseline evidence and limitations.",
        "priority": "medium",
    },
])

safe_to_csv(recommendations, OUT_RECOMMENDATIONS)

# ---------------------------------------------------------------------
# Main evidence report
# ---------------------------------------------------------------------

metrics_table = md_table(
    key_metrics.to_dict("records"),
    ["section", "metric", "value", "interpretation"],
)

limitations_table = md_table(
    limitations.to_dict("records"),
    ["limitation", "severity", "detail"],
)

recommendations_table = md_table(
    recommendations.to_dict("records"),
    ["recommendation", "priority", "detail"],
)

v39_test_per_label = pd.DataFrame()
if len(v39_per_label):
    v39_test_per_label = v39_per_label[
        (v39_per_label["model_name"] == "one_vs_rest_nearest_centroid_multilabel")
        & (v39_per_label["split"] == "test")
    ].copy()

per_label_table = md_table(
    v39_test_per_label.to_dict("records"),
    ["label", "support", "predicted_positive", "tp", "fp", "fn", "precision", "recall", "f1"],
)

if len(v39_data_audit):
    audit_table = md_table(
        v39_data_audit.to_dict("records"),
        list(v39_data_audit.columns),
    )
else:
    audit_table = "_No v39 data audit available._"

report = f"""# Week 7 Behaviour Temporal Evidence Package v40

## Purpose

This package summarizes the evidence produced after Week 7 dataset validation, crop-level baseline preparation, clip-level temporal feature preparation and multi-label temporal baseline dry-run.

This is **not** a final behaviour classifier package. It is an audit-ready evidence package showing what was prepared, what worked, what failed, and why clip-level temporal modelling is the correct next direction.

## Evidence chain

1. Dataset validation and split-fixed representation were prepared.
2. Crop-level lightweight features were extracted successfully.
3. Crop-only baseline classifier dry-run was executed as a sanity check.
4. Clip-level temporal features were extracted from 10-second clips.
5. Clip-level multi-label temporal baseline dry-run was executed as a sanity check.
6. Limitations and next-step recommendations were consolidated.

## Key metrics

{metrics_table}

## Main interpretation

The crop-only baseline was weak, which is expected because animal behaviour cannot reliably be inferred from a single crop and a single frame. The clip-level temporal baseline is more appropriate because the labels and behaviours are temporal, multi-pig and multi-label.

The clip-level temporal baseline should still not be overclaimed. The dataset is small, rare labels are underrepresented and test metrics are unstable. The correct conclusion is that the **temporal representation pipeline works** and should be developed further with stronger features and more labels.

## v39 test per-label metrics

{per_label_table}

## v39 data audit

{audit_table}

## Limitations

{limitations_table}

## Recommendations

{recommendations_table}

## Report-ready conclusion

The Week 7 workflow produced an auditable behaviour-modelling preparation pipeline. Crop-only features were useful as a sanity check but insufficient for reliable behaviour modelling. Clip-level temporal features and multi-label targets provide a more appropriate direction for future behaviour analysis, while rare behaviours and small split sizes must be handled conservatively.

## Claim scope

- This package is evidence/report-ready.
- This package is not a final behaviour classifier.
- This package is not a production tracking result.
- All baseline metrics should be described as sanity-check evidence only.
"""

OUT_REPORT.write_text(report)
copy_artifact(OUT_REPORT, "05_report_and_decision/Week7_Behaviour_Temporal_Evidence_Report_v40.md", "final_report", True, "Main v40 evidence report.")
copy_artifact(OUT_KEY_METRICS, "05_report_and_decision/week7_v40_key_metrics.csv", "final_report", True, "Key metrics table.")
copy_artifact(OUT_LIMITATIONS, "05_report_and_decision/week7_v40_limitations.csv", "final_report", True, "Consolidated limitations.")
copy_artifact(OUT_RECOMMENDATIONS, "05_report_and_decision/week7_v40_recommendations.csv", "final_report", True, "Consolidated recommendations.")

# ---------------------------------------------------------------------
# Manifest, zip, decision
# ---------------------------------------------------------------------

manifest = pd.DataFrame(manifest_rows)
safe_to_csv(manifest, OUT_MANIFEST)
copy_artifact(OUT_MANIFEST, "05_report_and_decision/week7_v40_artifact_manifest.csv", "final_report", True, "Artifact manifest.")

# Refresh manifest after copying manifest itself.
manifest = pd.DataFrame(manifest_rows)
safe_to_csv(manifest, OUT_MANIFEST)

hard_issues = [x for x in issues if str(x["issue_type"]).startswith("hard_")]
warnings = [x for x in issues if str(x["issue_type"]).startswith("warning_")]

issues_df = pd.DataFrame(issues, columns=["item", "issue_type", "issue_detail"])
safe_to_csv(issues_df, OUT_ISSUES)
copy_artifact(OUT_ISSUES, "05_report_and_decision/week7_v40_issues.csv", "final_report", True, "v40 package issues.")

# Create zip
if OUT_ZIP.exists():
    OUT_ZIP.unlink()

with zipfile.ZipFile(OUT_ZIP, "w", compression=zipfile.ZIP_DEFLATED) as zf:
    for p in PACKAGE_DIR.rglob("*"):
        if p.is_file():
            zf.write(p, p.relative_to(PACKAGE_DIR.parent))

zip_size = OUT_ZIP.stat().st_size
zip_sha = sha256_file(OUT_ZIP)

package_file_count = len([p for p in PACKAGE_DIR.rglob("*") if p.is_file()])
package_total_bytes = sum(p.stat().st_size for p in PACKAGE_DIR.rglob("*") if p.is_file())

evidence_ready = len(hard_issues) == 0 and OUT_ZIP.exists() and package_file_count > 0

decision = pd.DataFrame([{
    "v40_decision": "behaviour_temporal_evidence_package_created" if evidence_ready else "behaviour_temporal_evidence_package_issues_found",
    "package_dir": str(PACKAGE_DIR),
    "package_zip": str(OUT_ZIP),
    "package_file_count": int(package_file_count),
    "package_total_bytes": int(package_total_bytes),
    "zip_size_bytes": int(zip_size),
    "zip_sha256": zip_sha,
    "crop_baseline_test_macro_f1": crop_test_macro_f1,
    "clip_temporal_test_micro_f1": clip_test_micro_f1,
    "clip_temporal_test_macro_f1": clip_test_macro_f1,
    "clip_temporal_test_sample_f1": clip_test_sample_f1,
    "clip_temporal_test_hamming_loss": clip_test_hamming_loss,
    "final_classifier_claim": False,
    "claim_scope": "evidence_package_sanity_check_baselines_not_final_classifier",
    "hard_issue_count": int(len(hard_issues)),
    "warning_count": int(len(warnings)),
    "issue_count": int(len(issues_df)),
    "evidence_package_ready": bool(evidence_ready),
    "ready_for_v41_final_summary_or_visual_report": bool(evidence_ready),
    "generated_at": datetime.now().isoformat(timespec="seconds"),
}])

safe_to_csv(decision, OUT_DECISION)
copy_artifact(OUT_DECISION, "05_report_and_decision/week7_v40_decision_summary.csv", "final_report", True, "v40 decision summary.")

OUT_NOTE.write_text(
    "# Week 7 v40 Behaviour Temporal Evidence Package\n\n"
    "## Summary\n\n"
    f"- Package directory: `{PACKAGE_DIR}`\n"
    f"- Package zip: `{OUT_ZIP}`\n"
    f"- Package file count: `{package_file_count}`\n"
    f"- Package total bytes: `{package_total_bytes}`\n"
    f"- Zip size bytes: `{zip_size}`\n"
    f"- Zip SHA256: `{zip_sha}`\n"
    f"- Crop baseline test macro-F1: `{crop_test_macro_f1}`\n"
    f"- Clip temporal test micro-F1: `{clip_test_micro_f1}`\n"
    f"- Clip temporal test macro-F1: `{clip_test_macro_f1}`\n"
    f"- Clip temporal test sample-F1: `{clip_test_sample_f1}`\n"
    f"- Hard issue count: `{len(hard_issues)}`\n"
    f"- Warning count: `{len(warnings)}`\n"
    f"- Evidence package ready: `{evidence_ready}`\n\n"
    "## Important claim scope\n\n"
    "This is an evidence package with sanity-check baselines. It is not a final behaviour classifier.\n\n"
    "## Main outputs\n\n"
    f"- Main report: `{OUT_REPORT}`\n"
    f"- Key metrics: `{OUT_KEY_METRICS}`\n"
    f"- Limitations: `{OUT_LIMITATIONS}`\n"
    f"- Recommendations: `{OUT_RECOMMENDATIONS}`\n"
    f"- Manifest: `{OUT_MANIFEST}`\n"
    f"- Decision: `{OUT_DECISION}`\n"
    f"- Issues: `{OUT_ISSUES}`\n"
    f"- Zip: `{OUT_ZIP}`\n"
)

print("Saved:")
print(OUT_REPORT)
print(OUT_KEY_METRICS)
print(OUT_LIMITATIONS)
print(OUT_RECOMMENDATIONS)
print(OUT_MANIFEST)
print(OUT_DECISION)
print(OUT_ISSUES)
print(OUT_ZIP)
print(OUT_NOTE)

print()
print("=== v40 decision ===")
print(decision.to_string(index=False))

print()
print("=== v40 issues ===")
if len(issues_df):
    print(issues_df.to_string(index=False))
else:
    print("No issues found.")
