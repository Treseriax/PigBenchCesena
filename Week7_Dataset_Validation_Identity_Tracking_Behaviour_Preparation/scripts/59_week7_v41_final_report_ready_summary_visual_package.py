from pathlib import Path
from datetime import datetime
import csv
import json
import shutil
import hashlib
import zipfile
import pandas as pd

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    MATPLOTLIB_AVAILABLE = True
except Exception:
    MATPLOTLIB_AVAILABLE = False
    plt = None


ROOT = Path.home() / "PigBench"
W7 = ROOT / "Week7_Dataset_Validation_Identity_Tracking_Behaviour_Preparation"

V40 = W7 / "outputs" / "week7_behaviour_temporal_evidence_package_v40"
V39 = W7 / "outputs" / "week7_clip_multilabel_temporal_baseline_dryrun_v39"
V38 = W7 / "outputs" / "week7_clip_temporal_feature_preparation_v38"
V37 = W7 / "outputs" / "week7_crop_baseline_classifier_dryrun_v37"
V36B = W7 / "outputs" / "week7_full_crop_lightweight_feature_extraction_v36b"
V34B = W7 / "outputs" / "week7_baseline_behaviour_representation_dataset_v34b_split_fixed"

OUT_ROOT = W7 / "outputs" / "week7_final_report_ready_summary_visual_package_v41"
PACKAGE_DIR = OUT_ROOT / "Week7_Final_Report_Ready_Summary_Visual_Package_v41"
FIG_DIR = OUT_ROOT / "figures"
TABLE_DIR = OUT_ROOT / "tables"

OUT_ROOT.mkdir(parents=True, exist_ok=True)
PACKAGE_DIR.mkdir(parents=True, exist_ok=True)
FIG_DIR.mkdir(parents=True, exist_ok=True)
TABLE_DIR.mkdir(parents=True, exist_ok=True)

OUT_KEY_NUMBERS = TABLE_DIR / "week7_v41_key_numbers.csv"
OUT_CLAIM_SCOPE = TABLE_DIR / "week7_v41_claim_scope_checklist.csv"
OUT_FINAL_LIMITATIONS = TABLE_DIR / "week7_v41_final_limitations.csv"
OUT_FINAL_RECOMMENDATIONS = TABLE_DIR / "week7_v41_final_recommendations.csv"
OUT_REPORT_EN = OUT_ROOT / "Week7_Final_Report_Ready_Summary_v41.md"
OUT_REPORT_TR = OUT_ROOT / "Week7_Final_Report_Ready_Summary_v41_TR.md"
OUT_MANIFEST = OUT_ROOT / "week7_v41_package_manifest.csv"
OUT_DECISION = OUT_ROOT / "week7_v41_final_report_ready_summary_decision.csv"
OUT_ISSUES = OUT_ROOT / "week7_v41_final_report_ready_summary_issues.csv"
OUT_ZIP = OUT_ROOT / "Week7_Final_Report_Ready_Summary_Visual_Package_v41.zip"
OUT_NOTE = W7 / "notes" / "week7_v41_final_report_ready_summary_visual_package_notes.md"


def safe_to_csv(df, path):
    df.to_csv(
        path,
        index=False,
        quoting=csv.QUOTE_ALL,
        escapechar="\\",
        lineterminator="\n",
    )


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


def clean(v):
    if pd.isna(v):
        return ""
    return str(v).strip()


def sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def md_table(rows, columns):
    if not rows:
        return "_No rows available._"

    lines = []
    lines.append("| " + " | ".join(columns) + " |")
    lines.append("| " + " | ".join(["---"] * len(columns)) + " |")

    for r in rows:
        vals = []
        for c in columns:
            x = str(r.get(c, ""))
            x = x.replace("\n", " ").replace("|", "/")
            vals.append(x)
        lines.append("| " + " | ".join(vals) + " |")

    return "\n".join(lines)


issues = []
manifest_rows = []


def copy_artifact(src, rel_dst, group, required=True, description=""):
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
        issues.append({
            "item": str(src),
            "issue_type": "hard_missing_required_artifact" if required else "warning_missing_optional_artifact",
            "issue_detail": description,
        })

    manifest_rows.append({
        "group": group,
        "source_path": str(src),
        "package_relative_path": str(rel_dst),
        "required": bool(required),
        "exists": bool(exists),
        "size_bytes": size,
        "sha256": sha,
        "description": description,
    })


# ---------------------------------------------------------------------
# Inputs
# ---------------------------------------------------------------------

v40_decision_path = V40 / "week7_v40_behaviour_temporal_evidence_decision_summary.csv"
v40_key_metrics_path = V40 / "week7_v40_behaviour_temporal_evidence_key_metrics.csv"
v40_limitations_path = V40 / "week7_v40_behaviour_temporal_evidence_limitations.csv"
v40_recommendations_path = V40 / "week7_v40_behaviour_temporal_evidence_recommendations.csv"
v40_report_path = V40 / "Week7_Behaviour_Temporal_Evidence_Report_v40.md"
v40_zip_path = V40 / "Week7_Behaviour_Temporal_Evidence_Package_v40.zip"

v39_metrics_path = V39 / "week7_v39_clip_multilabel_metrics.csv"
v39_per_label_path = V39 / "week7_v39_clip_multilabel_per_label_metrics.csv"
v39_data_audit_path = V39 / "week7_v39_clip_multilabel_data_audit.csv"

v37_metrics_path = V37 / "week7_v37_crop_baseline_metrics.csv"
v38_label_summary_path = V38 / "week7_v38_multilabel_summary_by_split.csv"
v34b_class_split_path = V34B / "week7_v34b_class_by_split_readiness.csv"

v40_decision = first_row(v40_decision_path)
v40_key_metrics = read_df(v40_key_metrics_path)
v40_limitations = read_df(v40_limitations_path)
v40_recommendations = read_df(v40_recommendations_path)
v39_metrics = read_df(v39_metrics_path)
v39_per_label = read_df(v39_per_label_path)
v39_data_audit = read_df(v39_data_audit_path)
v37_metrics = read_df(v37_metrics_path)
v38_label_summary = read_df(v38_label_summary_path)
v34b_class_split = read_df(v34b_class_split_path)


required_inputs = [
    v40_decision_path,
    v40_key_metrics_path,
    v40_limitations_path,
    v40_recommendations_path,
    v40_report_path,
    v40_zip_path,
    v39_metrics_path,
    v37_metrics_path,
]

for p in required_inputs:
    if not Path(p).exists():
        issues.append({
            "item": str(p),
            "issue_type": "hard_missing_required_input",
            "issue_detail": "Required v41 input is missing.",
        })


# ---------------------------------------------------------------------
# Key numbers
# ---------------------------------------------------------------------

key_numbers = pd.DataFrame([
    {
        "category": "dataset",
        "metric": "model_ready_clip_rows",
        "value": clean(v40_decision.get("clip_temporal_test_micro_f1", "")),
        "note": "See v39/v40 metrics for clip-level baseline evidence.",
    },
    {
        "category": "crop_baseline",
        "metric": "crop_baseline_test_macro_f1",
        "value": clean(v40_decision.get("crop_baseline_test_macro_f1", "")),
        "note": "Crop-only single-frame lightweight baseline; sanity check only.",
    },
    {
        "category": "clip_temporal_baseline",
        "metric": "clip_temporal_test_micro_f1",
        "value": clean(v40_decision.get("clip_temporal_test_micro_f1", "")),
        "note": "Clip-level multi-label temporal sanity-check metric.",
    },
    {
        "category": "clip_temporal_baseline",
        "metric": "clip_temporal_test_macro_f1",
        "value": clean(v40_decision.get("clip_temporal_test_macro_f1", "")),
        "note": "Unstable because dataset and rare labels are small.",
    },
    {
        "category": "clip_temporal_baseline",
        "metric": "clip_temporal_test_sample_f1",
        "value": clean(v40_decision.get("clip_temporal_test_sample_f1", "")),
        "note": "Sample-level multi-label F1.",
    },
    {
        "category": "clip_temporal_baseline",
        "metric": "clip_temporal_test_hamming_loss",
        "value": clean(v40_decision.get("clip_temporal_test_hamming_loss", "")),
        "note": "Label-wise error rate.",
    },
    {
        "category": "package",
        "metric": "v40_package_file_count",
        "value": clean(v40_decision.get("package_file_count", "")),
        "note": "Number of files in v40 evidence package.",
    },
    {
        "category": "claim_scope",
        "metric": "final_classifier_claim",
        "value": clean(v40_decision.get("final_classifier_claim", "False")),
        "note": "Must remain False.",
    },
])

safe_to_csv(key_numbers, OUT_KEY_NUMBERS)


# ---------------------------------------------------------------------
# Claim scope
# ---------------------------------------------------------------------

claim_scope = pd.DataFrame([
    {
        "claim_type": "can_claim",
        "claim": "Week 7 produced an auditable dataset validation and behaviour-representation preparation pipeline.",
        "evidence": "v24/v30/v31/v32/v40 audit and report-ready outputs.",
    },
    {
        "claim_type": "can_claim",
        "claim": "Crop-level feature extraction and crop-only baseline sanity-check were executed.",
        "evidence": "v36b and v37 outputs.",
    },
    {
        "claim_type": "can_claim",
        "claim": "Clip-level temporal features and multi-label baseline sanity-check were executed.",
        "evidence": "v38 and v39 outputs.",
    },
    {
        "claim_type": "can_claim",
        "claim": "Clip-level temporal representation is a more appropriate direction than crop-only single-frame features.",
        "evidence": "Crop-only macro-F1 was weak; clip-level temporal pipeline produced stronger sanity-check evidence.",
    },
    {
        "claim_type": "cannot_claim",
        "claim": "This is a final behaviour classifier.",
        "evidence": "All v37/v39/v40 reports explicitly restrict claim scope to sanity-check baselines.",
    },
    {
        "claim_type": "cannot_claim",
        "claim": "The model generalizes robustly to unseen data.",
        "evidence": "Dataset is small; test split contains only 11 clips for v39.",
    },
    {
        "claim_type": "cannot_claim",
        "claim": "Rare behaviours are reliably modelled.",
        "evidence": "Rare labels have very few positive train examples.",
    },
    {
        "claim_type": "cannot_claim",
        "claim": "Tracking identities are final production-grade.",
        "evidence": "Identity-linking remains conservative and review-aware.",
    },
])

safe_to_csv(claim_scope, OUT_CLAIM_SCOPE)


# ---------------------------------------------------------------------
# Final limitations and recommendations
# ---------------------------------------------------------------------

if len(v40_limitations):
    final_limitations = v40_limitations.copy()
else:
    final_limitations = pd.DataFrame([
        {
            "limitation": "small_dataset",
            "detail": "Dataset and test split are small.",
            "severity": "high",
        }
    ])

safe_to_csv(final_limitations, OUT_FINAL_LIMITATIONS)

if len(v40_recommendations):
    final_recommendations = v40_recommendations.copy()
else:
    final_recommendations = pd.DataFrame([
        {
            "recommendation": "increase_annotation_volume",
            "detail": "Collect more labelled clips before making final model claims.",
            "priority": "high",
        }
    ])

safe_to_csv(final_recommendations, OUT_FINAL_RECOMMENDATIONS)


# ---------------------------------------------------------------------
# Optional figures
# ---------------------------------------------------------------------

figure_paths = []

if MATPLOTLIB_AVAILABLE:
    # Figure 1: metric comparison
    try:
        vals = {
            "Crop macro-F1": float(clean(v40_decision.get("crop_baseline_test_macro_f1", 0)) or 0),
            "Clip micro-F1": float(clean(v40_decision.get("clip_temporal_test_micro_f1", 0)) or 0),
            "Clip macro-F1": float(clean(v40_decision.get("clip_temporal_test_macro_f1", 0)) or 0),
            "Clip sample-F1": float(clean(v40_decision.get("clip_temporal_test_sample_f1", 0)) or 0),
        }

        fig_path = FIG_DIR / "week7_v41_crop_vs_clip_metric_comparison.png"
        plt.figure(figsize=(8, 4.5))
        plt.bar(list(vals.keys()), list(vals.values()))
        plt.ylim(0, 1)
        plt.ylabel("Score")
        plt.title("Week 7 sanity-check baseline metrics")
        plt.xticks(rotation=25, ha="right")
        plt.tight_layout()
        plt.savefig(fig_path, dpi=160)
        plt.close()
        figure_paths.append(fig_path)
    except Exception as e:
        issues.append({
            "item": "metric_comparison_figure",
            "issue_type": "warning_figure_generation_failed",
            "issue_detail": str(e),
        })

    # Figure 2: label counts by split
    try:
        if len(v39_data_audit):
            label_cols = [c for c in v39_data_audit.columns if c.startswith("label__")]
            plot_df = v39_data_audit.copy()
            plot_df = plot_df[plot_df["split"].isin(["train", "val", "test"])]

            fig_path = FIG_DIR / "week7_v41_clip_label_counts_by_split.png"
            plt.figure(figsize=(10, 5))
            x = range(len(label_cols))
            width = 0.25

            splits = ["train", "val", "test"]
            offsets = [-width, 0, width]

            for split, offset in zip(splits, offsets):
                row = plot_df[plot_df["split"] == split]
                if len(row):
                    vals = [int(row.iloc[0][c]) for c in label_cols]
                else:
                    vals = [0 for _ in label_cols]
                plt.bar([i + offset for i in x], vals, width=width, label=split)

            plt.xticks(list(x), [c.replace("label__", "") for c in label_cols], rotation=45, ha="right")
            plt.ylabel("Positive clip count")
            plt.title("Week 7 clip-level label counts by split")
            plt.legend()
            plt.tight_layout()
            plt.savefig(fig_path, dpi=160)
            plt.close()
            figure_paths.append(fig_path)
    except Exception as e:
        issues.append({
            "item": "label_counts_figure",
            "issue_type": "warning_figure_generation_failed",
            "issue_detail": str(e),
        })

    # Figure 3: report pipeline
    try:
        fig_path = FIG_DIR / "week7_v41_pipeline_summary.png"
        steps = [
            "Dataset\nvalidation",
            "Crop\nfeatures",
            "Crop baseline\nsanity-check",
            "Clip temporal\nfeatures",
            "Multi-label\nclip baseline",
            "Evidence\npackage",
        ]

        plt.figure(figsize=(11, 2.8))
        ax = plt.gca()
        ax.axis("off")

        for i, step in enumerate(steps):
            x = i / (len(steps) - 1)
            ax.text(
                x,
                0.5,
                step,
                ha="center",
                va="center",
                bbox=dict(boxstyle="round,pad=0.35", fc="white", ec="black"),
                transform=ax.transAxes,
            )
            if i < len(steps) - 1:
                ax.annotate(
                    "",
                    xy=((i + 0.75) / (len(steps) - 1), 0.5),
                    xytext=((i + 0.25) / (len(steps) - 1), 0.5),
                    arrowprops=dict(arrowstyle="->"),
                    xycoords=ax.transAxes,
                    textcoords=ax.transAxes,
                )

        plt.title("Week 7 report-ready behaviour-modelling evidence pipeline")
        plt.tight_layout()
        plt.savefig(fig_path, dpi=160)
        plt.close()
        figure_paths.append(fig_path)
    except Exception as e:
        issues.append({
            "item": "pipeline_figure",
            "issue_type": "warning_figure_generation_failed",
            "issue_detail": str(e),
        })
else:
    issues.append({
        "item": "matplotlib",
        "issue_type": "warning_matplotlib_not_available",
        "issue_detail": "Figures were skipped because matplotlib is not available.",
    })


# ---------------------------------------------------------------------
# Final reports
# ---------------------------------------------------------------------

key_numbers_table = md_table(
    key_numbers.to_dict("records"),
    ["category", "metric", "value", "note"],
)

claim_table = md_table(
    claim_scope.to_dict("records"),
    ["claim_type", "claim", "evidence"],
)

limitations_table = md_table(
    final_limitations.to_dict("records"),
    list(final_limitations.columns),
)

recommendations_table = md_table(
    final_recommendations.to_dict("records"),
    list(final_recommendations.columns),
)

figure_lines = []
for p in figure_paths:
    figure_lines.append(f"- `{p}`")

if not figure_lines:
    figure_lines.append("- No figures generated.")

figures_text = "\n".join(figure_lines)

report_en = f"""# Week 7 Final Report-Ready Summary v41

## Executive summary

Week 7 produced an audit-ready behaviour-modelling preparation workflow for the Unibo pig dataset. The work connects validated scanpoint labels, crop-level baselines, 10-second clip-level temporal features and multi-label baseline sanity checks.

The main conclusion is that crop-only single-frame features are not sufficient for reliable behaviour modelling, while clip-level temporal representation is the more appropriate direction. However, all model outputs are sanity-check baselines only. They must not be reported as final classifier performance.

## Key evidence

{key_numbers_table}

## Claim scope

{claim_table}

## Methodology summary

The Week 7 workflow followed this structure:

1. Validated and split-fixed behaviour representation.
2. Crop-level lightweight feature extraction.
3. Crop-only baseline sanity-check.
4. Clip-level temporal feature preparation from 10-second clips.
5. Clip-level multi-label temporal baseline sanity-check.
6. Evidence package creation and report-ready summary.

## Main interpretation

The crop-only baseline gives weak evidence because behaviour is temporal, contextual and often multi-pig. The clip-level representation provides more meaningful temporal evidence, but the dataset remains small and label-imbalanced.

Therefore, the correct report-ready statement is:

> The temporal/multi-label pipeline is operational and audit-ready, but further data and stronger temporal features are required before making final behaviour-classifier claims.

## Limitations

{limitations_table}

## Recommendations

{recommendations_table}

## Figures

{figures_text}

## Final claim statement

This package is report-ready evidence for Week 7 behaviour-modelling preparation. It is not a final classifier, not a production tracking system and not proof of robust generalization.
"""

OUT_REPORT_EN.write_text(report_en)

report_tr = f"""# Week 7 Final Report-Ready Summary v41 — Türkçe Özet

## Yönetici özeti

Week 7 sonunda Unibo pig dataset için audit-ready bir behaviour-modelling hazırlık hattı oluşturuldu. Bu hat; doğrulanmış scanpoint label’ları, crop-level baseline’ları, 10 saniyelik clip-level temporal feature’ları ve multi-label temporal baseline sanity-check sonuçlarını bir araya getirir.

Ana sonuç şudur: crop-only single-frame özellikler davranış modelleme için yeterli değildir. Clip-level temporal representation daha doğru yöndür. Ancak bütün baseline sonuçları yalnızca sanity-check olarak yorumlanmalıdır; final classifier performansı gibi sunulmamalıdır.

## Ana sayılar

{key_numbers_table}

## Ne iddia edebiliriz / edemeyiz?

{claim_table}

## Metodoloji özeti

1. Behaviour representation ve split düzeltmesi hazırlandı.
2. Crop-level lightweight feature extraction yapıldı.
3. Crop-only baseline sanity-check çalıştırıldı.
4. 10 saniyelik clip’lerden temporal feature hazırlandı.
5. Clip-level multi-label temporal baseline sanity-check çalıştırıldı.
6. Evidence package ve rapor-ready özet oluşturuldu.

## Ana yorum

Crop-only baseline zayıf kaldı çünkü davranış zamansal, bağlamsal ve çoklu-pig içerikli bir problem. Clip-level temporal representation daha anlamlı bir yön sağlıyor. Fakat veri seti hâlâ küçük ve label dağılımı dengesiz.

Bu yüzden raporda en doğru cümle şudur:

> Temporal / multi-label pipeline çalışır ve audit-ready durumdadır; fakat final behaviour classifier iddiası için daha fazla veri ve daha güçlü temporal feature’lar gerekir.

## Limitations

{limitations_table}

## Recommendations

{recommendations_table}

## Final claim scope

Bu paket Week 7 behaviour-modelling preparation için report-ready evidence paketidir. Final classifier değildir, production tracking sistemi değildir ve robust generalization kanıtı değildir.
"""

OUT_REPORT_TR.write_text(report_tr)


# ---------------------------------------------------------------------
# Copy artifacts
# ---------------------------------------------------------------------

copy_artifact(OUT_REPORT_EN, "01_final_reports/Week7_Final_Report_Ready_Summary_v41.md", "final_reports", True, "English final report-ready summary.")
copy_artifact(OUT_REPORT_TR, "01_final_reports/Week7_Final_Report_Ready_Summary_v41_TR.md", "final_reports", True, "Turkish final executive summary.")
copy_artifact(OUT_KEY_NUMBERS, "02_tables/week7_v41_key_numbers.csv", "tables", True, "Key numbers.")
copy_artifact(OUT_CLAIM_SCOPE, "02_tables/week7_v41_claim_scope_checklist.csv", "tables", True, "Can/cannot claim checklist.")
copy_artifact(OUT_FINAL_LIMITATIONS, "02_tables/week7_v41_final_limitations.csv", "tables", True, "Final limitations.")
copy_artifact(OUT_FINAL_RECOMMENDATIONS, "02_tables/week7_v41_final_recommendations.csv", "tables", True, "Final recommendations.")
copy_artifact(v40_report_path, "03_v40_evidence/Week7_Behaviour_Temporal_Evidence_Report_v40.md", "v40_evidence", True, "v40 evidence report.")
copy_artifact(v40_zip_path, "03_v40_evidence/Week7_Behaviour_Temporal_Evidence_Package_v40.zip", "v40_evidence", True, "v40 evidence package zip.")
copy_artifact(v39_metrics_path, "04_metrics/week7_v39_clip_multilabel_metrics.csv", "metrics", True, "v39 metrics.")
copy_artifact(v37_metrics_path, "04_metrics/week7_v37_crop_baseline_metrics.csv", "metrics", True, "v37 metrics.")

for p in figure_paths:
    copy_artifact(p, f"05_figures/{p.name}", "figures", False, "Generated v41 figure.")


# ---------------------------------------------------------------------
# Manifest / zip / decision
# ---------------------------------------------------------------------

manifest = pd.DataFrame(manifest_rows)
safe_to_csv(manifest, OUT_MANIFEST)
copy_artifact(OUT_MANIFEST, "06_manifest/week7_v41_package_manifest.csv", "manifest", True, "v41 package manifest.")

issues_df = pd.DataFrame(issues, columns=["item", "issue_type", "issue_detail"])
safe_to_csv(issues_df, OUT_ISSUES)
copy_artifact(OUT_ISSUES, "06_manifest/week7_v41_issues.csv", "manifest", True, "v41 issues.")

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

hard_issues = [x for x in issues if str(x["issue_type"]).startswith("hard_")]
warnings = [x for x in issues if str(x["issue_type"]).startswith("warning_")]

ready = len(hard_issues) == 0 and OUT_ZIP.exists() and package_file_count > 0

decision = pd.DataFrame([{
    "v41_decision": "final_report_ready_summary_visual_package_created" if ready else "final_report_ready_summary_visual_package_issues_found",
    "package_dir": str(PACKAGE_DIR),
    "package_zip": str(OUT_ZIP),
    "package_file_count": int(package_file_count),
    "package_total_bytes": int(package_total_bytes),
    "zip_size_bytes": int(zip_size),
    "zip_sha256": zip_sha,
    "matplotlib_available": bool(MATPLOTLIB_AVAILABLE),
    "figures_created": int(len(figure_paths)),
    "hard_issue_count": int(len(hard_issues)),
    "warning_count": int(len(warnings)),
    "issue_count": int(len(issues_df)),
    "final_classifier_claim": False,
    "claim_scope": "report_ready_summary_not_final_classifier",
    "report_ready": bool(ready),
    "ready_for_v42_final_audit_or_delivery": bool(ready),
    "generated_at": datetime.now().isoformat(timespec="seconds"),
}])

safe_to_csv(decision, OUT_DECISION)
copy_artifact(OUT_DECISION, "06_manifest/week7_v41_decision_summary.csv", "manifest", True, "v41 decision summary.")

OUT_NOTE.write_text(
    "# Week 7 v41 Final Report-Ready Summary / Visual Package\n\n"
    "## Summary\n\n"
    f"- Package directory: `{PACKAGE_DIR}`\n"
    f"- Package zip: `{OUT_ZIP}`\n"
    f"- Package file count: `{package_file_count}`\n"
    f"- Zip size bytes: `{zip_size}`\n"
    f"- Zip SHA256: `{zip_sha}`\n"
    f"- Matplotlib available: `{MATPLOTLIB_AVAILABLE}`\n"
    f"- Figures created: `{len(figure_paths)}`\n"
    f"- Hard issue count: `{len(hard_issues)}`\n"
    f"- Warning count: `{len(warnings)}`\n"
    f"- Report-ready: `{ready}`\n\n"
    "## Claim scope\n\n"
    "This package is a report-ready summary package. It is not a final classifier claim.\n\n"
    "## Outputs\n\n"
    f"- English report: `{OUT_REPORT_EN}`\n"
    f"- Turkish summary: `{OUT_REPORT_TR}`\n"
    f"- Key numbers: `{OUT_KEY_NUMBERS}`\n"
    f"- Claim scope: `{OUT_CLAIM_SCOPE}`\n"
    f"- Manifest: `{OUT_MANIFEST}`\n"
    f"- Decision: `{OUT_DECISION}`\n"
    f"- Issues: `{OUT_ISSUES}`\n"
    f"- Zip: `{OUT_ZIP}`\n"
)

print("Saved:")
print(OUT_REPORT_EN)
print(OUT_REPORT_TR)
print(OUT_KEY_NUMBERS)
print(OUT_CLAIM_SCOPE)
print(OUT_FINAL_LIMITATIONS)
print(OUT_FINAL_RECOMMENDATIONS)
print(OUT_MANIFEST)
print(OUT_DECISION)
print(OUT_ISSUES)
print(OUT_ZIP)
print(OUT_NOTE)

print()
print("=== v41 decision ===")
print(decision.to_string(index=False))

print()
print("=== v41 issues ===")
if len(issues_df):
    print(issues_df.to_string(index=False))
else:
    print("No issues found.")
