from pathlib import Path
from datetime import datetime
import shutil
import hashlib
import subprocess
import pandas as pd

F = Path.home() / "PigBench" / "Full_Unibo_Behaviour_Pipeline"
V80 = F / "outputs/v80_final_project_completion"
V81 = F / "outputs/v81_performance_improvement"
OUT = V81 / "04_v81_delivery_package"
PKG = OUT / "V81_Performance_Improvement_Delivery_Package"
OUT.mkdir(parents=True, exist_ok=True)

if PKG.exists():
    shutil.rmtree(PKG)
PKG.mkdir(parents=True, exist_ok=True)

m2 = pd.read_csv(V81 / "02_full_feature_models/v81d2_decision_summary.csv").iloc[0]
g = pd.read_csv(V81 / "03_comparison_with_v80/v81f1_decision_summary.csv").iloc[0]
e = pd.read_csv(V81 / "03_comparison_with_v80/v81e2_decision_summary.csv").iloc[0]

v80_macro = float(m2["v80_best_macro_f1"])
v80_acc = float(m2["v80_best_accuracy"])
v81_macro = float(m2["selected_test_macro_f1"])
v81_acc = float(m2["selected_test_accuracy"])

group_macro = float(g["selected_test_macro_f1"])
group_acc = float(g["selected_test_accuracy"])

comparison = pd.DataFrame([
    {
        "result_block": "v80_best_experimental_baseline",
        "task": "11_class_behaviour",
        "model": "limited_clip_visual_feature_baseline",
        "test_macro_f1": v80_macro,
        "test_accuracy": v80_acc,
        "selection_status": "previous_checkpoint",
        "claim_boundary": "experimental baseline",
    },
    {
        "result_block": "v81_selected_11class_model",
        "task": "11_class_behaviour",
        "model": str(m2["selected_model"]),
        "test_macro_f1": v81_macro,
        "test_accuracy": v81_acc,
        "selection_status": "selected_by_validation_macro_f1",
        "claim_boundary": "experimental 11-class classifier, not production-ready",
    },
    {
        "result_block": "v81_supplemental_grouped_classifier",
        "task": str(g["selected_group_scheme"]),
        "model": str(g["selected_model_id"]),
        "test_macro_f1": group_macro,
        "test_accuracy": group_acc,
        "selection_status": "selected_by_validation_macro_f1",
        "claim_boundary": "supplemental diagnostic grouped classifier, not replacement for 11-class",
    },
])

comparison.to_csv(OUT / "v81g1_final_model_comparison.csv", index=False)

issues = pd.DataFrame([
    {
        "item": "11_class_classifier",
        "issue_type": "macro_f1_not_improved_over_v80",
        "severity": "warning",
        "detail": "v81 selected 11-class model improves accuracy but not macro F1.",
    },
    {
        "item": "grouped_classifier",
        "issue_type": "supplemental_only",
        "severity": "info",
        "detail": "grouped classifier performs much better but must not replace the original 11-class task.",
    },
    {
        "item": "claim_boundary",
        "issue_type": "not_production_classifier",
        "severity": "info",
        "detail": "results are experimental and should be reported with limitations.",
    },
])
issues.to_csv(OUT / "v81g1_issues.csv", index=False)

decision = pd.DataFrame([{
    "v81g1_decision": "v81_performance_improvement_package_ready",
    "v80_11class_macro_f1": round(v80_macro, 6),
    "v80_11class_accuracy": round(v80_acc, 6),
    "v81_selected_11class_macro_f1": round(v81_macro, 6),
    "v81_selected_11class_accuracy": round(v81_acc, 6),
    "v81_grouped_macro_f1": round(group_macro, 6),
    "v81_grouped_accuracy": round(group_acc, 6),
    "main_finding": "11class_accuracy_improved_but_macro_f1_limited_grouped_classifier_successful",
    "hard_issue_count": 0,
    "warning_count": 1,
    "ready_for_final_handover": True,
    "claim_scope": "experimental_results_with_clear_limitations",
    "generated_at": datetime.now().isoformat(timespec="seconds"),
}])
decision.to_csv(OUT / "v81g1_decision_summary.csv", index=False)

report = OUT / "v81g1_final_comparison_report.md"
report.write_text(
    "# v81g1 Final Performance Improvement Comparison\n\n"
    "## Executive summary\n\n"
    "v81 completed a full 2768-clip visual-temporal feature extraction and trained several experimental classifiers.\n\n"
    "The selected 11-class model improves accuracy compared with v80, but it does not improve macro F1. "
    "Class-level analysis shows that rare classes and visually overlapping labels remain the main bottleneck.\n\n"
    "A supplemental grouped behaviour classifier was therefore added. This grouped classifier performs much better, "
    "showing that broader behaviour categories are more learnable under the current feature representation.\n\n"
    "## Main results\n\n"
    f"- v80 best 11-class macro F1: {v80_macro:.6f}\n"
    f"- v80 best 11-class accuracy: {v80_acc:.6f}\n"
    f"- v81 selected 11-class macro F1: {v81_macro:.6f}\n"
    f"- v81 selected 11-class accuracy: {v81_acc:.6f}\n"
    f"- v81 supplemental grouped macro F1: {group_macro:.6f}\n"
    f"- v81 supplemental grouped accuracy: {group_acc:.6f}\n\n"
    "## Interpretation\n\n"
    "The 11-class result should be reported as the original-label classifier. It is useful, but limited. "
    "The grouped classifier should be reported separately as a supplemental diagnostic model, not as a replacement.\n\n"
    "## Claim boundaries\n\n"
    "- No production-ready behaviour classifier is claimed.\n"
    "- The 11-class classifier remains experimental.\n"
    "- The grouped classifier is supplemental and diagnostic.\n"
    "- The exploratory best-test model is not selected as the official model because test-set model selection would be leakage.\n"
    "- The main technical bottleneck is class imbalance, rare labels, and visually overlapping behaviours.\n",
    encoding="utf-8",
)

readme = PKG / "README_V81_PERFORMANCE_IMPROVEMENT.md"
readme.write_text(
    "# V81 Performance Improvement Delivery Package\n\n"
    "This package contains the v81 performance improvement branch outputs.\n\n"
    "## Contents\n\n"
    "- Full visual-temporal feature extraction summaries\n"
    "- 11-class model metrics and diagnostics\n"
    "- Class imbalance/error analysis\n"
    "- Strategy selection report\n"
    "- Supplemental grouped classifier results\n"
    "- Final comparison report\n"
    "- Reproducibility scripts\n\n"
    "## Main result\n\n"
    f"- v81 selected 11-class macro F1: {v81_macro:.6f}\n"
    f"- v81 selected 11-class accuracy: {v81_acc:.6f}\n"
    f"- v81 supplemental grouped classifier macro F1: {group_macro:.6f}\n"
    f"- v81 supplemental grouped classifier accuracy: {group_acc:.6f}\n\n"
    "## Claim boundary\n\n"
    "The grouped classifier is a supplemental diagnostic model and does not replace the original 11-class classifier. "
    "No production-ready behaviour recognition claim is made.\n",
    encoding="utf-8",
)

# Copy notes.
notes_dst = PKG / "notes"
notes_dst.mkdir(exist_ok=True)
for p in (F / "notes").glob("v81*.md"):
    shutil.copy2(p, notes_dst / p.name)

# Copy scripts.
scripts_dst = PKG / "scripts"
scripts_dst.mkdir(exist_ok=True)
for p in (F / "scripts").glob("*v81*.py"):
    shutil.copy2(p, scripts_dst / p.name)
for p in (F / "scripts").glob("10*_v81*.py"):
    shutil.copy2(p, scripts_dst / p.name)

# Copy important outputs, excluding joblib models.
outputs_dst = PKG / "outputs"
outputs_dst.mkdir(exist_ok=True)

for sub in [
    "00_branch_registry",
    "01_full_visual_temporal_features",
    "02_full_feature_models",
    "03_comparison_with_v80",
    "04_v81_delivery_package",
]:
    src = V81 / sub
    dst = outputs_dst / sub
    if src.exists():
        shutil.copytree(
            src,
            dst,
            ignore=shutil.ignore_patterns("*.joblib", "*.pid", "*.log", "V81_Performance_Improvement_Delivery_Package", "*.zip"),
            dirs_exist_ok=True,
        )

# Include final comparison files explicitly.
for p in [OUT / "v81g1_final_model_comparison.csv", OUT / "v81g1_decision_summary.csv", OUT / "v81g1_final_comparison_report.md", OUT / "v81g1_issues.csv"]:
    shutil.copy2(p, PKG / p.name)

zip_base = OUT / "V81_Performance_Improvement_Delivery_Package_LITE"
zip_path = shutil.make_archive(str(zip_base), "zip", root_dir=OUT, base_dir="V81_Performance_Improvement_Delivery_Package")

h = hashlib.sha256()
with open(zip_path, "rb") as f:
    for chunk in iter(lambda: f.read(1024 * 1024), b""):
        h.update(chunk)
sha = h.hexdigest()

size_bytes = Path(zip_path).stat().st_size
file_count = sum(1 for x in PKG.rglob("*") if x.is_file())

manifest = pd.DataFrame([{
    "package_path": str(PKG),
    "zip_path": str(zip_path),
    "sha256": sha,
    "size_bytes": size_bytes,
    "file_count": file_count,
    "generated_at": datetime.now().isoformat(timespec="seconds"),
}])
manifest.to_csv(OUT / "v81g1_delivery_package_manifest.csv", index=False)

print(decision.to_string(index=False))
print("=== comparison ===")
print(comparison.to_string(index=False))
print("=== package ===")
print(manifest.to_string(index=False))
