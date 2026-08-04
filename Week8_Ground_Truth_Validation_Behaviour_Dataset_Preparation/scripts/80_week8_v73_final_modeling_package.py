from pathlib import Path
from datetime import datetime
import csv
import json
import hashlib
import zipfile
import pandas as pd


ROOT = Path.home() / "PigBench"
W8 = ROOT / "Week8_Ground_Truth_Validation_Behaviour_Dataset_Preparation"

V70B = W8 / "outputs" / "v70b_modeling_decision_package"
V72A = W8 / "outputs" / "v72a_frozen_detector_embedding_extraction"
V72B = W8 / "outputs" / "v72b_frozen_embedding_baseline"
V72C = W8 / "outputs" / "v72c_frozen_embedding_error_analysis_interface"

V70B_DECISION = V70B / "week8_v70b_decision_summary.csv"
V70B_CHAMPIONS = V70B / "Week8_Modeling_Decision_Package" / "week8_v70b_champion_baseline_selection.csv"
V70B_CLAIMS = V70B / "Week8_Modeling_Decision_Package" / "week8_v70b_claim_boundaries.csv"

V72A_DECISION = V72A / "week8_v72a_decision_summary.csv"
V72B_DECISION = V72B / "week8_v72b_decision_summary.csv"
V72B_BEST = V72B / "Week8_Frozen_Embedding_Baseline" / "week8_v72b_best_test_summary.csv"
V72B_COMPARE = V72B / "Week8_Frozen_Embedding_Baseline" / "week8_v72b_vs_v69b_champion_comparison.csv"

V72C_DECISION = V72C / "week8_v72c_decision_summary.csv"
V72C_CONF = V72C / "Week8_Frozen_Embedding_Error_Analysis_Interface" / "week8_v72c_top_confusions.csv"
V72C_CLASS = V72C / "Week8_Frozen_Embedding_Error_Analysis_Interface" / "week8_v72c_per_class_error_summary.csv"

OUT = W8 / "outputs" / "v73_final_modeling_package"
PKG = OUT / "Week8_Final_Modeling_Package"
NOTES = W8 / "notes"
REPORTS = W8 / "reports"
PROGRESS = W8 / "progress"

for p in [OUT, PKG, NOTES, REPORTS, PROGRESS]:
    p.mkdir(parents=True, exist_ok=True)

OUT_TIMELINE = PKG / "week8_v73_modeling_timeline.csv"
OUT_CHAMPION = PKG / "week8_v73_final_champion_selection.csv"
OUT_COMPARE = PKG / "week8_v73_v69b_vs_v72b_final_comparison.csv"
OUT_CLAIMS = PKG / "week8_v73_final_claim_boundaries.csv"
OUT_LIMITS = PKG / "week8_v73_limitations_and_next_steps.csv"
OUT_ERROR_SUMMARY = PKG / "week8_v73_error_summary.csv"
OUT_QA = PKG / "week8_v73_quality_checks.csv"
OUT_README = PKG / "README_Week8_Final_Modeling_Package.md"
OUT_MANIFEST = PKG / "week8_v73_manifest.json"

OUT_DECISION = OUT / "week8_v73_decision_summary.csv"
OUT_ISSUES = OUT / "week8_v73_issues.csv"
OUT_ZIP = OUT / "Week8_Final_Modeling_Package.zip"
OUT_SHA = OUT / "Week8_Final_Modeling_Package.sha256"
OUT_NOTE = NOTES / "week8_v73_final_modeling_package_notes.md"
OUT_REPORT = REPORTS / "week8_v73_final_modeling_package_report.md"
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


def bool_true(x):
    return str(x).strip().lower() == "true"


def fnum(x):
    try:
        return float(x)
    except Exception:
        return 0.0


def sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


issues = []

required = [
    V70B_DECISION,
    V70B_CHAMPIONS,
    V70B_CLAIMS,
    V72A_DECISION,
    V72B_DECISION,
    V72B_BEST,
    V72B_COMPARE,
    V72C_DECISION,
    V72C_CONF,
    V72C_CLASS,
]

for p in required:
    if not p.exists():
        issues.append({
            "item": str(p),
            "issue_type": "hard_missing_required_input",
            "issue_detail": "Required input for v73 final modeling package is missing.",
            "severity": "hard",
        })

if issues:
    issues_df = pd.DataFrame(issues)
    safe_to_csv(issues_df, OUT_ISSUES)
    decision = pd.DataFrame([{
        "v73_decision": "final_modeling_package_blocked",
        "hard_issue_count": int((issues_df["severity"] == "hard").sum()),
        "ready_for_v74_final_week8_delivery_package": False,
        "generated_at": datetime.now().isoformat(timespec="seconds"),
    }])
    safe_to_csv(decision, OUT_DECISION)
    print(decision.to_string(index=False))
    raise SystemExit(1)


d70b = read_csv_clean(V70B_DECISION)
d72a = read_csv_clean(V72A_DECISION)
d72b = read_csv_clean(V72B_DECISION)
d72c = read_csv_clean(V72C_DECISION)

old_champions = read_csv_clean(V70B_CHAMPIONS)
old_claims = read_csv_clean(V70B_CLAIMS)
best72b = read_csv_clean(V72B_BEST)
compare72b = read_csv_clean(V72B_COMPARE)
conf72c = read_csv_clean(V72C_CONF)
class72c = read_csv_clean(V72C_CLASS)

readiness = [
    ("v70b", bool_true(d70b.iloc[0].get("ready_for_v71_pretrained_embedding_audit", ""))),
    ("v72a", bool_true(d72a.iloc[0].get("ready_for_v72b_embedding_baseline", ""))),
    ("v72b", bool_true(d72b.iloc[0].get("ready_for_v72c_error_analysis", ""))),
    ("v72c", bool_true(d72c.iloc[0].get("ready_for_v73_final_modeling_package", ""))),
]

for stage, ok in readiness:
    if not ok:
        issues.append({
            "item": stage,
            "issue_type": "hard_previous_stage_not_ready",
            "issue_detail": f"{stage} readiness check failed.",
            "severity": "hard",
        })

timeline = pd.DataFrame([
    {
        "stage": "v69b",
        "role": "first champion baseline",
        "summary": "simple handcrafted image features with nearest centroid / kNN",
        "final_status": "superseded_by_v72b",
    },
    {
        "stage": "v70a",
        "role": "classical ablation",
        "summary": "class-balanced ridge + PCA did not beat v69b macro-F1",
        "final_status": "negative_ablation_documented",
    },
    {
        "stage": "v72a",
        "role": "embedding extraction",
        "summary": "frozen YOLOv8-s detector embeddings extracted for tight/context10 crops",
        "final_status": "completed",
    },
    {
        "stage": "v72b",
        "role": "new champion baseline",
        "summary": "frozen detector embedding baseline beat v69b macro-F1 on both split policies",
        "final_status": "selected_for_reporting",
    },
    {
        "stage": "v72c",
        "role": "error analysis and interface",
        "summary": "visual interface and error tables created for v72b champion predictions",
        "final_status": "completed",
    },
])

safe_to_csv(timeline, OUT_TIMELINE)

champion_rows = []

for _, r in best72b.iterrows():
    policy = clean(r["split_policy"])
    cmp = compare72b[compare72b["split_policy"] == policy]

    old_macro = fnum(cmp.iloc[0]["v69b_champion_macro_f1"]) if len(cmp) else 0.0
    old_acc = fnum(cmp.iloc[0]["v69b_champion_accuracy"]) if len(cmp) else 0.0
    new_macro = fnum(r["macro_f1"])
    new_acc = fnum(r["accuracy"])

    champion_rows.append({
        "split_policy": policy,
        "final_champion_stage": "v72b",
        "final_champion_experiment": "frozen_yolov8_detector_embedding_baseline",
        "crop_type": clean(r["crop_type"]),
        "selected_config": clean(r["selected_config_name"]),
        "test_accuracy": new_acc,
        "test_macro_f1": new_macro,
        "test_balanced_accuracy": fnum(r["balanced_accuracy"]),
        "v69b_previous_champion_macro_f1": old_macro,
        "delta_macro_f1_vs_v69b": new_macro - old_macro,
        "v69b_previous_champion_accuracy": old_acc,
        "delta_accuracy_vs_v69b": new_acc - old_acc,
        "champion_decision": "selected_for_reporting",
        "reason": "v72b improved test macro-F1 over v69b on this split policy",
    })

champion = pd.DataFrame(champion_rows)
safe_to_csv(champion, OUT_CHAMPION)
safe_to_csv(compare72b, OUT_COMPARE)

claims = pd.DataFrame([
    {
        "claim_topic": "GT source",
        "allowed_claim": "Manual GT v2 remains the source of truth.",
        "not_allowed_claim": "Model predictions or tracking outputs are GT.",
    },
    {
        "claim_topic": "Classifier status",
        "allowed_claim": "Frozen detector embedding baseline was evaluated on strict-gold crops.",
        "not_allowed_claim": "Production-grade behaviour classifier.",
    },
    {
        "claim_topic": "Champion model",
        "allowed_claim": "v72b frozen YOLOv8 detector embedding baseline is the final Week8 modeling champion.",
        "not_allowed_claim": "Fine-tuned deep model was trained.",
    },
    {
        "claim_topic": "Split policy",
        "allowed_claim": "Current split is exploratory; group-aware split is the conservative source-video-separated check.",
        "not_allowed_claim": "Current split alone proves source-video generalization.",
    },
    {
        "claim_topic": "Rare classes",
        "allowed_claim": "Rare classes remain limited and should be interpreted cautiously.",
        "not_allowed_claim": "Reliable standalone performance for BE, DE, IA.",
    },
    {
        "claim_topic": "v70a",
        "allowed_claim": "v70a was completed as a negative/ablation experiment.",
        "not_allowed_claim": "v70a outperformed v69b or v72b.",
    },
])

safe_to_csv(claims, OUT_CLAIMS)

limits = pd.DataFrame([
    {
        "item": "Dataset size",
        "limitation": "Only 372 strict-gold objects are used for classification baseline.",
        "next_step": "Add more manually verified windows/classes before strong classifier claims.",
    },
    {
        "item": "Class imbalance",
        "limitation": "STI is dominant; BE, DE, IA are rare.",
        "next_step": "Use macro-F1, per-class recall, and avoid rare-class standalone claims.",
    },
    {
        "item": "Temporal behavior",
        "limitation": "v72b uses crop-level frozen embeddings, not temporal sequence modeling.",
        "next_step": "Optional future work: clip-level temporal embedding or multi-frame model.",
    },
    {
        "item": "Generalization",
        "limitation": "Group-aware split is conservative but still small.",
        "next_step": "Validate on additional videos/days before deployment claims.",
    },
    {
        "item": "Model training",
        "limitation": "No fine-tuning was performed in this Week8 baseline.",
        "next_step": "Only consider fine-tuning after more labels and stronger validation protocol.",
    },
])

safe_to_csv(limits, OUT_LIMITS)

error_summary = pd.DataFrame([{
    "champion_prediction_rows": int(d72c.iloc[0].get("champion_prediction_rows", 0)),
    "error_rows": int(d72c.iloc[0].get("error_rows", 0)),
    "correct_rows": int(d72c.iloc[0].get("correct_rows", 0)),
    "current_errors": int(d72c.iloc[0].get("current_errors", 0)),
    "group_errors": int(d72c.iloc[0].get("group_errors", 0)),
    "top_confusion_rows": int(d72c.iloc[0].get("top_confusion_rows", 0)),
    "class_summary_rows": int(d72c.iloc[0].get("class_summary_rows", 0)),
    "interface_server": clean(d72c.iloc[0].get("server_path", "")),
    "interface_port": clean(d72c.iloc[0].get("default_port", "")),
}])

safe_to_csv(error_summary, OUT_ERROR_SUMMARY)

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

add_qa("readiness_checks_passed", 4, sum(1 for _, ok in readiness if ok), all(ok for _, ok in readiness), "hard", "v70b, v72a, v72b, v72c should be ready.")
add_qa("champion_rows", 2, len(champion), len(champion) == 2, "hard", "Expected one final champion per split policy.")
add_qa("current_champion_v72b", "v72b", champion[champion["split_policy"] == "current_recommended_split"]["final_champion_stage"].iloc[0], champion[champion["split_policy"] == "current_recommended_split"]["final_champion_stage"].iloc[0] == "v72b", "hard", "Current split champion should be v72b.")
add_qa("group_champion_v72b", "v72b", champion[champion["split_policy"] == "selected_group_aware_split"]["final_champion_stage"].iloc[0], champion[champion["split_policy"] == "selected_group_aware_split"]["final_champion_stage"].iloc[0] == "v72b", "hard", "Group-aware split champion should be v72b.")
add_qa("v72b_beats_v69b_all_macro_f1", True, bool(compare72b["v72b_beats_v69b_macro_f1"].astype(str).str.lower().eq("true").all()), bool(compare72b["v72b_beats_v69b_macro_f1"].astype(str).str.lower().eq("true").all()), "hard", "v72b should beat v69b macro-F1 for both split policies.")
add_qa("v72c_errors_documented", ">0", int(d72c.iloc[0].get("error_rows", 0)), int(d72c.iloc[0].get("error_rows", 0)) > 0, "info", "Errors are expected and documented.")
add_qa("claim_boundaries_created", ">0", len(claims), len(claims) > 0, "hard", "Claim boundaries should exist.")
add_qa("limitations_created", ">0", len(limits), len(limits) > 0, "hard", "Limitations and next steps should exist.")

qa = pd.DataFrame(qa_rows)
safe_to_csv(qa, OUT_QA)

hard_quality_failures = int(((qa["severity"] == "hard") & (~qa["passed"])).sum())

if hard_quality_failures:
    issues.append({
        "item": "v73_quality_checks",
        "issue_type": "hard_final_modeling_quality_failed",
        "issue_detail": f"{hard_quality_failures} hard checks failed.",
        "severity": "hard",
    })

issues.append({
    "item": "claim_scope",
    "issue_type": "info_final_modeling_package_only",
    "issue_detail": "v73 summarizes baseline modeling results. It does not train a new model.",
    "severity": "info",
})

issues_df = pd.DataFrame(issues, columns=["item", "issue_type", "issue_detail", "severity"])
safe_to_csv(issues_df, OUT_ISSUES)

hard_issue_count = int((issues_df["severity"] == "hard").sum()) if len(issues_df) else 0
warning_count = int((issues_df["severity"] == "warning").sum()) if len(issues_df) else 0
info_count = int((issues_df["severity"] == "info").sum()) if len(issues_df) else 0

cur = champion[champion["split_policy"] == "current_recommended_split"].iloc[0]
grp = champion[champion["split_policy"] == "selected_group_aware_split"].iloc[0]

report_text = f"""# Week 8 Final Modeling Package

## Final modeling decision

The final Week8 modeling champion is v72b: frozen YOLOv8 detector embedding baseline.

## Champion results

Current split:
- crop: {cur['crop_type']}
- config: {cur['selected_config']}
- test accuracy: {float(cur['test_accuracy']):.4f}
- test macro-F1: {float(cur['test_macro_f1']):.4f}
- delta macro-F1 vs v69b: {float(cur['delta_macro_f1_vs_v69b']):.4f}

Group-aware split:
- crop: {grp['crop_type']}
- config: {grp['selected_config']}
- test accuracy: {float(grp['test_accuracy']):.4f}
- test macro-F1: {float(grp['test_macro_f1']):.4f}
- delta macro-F1 vs v69b: {float(grp['delta_macro_f1_vs_v69b']):.4f}

## Claim boundary

This is a frozen embedding baseline, not a production behaviour classifier.
No GT is modified by model outputs.
Manual GT v2 remains the source of truth.

## Error inspection

v72c created a visual interface for manual error inspection.
"""

OUT_README.write_text(report_text)
OUT_REPORT.write_text(report_text)

manifest = {
    "version": "week8_v73_final_modeling_package",
    "created_at": datetime.now().isoformat(timespec="seconds"),
    "final_champion": "v72b_frozen_yolov8_detector_embedding_baseline",
    "champion_rows": champion.to_dict(orient="records"),
    "error_summary": error_summary.to_dict(orient="records"),
    "claim_boundary": "baseline modeling package only; no production classifier claim",
    "ready_for_v74": bool(hard_issue_count == 0),
}

OUT_MANIFEST.write_text(json.dumps(manifest, indent=2, ensure_ascii=False))

if OUT_ZIP.exists():
    OUT_ZIP.unlink()

with zipfile.ZipFile(OUT_ZIP, "w", compression=zipfile.ZIP_DEFLATED) as z:
    for p in sorted(PKG.rglob("*")):
        if p.is_file():
            z.write(p, p.relative_to(OUT))

zip_hash = sha256_file(OUT_ZIP)
OUT_SHA.write_text(f"{zip_hash}  {OUT_ZIP.name}\n")

decision = pd.DataFrame([{
    "v73_decision": "final_modeling_package_completed" if hard_issue_count == 0 else "final_modeling_package_has_blocking_issues",
    "final_champion_stage": "v72b",
    "final_champion_name": "frozen_yolov8_detector_embedding_baseline",
    "current_champion_crop": clean(cur["crop_type"]),
    "current_champion_config": clean(cur["selected_config"]),
    "current_champion_accuracy": float(cur["test_accuracy"]),
    "current_champion_macro_f1": float(cur["test_macro_f1"]),
    "current_delta_macro_f1_vs_v69b": float(cur["delta_macro_f1_vs_v69b"]),
    "group_champion_crop": clean(grp["crop_type"]),
    "group_champion_config": clean(grp["selected_config"]),
    "group_champion_accuracy": float(grp["test_accuracy"]),
    "group_champion_macro_f1": float(grp["test_macro_f1"]),
    "group_delta_macro_f1_vs_v69b": float(grp["delta_macro_f1_vs_v69b"]),
    "v72c_error_rows": int(d72c.iloc[0].get("error_rows", 0)),
    "v72c_correct_rows": int(d72c.iloc[0].get("correct_rows", 0)),
    "zip_path": str(OUT_ZIP),
    "zip_sha256": zip_hash,
    "hard_quality_failures": hard_quality_failures,
    "hard_issue_count": hard_issue_count,
    "warning_count": warning_count,
    "info_count": info_count,
    "issue_count": int(len(issues_df)),
    "ready_for_v74_final_week8_delivery_package": bool(hard_issue_count == 0),
    "claim_scope": "final_modeling_package_only_no_production_classifier_claim",
    "generated_at": datetime.now().isoformat(timespec="seconds"),
}])

safe_to_csv(decision, OUT_DECISION)

OUT_NOTE.write_text(
    "# Week 8 v73 Final Modeling Package\n\n"
    f"- v73 decision: {decision.iloc[0]['v73_decision']}\n"
    f"- Final champion: v72b frozen YOLOv8 detector embedding baseline\n"
    f"- Current split: {cur['crop_type']} / {cur['selected_config']} / macro-F1={float(cur['test_macro_f1']):.4f} / acc={float(cur['test_accuracy']):.4f}\n"
    f"- Group-aware split: {grp['crop_type']} / {grp['selected_config']} / macro-F1={float(grp['test_macro_f1']):.4f} / acc={float(grp['test_accuracy']):.4f}\n"
    f"- Current delta macro-F1 vs v69b: {float(cur['delta_macro_f1_vs_v69b']):.4f}\n"
    f"- Group delta macro-F1 vs v69b: {float(grp['delta_macro_f1_vs_v69b']):.4f}\n"
    f"- v72c errors: {int(d72c.iloc[0].get('error_rows', 0))}\n"
    f"- Hard issues: {hard_issue_count}\n"
    f"- Ready for v74 final Week8 delivery package: {bool(hard_issue_count == 0)}\n\n"
    "This is still a baseline/proof-of-concept modeling result, not a production classifier.\n"
)

progress_row = pd.DataFrame([{
    "date": datetime.now().date().isoformat(),
    "stage": "v73",
    "task_name": "Final modeling package",
    "status": "PASS" if hard_issue_count == 0 else "NEEDS_FIX",
    "input_summary": f"{V72B}; {V72C}",
    "output_summary": str(PKG),
    "hard_issues": hard_issue_count,
    "warnings": warning_count,
    "next_action": "Create v74 final Week8 delivery package." if hard_issue_count == 0 else "Fix v73 hard issues.",
}])

if OUT_PROGRESS.exists():
    old = pd.read_csv(OUT_PROGRESS)
    progress = pd.concat([old, progress_row], ignore_index=True)
else:
    progress = progress_row

safe_to_csv(progress, OUT_PROGRESS)

print("=== v73 decision ===")
print(decision.to_string(index=False))
print("\n=== final champion ===")
print(champion.to_string(index=False))
print("\n=== QA ===")
print(qa.to_string(index=False))
print("\n=== issues ===")
print(issues_df.to_string(index=False) if len(issues_df) else "No issues found.")
