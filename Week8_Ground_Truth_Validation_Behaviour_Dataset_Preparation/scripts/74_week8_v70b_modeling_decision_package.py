from pathlib import Path
from datetime import datetime
import csv
import json
import hashlib
import zipfile
import pandas as pd


ROOT = Path.home() / "PigBench"
W8 = ROOT / "Week8_Ground_Truth_Validation_Behaviour_Dataset_Preparation"

V69A = W8 / "outputs" / "v69a_baseline_sanity"
V69B = W8 / "outputs" / "v69b_feature_baseline"
V69C = W8 / "outputs" / "v69c_error_analysis"
V69D = W8 / "outputs" / "v69d_error_analysis_interface"
V70A = W8 / "outputs" / "v70a_improved_feature_baseline"

V69A_DECISION = V69A / "week8_v69a_decision_summary.csv"
V69A_METRICS = V69A / "Week8_Baseline_Sanity" / "week8_v69a_baseline_sanity_metrics.csv"

V69B_DECISION = V69B / "week8_v69b_decision_summary.csv"
V69B_BEST = V69B / "Week8_Feature_Baseline" / "week8_v69b_best_model_summary.csv"
V69B_ISSUES = V69B / "week8_v69b_issues.csv"

V69C_DECISION = V69C / "week8_v69c_decision_summary.csv"
V69C_TOP_CONF = V69C / "Week8_Feature_Baseline_Error_Analysis" / "week8_v69c_top_confusions.csv"
V69C_CLASS = V69C / "Week8_Feature_Baseline_Error_Analysis" / "week8_v69c_per_class_error_summary.csv"

V69D_DECISION = V69D / "week8_v69d_decision_summary.csv"

V70A_DECISION = V70A / "week8_v70a_decision_summary.csv"
V70A_BEST = V70A / "Week8_Improved_Feature_Baseline" / "week8_v70a_best_test_summary.csv"
V70A_COMPARE = V70A / "Week8_Improved_Feature_Baseline" / "week8_v70a_vs_v69b_comparison.csv"
V70A_ISSUES = V70A / "week8_v70a_issues.csv"

OUT = W8 / "outputs" / "v70b_modeling_decision_package"
PKG = OUT / "Week8_Modeling_Decision_Package"
NOTES = W8 / "notes"
REPORTS = W8 / "reports"
PROGRESS = W8 / "progress"

for p in [OUT, PKG, NOTES, REPORTS, PROGRESS]:
    p.mkdir(parents=True, exist_ok=True)

OUT_EXPERIMENTS = PKG / "week8_v70b_experiment_comparison.csv"
OUT_CHAMPION = PKG / "week8_v70b_champion_baseline_selection.csv"
OUT_CLAIMS = PKG / "week8_v70b_claim_boundaries.csv"
OUT_NEXT = PKG / "week8_v70b_next_step_recommendations.csv"
OUT_QA = PKG / "week8_v70b_quality_checks.csv"
OUT_README = PKG / "README_Week8_Modeling_Decision_Package.md"
OUT_MANIFEST = PKG / "week8_v70b_manifest.json"

OUT_DECISION = OUT / "week8_v70b_decision_summary.csv"
OUT_ISSUES = OUT / "week8_v70b_issues.csv"
OUT_ZIP = OUT / "Week8_Modeling_Decision_Package.zip"
OUT_SHA = OUT / "Week8_Modeling_Decision_Package.sha256"
OUT_NOTE = NOTES / "week8_v70b_modeling_decision_package_notes.md"
OUT_REPORT = REPORTS / "week8_v70b_modeling_decision_package_report.md"
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


def sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def fnum(x, default=0.0):
    try:
        return float(x)
    except Exception:
        return default


def get_sanity_macro(sanity, policy, split, baseline):
    r = sanity[
        (sanity["split_policy"] == policy)
        & (sanity["eval_split"] == split)
        & (sanity["baseline_type"] == baseline)
    ]
    if len(r) == 0:
        return ""
    return float(r.iloc[0]["macro_f1"])


issues = []

required = [
    V69A_DECISION, V69A_METRICS,
    V69B_DECISION, V69B_BEST, V69B_ISSUES,
    V69C_DECISION, V69C_TOP_CONF, V69C_CLASS,
    V69D_DECISION,
    V70A_DECISION, V70A_BEST, V70A_COMPARE, V70A_ISSUES,
]

for p in required:
    if not p.exists():
        issues.append({
            "item": str(p),
            "issue_type": "hard_missing_required_input",
            "issue_detail": "Required modeling-stage input missing.",
            "severity": "hard",
        })

if issues:
    issues_df = pd.DataFrame(issues)
    safe_to_csv(issues_df, OUT_ISSUES)
    decision = pd.DataFrame([{
        "v70b_decision": "modeling_decision_package_blocked",
        "hard_issue_count": int((issues_df["severity"] == "hard").sum()),
        "ready_for_v71_pretrained_embedding_audit": False,
        "generated_at": datetime.now().isoformat(timespec="seconds"),
    }])
    safe_to_csv(decision, OUT_DECISION)
    print(decision.to_string(index=False))
    raise SystemExit(1)


d69a = read_csv_clean(V69A_DECISION)
d69b = read_csv_clean(V69B_DECISION)
d69c = read_csv_clean(V69C_DECISION)
d69d = read_csv_clean(V69D_DECISION)
d70a = read_csv_clean(V70A_DECISION)

sanity = read_csv_clean(V69A_METRICS)
best69b = read_csv_clean(V69B_BEST)
best70a = read_csv_clean(V70A_BEST)
compare70a = read_csv_clean(V70A_COMPARE)
conf69c = read_csv_clean(V69C_TOP_CONF)
class69c = read_csv_clean(V69C_CLASS)

readiness_checks = [
    ("v69a_ready", bool_true(d69a.iloc[0].get("ready_for_v69b_feature_baseline", ""))),
    ("v69b_ready", bool_true(d69b.iloc[0].get("ready_for_v69c_error_analysis", ""))),
    ("v69c_ready", bool_true(d69c.iloc[0].get("ready_for_v70a_improved_baseline", ""))),
    ("v69d_ready", bool_true(d69d.iloc[0].get("ready_for_v70a_improved_baseline", ""))),
    ("v70a_ready", bool_true(d70a.iloc[0].get("ready_for_v70b_error_interface", ""))),
]

for name, ok in readiness_checks:
    if not ok:
        issues.append({
            "item": name,
            "issue_type": "hard_stage_not_ready",
            "issue_detail": f"{name} is not ready.",
            "severity": "hard",
        })

experiment_rows = []

for _, r in best69b.iterrows():
    policy = clean(r["split_policy"])
    experiment_rows.append({
        "experiment_stage": "v69b",
        "experiment_name": "simple_image_feature_baseline",
        "split_policy": policy,
        "crop_type": clean(r["crop_type"]),
        "model_type": clean(r["model_type"]),
        "selection_policy": "best_test_in_v69b_summary",
        "test_accuracy": fnum(r["accuracy"]),
        "test_macro_f1": fnum(r["macro_f1"]),
        "test_balanced_accuracy": fnum(r["balanced_accuracy"]),
        "beats_majority_macro_f1": clean(r.get("beats_majority_macro_f1", "")),
        "beats_random_macro_f1": clean(r.get("beats_random_macro_f1", "")),
        "claim_scope": "simple_feature_baseline",
    })

for _, r in best70a.iterrows():
    policy = clean(r["split_policy"])
    experiment_rows.append({
        "experiment_stage": "v70a",
        "experiment_name": "class_balanced_ridge_with_pca",
        "split_policy": policy,
        "crop_type": clean(r["crop_type"]),
        "model_type": clean(r["model_type"]),
        "selection_policy": "validation_macro_f1_selected",
        "test_accuracy": fnum(r["accuracy"]),
        "test_macro_f1": fnum(r["macro_f1"]),
        "test_balanced_accuracy": fnum(r["balanced_accuracy"]),
        "beats_majority_macro_f1": "",
        "beats_random_macro_f1": "",
        "claim_scope": "improved_classical_feature_baseline_ablation",
    })

experiments = pd.DataFrame(experiment_rows)
safe_to_csv(experiments, OUT_EXPERIMENTS)

champion_rows = []

for policy in sorted(experiments["split_policy"].unique()):
    sub = experiments[experiments["split_policy"] == policy].copy()
    sub = sub.sort_values(["test_macro_f1", "test_balanced_accuracy", "test_accuracy"], ascending=[False, False, False])
    champ = sub.iloc[0]

    majority_macro = get_sanity_macro(sanity, policy, "test", "majority_train_class")
    random_macro = get_sanity_macro(sanity, policy, "test", "train_prior_random_mean")

    champion_rows.append({
        "split_policy": policy,
        "champion_stage": clean(champ["experiment_stage"]),
        "champion_experiment": clean(champ["experiment_name"]),
        "champion_crop_type": clean(champ["crop_type"]),
        "champion_model_type": clean(champ["model_type"]),
        "champion_test_accuracy": fnum(champ["test_accuracy"]),
        "champion_test_macro_f1": fnum(champ["test_macro_f1"]),
        "champion_test_balanced_accuracy": fnum(champ["test_balanced_accuracy"]),
        "sanity_majority_test_macro_f1": majority_macro,
        "sanity_random_test_macro_f1": random_macro,
        "delta_vs_majority_macro_f1": fnum(champ["test_macro_f1"]) - fnum(majority_macro),
        "delta_vs_random_macro_f1": fnum(champ["test_macro_f1"]) - fnum(random_macro),
        "champion_decision": "selected_for_reporting",
        "reason": "highest test macro-F1 among completed non-deep baselines",
    })

champion = pd.DataFrame(champion_rows)
safe_to_csv(champion, OUT_CHAMPION)

v70a_negative = False
for _, r in compare70a.iterrows():
    if clean(r.get("v70a_beats_v69b_macro_f1", "")).lower() != "true":
        v70a_negative = True

claim_rows = [
    {
        "claim_topic": "GT source",
        "allowed_claim": "Manual GT v2 is source of truth.",
        "not_allowed_claim": "Tracking or model prediction is source of truth.",
    },
    {
        "claim_topic": "Classification",
        "allowed_claim": "Baseline/proof-of-concept classification experiments were run on strict-gold crops.",
        "not_allowed_claim": "Production-grade behaviour classifier.",
    },
    {
        "claim_topic": "Champion baseline",
        "allowed_claim": "v69b simple feature baseline is the current champion among completed classical baselines.",
        "not_allowed_claim": "v70a improved classifier outperformed v69b.",
    },
    {
        "claim_topic": "Split policy",
        "allowed_claim": "Current split is exploratory; group-aware split is conservative and source-video-overlap-free.",
        "not_allowed_claim": "Current split proves source-video generalization.",
    },
    {
        "claim_topic": "Rare classes",
        "allowed_claim": "BE, DE, IA are rare and should not receive strong standalone claims.",
        "not_allowed_claim": "Reliable per-class performance for rare classes.",
    },
]
claims = pd.DataFrame(claim_rows)
safe_to_csv(claims, OUT_CLAIMS)

next_rows = [
    {
        "next_stage": "v71",
        "task": "pretrained/deep embedding feasibility audit",
        "reason": "Before trying stronger visual baselines, check whether usable local pretrained weights/features exist without internet-dependent downloads.",
        "priority": "high",
    },
    {
        "next_stage": "v72",
        "task": "frozen embedding baseline if v71 passes",
        "reason": "A frozen CNN/YOLO embedding baseline may outperform handcrafted features without risky fine-tuning.",
        "priority": "high",
    },
    {
        "next_stage": "v73",
        "task": "final modeling report package",
        "reason": "Consolidate GT, crop dataset, split policies, baselines, errors, and limitations.",
        "priority": "medium",
    },
    {
        "next_stage": "optional",
        "task": "deep fine-tuning",
        "reason": "Only after frozen embedding baseline; dataset is small and imbalanced.",
        "priority": "low",
    },
]
next_steps = pd.DataFrame(next_rows)
safe_to_csv(next_steps, OUT_NEXT)

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

add_qa("readiness_checks_passed", 5, sum(1 for _, ok in readiness_checks if ok), all(ok for _, ok in readiness_checks), "hard", "v69a-v70a stages should be ready.")
add_qa("experiment_rows", 4, len(experiments), len(experiments) == 4, "hard", "Expected v69b/v70a results for two split policies.")
add_qa("champion_rows", 2, len(champion), len(champion) == 2, "hard", "Expected one champion per split policy.")
add_qa("champion_is_v69b_current", "v69b", champion[champion["split_policy"] == "current_recommended_split"]["champion_stage"].iloc[0], champion[champion["split_policy"] == "current_recommended_split"]["champion_stage"].iloc[0] == "v69b", "hard", "Current split champion should remain v69b.")
add_qa("champion_is_v69b_group", "v69b", champion[champion["split_policy"] == "selected_group_aware_split"]["champion_stage"].iloc[0], champion[champion["split_policy"] == "selected_group_aware_split"]["champion_stage"].iloc[0] == "v69b", "hard", "Group-aware split champion should remain v69b.")
add_qa("v70a_negative_result_documented", True, v70a_negative, v70a_negative, "info", "v70a did not beat v69b and should be documented as negative/ablation experiment.")
add_qa("top_confusions_available", ">0", len(conf69c), len(conf69c) > 0, "hard", "v69c top confusions should be available.")
add_qa("class_summary_available", ">0", len(class69c), len(class69c) > 0, "hard", "v69c class summary should be available.")

qa = pd.DataFrame(qa_rows)
safe_to_csv(qa, OUT_QA)

hard_quality_failures = int(((qa["severity"] == "hard") & (~qa["passed"])).sum())

if hard_quality_failures:
    issues.append({
        "item": "v70b_quality_checks",
        "issue_type": "hard_modeling_decision_quality_check_failed",
        "issue_detail": f"{hard_quality_failures} hard checks failed.",
        "severity": "hard",
    })

if v70a_negative:
    issues.append({
        "item": "v70a",
        "issue_type": "info_negative_ablation_result",
        "issue_detail": "v70a did not beat v69b on test macro-F1; keep v69b as champion.",
        "severity": "info",
    })

issues.append({
    "item": "claim_scope",
    "issue_type": "info_baseline_only",
    "issue_detail": "Current modeling results are baseline/proof-of-concept only.",
    "severity": "info",
})

issues_df = pd.DataFrame(issues, columns=["item", "issue_type", "issue_detail", "severity"])
safe_to_csv(issues_df, OUT_ISSUES)

hard_issue_count = int((issues_df["severity"] == "hard").sum()) if len(issues_df) else 0
warning_count = int((issues_df["severity"] == "warning").sum()) if len(issues_df) else 0
info_count = int((issues_df["severity"] == "info").sum()) if len(issues_df) else 0

manifest = {
    "version": "week8_v70b_modeling_decision_package",
    "created_at": datetime.now().isoformat(timespec="seconds"),
    "champions": champion.to_dict(orient="records"),
    "v70a_negative_ablation": bool(v70a_negative),
    "claim_boundary": "baseline/proof-of-concept only; champion is not production classifier",
    "next_recommended_stage": "v71_pretrained_embedding_feasibility_audit",
}
OUT_MANIFEST.write_text(json.dumps(manifest, indent=2, ensure_ascii=False))

champ_text = champion.to_string(index=False)
exp_text = experiments.to_string(index=False)
claims_text = claims.to_string(index=False)
next_text = next_steps.to_string(index=False)

OUT_README.write_text(
    "# Week8 v70b Modeling Decision Package\n\n"
    "## Decision\n\n"
    "The v69b simple image feature baseline remains the champion among completed non-deep baselines.\n\n"
    "v70a class-balanced ridge + PCA was completed but did not improve test macro-F1 over v69b, so it is documented as a negative/ablation experiment.\n\n"
    "## Champions\n\n"
    f"{champ_text}\n\n"
    "## Experiment Comparison\n\n"
    f"{exp_text}\n\n"
    "## Claim Boundaries\n\n"
    f"{claims_text}\n\n"
    "## Next Steps\n\n"
    f"{next_text}\n"
)

if OUT_ZIP.exists():
    OUT_ZIP.unlink()

with zipfile.ZipFile(OUT_ZIP, "w", compression=zipfile.ZIP_DEFLATED) as z:
    for p in sorted(PKG.rglob("*")):
        if p.is_file():
            z.write(p, p.relative_to(OUT))

zip_hash = sha256_file(OUT_ZIP)
OUT_SHA.write_text(f"{zip_hash}  {OUT_ZIP.name}\n")

cur = champion[champion["split_policy"] == "current_recommended_split"].iloc[0]
grp = champion[champion["split_policy"] == "selected_group_aware_split"].iloc[0]

decision = pd.DataFrame([{
    "v70b_decision": "modeling_decision_package_completed" if hard_issue_count == 0 else "modeling_decision_package_has_blocking_issues",
    "current_champion_stage": clean(cur["champion_stage"]),
    "current_champion_model": clean(cur["champion_model_type"]),
    "current_champion_crop": clean(cur["champion_crop_type"]),
    "current_champion_macro_f1": fnum(cur["champion_test_macro_f1"]),
    "current_champion_accuracy": fnum(cur["champion_test_accuracy"]),
    "group_champion_stage": clean(grp["champion_stage"]),
    "group_champion_model": clean(grp["champion_model_type"]),
    "group_champion_crop": clean(grp["champion_crop_type"]),
    "group_champion_macro_f1": fnum(grp["champion_test_macro_f1"]),
    "group_champion_accuracy": fnum(grp["champion_test_accuracy"]),
    "v70a_negative_ablation_documented": bool(v70a_negative),
    "zip_path": str(OUT_ZIP),
    "zip_sha256": zip_hash,
    "hard_quality_failures": hard_quality_failures,
    "hard_issue_count": hard_issue_count,
    "warning_count": warning_count,
    "info_count": info_count,
    "issue_count": int(len(issues_df)),
    "ready_for_v71_pretrained_embedding_audit": bool(hard_issue_count == 0),
    "claim_scope": "baseline_modeling_decision_only_no_production_classifier_claim",
    "generated_at": datetime.now().isoformat(timespec="seconds"),
}])

safe_to_csv(decision, OUT_DECISION)

OUT_NOTE.write_text(
    "# Week 8 v70b Modeling Decision Package\n\n"
    f"- v70b decision: {decision.iloc[0]['v70b_decision']}\n"
    f"- Current split champion: {cur['champion_stage']} / {cur['champion_crop_type']} / {cur['champion_model_type']} / macro-F1={float(cur['champion_test_macro_f1']):.4f}\n"
    f"- Group-aware champion: {grp['champion_stage']} / {grp['champion_crop_type']} / {grp['champion_model_type']} / macro-F1={float(grp['champion_test_macro_f1']):.4f}\n"
    f"- v70a negative ablation documented: {bool(v70a_negative)}\n"
    f"- Hard issues: {hard_issue_count}\n"
    f"- Ready for v71 pretrained embedding audit: {bool(hard_issue_count == 0)}\n\n"
    "Conclusion: keep v69b as champion baseline. v70a is documented as a completed but non-improving ablation.\n"
)

OUT_REPORT.write_text(
    "# Week 8 v70b Modeling Decision Package Report\n\n"
    f"Decision: {decision.iloc[0]['v70b_decision']}\n\n"
    f"Current champion macro-F1: {float(cur['champion_test_macro_f1'])}\n\n"
    f"Group-aware champion macro-F1: {float(grp['champion_test_macro_f1'])}\n\n"
    f"ZIP: {OUT_ZIP}\n\n"
    f"SHA256: {zip_hash}\n"
)

progress_row = pd.DataFrame([{
    "date": datetime.now().date().isoformat(),
    "stage": "v70b",
    "task_name": "Modeling decision package",
    "status": "PASS" if hard_issue_count == 0 else "NEEDS_FIX",
    "input_summary": f"{V69B}; {V70A}",
    "output_summary": str(PKG),
    "hard_issues": hard_issue_count,
    "warnings": warning_count,
    "next_action": "Run v71 pretrained/deep embedding feasibility audit." if hard_issue_count == 0 else "Fix v70b hard issues.",
}])

if OUT_PROGRESS.exists():
    old = pd.read_csv(OUT_PROGRESS)
    progress = pd.concat([old, progress_row], ignore_index=True)
else:
    progress = progress_row

safe_to_csv(progress, OUT_PROGRESS)

print("=== v70b decision ===")
print(decision.to_string(index=False))
print("\n=== champion selection ===")
print(champion.to_string(index=False))
print("\n=== issues ===")
print(issues_df.to_string(index=False) if len(issues_df) else "No issues found.")
