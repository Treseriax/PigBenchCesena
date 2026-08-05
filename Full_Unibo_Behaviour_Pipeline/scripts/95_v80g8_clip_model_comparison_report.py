from pathlib import Path
from datetime import datetime
import pandas as pd

F = Path.home() / "PigBench" / "Full_Unibo_Behaviour_Pipeline"
G = F / "outputs/v80_final_project_completion/05_clip_based_videomae"
OUT = G / "v80g8_clip_model_comparison"
OUT.mkdir(parents=True, exist_ok=True)

frame_dec = pd.read_csv(
    F / "outputs/v80_final_project_completion/04_frame_based_baseline/v80f5_decision_summary.csv"
).fillna("")

visual_dec = pd.read_csv(
    G / "v80g5_limited_clip_visual_feature_baseline/v80g5_decision_summary.csv"
).fillna("")

videomae_dec = pd.read_csv(
    G / "v80g7_tiny_videomae_limited_training/v80g7b_decision_summary.csv"
).fillna("")

rows = []

rows.append({
    "model_family": "frame_proxy_baseline",
    "model_type": "logistic_regression_on_roi_bbox_track_features",
    "dataset_scope": "full_split_A_and_split_B_frame_proxy_dataset",
    "test_clips": "",
    "test_accuracy": float(frame_dec.iloc[0]["overall_best_test_accuracy"]),
    "test_macro_f1": float(frame_dec.iloc[0]["overall_best_test_macro_f1"]),
    "test_weighted_f1": "",
    "best_protocol_or_split": str(frame_dec.iloc[0]["overall_best_protocol"]),
    "aggregation": str(frame_dec.iloc[0]["overall_best_aggregation"]),
    "pretrained_weights_used": False,
    "claim_scope": "baseline_not_visual_temporal_not_production_classifier",
})

rows.append({
    "model_family": "limited_clip_visual_feature_baseline",
    "model_type": str(visual_dec.iloc[0]["model_type"]),
    "dataset_scope": "limited_73_clip_split_A_dataset",
    "test_clips": int(visual_dec.iloc[0]["test_clips"]),
    "test_accuracy": float(visual_dec.iloc[0]["test_accuracy"]),
    "test_macro_f1": float(visual_dec.iloc[0]["test_macro_f1"]),
    "test_weighted_f1": float(visual_dec.iloc[0]["test_weighted_f1"]),
    "best_protocol_or_split": "Split A limited test",
    "aggregation": "clip_level_direct_prediction",
    "pretrained_weights_used": False,
    "claim_scope": "limited_visual_temporal_feature_baseline_not_final_videomae",
})

rows.append({
    "model_family": "tiny_random_videomae",
    "model_type": str(videomae_dec.iloc[0]["model_type"]),
    "dataset_scope": "limited_73_clip_split_A_dataset",
    "test_clips": int(videomae_dec.iloc[0]["test_clips"]),
    "test_accuracy": float(videomae_dec.iloc[0]["test_accuracy"]),
    "test_macro_f1": float(videomae_dec.iloc[0]["test_macro_f1"]),
    "test_weighted_f1": float(videomae_dec.iloc[0]["test_weighted_f1"]),
    "best_protocol_or_split": "Split A limited test",
    "aggregation": "clip_level_direct_prediction",
    "pretrained_weights_used": bool(videomae_dec.iloc[0]["pretrained_weights_used"]),
    "claim_scope": "limited_tiny_random_videomae_not_pretrained_not_final",
})

comparison = pd.DataFrame(rows)
comparison = comparison.sort_values("test_macro_f1", ascending=False).reset_index(drop=True)
comparison.to_csv(OUT / "v80g8_clip_model_comparison_table.csv", index=False)

best = comparison.iloc[0]

issues = []
if float(best["test_macro_f1"]) < 0.15:
    issues.append({
        "item": "clip_models",
        "issue_type": "low_absolute_performance",
        "severity": "warning",
        "detail": "best clip model macro F1 remains low; report as limited experimental result, not final production classifier",
    })

if not issues:
    issues = [{
        "item": "none",
        "issue_type": "none",
        "severity": "info",
        "detail": "clip model comparison completed",
    }]

pd.DataFrame(issues).to_csv(OUT / "v80g8_issues.csv", index=False)

hard = sum(1 for x in issues if x["severity"] == "hard")
warning = sum(1 for x in issues if x["severity"] == "warning")

decision = pd.DataFrame([{
    "v80g8_decision": "clip_model_comparison_completed" if hard == 0 else "clip_model_comparison_has_blocking_issues",
    "best_model_family": str(best["model_family"]),
    "best_model_type": str(best["model_type"]),
    "best_test_macro_f1": round(float(best["test_macro_f1"]), 6),
    "best_test_accuracy": round(float(best["test_accuracy"]), 6),
    "hard_issue_count": hard,
    "warning_count": warning,
    "ready_for_v80h_evaluation_summary": hard == 0,
    "claim_scope": "experimental_model_comparison_not_final_production_classifier",
    "generated_at": datetime.now().isoformat(timespec="seconds"),
}])

decision.to_csv(OUT / "v80g8_decision_summary.csv", index=False)

report = OUT / "v80g8_clip_model_comparison_report.md"
report.write_text(
    "# v80g8 Clip Model Comparison Report\n\n"
    "## Scope\n\n"
    "This report compares the experimental behaviour-classification models produced in the final project pipeline. "
    "The comparison includes a frame-proxy baseline, a limited clip visual-temporal feature baseline, and a tiny random VideoMAE-compatible model.\n\n"
    "None of these models is claimed as a production classifier. The tiny VideoMAE model was trained from random initialization and does not use pretrained weights.\n\n"
    "## Best result\n\n"
    f"- Best model family: {decision.iloc[0]['best_model_family']}\n"
    f"- Best model type: {decision.iloc[0]['best_model_type']}\n"
    f"- Best test macro F1: {decision.iloc[0]['best_test_macro_f1']:.6f}\n"
    f"- Best test accuracy: {decision.iloc[0]['best_test_accuracy']:.6f}\n\n"
    "## Interpretation\n\n"
    "The frame-proxy baseline performed poorly, showing that bbox/track geometry alone is weak for behaviour recognition. "
    "The limited clip visual-temporal feature baseline improved over the frame-proxy baseline, showing that information from RGB video frames is useful. "
    "The tiny random VideoMAE stage validates that a VideoMAE-compatible video-transformer training pipeline can run on the extracted clip tensors, but the model is intentionally small, randomly initialized, and trained on only 73 clips.\n\n"
    "The correct final interpretation is that the dataset, extraction, split, evaluation, and model-training infrastructure are now working, while stronger behaviour-classification performance would require a larger extracted dataset, pretrained video model weights, more training clips, and finalized identity-to-track assignment.\n\n",
    encoding="utf-8",
)

note = F / "notes/v80g8_clip_model_comparison_notes.md"
note.write_text(
    "# v80g8 Clip Model Comparison\n\n"
    f"- Decision: {decision.iloc[0]['v80g8_decision']}\n"
    f"- Best model family: {decision.iloc[0]['best_model_family']}\n"
    f"- Best test macro F1: {decision.iloc[0]['best_test_macro_f1']:.6f}\n"
    f"- Best test accuracy: {decision.iloc[0]['best_test_accuracy']:.6f}\n"
    f"- Hard issues: {hard}\n"
    f"- Warnings: {warning}\n"
    f"- Ready for v80h evaluation summary: {hard == 0}\n",
    encoding="utf-8",
)

print(decision.to_string(index=False))
print("=== comparison ===")
print(comparison.to_string(index=False))
print("=== issues ===")
print(pd.DataFrame(issues).to_string(index=False))
