from pathlib import Path
from datetime import datetime
import pandas as pd

F = Path.home() / "PigBench" / "Full_Unibo_Behaviour_Pipeline"
O = F / "outputs/v80_final_project_completion/04_frame_based_baseline"
O.mkdir(parents=True, exist_ok=True)

A_path = O / "v80f3_splitA_frame_based_baseline_metrics_summary.csv"
B_path = O / "v80f4_splitB_frame_based_baseline_metrics_summary.csv"
A_dec_path = O / "v80f3_decision_summary.csv"
B_dec_path = O / "v80f4_decision_summary.csv"

A = pd.read_csv(A_path).fillna("")
B = pd.read_csv(B_path).fillna("")
A_dec = pd.read_csv(A_dec_path).fillna("")
B_dec = pd.read_csv(B_dec_path).fillna("")

A["split_protocol"] = "Split A - grouped video-level"
B["split_protocol"] = "Split B - cross-camera/pen"

comparison = pd.concat([A, B], ignore_index=True)
comparison["aggregation"] = comparison["aggregation"].replace("", "frame_level_no_aggregation")
comparison.to_csv(O / "v80f5_frame_baseline_comparison_metrics.csv", index=False)

clip_test = comparison[
    (comparison["split"] == "test") & (comparison["level"] == "clip")
].copy()

best_by_protocol = (
    clip_test.sort_values(["split_protocol", "macro_f1", "weighted_f1"], ascending=[True, False, False])
    .groupby("split_protocol", as_index=False)
    .head(1)
    .reset_index(drop=True)
)

best_by_protocol.to_csv(O / "v80f5_best_test_clip_results_by_split.csv", index=False)

overall_best = best_by_protocol.sort_values(["macro_f1", "weighted_f1"], ascending=False).iloc[0]

issues = []

if len(best_by_protocol) < 2:
    issues.append({
        "item": "comparison",
        "issue_type": "missing_split_protocol_result",
        "severity": "hard",
        "detail": "expected both Split A and Split B test clip results",
    })

if float(overall_best["macro_f1"]) < 0.10:
    issues.append({
        "item": "frame_proxy_baseline",
        "issue_type": "low_macro_f1_expected_limitation",
        "severity": "warning",
        "detail": "best test clip macro F1 is below 0.10; bbox/track proxy features are weak for behaviour classification",
    })

if not issues:
    issues = [{
        "item": "none",
        "issue_type": "none",
        "severity": "info",
        "detail": "frame baseline comparison completed",
    }]

pd.DataFrame(issues).to_csv(O / "v80f5_issues.csv", index=False)

hard = sum(1 for x in issues if x["severity"] == "hard")
warning = sum(1 for x in issues if x["severity"] == "warning")

decision = pd.DataFrame([{
    "v80f5_decision": "frame_baseline_comparison_completed" if hard == 0 else "frame_baseline_comparison_has_blocking_issues",
    "split_A_best_test_aggregation": str(A_dec.iloc[0]["best_test_clip_aggregation"]),
    "split_A_best_test_macro_f1": float(A_dec.iloc[0]["best_test_clip_macro_f1"]),
    "split_A_best_test_accuracy": float(A_dec.iloc[0]["best_test_clip_accuracy"]),
    "split_B_best_test_aggregation": str(B_dec.iloc[0]["best_test_clip_aggregation"]),
    "split_B_best_test_macro_f1": float(B_dec.iloc[0]["best_test_clip_macro_f1"]),
    "split_B_best_test_accuracy": float(B_dec.iloc[0]["best_test_clip_accuracy"]),
    "overall_best_protocol": str(overall_best["split_protocol"]),
    "overall_best_aggregation": str(overall_best["aggregation"]),
    "overall_best_test_macro_f1": float(overall_best["macro_f1"]),
    "overall_best_test_accuracy": float(overall_best["accuracy"]),
    "hard_issue_count": hard,
    "warning_count": warning,
    "ready_for_v80g_clip_based_videomae": hard == 0,
    "claim_scope": "frame_proxy_baseline_comparison_not_rgb_cnn_not_production_classifier",
    "generated_at": datetime.now().isoformat(timespec="seconds"),
}])

decision.to_csv(O / "v80f5_decision_summary.csv", index=False)

report_path = O / "v80f5_frame_based_baseline_report.md"
report_path.write_text(
    "# v80f5 Frame-based Baseline Comparison Report\n\n"
    "## Scope\n\n"
    "This report compares a frame-proxy baseline on two split protocols. "
    "The model uses ROI-filtered bbox/track/frame-proxy features and logistic regression. "
    "It is not an RGB image CNN and not a production behaviour classifier.\n\n"
    "## Split protocols\n\n"
    "- Split A: grouped video-level split; no same video appears across train/val/test.\n"
    "- Split B: cross-camera/pen split; TLC6/M4 is held out as test.\n\n"
    "## Best test results\n\n"
    f"- Split A best aggregation: {decision.iloc[0]['split_A_best_test_aggregation']}\n"
    f"- Split A test macro F1: {decision.iloc[0]['split_A_best_test_macro_f1']:.6f}\n"
    f"- Split A test accuracy: {decision.iloc[0]['split_A_best_test_accuracy']:.6f}\n\n"
    f"- Split B best aggregation: {decision.iloc[0]['split_B_best_test_aggregation']}\n"
    f"- Split B test macro F1: {decision.iloc[0]['split_B_best_test_macro_f1']:.6f}\n"
    f"- Split B test accuracy: {decision.iloc[0]['split_B_best_test_accuracy']:.6f}\n\n"
    "## Interpretation\n\n"
    "The frame-proxy baseline performs poorly on both the grouped video split and the cross-camera split. "
    "This indicates that simple ROI-filtered bounding-box geometry, detection confidence, candidate rank, "
    "camera/pen metadata, and identity colour are not sufficient to classify pig behaviour reliably.\n\n"
    "The result is still useful as a baseline because it establishes that behaviour recognition requires "
    "visual-temporal information from the video frames, motivating the next clip-based model stage.\n\n"
    "## Next step\n\n"
    "Proceed to a clip-based action classifier using sampled frames from each 10-second behaviour clip. "
    "The planned model family is VideoMAE / video transformer, using the same validated split definitions "
    "so that frame-based and clip-based approaches can be compared fairly.\n\n",
    encoding="utf-8",
)

note = F / "notes/v80f5_frame_baseline_comparison_notes.md"
note.write_text(
    "# v80f5 Frame Baseline Comparison\n\n"
    f"- Decision: {decision.iloc[0]['v80f5_decision']}\n"
    f"- Split A best test macro F1: {decision.iloc[0]['split_A_best_test_macro_f1']:.6f}\n"
    f"- Split B best test macro F1: {decision.iloc[0]['split_B_best_test_macro_f1']:.6f}\n"
    f"- Overall best protocol: {decision.iloc[0]['overall_best_protocol']}\n"
    f"- Overall best aggregation: {decision.iloc[0]['overall_best_aggregation']}\n"
    f"- Hard issues: {hard}\n"
    f"- Warnings: {warning}\n"
    f"- Ready for v80g clip-based VideoMAE: {hard == 0}\n\n"
    "The frame-proxy baseline is weak, which supports moving to a visual-temporal clip-based classifier.\n",
    encoding="utf-8",
)

print(decision.to_string(index=False))
print("=== best by protocol ===")
print(best_by_protocol.to_string(index=False))
print("=== issues ===")
print(pd.DataFrame(issues).to_string(index=False))
print("report", report_path)
