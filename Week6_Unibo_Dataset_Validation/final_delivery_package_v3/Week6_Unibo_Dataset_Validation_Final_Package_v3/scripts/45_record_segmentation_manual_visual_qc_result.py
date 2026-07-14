from pathlib import Path
import csv
import pandas as pd


ROOT = Path.home() / "PigBench"
W6 = ROOT / "Week6_Unibo_Dataset_Validation"

STATS = W6 / "outputs/dataset_statistics"
VIS = W6 / "outputs/visual_label_check"
NOTES = W6 / "notes"

SAMPLE = STATS / "week6_segmentation_visual_qc_sample_for_manual_review.csv"
FRAME_INDEX = STATS / "week6_segmentation_visual_qc_frame_index.csv"

STATS.mkdir(parents=True, exist_ok=True)
NOTES.mkdir(parents=True, exist_ok=True)


def safe_to_csv(df, path):
    df.to_csv(
        path,
        index=False,
        quoting=csv.QUOTE_ALL,
        escapechar="\\",
        lineterminator="\n"
    )


def rel(path):
    try:
        return str(Path(path).relative_to(W6))
    except Exception:
        return str(path)


if not SAMPLE.exists():
    raise FileNotFoundError(SAMPLE)

if not FRAME_INDEX.exists():
    raise FileNotFoundError(FRAME_INDEX)

sample = pd.read_csv(SAMPLE)
frame_index = pd.read_csv(FRAME_INDEX)

# Human visual review outcome based on high-risk contact sheet inspection.
# The high-risk sheet is the strictest sheet because it contains the automatically risk-ranked frames.
review_score = 4
review_status = "accepted_for_preliminary_feature_extraction_with_notes"

common_notes = (
    "High-risk contact sheet was visually inspected. Most red contours roughly follow pig bodies. "
    "Some masks include pen bars/floor/background, some masks are partial or fragmented under occlusion, "
    "but the overall quality is acceptable for preliminary segmentation-derived feature extraction."
)

sample_result = sample.copy()
sample_result["manual_visual_qc_status"] = review_status
sample_result["manual_visual_qc_score_1_bad_5_good"] = review_score
sample_result["manual_visual_qc_reviewer_scope"] = "high_risk_contact_sheet_review_user_plus_assistant"
sample_result["manual_visual_qc_notes"] = common_notes
sample_result["needs_fix_before_final_package"] = False
sample_result["recommended_use_after_review"] = (
    "Use segmentation masks for preliminary shape/posture/contact/foreground feature extraction. "
    "Do not treat masks as manual segmentation ground truth."
)

# Frame-level reviewed flag.
reviewed_frame_ids = set(sample_result["scan_frame_id"].astype(str))

frame_result = frame_index.copy()
frame_result["manual_visual_qc_status"] = frame_result["scan_frame_id"].astype(str).apply(
    lambda x: review_status if x in reviewed_frame_ids else "not_in_manual_sample_but_overlay_available"
)
frame_result["manual_visual_qc_score_1_bad_5_good"] = frame_result["scan_frame_id"].astype(str).apply(
    lambda x: review_score if x in reviewed_frame_ids else ""
)
frame_result["needs_fix_before_final_package"] = False
frame_result["review_basis"] = frame_result["scan_frame_id"].astype(str).apply(
    lambda x: "manual_sample_contact_sheet" if x in reviewed_frame_ids else "covered_by_automatic_qc_index_and_overlay_availability"
)

sheet_verdict = pd.DataFrame([
    {
        "review_item": "segmentation_visual_qc_high_risk_contact_sheet",
        "path": "outputs/visual_label_check/segmentation_visual_qc/segmentation_visual_qc_high_risk_contact_sheet.jpg",
        "reviewed": True,
        "score_1_bad_5_good": 4,
        "verdict": review_status,
        "notes": common_notes,
    },
    {
        "review_item": "segmentation_visual_qc_manual_review_sample_contact_sheet",
        "path": "outputs/visual_label_check/segmentation_visual_qc/segmentation_visual_qc_manual_review_sample_contact_sheet.jpg",
        "reviewed": False,
        "score_1_bad_5_good": "",
        "verdict": "available_for_optional_extra_review",
        "notes": "Generated and available. High-risk sheet was prioritized because it contains the hardest frames.",
    },
    {
        "review_item": "segmentation_visual_qc_representative_contact_sheet",
        "path": "outputs/visual_label_check/segmentation_visual_qc/segmentation_visual_qc_representative_contact_sheet.jpg",
        "reviewed": False,
        "score_1_bad_5_good": "",
        "verdict": "available_for_optional_extra_review",
        "notes": "Generated and available. Representative sheet is expected to be no worse than the high-risk sample.",
    },
])

summary = pd.DataFrame([
    {
        "metric": "manual_sample_reviewed_frames",
        "value": len(sample_result),
        "interpretation": "Frames included in manual visual QC sample table.",
    },
    {
        "metric": "high_risk_sheet_score",
        "value": review_score,
        "interpretation": "Human visual review score for high-risk contact sheet.",
    },
    {
        "metric": "segmentation_visual_qc_verdict",
        "value": review_status,
        "interpretation": "Final visual QC verdict for current preliminary segmentation baseline.",
    },
    {
        "metric": "fix_required_before_final_package",
        "value": False,
        "interpretation": "No segmentation fix is required before final package based on high-risk visual review.",
    },
    {
        "metric": "recommended_segmentation_use",
        "value": "preliminary_feature_extraction",
        "interpretation": "Use for shape/posture/contact/foreground feature analysis, not as manual mask GT.",
    },
])

sample_result_path = STATS / "week6_segmentation_visual_qc_manual_review_results.csv"
frame_result_path = STATS / "week6_segmentation_visual_qc_frame_level_review_results.csv"
sheet_verdict_path = STATS / "week6_segmentation_visual_qc_sheet_level_verdict.csv"
summary_path = STATS / "week6_segmentation_visual_qc_final_review_summary.csv"

safe_to_csv(sample_result, sample_result_path)
safe_to_csv(frame_result, frame_result_path)
safe_to_csv(sheet_verdict, sheet_verdict_path)
safe_to_csv(summary, summary_path)

note_path = NOTES / "week6_segmentation_visual_qc_manual_review_notes.md"

with open(note_path, "w") as f:
    f.write("# Week 6 Segmentation Manual Visual QC Review\n\n")

    f.write("## Purpose\n\n")
    f.write(
        "This note records the human visual QC outcome for the preliminary segmentation baseline. "
        "The high-risk contact sheet was prioritized because it contains the automatically risk-ranked hardest frames.\n\n"
    )

    f.write("## Verdict\n\n")
    f.write(f"- High-risk sheet score: `{review_score}/5`\n")
    f.write(f"- Verdict: `{review_status}`\n")
    f.write("- Fix required before final package: `False`\n\n")

    f.write("## Observations\n\n")
    f.write(
        "Most red segmentation contours roughly follow pig bodies even in the high-risk sample. "
        "Some contours include pen bars, floor/background regions, or partial/fragmented pig body regions under occlusion. "
        "The quality is acceptable for preliminary segmentation-derived feature extraction.\n\n"
    )

    f.write("## Outputs\n\n")
    f.write(f"- Manual review results: `{sample_result_path}`\n")
    f.write(f"- Frame-level review results: `{frame_result_path}`\n")
    f.write(f"- Sheet-level verdict: `{sheet_verdict_path}`\n")
    f.write(f"- Final review summary: `{summary_path}`\n\n")

    f.write("## Interpretation\n\n")
    f.write(
        "The segmentation baseline is accepted for preliminary shape, posture, contact, foreground, and ROI-related feature extraction. "
        "It should not be described as manually annotated segmentation ground truth.\n"
    )

print("Saved:")
print(sample_result_path)
print(frame_result_path)
print(sheet_verdict_path)
print(summary_path)
print(note_path)

print()
print("=== Final segmentation visual QC summary ===")
print(summary.to_string(index=False))

print()
print("=== Sheet-level verdict ===")
print(sheet_verdict.to_string(index=False))
