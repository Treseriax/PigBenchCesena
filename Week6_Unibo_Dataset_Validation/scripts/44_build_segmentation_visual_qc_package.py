from pathlib import Path
import csv
import math
import html
import pandas as pd
import numpy as np

try:
    import cv2
    CV2_AVAILABLE = True
except Exception:
    CV2_AVAILABLE = False


ROOT = Path.home() / "PigBench"
W6 = ROOT / "Week6_Unibo_Dataset_Validation"

FEAT = W6 / "outputs/feature_extractors"
GT = W6 / "outputs/unified_ground_truth"
STATS = W6 / "outputs/dataset_statistics"
VIS = W6 / "outputs/visual_label_check"
NOTES = W6 / "notes"

SEG_FEATURES = FEAT / "week6_preliminary_bbox_guided_segmentation_features.csv"
SEG_FRAME_SUMMARY = FEAT / "week6_preliminary_segmentation_frame_summary.csv"
FRAME_INDEX = GT / "week6_scanpoint_frame_index.csv"
SEG_OVERLAY_DIR = VIS / "preliminary_segmentation_baseline"

QC_DIR = VIS / "segmentation_visual_qc"
QC_DIR.mkdir(parents=True, exist_ok=True)
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


def make_contact_sheet(rows, out_path, title, thumb_w=420, max_cols=3):
    if not CV2_AVAILABLE or len(rows) == 0:
        return False

    thumbs = []

    for _, r in rows.iterrows():
        p = Path(str(r["segmentation_overlay_path"]))

        if not p.exists():
            continue

        img = cv2.imread(str(p))

        if img is None:
            continue

        h, w = img.shape[:2]
        scale = thumb_w / max(w, 1)
        thumb_h = int(h * scale)
        thumb = cv2.resize(img, (thumb_w, thumb_h))

        label_lines = [
            str(r["scan_frame_id"]),
            f"risk={r['visual_qc_risk_score']}",
            f"bbox={r['segmented_bbox_count']}, fallback={r['fallback_count']}",
        ]

        label_h = 72
        canvas = np.full((thumb_h + label_h, thumb_w, 3), 255, dtype=np.uint8)
        canvas[label_h:label_h+thumb_h, :, :] = thumb

        y = 18
        for line in label_lines:
            cv2.putText(
                canvas,
                line,
                (8, y),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (0, 0, 0),
                1,
                cv2.LINE_AA,
            )
            y += 20

        thumbs.append(canvas)

    if not thumbs:
        return False

    cols = min(max_cols, len(thumbs))
    rows_n = int(math.ceil(len(thumbs) / cols))
    cell_h = max(t.shape[0] for t in thumbs)

    title_h = 48
    sheet = np.full((title_h + rows_n * cell_h, cols * thumb_w, 3), 255, dtype=np.uint8)

    cv2.putText(
        sheet,
        title,
        (10, 30),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (0, 0, 0),
        2,
        cv2.LINE_AA,
    )

    for i, thumb in enumerate(thumbs):
        rr = i // cols
        cc = i % cols
        y = title_h + rr * cell_h
        x = cc * thumb_w
        sheet[y:y+thumb.shape[0], x:x+thumb.shape[1]] = thumb

    cv2.imwrite(str(out_path), sheet)
    return True


if not SEG_FEATURES.exists():
    raise FileNotFoundError(SEG_FEATURES)

if not SEG_FRAME_SUMMARY.exists():
    raise FileNotFoundError(SEG_FRAME_SUMMARY)

seg = pd.read_csv(SEG_FEATURES)
frame = pd.read_csv(SEG_FRAME_SUMMARY)
frames = pd.read_csv(FRAME_INDEX) if FRAME_INDEX.exists() else pd.DataFrame()

# Status counts per frame.
status_counts = (
    seg.groupby(["scan_frame_id", "segmentation_status"])
    .size()
    .reset_index(name="count")
    .pivot(index="scan_frame_id", columns="segmentation_status", values="count")
    .fillna(0)
    .reset_index()
)

# Per-frame extra quality metrics.
extra = (
    seg.groupby("scan_frame_id")
    .agg(
        bbox_count=("det_id", "count"),
        mean_mask_area_fraction=("mask_area_fraction_of_bbox", "mean"),
        std_mask_area_fraction=("mask_area_fraction_of_bbox", "std"),
        min_mask_area_fraction=("mask_area_fraction_of_bbox", "min"),
        max_mask_area_fraction=("mask_area_fraction_of_bbox", "max"),
        mean_shape_solidity=("shape_solidity", "mean"),
        min_shape_solidity=("shape_solidity", "min"),
        mean_shape_extent=("shape_extent", "mean"),
        min_shape_extent=("shape_extent", "min"),
        mean_mask_aspect_ratio=("mask_aspect_ratio", "mean"),
        max_mask_aspect_ratio=("mask_aspect_ratio", "max"),
    )
    .reset_index()
)

qc = frame.merge(extra, on="scan_frame_id", how="left")
qc = qc.merge(status_counts, on="scan_frame_id", how="left")

for c in ["grabcut_success", "otsu_fallback_after_grabcut"]:
    if c not in qc.columns:
        qc[c] = 0

qc["fallback_count"] = qc["otsu_fallback_after_grabcut"].fillna(0)
qc["grabcut_count"] = qc["grabcut_success"].fillna(0)

# Add frame metadata.
if len(frames):
    keep = ["scan_frame_id"]
    for c in ["timestamp", "video_id", "video_match_status", "frame_image_path"]:
        if c in frames.columns:
            keep.append(c)

    keep = list(dict.fromkeys(keep))
    qc = qc.merge(frames[keep], on="scan_frame_id", how="left", suffixes=("", "_from_frame_index"))

# Overlay paths.
overlay_paths = []

for sid in qc["scan_frame_id"]:
    p = SEG_OVERLAY_DIR / f"{sid}_preliminary_segmentation_overlay.jpg"
    overlay_paths.append(str(p))

qc["segmentation_overlay_path"] = overlay_paths
qc["segmentation_overlay_exists"] = qc["segmentation_overlay_path"].apply(lambda p: Path(str(p)).exists())

# Risk score.
risk = []

for _, r in qc.iterrows():
    score = 0
    reasons = []

    bbox_count = r.get("bbox_count", np.nan)
    fallback_count = r.get("fallback_count", 0)
    mean_frac = r.get("mean_mask_area_fraction", np.nan)
    min_frac = r.get("min_mask_area_fraction", np.nan)
    max_frac = r.get("max_mask_area_fraction", np.nan)
    min_solidity = r.get("min_shape_solidity", np.nan)
    max_aspect = r.get("max_mask_aspect_ratio", np.nan)

    if pd.notna(bbox_count) and (bbox_count < 6 or bbox_count > 10):
        score += 2
        reasons.append("unusual_bbox_count")

    if pd.notna(fallback_count) and fallback_count > 0:
        score += 1
        reasons.append("has_otsu_fallback")

    if pd.notna(fallback_count) and pd.notna(bbox_count) and bbox_count > 0 and fallback_count / bbox_count >= 0.5:
        score += 2
        reasons.append("fallback_majority")

    if pd.notna(min_frac) and min_frac < 0.10:
        score += 2
        reasons.append("very_small_mask_fraction")

    if pd.notna(max_frac) and max_frac > 0.95:
        score += 2
        reasons.append("very_large_mask_fraction")

    if pd.notna(min_solidity) and min_solidity < 0.35:
        score += 1
        reasons.append("low_solidity")

    if pd.notna(max_aspect) and max_aspect > 4.0:
        score += 1
        reasons.append("extreme_aspect_ratio")

    if not bool(r.get("segmentation_overlay_exists", False)):
        score += 5
        reasons.append("missing_overlay")

    risk.append((score, ";".join(reasons) if reasons else "low_risk"))

qc["visual_qc_risk_score"] = [x[0] for x in risk]
qc["visual_qc_risk_reasons"] = [x[1] for x in risk]
qc["visual_qc_priority"] = pd.cut(
    qc["visual_qc_risk_score"],
    bins=[-1, 0, 2, 4, 999],
    labels=["low", "medium", "high", "critical"],
)

# Sort high risk first.
qc_sorted = qc.sort_values(
    ["visual_qc_risk_score", "scan_frame_id"],
    ascending=[False, True],
).reset_index(drop=True)

# High-risk and representative sample.
high_risk = qc_sorted[qc_sorted["visual_qc_risk_score"] >= 3].copy()

# If too many, keep top 24; if too few, supplement representative frames.
high_risk_sample = high_risk.head(24).copy()

# Representative: every 6th frame + top risk if not included.
representative = qc.sort_values("scan_frame_id").iloc[::6].copy()

sample = pd.concat([high_risk_sample, representative], ignore_index=True)
sample = sample.drop_duplicates(subset=["scan_frame_id"]).sort_values(
    ["visual_qc_risk_score", "scan_frame_id"],
    ascending=[False, True],
).reset_index(drop=True)

# Manual review template.
manual_template = sample[
    [
        "scan_frame_id",
        "timestamp",
        "video_id",
        "segmented_bbox_count",
        "fallback_count",
        "mean_mask_area_fraction",
        "min_mask_area_fraction",
        "max_mask_area_fraction",
        "visual_qc_risk_score",
        "visual_qc_risk_reasons",
        "segmentation_overlay_path",
    ]
].copy()

manual_template["manual_visual_qc_status"] = ""
manual_template["manual_visual_qc_score_1_bad_5_good"] = ""
manual_template["manual_visual_qc_notes"] = ""
manual_template["needs_fix"] = ""

# Outputs.
qc_index_path = STATS / "week6_segmentation_visual_qc_frame_index.csv"
sample_path = STATS / "week6_segmentation_visual_qc_sample_for_manual_review.csv"
manual_template_path = STATS / "week6_segmentation_visual_qc_manual_review_template.csv"

safe_to_csv(qc_sorted, qc_index_path)
safe_to_csv(sample, sample_path)
safe_to_csv(manual_template, manual_template_path)

# Contact sheets.
high_sheet_path = QC_DIR / "segmentation_visual_qc_high_risk_contact_sheet.jpg"
sample_sheet_path = QC_DIR / "segmentation_visual_qc_manual_review_sample_contact_sheet.jpg"
representative_sheet_path = QC_DIR / "segmentation_visual_qc_representative_contact_sheet.jpg"

make_contact_sheet(high_risk_sample, high_sheet_path, "Segmentation Visual QC - High Risk Sample")
make_contact_sheet(sample.head(30), sample_sheet_path, "Segmentation Visual QC - Manual Review Sample")
make_contact_sheet(representative, representative_sheet_path, "Segmentation Visual QC - Representative Frames")

# HTML reviewer.
html_path = QC_DIR / "segmentation_visual_qc_static_reviewer.html"

cards = []

for _, r in sample.iterrows():
    overlay_rel = "../" + rel(r["segmentation_overlay_path"])
    risk_score = html.escape(str(r["visual_qc_risk_score"]))
    reasons = html.escape(str(r["visual_qc_risk_reasons"]))
    sid = html.escape(str(r["scan_frame_id"]))
    timestamp = html.escape(str(r.get("timestamp", "")))
    video_id = html.escape(str(r.get("video_id", "")))

    cards.append(f"""
    <div class="card">
      <h3>{sid} — risk {risk_score}</h3>
      <p><b>Timestamp:</b> {timestamp} | <b>Video:</b> {video_id}</p>
      <p><b>Reasons:</b> {reasons}</p>
      <p><b>Bboxes:</b> {r.get('segmented_bbox_count', '')} | <b>Fallback:</b> {r.get('fallback_count', '')} | <b>Mean mask fraction:</b> {r.get('mean_mask_area_fraction', ''):.3f}</p>
      <img src="{html.escape(overlay_rel)}">
    </div>
    """)

html_text = f"""<!DOCTYPE html>
<html>
<head>
<meta charset="UTF-8">
<title>Week 6 Segmentation Visual QC Reviewer</title>
<style>
body {{
  font-family: Arial, sans-serif;
  margin: 24px;
  background: #f7f7f7;
}}
.card {{
  background: white;
  border: 1px solid #ddd;
  padding: 14px;
  margin-bottom: 22px;
  border-radius: 8px;
}}
.card img {{
  max-width: 100%;
  border: 1px solid #ccc;
}}
.summary {{
  background: white;
  border: 1px solid #ddd;
  padding: 16px;
  margin-bottom: 20px;
}}
</style>
</head>
<body>
<h1>Week 6 Segmentation Visual QC Reviewer</h1>
<div class="summary">
<p>This page lists selected segmentation overlay frames for manual visual quality control.</p>
<p><b>Total frames:</b> {len(qc_sorted)} | <b>Sample frames:</b> {len(sample)} | <b>High-risk frames:</b> {len(high_risk)}</p>
<p>Manual review template: <code>outputs/dataset_statistics/week6_segmentation_visual_qc_manual_review_template.csv</code></p>
</div>
{''.join(cards)}
</body>
</html>
"""

html_path.write_text(html_text)

# Summary.
summary = pd.DataFrame([
    {
        "metric": "total_segmentation_frames",
        "value": len(qc_sorted),
        "interpretation": "All scanpoint frames with segmentation overlays.",
    },
    {
        "metric": "overlay_exists_count",
        "value": int(qc_sorted["segmentation_overlay_exists"].sum()),
        "interpretation": "Frames with overlay image available.",
    },
    {
        "metric": "high_risk_frame_count",
        "value": len(high_risk),
        "interpretation": "Frames with visual QC risk score >= 3.",
    },
    {
        "metric": "manual_review_sample_count",
        "value": len(sample),
        "interpretation": "Frames selected for manual visual review.",
    },
    {
        "metric": "manual_review_required",
        "value": True,
        "interpretation": "Human visual confirmation is needed before claiming segmentation quality.",
    },
])

summary_path = STATS / "week6_segmentation_visual_qc_summary.csv"
safe_to_csv(summary, summary_path)

# Notes.
note_path = NOTES / "week6_segmentation_visual_qc_package_notes.md"

with open(note_path, "w") as f:
    f.write("# Week 6 Segmentation Visual QC Package\n\n")

    f.write("## Purpose\n\n")
    f.write(
        "This step creates a visual quality-control package for the preliminary segmentation baseline. "
        "It does not automatically declare segmentation masks correct; instead, it prepares risk-ranked overlays for manual review.\n\n"
    )

    f.write("## Outputs\n\n")
    f.write(f"- Full frame QC index: `{qc_index_path}`\n")
    f.write(f"- Manual review sample: `{sample_path}`\n")
    f.write(f"- Manual review template: `{manual_template_path}`\n")
    f.write(f"- High-risk contact sheet: `{high_sheet_path}`\n")
    f.write(f"- Manual review sample contact sheet: `{sample_sheet_path}`\n")
    f.write(f"- Representative contact sheet: `{representative_sheet_path}`\n")
    f.write(f"- Static HTML reviewer: `{html_path}`\n")
    f.write(f"- Summary: `{summary_path}`\n\n")

    f.write("## Summary\n\n")
    f.write(summary.to_markdown(index=False))
    f.write("\n\n")

    f.write("## What needs human review\n\n")
    f.write(
        "Open the contact sheets or the static HTML reviewer and inspect whether the red segmentation contours roughly follow pig bodies. "
        "Use the manual review template to record obvious failures, acceptable masks, and frames that need correction.\n\n"
    )

    f.write("## Interpretation\n\n")
    f.write(
        "This package upgrades the segmentation baseline from pure automatic output to an inspectable QC artifact. "
        "Final segmentation quality should be reported based on the manual visual review outcome.\n"
    )

print("Saved:")
print(qc_index_path)
print(sample_path)
print(manual_template_path)
print(high_sheet_path)
print(sample_sheet_path)
print(representative_sheet_path)
print(html_path)
print(summary_path)
print(note_path)

print()
print("=== Segmentation visual QC summary ===")
print(summary.to_string(index=False))

print()
print("=== Top high-risk frames ===")
cols = [
    "scan_frame_id",
    "timestamp",
    "video_id",
    "segmented_bbox_count",
    "fallback_count",
    "visual_qc_risk_score",
    "visual_qc_risk_reasons",
]
print(qc_sorted[cols].head(20).to_string(index=False))
