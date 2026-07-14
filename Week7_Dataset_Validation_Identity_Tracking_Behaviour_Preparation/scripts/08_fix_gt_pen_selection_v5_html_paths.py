from pathlib import Path
import pandas as pd
import csv


ROOT = Path.home() / "PigBench"
W7 = ROOT / "Week7_Dataset_Validation_Identity_Tracking_Behaviour_Preparation"

REVIEW_TEMPLATE = W7 / "outputs" / "roi_and_crate_mapping" / "week7_gt_pen_detection_selection_v5_manual_override_template.csv"
OUT_HTML = W7 / "outputs" / "visualizations" / "week7_gt_pen_selection_v5_static_review_fixed.html"
OUT_SUMMARY = W7 / "outputs" / "dataset_statistics" / "week7_gt_pen_selection_v5_html_fix_summary.csv"
NOTE = W7 / "notes" / "week7_gt_pen_selection_v5_html_fix_notes.md"


def safe_to_csv(df, path):
    df.to_csv(
        path,
        index=False,
        quoting=csv.QUOTE_ALL,
        escapechar="\\",
        lineterminator="\n",
    )


review = pd.read_csv(REVIEW_TEMPLATE)

rows = []

for _, r in review.sort_values("scan_frame_id").iterrows():
    img_path = Path(str(r["review_image_path"]))

    # HTML file is inside:
    # Week7/.../outputs/visualizations/
    # Review images are inside:
    # Week7/.../outputs/visualizations/gt_pen_selection_v5_review_frames/
    # Therefore the correct relative src is only:
    # gt_pen_selection_v5_review_frames/<image>.jpg
    src = "gt_pen_selection_v5_review_frames/" + img_path.name

    risky_class = "risky" if bool(r.get("frame_needs_review", False)) else "clean"

    rows.append(
        f"""
        <div class="frame-card {risky_class}">
          <h3>{r['scan_frame_id']}</h3>
          <p>
            selected={r['selected_count']} |
            clean={r['clean_selected_count']} |
            review={r['selected_needs_review_count']} |
            frame needs review={r.get('frame_needs_review', '')}
          </p>
          <img src="{src}" alt="{r['scan_frame_id']} review image">
        </div>
        """
    )

html = f"""
<!DOCTYPE html>
<html>
<head>
  <meta charset="utf-8">
  <title>Week 7 GT Pen Selection v5 Review Fixed</title>
  <style>
    body {{
      font-family: Arial, sans-serif;
      margin: 24px;
      background: #f7f7f7;
    }}
    .legend {{
      background: white;
      padding: 14px;
      border-radius: 8px;
      margin-bottom: 20px;
      border-left: 6px solid #333;
    }}
    .frame-card {{
      background: white;
      margin: 18px 0;
      padding: 12px;
      border-radius: 8px;
      box-shadow: 0 1px 4px rgba(0,0,0,0.15);
    }}
    .frame-card.risky {{
      border-left: 8px solid darkorange;
    }}
    .frame-card.clean {{
      border-left: 8px solid green;
    }}
    img {{
      max-width: 100%;
      border: 1px solid #ddd;
      display: block;
    }}
    .clean-text {{
      color: green;
      font-weight: bold;
    }}
    .review-text {{
      color: darkorange;
      font-weight: bold;
    }}
    .risk-text {{
      color: red;
      font-weight: bold;
    }}
  </style>
</head>
<body>
  <h1>Week 7 Ground Truth Pen Selection v5 Review</h1>

  <div class="legend">
    <p><span class="clean-text">Green / CLEAN</span>: selected and high-confidence.</p>
    <p><span class="review-text">Orange / REVIEW</span>: selected but should be visually checked.</p>
    <p><span class="risk-text">Red / IGN_RISK</span>: ignored but suspicious.</p>
    <p>The goal is to avoid using wrong-pen detections for colour and behaviour matching.</p>
  </div>

  {''.join(rows)}
</body>
</html>
"""

OUT_HTML.write_text(html)

summary = pd.DataFrame([
    {
        "item": "input_review_template",
        "path": str(REVIEW_TEMPLATE),
        "exists": REVIEW_TEMPLATE.exists(),
    },
    {
        "item": "fixed_html",
        "path": str(OUT_HTML),
        "exists": OUT_HTML.exists(),
    },
    {
        "item": "review_frame_rows",
        "path": "",
        "exists": len(review),
    },
    {
        "item": "relative_image_path_rule",
        "path": "gt_pen_selection_v5_review_frames/<image>.jpg",
        "exists": True,
    },
])

safe_to_csv(summary, OUT_SUMMARY)

NOTE.write_text(
    "# Week 7 GT Pen Selection v5 HTML Path Fix\n\n"
    "## Problem\n\n"
    "The original HTML viewer used incorrect relative image paths, so review images did not appear in the browser.\n\n"
    "## Fix\n\n"
    "The fixed HTML file uses image paths relative to the `outputs/visualizations` directory:\n\n"
    "`gt_pen_selection_v5_review_frames/<image>.jpg`\n\n"
    "## Output\n\n"
    f"- Fixed HTML: `{OUT_HTML}`\n"
)

print("Saved fixed HTML:")
print(OUT_HTML)
print()
print("Saved summary:")
print(OUT_SUMMARY)
print()
print("Saved note:")
print(NOTE)
print()
print(summary.to_string(index=False))
