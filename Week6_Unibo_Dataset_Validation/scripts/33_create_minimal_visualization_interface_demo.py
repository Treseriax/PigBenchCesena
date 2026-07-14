from pathlib import Path
import json
import csv
import html
import pandas as pd


ROOT = Path.home() / "PigBench"
W6 = ROOT / "Week6_Unibo_Dataset_Validation"

OUT_GT = W6 / "outputs/unified_ground_truth"
OUT_VIS = W6 / "outputs/visual_label_check"
OUT_STATS = W6 / "outputs/dataset_statistics"
NOTES = W6 / "notes"
INTERFACE_DIR = W6 / "interface_demo"

NESTED_JSON = OUT_GT / "week6_scanpoint_annotations_nested_for_viewer.json"
MARKER_OVERLAY_DIR = OUT_VIS / "marker_bbox_overlay_v3"

INTERFACE_DIR.mkdir(parents=True, exist_ok=True)
OUT_STATS.mkdir(parents=True, exist_ok=True)
NOTES.mkdir(parents=True, exist_ok=True)


def safe_to_csv(df, path):
    df.to_csv(
        path,
        index=False,
        quoting=csv.QUOTE_ALL,
        escapechar="\\",
        lineterminator="\n"
    )


def rel_to_w6(path):
    try:
        return str(Path(path).relative_to(W6))
    except Exception:
        return str(path)


if not NESTED_JSON.exists():
    raise FileNotFoundError(NESTED_JSON)

data = json.loads(NESTED_JSON.read_text())
frames = data.get("frames", [])

# Build compact interface index.
rows = []

for fr in frames:
    scan_frame_id = fr.get("scan_frame_id")
    timestamp = fr.get("timestamp")
    video_id = fr.get("video_id")
    video_match_status = fr.get("video_match_status")
    manual_labels = fr.get("manual_labels", [])
    bboxes = fr.get("detector_bboxes", [])

    # Find overlay image if exists.
    overlay_candidates = sorted(MARKER_OVERLAY_DIR.glob(f"{scan_frame_id}__*marker_bbox_overlay_v3.jpg"))
    overlay_path = overlay_candidates[0] if overlay_candidates else ""

    behaviour_summary = "; ".join([
        f"{lab.get('colour_id')}={lab.get('behaviour_code')}"
        for lab in manual_labels
    ])

    rows.append({
        "scan_frame_id": scan_frame_id,
        "timestamp": timestamp,
        "video_id": video_id,
        "video_match_status": video_match_status,
        "manual_label_count": len(manual_labels),
        "detector_bbox_count": len(bboxes),
        "behaviour_summary": behaviour_summary,
        "frame_image_path": fr.get("frame_image_path"),
        "overlay_image_path": str(overlay_path) if overlay_path else "",
        "overlay_image_exists": bool(overlay_path),
    })

index_df = pd.DataFrame(rows)
index_path = INTERFACE_DIR / "week6_visualization_interface_index.csv"
safe_to_csv(index_df, index_path)

# ------------------------------------------------------------
# Streamlit app
# ------------------------------------------------------------
streamlit_path = INTERFACE_DIR / "week6_visualization_streamlit_app.py"

streamlit_code = f'''from pathlib import Path
import json
import pandas as pd
import streamlit as st

W6 = Path.home() / "PigBench" / "Week6_Unibo_Dataset_Validation"
NESTED_JSON = W6 / "outputs/unified_ground_truth/week6_scanpoint_annotations_nested_for_viewer.json"
INDEX_CSV = W6 / "interface_demo/week6_visualization_interface_index.csv"

st.set_page_config(page_title="Week 6 Unibo Visualization Demo", layout="wide")

st.title("Week 6 Unibo Visualization Demo")
st.caption("Loads scanpoint annotations, manual colour-ID behaviour labels, and detector bbox visualization outputs.")

data = json.loads(NESTED_JSON.read_text())
frames = data.get("frames", [])
index = pd.read_csv(INDEX_CSV)

scan_ids = index["scan_frame_id"].tolist()
selected = st.selectbox("Select scan frame", scan_ids)

row = index[index["scan_frame_id"] == selected].iloc[0]
frame = next(fr for fr in frames if fr.get("scan_frame_id") == selected)

left, right = st.columns([2, 1])

with left:
    st.subheader("Overlay visualization")
    overlay_path = Path(str(row["overlay_image_path"]))
    if overlay_path.exists():
        st.image(str(overlay_path), caption=str(overlay_path), use_container_width=True)
    else:
        st.warning("Overlay image not found.")
        frame_path = Path(str(row["frame_image_path"]))
        if frame_path.exists():
            st.image(str(frame_path), caption=str(frame_path), use_container_width=True)

with right:
    st.subheader("Frame metadata")
    st.write({{
        "scan_frame_id": frame.get("scan_frame_id"),
        "timestamp": frame.get("timestamp"),
        "video_id": frame.get("video_id"),
        "video_match_status": frame.get("video_match_status"),
        "camera_id": frame.get("camera_id"),
        "crate_id": frame.get("crate_id"),
        "pen_id": frame.get("pen_id"),
        "frame_index": frame.get("frame_index"),
        "timestamp_sec_in_video": frame.get("timestamp_sec_in_video"),
    }})

    st.subheader("Manual labels")
    st.dataframe(pd.DataFrame(frame.get("manual_labels", [])), use_container_width=True)

    st.subheader("Detector bboxes")
    st.dataframe(pd.DataFrame(frame.get("detector_bboxes", [])), use_container_width=True)

st.subheader("All scan frames")
st.dataframe(index, use_container_width=True)
'''

streamlit_path.write_text(streamlit_code)

# ------------------------------------------------------------
# Static HTML demo
# ------------------------------------------------------------
html_path = INTERFACE_DIR / "week6_static_visualization_viewer.html"

# To keep HTML portable inside package, use relative paths from W6.
cards = []

for _, r in index_df.iterrows():
    overlay_path = r["overlay_image_path"]
    if overlay_path:
        overlay_rel = rel_to_w6(overlay_path)
    else:
        overlay_rel = ""

    labels = html.escape(str(r["behaviour_summary"]))

    card = f"""
    <div class="card">
      <h3>{html.escape(str(r['scan_frame_id']))} — {html.escape(str(r['timestamp']))}</h3>
      <p><b>Video:</b> {html.escape(str(r['video_id']))}</p>
      <p><b>Video match:</b> {html.escape(str(r['video_match_status']))}</p>
      <p><b>Manual labels:</b> {html.escape(str(r['manual_label_count']))} | <b>Detector bboxes:</b> {html.escape(str(r['detector_bbox_count']))}</p>
      <p><b>Behaviour summary:</b> {labels}</p>
      {"<img src='../" + html.escape(overlay_rel) + "'>" if overlay_rel else "<p>No overlay image found.</p>"}
    </div>
    """
    cards.append(card)

html_text = f"""<!DOCTYPE html>
<html>
<head>
<meta charset="UTF-8">
<title>Week 6 Unibo Static Visualization Viewer</title>
<style>
body {{
  font-family: Arial, sans-serif;
  margin: 24px;
  background: #f7f7f7;
}}
h1 {{
  margin-bottom: 4px;
}}
.summary {{
  background: white;
  border: 1px solid #ddd;
  padding: 16px;
  margin-bottom: 20px;
}}
.card {{
  background: white;
  border: 1px solid #ddd;
  padding: 14px;
  margin-bottom: 20px;
  border-radius: 8px;
}}
.card img {{
  max-width: 100%;
  border: 1px solid #ccc;
}}
code {{
  background: #eee;
  padding: 2px 4px;
}}
</style>
</head>
<body>
<h1>Week 6 Unibo Static Visualization Viewer</h1>
<div class="summary">
<p>This static demo lists all scanpoint frames with manual colour-ID behaviour labels and detector bbox overlay images.</p>
<p><b>Frame count:</b> {len(index_df)} | <b>Source JSON:</b> <code>outputs/unified_ground_truth/week6_scanpoint_annotations_nested_for_viewer.json</code></p>
<p><b>Note:</b> Detector bboxes are automatic YOLOv8-s outputs, not manual bbox ground truth. Colour-marker links are candidate-level only.</p>
</div>
{''.join(cards)}
</body>
</html>
"""

html_path.write_text(html_text)

# ------------------------------------------------------------
# Viewer README / notes
# ------------------------------------------------------------
readme_path = INTERFACE_DIR / "README_visualization_interface_demo.md"

with open(readme_path, "w") as f:
    f.write("# Week 6 Visualization Interface Demo\n\n")

    f.write("## Purpose\n\n")
    f.write(
        "This demo addresses the visualization interface requirement. "
        "It loads the nested scanpoint annotation JSON and displays frame metadata, manual colour-ID behaviour labels, detector bboxes, and overlay visualizations.\n\n"
    )

    f.write("## Files\n\n")
    f.write(f"- Streamlit app: `{streamlit_path}`\n")
    f.write(f"- Static HTML viewer: `{html_path}`\n")
    f.write(f"- Interface index CSV: `{index_path}`\n")
    f.write(f"- Source JSON: `{NESTED_JSON}`\n\n")

    f.write("## How to run Streamlit\n\n")
    f.write("```bash\n")
    f.write("cd ~/PigBench\n")
    f.write("streamlit run Week6_Unibo_Dataset_Validation/interface_demo/week6_visualization_streamlit_app.py --server.port 8504\n")
    f.write("```\n\n")

    f.write("## Interpretation\n\n")
    f.write(
        "The demo is intended for visual quality control and dataset exploration. "
        "It does not claim final identity-resolved bbox-to-colour ground truth. "
        "Manual labels are shown separately from detector bboxes because bbox-to-colour assignment remains candidate-level.\n"
    )

# Summary table.
summary = pd.DataFrame([
    {
        "artifact": "nested_annotation_json",
        "path": rel_to_w6(NESTED_JSON),
        "exists": NESTED_JSON.exists(),
        "purpose": "Frame-level JSON with manual labels and detector bboxes.",
    },
    {
        "artifact": "streamlit_app",
        "path": rel_to_w6(streamlit_path),
        "exists": streamlit_path.exists(),
        "purpose": "Interactive interface demo.",
    },
    {
        "artifact": "static_html_viewer",
        "path": rel_to_w6(html_path),
        "exists": html_path.exists(),
        "purpose": "Portable static visualization demo.",
    },
    {
        "artifact": "interface_index_csv",
        "path": rel_to_w6(index_path),
        "exists": index_path.exists(),
        "purpose": "Compact index of 72 scan frames for the viewer.",
    },
])

summary_path = OUT_STATS / "week6_visualization_interface_demo_summary.csv"
safe_to_csv(summary, summary_path)

note_path = NOTES / "week6_visualization_interface_demo_notes.md"

with open(note_path, "w") as f:
    f.write("# Week 6 Visualization Interface Demo Notes\n\n")

    f.write("## Purpose\n\n")
    f.write(
        "This step creates a minimal visualization interface demo using the nested scanpoint annotation JSON. "
        "It closes the interface gap by providing both a Streamlit app and a static HTML viewer.\n\n"
    )

    f.write("## Outputs\n\n")
    f.write(summary.to_markdown(index=False))
    f.write("\n\n")

    f.write("## Viewer coverage\n\n")
    f.write(f"- Indexed scan frames: `{len(index_df)}`\n")
    f.write(f"- Frames with overlay image: `{int(index_df['overlay_image_exists'].sum())}`\n")
    f.write(f"- Total manual labels listed: `{int(index_df['manual_label_count'].sum())}`\n")
    f.write(f"- Total detector bboxes listed: `{int(index_df['detector_bbox_count'].sum())}`\n\n")

    f.write("## Interpretation\n\n")
    f.write(
        "The visualization demo loads scanpoint annotation JSON and displays frame-level metadata, manual behaviour labels, detector bboxes, and overlay images. "
        "It is a dataset exploration/QC interface rather than a final annotation editing tool.\n"
    )

print("Saved:")
print(streamlit_path)
print(html_path)
print(index_path)
print(readme_path)
print(summary_path)
print(note_path)

print()
print("=== Interface summary ===")
print(summary.to_string(index=False))

print()
print("=== Viewer coverage ===")
print(index_df[["scan_frame_id", "manual_label_count", "detector_bbox_count", "overlay_image_exists"]].head(10).to_string(index=False))
print("frames:", len(index_df))
print("overlays:", int(index_df["overlay_image_exists"].sum()))
print("manual labels:", int(index_df["manual_label_count"].sum()))
print("detector bboxes:", int(index_df["detector_bbox_count"].sum()))
