from pathlib import Path
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
    st.write({
        "scan_frame_id": frame.get("scan_frame_id"),
        "timestamp": frame.get("timestamp"),
        "video_id": frame.get("video_id"),
        "video_match_status": frame.get("video_match_status"),
        "camera_id": frame.get("camera_id"),
        "crate_id": frame.get("crate_id"),
        "pen_id": frame.get("pen_id"),
        "frame_index": frame.get("frame_index"),
        "timestamp_sec_in_video": frame.get("timestamp_sec_in_video"),
    })

    st.subheader("Manual labels")
    st.dataframe(pd.DataFrame(frame.get("manual_labels", [])), use_container_width=True)

    st.subheader("Detector bboxes")
    st.dataframe(pd.DataFrame(frame.get("detector_bboxes", [])), use_container_width=True)

st.subheader("All scan frames")
st.dataframe(index, use_container_width=True)
