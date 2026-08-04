from pathlib import Path
import json
import pandas as pd
import streamlit as st

BASE = Path.home() / "PigBench" / "Full_Unibo_Behaviour_Pipeline"
V80 = BASE / "outputs/v80_final_project_completion"
JSON_DIR = V80 / "01_dataset_json_export"
APP_OUT = V80 / "02_annotation_visualizer_app"
CORR_DIR = APP_OUT / "manual_corrections"
CORR_DIR.mkdir(parents=True, exist_ok=True)

MANIFEST = JSON_DIR / "v80c_json_export_manifest.csv"

st.set_page_config(page_title="Unibo Pig Annotation Visualizer", layout="wide")
st.title("Unibo Pig Behaviour Annotation Visualizer / Corrector")

@st.cache_data
def load_manifest():
    return pd.read_csv(MANIFEST).fillna("")

def load_json(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

def save_json(path, data):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)

def list_to_text(x):
    if isinstance(x, list):
        return ";".join(map(str, x))
    return str(x)

def text_to_list(x):
    x = str(x)
    if not x:
        return []
    return [p for p in x.split(";") if p]

def annotations_to_df(annotations):
    rows = []
    for a in annotations:
        rows.append({
            "clip_id": a.get("clip_id", ""),
            "window_id": a.get("window_id", ""),
            "start_sec": a.get("start_sec", ""),
            "end_sec": a.get("end_sec", ""),
            "behaviour_label": a.get("behaviour_label", ""),
            "identity_colour": a.get("identity_colour", ""),
            "candidate_tracklets": list_to_text(a.get("candidate_tracklets", [])),
            "identity_dataset_status": a.get("identity_dataset_status", ""),
            "usable_for_behaviour_clip_dataset": a.get("usable_for_behaviour_clip_dataset", True),
            "usable_for_identity_training": a.get("usable_for_identity_training", False),
        })
    return pd.DataFrame(rows)

def df_to_annotations(df):
    annotations = []
    for _, r in df.fillna("").iterrows():
        if not str(r.get("behaviour_label", "")) and not str(r.get("start_sec", "")):
            continue
        try:
            start = float(r.get("start_sec", 0))
            end = float(r.get("end_sec", 0))
        except Exception:
            start = None
            end = None
        annotations.append({
            "clip_id": str(r.get("clip_id", "")),
            "window_id": str(r.get("window_id", "")),
            "start_sec": start,
            "end_sec": end,
            "duration_sec": (end - start) if start is not None and end is not None else None,
            "behaviour_label": str(r.get("behaviour_label", "")),
            "identity_colour": str(r.get("identity_colour", "")),
            "candidate_tracklets": text_to_list(r.get("candidate_tracklets", "")),
            "identity_dataset_status": str(r.get("identity_dataset_status", "")),
            "usable_for_behaviour_clip_dataset": bool(r.get("usable_for_behaviour_clip_dataset", True)),
            "usable_for_identity_training": bool(r.get("usable_for_identity_training", False)),
            "bbox_source": "roi_filtered_top6_candidate_tracklets",
            "claim_boundary": "corrected_json_copy_not_original_overwrite",
        })
    return annotations

manifest = load_manifest()

with st.sidebar:
    mode = st.radio("Mode", ["Visualization-only", "Manual correction / labeling"])
    file_type = st.selectbox(
        "JSON type",
        ["all", "behaviour_annotations", "metadata_status_only_no_behaviour_labels"],
    )
    search = st.text_input("Search", "")

filtered = manifest.copy()
if file_type != "all":
    filtered = filtered[filtered["annotation_file_type"] == file_type]

if search:
    s = search.lower()
    filtered = filtered[
        filtered.apply(lambda r: s in " ".join(map(str, r.values)).lower(), axis=1)
    ]

if len(filtered) == 0:
    st.error("No matching video.")
    st.stop()

labels = (
    filtered["video_id"].astype(str)
    + " | "
    + filtered["video_filename"].astype(str)
    + " | "
    + filtered["annotation_file_type"].astype(str)
).tolist()

selected = st.sidebar.selectbox("Video", labels)
video_id = selected.split(" | ")[0]
row = manifest[manifest["video_id"].astype(str) == video_id].iloc[0]

data = load_json(row["json_path"])
video = data.get("video", {})
annotations = data.get("annotations", [])

left, right = st.columns([1, 1])

with left:
    st.subheader("Video metadata")
    st.json(video)

    st.subheader("Video player")
    video_path = video.get("video_path", "")
    st.write(video_path)
    try:
        st.video(video_path)
    except Exception as e:
        st.warning(f"Video could not be loaded by Streamlit: {e}")

    if annotations:
        choices = [
            f"{i}: {a.get('start_sec')} - {a.get('end_sec')} | {a.get('identity_colour')} | {a.get('behaviour_label')}"
            for i, a in enumerate(annotations)
        ]
        chosen = st.selectbox("Current annotation interval", choices)
        idx = int(chosen.split(":")[0])
        st.subheader("Selected annotation")
        st.json(annotations[idx])
    else:
        st.info("Metadata-only JSON: no behaviour annotations yet.")

with right:
    st.subheader("Annotations")
    table = annotations_to_df(annotations)

    if mode == "Visualization-only":
        st.dataframe(table, use_container_width=True)
    else:
        st.info("Edited annotations are saved as a corrected JSON copy. Original JSON is not overwritten.")
        edited = st.data_editor(table, use_container_width=True, num_rows="dynamic")
        if st.button("Save corrected JSON copy"):
            corrected = dict(data)
            corrected["annotations"] = df_to_annotations(edited)
            corrected["annotation_file_type"] = (
                "behaviour_annotations"
                if len(corrected["annotations"]) > 0
                else "metadata_status_only_no_behaviour_labels"
            )
            corrected["summary"]["annotation_count"] = len(corrected["annotations"])
            corrected["summary"]["has_behaviour_annotations"] = len(corrected["annotations"]) > 0
            corrected["manual_correction"] = {
                "source_json_path": str(row["json_path"]),
                "note": "Original exported JSON was not overwritten.",
            }
            out_path = CORR_DIR / f"{video_id}__corrected.json"
            save_json(out_path, corrected)
            st.success(f"Saved corrected JSON: {out_path}")

st.divider()
st.caption(
    "Candidate tracklets are not final colour-to-track identity assignments unless manually reviewed."
)
