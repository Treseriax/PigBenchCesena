from pathlib import Path
import json
from collections import Counter, defaultdict
import pandas as pd


ROOT = Path("Week4_5_Behaviour_Representation")
OUT = ROOT / "outputs/dataset_comparison"
NOTES = ROOT / "notes/public_dataset_notes"

UNIBO_JSON = ROOT / "data/unibo_inputs/corrected_scan_window_behaviour_annotations.json"

ED_SCHEMA = OUT / "edinburgh_sample_output_json_schema_summary.csv"
ED_BEHAV = OUT / "edinburgh_sample_behaviour_counts.csv"
ED_KEYS = OUT / "edinburgh_sample_json_key_counts.csv"

OUT.mkdir(parents=True, exist_ok=True)
NOTES.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------------
# Load Edinburgh inspection outputs
# ---------------------------------------------------------------------

ed_schema = pd.read_csv(ED_SCHEMA)
ed_behav = pd.read_csv(ED_BEHAV)
ed_keys = pd.read_csv(ED_KEYS)

ed_schema_map = dict(zip(ed_schema["field"], ed_schema["value"]))
ed_labels = ", ".join(ed_behav["behaviour"].astype(str).tolist())

# ---------------------------------------------------------------------
# Load and inspect Unibo annotation JSON
# ---------------------------------------------------------------------

with open(UNIBO_JSON) as f:
    unibo_data = json.load(f)

key_counter = Counter()
dict_count = 0
list_count = 0
bbox_like_count = 0

segment_counter = Counter()
track_id_counter = Counter()
frame_values = []

behaviour_field_counter = Counter()
behaviour_value_counter = Counter()

identity_field_counter = Counter()
identity_value_counter = Counter()

bbox_key_patterns = [
    {"x1", "y1", "x2", "y2"},
    {"x", "y", "width", "height"},
    {"bbox"},
]

def scalar(v):
    return isinstance(v, (str, int, float, bool)) or v is None

def walk(obj):
    global dict_count, list_count, bbox_like_count

    if isinstance(obj, dict):
        dict_count += 1

        keys = set(obj.keys())
        for k, v in obj.items():
            key_counter[k] += 1

            lk = k.lower()

            if scalar(v):
                if "segment" in lk:
                    segment_counter[str(v)] += 1

                if "track" in lk and "id" in lk:
                    track_id_counter[str(v)] += 1

                if lk in {"frame", "frame_id", "frame_number", "framenumber"} or "frame" == lk:
                    try:
                        frame_values.append(int(v))
                    except Exception:
                        pass

                if "behaviour" in lk or "behavior" in lk:
                    behaviour_field_counter[k] += 1
                    behaviour_value_counter[str(v)] += 1

                if "identity" in lk or "pig_id" in lk or "pig_identity" in lk or "colour" in lk or "color" in lk:
                    identity_field_counter[k] += 1
                    identity_value_counter[str(v)] += 1

        if "bbox" in keys:
            bbox_like_count += 1
        elif {"x1", "y1", "x2", "y2"}.issubset(keys):
            bbox_like_count += 1
        elif {"x", "y", "width", "height"}.issubset(keys):
            bbox_like_count += 1

        for v in obj.values():
            walk(v)

    elif isinstance(obj, list):
        list_count += 1
        for item in obj:
            walk(item)

walk(unibo_data)

unibo_top_type = type(unibo_data).__name__
unibo_top_keys = ", ".join(unibo_data.keys()) if isinstance(unibo_data, dict) else ""
unibo_top_len = len(unibo_data) if isinstance(unibo_data, list) else ""

unibo_unique_segments = len(segment_counter)
unibo_unique_track_ids = len(track_id_counter)

unibo_min_frame = min(frame_values) if frame_values else ""
unibo_max_frame = max(frame_values) if frame_values else ""
unibo_frame_count = len(frame_values)

unibo_behaviour_values = ", ".join([k for k, _ in behaviour_value_counter.most_common(20)])
unibo_behaviour_fields = ", ".join(behaviour_field_counter.keys())

unibo_identity_fields = ", ".join(identity_field_counter.keys())
unibo_identity_values = ", ".join([k for k, _ in identity_value_counter.most_common(20)])

# ---------------------------------------------------------------------
# Save Unibo inspection outputs
# ---------------------------------------------------------------------

unibo_summary_rows = [
    {"field": "top_level_type", "value": unibo_top_type},
    {"field": "top_level_keys", "value": unibo_top_keys},
    {"field": "top_level_length_if_list", "value": unibo_top_len},
    {"field": "dict_count_recursive", "value": dict_count},
    {"field": "list_count_recursive", "value": list_count},
    {"field": "bbox_like_count", "value": bbox_like_count},
    {"field": "unique_segments", "value": unibo_unique_segments},
    {"field": "unique_track_ids", "value": unibo_unique_track_ids},
    {"field": "min_frame", "value": unibo_min_frame},
    {"field": "max_frame", "value": unibo_max_frame},
    {"field": "frame_value_count", "value": unibo_frame_count},
    {"field": "behaviour_fields_detected", "value": unibo_behaviour_fields},
    {"field": "identity_fields_detected", "value": unibo_identity_fields},
]

unibo_summary = pd.DataFrame(unibo_summary_rows)
unibo_summary_path = OUT / "unibo_annotation_schema_summary.csv"
unibo_summary.to_csv(unibo_summary_path, index=False)

unibo_keys = pd.DataFrame(
    [{"key": k, "count": v} for k, v in key_counter.most_common()]
)
unibo_keys_path = OUT / "unibo_annotation_json_key_counts.csv"
unibo_keys.to_csv(unibo_keys_path, index=False)

unibo_behav = pd.DataFrame(
    [{"behaviour_or_behavior_value": k, "count": v} for k, v in behaviour_value_counter.most_common()]
)
unibo_behav_path = OUT / "unibo_annotation_behaviour_value_counts.csv"
unibo_behav.to_csv(unibo_behav_path, index=False)

unibo_identity = pd.DataFrame(
    [{"identity_or_colour_value": k, "count": v} for k, v in identity_value_counter.most_common()]
)
unibo_identity_path = OUT / "unibo_annotation_identity_value_counts.csv"
unibo_identity.to_csv(unibo_identity_path, index=False)

# ---------------------------------------------------------------------
# Comparison table
# ---------------------------------------------------------------------

comparison_rows = [
    {
        "aspect": "Data source",
        "Edinburgh manually annotated dataset": "Public Edinburgh annotated.tar ground-truth package",
        "Unibo corrected scan-window dataset": "Our Week 3 corrected scan-window outputs"
    },
    {
        "aspect": "Main annotation file",
        "Edinburgh manually annotated dataset": "output.json inside each annotated sequence folder",
        "Unibo corrected scan-window dataset": "corrected_scan_window_behaviour_annotations.json"
    },
    {
        "aspect": "Top-level JSON structure",
        "Edinburgh manually annotated dataset": str(ed_schema_map.get("top_level_type", "")) + " with keys: " + str(ed_schema_map.get("top_level_keys", "")),
        "Unibo corrected scan-window dataset": str(unibo_top_type) + (" with keys: " + unibo_top_keys if unibo_top_keys else "")
    },
    {
        "aspect": "Object / track representation",
        "Edinburgh manually annotated dataset": "objects list; each object corresponds to one tracked pig; inspected sample has 8 objects",
        "Unibo corrected scan-window dataset": f"frame-level track records with track IDs; detected unique track-id values: {unibo_unique_track_ids}"
    },
    {
        "aspect": "Frame range / temporal representation",
        "Edinburgh manually annotated dataset": f"Ground-truth frameNumber range {ed_schema_map.get('min_frameNumber')}–{ed_schema_map.get('max_frameNumber')}; source page notes GT corresponds to every third raw video frame",
        "Unibo corrected scan-window dataset": f"Detected frame value range {unibo_min_frame}–{unibo_max_frame}; corrected scan-window segments are aligned to Excel observation windows"
    },
    {
        "aspect": "Bounding box format",
        "Edinburgh manually annotated dataset": "bbox object with x, y, width, height",
        "Unibo corrected scan-window dataset": "bbox-like frame-level coordinates from YOLO/ByteTrack tracking output"
    },
    {
        "aspect": "Behaviour labels",
        "Edinburgh manually annotated dataset": ed_labels,
        "Unibo corrected scan-window dataset": unibo_behaviour_values
    },
    {
        "aspect": "Behaviour granularity",
        "Edinburgh manually annotated dataset": "Manual ground-truth frame descriptors; behaviour forward-propagated between changes according to source-page notes",
        "Unibo corrected scan-window dataset": "Segment/window-level Excel ethogram labels attached to frame-level tracks"
    },
    {
        "aspect": "Identity handling",
        "Edinburgh manually annotated dataset": "Persistent object IDs in ground-truth object list",
        "Unibo corrected scan-window dataset": "ByteTrack IDs available; true pig identity remains conservative / not fully verified"
    },
    {
        "aspect": "Depth information",
        "Edinburgh manually annotated dataset": "Depth video and background_depth are available",
        "Unibo corrected scan-window dataset": "RGB/video based; no depth information"
    },
    {
        "aspect": "Strength for Week 4",
        "Edinburgh manually annotated dataset": "Strong public reference for bbox + tracking + behaviour annotation format",
        "Unibo corrected scan-window dataset": "Primary internal dataset for trajectory, ROI, and representation prototype"
    },
    {
        "aspect": "Main limitation",
        "Edinburgh manually annotated dataset": "Different camera/pen setup; only selected sequences manually ground-truthed",
        "Unibo corrected scan-window dataset": "Labels are scan-window level, identity mapping remains conservative, no snout/skeleton/depth yet"
    },
]

comparison = pd.DataFrame(comparison_rows)
comparison_path = OUT / "edinburgh_vs_unibo_annotation_format_comparison.csv"
comparison.to_csv(comparison_path, index=False)

# ---------------------------------------------------------------------
# Markdown note
# ---------------------------------------------------------------------

note_path = NOTES / "edinburgh_vs_unibo_annotation_format_comparison.md"

with open(note_path, "w") as f:
    f.write("# Edinburgh vs Unibo Annotation Format Comparison\n\n")

    f.write("## Purpose\n\n")
    f.write(
        "This note compares the Edinburgh manually annotated `output.json` format with "
        "the Week 3 corrected Unibo scan-window annotation JSON.\n\n"
    )

    f.write("## Comparison table\n\n")
    f.write(comparison.to_markdown(index=False))
    f.write("\n\n")

    f.write("## Unibo schema summary\n\n")
    f.write(unibo_summary.to_markdown(index=False))
    f.write("\n\n")

    f.write("## Unibo most common keys\n\n")
    f.write(unibo_keys.head(30).to_markdown(index=False))
    f.write("\n\n")

    f.write("## Unibo behaviour/value counts\n\n")
    if len(unibo_behav):
        f.write(unibo_behav.head(50).to_markdown(index=False))
    else:
        f.write("No behaviour/behavior fields were detected by key-name search.")
    f.write("\n\n")

    f.write("## Interpretation\n\n")
    f.write(
        "The Edinburgh dataset provides a clean public reference for manual ground-truth "
        "annotation with persistent object IDs, bounding boxes, and frame-level behaviour descriptors. "
        "The Unibo dataset is more directly connected to the internship task and contains corrected "
        "scan-window alignment, tracking outputs, and behaviour labels, but its labels are coarser "
        "because they are attached at scan-window level rather than fully manual frame-level annotation. "
        "Together, the two datasets are complementary for Week 4 behaviour representation analysis.\n"
    )

print("Saved:")
print(unibo_summary_path)
print(unibo_keys_path)
print(unibo_behav_path)
print(unibo_identity_path)
print(comparison_path)
print(note_path)

print()
print("=== Comparison table ===")
print(comparison.to_string(index=False))

print()
print("=== Unibo summary ===")
print(unibo_summary.to_string(index=False))

print()
print("=== Unibo behaviour values ===")
print(unibo_behav.head(30).to_string(index=False) if len(unibo_behav) else "No behaviour/behavior values found.")
