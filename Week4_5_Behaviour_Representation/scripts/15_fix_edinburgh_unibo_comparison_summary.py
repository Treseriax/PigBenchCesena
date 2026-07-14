from pathlib import Path
import json
from collections import Counter
import pandas as pd


ROOT = Path("Week4_5_Behaviour_Representation")
OUT = ROOT / "outputs/dataset_comparison"
NOTES = ROOT / "notes/public_dataset_notes"

UNIBO_JSON = ROOT / "data/unibo_inputs/corrected_scan_window_behaviour_annotations.json"

ED_SCHEMA = OUT / "edinburgh_sample_output_json_schema_summary.csv"
ED_BEHAV = OUT / "edinburgh_sample_behaviour_counts.csv"

ed_schema = pd.read_csv(ED_SCHEMA)
ed_behav = pd.read_csv(ED_BEHAV)

ed_schema_map = dict(zip(ed_schema["field"], ed_schema["value"]))
ed_labels = ", ".join(ed_behav["behaviour"].astype(str).tolist())

with open(UNIBO_JSON) as f:
    data = json.load(f)

segments = data.get("segments", [])

segment_ids = []
frame_count = 0
track_instance_count = 0
track_ids = Counter()
bbox_xyxy_count = 0
bbox_xywh_count = 0
frame_values = []

track_behaviour_values = Counter()
available_segment_label_values = Counter()
identity_status_values = Counter()
assigned_colour_values = Counter()

for seg in segments:
    seg_id = seg.get("segment_id", "unknown_segment")
    segment_ids.append(seg_id)

    # Segment-level available Excel labels
    for item in seg.get("available_excel_behaviour_labels", []):
        if isinstance(item, dict):
            code = item.get("behaviour_code")
            label = item.get("behaviour_label")
            if code is not None:
                available_segment_label_values[str(code)] += 1
            if label is not None:
                available_segment_label_values[str(label)] += 1

    frames = seg.get("frames", [])
    frame_count += len(frames)

    for fr in frames:
        if "frame" in fr:
            frame_values.append(fr["frame"])

        tracks = fr.get("tracks", [])
        for tr in tracks:
            track_instance_count += 1

            if "track_id" in tr:
                track_ids[str(tr["track_id"])] += 1

            if "bbox_xyxy" in tr:
                bbox_xyxy_count += 1

            if "bbox_xywh" in tr:
                bbox_xywh_count += 1

            if "identity_status" in tr:
                identity_status_values[str(tr["identity_status"])] += 1

            if "assigned_colour" in tr:
                assigned_colour_values[str(tr["assigned_colour"])] += 1

            beh = tr.get("behaviour", None)

            if beh is None:
                track_behaviour_values["None"] += 1
            elif isinstance(beh, dict):
                code = beh.get("behaviour_code")
                label = beh.get("behaviour_label")
                if code is not None:
                    track_behaviour_values[str(code)] += 1
                if label is not None:
                    track_behaviour_values[str(label)] += 1
            else:
                track_behaviour_values[str(beh)] += 1

fixed_summary = pd.DataFrame([
    {"field": "top_level_type", "value": type(data).__name__},
    {"field": "top_level_keys", "value": ", ".join(data.keys())},
    {"field": "segment_count", "value": len(segments)},
    {"field": "segment_ids", "value": ", ".join(segment_ids)},
    {"field": "frame_count", "value": frame_count},
    {"field": "track_instance_count", "value": track_instance_count},
    {"field": "unique_track_ids_global", "value": len(track_ids)},
    {"field": "min_frame", "value": min(frame_values) if frame_values else ""},
    {"field": "max_frame", "value": max(frame_values) if frame_values else ""},
    {"field": "bbox_xyxy_count", "value": bbox_xyxy_count},
    {"field": "bbox_xywh_count", "value": bbox_xywh_count},
    {"field": "identity_status_values", "value": ", ".join([f"{k}:{v}" for k, v in identity_status_values.most_common()])},
    {"field": "assigned_colour_values_top", "value": ", ".join([f"{k}:{v}" for k, v in assigned_colour_values.most_common(10)])},
])

fixed_summary_path = OUT / "unibo_annotation_schema_summary_fixed.csv"
fixed_summary.to_csv(fixed_summary_path, index=False)

track_behaviour_df = pd.DataFrame(
    [{"track_behaviour_value": k, "count": v} for k, v in track_behaviour_values.most_common()]
)
track_behaviour_path = OUT / "unibo_track_level_behaviour_counts_fixed.csv"
track_behaviour_df.to_csv(track_behaviour_path, index=False)

segment_label_df = pd.DataFrame(
    [{"segment_available_label_value": k, "count": v} for k, v in available_segment_label_values.most_common()]
)
segment_label_path = OUT / "unibo_segment_available_label_counts_fixed.csv"
segment_label_df.to_csv(segment_label_path, index=False)

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
        "Edinburgh manually annotated dataset": f"dict with keys: {ed_schema_map.get('top_level_keys', '')}",
        "Unibo corrected scan-window dataset": f"dict with keys: {', '.join(data.keys())}"
    },
    {
        "aspect": "Number of sequences / segments",
        "Edinburgh manually annotated dataset": "12 manually ground-truthed sequences in annotated.tar",
        "Unibo corrected scan-window dataset": f"{len(segments)} corrected scan-window segments: {', '.join(segment_ids)}"
    },
    {
        "aspect": "Object / track representation",
        "Edinburgh manually annotated dataset": "objects list; each object corresponds to one manually tracked pig; inspected sample has 8 objects",
        "Unibo corrected scan-window dataset": f"{track_instance_count} frame-level track instances from YOLOv8-s + ByteTrack; {len(track_ids)} unique global track IDs"
    },
    {
        "aspect": "Frame range / temporal representation",
        "Edinburgh manually annotated dataset": f"Ground-truth frameNumber range {ed_schema_map.get('min_frameNumber')}–{ed_schema_map.get('max_frameNumber')}; source page notes GT corresponds to every third raw video frame",
        "Unibo corrected scan-window dataset": f"Frame range {min(frame_values) if frame_values else ''}–{max(frame_values) if frame_values else ''}; segments aligned to Excel scan-window observations"
    },
    {
        "aspect": "Bounding box format",
        "Edinburgh manually annotated dataset": "bbox object with x, y, width, height",
        "Unibo corrected scan-window dataset": f"bbox_xyxy and bbox_xywh stored for each track instance; bbox_xyxy count = {bbox_xyxy_count}, bbox_xywh count = {bbox_xywh_count}"
    },
    {
        "aspect": "Behaviour labels",
        "Edinburgh manually annotated dataset": ed_labels,
        "Unibo corrected scan-window dataset": ", ".join([k for k, _ in available_segment_label_values.most_common()])
    },
    {
        "aspect": "Behaviour granularity",
        "Edinburgh manually annotated dataset": "Manual frame descriptors; behaviour changes stored sparsely and propagated according to source-page notes",
        "Unibo corrected scan-window dataset": "Excel ethogram labels are attached at scan-window/segment level, then linked to frame-level tracks"
    },
    {
        "aspect": "Identity handling",
        "Edinburgh manually annotated dataset": "Persistent object IDs in manually ground-truthed object list",
        "Unibo corrected scan-window dataset": "ByteTrack IDs are available; true pig identity is still conservative / not fully verified"
    },
    {
        "aspect": "Depth information",
        "Edinburgh manually annotated dataset": "Colour video, depth video, masks, and calibration-related files are available",
        "Unibo corrected scan-window dataset": "RGB/video-based tracking outputs; no depth information"
    },
    {
        "aspect": "Best use in Week 4",
        "Edinburgh manually annotated dataset": "Public reference for annotation structure, bbox + track + behaviour representation",
        "Unibo corrected scan-window dataset": "Primary dataset for our trajectory, ROI, and representation feature engineering"
    },
]

comparison = pd.DataFrame(comparison_rows)
comparison_path = OUT / "edinburgh_vs_unibo_annotation_format_comparison_fixed.csv"
comparison.to_csv(comparison_path, index=False)

note_path = NOTES / "edinburgh_vs_unibo_annotation_format_comparison_fixed.md"

with open(note_path, "w") as f:
    f.write("# Edinburgh vs Unibo Annotation Format Comparison — Fixed Summary\n\n")

    f.write("## Why this fixed version was created\n\n")
    f.write(
        "The first automatic comparison used a broad recursive key search. It over-counted Unibo segments "
        "because keys such as `segment_time_sec` were also matched, and it missed bounding boxes because "
        "Unibo stores them as `bbox_xyxy` and `bbox_xywh`. This fixed version uses the known Unibo JSON "
        "structure directly.\n\n"
    )

    f.write("## Fixed comparison table\n\n")
    f.write(comparison.to_markdown(index=False))
    f.write("\n\n")

    f.write("## Fixed Unibo schema summary\n\n")
    f.write(fixed_summary.to_markdown(index=False))
    f.write("\n\n")

    f.write("## Track-level behaviour values\n\n")
    f.write(track_behaviour_df.to_markdown(index=False))
    f.write("\n\n")

    f.write("## Segment-level available behaviour labels\n\n")
    f.write(segment_label_df.to_markdown(index=False))
    f.write("\n\n")

    f.write("## Interpretation\n\n")
    f.write(
        "The Edinburgh dataset is cleaner as a public manual ground-truth reference because it contains "
        "persistent object IDs, sparse frame descriptors, bounding boxes, and behaviour labels in `output.json`. "
        "The Unibo dataset is more project-specific and has corrected scan-window alignment plus detector/tracker "
        "outputs, but its behaviour labels are coarser because they come from scan-window Excel observations rather "
        "than full manual frame-by-frame annotation.\n"
    )

print("Saved:")
print(fixed_summary_path)
print(track_behaviour_path)
print(segment_label_path)
print(comparison_path)
print(note_path)

print()
print("=== Fixed Unibo summary ===")
print(fixed_summary.to_string(index=False))

print()
print("=== Fixed comparison ===")
print(comparison.to_string(index=False))
