from pathlib import Path
import tarfile
import json
from collections import Counter
import pandas as pd


ROOT = Path("Week4_5_Behaviour_Representation")
EDIN = ROOT / "data/public_datasets/edinburgh"
TAR_PATH = EDIN / "annotations/annotated.tar"
OUT = ROOT / "outputs/dataset_comparison"
NOTES = ROOT / "notes/public_dataset_notes"

OUT.mkdir(parents=True, exist_ok=True)
NOTES.mkdir(parents=True, exist_ok=True)

if not TAR_PATH.exists():
    raise FileNotFoundError(TAR_PATH)

# ---------------------------------------------------------------------
# Find output.json files inside annotated.tar
# ---------------------------------------------------------------------

with tarfile.open(TAR_PATH, "r") as tar:
    members = tar.getnames()
    output_json_members = [m for m in members if m.endswith("output.json")]

paths_txt = OUT / "edinburgh_annotated_output_json_paths.txt"
paths_txt.write_text("\n".join(output_json_members) + "\n")

if not output_json_members:
    raise RuntimeError("No output.json files found in annotated.tar")

sample_member = output_json_members[0]

# ---------------------------------------------------------------------
# Load one sample output.json directly from tar, without extracting all tar
# ---------------------------------------------------------------------

with tarfile.open(TAR_PATH, "r") as tar:
    f = tar.extractfile(sample_member)
    if f is None:
        raise RuntimeError(f"Could not read {sample_member}")
    data = json.load(f)

# ---------------------------------------------------------------------
# Recursive inspection helpers
# ---------------------------------------------------------------------

key_counter = Counter()
behaviour_counter = Counter()
bbox_count = 0
frame_numbers = []
visible_counter = Counter()
ground_truth_counter = Counter()
list_lengths = []
dict_count = 0
list_count = 0

bbox_widths = []
bbox_heights = []
bbox_xs = []
bbox_ys = []


def walk(obj, path="root"):
    global bbox_count, dict_count, list_count

    if isinstance(obj, dict):
        dict_count += 1

        for k, v in obj.items():
            key_counter[k] += 1

        if "behaviour" in obj:
            behaviour_counter[str(obj.get("behaviour"))] += 1

        if "visible" in obj:
            visible_counter[str(obj.get("visible"))] += 1

        if "isGroundTruth" in obj:
            ground_truth_counter[str(obj.get("isGroundTruth"))] += 1

        if "frameNumber" in obj:
            try:
                frame_numbers.append(int(obj["frameNumber"]))
            except Exception:
                pass

        if "bbox" in obj and isinstance(obj["bbox"], dict):
            bbox = obj["bbox"]
            bbox_count += 1

            for source_key, target_list in [
                ("x", bbox_xs),
                ("y", bbox_ys),
                ("width", bbox_widths),
                ("height", bbox_heights),
            ]:
                if source_key in bbox:
                    try:
                        target_list.append(float(bbox[source_key]))
                    except Exception:
                        pass

        for k, v in obj.items():
            walk(v, f"{path}.{k}")

    elif isinstance(obj, list):
        list_count += 1
        list_lengths.append({
            "path": path,
            "length": len(obj)
        })

        for i, item in enumerate(obj):
            # Avoid extremely long path strings
            walk(item, f"{path}[{i}]")

walk(data)

# ---------------------------------------------------------------------
# Build summaries
# ---------------------------------------------------------------------

top_level_type = type(data).__name__
top_level_keys = list(data.keys()) if isinstance(data, dict) else []

schema_rows = [
    {"field": "sample_member", "value": sample_member},
    {"field": "top_level_type", "value": top_level_type},
    {"field": "top_level_keys", "value": ", ".join(top_level_keys)},
    {"field": "dict_count_recursive", "value": dict_count},
    {"field": "list_count_recursive", "value": list_count},
    {"field": "bbox_count", "value": bbox_count},
    {"field": "unique_behaviour_labels", "value": len(behaviour_counter)},
    {"field": "min_frameNumber", "value": min(frame_numbers) if frame_numbers else ""},
    {"field": "max_frameNumber", "value": max(frame_numbers) if frame_numbers else ""},
    {"field": "frameNumber_count", "value": len(frame_numbers)},
]

schema_df = pd.DataFrame(schema_rows)
schema_csv = OUT / "edinburgh_sample_output_json_schema_summary.csv"
schema_df.to_csv(schema_csv, index=False)

behaviour_df = pd.DataFrame(
    [{"behaviour": k, "count": v} for k, v in behaviour_counter.most_common()]
)
behaviour_csv = OUT / "edinburgh_sample_behaviour_counts.csv"
behaviour_df.to_csv(behaviour_csv, index=False)

key_df = pd.DataFrame(
    [{"key": k, "count": v} for k, v in key_counter.most_common()]
)
key_csv = OUT / "edinburgh_sample_json_key_counts.csv"
key_df.to_csv(key_csv, index=False)

list_df = pd.DataFrame(list_lengths).sort_values("length", ascending=False)
list_csv = OUT / "edinburgh_sample_json_list_lengths.csv"
list_df.to_csv(list_csv, index=False)

bbox_summary = pd.DataFrame([
    {
        "bbox_count": bbox_count,
        "mean_width": sum(bbox_widths) / len(bbox_widths) if bbox_widths else "",
        "mean_height": sum(bbox_heights) / len(bbox_heights) if bbox_heights else "",
        "min_x": min(bbox_xs) if bbox_xs else "",
        "max_x": max(bbox_xs) if bbox_xs else "",
        "min_y": min(bbox_ys) if bbox_ys else "",
        "max_y": max(bbox_ys) if bbox_ys else "",
    }
])
bbox_csv = OUT / "edinburgh_sample_bbox_summary.csv"
bbox_summary.to_csv(bbox_csv, index=False)

# Save compact raw preview
preview_txt = OUT / "edinburgh_sample_output_json_preview.txt"
preview = json.dumps(data, indent=2)[:12000]
preview_txt.write_text(preview)

# Markdown note
note_path = NOTES / "edinburgh_output_json_schema_inspection.md"

with open(note_path, "w") as f:
    f.write("# Edinburgh `output.json` Schema Inspection\n\n")

    f.write("## Purpose\n\n")
    f.write(
        "This step inspects one manually ground-truthed Edinburgh `output.json` file "
        "directly from `annotated.tar`, without extracting the full archive.\n\n"
    )

    f.write("## Sample inspected\n\n")
    f.write(f"- `{sample_member}`\n\n")

    f.write("## Schema summary\n\n")
    f.write(schema_df.to_markdown(index=False))
    f.write("\n\n")

    f.write("## Behaviour label counts\n\n")
    if len(behaviour_df):
        f.write(behaviour_df.to_markdown(index=False))
    else:
        f.write("No `behaviour` fields found by recursive inspection.")
    f.write("\n\n")

    f.write("## Most common JSON keys\n\n")
    f.write(key_df.head(30).to_markdown(index=False))
    f.write("\n\n")

    f.write("## Largest lists in JSON\n\n")
    f.write(list_df.head(20).to_markdown(index=False))
    f.write("\n\n")

    f.write("## Bounding box summary\n\n")
    f.write(bbox_summary.to_markdown(index=False))
    f.write("\n\n")

    f.write("## Interpretation\n\n")
    f.write(
        "This inspection confirms the internal structure of the Edinburgh manually annotated "
        "ground-truth file. The extracted behaviour labels, bounding box fields, frame numbers, "
        "visibility flags, and ground-truth flags can be compared directly with our Unibo "
        "Week 3 corrected annotation JSON.\n"
    )

print("Saved:")
print(paths_txt)
print(schema_csv)
print(behaviour_csv)
print(key_csv)
print(list_csv)
print(bbox_csv)
print(preview_txt)
print(note_path)

print()
print("=== sample member ===")
print(sample_member)

print()
print("=== schema summary ===")
print(schema_df.to_string(index=False))

print()
print("=== behaviour counts ===")
print(behaviour_df.to_string(index=False) if len(behaviour_df) else "No behaviour fields found.")

print()
print("=== top keys ===")
print(key_df.head(30).to_string(index=False))

print()
print("=== largest lists ===")
print(list_df.head(20).to_string(index=False))
