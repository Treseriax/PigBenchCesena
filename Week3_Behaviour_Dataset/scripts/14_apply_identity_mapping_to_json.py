import json
from pathlib import Path


ANNOTATION_JSON = Path("Week3_Behaviour_Dataset/outputs/json/UniboVid2_sample_behaviour_annotations.json")
MAPPING_JSON = Path("Week3_Behaviour_Dataset/outputs/identity_mapping/track_id_to_pig_identity_manual.json")
OUT_JSON = Path("Week3_Behaviour_Dataset/outputs/json/UniboVid2_sample_behaviour_annotations_with_identity.json")


with open(ANNOTATION_JSON) as f:
    annotation = json.load(f)

with open(MAPPING_JSON) as f:
    mapping_data = json.load(f)

mapping_by_track_id = {
    int(item["track_id"]): item
    for item in mapping_data["mappings"]
}

updated_count = 0
missing_mapping_count = 0

for frame in annotation["frames"]:
    for pig in frame["pigs"]:
        track_id = int(pig["track_id"])
        mapping = mapping_by_track_id.get(track_id)

        if mapping is None:
            pig["identity_status"] = "missing_mapping"
            pig["fragment_group"] = None
            pig["identity_note"] = "No manual identity mapping entry found for this track_id."
            missing_mapping_count += 1
            continue

        pig["pig_id"] = mapping["pig_id"]
        pig["colour"] = mapping["colour"]
        pig["identity_status"] = mapping["identity_status"]
        pig["fragment_group"] = mapping["fragment_group"]
        pig["identity_note"] = mapping["notes"]

        updated_count += 1

annotation["video"]["identity_mapping_file"] = str(MAPPING_JSON)
annotation["video"]["identity_mapping_status"] = "manual_review_template_applied"
annotation["video"]["identity_mapping_note"] = (
    "Exact colour identities are not assigned automatically. "
    "Stable, fragmented, and uncertain track statuses are included for manual validation."
)

with open(OUT_JSON, "w") as f:
    json.dump(annotation, f, indent=2)

print("Input annotation:", ANNOTATION_JSON)
print("Input mapping:", MAPPING_JSON)
print("Output:", OUT_JSON)
print("Updated pig entries:", updated_count)
print("Missing mapping entries:", missing_mapping_count)
print("Frames:", len(annotation["frames"]))

print()
print("Example first frame with identity fields:")
print(json.dumps(annotation["frames"][0], indent=2)[:3500])
