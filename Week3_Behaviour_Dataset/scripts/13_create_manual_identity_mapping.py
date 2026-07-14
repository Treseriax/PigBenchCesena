import json
from pathlib import Path
import pandas as pd


SUMMARY_CSV = Path("Week3_Behaviour_Dataset/outputs/identity_mapping/track_id_summary.csv")
OUT_JSON = Path("Week3_Behaviour_Dataset/outputs/identity_mapping/track_id_to_pig_identity_manual.json")
OUT_CSV = Path("Week3_Behaviour_Dataset/outputs/identity_mapping/track_id_to_pig_identity_manual.csv")

summary = pd.read_csv(SUMMARY_CSV)

# Conservative manual mapping:
# We do not assign exact colour labels unless validated by visual/manual review.
fragment_groups = {
    "fragment_group_left_bar_pig": [8, 16, 19, 24],
    "fragment_group_left_edge_pig": [9, 17],
}

mappings = []

for _, row in summary.iterrows():
    track_id = int(row["track_id"])

    fragment_group = None
    for group_name, ids in fragment_groups.items():
        if track_id in ids:
            fragment_group = group_name

    if track_id in [1, 2, 3, 4, 5, 6]:
        identity_status = "stable_track_colour_validation_needed"
        pig_id = None
        colour = None
        notes = "Stable short-term track in the sample, but exact colour identity is not assigned from montage alone."
    elif fragment_group is not None:
        identity_status = "fragmented_track_colour_validation_needed"
        pig_id = None
        colour = None
        notes = f"Likely belongs to {fragment_group}; exact colour identity requires manual validation."
    else:
        identity_status = "uncertain_track"
        pig_id = None
        colour = None
        notes = "Short or uncertain track; keep unassigned in prototype."

    mappings.append({
        "video_id": "UniboVid2_sample",
        "track_id": track_id,
        "pig_id": pig_id,
        "colour": colour,
        "identity_status": identity_status,
        "fragment_group": fragment_group,
        "first_frame": int(row["first_frame"]),
        "last_frame": int(row["last_frame"]),
        "num_frames": int(row["num_frames"]),
        "mean_score": round(float(row["mean_score"]), 4),
        "notes": notes
    })

with open(OUT_JSON, "w") as f:
    json.dump({
        "video_id": "UniboVid2_sample",
        "mapping_status": "manual_review_template",
        "notes": "Exact pig colour identities were not assigned automatically because colour markers are partially occluded and tracker fragmentation is present.",
        "mappings": mappings
    }, f, indent=2)

pd.DataFrame(mappings).to_csv(OUT_CSV, index=False)

print("Saved:", OUT_JSON)
print("Saved:", OUT_CSV)
print()
print(pd.DataFrame(mappings).to_string(index=False))
