from pathlib import Path
import json
import pandas as pd


OBS_CSV = Path("Week3_Behaviour_Dataset/outputs/behaviour_annotations/parsed_behaviour_observations.csv")
SEG_JSON = Path("Week3_Behaviour_Dataset/data/scan_window_segments_tentative.json")

OUT_DIR = Path("Week3_Behaviour_Dataset/outputs/scan_window_aligned/behaviour_labels")
OUT_DIR.mkdir(parents=True, exist_ok=True)

with open(SEG_JSON) as f:
    seg_data = json.load(f)

obs = pd.read_csv(OBS_CSV)

rows = []

for seg in seg_data["segments"]:
    segment_id = seg["segment_id"]
    interval_start = seg["excel_interval_start"]
    interval_end = seg["excel_interval_end"]

    subset = obs[
        (obs["interval_start"] == interval_start) &
        (obs["interval_end"] == interval_end)
    ].copy()

    if subset.empty:
        print(f"WARNING: no labels found for {segment_id} {interval_start}-{interval_end}")
        continue

    for _, row in subset.iterrows():
        rows.append({
            "segment_id": segment_id,
            "excel_interval_start": interval_start,
            "excel_interval_end": interval_end,
            "pig_id": row["pig_id"],
            "colour": row["colour"],
            "behaviour_code": row["behaviour_code"],
            "behaviour_label": row["behaviour_label"]
        })

labels = pd.DataFrame(rows)

out_csv = OUT_DIR / "scan_window_behaviour_labels.csv"
out_json = OUT_DIR / "scan_window_behaviour_labels.json"

labels.to_csv(out_csv, index=False)

with open(out_json, "w") as f:
    json.dump(labels.to_dict(orient="records"), f, indent=2)

summary = (
    labels
    .groupby(["segment_id", "excel_interval_start", "excel_interval_end"])
    .agg(
        num_pigs=("pig_id", "count"),
        colours=("colour", lambda x: ", ".join(x.astype(str))),
        behaviour_codes=("behaviour_code", lambda x: ", ".join(x.astype(str))),
        behaviour_labels=("behaviour_label", lambda x: ", ".join(x.astype(str)))
    )
    .reset_index()
)

summary_csv = OUT_DIR / "scan_window_behaviour_label_summary.csv"
summary.to_csv(summary_csv, index=False)

print("Saved:", out_csv)
print("Saved:", out_json)
print("Saved:", summary_csv)
print()
print("=== Labels ===")
print(labels.to_string(index=False))
print()
print("=== Summary ===")
print(summary.to_string(index=False))
