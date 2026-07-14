import json
from pathlib import Path
import pandas as pd


VIDEO_MAP = Path("Week3_Behaviour_Dataset/data/video_time_mapping.json")
OBS_CSV = Path("Week3_Behaviour_Dataset/outputs/behaviour_annotations/parsed_behaviour_observations.csv")
OUT_DIR = Path("Week3_Behaviour_Dataset/outputs/behaviour_annotations")
OUT_DIR.mkdir(parents=True, exist_ok=True)


def time_to_sec(t):
    h, m, s = map(int, str(t).split(":"))
    return h * 3600 + m * 60 + s


def sec_to_time(sec):
    h = sec // 3600
    m = (sec % 3600) // 60
    s = sec % 60
    return f"{h:02d}:{m:02d}:{s:02d}"


with open(VIDEO_MAP) as f:
    video_map = json.load(f)

obs = pd.read_csv(OBS_CSV)

results = []

for video in video_map["videos"]:
    video_id = video["video_id"]
    v_start = time_to_sec(video["estimated_start_time"])
    v_end = time_to_sec(video["estimated_end_time"])

    for _, row in obs.iterrows():
        interval_start = time_to_sec(row["interval_start"])
        interval_end = time_to_sec(row["interval_end"])

        overlap_start = max(v_start, interval_start)
        overlap_end = min(v_end, interval_end)
        overlap_sec = max(0, overlap_end - overlap_start)

        if overlap_sec > 0:
            results.append({
                "video_id": video_id,
                "video_start": video["estimated_start_time"],
                "video_end": video["estimated_end_time"],
                "pig_id": row["pig_id"],
                "colour": row["colour"],
                "behaviour_code": row["behaviour_code"],
                "behaviour_label": row["behaviour_label"],
                "observation_start": row["interval_start"],
                "observation_end": row["interval_end"],
                "overlap_start": sec_to_time(overlap_start),
                "overlap_end": sec_to_time(overlap_end),
                "overlap_sec": overlap_sec
            })

out_df = pd.DataFrame(results)
out_csv = OUT_DIR / "video_excel_overlaps.csv"
out_df.to_csv(out_csv, index=False)

print("Saved:", out_csv)
print("Number of overlapping records:", len(out_df))

if len(out_df) > 0:
    print(out_df.to_string(index=False))
else:
    print()
    print("No exact overlap found between available MP4 clips and Excel scan-sampling intervals.")
    print("This means the clips are still useful for interface/tracking prototype,")
    print("but exact behaviour labels require clips from scan windows such as 09:00:00–09:00:10 or 09:10:00–09:10:10.")

print()
print("Video time windows:")
for video in video_map["videos"]:
    print(f"- {video['video_id']}: {video['estimated_start_time']} -> {video['estimated_end_time']}")

print()
print("Available 09:00–10:00 scan windows:")
subset = obs[obs["hour_start"] == "09:00"][["interval_start", "interval_end"]].drop_duplicates()
print(subset.to_string(index=False))
