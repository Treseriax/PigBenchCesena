from pathlib import Path
import configparser
import pandas as pd


SEQ_DIR = Path("Week3_Behaviour_Dataset/data/sequences/UniboVid2_sample")
TRACK_CSV = Path("Week3_Behaviour_Dataset/outputs/tracking/UniboVid2_sample_bytetrack/UniboVid2_sample_bytetrack_tracks.csv")
OUT_CSV = Path("Week3_Behaviour_Dataset/outputs/tracking/UniboVid2_sample_bytetrack/UniboVid2_sample_bytetrack_tracks_with_time.csv")


def time_to_sec(t):
    h, m, s = map(int, t.split(":"))
    return h * 3600 + m * 60 + s


def sec_to_time_float(sec):
    h = int(sec // 3600)
    m = int((sec % 3600) // 60)
    s = sec % 60
    return f"{h:02d}:{m:02d}:{s:06.3f}"


seqinfo = SEQ_DIR / "seqinfo.ini"

config = configparser.ConfigParser()
config.read(seqinfo)

fps = float(config["Sequence"]["frameRate"])
estimated_start_time = config["Sequence"].get("estimatedStartTime", "09:04:38")
start_sec = time_to_sec(estimated_start_time)

df = pd.read_csv(TRACK_CSV)

# Frame 1 corresponds to elapsed time 0.0 sec
df["time_from_video_start_sec"] = (df["frame"] - 1) / fps
df["estimated_time_sec"] = start_sec + df["time_from_video_start_sec"]
df["estimated_timestamp"] = df["estimated_time_sec"].apply(sec_to_time_float)

df["video_id"] = "UniboVid2_sample"
df["source_video"] = "UniboVid2.mp4"
df["timestamp_source"] = "estimated_from_camera_overlay_and_fps"

OUT_CSV.parent.mkdir(parents=True, exist_ok=True)
df.to_csv(OUT_CSV, index=False)

print("Input:", TRACK_CSV)
print("Output:", OUT_CSV)
print("FPS:", fps)
print("Estimated start time:", estimated_start_time)
print("Rows:", len(df))
print()
print(df.head(20).to_string(index=False))
