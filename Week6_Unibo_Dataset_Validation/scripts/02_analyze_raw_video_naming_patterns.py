from pathlib import Path
import re
import pandas as pd


ROOT = Path("Week6_Unibo_Dataset_Validation")
INVENTORY = ROOT / "outputs/dataset_statistics/unibo_candidate_raw_work_videos.csv"
OUT = ROOT / "outputs/dataset_statistics"
NOTES = ROOT / "notes"

OUT.mkdir(parents=True, exist_ok=True)
NOTES.mkdir(parents=True, exist_ok=True)

df = pd.read_csv(INVENTORY)

def parse_video_name(path_str):
    p = Path(path_str)
    name = p.name
    stem = p.stem

    result = {
        "filename": name,
        "naming_family": "unknown",
        "parsed_camera_id": "",
        "parsed_pen_or_box_id": "",
        "parsed_date_token": "",
        "parsed_start_time": "",
        "parsed_end_time": "",
        "parsed_raw_id": stem,
    }

    # Pattern 1: TLC1 B1 1000-1100.mp4 or TLC 1 -B1 0700-0800.mp4
    tlc = re.search(
        r"TLC\s*([0-9]+)\s*[-_ ]*\s*B\s*([0-9]+)\s+([0-9]{3,4})[-_ ]+([0-9]{3,4})",
        stem,
        flags=re.I
    )
    if tlc:
        cam = tlc.group(1)
        box = tlc.group(2)
        start = tlc.group(3).zfill(4)
        end = tlc.group(4).zfill(4)
        result.update({
            "naming_family": "TLC_B_time_range",
            "parsed_camera_id": f"TLC{cam}",
            "parsed_pen_or_box_id": f"B{box}",
            "parsed_start_time": f"{start[:2]}:{start[2:]}",
            "parsed_end_time": f"{end[:2]}:{end[2:]}",
        })
        return result

    # Pattern 2: c0001210722100000
    # Conservative parse:
    # c + camera-like digits + date/time-like token.
    # We do NOT assume exact semantics yet, but we expose possible camera and timestamp tokens.
    cpat = re.search(r"^c([0-9]{4})([0-9]{6})([0-9]{6})$", stem, flags=re.I)
    if cpat:
        cam_token = cpat.group(1)
        date_token = cpat.group(2)
        time_token = cpat.group(3)
        result.update({
            "naming_family": "c_token_datetime_like",
            "parsed_camera_id": f"c{cam_token}",
            "parsed_date_token": date_token,
            "parsed_start_time": f"{time_token[:2]}:{time_token[2:4]}:{time_token[4:6]}",
        })
        return result

    # Fallback for c... files with unexpected length
    c_any = re.search(r"^c([0-9]+)$", stem, flags=re.I)
    if c_any:
        token = c_any.group(1)
        result.update({
            "naming_family": "c_unknown_token",
            "parsed_raw_id": token,
        })
        return result

    return result

parsed_rows = []
for _, row in df.iterrows():
    parsed = parse_video_name(row["absolute_path"])
    merged = row.to_dict()
    merged.update(parsed)
    parsed_rows.append(merged)

parsed_df = pd.DataFrame(parsed_rows)

parsed_path = OUT / "unibo_raw_video_naming_pattern_analysis.csv"
parsed_df.to_csv(parsed_path, index=False)

family_summary = (
    parsed_df.groupby("naming_family")
    .agg(
        video_count=("absolute_path", "count"),
        total_size_gb=("size_bytes", lambda x: round(x.sum() / (1024**3), 3)),
        mean_duration_sec=("duration_sec", "mean"),
    )
    .reset_index()
    .sort_values("video_count", ascending=False)
)

family_summary_path = OUT / "unibo_raw_video_naming_family_summary.csv"
family_summary.to_csv(family_summary_path, index=False)

camera_summary = (
    parsed_df.groupby(["naming_family", "parsed_camera_id"])
    .size()
    .reset_index(name="video_count")
    .sort_values(["naming_family", "parsed_camera_id"])
)

camera_summary_path = OUT / "unibo_raw_video_camera_token_summary.csv"
camera_summary.to_csv(camera_summary_path, index=False)

pen_summary = (
    parsed_df.groupby(["naming_family", "parsed_pen_or_box_id"])
    .size()
    .reset_index(name="video_count")
    .sort_values(["naming_family", "parsed_pen_or_box_id"])
)

pen_summary_path = OUT / "unibo_raw_video_pen_token_summary.csv"
pen_summary.to_csv(pen_summary_path, index=False)

note_path = NOTES / "unibo_raw_video_naming_pattern_analysis.md"

with open(note_path, "w") as f:
    f.write("# Unibo Raw Video Naming Pattern Analysis\n\n")

    f.write("## Purpose\n\n")
    f.write(
        "This note analyses the naming conventions of the 84 raw MP4 videos found under `/work/pig/datasets/Unibo`. "
        "The goal is to infer possible camera, pen/crate, date, and time information before building the unified ground-truth table.\n\n"
    )

    f.write("## Naming family summary\n\n")
    f.write(family_summary.to_markdown(index=False))
    f.write("\n\n")

    f.write("## Camera token summary\n\n")
    f.write(camera_summary.to_markdown(index=False))
    f.write("\n\n")

    f.write("## Pen/box token summary\n\n")
    f.write(pen_summary.to_markdown(index=False))
    f.write("\n\n")

    f.write("## First parsed videos\n\n")
    cols = [
        "filename",
        "naming_family",
        "parsed_camera_id",
        "parsed_pen_or_box_id",
        "parsed_date_token",
        "parsed_start_time",
        "parsed_end_time",
        "size_mb",
        "duration_sec",
    ]
    f.write(parsed_df[cols].head(40).to_markdown(index=False))
    f.write("\n\n")

    f.write("## Interpretation\n\n")
    f.write(
        "The TLC-style filenames appear to explicitly encode a camera token, a B/box token, and a time range. "
        "The c-token filenames appear to encode camera-like and date/time-like tokens, but their exact semantics should be confirmed using annotation files or supervisor metadata before being treated as ground truth.\n"
    )

print("Saved:")
print(parsed_path)
print(family_summary_path)
print(camera_summary_path)
print(pen_summary_path)
print(note_path)

print()
print("=== Naming family summary ===")
print(family_summary.to_string(index=False))

print()
print("=== Camera token summary ===")
print(camera_summary.to_string(index=False))

print()
print("=== First parsed videos ===")
cols = ["filename", "naming_family", "parsed_camera_id", "parsed_pen_or_box_id", "parsed_date_token", "parsed_start_time", "parsed_end_time"]
print(parsed_df[cols].head(40).to_string(index=False))
