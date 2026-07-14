from pathlib import Path
import os
import re
import json
import subprocess
from datetime import datetime
import pandas as pd


PROJECT_ROOT = Path.home() / "PigBench"
W6 = PROJECT_ROOT / "Week6_Unibo_Dataset_Validation"
OUT = W6 / "outputs/dataset_statistics"
NOTES = W6 / "notes"

OUT.mkdir(parents=True, exist_ok=True)
NOTES.mkdir(parents=True, exist_ok=True)

SEARCH_ROOTS = [
    PROJECT_ROOT,
]

EXCLUDE_PARTS = {
    ".git",
    "__pycache__",
    "Week4_5_Final_Package",
    "final_outputs/Week4_5_Final_Package",
}


def is_excluded(path: Path) -> bool:
    s = str(path)
    return any(part in s for part in EXCLUDE_PARTS)


def infer_metadata_from_path(path: Path):
    s = str(path)
    name = path.name

    # Flexible patterns because we do not yet know the exact Unibo naming convention.
    camera_match = re.search(r"(?:cam|camera|c)[_\- ]?(\d+)", s, flags=re.I)
    crate_match = re.search(r"(?:crate|pen|box|stall)[_\- ]?(\d+)", s, flags=re.I)
    scan_match = re.search(r"scan[_\- ]?(\d{2})[_\-: ]?(\d{2})", s, flags=re.I)
    date_match = re.search(r"(20\d{2}[-_]\d{2}[-_]\d{2}|\d{8})", s)
    session_match = re.search(r"(?:session|sess|day|d)[_\- ]?(\d+)", s, flags=re.I)

    return {
        "inferred_camera_id": camera_match.group(1) if camera_match else "",
        "inferred_crate_or_pen_id": crate_match.group(1) if crate_match else "",
        "inferred_scan_window": f"{scan_match.group(1)}:{scan_match.group(2)}" if scan_match else "",
        "inferred_date": date_match.group(1) if date_match else "",
        "inferred_session": session_match.group(1) if session_match else "",
    }


def get_video_metadata_ffprobe(path: Path):
    try:
        cmd = [
            "ffprobe",
            "-v", "error",
            "-select_streams", "v:0",
            "-show_entries", "stream=width,height,r_frame_rate,avg_frame_rate,nb_frames,duration",
            "-of", "json",
            str(path),
        ]
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=20)
        if result.returncode != 0:
            return {}
        data = json.loads(result.stdout)
        streams = data.get("streams", [])
        if not streams:
            return {}
        st = streams[0]

        def parse_rate(rate):
            if not rate or rate == "0/0":
                return None
            if "/" in rate:
                a, b = rate.split("/")
                try:
                    return float(a) / float(b)
                except Exception:
                    return None
            try:
                return float(rate)
            except Exception:
                return None

        fps = parse_rate(st.get("avg_frame_rate")) or parse_rate(st.get("r_frame_rate"))

        return {
            "width": st.get("width", ""),
            "height": st.get("height", ""),
            "fps": fps if fps is not None else "",
            "frame_count": st.get("nb_frames", ""),
            "duration_sec": st.get("duration", ""),
            "metadata_source": "ffprobe",
        }
    except Exception:
        return {}


def get_video_metadata_cv2(path: Path):
    try:
        import cv2
        cap = cv2.VideoCapture(str(path))
        if not cap.isOpened():
            return {}
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = float(cap.get(cv2.CAP_PROP_FPS))
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        duration_sec = frame_count / fps if fps and fps > 0 else ""
        cap.release()
        return {
            "width": width,
            "height": height,
            "fps": fps,
            "frame_count": frame_count,
            "duration_sec": duration_sec,
            "metadata_source": "cv2",
        }
    except Exception:
        return {}


def get_video_metadata(path: Path):
    meta = get_video_metadata_ffprobe(path)
    if meta:
        return meta
    meta = get_video_metadata_cv2(path)
    if meta:
        return meta
    return {
        "width": "",
        "height": "",
        "fps": "",
        "frame_count": "",
        "duration_sec": "",
        "metadata_source": "unavailable",
    }


rows = []

for root in SEARCH_ROOTS:
    if not root.exists():
        continue

    for path in sorted(root.rglob("*.mp4")):
        if is_excluded(path):
            continue

        try:
            stat = path.stat()
        except Exception:
            continue

        inferred = infer_metadata_from_path(path)
        video_meta = get_video_metadata(path)

        try:
            rel_project = str(path.relative_to(PROJECT_ROOT))
        except Exception:
            rel_project = str(path)

        rows.append({
            "absolute_path": str(path),
            "relative_to_project": rel_project,
            "filename": path.name,
            "parent_folder": str(path.parent),
            "size_bytes": stat.st_size,
            "size_mb": round(stat.st_size / (1024 * 1024), 3),
            "modified_time": datetime.fromtimestamp(stat.st_mtime).isoformat(timespec="seconds"),
            **inferred,
            **video_meta,
        })

df = pd.DataFrame(rows)

inventory_path = OUT / "unibo_mp4_video_inventory.csv"
df.to_csv(inventory_path, index=False)

summary_rows = []

summary_rows.append({"metric": "total_mp4_videos_found", "value": len(df)})

if len(df):
    summary_rows.append({"metric": "total_size_gb", "value": round(df["size_bytes"].sum() / (1024 ** 3), 3)})
    summary_rows.append({"metric": "unique_parent_folders", "value": df["parent_folder"].nunique()})
    summary_rows.append({"metric": "videos_with_inferred_camera_id", "value": int((df["inferred_camera_id"].astype(str) != "").sum())})
    summary_rows.append({"metric": "videos_with_inferred_crate_or_pen_id", "value": int((df["inferred_crate_or_pen_id"].astype(str) != "").sum())})
    summary_rows.append({"metric": "videos_with_inferred_scan_window", "value": int((df["inferred_scan_window"].astype(str) != "").sum())})
    summary_rows.append({"metric": "videos_with_metadata", "value": int((df["metadata_source"].astype(str) != "unavailable").sum())})

summary = pd.DataFrame(summary_rows)
summary_path = OUT / "unibo_mp4_video_inventory_summary.csv"
summary.to_csv(summary_path, index=False)

folder_counts = (
    df.groupby("parent_folder")
    .size()
    .reset_index(name="video_count")
    .sort_values("video_count", ascending=False)
    if len(df) else pd.DataFrame(columns=["parent_folder", "video_count"])
)

folder_counts_path = OUT / "unibo_mp4_parent_folder_counts.csv"
folder_counts.to_csv(folder_counts_path, index=False)

camera_counts = (
    df.groupby("inferred_camera_id")
    .size()
    .reset_index(name="video_count")
    .sort_values("video_count", ascending=False)
    if len(df) else pd.DataFrame(columns=["inferred_camera_id", "video_count"])
)

camera_counts_path = OUT / "unibo_mp4_inferred_camera_counts.csv"
camera_counts.to_csv(camera_counts_path, index=False)

pen_counts = (
    df.groupby("inferred_crate_or_pen_id")
    .size()
    .reset_index(name="video_count")
    .sort_values("video_count", ascending=False)
    if len(df) else pd.DataFrame(columns=["inferred_crate_or_pen_id", "video_count"])
)

pen_counts_path = OUT / "unibo_mp4_inferred_pen_counts.csv"
pen_counts.to_csv(pen_counts_path, index=False)

note_path = NOTES / "week6_mp4_video_inventory_notes.md"

with open(note_path, "w") as f:
    f.write("# Week 6 MP4 Video Inventory Notes\n\n")
    f.write("## Purpose\n\n")
    f.write(
        "This note documents the first Week 6 dataset inventory step. "
        "The goal is to identify all MP4 videos currently available under the PigBench project folder and extract basic metadata for later Unibo dataset analysis.\n\n"
    )

    f.write("## Summary\n\n")
    f.write(summary.to_markdown(index=False))
    f.write("\n\n")

    f.write("## Parent folder counts\n\n")
    if len(folder_counts):
        f.write(folder_counts.head(30).to_markdown(index=False))
    else:
        f.write("No MP4 videos found.")
    f.write("\n\n")

    f.write("## Interpretation\n\n")
    f.write(
        "This inventory is the first step before extracting ground-truth labels, building a unified annotation table, "
        "and proposing train/test splits. If crate, pen, camera, or session IDs cannot be inferred from filenames or folders, "
        "they will need to be mapped manually or extracted from associated annotation/metadata files.\n"
    )

print("Saved:")
print(inventory_path)
print(summary_path)
print(folder_counts_path)
print(camera_counts_path)
print(pen_counts_path)
print(note_path)

print()
print("=== MP4 inventory summary ===")
print(summary.to_string(index=False))

print()
print("=== Top parent folders ===")
print(folder_counts.head(20).to_string(index=False) if len(folder_counts) else "No MP4 videos found.")

print()
print("=== First 20 videos ===")
cols = [
    "relative_to_project",
    "size_mb",
    "width",
    "height",
    "fps",
    "frame_count",
    "duration_sec",
    "inferred_crate_or_pen_id",
    "inferred_camera_id",
    "inferred_scan_window",
]
print(df[cols].head(20).to_string(index=False) if len(df) else "No MP4 videos found.")
