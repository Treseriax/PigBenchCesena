from pathlib import Path
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

# Main correction: include shared server work folders, not only ~/PigBench.
CANDIDATE_ROOTS = [
    Path("/work"),
    Path("/workspace"),
    Path("/data"),
    Path("/datasets"),
    Path("/scratch"),
    PROJECT_ROOT,
]

SEARCH_ROOTS = [p for p in CANDIDATE_ROOTS if p.exists()]

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

    camera_match = re.search(r"(?:cam|camera|c)[_\- ]?(\d+)", s, flags=re.I)
    crate_match = re.search(r"(?:crate|pen|box|stall)[_\- ]?(\d+)", s, flags=re.I)
    scan_match = re.search(r"scan[_\- ]?(\d{2})[_\-: ]?(\d{2})", s, flags=re.I)
    date_match = re.search(r"(20\d{2}[-_]\d{2}[-_]\d{2}|\d{8})", s)
    session_match = re.search(r"(?:session|sess|day|d)[_\- ]?(\d+)", s, flags=re.I)

    # Extra generic IDs from filenames like UniboVid1, pigtrack0028, etc.
    unibo_vid_match = re.search(r"UniboVid[_\- ]?(\d+)", name, flags=re.I)
    pigtrack_match = re.search(r"pigtrack[_\- ]?(\d+)", name, flags=re.I)

    return {
        "inferred_camera_id": camera_match.group(1) if camera_match else "",
        "inferred_crate_or_pen_id": crate_match.group(1) if crate_match else "",
        "inferred_scan_window": f"{scan_match.group(1)}:{scan_match.group(2)}" if scan_match else "",
        "inferred_date": date_match.group(1) if date_match else "",
        "inferred_session": session_match.group(1) if session_match else "",
        "inferred_unibo_video_id": unibo_vid_match.group(1) if unibo_vid_match else "",
        "inferred_pigtrack_id": pigtrack_match.group(1) if pigtrack_match else "",
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
        width = int(cap.get(3))
        height = int(cap.get(4))
        fps = float(cap.get(5))
        frame_count = int(cap.get(7))
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
            rel_to_project = str(path.relative_to(PROJECT_ROOT))
        except Exception:
            rel_to_project = ""

        rows.append({
            "absolute_path": str(path),
            "search_root": str(root),
            "relative_to_project": rel_to_project,
            "filename": path.name,
            "parent_folder": str(path.parent),
            "size_bytes": stat.st_size,
            "size_mb": round(stat.st_size / (1024 * 1024), 3),
            "modified_time": datetime.fromtimestamp(stat.st_mtime).isoformat(timespec="seconds"),
            **inferred,
            **video_meta,
        })

df = pd.DataFrame(rows)

inventory_path = OUT / "unibo_all_server_mp4_video_inventory.csv"
df.to_csv(inventory_path, index=False)

summary_rows = [
    {"metric": "search_roots", "value": ", ".join(str(p) for p in SEARCH_ROOTS)},
    {"metric": "total_mp4_videos_found", "value": len(df)},
]

if len(df):
    summary_rows += [
        {"metric": "total_size_gb", "value": round(df["size_bytes"].sum() / (1024 ** 3), 3)},
        {"metric": "unique_parent_folders", "value": df["parent_folder"].nunique()},
        {"metric": "videos_under_work", "value": int(df["absolute_path"].str.startswith("/work").sum())},
        {"metric": "videos_under_home_project", "value": int(df["absolute_path"].str.startswith(str(PROJECT_ROOT)).sum())},
        {"metric": "videos_with_metadata", "value": int((df["metadata_source"].astype(str) != "unavailable").sum())},
        {"metric": "videos_with_inferred_camera_id", "value": int((df["inferred_camera_id"].fillna("").astype(str) != "").sum())},
        {"metric": "videos_with_inferred_crate_or_pen_id", "value": int((df["inferred_crate_or_pen_id"].fillna("").astype(str) != "").sum())},
        {"metric": "videos_with_inferred_scan_window", "value": int((df["inferred_scan_window"].fillna("").astype(str) != "").sum())},
    ]

summary = pd.DataFrame(summary_rows)
summary_path = OUT / "unibo_all_server_mp4_video_inventory_summary.csv"
summary.to_csv(summary_path, index=False)

folder_counts = (
    df.groupby("parent_folder")
    .size()
    .reset_index(name="video_count")
    .sort_values("video_count", ascending=False)
    if len(df) else pd.DataFrame(columns=["parent_folder", "video_count"])
)
folder_counts_path = OUT / "unibo_all_server_mp4_parent_folder_counts.csv"
folder_counts.to_csv(folder_counts_path, index=False)

root_counts = (
    df.groupby("search_root")
    .size()
    .reset_index(name="video_count")
    .sort_values("video_count", ascending=False)
    if len(df) else pd.DataFrame(columns=["search_root", "video_count"])
)
root_counts_path = OUT / "unibo_all_server_mp4_search_root_counts.csv"
root_counts.to_csv(root_counts_path, index=False)

# Candidate raw dataset videos: prefer /work and exclude obvious generated output/demo videos.
if len(df):
    generated_keywords = [
        "tracking",
        "detections",
        "visualization",
        "final_outputs",
        "outputs/",
        "Week2_Tracking_Study",
        "Week3_Behaviour_Dataset",
        "Week4_5_Behaviour_Representation",
    ]

    def looks_generated(row):
        s = (str(row["absolute_path"]) + " " + str(row["filename"])).lower()
        return any(k.lower() in s for k in generated_keywords)

    df["looks_generated_output"] = df.apply(looks_generated, axis=1)
    candidates = df[(df["absolute_path"].str.startswith("/work")) & (~df["looks_generated_output"])].copy()
else:
    candidates = pd.DataFrame()

candidate_path = OUT / "unibo_candidate_raw_work_videos.csv"
candidates.to_csv(candidate_path, index=False)

note_path = NOTES / "week6_all_server_mp4_video_inventory_notes.md"
with open(note_path, "w") as f:
    f.write("# Week 6 All-Server MP4 Video Inventory Notes\n\n")
    f.write("## Purpose\n\n")
    f.write(
        "This corrected inventory scans shared server work folders in addition to the user's home PigBench folder. "
        "The previous inventory only scanned ~/PigBench and therefore mostly captured generated demo/tracking videos.\n\n"
    )

    f.write("## Summary\n\n")
    f.write(summary.to_markdown(index=False))
    f.write("\n\n")

    f.write("## Search root counts\n\n")
    f.write(root_counts.to_markdown(index=False) if len(root_counts) else "No MP4 videos found.")
    f.write("\n\n")

    f.write("## Top parent folders\n\n")
    f.write(folder_counts.head(40).to_markdown(index=False) if len(folder_counts) else "No MP4 videos found.")
    f.write("\n\n")

    f.write("## Candidate raw work videos\n\n")
    if len(candidates):
        f.write(candidates[["absolute_path", "size_mb", "width", "height", "fps", "frame_count", "duration_sec"]].head(50).to_markdown(index=False))
    else:
        f.write("No candidate raw work videos were identified automatically. The folder naming convention may require manual selection.")
    f.write("\n\n")

    f.write("## Interpretation\n\n")
    f.write(
        "For Week 6, the important videos are likely the raw MP4 files under the shared work folder, not generated visualizations. "
        "The next step is to inspect the candidate raw work videos and identify which ones correspond to Unibo crates/pens, cameras, and annotation files.\n"
    )

print("Saved:")
print(inventory_path)
print(summary_path)
print(folder_counts_path)
print(root_counts_path)
print(candidate_path)
print(note_path)

print()
print("=== Corrected all-server MP4 summary ===")
print(summary.to_string(index=False))

print()
print("=== Search root counts ===")
print(root_counts.to_string(index=False) if len(root_counts) else "No MP4 videos found.")

print()
print("=== Top parent folders ===")
print(folder_counts.head(30).to_string(index=False) if len(folder_counts) else "No MP4 videos found.")

print()
print("=== Candidate raw work videos first 30 ===")
cols = ["absolute_path", "size_mb", "width", "height", "fps", "frame_count", "duration_sec"]
print(candidates[cols].head(30).to_string(index=False) if len(candidates) else "No candidate raw work videos identified automatically.")
