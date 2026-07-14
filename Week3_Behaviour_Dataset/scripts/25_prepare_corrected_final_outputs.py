from pathlib import Path
import shutil
import json
import pandas as pd


ROOT = Path("Week3_Behaviour_Dataset")
OUT = ROOT / "final_outputs_corrected_scan_windows"

subdirs = [
    "json",
    "csv",
    "figures",
    "videos",
    "scripts",
    "notes"
]

for sub in subdirs:
    (OUT / sub).mkdir(parents=True, exist_ok=True)


def copy_file(src, dst_dir, new_name=None):
    src = Path(src)
    dst_dir = Path(dst_dir)
    if not src.exists():
        print("MISSING:", src)
        return None

    dst = dst_dir / (new_name if new_name else src.name)
    shutil.copy2(src, dst)
    print("COPIED:", src, "->", dst)
    return dst


# JSON outputs
copy_file(
    ROOT / "outputs/scan_window_aligned/json/corrected_scan_window_behaviour_annotations.json",
    OUT / "json"
)

copy_file(
    ROOT / "outputs/scan_window_aligned/json/corrected_scan_window_json_summary.csv",
    OUT / "csv"
)

copy_file(
    ROOT / "outputs/scan_window_aligned/behaviour_labels/scan_window_behaviour_labels.json",
    OUT / "json"
)

# CSV outputs
csv_files = [
    ROOT / "data/scan_window_sequences/scan_window_frame_mapping.csv",
    ROOT / "outputs/scan_window_aligned/tracking_with_time/all_scan_windows_bytetrack_tracks_with_excel_window.csv",
    ROOT / "outputs/scan_window_aligned/tracking_with_time/scan_window_tracking_with_time_summary.csv",
    ROOT / "outputs/scan_window_aligned/behaviour_labels/scan_window_behaviour_labels.csv",
    ROOT / "outputs/scan_window_aligned/behaviour_labels/scan_window_behaviour_label_summary.csv",
    ROOT / "outputs/scan_window_aligned/identity_mapping/all_segments_track_id_summary.csv",
    ROOT / "outputs/scan_window_aligned/identity_mapping/track_id_to_colour_mapping_template.csv",
    ROOT / "outputs/scan_window_aligned/identity_mapping/dominant_track_id_to_colour_mapping_table.csv",
]

for f in csv_files:
    copy_file(f, OUT / "csv")

# Segment mapping JSON
copy_file(
    ROOT / "data/scan_window_segments_tentative.json",
    OUT / "json",
    "scan_window_segments_manual_mapping.json"
)

# Figures / montages
figures = [
    ROOT / "outputs/scan_window_video_check/scan_window_video_montage_5sec.jpg",
    ROOT / "outputs/scan_window_video_check/segment_confirmation_montage.jpg",
]

for f in figures:
    copy_file(f, OUT / "figures")

for seg in ["scan_09_00", "scan_09_10", "scan_09_20", "scan_09_30", "scan_09_40", "scan_09_50"]:
    copy_file(
        ROOT / f"outputs/scan_window_aligned/identity_mapping/{seg}_track_id_montage.jpg",
        OUT / "figures"
    )

    # corrected visualization video
    copy_file(
        ROOT / f"outputs/scan_window_aligned/visualization/corrected_scan_windows/{seg}/{seg}_corrected_behaviour_visualization.mp4",
        OUT / "videos"
    )

    # one or more screenshots if available
    vis_dir = ROOT / f"outputs/scan_window_aligned/visualization/corrected_scan_windows/{seg}"
    for jpg in sorted(vis_dir.glob("*.jpg"))[:3]:
        copy_file(jpg, OUT / "figures")

# Scripts used in corrected pipeline
for script_num in range(16, 25):
    for script in (ROOT / "scripts").glob(f"{script_num}_*.py"):
        copy_file(script, OUT / "scripts")

# Build compact README / final summary
summary_csv = ROOT / "outputs/scan_window_aligned/json/corrected_scan_window_json_summary.csv"
tracking_summary_csv = ROOT / "outputs/scan_window_aligned/tracking_with_time/scan_window_tracking_with_time_summary.csv"
label_summary_csv = ROOT / "outputs/scan_window_aligned/behaviour_labels/scan_window_behaviour_label_summary.csv"

json_summary = pd.read_csv(summary_csv) if summary_csv.exists() else pd.DataFrame()
tracking_summary = pd.read_csv(tracking_summary_csv) if tracking_summary_csv.exists() else pd.DataFrame()
label_summary = pd.read_csv(label_summary_csv) if label_summary_csv.exists() else pd.DataFrame()

readme = OUT / "notes" / "corrected_week3_summary.md"

with open(readme, "w") as f:
    f.write("# Week 3 Corrected Scan-Window Behaviour Dataset\n\n")

    f.write("## Purpose\n\n")
    f.write(
        "This corrected version uses a manually concatenated scan-window video containing "
        "the relevant 09:00, 09:10, 09:20, 09:30, 09:40, and 09:50 behavioural observation intervals. "
        "The previous limitation of no exact overlap between the video and Excel scan-sampling windows "
        "is therefore addressed using manual segment-to-Excel-window alignment.\n\n"
    )

    f.write("## Important methodological note\n\n")
    f.write(
        "The corrected video is not a continuous camera recording. It is a manually concatenated screen recording "
        "of selected observation windows. Therefore, timestamp alignment is performed at the segment level instead "
        "of treating video elapsed time as continuous real camera time.\n\n"
    )

    f.write("## Pipeline\n\n")
    f.write("1. Extract scan-window segments from the clean screen recording.\n")
    f.write("2. Run YOLOv8-s pig detection on each segment.\n")
    f.write("3. Run ByteTrack tracking on each segment.\n")
    f.write("4. Attach segment and Excel scan-window timing to each track row.\n")
    f.write("5. Parse Excel behaviour labels for each scan window.\n")
    f.write("6. Build a corrected JSON dataset linking frame, track ID, segment, and available behaviour labels.\n")
    f.write("7. Keep track-to-colour identity mapping as a manual verification step to avoid false behaviour assignment.\n\n")

    f.write("## Corrected JSON summary\n\n")
    if not json_summary.empty:
        f.write(json_summary.to_markdown(index=False))
        f.write("\n\n")

    f.write("## Tracking summary\n\n")
    if not tracking_summary.empty:
        f.write(tracking_summary.to_markdown(index=False))
        f.write("\n\n")

    f.write("## Behaviour label summary\n\n")
    if not label_summary.empty:
        f.write(label_summary.to_markdown(index=False))
        f.write("\n\n")

    f.write("## Current identity status\n\n")
    f.write(
        "The generated JSON contains all track instances and the correct segment-level Excel behaviour candidates. "
        "However, individual track IDs are not forced into colour identities unless manually verified. "
        "This avoids assigning incorrect behaviour labels when colour markers are ambiguous or tracking fragments occur.\n"
    )

print("Saved README:", readme)

# Zip final outputs
zip_base = ROOT / "Week3_corrected_scan_window_final_outputs"
zip_path = shutil.make_archive(str(zip_base), "zip", OUT)

print()
print("Final corrected outputs folder:", OUT)
print("ZIP:", zip_path)
