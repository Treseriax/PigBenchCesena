from pathlib import Path
import pandas as pd


FRAME_MAP_CSV = Path("Week3_Behaviour_Dataset/data/scan_window_sequences/scan_window_frame_mapping.csv")
TRACK_ROOT = Path("Week3_Behaviour_Dataset/outputs/scan_window_aligned/tracking")
OUT_ROOT = Path("Week3_Behaviour_Dataset/outputs/scan_window_aligned/tracking_with_time")
OUT_ROOT.mkdir(parents=True, exist_ok=True)

segments = [
    "scan_09_00",
    "scan_09_10",
    "scan_09_20",
    "scan_09_30",
    "scan_09_40",
    "scan_09_50",
]

frame_map = pd.read_csv(FRAME_MAP_CSV)

all_tracks = []

for seg in segments:
    print(f"=== Processing {seg} ===")

    track_csv = TRACK_ROOT / f"{seg}_bytetrack" / f"{seg}_bytetrack_tracks.csv"
    out_dir = OUT_ROOT / seg
    out_dir.mkdir(parents=True, exist_ok=True)
    out_csv = out_dir / f"{seg}_bytetrack_tracks_with_excel_window.csv"

    tracks = pd.read_csv(track_csv)
    seg_map = frame_map[frame_map["segment_id"] == seg].copy()

    # Merge local tracker frame with extracted segment frame map
    merged = tracks.merge(
        seg_map,
        left_on="frame",
        right_on="local_frame",
        how="left"
    )

    merged["label_assignment_status"] = "segment_level_excel_window_assigned"
    merged["tracker"] = "bytetrack"
    merged["source_track_csv"] = str(track_csv)

    # This is intentionally segment-level:
    # the video is manually concatenated scan-window material, so we avoid pretending
    # that video elapsed time is a continuous real camera timestamp.
    merged["timestamp_mapping_note"] = (
        "Manually concatenated scan-window video; behaviour labels are assigned "
        "using segment-to-Excel-window mapping."
    )

    merged.to_csv(out_csv, index=False)
    all_tracks.append(merged)

    print("Input rows:", len(tracks))
    print("Output:", out_csv)
    print("Output rows:", len(merged))
    print("Excel window:", merged["excel_interval_start"].iloc[0], "->", merged["excel_interval_end"].iloc[0])
    print("Unique track IDs:", merged["track_id"].nunique())
    print()

combined = pd.concat(all_tracks, ignore_index=True)
combined_csv = OUT_ROOT / "all_scan_windows_bytetrack_tracks_with_excel_window.csv"
combined.to_csv(combined_csv, index=False)

summary = (
    combined
    .groupby("segment_id")
    .agg(
        frames=("frame", "nunique"),
        track_rows=("track_id", "count"),
        unique_track_ids=("track_id", "nunique"),
        excel_interval_start=("excel_interval_start", "first"),
        excel_interval_end=("excel_interval_end", "first")
    )
    .reset_index()
)

summary_csv = OUT_ROOT / "scan_window_tracking_with_time_summary.csv"
summary.to_csv(summary_csv, index=False)

print("=== Combined output ===")
print("Saved:", combined_csv)
print("Rows:", len(combined))
print()
print("=== Summary ===")
print(summary.to_string(index=False))
print("Saved:", summary_csv)
