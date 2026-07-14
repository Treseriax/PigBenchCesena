from pathlib import Path
import pandas as pd


ROOT = Path("Week4_5_Behaviour_Representation")
INPUT = ROOT / "data/unibo_inputs"
OUT = ROOT / "outputs/trajectory_features"
NOTES = ROOT / "notes"

OUT.mkdir(parents=True, exist_ok=True)
NOTES.mkdir(parents=True, exist_ok=True)

tracks_path = INPUT / "all_scan_windows_bytetrack_tracks_with_excel_window.csv"
labels_path = INPUT / "scan_window_behaviour_labels.csv"
tracking_summary_path = INPUT / "scan_window_tracking_with_time_summary.csv"

tracks = pd.read_csv(tracks_path)
labels = pd.read_csv(labels_path)
tracking_summary = pd.read_csv(tracking_summary_path)

print("=== Tracks CSV ===")
print("Path:", tracks_path)
print("Shape:", tracks.shape)
print("Columns:")
print(list(tracks.columns))
print()

print("=== Behaviour labels CSV ===")
print("Path:", labels_path)
print("Shape:", labels.shape)
print(labels.head(12).to_string(index=False))
print()

# Basic derived bbox values
tracks["bbox_w"] = tracks["x2"] - tracks["x1"]
tracks["bbox_h"] = tracks["y2"] - tracks["y1"]
tracks["bbox_area"] = tracks["bbox_w"] * tracks["bbox_h"]

segment_overview = (
    tracks
    .groupby("segment_id")
    .agg(
        frames=("frame", "nunique"),
        track_rows=("track_id", "count"),
        unique_track_ids=("track_id", "nunique"),
        min_frame=("frame", "min"),
        max_frame=("frame", "max"),
        min_segment_time=("segment_time_sec", "min"),
        max_segment_time=("segment_time_sec", "max"),
        mean_tracks_per_frame=("track_id", lambda x: len(x) / tracks.loc[x.index, "frame"].nunique()),
        mean_score=("score", "mean"),
        mean_bbox_area=("bbox_area", "mean"),
        excel_interval_start=("excel_interval_start", "first"),
        excel_interval_end=("excel_interval_end", "first"),
    )
    .reset_index()
)

track_lifetime = (
    tracks
    .groupby(["segment_id", "track_id"])
    .agg(
        first_frame=("frame", "min"),
        last_frame=("frame", "max"),
        num_frames=("frame", "nunique"),
        mean_score=("score", "mean"),
        mean_cx=("cx", "mean"),
        mean_cy=("cy", "mean"),
        mean_bbox_area=("bbox_area", "mean"),
        excel_interval_start=("excel_interval_start", "first"),
        excel_interval_end=("excel_interval_end", "first"),
    )
    .reset_index()
)

track_lifetime["duration_sec"] = track_lifetime["num_frames"] / 25.0
track_lifetime = track_lifetime.sort_values(
    ["segment_id", "num_frames", "mean_score"],
    ascending=[True, False, False]
)

missing_summary = tracks.isna().sum().reset_index()
missing_summary.columns = ["column", "missing_count"]
missing_summary = missing_summary[missing_summary["missing_count"] > 0]

segment_overview_path = OUT / "unibo_tracking_segment_overview.csv"
track_lifetime_path = OUT / "unibo_track_lifetime_summary.csv"
missing_summary_path = OUT / "unibo_tracking_missing_values.csv"

segment_overview.to_csv(segment_overview_path, index=False)
track_lifetime.to_csv(track_lifetime_path, index=False)
missing_summary.to_csv(missing_summary_path, index=False)

print("=== Segment overview ===")
print(segment_overview.to_string(index=False))
print()

print("=== Top 10 longest tracks per segment ===")
for seg in sorted(tracks["segment_id"].unique()):
    print()
    print("---", seg, "---")
    top = track_lifetime[track_lifetime["segment_id"] == seg].head(10)
    print(
        top[
            [
                "track_id",
                "first_frame",
                "last_frame",
                "num_frames",
                "duration_sec",
                "mean_score",
                "mean_cx",
                "mean_cy",
            ]
        ].to_string(index=False)
    )

print()
print("=== Missing values ===")
if missing_summary.empty:
    print("No missing values found.")
else:
    print(missing_summary.to_string(index=False))

report_path = NOTES / "unibo_tracking_input_inspection.md"

with open(report_path, "w") as f:
    f.write("# Unibo Tracking Input Inspection\n\n")
    f.write("## Input files\n\n")
    f.write(f"- Tracks: `{tracks_path}`\n")
    f.write(f"- Behaviour labels: `{labels_path}`\n")
    f.write(f"- Tracking summary: `{tracking_summary_path}`\n\n")

    f.write("## Tracks shape\n\n")
    f.write(f"- Rows: {tracks.shape[0]}\n")
    f.write(f"- Columns: {tracks.shape[1]}\n\n")

    f.write("## Segment overview\n\n")
    f.write(segment_overview.to_markdown(index=False))
    f.write("\n\n")

    f.write("## Data quality note\n\n")
    if missing_summary.empty:
        f.write("No missing values were found in the tracking CSV.\n\n")
    else:
        f.write("Missing values were found in the following columns:\n\n")
        f.write(missing_summary.to_markdown(index=False))
        f.write("\n\n")

    f.write("## Interpretation\n\n")
    f.write(
        "The Week 3 corrected tracking outputs contain frame-level bounding boxes, "
        "centroids, track IDs, segment IDs, and Excel scan-window timing. "
        "These are suitable for trajectory-based behaviour representation features "
        "such as speed, acceleration, displacement, trajectory length, turning angle, "
        "occupancy maps, heatmaps, nearest-neighbour distance, local density, and ROI time.\n"
    )

print()
print("Saved:")
print(segment_overview_path)
print(track_lifetime_path)
print(missing_summary_path)
print(report_path)
