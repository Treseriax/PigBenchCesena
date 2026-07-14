from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


ROOT = Path("Week4_5_Behaviour_Representation")
TRAJ = ROOT / "outputs/trajectory_features"
OUT = ROOT / "outputs/visualizations/trajectory"
NOTES = ROOT / "notes"

OUT.mkdir(parents=True, exist_ok=True)
NOTES.mkdir(parents=True, exist_ok=True)

frame_features = pd.read_csv(TRAJ / "trajectory_frame_level_features.csv")
track_summary = pd.read_csv(TRAJ / "trajectory_track_level_summary.csv")
segment_summary = pd.read_csv(TRAJ / "trajectory_segment_motion_summary.csv")
group_features = pd.read_csv(TRAJ / "trajectory_group_frame_features.csv")
group_summary = pd.read_csv(TRAJ / "trajectory_group_segment_summary.csv")

# Estimate image dimensions from data
image_width = float(np.ceil(frame_features["x2"].max()))
image_height = float(np.ceil(frame_features["y2"].max()))

segments = sorted(frame_features["segment_id"].unique().tolist())

print("Image size estimate:", image_width, image_height)
print("Segments:", segments)

# Keep visualizations readable by focusing on longer tracks
MIN_TRACK_FRAMES = 40

for seg in segments:
    print("===", seg, "===")

    seg_frame = frame_features[frame_features["segment_id"] == seg].copy()
    seg_tracks = track_summary[track_summary["segment_id"] == seg].copy()
    seg_group = group_features[group_features["segment_id"] == seg].copy()

    dominant_ids = (
        seg_tracks[seg_tracks["num_frames"] >= MIN_TRACK_FRAMES]
        .sort_values(["num_frames", "mean_score"], ascending=[False, False])
        .head(12)["track_id"]
        .astype(int)
        .tolist()
    )

    dominant_frame = seg_frame[seg_frame["track_id"].astype(int).isin(dominant_ids)].copy()

    # 1. Trajectory plot
    plt.figure(figsize=(10, 6))
    for tid, g in dominant_frame.groupby("track_id"):
        g = g.sort_values("frame")
        plt.plot(g["cx"], g["cy"], linewidth=1.5, label=f"ID {int(tid)}")
        plt.scatter(g["cx"].iloc[0], g["cy"].iloc[0], s=20)
        plt.scatter(g["cx"].iloc[-1], g["cy"].iloc[-1], s=20, marker="x")

    plt.gca().invert_yaxis()
    plt.xlim(0, image_width)
    plt.ylim(image_height, 0)
    plt.title(f"{seg} dominant track trajectories")
    plt.xlabel("x centroid")
    plt.ylabel("y centroid")
    plt.legend(fontsize=7, ncol=2)
    plt.tight_layout()
    out_path = OUT / f"{seg}_dominant_trajectories.png"
    plt.savefig(out_path, dpi=180)
    plt.close()
    print("Saved:", out_path)

    # 2. Heatmap of all centroid positions
    plt.figure(figsize=(10, 6))
    plt.hist2d(
        seg_frame["cx"],
        seg_frame["cy"],
        bins=[60, 40],
        range=[[0, image_width], [0, image_height]]
    )
    plt.gca().invert_yaxis()
    plt.title(f"{seg} centroid heatmap")
    plt.xlabel("x centroid")
    plt.ylabel("y centroid")
    plt.colorbar(label="track-instance count")
    plt.tight_layout()
    out_path = OUT / f"{seg}_centroid_heatmap.png"
    plt.savefig(out_path, dpi=180)
    plt.close()
    print("Saved:", out_path)

    # 3. Occupancy grid map
    x_bins = np.linspace(0, image_width, 9)
    y_bins = np.linspace(0, image_height, 7)

    occupancy, _, _ = np.histogram2d(
        seg_frame["cy"],
        seg_frame["cx"],
        bins=[y_bins, x_bins]
    )

    occupancy_norm = occupancy / max(1, occupancy.sum())

    plt.figure(figsize=(10, 6))
    plt.imshow(
        occupancy_norm,
        origin="upper",
        extent=[0, image_width, image_height, 0],
        aspect="auto"
    )
    plt.title(f"{seg} normalized occupancy map")
    plt.xlabel("x centroid")
    plt.ylabel("y centroid")
    plt.colorbar(label="relative occupancy")
    plt.tight_layout()
    out_path = OUT / f"{seg}_occupancy_map.png"
    plt.savefig(out_path, dpi=180)
    plt.close()
    print("Saved:", out_path)

    # 4. Speed over time for dominant tracks
    plt.figure(figsize=(10, 6))
    for tid, g in dominant_frame.groupby("track_id"):
        g = g.sort_values("frame")
        # Clip only for visualization so extreme tracking jumps do not dominate the plot
        speed_vis = g["speed_px_s"].clip(upper=300)
        plt.plot(g["segment_time_sec"], speed_vis, linewidth=1.2, label=f"ID {int(tid)}")

    plt.title(f"{seg} speed over time for dominant tracks")
    plt.xlabel("segment time (s)")
    plt.ylabel("speed px/s, clipped at 300 for visualization")
    plt.legend(fontsize=7, ncol=2)
    plt.tight_layout()
    out_path = OUT / f"{seg}_dominant_speed_over_time.png"
    plt.savefig(out_path, dpi=180)
    plt.close()
    print("Saved:", out_path)

    # 5. Group/social features over time
    plt.figure(figsize=(10, 6))
    plt.plot(seg_group["frame"], seg_group["visible_tracks"], label="visible tracks")
    plt.plot(
        seg_group["frame"],
        seg_group["mean_local_density_radius_180px"],
        label="mean local density"
    )
    plt.title(f"{seg} group-level features over time")
    plt.xlabel("frame")
    plt.ylabel("value")
    plt.legend()
    plt.tight_layout()
    out_path = OUT / f"{seg}_group_features_over_time.png"
    plt.savefig(out_path, dpi=180)
    plt.close()
    print("Saved:", out_path)

# Cross-segment summary plots

# 6. Segment mean speed
plt.figure(figsize=(9, 5))
plt.bar(segment_summary["segment_id"], segment_summary["mean_speed_px_s"])
plt.xticks(rotation=30, ha="right")
plt.title("Mean track speed by scan-window segment")
plt.ylabel("mean speed px/s")
plt.tight_layout()
out_path = OUT / "all_segments_mean_speed.png"
plt.savefig(out_path, dpi=180)
plt.close()
print("Saved:", out_path)

# 7. Segment stationary ratio
plt.figure(figsize=(9, 5))
plt.bar(segment_summary["segment_id"], segment_summary["mean_stationary_ratio"])
plt.xticks(rotation=30, ha="right")
plt.title("Mean stationary ratio by scan-window segment")
plt.ylabel("stationary ratio")
plt.ylim(0, 1)
plt.tight_layout()
out_path = OUT / "all_segments_stationary_ratio.png"
plt.savefig(out_path, dpi=180)
plt.close()
print("Saved:", out_path)

# 8. Segment group spread
plt.figure(figsize=(9, 5))
plt.bar(group_summary["segment_id"], group_summary["mean_group_spread_px"])
plt.xticks(rotation=30, ha="right")
plt.title("Mean group spread by scan-window segment")
plt.ylabel("group spread px")
plt.tight_layout()
out_path = OUT / "all_segments_group_spread.png"
plt.savefig(out_path, dpi=180)
plt.close()
print("Saved:", out_path)

# 9. Segment local density
plt.figure(figsize=(9, 5))
plt.bar(group_summary["segment_id"], group_summary["mean_local_density_radius_180px"])
plt.xticks(rotation=30, ha="right")
plt.title("Mean local density by scan-window segment")
plt.ylabel("mean local density within 180 px")
plt.tight_layout()
out_path = OUT / "all_segments_local_density.png"
plt.savefig(out_path, dpi=180)
plt.close()
print("Saved:", out_path)

# Save visualization note
note_path = NOTES / "trajectory_visualization_notes.md"

with open(note_path, "w") as f:
    f.write("# Trajectory Visualization Notes\n\n")
    f.write("## Purpose\n\n")
    f.write(
        "This step generates visual examples of trajectory-derived behaviour representations "
        "from the Week 3 corrected tracking outputs.\n\n"
    )

    f.write("## Generated visualizations\n\n")
    f.write("- Dominant track trajectory plots\n")
    f.write("- Centroid heatmaps\n")
    f.write("- Normalized occupancy maps\n")
    f.write("- Speed-over-time plots\n")
    f.write("- Group/social feature plots over time\n")
    f.write("- Cross-segment summary plots for speed, stationary ratio, group spread, and local density\n\n")

    f.write("## Interpretation\n\n")
    f.write(
        "Trajectory plots show movement paths of dominant track IDs. Heatmaps and occupancy maps "
        "show where pigs spend more time. Speed-over-time plots reveal active or stationary periods. "
        "Group feature plots summarize social/spatial patterns such as visible track count and local density.\n\n"
    )

    f.write("## Important limitation\n\n")
    f.write(
        "Some speed spikes may be caused by tracking fragmentation, ID switches, or bounding-box jitter. "
        "Therefore, speed plots clip extreme values for visualization only, and future robust feature extraction "
        "should include smoothing or outlier filtering.\n"
    )

print("Saved note:", note_path)
