from pathlib import Path
import numpy as np
import pandas as pd


ROOT = Path("Week4_5_Behaviour_Representation")
INPUT = ROOT / "data/unibo_inputs"
OUT = ROOT / "outputs/trajectory_features"
NOTES = ROOT / "notes"

OUT.mkdir(parents=True, exist_ok=True)
NOTES.mkdir(parents=True, exist_ok=True)

TRACKS_CSV = INPUT / "all_scan_windows_bytetrack_tracks_with_excel_window.csv"

FPS = 25.0

tracks = pd.read_csv(TRACKS_CSV)

# Basic geometry
tracks["bbox_w"] = tracks["x2"] - tracks["x1"]
tracks["bbox_h"] = tracks["y2"] - tracks["y1"]
tracks["bbox_area"] = tracks["bbox_w"] * tracks["bbox_h"]

# Estimate image size from tracking coordinates.
# This avoids hard-coding resolution, although current video is 1920x1020.
image_width = float(np.ceil(tracks["x2"].max()))
image_height = float(np.ceil(tracks["y2"].max()))
image_diag = float(np.sqrt(image_width ** 2 + image_height ** 2))

tracks["cx_norm"] = tracks["cx"] / image_width
tracks["cy_norm"] = tracks["cy"] / image_height
tracks["bbox_area_norm"] = tracks["bbox_area"] / (image_width * image_height)

tracks = tracks.sort_values(["segment_id", "track_id", "frame"]).reset_index(drop=True)

feature_rows = []

for (seg, tid), g in tracks.groupby(["segment_id", "track_id"], sort=False):
    g = g.sort_values("frame").copy()

    prev_cx = g["cx"].shift(1)
    prev_cy = g["cy"].shift(1)
    prev_frame = g["frame"].shift(1)

    frame_gap = g["frame"] - prev_frame
    dt = frame_gap / FPS

    dx = g["cx"] - prev_cx
    dy = g["cy"] - prev_cy

    step_dist = np.sqrt(dx ** 2 + dy ** 2)

    # If the tracker disappears and reappears after a gap, keep the distance but mark the gap.
    # For speed, avoid division by zero.
    speed_px_s = step_dist / dt.replace(0, np.nan)
    speed_norm_s = speed_px_s / image_diag

    prev_speed = speed_px_s.shift(1)
    acceleration_px_s2 = (speed_px_s - prev_speed) / dt.replace(0, np.nan)

    # Turning angle: angle between previous displacement vector and current displacement vector.
    prev_dx = dx.shift(1)
    prev_dy = dy.shift(1)

    dot = dx * prev_dx + dy * prev_dy
    norm1 = np.sqrt(dx ** 2 + dy ** 2)
    norm2 = np.sqrt(prev_dx ** 2 + prev_dy ** 2)

    cos_angle = dot / (norm1 * norm2)
    cos_angle = cos_angle.clip(-1, 1)
    turning_angle_deg = np.degrees(np.arccos(cos_angle))

    g["frame_gap"] = frame_gap
    g["dt_sec"] = dt
    g["dx"] = dx
    g["dy"] = dy
    g["step_distance_px"] = step_dist
    g["speed_px_s"] = speed_px_s
    g["speed_norm_s"] = speed_norm_s
    g["acceleration_px_s2"] = acceleration_px_s2
    g["turning_angle_deg"] = turning_angle_deg

    feature_rows.append(g)

frame_level_features = pd.concat(feature_rows, ignore_index=True)

# Fill first-row NaNs per track with 0 for movement descriptors
movement_cols = [
    "frame_gap",
    "dt_sec",
    "dx",
    "dy",
    "step_distance_px",
    "speed_px_s",
    "speed_norm_s",
    "acceleration_px_s2",
    "turning_angle_deg",
]

for col in movement_cols:
    frame_level_features[col] = frame_level_features[col].replace([np.inf, -np.inf], np.nan)
    frame_level_features[col] = frame_level_features[col].fillna(0)

# Stationary threshold in pixels/second.
# Conservative: below 20 px/s in a 1920x1020 screen recording is treated as nearly stationary.
STATIONARY_SPEED_THR = 20.0
FAST_SPEED_THR = 120.0

frame_level_features["is_stationary"] = frame_level_features["speed_px_s"] < STATIONARY_SPEED_THR
frame_level_features["is_fast_motion"] = frame_level_features["speed_px_s"] > FAST_SPEED_THR

# Track-level summaries
track_summary = (
    frame_level_features
    .groupby(["segment_id", "track_id"])
    .agg(
        first_frame=("frame", "min"),
        last_frame=("frame", "max"),
        num_frames=("frame", "nunique"),
        duration_sec=("frame", lambda x: x.nunique() / FPS),
        mean_score=("score", "mean"),
        mean_bbox_area=("bbox_area", "mean"),
        mean_bbox_area_norm=("bbox_area_norm", "mean"),
        mean_cx=("cx", "mean"),
        mean_cy=("cy", "mean"),
        mean_cx_norm=("cx_norm", "mean"),
        mean_cy_norm=("cy_norm", "mean"),
        trajectory_length_px=("step_distance_px", "sum"),
        mean_step_distance_px=("step_distance_px", "mean"),
        mean_speed_px_s=("speed_px_s", "mean"),
        max_speed_px_s=("speed_px_s", "max"),
        std_speed_px_s=("speed_px_s", "std"),
        mean_speed_norm_s=("speed_norm_s", "mean"),
        mean_abs_acceleration_px_s2=("acceleration_px_s2", lambda x: np.mean(np.abs(x))),
        max_abs_acceleration_px_s2=("acceleration_px_s2", lambda x: np.max(np.abs(x))),
        mean_turning_angle_deg=("turning_angle_deg", "mean"),
        max_turning_angle_deg=("turning_angle_deg", "max"),
        stationary_ratio=("is_stationary", "mean"),
        fast_motion_ratio=("is_fast_motion", "mean"),
        excel_interval_start=("excel_interval_start", "first"),
        excel_interval_end=("excel_interval_end", "first"),
    )
    .reset_index()
)

track_summary["std_speed_px_s"] = track_summary["std_speed_px_s"].fillna(0)

# Net displacement between first and last centroid per track
first_points = (
    frame_level_features
    .sort_values(["segment_id", "track_id", "frame"])
    .groupby(["segment_id", "track_id"])
    .first()
    .reset_index()[["segment_id", "track_id", "cx", "cy"]]
    .rename(columns={"cx": "first_cx", "cy": "first_cy"})
)

last_points = (
    frame_level_features
    .sort_values(["segment_id", "track_id", "frame"])
    .groupby(["segment_id", "track_id"])
    .last()
    .reset_index()[["segment_id", "track_id", "cx", "cy"]]
    .rename(columns={"cx": "last_cx", "cy": "last_cy"})
)

track_summary = track_summary.merge(first_points, on=["segment_id", "track_id"], how="left")
track_summary = track_summary.merge(last_points, on=["segment_id", "track_id"], how="left")

track_summary["net_displacement_px"] = np.sqrt(
    (track_summary["last_cx"] - track_summary["first_cx"]) ** 2
    + (track_summary["last_cy"] - track_summary["first_cy"]) ** 2
)

track_summary["straightness_index"] = (
    track_summary["net_displacement_px"] / track_summary["trajectory_length_px"].replace(0, np.nan)
).fillna(0)

# Segment-level summaries
segment_motion_summary = (
    track_summary
    .groupby("segment_id")
    .agg(
        tracks=("track_id", "nunique"),
        mean_track_duration_sec=("duration_sec", "mean"),
        mean_trajectory_length_px=("trajectory_length_px", "mean"),
        mean_speed_px_s=("mean_speed_px_s", "mean"),
        max_speed_px_s=("max_speed_px_s", "max"),
        mean_stationary_ratio=("stationary_ratio", "mean"),
        mean_fast_motion_ratio=("fast_motion_ratio", "mean"),
        mean_turning_angle_deg=("mean_turning_angle_deg", "mean"),
        excel_interval_start=("excel_interval_start", "first"),
        excel_interval_end=("excel_interval_end", "first"),
    )
    .reset_index()
)

# Frame-level group features
group_rows = []

LOCAL_DENSITY_RADIUS_PX = 180.0

for (seg, frame), g in frame_level_features.groupby(["segment_id", "frame"]):
    coords = g[["cx", "cy"]].to_numpy(dtype=float)
    n = len(coords)

    if n == 0:
        continue

    group_cx = float(coords[:, 0].mean())
    group_cy = float(coords[:, 1].mean())

    if n == 1:
        mean_nn = 0.0
        min_nn = 0.0
        mean_local_density = 0.0
        group_spread = 0.0
    else:
        diff = coords[:, None, :] - coords[None, :, :]
        dist = np.sqrt((diff ** 2).sum(axis=2))
        np.fill_diagonal(dist, np.nan)

        nn = np.nanmin(dist, axis=1)
        mean_nn = float(np.nanmean(nn))
        min_nn = float(np.nanmin(nn))

        local_counts = np.nansum(dist < LOCAL_DENSITY_RADIUS_PX, axis=1)
        mean_local_density = float(np.mean(local_counts))

        dist_to_group_center = np.sqrt(
            (coords[:, 0] - group_cx) ** 2 + (coords[:, 1] - group_cy) ** 2
        )
        group_spread = float(dist_to_group_center.mean())

    group_rows.append({
        "segment_id": seg,
        "frame": int(frame),
        "visible_tracks": int(n),
        "group_cx": group_cx,
        "group_cy": group_cy,
        "group_cx_norm": group_cx / image_width,
        "group_cy_norm": group_cy / image_height,
        "mean_nearest_neighbour_distance_px": mean_nn,
        "min_nearest_neighbour_distance_px": min_nn,
        "mean_local_density_radius_180px": mean_local_density,
        "group_spread_px": group_spread,
        "excel_interval_start": g["excel_interval_start"].iloc[0],
        "excel_interval_end": g["excel_interval_end"].iloc[0],
    })

group_features = pd.DataFrame(group_rows)

group_segment_summary = (
    group_features
    .groupby("segment_id")
    .agg(
        frames=("frame", "nunique"),
        mean_visible_tracks=("visible_tracks", "mean"),
        max_visible_tracks=("visible_tracks", "max"),
        mean_nearest_neighbour_distance_px=("mean_nearest_neighbour_distance_px", "mean"),
        min_nearest_neighbour_distance_px=("min_nearest_neighbour_distance_px", "min"),
        mean_local_density_radius_180px=("mean_local_density_radius_180px", "mean"),
        mean_group_spread_px=("group_spread_px", "mean"),
        excel_interval_start=("excel_interval_start", "first"),
        excel_interval_end=("excel_interval_end", "first"),
    )
    .reset_index()
)

# Save outputs
frame_features_path = OUT / "trajectory_frame_level_features.csv"
track_summary_path = OUT / "trajectory_track_level_summary.csv"
segment_motion_path = OUT / "trajectory_segment_motion_summary.csv"
group_features_path = OUT / "trajectory_group_frame_features.csv"
group_segment_path = OUT / "trajectory_group_segment_summary.csv"

frame_level_features.to_csv(frame_features_path, index=False)
track_summary.to_csv(track_summary_path, index=False)
segment_motion_summary.to_csv(segment_motion_path, index=False)
group_features.to_csv(group_features_path, index=False)
group_segment_summary.to_csv(group_segment_path, index=False)

print("=== Image size estimate ===")
print("width:", image_width)
print("height:", image_height)
print("diag:", round(image_diag, 2))
print()

print("=== Saved trajectory outputs ===")
for p in [
    frame_features_path,
    track_summary_path,
    segment_motion_path,
    group_features_path,
    group_segment_path,
]:
    print(p)
print()

print("=== Segment motion summary ===")
print(segment_motion_summary.to_string(index=False))
print()

print("=== Group segment summary ===")
print(group_segment_summary.to_string(index=False))
print()

# Write feature definition note
note_path = NOTES / "trajectory_feature_extraction_notes.md"

with open(note_path, "w") as f:
    f.write("# Trajectory Feature Extraction Notes\n\n")
    f.write("## Purpose\n\n")
    f.write(
        "This step converts Week 3 corrected tracking outputs into interpretable "
        "trajectory-based behaviour representation features.\n\n"
    )

    f.write("## Input\n\n")
    f.write(f"- `{TRACKS_CSV}`\n\n")

    f.write("## Feature families\n\n")
    f.write("### Track-level motion features\n\n")
    f.write("- Speed in pixels per second\n")
    f.write("- Normalized speed using image diagonal\n")
    f.write("- Acceleration in pixels per second squared\n")
    f.write("- Step displacement\n")
    f.write("- Trajectory length\n")
    f.write("- Net displacement\n")
    f.write("- Straightness index\n")
    f.write("- Turning angle\n")
    f.write("- Stationary ratio\n")
    f.write("- Fast motion ratio\n\n")

    f.write("### Group/social features\n\n")
    f.write("- Visible track count per frame\n")
    f.write("- Group centroid\n")
    f.write("- Group spread\n")
    f.write("- Nearest-neighbour distance\n")
    f.write("- Local density within 180 px radius\n\n")

    f.write("## Interpretation examples\n\n")
    f.write("- High speed and acceleration may indicate active movement or interaction.\n")
    f.write("- Low speed and high stationary ratio may indicate resting or lying behaviour.\n")
    f.write("- Low nearest-neighbour distance and high local density may indicate social contact or crowding.\n")
    f.write("- High group spread means animals are spatially dispersed.\n")
    f.write("- Low group spread means animals are clustered.\n\n")

    f.write("## Limitations\n\n")
    f.write(
        "The features are based on centroid and bounding-box trajectories only. "
        "The current tracking data does not include snout keypoints, rotated bounding boxes, or skeletons. "
        "Therefore, feeding/drinking resource detection is approximated only in later ROI-based steps.\n"
    )

print("Saved note:", note_path)
