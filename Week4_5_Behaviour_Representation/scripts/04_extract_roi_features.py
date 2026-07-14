from pathlib import Path
import json
import numpy as np
import pandas as pd


ROOT = Path("Week4_5_Behaviour_Representation")
TRAJ = ROOT / "outputs/trajectory_features"
OUT = ROOT / "outputs/roi_features"
NOTES = ROOT / "notes"
DATA = ROOT / "data"

OUT.mkdir(parents=True, exist_ok=True)
NOTES.mkdir(parents=True, exist_ok=True)

FRAME_CSV = TRAJ / "trajectory_frame_level_features.csv"
ROI_JSON = DATA / "manual_roi_zones_initial.json"

FPS = 25.0

df = pd.read_csv(FRAME_CSV)

image_width = float(np.ceil(df["x2"].max()))
image_height = float(np.ceil(df["y2"].max()))

# ---------------------------------------------------------------------
# Initial ROI definition
# ---------------------------------------------------------------------
# These are intentionally approximate. They are not final ethological
# feeder/drinker annotations. They are a first representation prototype.
#
# spatial_grid zones are non-overlapping coarse spatial regions.
# resource_candidate zones are approximate resource/interaction regions.
# ---------------------------------------------------------------------

if not ROI_JSON.exists():
    roi_def = {
        "description": (
            "Initial manually defined ROI zones for Week 4 feature engineering. "
            "Coordinates are normalized to image width and height. "
            "These zones are approximate and should be visually refined later."
        ),
        "coordinate_type": "normalized",
        "image_width_estimate": image_width,
        "image_height_estimate": image_height,
        "zones": [
            {
                "zone_id": "upper_left",
                "zone_type": "spatial_grid",
                "x1": 0.00, "y1": 0.00, "x2": 0.33, "y2": 0.33,
                "description": "Upper-left coarse spatial grid zone."
            },
            {
                "zone_id": "upper_center",
                "zone_type": "spatial_grid",
                "x1": 0.33, "y1": 0.00, "x2": 0.66, "y2": 0.33,
                "description": "Upper-center coarse spatial grid zone."
            },
            {
                "zone_id": "upper_right",
                "zone_type": "spatial_grid",
                "x1": 0.66, "y1": 0.00, "x2": 1.00, "y2": 0.33,
                "description": "Upper-right coarse spatial grid zone."
            },
            {
                "zone_id": "middle_left",
                "zone_type": "spatial_grid",
                "x1": 0.00, "y1": 0.33, "x2": 0.33, "y2": 0.66,
                "description": "Middle-left coarse spatial grid zone."
            },
            {
                "zone_id": "middle_center",
                "zone_type": "spatial_grid",
                "x1": 0.33, "y1": 0.33, "x2": 0.66, "y2": 0.66,
                "description": "Middle-center coarse spatial grid zone."
            },
            {
                "zone_id": "middle_right",
                "zone_type": "spatial_grid",
                "x1": 0.66, "y1": 0.33, "x2": 1.00, "y2": 0.66,
                "description": "Middle-right coarse spatial grid zone."
            },
            {
                "zone_id": "lower_left",
                "zone_type": "spatial_grid",
                "x1": 0.00, "y1": 0.66, "x2": 0.33, "y2": 1.00,
                "description": "Lower-left coarse spatial grid zone."
            },
            {
                "zone_id": "lower_center",
                "zone_type": "spatial_grid",
                "x1": 0.33, "y1": 0.66, "x2": 0.66, "y2": 1.00,
                "description": "Lower-center coarse spatial grid zone."
            },
            {
                "zone_id": "lower_right",
                "zone_type": "spatial_grid",
                "x1": 0.66, "y1": 0.66, "x2": 1.00, "y2": 1.00,
                "description": "Lower-right coarse spatial grid zone."
            },

            {
                "zone_id": "upper_wall_resource_candidate",
                "zone_type": "resource_candidate",
                "x1": 0.38, "y1": 0.02, "x2": 0.98, "y2": 0.32,
                "description": (
                    "Approximate upper-wall resource/contact candidate zone. "
                    "This can be refined after visual inspection."
                )
            },
            {
                "zone_id": "right_wall_candidate",
                "zone_type": "resource_candidate",
                "x1": 0.78, "y1": 0.15, "x2": 1.00, "y2": 0.80,
                "description": "Approximate right-wall candidate zone."
            },
            {
                "zone_id": "front_gate_lower_candidate",
                "zone_type": "resource_candidate",
                "x1": 0.25, "y1": 0.62, "x2": 0.96, "y2": 1.00,
                "description": "Approximate lower/front gate or close-contact candidate zone."
            },
            {
                "zone_id": "central_activity_candidate",
                "zone_type": "resource_candidate",
                "x1": 0.32, "y1": 0.25, "x2": 0.78, "y2": 0.72,
                "description": "Approximate central activity/contact zone."
            }
        ]
    }

    with open(ROI_JSON, "w") as f:
        json.dump(roi_def, f, indent=2)

with open(ROI_JSON) as f:
    roi_def = json.load(f)

zones = roi_def["zones"]

# Convert normalized coordinates to absolute pixel coordinates
zone_rows = []

for z in zones:
    x1 = float(z["x1"]) * image_width
    y1 = float(z["y1"]) * image_height
    x2 = float(z["x2"]) * image_width
    y2 = float(z["y2"]) * image_height

    zone_rows.append({
        "zone_id": z["zone_id"],
        "zone_type": z["zone_type"],
        "x1": x1,
        "y1": y1,
        "x2": x2,
        "y2": y2,
        "center_x": (x1 + x2) / 2,
        "center_y": (y1 + y2) / 2,
        "description": z.get("description", "")
    })

zones_df = pd.DataFrame(zone_rows)

zones_csv = OUT / "manual_roi_zones_initial_absolute.csv"
zones_df.to_csv(zones_csv, index=False)

# ---------------------------------------------------------------------
# Frame-level ROI membership
# ---------------------------------------------------------------------

roi_df = df.copy()

for _, z in zones_df.iterrows():
    zid = z["zone_id"]

    inside = (
        (roi_df["cx"] >= z["x1"]) &
        (roi_df["cx"] < z["x2"]) &
        (roi_df["cy"] >= z["y1"]) &
        (roi_df["cy"] < z["y2"])
    )

    roi_df[f"inside_{zid}"] = inside.astype(int)

    dist = np.sqrt(
        (roi_df["cx"] - z["center_x"]) ** 2 +
        (roi_df["cy"] - z["center_y"]) ** 2
    )

    roi_df[f"distance_to_{zid}_center_px"] = dist

# Assign one non-overlapping spatial grid zone to each row
grid_zones = zones_df[zones_df["zone_type"] == "spatial_grid"].copy()

def assign_grid_zone(row):
    cx = row["cx"]
    cy = row["cy"]

    for _, z in grid_zones.iterrows():
        if cx >= z["x1"] and cx < z["x2"] and cy >= z["y1"] and cy < z["y2"]:
            return z["zone_id"]

    return "outside_grid"

roi_df["primary_spatial_grid_zone"] = roi_df.apply(assign_grid_zone, axis=1)

frame_roi_path = OUT / "roi_frame_level_features.csv"
roi_df.to_csv(frame_roi_path, index=False)

# ---------------------------------------------------------------------
# Track-level ROI summary
# ---------------------------------------------------------------------

track_rows = []

for (seg, tid), g in roi_df.groupby(["segment_id", "track_id"]):
    row = {
        "segment_id": seg,
        "track_id": int(tid),
        "first_frame": int(g["frame"].min()),
        "last_frame": int(g["frame"].max()),
        "num_frames": int(g["frame"].nunique()),
        "duration_sec": float(g["frame"].nunique() / FPS),
        "mean_score": float(g["score"].mean()),
        "mean_cx": float(g["cx"].mean()),
        "mean_cy": float(g["cy"].mean()),
        "excel_interval_start": g["excel_interval_start"].iloc[0],
        "excel_interval_end": g["excel_interval_end"].iloc[0],
    }

    # Time spent in every zone
    for _, z in zones_df.iterrows():
        zid = z["zone_id"]
        inside_col = f"inside_{zid}"
        dist_col = f"distance_to_{zid}_center_px"

        inside_frames = int(g[inside_col].sum())
        row[f"{zid}_time_sec"] = inside_frames / FPS
        row[f"{zid}_ratio"] = float(g[inside_col].mean())
        row[f"{zid}_mean_distance_px"] = float(g[dist_col].mean())
        row[f"{zid}_min_distance_px"] = float(g[dist_col].min())

    # Primary spatial zone = most frequent grid zone
    if len(g) > 0:
        row["dominant_spatial_grid_zone"] = g["primary_spatial_grid_zone"].mode().iloc[0]
    else:
        row["dominant_spatial_grid_zone"] = "unknown"

    # Spatial zone switches over time
    ordered_zones = g.sort_values("frame")["primary_spatial_grid_zone"].tolist()
    switches = 0
    for a, b in zip(ordered_zones[:-1], ordered_zones[1:]):
        if a != b:
            switches += 1

    row["spatial_grid_zone_switches"] = switches
    row["spatial_grid_zone_switch_rate"] = switches / max(1, len(ordered_zones) - 1)

    track_rows.append(row)

track_roi = pd.DataFrame(track_rows)

track_roi_path = OUT / "roi_track_level_summary.csv"
track_roi.to_csv(track_roi_path, index=False)

# ---------------------------------------------------------------------
# Segment-zone summaries
# ---------------------------------------------------------------------

segment_zone_rows = []

for seg, seg_df in roi_df.groupby("segment_id"):
    frame_count = seg_df["frame"].nunique()

    for _, z in zones_df.iterrows():
        zid = z["zone_id"]
        inside_col = f"inside_{zid}"

        per_frame_counts = (
            seg_df
            .groupby("frame")[inside_col]
            .sum()
            .reset_index(name="tracks_inside_zone")
        )

        segment_zone_rows.append({
            "segment_id": seg,
            "zone_id": zid,
            "zone_type": z["zone_type"],
            "frames": int(frame_count),
            "track_instances": int(len(seg_df)),
            "zone_track_instances": int(seg_df[inside_col].sum()),
            "zone_instance_ratio": float(seg_df[inside_col].mean()),
            "zone_track_seconds": float(seg_df[inside_col].sum() / FPS),
            "mean_tracks_inside_zone_per_frame": float(per_frame_counts["tracks_inside_zone"].mean()),
            "max_tracks_inside_zone_per_frame": int(per_frame_counts["tracks_inside_zone"].max()),
            "unique_tracks_visiting_zone": int(seg_df[seg_df[inside_col] == 1]["track_id"].nunique()),
            "excel_interval_start": seg_df["excel_interval_start"].iloc[0],
            "excel_interval_end": seg_df["excel_interval_end"].iloc[0],
        })

segment_zone_summary = pd.DataFrame(segment_zone_rows)

segment_zone_path = OUT / "roi_zone_segment_summary.csv"
segment_zone_summary.to_csv(segment_zone_path, index=False)

# Segment-level compact summary: dominant resource candidate zones
resource_summary = (
    segment_zone_summary[segment_zone_summary["zone_type"] == "resource_candidate"]
    .sort_values(["segment_id", "zone_instance_ratio"], ascending=[True, False])
    .copy()
)

resource_summary_path = OUT / "roi_resource_candidate_segment_summary.csv"
resource_summary.to_csv(resource_summary_path, index=False)

# Track dominant zone compact output
track_dominant = track_roi[
    [
        "segment_id",
        "track_id",
        "first_frame",
        "last_frame",
        "num_frames",
        "duration_sec",
        "dominant_spatial_grid_zone",
        "spatial_grid_zone_switches",
        "spatial_grid_zone_switch_rate",
        "excel_interval_start",
        "excel_interval_end"
    ]
].copy()

track_dominant_path = OUT / "roi_track_dominant_zone_summary.csv"
track_dominant.to_csv(track_dominant_path, index=False)

# ---------------------------------------------------------------------
# Notes
# ---------------------------------------------------------------------

note_path = NOTES / "roi_feature_extraction_notes.md"

with open(note_path, "w") as f:
    f.write("# ROI / Resource-Based Feature Extraction Notes\n\n")

    f.write("## Purpose\n\n")
    f.write(
        "This step creates a first ROI-based behaviour representation prototype from "
        "Week 3 corrected tracking outputs. It estimates how much time each track spends "
        "inside coarse spatial zones and approximate resource/contact candidate zones.\n\n"
    )

    f.write("## Connection to Larsen et al. 2026\n\n")
    f.write(
        "The Larsen et al. 2026 feeding/drinking paper uses pig detection, rotated bounding boxes, "
        "head direction, snout keypoints, and feeder/drinker ROIs to estimate resource occupation. "
        "Our current Unibo tracking data does not contain snout keypoints or rotated bounding boxes, "
        "so this prototype uses centroid and bounding-box based ROI approximation only.\n\n"
    )

    f.write("## Generated feature files\n\n")
    f.write("- `roi_frame_level_features.csv`\n")
    f.write("- `roi_track_level_summary.csv`\n")
    f.write("- `roi_track_dominant_zone_summary.csv`\n")
    f.write("- `roi_zone_segment_summary.csv`\n")
    f.write("- `roi_resource_candidate_segment_summary.csv`\n")
    f.write("- `manual_roi_zones_initial_absolute.csv`\n")
    f.write("- `manual_roi_zones_initial.json`\n\n")

    f.write("## Interpretation\n\n")
    f.write(
        "High time spent in a zone may indicate resting, activity, resource use, or social contact, "
        "depending on the zone location. Resource candidate zones are not final feeder/drinker labels; "
        "they are approximate regions for feature engineering and should be visually refined.\n\n"
    )

    f.write("## Limitation\n\n")
    f.write(
        "Without snout keypoints and rotated bounding boxes, this approach cannot reliably distinguish "
        "actual feeding/drinking from simply being near a feeder/drinker area. It should therefore be "
        "treated as a representation prototype, not a final behaviour detector.\n"
    )

print("=== ROI zones ===")
print(zones_df.to_string(index=False))
print()

print("=== Saved ROI outputs ===")
for p in [
    ROI_JSON,
    zones_csv,
    frame_roi_path,
    track_roi_path,
    track_dominant_path,
    segment_zone_path,
    resource_summary_path,
    note_path,
]:
    print(p)

print()
print("=== Resource candidate segment summary ===")
print(resource_summary.to_string(index=False))
