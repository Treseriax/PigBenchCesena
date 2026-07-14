from pathlib import Path
import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


ROOT = Path("Week4_5_Behaviour_Representation")
ROI = ROOT / "outputs/roi_features"
OUT = ROOT / "outputs/visualizations/roi"
NOTES = ROOT / "notes"

SEQ_ROOT = Path("Week3_Behaviour_Dataset/data/scan_window_sequences")

OUT.mkdir(parents=True, exist_ok=True)
NOTES.mkdir(parents=True, exist_ok=True)

zones = pd.read_csv(ROI / "manual_roi_zones_initial_absolute.csv")
resource_summary = pd.read_csv(ROI / "roi_resource_candidate_segment_summary.csv")
zone_summary = pd.read_csv(ROI / "roi_zone_segment_summary.csv")
track_zone = pd.read_csv(ROI / "roi_track_dominant_zone_summary.csv")

segments = sorted(resource_summary["segment_id"].unique().tolist())

# Simple deterministic colors in BGR for OpenCV
ZONE_COLORS = {
    "spatial_grid": (180, 180, 180),
    "resource_candidate": (0, 255, 255),
}

TEXT_COLOR = (255, 255, 255)


def draw_roi_overlay(img, zones_df):
    overlay = img.copy()

    for _, z in zones_df.iterrows():
        x1, y1, x2, y2 = int(z["x1"]), int(z["y1"]), int(z["x2"]), int(z["y2"])
        zone_id = z["zone_id"]
        zone_type = z["zone_type"]

        color = ZONE_COLORS.get(zone_type, (0, 255, 0))
        thickness = 2 if zone_type == "spatial_grid" else 4

        cv2.rectangle(overlay, (x1, y1), (x2, y2), color, thickness)

        if zone_type == "resource_candidate":
            # dark label background
            label = zone_id
            (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.65, 2)
            cv2.rectangle(overlay, (x1, max(0, y1 - th - 10)), (x1 + tw + 8, y1), (0, 0, 0), -1)
            cv2.putText(
                overlay,
                label,
                (x1 + 4, max(18, y1 - 6)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.65,
                TEXT_COLOR,
                2,
                cv2.LINE_AA,
            )

    return overlay


# ---------------------------------------------------------------------
# 1. ROI overlay images on representative frames
# ---------------------------------------------------------------------

for seg in segments:
    img_dir = SEQ_ROOT / seg / "img1"

    # Try middle frame based on available images
    imgs = sorted(img_dir.glob("*.jpg"))
    if not imgs:
        print("No images found for", seg)
        continue

    mid_img = imgs[len(imgs) // 2]
    img = cv2.imread(str(mid_img))

    if img is None:
        print("Could not read", mid_img)
        continue

    overlay = draw_roi_overlay(img, zones)

    # Resize for easier viewing
    h, w = overlay.shape[:2]
    out_w = 1280
    out_h = int(h * out_w / w)
    overlay_resized = cv2.resize(overlay, (out_w, out_h))

    out_path = OUT / f"{seg}_roi_overlay.jpg"
    cv2.imwrite(str(out_path), overlay_resized)
    print("Saved:", out_path)


# ---------------------------------------------------------------------
# 2. Resource candidate zone occupancy per segment
# ---------------------------------------------------------------------

pivot = resource_summary.pivot(
    index="segment_id",
    columns="zone_id",
    values="zone_instance_ratio"
).fillna(0)

plt.figure(figsize=(11, 6))
pivot.plot(kind="bar", figsize=(11, 6))
plt.title("Resource candidate zone instance ratio by segment")
plt.xlabel("segment")
plt.ylabel("zone instance ratio")
plt.xticks(rotation=30, ha="right")
plt.legend(fontsize=8)
plt.tight_layout()
out_path = OUT / "resource_candidate_zone_instance_ratio_by_segment.png"
plt.savefig(out_path, dpi=180)
plt.close()
print("Saved:", out_path)


# ---------------------------------------------------------------------
# 3. Resource candidate zone track seconds per segment
# ---------------------------------------------------------------------

pivot_sec = resource_summary.pivot(
    index="segment_id",
    columns="zone_id",
    values="zone_track_seconds"
).fillna(0)

plt.figure(figsize=(11, 6))
pivot_sec.plot(kind="bar", figsize=(11, 6))
plt.title("Resource candidate zone track-seconds by segment")
plt.xlabel("segment")
plt.ylabel("track-seconds inside zone")
plt.xticks(rotation=30, ha="right")
plt.legend(fontsize=8)
plt.tight_layout()
out_path = OUT / "resource_candidate_zone_track_seconds_by_segment.png"
plt.savefig(out_path, dpi=180)
plt.close()
print("Saved:", out_path)


# ---------------------------------------------------------------------
# 4. Spatial grid occupancy heatmaps
# ---------------------------------------------------------------------

grid_order = [
    ["upper_left", "upper_center", "upper_right"],
    ["middle_left", "middle_center", "middle_right"],
    ["lower_left", "lower_center", "lower_right"],
]

for seg in segments:
    seg_zone = zone_summary[
        (zone_summary["segment_id"] == seg) &
        (zone_summary["zone_type"] == "spatial_grid")
    ].copy()

    grid = np.zeros((3, 3), dtype=float)

    for i, row_names in enumerate(grid_order):
        for j, zone_id in enumerate(row_names):
            val = seg_zone[seg_zone["zone_id"] == zone_id]["zone_instance_ratio"]
            grid[i, j] = float(val.iloc[0]) if len(val) else 0.0

    plt.figure(figsize=(7, 5))
    plt.imshow(grid, aspect="auto")
    plt.title(f"{seg} spatial grid occupancy ratio")
    plt.xticks([0, 1, 2], ["left", "center", "right"])
    plt.yticks([0, 1, 2], ["upper", "middle", "lower"])
    plt.colorbar(label="zone instance ratio")

    for i in range(3):
        for j in range(3):
            plt.text(j, i, f"{grid[i, j]:.2f}", ha="center", va="center")

    plt.tight_layout()
    out_path = OUT / f"{seg}_spatial_grid_occupancy_heatmap.png"
    plt.savefig(out_path, dpi=180)
    plt.close()
    print("Saved:", out_path)


# ---------------------------------------------------------------------
# 5. Dominant spatial grid zone distribution
# ---------------------------------------------------------------------

dist = (
    track_zone
    .groupby(["segment_id", "dominant_spatial_grid_zone"])
    .size()
    .reset_index(name="track_count")
)

pivot_dist = dist.pivot(
    index="segment_id",
    columns="dominant_spatial_grid_zone",
    values="track_count"
).fillna(0)

plt.figure(figsize=(12, 6))
pivot_dist.plot(kind="bar", stacked=True, figsize=(12, 6))
plt.title("Dominant spatial grid zone distribution by segment")
plt.xlabel("segment")
plt.ylabel("track count")
plt.xticks(rotation=30, ha="right")
plt.legend(fontsize=8, ncol=2)
plt.tight_layout()
out_path = OUT / "dominant_spatial_grid_zone_distribution.png"
plt.savefig(out_path, dpi=180)
plt.close()
print("Saved:", out_path)


# ---------------------------------------------------------------------
# 6. Notes
# ---------------------------------------------------------------------

note_path = NOTES / "roi_visualization_notes.md"

with open(note_path, "w") as f:
    f.write("# ROI Visualization Notes\n\n")

    f.write("## Purpose\n\n")
    f.write(
        "This step visualizes the initial manually defined ROI zones and summarizes "
        "zone occupancy patterns across scan-window segments.\n\n"
    )

    f.write("## Generated visualizations\n\n")
    f.write("- ROI overlay images for each scan-window segment\n")
    f.write("- Resource candidate zone instance ratio by segment\n")
    f.write("- Resource candidate zone track-seconds by segment\n")
    f.write("- Spatial grid occupancy heatmaps\n")
    f.write("- Dominant spatial grid zone distribution\n\n")

    f.write("## Interpretation\n\n")
    f.write(
        "The ROI overlays are used to visually inspect whether approximate zones match "
        "meaningful areas in the pen. Occupancy charts summarize how much tracked pig "
        "centroids fall inside each zone. These features are useful as interpretable "
        "spatial behaviour representations.\n\n"
    )

    f.write("## Connection to resource-based behaviour detection\n\n")
    f.write(
        "The Larsen et al. feeding/drinking paper uses precise feeder and drinker ROIs "
        "with head/snout information. In our current prototype, ROI features are based only "
        "on centroid and bounding-box tracking. Therefore, they should be treated as "
        "resource/contact candidate features rather than final feeding or drinking detections.\n"
    )

print("Saved note:", note_path)
