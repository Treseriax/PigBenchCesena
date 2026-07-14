import argparse
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def read_mot_gt(gt_path: Path) -> pd.DataFrame:
    cols = ["frame", "track_id", "x", "y", "w", "h", "conf", "class_id", "visibility"]
    df = pd.read_csv(gt_path, header=None, names=cols)

    df["x2"] = df["x"] + df["w"]
    df["y2"] = df["y"] + df["h"]
    df["cx"] = df["x"] + df["w"] / 2.0
    df["cy"] = df["y"] + df["h"] / 2.0

    return df


def load_first_image(seq_dir: Path) -> np.ndarray:
    img_path = seq_dir / "img1" / "00000001.jpg"
    img = cv2.imread(str(img_path))
    if img is None:
        raise FileNotFoundError(f"Could not read first image: {img_path}")
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


def plot_centroid_tracks(df: pd.DataFrame, background: np.ndarray, out_path: Path):
    plt.figure(figsize=(12, 8))
    plt.imshow(background)

    for track_id, group in df.groupby("track_id"):
        group = group.sort_values("frame")
        plt.plot(group["cx"], group["cy"], marker="o", markersize=2, linewidth=1, label=f"ID {track_id}")

        first = group.iloc[0]
        plt.text(first["cx"], first["cy"], str(track_id), fontsize=8)

    plt.title("PigTrack GT Centroid Tracks - pigtrack0028")
    plt.axis("off")
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()


def plot_heatmap(df: pd.DataFrame, width: int, height: int, out_path: Path):
    plt.figure(figsize=(12, 8))
    plt.hist2d(df["cx"], df["cy"], bins=[64, 40], range=[[0, width], [0, height]])
    plt.gca().invert_yaxis()
    plt.colorbar(label="Centroid count")
    plt.title("PigTrack GT Centroid Heatmap - pigtrack0028")
    plt.xlabel("x")
    plt.ylabel("y")
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()


def plot_occupancy_map(df: pd.DataFrame, width: int, height: int, out_path: Path, grid_x: int = 8, grid_y: int = 5):
    x_bins = np.linspace(0, width, grid_x + 1)
    y_bins = np.linspace(0, height, grid_y + 1)

    occ, _, _ = np.histogram2d(df["cy"], df["cx"], bins=[y_bins, x_bins])

    plt.figure(figsize=(10, 6))
    plt.imshow(occ, origin="upper")
    plt.colorbar(label="Centroid count")
    plt.title(f"PigTrack GT Occupancy Map - {grid_x}x{grid_y} grid")
    plt.xlabel("Grid column")
    plt.ylabel("Grid row")

    for r in range(grid_y):
        for c in range(grid_x):
            plt.text(c, r, int(occ[r, c]), ha="center", va="center", fontsize=8)

    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seq-dir", required=True)
    parser.add_argument("--out-dir", required=True)
    args = parser.parse_args()

    seq_dir = Path(args.seq_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    gt_path = seq_dir / "gt" / "gt_mot1.1.txt"
    df = read_mot_gt(gt_path)

    background = load_first_image(seq_dir)
    height, width = background.shape[:2]

    trajectory_csv = out_dir / "pigtrack0028_gt_trajectories.csv"
    df.to_csv(trajectory_csv, index=False)

    plot_centroid_tracks(df, background, out_dir / "pigtrack0028_gt_centroid_tracks.png")
    plot_heatmap(df, width, height, out_dir / "pigtrack0028_gt_heatmap.png")
    plot_occupancy_map(df, width, height, out_dir / "pigtrack0028_gt_occupancy_map.png")

    print("Saved:")
    print(trajectory_csv)
    print(out_dir / "pigtrack0028_gt_centroid_tracks.png")
    print(out_dir / "pigtrack0028_gt_heatmap.png")
    print(out_dir / "pigtrack0028_gt_occupancy_map.png")
    print()
    print("Summary:")
    print("frames:", df["frame"].nunique())
    print("track IDs:", df["track_id"].nunique())
    print("detections:", len(df))
    print("image size:", width, "x", height)


if __name__ == "__main__":
    main()
