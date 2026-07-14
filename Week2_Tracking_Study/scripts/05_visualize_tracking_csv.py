import argparse
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def load_first_image(seq_dir: Path) -> np.ndarray:
    img_path = seq_dir / "img1" / "00000001.jpg"
    img = cv2.imread(str(img_path))
    if img is None:
        raise FileNotFoundError(f"Could not read first image: {img_path}")
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


def plot_centroid_tracks(df: pd.DataFrame, background: np.ndarray, out_path: Path, title: str):
    plt.figure(figsize=(12, 8))
    plt.imshow(background)

    for track_id, group in df.groupby("track_id"):
        group = group.sort_values("frame")
        plt.plot(group["cx"], group["cy"], marker="o", markersize=2, linewidth=1, label=f"ID {track_id}")

        first = group.iloc[0]
        plt.text(first["cx"], first["cy"], str(int(track_id)), fontsize=8)

    plt.title(title)
    plt.axis("off")
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()


def plot_heatmap(df: pd.DataFrame, width: int, height: int, out_path: Path, title: str):
    plt.figure(figsize=(12, 8))
    plt.hist2d(df["cx"], df["cy"], bins=[64, 40], range=[[0, width], [0, height]])
    plt.gca().invert_yaxis()
    plt.colorbar(label="Centroid count")
    plt.title(title)
    plt.xlabel("x")
    plt.ylabel("y")
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()


def plot_occupancy_map(df: pd.DataFrame, width: int, height: int, out_path: Path, title: str, grid_x: int = 8, grid_y: int = 5):
    x_bins = np.linspace(0, width, grid_x + 1)
    y_bins = np.linspace(0, height, grid_y + 1)

    occ, _, _ = np.histogram2d(df["cy"], df["cx"], bins=[y_bins, x_bins])

    plt.figure(figsize=(10, 6))
    plt.imshow(occ, origin="upper")
    plt.colorbar(label="Centroid count")
    plt.title(title)
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
    parser.add_argument("--tracks-csv", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--name", required=True)
    args = parser.parse_args()

    seq_dir = Path(args.seq_dir)
    tracks_csv = Path(args.tracks_csv)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(tracks_csv)

    background = load_first_image(seq_dir)
    height, width = background.shape[:2]

    plot_centroid_tracks(
        df,
        background,
        out_dir / f"{args.name}_centroid_tracks.png",
        f"{args.name} Centroid Tracks"
    )

    plot_heatmap(
        df,
        width,
        height,
        out_dir / f"{args.name}_heatmap.png",
        f"{args.name} Centroid Heatmap"
    )

    plot_occupancy_map(
        df,
        width,
        height,
        out_dir / f"{args.name}_occupancy_map.png",
        f"{args.name} Occupancy Map"
    )

    print("Saved visualizations to:", out_dir)
    print("frames:", df["frame"].nunique())
    print("track IDs:", df["track_id"].nunique())
    print("rows:", len(df))


if __name__ == "__main__":
    main()
