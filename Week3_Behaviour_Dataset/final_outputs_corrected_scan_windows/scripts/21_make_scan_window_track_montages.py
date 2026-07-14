from pathlib import Path
import cv2
import numpy as np
import pandas as pd


SEQ_ROOT = Path("Week3_Behaviour_Dataset/data/scan_window_sequences")
TRACK_ROOT = Path("Week3_Behaviour_Dataset/outputs/scan_window_aligned/tracking_with_time")
LABEL_CSV = Path("Week3_Behaviour_Dataset/outputs/scan_window_aligned/behaviour_labels/scan_window_behaviour_labels.csv")

OUT_DIR = Path("Week3_Behaviour_Dataset/outputs/scan_window_aligned/identity_mapping")
OUT_DIR.mkdir(parents=True, exist_ok=True)

segments = [
    "scan_09_00",
    "scan_09_10",
    "scan_09_20",
    "scan_09_30",
    "scan_09_40",
    "scan_09_50",
]

labels = pd.read_csv(LABEL_CSV)

all_summary_rows = []
all_template_rows = []


def draw_tracks(frame, frame_df):
    for _, row in frame_df.iterrows():
        x1, y1, x2, y2 = map(lambda v: int(round(v)), [row["x1"], row["y1"], row["x2"], row["y2"]])
        track_id = int(row["track_id"])

        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 3)
        cv2.putText(
            frame,
            f"ID {track_id}",
            (x1, max(35, y1 - 8)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.9,
            (0, 255, 0),
            3,
            cv2.LINE_AA,
        )

    return frame


for seg in segments:
    print(f"=== {seg} ===")

    seq_dir = SEQ_ROOT / seg
    img_dir = seq_dir / "img1"
    track_csv = TRACK_ROOT / seg / f"{seg}_bytetrack_tracks_with_excel_window.csv"

    df = pd.read_csv(track_csv)
    max_frame = int(df["frame"].max())

    sample_frames = sorted(set([
        1,
        max(1, int(max_frame * 0.25)),
        max(1, int(max_frame * 0.50)),
        max(1, int(max_frame * 0.75)),
        max_frame,
    ]))

    tiles = []

    for frame_idx in sample_frames:
        img_path = img_dir / f"{frame_idx:08d}.jpg"
        img = cv2.imread(str(img_path))

        if img is None:
            print("Missing image:", img_path)
            continue

        frame_df = df[df["frame"] == frame_idx]
        img = draw_tracks(img, frame_df)

        interval_start = str(frame_df["excel_interval_start"].iloc[0]) if len(frame_df) else "unknown"
        interval_end = str(frame_df["excel_interval_end"].iloc[0]) if len(frame_df) else "unknown"

        target_w = 640
        h, w = img.shape[:2]
        target_h = int(h * target_w / w)
        thumb = cv2.resize(img, (target_w, target_h))

        bar = np.ones((48, target_w, 3), dtype=np.uint8) * 255
        title = f"{seg} | frame {frame_idx} | {interval_start}-{interval_end}"
        cv2.putText(
            bar,
            title,
            (10, 32),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.65,
            (0, 0, 0),
            2,
            cv2.LINE_AA,
        )

        tiles.append(np.vstack([bar, thumb]))

    if tiles:
        max_h = max(t.shape[0] for t in tiles)
        padded = []

        for t in tiles:
            if t.shape[0] < max_h:
                pad = np.ones((max_h - t.shape[0], t.shape[1], 3), dtype=np.uint8) * 255
                t = np.vstack([t, pad])
            padded.append(t)

        rows = []
        for i in range(0, len(padded), 2):
            row_imgs = padded[i:i+2]
            while len(row_imgs) < 2:
                row_imgs.append(np.ones_like(padded[0]) * 255)
            rows.append(np.hstack(row_imgs))

        montage = np.vstack(rows)
        out_montage = OUT_DIR / f"{seg}_track_id_montage.jpg"
        cv2.imwrite(str(out_montage), montage)
        print("Saved montage:", out_montage)

    summary = (
        df.groupby("track_id")
        .agg(
            first_frame=("frame", "min"),
            last_frame=("frame", "max"),
            num_frames=("frame", "nunique"),
            mean_score=("score", "mean"),
            mean_cx=("cx", "mean"),
            mean_cy=("cy", "mean"),
            excel_interval_start=("excel_interval_start", "first"),
            excel_interval_end=("excel_interval_end", "first"),
        )
        .reset_index()
        .sort_values(["num_frames", "mean_score"], ascending=[False, False])
    )

    for _, row in summary.iterrows():
        all_summary_rows.append({
            "segment_id": seg,
            "track_id": int(row["track_id"]),
            "first_frame": int(row["first_frame"]),
            "last_frame": int(row["last_frame"]),
            "num_frames": int(row["num_frames"]),
            "mean_score": round(float(row["mean_score"]), 4),
            "mean_cx": round(float(row["mean_cx"]), 2),
            "mean_cy": round(float(row["mean_cy"]), 2),
            "excel_interval_start": row["excel_interval_start"],
            "excel_interval_end": row["excel_interval_end"],
        })

        all_template_rows.append({
            "segment_id": seg,
            "track_id": int(row["track_id"]),
            "first_frame": int(row["first_frame"]),
            "last_frame": int(row["last_frame"]),
            "num_frames": int(row["num_frames"]),
            "mean_score": round(float(row["mean_score"]), 4),
            "mean_cx": round(float(row["mean_cx"]), 2),
            "mean_cy": round(float(row["mean_cy"]), 2),
            "assigned_pig_id": "",
            "assigned_colour": "",
            "behaviour_code": "",
            "behaviour_label": "",
            "identity_confidence": "",
            "notes": "Fill assigned_colour manually if the colour marker is visible."
        })

    # Also save segment-specific label table for quick review
    seg_labels = labels[labels["segment_id"] == seg]
    seg_labels.to_csv(OUT_DIR / f"{seg}_excel_behaviour_labels.csv", index=False)

summary_df = pd.DataFrame(all_summary_rows)
template_df = pd.DataFrame(all_template_rows)

summary_path = OUT_DIR / "all_segments_track_id_summary.csv"
template_path = OUT_DIR / "track_id_to_colour_mapping_template.csv"

summary_df.to_csv(summary_path, index=False)
template_df.to_csv(template_path, index=False)

print()
print("Saved summary:", summary_path)
print("Saved mapping template:", template_path)
print()
print("Top rows:")
print(template_df.head(30).to_string(index=False))
