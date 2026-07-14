from pathlib import Path
import cv2
import pandas as pd
import numpy as np


SEQ_DIR = Path("Week3_Behaviour_Dataset/data/sequences/UniboVid2_sample")
TRACK_CSV = Path("Week3_Behaviour_Dataset/outputs/tracking/UniboVid2_sample_bytetrack/UniboVid2_sample_bytetrack_tracks_with_time.csv")
OUT_DIR = Path("Week3_Behaviour_Dataset/outputs/identity_mapping")
OUT_DIR.mkdir(parents=True, exist_ok=True)

df = pd.read_csv(TRACK_CSV)

sample_frames = [1, 50, 100, 150, 200, 250, 300]

def draw_tracks(frame, frame_df):
    for _, row in frame_df.iterrows():
        x1, y1, x2, y2 = map(lambda v: int(round(v)), [row["x1"], row["y1"], row["x2"], row["y2"]])
        track_id = int(row["track_id"])

        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(
            frame,
            f"ID {track_id}",
            (x1, max(20, y1 - 5)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.65,
            (0, 255, 0),
            2,
            cv2.LINE_AA,
        )
    return frame

images = []

for frame_idx in sample_frames:
    img_path = SEQ_DIR / "img1" / f"{frame_idx:08d}.jpg"
    img = cv2.imread(str(img_path))

    if img is None:
        print("Missing frame:", img_path)
        continue

    frame_df = df[df["frame"] == frame_idx]
    img = draw_tracks(img, frame_df)

    timestamp = frame_df["estimated_timestamp"].iloc[0] if len(frame_df) else "unknown"

    bar_h = 35
    bar = np.ones((bar_h, img.shape[1], 3), dtype=np.uint8) * 255
    cv2.putText(
        bar,
        f"Frame {frame_idx} | {timestamp}",
        (10, 24),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (0, 0, 0),
        2,
        cv2.LINE_AA,
    )

    img = np.vstack([bar, img])
    img = cv2.resize(img, (704, int(img.shape[0] * 704 / img.shape[1])))
    images.append(img)

# pad all images to same height
max_h = max(img.shape[0] for img in images)
padded = []

for img in images:
    if img.shape[0] < max_h:
        pad = np.ones((max_h - img.shape[0], img.shape[1], 3), dtype=np.uint8) * 255
        img = np.vstack([img, pad])
    padded.append(img)

# 2 columns layout
rows = []
for i in range(0, len(padded), 2):
    if i + 1 < len(padded):
        row = np.hstack([padded[i], padded[i + 1]])
    else:
        blank = np.ones_like(padded[i]) * 255
        row = np.hstack([padded[i], blank])
    rows.append(row)

montage = np.vstack(rows)

out_path = OUT_DIR / "UniboVid2_sample_track_id_montage.jpg"
cv2.imwrite(str(out_path), montage)

summary = (
    df.groupby("track_id")
    .agg(
        first_frame=("frame", "min"),
        last_frame=("frame", "max"),
        num_frames=("frame", "nunique"),
        mean_score=("score", "mean"),
        mean_cx=("cx", "mean"),
        mean_cy=("cy", "mean"),
    )
    .reset_index()
    .sort_values(["first_frame", "track_id"])
)

summary_path = OUT_DIR / "track_id_summary.csv"
summary.to_csv(summary_path, index=False)

template = summary.copy()
template["pig_id"] = ""
template["colour"] = ""
template["identity_status"] = "unassigned_manual_review_needed"
template["notes"] = ""
template_path = OUT_DIR / "track_id_to_pig_identity_template.csv"
template.to_csv(template_path, index=False)

print("Saved montage:", out_path)
print("Saved summary:", summary_path)
print("Saved mapping template:", template_path)
print()
print(summary.to_string(index=False))
