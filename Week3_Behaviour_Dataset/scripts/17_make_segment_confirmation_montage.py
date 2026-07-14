from pathlib import Path
import json
import cv2
import numpy as np

SEG_JSON = Path("Week3_Behaviour_Dataset/data/scan_window_segments_tentative.json")
OUT_DIR = Path("Week3_Behaviour_Dataset/outputs/scan_window_video_check")
OUT_DIR.mkdir(parents=True, exist_ok=True)

with open(SEG_JSON) as f:
    data = json.load(f)

video_path = Path(data["video_path"])
cap = cv2.VideoCapture(str(video_path))

if not cap.isOpened():
    raise RuntimeError(f"Could not open video: {video_path}")

fps = cap.get(cv2.CAP_PROP_FPS)
tiles = []

for seg in data["segments"]:
    sid = seg["segment_id"]
    start = float(seg["video_start_sec"])
    end = float(seg["video_end_sec"])
    mid = (start + end) / 2.0

    sample_points = [
        ("start", start),
        ("middle", mid),
        ("end", max(start, end - 0.5))
    ]

    for label, sec in sample_points:
        frame_idx = int(round(sec * fps))
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ok, frame = cap.read()

        if not ok:
            print("Could not read", sid, label, sec)
            continue

        out_img = OUT_DIR / f"{sid}_{label}_{sec:.1f}s_frame_{frame_idx}.jpg"
        cv2.imwrite(str(out_img), frame)

        target_w = 420
        h, w = frame.shape[:2]
        target_h = int(h * target_w / w)
        thumb = cv2.resize(frame, (target_w, target_h))

        bar = np.ones((45, target_w, 3), dtype=np.uint8) * 255
        title = f"{sid} | {label} | {sec:.1f}s"
        cv2.putText(
            bar,
            title,
            (8, 29),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.58,
            (0, 0, 0),
            2,
            cv2.LINE_AA,
        )

        tiles.append(np.vstack([bar, thumb]))

cap.release()

if not tiles:
    raise RuntimeError("No frames were extracted for montage.")

max_h = max(t.shape[0] for t in tiles)
padded = []

for t in tiles:
    if t.shape[0] < max_h:
        pad = np.ones((max_h - t.shape[0], t.shape[1], 3), dtype=np.uint8) * 255
        t = np.vstack([t, pad])
    padded.append(t)

rows = []
for i in range(0, len(padded), 3):
    row_imgs = padded[i:i+3]
    while len(row_imgs) < 3:
        row_imgs.append(np.ones_like(padded[0]) * 255)
    rows.append(np.hstack(row_imgs))

montage = np.vstack(rows)
out_path = OUT_DIR / "segment_confirmation_montage.jpg"
cv2.imwrite(str(out_path), montage)

print("Saved:", out_path)
