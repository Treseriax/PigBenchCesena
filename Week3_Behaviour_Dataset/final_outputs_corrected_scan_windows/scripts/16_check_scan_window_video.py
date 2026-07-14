from pathlib import Path
import cv2
import numpy as np
import json

VIDEO = Path("Week3_Behaviour_Dataset/data/videos/Unibo_scan_windows_clean.mp4")
OUT_DIR = Path("Week3_Behaviour_Dataset/outputs/scan_window_video_check")
OUT_DIR.mkdir(parents=True, exist_ok=True)

cap = cv2.VideoCapture(str(VIDEO))

if not cap.isOpened():
    raise RuntimeError(f"Could not open video: {VIDEO}")

fps = cap.get(cv2.CAP_PROP_FPS)
frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
duration = frame_count / fps if fps > 0 else 0

metadata = {
    "video_path": str(VIDEO),
    "fps": fps,
    "frame_count": frame_count,
    "duration_sec": duration,
    "width": width,
    "height": height
}

with open(OUT_DIR / "scan_window_video_metadata.json", "w") as f:
    json.dump(metadata, f, indent=2)

print("Video:", VIDEO)
print("FPS:", fps)
print("Frame count:", frame_count)
print("Duration sec:", duration)
print("Resolution:", width, "x", height)

# sample every 5 seconds
sample_seconds = list(range(0, int(duration) + 1, 5))
if duration not in sample_seconds:
    sample_seconds.append(duration)

thumbs = []

for sec in sample_seconds:
    frame_idx = min(int(round(sec * fps)), frame_count - 1)
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
    ok, frame = cap.read()

    if not ok:
        continue

    out_img = OUT_DIR / f"frame_{frame_idx:06d}_{sec:.1f}s.jpg"
    cv2.imwrite(str(out_img), frame)

    target_w = 420
    h, w = frame.shape[:2]
    target_h = int(h * target_w / w)
    thumb = cv2.resize(frame, (target_w, target_h))

    bar = np.ones((36, target_w, 3), dtype=np.uint8) * 255
    cv2.putText(
        bar,
        f"{sec:.1f}s | frame {frame_idx}",
        (10, 25),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.65,
        (0, 0, 0),
        2,
        cv2.LINE_AA,
    )

    thumb = np.vstack([bar, thumb])
    thumbs.append(thumb)

cap.release()

# make montage, 3 columns
if thumbs:
    max_h = max(t.shape[0] for t in thumbs)
    padded = []
    for t in thumbs:
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
    montage_path = OUT_DIR / "scan_window_video_montage_5sec.jpg"
    cv2.imwrite(str(montage_path), montage)
    print("Saved montage:", montage_path)

print("Saved outputs in:", OUT_DIR)
