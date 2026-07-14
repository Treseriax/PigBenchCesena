from pathlib import Path
import cv2
import numpy as np

IN_DIR = Path("Week3_Behaviour_Dataset/outputs/video_metadata")
OUT_DIR = Path("Week3_Behaviour_Dataset/outputs/video_metadata")
OUT_DIR.mkdir(parents=True, exist_ok=True)

items = [
    ("UniboVid1 first", IN_DIR / "UniboVid1_first_frame_0.jpg"),
    ("UniboVid1 middle", IN_DIR / "UniboVid1_middle_frame_2372.jpg"),
    ("UniboVid1 last", IN_DIR / "UniboVid1_last_frame_4743.jpg"),
    ("UniboVid2 first", IN_DIR / "UniboVid2_first_frame_0.jpg"),
    ("UniboVid2 middle", IN_DIR / "UniboVid2_middle_frame_2859.jpg"),
    ("UniboVid2 last", IN_DIR / "UniboVid2_last_frame_5718.jpg"),
]

thumbs = []

for label, path in items:
    img = cv2.imread(str(path))
    if img is None:
        raise FileNotFoundError(path)

    # resize to same width
    target_w = 420
    h, w = img.shape[:2]
    target_h = int(h * target_w / w)
    img = cv2.resize(img, (target_w, target_h))

    # add label bar
    bar = np.ones((35, target_w, 3), dtype=np.uint8) * 255
    cv2.putText(bar, label, (10, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 0, 0), 2)
    combined = np.vstack([bar, img])
    thumbs.append(combined)

# make all same height
max_h = max(t.shape[0] for t in thumbs)
padded = []
for t in thumbs:
    if t.shape[0] < max_h:
        pad = np.ones((max_h - t.shape[0], t.shape[1], 3), dtype=np.uint8) * 255
        t = np.vstack([t, pad])
    padded.append(t)

row1 = np.hstack(padded[:3])
row2 = np.hstack(padded[3:])
montage = np.vstack([row1, row2])

out_path = OUT_DIR / "unibo_video_sample_frames_montage.jpg"
cv2.imwrite(str(out_path), montage)

print("Saved:", out_path)
