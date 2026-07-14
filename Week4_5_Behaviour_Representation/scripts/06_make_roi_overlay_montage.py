from pathlib import Path
import cv2
import numpy as np


IN_DIR = Path("Week4_5_Behaviour_Representation/outputs/visualizations/roi")
OUT_DIR = IN_DIR
OUT_PATH = OUT_DIR / "all_segments_roi_overlay_montage.jpg"

segments = [
    "scan_09_00",
    "scan_09_10",
    "scan_09_20",
    "scan_09_30",
    "scan_09_40",
    "scan_09_50",
]

tiles = []

for seg in segments:
    img_path = IN_DIR / f"{seg}_roi_overlay.jpg"
    img = cv2.imread(str(img_path))

    if img is None:
        print("Missing:", img_path)
        continue

    # Make each tile smaller for montage
    target_w = 640
    h, w = img.shape[:2]
    target_h = int(h * target_w / w)
    img = cv2.resize(img, (target_w, target_h))

    header_h = 42
    header = np.ones((header_h, target_w, 3), dtype=np.uint8) * 255
    cv2.putText(
        header,
        seg,
        (12, 28),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (0, 0, 0),
        2,
        cv2.LINE_AA,
    )

    tile = np.vstack([header, img])
    tiles.append(tile)

if not tiles:
    raise RuntimeError("No ROI overlay images found.")

# Pad same height
max_h = max(t.shape[0] for t in tiles)
padded = []

for t in tiles:
    if t.shape[0] < max_h:
        pad = np.ones((max_h - t.shape[0], t.shape[1], 3), dtype=np.uint8) * 255
        t = np.vstack([t, pad])
    padded.append(t)

rows = []
for i in range(0, len(padded), 2):
    row = padded[i:i+2]
    while len(row) < 2:
        row.append(np.ones_like(padded[0]) * 255)
    rows.append(np.hstack(row))

montage = np.vstack(rows)
cv2.imwrite(str(OUT_PATH), montage)

print("Saved:", OUT_PATH)
