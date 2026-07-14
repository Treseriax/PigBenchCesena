from pathlib import Path
import cv2
import pandas as pd

seq_dir = Path("Week2_Tracking_Study/data/PigTrack/pigtrack0028")
out_dir = Path("Week2_Tracking_Study/outputs/visualizations/pigtrack0028_gt")
out_dir.mkdir(parents=True, exist_ok=True)

img_path = seq_dir / "img1" / "00000001.jpg"
gt_path = seq_dir / "gt" / "gt_mot1.1.txt"

img = cv2.imread(str(img_path))
assert img is not None, f"Could not read image: {img_path}"

cols = ["frame", "track_id", "x", "y", "w", "h", "conf", "class_id", "visibility"]
df = pd.read_csv(gt_path, header=None, names=cols)

frame1 = df[df["frame"] == 1].copy()

for _, row in frame1.iterrows():
    x1 = int(row["x"])
    y1 = int(row["y"])
    x2 = int(row["x"] + row["w"])
    y2 = int(row["y"] + row["h"])
    tid = int(row["track_id"])

    cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
    cv2.putText(
        img,
        f"ID {tid}",
        (x1, max(20, y1 - 5)),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        (0, 255, 0),
        2,
        cv2.LINE_AA
    )

out_path = out_dir / "pigtrack0028_gt_frame1_overlay.jpg"
cv2.imwrite(str(out_path), img)

print("Saved:", out_path)
print("Objects on frame 1:", len(frame1))
