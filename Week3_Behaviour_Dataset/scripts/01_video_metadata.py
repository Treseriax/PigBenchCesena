from pathlib import Path
import json
import cv2
import pandas as pd


VIDEO_DIR = Path("Week3_Behaviour_Dataset/data/videos")
OUT_DIR = Path("Week3_Behaviour_Dataset/outputs/video_metadata")
OUT_DIR.mkdir(parents=True, exist_ok=True)

videos = [
    VIDEO_DIR / "UniboVid1.mp4",
    VIDEO_DIR / "UniboVid2.mp4",
]

rows = []

for video_path in videos:
    cap = cv2.VideoCapture(str(video_path))

    if not cap.isOpened():
        print(f"ERROR: Could not open {video_path}")
        continue

    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    duration_sec = frame_count / fps if fps > 0 else None

    video_id = video_path.stem

    print()
    print("Video:", video_id)
    print("Path:", video_path)
    print("FPS:", fps)
    print("Frame count:", frame_count)
    print("Duration seconds:", duration_sec)
    print("Resolution:", width, "x", height)

    sample_indices = {
        "first": 0,
        "middle": frame_count // 2,
        "last": max(0, frame_count - 1),
    }

    for label, frame_idx in sample_indices.items():
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ok, frame = cap.read()

        if ok:
            out_img = OUT_DIR / f"{video_id}_{label}_frame_{frame_idx}.jpg"
            cv2.imwrite(str(out_img), frame)
            print(f"Saved sample frame: {out_img}")

    cap.release()

    rows.append({
        "video_id": video_id,
        "path": str(video_path),
        "fps": fps,
        "frame_count": frame_count,
        "duration_sec": duration_sec,
        "width": width,
        "height": height,
        "notes": "MP4 clip converted/screen-recorded from original Unibo .vgz"
    })

df = pd.DataFrame(rows)

csv_path = OUT_DIR / "video_metadata.csv"
json_path = OUT_DIR / "video_metadata.json"

df.to_csv(csv_path, index=False)

with open(json_path, "w") as f:
    json.dump(rows, f, indent=2)

print()
print("Saved:", csv_path)
print("Saved:", json_path)
