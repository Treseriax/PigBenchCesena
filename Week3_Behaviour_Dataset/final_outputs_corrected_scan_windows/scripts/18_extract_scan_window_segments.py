from pathlib import Path
import json
import cv2
import pandas as pd


SEG_JSON = Path("Week3_Behaviour_Dataset/data/scan_window_segments_tentative.json")
OUT_ROOT = Path("Week3_Behaviour_Dataset/data/scan_window_sequences")
OUT_ROOT.mkdir(parents=True, exist_ok=True)

with open(SEG_JSON) as f:
    segment_data = json.load(f)

video_path = Path(segment_data["video_path"])
cap = cv2.VideoCapture(str(video_path))

if not cap.isOpened():
    raise RuntimeError(f"Could not open video: {video_path}")

fps = cap.get(cv2.CAP_PROP_FPS)
total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

print("Input video:", video_path)
print("FPS:", fps)
print("Total frames:", total_frames)
print("Resolution:", width, "x", height)
print()

all_frame_rows = []

for seg in segment_data["segments"]:
    segment_id = seg["segment_id"]
    start_sec = float(seg["video_start_sec"])
    end_sec = float(seg["video_end_sec"])

    start_frame = int(round(start_sec * fps))
    end_frame = int(round(end_sec * fps))

    segment_dir = OUT_ROOT / segment_id
    img_dir = segment_dir / "img1"
    img_dir.mkdir(parents=True, exist_ok=True)

    # Clean old frames if script is rerun
    for old_img in img_dir.glob("*.jpg"):
        old_img.unlink()

    saved = 0

    for original_frame_idx in range(start_frame, min(end_frame, total_frames)):
        cap.set(cv2.CAP_PROP_POS_FRAMES, original_frame_idx)
        ok, frame = cap.read()

        if not ok:
            continue

        saved += 1
        local_frame_idx = saved
        out_img = img_dir / f"{local_frame_idx:08d}.jpg"
        cv2.imwrite(str(out_img), frame)

        video_time_sec = original_frame_idx / fps
        segment_time_sec = video_time_sec - start_sec

        all_frame_rows.append({
            "segment_id": segment_id,
            "local_frame": local_frame_idx,
            "original_video_frame": original_frame_idx,
            "video_time_sec": round(video_time_sec, 6),
            "segment_time_sec": round(segment_time_sec, 6),
            "excel_interval_start": seg["excel_interval_start"],
            "excel_interval_end": seg["excel_interval_end"],
            "segment_start_sec": start_sec,
            "segment_end_sec": end_sec,
            "source_video": str(video_path)
        })

    seqinfo_path = segment_dir / "seqinfo.ini"
    with open(seqinfo_path, "w") as f:
        f.write("[Sequence]\n")
        f.write(f"name={segment_id}\n")
        f.write("imDir=img1\n")
        f.write(f"frameRate={fps:.6f}\n")
        f.write(f"seqLength={saved}\n")
        f.write(f"imWidth={width}\n")
        f.write(f"imHeight={height}\n")
        f.write("imExt=.jpg\n")
        f.write(f"sourceVideo={video_path.name}\n")
        f.write(f"segmentStartSec={start_sec}\n")
        f.write(f"segmentEndSec={end_sec}\n")
        f.write(f"excelIntervalStart={seg['excel_interval_start']}\n")
        f.write(f"excelIntervalEnd={seg['excel_interval_end']}\n")

    print(f"{segment_id}:")
    print(f"  video sec: {start_sec} -> {end_sec}")
    print(f"  frames: {start_frame} -> {end_frame}")
    print(f"  saved frames: {saved}")
    print(f"  output: {segment_dir}")
    print()

cap.release()

frame_map_df = pd.DataFrame(all_frame_rows)
frame_map_path = OUT_ROOT / "scan_window_frame_mapping.csv"
frame_map_df.to_csv(frame_map_path, index=False)

print("Saved frame mapping:", frame_map_path)
print("Total extracted frame rows:", len(frame_map_df))
