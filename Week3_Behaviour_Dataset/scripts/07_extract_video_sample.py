from pathlib import Path
import cv2

video_path = Path("Week3_Behaviour_Dataset/data/videos/UniboVid2.mp4")
out_dir = Path("Week3_Behaviour_Dataset/data/sequences/UniboVid2_sample")
img_dir = out_dir / "img1"
img_dir.mkdir(parents=True, exist_ok=True)

start_frame = 0
num_frames = 300

cap = cv2.VideoCapture(str(video_path))

if not cap.isOpened():
    raise RuntimeError(f"Could not open video: {video_path}")

fps = cap.get(cv2.CAP_PROP_FPS)
total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)

saved = 0

for i in range(num_frames):
    ok, frame = cap.read()
    if not ok:
        break

    frame_index = i + 1
    out_path = img_dir / f"{frame_index:08d}.jpg"
    cv2.imwrite(str(out_path), frame)
    saved += 1

cap.release()

seqinfo = out_dir / "seqinfo.ini"
with open(seqinfo, "w") as f:
    f.write("[Sequence]\n")
    f.write("name=UniboVid2_sample\n")
    f.write("imDir=img1\n")
    f.write(f"frameRate={fps:.6f}\n")
    f.write(f"seqLength={saved}\n")
    f.write(f"imWidth={width}\n")
    f.write(f"imHeight={height}\n")
    f.write("imExt=.jpg\n")
    f.write("estimatedStartTime=09:04:38\n")
    f.write("sourceVideo=UniboVid2.mp4\n")

print("Video:", video_path)
print("FPS:", fps)
print("Original frames:", total_frames)
print("Resolution:", width, "x", height)
print("Saved frames:", saved)
print("Output:", out_dir)
print("Seqinfo:", seqinfo)
