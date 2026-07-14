from pathlib import Path
import cv2
import numpy as np
import pandas as pd
import textwrap


SEQ_ROOT = Path("Week3_Behaviour_Dataset/data/scan_window_sequences")
TRACK_ROOT = Path("Week3_Behaviour_Dataset/outputs/scan_window_aligned/tracking_with_time")
LABEL_CSV = Path("Week3_Behaviour_Dataset/outputs/scan_window_aligned/behaviour_labels/scan_window_behaviour_labels.csv")
DOMINANT_CSV = Path("Week3_Behaviour_Dataset/outputs/scan_window_aligned/identity_mapping/dominant_track_id_to_colour_mapping_table.csv")

OUT_ROOT = Path("Week3_Behaviour_Dataset/outputs/scan_window_aligned/visualization/corrected_scan_windows")
OUT_ROOT.mkdir(parents=True, exist_ok=True)

segments = [
    "scan_09_00",
    "scan_09_10",
    "scan_09_20",
    "scan_09_30",
    "scan_09_40",
    "scan_09_50",
]

labels = pd.read_csv(LABEL_CSV)
dominant = pd.read_csv(DOMINANT_CSV)

# Use dominant tracks only for readable visualization
dominant_lookup = {
    seg: set(dominant[dominant["segment_id"] == seg]["track_id"].astype(int).tolist())
    for seg in segments
}


def label_text_for_segment(seg):
    seg_labels = labels[labels["segment_id"] == seg].copy()
    parts = []
    for _, r in seg_labels.iterrows():
        parts.append(f"{r['colour']}={r['behaviour_code']}({r['behaviour_label']})")
    return "; ".join(parts)


def draw_info_panel(frame, seg, excel_start, excel_end, frame_idx, candidate_text):
    h, w = frame.shape[:2]

    panel_h = 125
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, 0), (w, panel_h), (0, 0, 0), -1)
    frame[:] = cv2.addWeighted(overlay, 0.55, frame, 0.45, 0)

    line1 = f"{seg} | frame {frame_idx} | Excel window: {excel_start}-{excel_end}"
    line2 = "Identity status: unverified unless manually mapped | Behaviour: segment-level Excel candidates"
    wrapped = textwrap.wrap("Candidates: " + candidate_text, width=135)

    cv2.putText(frame, line1, (18, 34), cv2.FONT_HERSHEY_SIMPLEX, 0.82, (255, 255, 255), 2, cv2.LINE_AA)
    cv2.putText(frame, line2, (18, 67), cv2.FONT_HERSHEY_SIMPLEX, 0.62, (255, 255, 255), 2, cv2.LINE_AA)

    y = 99
    for line in wrapped[:1]:
        cv2.putText(frame, line, (18, y), cv2.FONT_HERSHEY_SIMPLEX, 0.54, (255, 255, 255), 1, cv2.LINE_AA)
        y += 24

    return frame


def draw_track(frame, row):
    x1, y1, x2, y2 = map(lambda v: int(round(v)), [row["x1"], row["y1"], row["x2"], row["y2"]])
    tid = int(row["track_id"])
    score = float(row["score"])

    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 3)

    label = f"ID {tid} | unverified | {score:.2f}"
    label_y = max(145, y1 - 8)
    cv2.putText(frame, label, (x1, label_y), cv2.FONT_HERSHEY_SIMPLEX, 0.70, (0, 255, 0), 2, cv2.LINE_AA)

    return frame


for seg in segments:
    print(f"=== Visualizing {seg} ===")

    seq_dir = SEQ_ROOT / seg
    img_dir = seq_dir / "img1"
    track_csv = TRACK_ROOT / seg / f"{seg}_bytetrack_tracks_with_excel_window.csv"

    out_dir = OUT_ROOT / seg
    out_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(track_csv)
    allowed_ids = dominant_lookup.get(seg, set())

    # Filter to dominant tracks for readability
    df = df[df["track_id"].astype(int).isin(allowed_ids)].copy()

    if df.empty:
        print("No dominant tracks found for", seg)
        continue

    max_frame = int(df["frame"].max())
    excel_start = str(df["excel_interval_start"].iloc[0])
    excel_end = str(df["excel_interval_end"].iloc[0])
    candidate_text = label_text_for_segment(seg)

    first_img = cv2.imread(str(img_dir / "00000001.jpg"))
    if first_img is None:
        raise RuntimeError(f"Could not read first image for {seg}")

    original_h, original_w = first_img.shape[:2]

    # Make output smaller than original screen recording but keep aspect ratio
    out_w = 1280
    out_h = int(original_h * out_w / original_w)

    out_video = out_dir / f"{seg}_corrected_behaviour_visualization.mp4"
    writer = cv2.VideoWriter(
        str(out_video),
        cv2.VideoWriter_fourcc(*"mp4v"),
        25.0,
        (out_w, out_h)
    )

    screenshot_frames = sorted(set([1, max(1, max_frame // 2), max_frame]))

    for frame_idx in range(1, max_frame + 1):
        img_path = img_dir / f"{frame_idx:08d}.jpg"
        frame = cv2.imread(str(img_path))

        if frame is None:
            continue

        frame_df = df[df["frame"] == frame_idx]

        frame = draw_info_panel(frame, seg, excel_start, excel_end, frame_idx, candidate_text)

        for _, row in frame_df.iterrows():
            frame = draw_track(frame, row)

        frame_resized = cv2.resize(frame, (out_w, out_h))
        writer.write(frame_resized)

        if frame_idx in screenshot_frames:
            cv2.imwrite(str(out_dir / f"{seg}_viewer_frame_{frame_idx:04d}.jpg"), frame_resized)

    writer.release()

    print("Saved video:", out_video)
    print("Saved screenshots in:", out_dir)

print("Done.")
