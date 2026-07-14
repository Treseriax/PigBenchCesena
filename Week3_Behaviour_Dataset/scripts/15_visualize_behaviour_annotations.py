import argparse
import json
from pathlib import Path

import cv2


def short_status(status):
    if status == "stable_track_colour_validation_needed":
        return "stable"
    if status == "fragmented_track_colour_validation_needed":
        return "fragmented"
    if status == "uncertain_track":
        return "uncertain"
    return str(status)


def draw_annotation(frame, frame_ann):
    timestamp = frame_ann.get("timestamp", "unknown")
    frame_index = frame_ann.get("frame_index", -1)

    # Header
    cv2.rectangle(frame, (0, 0), (frame.shape[1], 38), (255, 255, 255), -1)
    cv2.putText(
        frame,
        f"Frame {frame_index} | Timestamp {timestamp}",
        (10, 26),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (0, 0, 0),
        2,
        cv2.LINE_AA,
    )

    for pig in frame_ann.get("pigs", []):
        x1, y1, x2, y2 = [int(round(v)) for v in pig["bbox_xyxy"]]
        track_id = pig["track_id"]
        identity_status = short_status(pig.get("identity_status"))
        behaviour_label = pig.get("behaviour", {}).get("label", "unknown")

        if identity_status == "stable":
            color = (0, 255, 0)
        elif identity_status == "fragmented":
            color = (0, 165, 255)
        else:
            color = (0, 0, 255)

        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

        label1 = f"ID {track_id} | {identity_status}"
        label2 = f"beh: {behaviour_label}"

        y_text = max(55, y1 - 24)

        cv2.putText(
            frame,
            label1,
            (x1, y_text),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            color,
            2,
            cv2.LINE_AA,
        )

        cv2.putText(
            frame,
            label2,
            (x1, y_text + 16),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.42,
            color,
            1,
            cv2.LINE_AA,
        )

    return frame


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seq-dir", required=True)
    parser.add_argument("--json", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--fps", type=float, default=25.091418)
    args = parser.parse_args()

    seq_dir = Path(args.seq_dir)
    img_dir = seq_dir / "img1"
    json_path = Path(args.json)
    out_dir = Path(args.out_dir)
    screenshot_dir = out_dir / "screenshots"
    screenshot_dir.mkdir(parents=True, exist_ok=True)

    with open(json_path) as f:
        data = json.load(f)

    frames = data["frames"]
    frame_by_index = {int(f["frame_index"]): f for f in frames}

    first_img_path = img_dir / "00000001.jpg"
    first_img = cv2.imread(str(first_img_path))

    if first_img is None:
        raise RuntimeError(f"Could not read first image: {first_img_path}")

    height, width = first_img.shape[:2]

    video_path = out_dir / "UniboVid2_sample_behaviour_visualization.mp4"
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(video_path), fourcc, args.fps, (width, height))

    selected_screenshot_frames = {1, 50, 100, 150, 200, 250, 300}
    written = 0

    for frame_idx in sorted(frame_by_index.keys()):
        img_path = img_dir / f"{frame_idx:08d}.jpg"
        frame = cv2.imread(str(img_path))

        if frame is None:
            print("Missing frame:", img_path)
            continue

        ann = frame_by_index[frame_idx]
        vis = draw_annotation(frame, ann)

        writer.write(vis)
        written += 1

        if frame_idx in selected_screenshot_frames:
            out_img = screenshot_dir / f"viewer_frame_{frame_idx:04d}.jpg"
            cv2.imwrite(str(out_img), vis)

    writer.release()

    print("Input JSON:", json_path)
    print("Output video:", video_path)
    print("Output screenshots:", screenshot_dir)
    print("Frames visualized:", written)
    print("Screenshots:")
    for p in sorted(screenshot_dir.glob("*.jpg")):
        print("-", p)


if __name__ == "__main__":
    main()
