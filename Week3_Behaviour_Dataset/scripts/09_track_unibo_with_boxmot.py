import argparse
import configparser
import sys
from pathlib import Path

import cv2
import numpy as np
import pandas as pd


def read_fps_from_seqinfo(seq_dir: Path, default_fps: float = 25.0) -> float:
    seqinfo = seq_dir / "seqinfo.ini"
    if not seqinfo.exists():
        return default_fps

    config = configparser.ConfigParser()
    config.read(seqinfo)

    try:
        return float(config["Sequence"]["frameRate"])
    except Exception:
        return default_fps


def iou_xyxy(box, boxes):
    if len(boxes) == 0:
        return np.array([])

    x1 = np.maximum(box[0], boxes[:, 0])
    y1 = np.maximum(box[1], boxes[:, 1])
    x2 = np.minimum(box[2], boxes[:, 2])
    y2 = np.minimum(box[3], boxes[:, 3])

    inter_w = np.maximum(0, x2 - x1)
    inter_h = np.maximum(0, y2 - y1)
    inter = inter_w * inter_h

    area_box = max(0, box[2] - box[0]) * max(0, box[3] - box[1])
    area_boxes = np.maximum(0, boxes[:, 2] - boxes[:, 0]) * np.maximum(0, boxes[:, 3] - boxes[:, 1])

    union = area_box + area_boxes - inter + 1e-6
    return inter / union


def nms_detections(dets, iou_thr=0.45):
    if len(dets) == 0:
        return dets

    order = np.argsort(-dets[:, 4])
    dets = dets[order]

    keep = []
    while len(dets) > 0:
        current = dets[0]
        keep.append(current)

        if len(dets) == 1:
            break

        ious = iou_xyxy(current[:4], dets[1:, :4])
        dets = dets[1:][ious < iou_thr]

    return np.array(keep, dtype=float)


def draw_track(img, x1, y1, x2, y2, track_id, score=None):
    color = (0, 255, 0)

    cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)

    label = f"ID {track_id}"
    if score is not None and score >= 0:
        label += f" {score:.2f}"

    cv2.putText(
        img,
        label,
        (x1, max(20, y1 - 5)),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        color,
        2,
        cv2.LINE_AA,
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seq-dir", required=True)
    parser.add_argument("--name", required=True)
    parser.add_argument("--detections-csv", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--tracker", default="bytetrack")
    parser.add_argument("--tracker-config", required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--nms-iou", type=float, default=0.45)
    args = parser.parse_args()

    repo_root = Path.cwd()
    sys.path.insert(0, str(repo_root / "tracking" / "boxmot"))

    try:
        from boxmot import create_tracker
    except Exception:
        from boxmot.tracker_zoo import create_tracker

    seq_dir = Path(args.seq_dir)
    img_dir = seq_dir / "img1"
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    det_df = pd.read_csv(args.detections_csv)
    image_paths = sorted(img_dir.glob("*.jpg"))

    if not image_paths:
        raise FileNotFoundError(f"No images found in {img_dir}")

    first_img = cv2.imread(str(image_paths[0]))
    if first_img is None:
        raise RuntimeError(f"Could not read first image: {image_paths[0]}")

    height, width = first_img.shape[:2]
    fps = read_fps_from_seqinfo(seq_dir)

    tracker = create_tracker(
        tracker_type=args.tracker,
        tracker_config=Path(args.tracker_config),
        reid_weights=None,
        device=args.device,
        half=False,
        per_class=False,
    )

    output_video = out_dir / f"{args.name}_{args.tracker}_tracking.mp4"
    output_csv = out_dir / f"{args.name}_{args.tracker}_tracks.csv"

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    video_writer = cv2.VideoWriter(str(output_video), fourcc, fps, (width, height))

    rows = []

    for frame_idx, img_path in enumerate(image_paths, start=1):
        img = cv2.imread(str(img_path))

        frame_dets = det_df[det_df["frame"] == frame_idx]

        if len(frame_dets) > 0:
            dets = frame_dets[["x1", "y1", "x2", "y2", "score", "class_id"]].to_numpy(dtype=float)
            dets = nms_detections(dets, iou_thr=args.nms_iou)
        else:
            dets = np.empty((0, 6), dtype=float)

        tracks = tracker.update(dets, img)

        frame_track_count = 0

        for trk in tracks:
            if len(trk) < 5:
                continue

            x1, y1, x2, y2 = trk[:4]
            track_id = int(trk[4])

            score = float(trk[5]) if len(trk) > 5 else -1.0
            class_id = int(trk[6]) if len(trk) > 6 else 0

            cx = (float(x1) + float(x2)) / 2.0
            cy = (float(y1) + float(y2)) / 2.0

            rows.append([
                frame_idx,
                track_id,
                float(x1),
                float(y1),
                float(x2),
                float(y2),
                score,
                class_id,
                cx,
                cy,
            ])

            draw_track(
                img,
                int(round(x1)),
                int(round(y1)),
                int(round(x2)),
                int(round(y2)),
                track_id,
                score,
            )

            frame_track_count += 1

        video_writer.write(img)

        if frame_idx % 50 == 0 or frame_idx == 1:
            print(f"Frame {frame_idx:04d}: detections after NMS={len(dets)}, tracks={frame_track_count}")

    video_writer.release()

    out_df = pd.DataFrame(
        rows,
        columns=[
            "frame",
            "track_id",
            "x1",
            "y1",
            "x2",
            "y2",
            "score",
            "class_id",
            "cx",
            "cy",
        ],
    )

    out_df.to_csv(output_csv, index=False)

    print()
    print("Saved tracking CSV:", output_csv)
    print("Saved tracking video:", output_video)
    print("Total track rows:", len(out_df))
    print("Unique track IDs:", out_df["track_id"].nunique() if len(out_df) else 0)


if __name__ == "__main__":
    main()
