import argparse
import csv
import sys
from pathlib import Path

import cv2
import torch
from mmdet.apis import init_detector, inference_detector
from mmdet.utils import register_all_modules


def draw_box(img, x1, y1, x2, y2, score):
    cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
    cv2.putText(
        img,
        f"pig {score:.2f}",
        (x1, max(20, y1 - 5)),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.5,
        (0, 255, 0),
        2,
        cv2.LINE_AA,
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seq-dir", required=True)
    parser.add_argument("--name", required=True)
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--score-thr", type=float, default=0.05)
    parser.add_argument("--vis-thr", type=float, default=0.25)
    args = parser.parse_args()

    repo_root = Path.cwd()
    sys.path.insert(0, str(repo_root))
    sys.path.insert(0, str(repo_root / "detection"))

    seq_dir = Path(args.seq_dir)
    img_dir = seq_dir / "img1"
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    output_csv = out_dir / f"{args.name}_yolov8s_detections.csv"
    output_video = out_dir / f"{args.name}_yolov8s_detections.mp4"

    image_paths = sorted(img_dir.glob("*.jpg"))
    if not image_paths:
        raise FileNotFoundError(f"No jpg images found in {img_dir}")

    first_img = cv2.imread(str(image_paths[0]))
    if first_img is None:
        raise RuntimeError(f"Could not read first image: {image_paths[0]}")

    height, width = first_img.shape[:2]

    register_all_modules(init_default_scope=False)

    print("Sequence:", args.name)
    print("Frames:", len(image_paths))
    print("Image size:", width, "x", height)
    print("Config:", args.config)
    print("Checkpoint:", args.checkpoint)
    print("Device:", args.device)
    print("Score threshold:", args.score_thr)

    model = init_detector(args.config, args.checkpoint, device=args.device)

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    video_writer = cv2.VideoWriter(str(output_video), fourcc, 25, (width, height))

    rows = []

    with torch.no_grad():
        for frame_idx, img_path in enumerate(image_paths, start=1):
            img = cv2.imread(str(img_path))
            result = inference_detector(model, img)

            pred = result.pred_instances
            bboxes = pred.bboxes.detach().cpu().numpy()
            scores = pred.scores.detach().cpu().numpy()
            labels = pred.labels.detach().cpu().numpy()

            kept = 0

            for bbox, score, label in zip(bboxes, scores, labels):
                if float(score) < args.score_thr:
                    continue

                x1, y1, x2, y2 = bbox.tolist()

                rows.append([
                    frame_idx,
                    float(x1),
                    float(y1),
                    float(x2),
                    float(y2),
                    float(score),
                    int(label),
                ])

                if float(score) >= args.vis_thr:
                    draw_box(
                        img,
                        int(round(x1)),
                        int(round(y1)),
                        int(round(x2)),
                        int(round(y2)),
                        float(score),
                    )

                kept += 1

            video_writer.write(img)

            if frame_idx % 50 == 0 or frame_idx == 1:
                print(f"Frame {frame_idx:04d}: kept {kept} detections")

    video_writer.release()

    with open(output_csv, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["frame", "x1", "y1", "x2", "y2", "score", "class_id"])
        writer.writerows(rows)

    print()
    print("Saved detection CSV:", output_csv)
    print("Saved annotated video:", output_video)
    print("Total detections:", len(rows))


if __name__ == "__main__":
    main()
