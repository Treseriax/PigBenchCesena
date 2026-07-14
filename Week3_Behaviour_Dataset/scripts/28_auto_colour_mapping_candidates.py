from pathlib import Path
import cv2
import numpy as np
import pandas as pd
import json


SEQ_ROOT = Path("Week3_Behaviour_Dataset/data/scan_window_sequences")
TRACK_ROOT = Path("Week3_Behaviour_Dataset/outputs/scan_window_aligned/tracking_with_time")
DOMINANT_CSV = Path("Week3_Behaviour_Dataset/outputs/scan_window_aligned/identity_mapping/dominant_track_id_to_colour_mapping_table.csv")
LABEL_CSV = Path("Week3_Behaviour_Dataset/outputs/scan_window_aligned/behaviour_labels/scan_window_behaviour_labels.csv")

OUT_DIR = Path("Week3_Behaviour_Dataset/outputs/scan_window_aligned/identity_mapping/auto_colour_mapping")
OUT_DIR.mkdir(parents=True, exist_ok=True)

dominant = pd.read_csv(DOMINANT_CSV)
labels = pd.read_csv(LABEL_CSV)

segments = sorted(dominant["segment_id"].unique().tolist())


# HSV masks.
# OpenCV HSV hue range: 0-179.
COLOUR_RANGES = {
    "green": [
        ((35, 50, 40), (90, 255, 255)),
    ],
    "blue": [
        ((90, 45, 40), (135, 255, 255)),
    ],
    "purple": [
        ((125, 35, 35), (165, 255, 255)),
    ],
    "red": [
        ((0, 45, 35), (12, 255, 255)),
        ((168, 45, 35), (179, 255, 255)),
    ],
}


def make_mask(hsv, ranges):
    mask_total = np.zeros(hsv.shape[:2], dtype=np.uint8)

    for lower, upper in ranges:
        lower = np.array(lower, dtype=np.uint8)
        upper = np.array(upper, dtype=np.uint8)
        mask = cv2.inRange(hsv, lower, upper)
        mask_total = cv2.bitwise_or(mask_total, mask)

    kernel = np.ones((3, 3), np.uint8)
    mask_total = cv2.morphologyEx(mask_total, cv2.MORPH_OPEN, kernel)
    mask_total = cv2.morphologyEx(mask_total, cv2.MORPH_CLOSE, kernel)

    return mask_total


def crop_bbox(img, row, pad_ratio=0.04):
    h, w = img.shape[:2]

    x1 = float(row["x1"])
    y1 = float(row["y1"])
    x2 = float(row["x2"])
    y2 = float(row["y2"])

    bw = x2 - x1
    bh = y2 - y1

    pad_x = bw * pad_ratio
    pad_y = bh * pad_ratio

    x1 = max(0, int(round(x1 - pad_x)))
    y1 = max(0, int(round(y1 - pad_y)))
    x2 = min(w - 1, int(round(x2 + pad_x)))
    y2 = min(h - 1, int(round(y2 + pad_y)))

    if x2 <= x1 or y2 <= y1:
        return None

    return img[y1:y2, x1:x2].copy()


def colour_ratios(crop):
    if crop is None or crop.size == 0:
        return {c: 0.0 for c in COLOUR_RANGES}

    # Ignore extremely tiny crops
    area = crop.shape[0] * crop.shape[1]
    if area < 100:
        return {c: 0.0 for c in COLOUR_RANGES}

    hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)

    ratios = {}
    for colour, ranges in COLOUR_RANGES.items():
        mask = make_mask(hsv, ranges)
        pixels = int(np.count_nonzero(mask))
        ratios[colour] = pixels / area

    return ratios


def decide_colour(scores, support_counts):
    sorted_scores = sorted(scores.items(), key=lambda x: x[1], reverse=True)
    best_colour, best_score = sorted_scores[0]
    second_score = sorted_scores[1][1] if len(sorted_scores) > 1 else 0.0

    margin = best_score / (second_score + 1e-9)
    support = support_counts.get(best_colour, 0)

    if best_score < 0.0015 or support < 2:
        return {
            "predicted_colour": "no_reliable_colour_marker",
            "confidence_level": "low",
            "recommended_action": "leave_unverified",
            "score_margin": margin,
        }

    if best_colour in ["green", "blue", "purple"]:
        if best_score >= 0.006 and margin >= 1.8 and support >= 3:
            confidence = "high"
            action = "can_auto_assign_if_visually_reasonable"
        elif best_score >= 0.003 and margin >= 1.3:
            confidence = "medium"
            action = "manual_check_recommended"
        else:
            confidence = "low"
            action = "leave_unverified"

        return {
            "predicted_colour": best_colour,
            "confidence_level": confidence,
            "recommended_action": action,
            "score_margin": margin,
        }

    if best_colour == "red":
        if best_score >= 0.004 and margin >= 1.4:
            confidence = "medium"
            action = "red_candidate_requires_neck_tail_manual_check"
        else:
            confidence = "low"
            action = "leave_unverified"

        return {
            "predicted_colour": "red_candidate",
            "confidence_level": confidence,
            "recommended_action": action,
            "score_margin": margin,
        }

    return {
        "predicted_colour": "unknown",
        "confidence_level": "low",
        "recommended_action": "leave_unverified",
        "score_margin": margin,
    }


all_rows = []
montage_tiles = []

for seg in segments:
    print(f"=== Processing {seg} ===")

    seq_dir = SEQ_ROOT / seg
    img_dir = seq_dir / "img1"
    track_csv = TRACK_ROOT / seg / f"{seg}_bytetrack_tracks_with_excel_window.csv"

    tracks = pd.read_csv(track_csv)
    seg_dominant = dominant[dominant["segment_id"] == seg].copy()

    for _, dom_row in seg_dominant.iterrows():
        track_id = int(dom_row["track_id"])

        track_df = tracks[tracks["track_id"].astype(int) == track_id].sort_values("frame").copy()

        if track_df.empty:
            continue

        # Sample up to 20 frames evenly across the track.
        n_samples = min(20, len(track_df))
        sample_indices = np.linspace(0, len(track_df) - 1, n_samples).astype(int)
        sample_rows = track_df.iloc[sample_indices]

        per_frame_ratios = []
        crop_examples = []

        for _, row in sample_rows.iterrows():
            frame_idx = int(row["frame"])
            img_path = img_dir / f"{frame_idx:08d}.jpg"

            img = cv2.imread(str(img_path))
            if img is None:
                continue

            crop = crop_bbox(img, row)

            if crop is None:
                continue

            ratios = colour_ratios(crop)
            ratios["frame"] = frame_idx
            per_frame_ratios.append(ratios)

            if len(crop_examples) < 3:
                crop_vis = crop.copy()
                crop_vis = cv2.resize(crop_vis, (180, 120))
                crop_examples.append(crop_vis)

        if not per_frame_ratios:
            continue

        ratio_df = pd.DataFrame(per_frame_ratios)

        scores = {}
        support_counts = {}

        for colour in COLOUR_RANGES.keys():
            values = ratio_df[colour].fillna(0).values
            values_sorted = np.sort(values)[::-1]

            # Top-k mean is more robust because colour markers may be visible only in some frames.
            k = min(5, len(values_sorted))
            topk_mean = float(values_sorted[:k].mean()) if k > 0 else 0.0

            scores[colour] = topk_mean
            support_counts[colour] = int((values > 0.0015).sum())

        decision = decide_colour(scores, support_counts)

        seg_labels = labels[labels["segment_id"] == seg]
        available_labels = "; ".join(
            f"{r['colour']}={r['behaviour_code']}({r['behaviour_label']})"
            for _, r in seg_labels.iterrows()
        )

        row_out = {
            "segment_id": seg,
            "track_id": track_id,
            "first_frame": int(dom_row["first_frame"]),
            "last_frame": int(dom_row["last_frame"]),
            "num_frames": int(dom_row["num_frames"]),
            "mean_score": float(dom_row["mean_score"]),
            "mean_cx": float(dom_row["mean_cx"]),
            "mean_cy": float(dom_row["mean_cy"]),
            "green_score": round(scores["green"], 6),
            "blue_score": round(scores["blue"], 6),
            "purple_score": round(scores["purple"], 6),
            "red_score": round(scores["red"], 6),
            "green_support_frames": support_counts["green"],
            "blue_support_frames": support_counts["blue"],
            "purple_support_frames": support_counts["purple"],
            "red_support_frames": support_counts["red"],
            "predicted_colour": decision["predicted_colour"],
            "confidence_level": decision["confidence_level"],
            "score_margin": round(float(decision["score_margin"]), 3),
            "recommended_action": decision["recommended_action"],
            "available_excel_labels": available_labels,
            "notes": (
                "HSV-based colour-marker candidate. Red needs manual red-neck/red-tail verification; "
                "no reliable marker should remain unverified."
            )
        }

        all_rows.append(row_out)

        # Create small crop montage tile for visual auditing
        if crop_examples:
            tile_w = 540
            header_h = 70
            tile_h = 120
            tile = np.ones((header_h + tile_h, tile_w, 3), dtype=np.uint8) * 255

            text1 = f"{seg} | ID {track_id} | pred={decision['predicted_colour']} | {decision['confidence_level']}"
            text2 = f"G:{scores['green']:.4f} B:{scores['blue']:.4f} P:{scores['purple']:.4f} R:{scores['red']:.4f}"

            cv2.putText(tile, text1, (8, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.52, (0, 0, 0), 1, cv2.LINE_AA)
            cv2.putText(tile, text2, (8, 52), cv2.FONT_HERSHEY_SIMPLEX, 0.52, (0, 0, 0), 1, cv2.LINE_AA)

            x = 0
            for c in crop_examples:
                tile[header_h:header_h + tile_h, x:x + 180] = c
                x += 180

            montage_tiles.append(tile)

candidates = pd.DataFrame(all_rows)
out_csv = OUT_DIR / "auto_colour_mapping_candidates.csv"
out_json = OUT_DIR / "auto_colour_mapping_candidates.json"

candidates.to_csv(out_csv, index=False)

with open(out_json, "w") as f:
    json.dump(candidates.to_dict(orient="records"), f, indent=2)

print()
print("Saved:", out_csv)
print("Saved:", out_json)

if not candidates.empty:
    print()
    print("=== Candidate summary ===")
    print(
        candidates
        .groupby(["predicted_colour", "confidence_level", "recommended_action"])
        .size()
        .reset_index(name="count")
        .to_string(index=False)
    )

# Make visual crop montage
if montage_tiles:
    cols = 2
    rows = []
    for i in range(0, len(montage_tiles), cols):
        row_tiles = montage_tiles[i:i + cols]
        while len(row_tiles) < cols:
            row_tiles.append(np.ones_like(montage_tiles[0]) * 255)
        rows.append(np.hstack(row_tiles))

    montage = np.vstack(rows)
    montage_path = OUT_DIR / "auto_colour_candidate_crop_montage.jpg"
    cv2.imwrite(str(montage_path), montage)
    print("Saved crop montage:", montage_path)

# Also save a high-confidence-only table
if not candidates.empty:
    high = candidates[
        (
            candidates["confidence_level"].isin(["high", "medium"])
        ) &
        (
            candidates["recommended_action"] != "leave_unverified"
        )
    ].copy()

    high_path = OUT_DIR / "auto_colour_mapping_candidates_review_needed.csv"
    high.to_csv(high_path, index=False)
    print("Saved review table:", high_path)
