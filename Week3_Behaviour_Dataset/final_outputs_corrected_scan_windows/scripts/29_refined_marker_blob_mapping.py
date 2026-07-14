from pathlib import Path
import cv2
import numpy as np
import pandas as pd
import json


SEQ_ROOT = Path("Week3_Behaviour_Dataset/data/scan_window_sequences")
TRACK_ROOT = Path("Week3_Behaviour_Dataset/outputs/scan_window_aligned/tracking_with_time")
DOMINANT_CSV = Path("Week3_Behaviour_Dataset/outputs/scan_window_aligned/identity_mapping/dominant_track_id_to_colour_mapping_table.csv")
LABEL_CSV = Path("Week3_Behaviour_Dataset/outputs/scan_window_aligned/behaviour_labels/scan_window_behaviour_labels.csv")

OUT_DIR = Path("Week3_Behaviour_Dataset/outputs/scan_window_aligned/identity_mapping/refined_marker_mapping")
OUT_DIR.mkdir(parents=True, exist_ok=True)

dominant = pd.read_csv(DOMINANT_CSV)
labels = pd.read_csv(LABEL_CSV)

COLOUR_RANGES = {
    "green": [
        ((35, 70, 45), (88, 255, 255)),
    ],
    "blue": [
        ((92, 70, 45), (130, 255, 255)),
    ],
    "purple": [
        ((128, 45, 35), (165, 255, 255)),
    ],
    "red": [
        ((0, 65, 45), (12, 255, 255)),
        ((168, 65, 45), (179, 255, 255)),
    ],
}


def make_mask(hsv, ranges):
    total = np.zeros(hsv.shape[:2], dtype=np.uint8)
    for lo, hi in ranges:
        lo = np.array(lo, dtype=np.uint8)
        hi = np.array(hi, dtype=np.uint8)
        total = cv2.bitwise_or(total, cv2.inRange(hsv, lo, hi))

    kernel = np.ones((3, 3), np.uint8)
    total = cv2.morphologyEx(total, cv2.MORPH_OPEN, kernel)
    total = cv2.morphologyEx(total, cv2.MORPH_CLOSE, kernel)
    return total


def crop_bbox(img, row):
    h, w = img.shape[:2]
    x1, y1, x2, y2 = float(row["x1"]), float(row["y1"]), float(row["x2"]), float(row["y2"])

    # Slightly shrink crop to avoid background/bars around bbox edges.
    bw = x2 - x1
    bh = y2 - y1
    x1 = int(max(0, x1 + 0.04 * bw))
    x2 = int(min(w - 1, x2 - 0.04 * bw))
    y1 = int(max(0, y1 + 0.04 * bh))
    y2 = int(min(h - 1, y2 - 0.04 * bh))

    if x2 <= x1 or y2 <= y1:
        return None

    return img[y1:y2, x1:x2].copy()


def blob_score_for_colour(crop, colour):
    if crop is None or crop.size == 0:
        return 0.0, 0, None

    h, w = crop.shape[:2]
    crop_area = h * w
    if crop_area < 500:
        return 0.0, 0, None

    hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
    mask = make_mask(hsv, COLOUR_RANGES[colour])

    num_labels, labels_cc, stats, centroids = cv2.connectedComponentsWithStats(mask, connectivity=8)

    best_score = 0.0
    best_blob = None
    blob_count = 0

    for i in range(1, num_labels):
        x, y, bw, bh, area = stats[i]
        if area < 20:
            continue

        # Reject very large regions; markers should be compact, not whole body/background.
        if area > 0.08 * crop_area:
            continue

        # Reject long thin lines, often pen bars / UI artefacts.
        aspect = max(bw / max(bh, 1), bh / max(bw, 1))
        if aspect > 8:
            continue

        # Reject blobs touching crop boundary too much; often background artefact.
        touches_edge = x <= 2 or y <= 2 or (x + bw) >= (w - 3) or (y + bh) >= (h - 3)
        if touches_edge and area < 80:
            continue

        blob_count += 1

        # Score rewards visible compact marker but normalizes by crop size.
        score = area / crop_area

        if score > best_score:
            best_score = score
            best_blob = (int(x), int(y), int(bw), int(bh), int(area))

    return best_score, blob_count, best_blob


def decide(scores, supports):
    sorted_scores = sorted(scores.items(), key=lambda x: x[1], reverse=True)
    best_colour, best_score = sorted_scores[0]
    second = sorted_scores[1][1] if len(sorted_scores) > 1 else 0.0
    margin = best_score / (second + 1e-9)
    support = supports.get(best_colour, 0)

    # Much stricter than previous script.
    if best_score < 0.0009 or support < 3:
        return "no_reliable_marker", "low", margin, "leave_unverified"

    if best_colour in ["green", "blue", "purple"]:
        if best_score >= 0.0025 and margin >= 2.0 and support >= 4:
            return best_colour, "high", margin, "can_auto_assign_after_visual_review"
        if best_score >= 0.0013 and margin >= 1.5 and support >= 3:
            return best_colour, "medium", margin, "manual_check_recommended"
        return best_colour, "low", margin, "leave_unverified"

    if best_colour == "red":
        if best_score >= 0.0013 and margin >= 1.5 and support >= 3:
            return "red_candidate", "medium", margin, "requires_red_neck_tail_manual_check"
        return "red_candidate", "low", margin, "leave_unverified"

    return "unknown", "low", margin, "leave_unverified"


rows = []
tiles = []

for seg in sorted(dominant["segment_id"].unique()):
    print("===", seg, "===")
    img_dir = SEQ_ROOT / seg / "img1"
    track_csv = TRACK_ROOT / seg / f"{seg}_bytetrack_tracks_with_excel_window.csv"

    tracks = pd.read_csv(track_csv)
    seg_dom = dominant[dominant["segment_id"] == seg].copy()

    for _, dom in seg_dom.iterrows():
        tid = int(dom["track_id"])
        tdf = tracks[tracks["track_id"].astype(int) == tid].sort_values("frame").copy()
        if tdf.empty:
            continue

        n_samples = min(18, len(tdf))
        sample_idx = np.linspace(0, len(tdf) - 1, n_samples).astype(int)
        sample_rows = tdf.iloc[sample_idx]

        frame_scores = []
        examples = []

        for _, r in sample_rows.iterrows():
            frame_idx = int(r["frame"])
            img = cv2.imread(str(img_dir / f"{frame_idx:08d}.jpg"))
            if img is None:
                continue

            crop = crop_bbox(img, r)
            if crop is None:
                continue

            colour_scores = {}
            blob_infos = {}

            for colour in COLOUR_RANGES:
                s, count, blob = blob_score_for_colour(crop, colour)
                colour_scores[colour] = s
                blob_infos[colour] = blob

            frame_scores.append(colour_scores)

            if len(examples) < 3:
                vis = crop.copy()
                # Draw best blob for visually strongest colour in this crop.
                best_colour = max(colour_scores, key=colour_scores.get)
                blob = blob_infos.get(best_colour)
                if blob:
                    x, y, bw, bh, area = blob
                    cv2.rectangle(vis, (x, y), (x + bw, y + bh), (0, 255, 255), 2)
                    cv2.putText(vis, best_colour, (x, max(16, y - 4)), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 255), 1)

                vis = cv2.resize(vis, (180, 120))
                examples.append(vis)

        if not frame_scores:
            continue

        fs = pd.DataFrame(frame_scores)

        scores = {}
        supports = {}

        for colour in COLOUR_RANGES:
            vals = fs[colour].fillna(0).values
            vals_sorted = np.sort(vals)[::-1]
            k = min(5, len(vals_sorted))
            scores[colour] = float(vals_sorted[:k].mean()) if k > 0 else 0.0
            supports[colour] = int((vals > 0.0008).sum())

        pred, conf, margin, action = decide(scores, supports)

        seg_labels = labels[labels["segment_id"] == seg]
        available = "; ".join(
            f"{r['colour']}={r['behaviour_code']}({r['behaviour_label']})"
            for _, r in seg_labels.iterrows()
        )

        rows.append({
            "segment_id": seg,
            "track_id": tid,
            "first_frame": int(dom["first_frame"]),
            "last_frame": int(dom["last_frame"]),
            "num_frames": int(dom["num_frames"]),
            "mean_score": float(dom["mean_score"]),
            "mean_cx": float(dom["mean_cx"]),
            "mean_cy": float(dom["mean_cy"]),
            "green_blob_score": round(scores["green"], 7),
            "blue_blob_score": round(scores["blue"], 7),
            "purple_blob_score": round(scores["purple"], 7),
            "red_blob_score": round(scores["red"], 7),
            "green_support": supports["green"],
            "blue_support": supports["blue"],
            "purple_support": supports["purple"],
            "red_support": supports["red"],
            "predicted_colour": pred,
            "confidence_level": conf,
            "score_margin": round(float(margin), 3),
            "recommended_action": action,
            "available_excel_labels": available
        })

        if examples:
            tile = np.ones((190, 540, 3), dtype=np.uint8) * 255
            t1 = f"{seg} | ID {tid} | {pred} | {conf}"
            t2 = f"G:{scores['green']:.5f} B:{scores['blue']:.5f} P:{scores['purple']:.5f} R:{scores['red']:.5f}"
            t3 = f"support G/B/P/R: {supports['green']}/{supports['blue']}/{supports['purple']}/{supports['red']}"
            cv2.putText(tile, t1, (8, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.52, (0, 0, 0), 1, cv2.LINE_AA)
            cv2.putText(tile, t2, (8, 45), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (0, 0, 0), 1, cv2.LINE_AA)
            cv2.putText(tile, t3, (8, 66), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 0), 1, cv2.LINE_AA)

            x = 0
            for ex in examples:
                tile[70:190, x:x+180] = ex
                x += 180
            tiles.append(tile)

df = pd.DataFrame(rows)

out_csv = OUT_DIR / "refined_marker_mapping_candidates.csv"
out_json = OUT_DIR / "refined_marker_mapping_candidates.json"

df.to_csv(out_csv, index=False)
with open(out_json, "w") as f:
    json.dump(df.to_dict(orient="records"), f, indent=2)

print("Saved:", out_csv)
print("Saved:", out_json)

if not df.empty:
    print()
    print(df.groupby(["predicted_colour", "confidence_level", "recommended_action"]).size().reset_index(name="count").to_string(index=False))

if tiles:
    cols = 2
    rows_img = []
    for i in range(0, len(tiles), cols):
        row = tiles[i:i+cols]
        while len(row) < cols:
            row.append(np.ones_like(tiles[0]) * 255)
        rows_img.append(np.hstack(row))
    montage = np.vstack(rows_img)
    montage_path = OUT_DIR / "refined_marker_candidate_crop_montage.jpg"
    cv2.imwrite(str(montage_path), montage)
    print("Saved montage:", montage_path)

review = df[df["recommended_action"] != "leave_unverified"].copy()
review_path = OUT_DIR / "refined_marker_candidates_review_needed.csv"
review.to_csv(review_path, index=False)
print("Saved review table:", review_path)
