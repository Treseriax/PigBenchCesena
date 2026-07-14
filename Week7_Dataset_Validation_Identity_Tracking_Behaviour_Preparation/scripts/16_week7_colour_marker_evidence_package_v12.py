from pathlib import Path
from datetime import datetime
import csv
import math

import cv2
import numpy as np
import pandas as pd


ROOT = Path.home() / "PigBench"
W7 = ROOT / "Week7_Dataset_Validation_Identity_Tracking_Behaviour_Preparation"

FINAL_BOXES_PATH = W7 / "outputs" / "final_corrected_gt_pen_boxes_v11" / "week7_final_corrected_gt_pen_boxes_v11_for_colour_matching.csv"
IMG_DIR = W7 / "outputs" / "manual_annotation_v9" / "cvat_coco_import" / "images"

OUT_ROOT = W7 / "outputs" / "colour_identity" / "colour_marker_evidence_v12"
OUT_CROPS = OUT_ROOT / "pig_crops"
OUT_ENHANCED = OUT_ROOT / "pig_crops_enhanced"
OUT_OVERLAYS = OUT_ROOT / "frame_overlays"

for p in [OUT_ROOT, OUT_CROPS, OUT_ENHANCED, OUT_OVERLAYS]:
    p.mkdir(parents=True, exist_ok=True)

OUT_BOX_FEATURES = OUT_ROOT / "week7_colour_marker_evidence_v12_per_box_features.csv"
OUT_MARKER_CANDIDATES = OUT_ROOT / "week7_colour_marker_evidence_v12_marker_candidates.csv"
OUT_FRAME_SUMMARY = OUT_ROOT / "week7_colour_marker_evidence_v12_frame_summary.csv"
OUT_CONTACT = OUT_ROOT / "week7_colour_marker_evidence_v12_frame_overlay_contact_sheet.jpg"
OUT_HTML = OUT_ROOT / "week7_colour_marker_evidence_v12_static_review.html"
OUT_NOTE = W7 / "notes" / "week7_colour_marker_evidence_v12_notes.md"


def safe_to_csv(df, path):
    df.to_csv(
        path,
        index=False,
        quoting=csv.QUOTE_ALL,
        escapechar="\\",
        lineterminator="\n",
    )


def clamp_box(x1, y1, x2, y2, w, h):
    x1 = max(0, min(w - 1, int(round(x1))))
    y1 = max(0, min(h - 1, int(round(y1))))
    x2 = max(1, min(w, int(round(x2))))
    y2 = max(1, min(h, int(round(y2))))

    if x2 <= x1:
        x2 = min(w, x1 + 1)
    if y2 <= y1:
        y2 = min(h, y1 + 1)

    return x1, y1, x2, y2


def enhance_crop(crop):
    lab = cv2.cvtColor(crop, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=2.5, tileGridSize=(8, 8))
    l2 = clahe.apply(l)
    out = cv2.merge([l2, a, b])
    out = cv2.cvtColor(out, cv2.COLOR_LAB2BGR)
    return out


def rough_colour_label(h, s, v):
    # OpenCV HSV hue range: 0-179
    if v < 45:
        return "dark"
    if s < 35 and v > 185:
        return "white_or_light"
    if s < 35:
        return "low_saturation"

    if h < 10 or h >= 170:
        return "red"
    if 10 <= h < 23:
        return "orange"
    if 23 <= h < 36:
        return "yellow"
    if 36 <= h < 86:
        return "green"
    if 86 <= h < 101:
        return "cyan"
    if 101 <= h < 131:
        return "blue"
    if 131 <= h < 161:
        return "purple"
    if 161 <= h < 170:
        return "pink"
    return "unknown"


def extract_marker_candidates(crop, full_x1, full_y1, box_id, scan_frame_id):
    h, w = crop.shape[:2]

    hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
    lab = cv2.cvtColor(crop, cv2.COLOR_BGR2LAB)

    H = hsv[:, :, 0]
    S = hsv[:, :, 1]
    V = hsv[:, :, 2]

    # Coloured markers tend to be higher saturation than pig skin/background.
    # This is intentionally broad; candidates are evidence, not final labels.
    mask = ((S > 55) & (V > 45)).astype(np.uint8) * 255

    # Remove tiny noise.
    kernel = np.ones((3, 3), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    rows = []

    crop_area = max(1, h * w)

    for idx, cnt in enumerate(contours):
        area = float(cv2.contourArea(cnt))

        if area < 12:
            continue

        if area > crop_area * 0.25:
            continue

        x, y, bw, bh = cv2.boundingRect(cnt)

        if bw < 3 or bh < 3:
            continue

        comp_mask = np.zeros((h, w), dtype=np.uint8)
        cv2.drawContours(comp_mask, [cnt], -1, 255, -1)
        pix = comp_mask > 0

        if pix.sum() == 0:
            continue

        mean_h = float(np.mean(H[pix]))
        mean_s = float(np.mean(S[pix]))
        mean_v = float(np.mean(V[pix]))

        mean_bgr = np.mean(crop[pix], axis=0)
        mean_lab = np.mean(lab[pix], axis=0)

        colour_label = rough_colour_label(mean_h, mean_s, mean_v)

        # Score is just a prioritization metric for visual inspection.
        score = float((pix.sum() / crop_area) * (mean_s / 255.0) * 1000.0)

        rows.append({
            "scan_frame_id": scan_frame_id,
            "final_box_id": box_id,
            "candidate_index_raw": idx,
            "candidate_score": score,
            "candidate_area_pixels": int(pix.sum()),
            "candidate_area_fraction_of_crop": float(pix.sum() / crop_area),
            "crop_x1": int(x),
            "crop_y1": int(y),
            "crop_x2": int(x + bw),
            "crop_y2": int(y + bh),
            "full_x1": int(full_x1 + x),
            "full_y1": int(full_y1 + y),
            "full_x2": int(full_x1 + x + bw),
            "full_y2": int(full_y1 + y + bh),
            "mean_h": mean_h,
            "mean_s": mean_s,
            "mean_v": mean_v,
            "mean_b": float(mean_bgr[0]),
            "mean_g": float(mean_bgr[1]),
            "mean_r": float(mean_bgr[2]),
            "mean_l": float(mean_lab[0]),
            "mean_a": float(mean_lab[1]),
            "mean_lab_b": float(mean_lab[2]),
            "rough_colour_label": colour_label,
        })

    rows = sorted(rows, key=lambda r: r["candidate_score"], reverse=True)

    for rank, r in enumerate(rows, start=1):
        r["candidate_rank_in_box"] = rank

    return rows[:5]


def make_contact_sheet(paths, output_path):
    if not paths:
        return False

    sample = paths
    if len(sample) > 16:
        idx = np.linspace(0, len(sample) - 1, 16).round().astype(int)
        sample = [sample[i] for i in idx]

    thumbs = []
    thumb_w, thumb_h = 380, 250

    for p in sample:
        img = cv2.imread(str(p))
        if img is None:
            continue
        thumbs.append(cv2.resize(img, (thumb_w, thumb_h), interpolation=cv2.INTER_AREA))

    if not thumbs:
        return False

    cols = 4
    rows = math.ceil(len(thumbs) / cols)
    sheet = np.full((rows * thumb_h, cols * thumb_w, 3), 255, dtype=np.uint8)

    for i, img in enumerate(thumbs):
        r = i // cols
        c = i % cols
        sheet[r * thumb_h:(r + 1) * thumb_h, c * thumb_w:(c + 1) * thumb_w] = img

    cv2.imwrite(str(output_path), sheet)
    return True


boxes = pd.read_csv(FINAL_BOXES_PATH)

for c in ["x1", "y1", "x2", "y2"]:
    boxes[c] = pd.to_numeric(boxes[c], errors="coerce")

boxes = boxes.dropna(subset=["x1", "y1", "x2", "y2"]).copy()
boxes = boxes.sort_values(["scan_frame_id", "final_box_id"]).copy()

box_feature_rows = []
marker_rows = []
overlay_paths = []
frame_summary_rows = []

for scan_frame_id, g in boxes.groupby("scan_frame_id", sort=True):
    img_path = IMG_DIR / f"{scan_frame_id}.jpg"

    if not img_path.exists():
        continue

    img = cv2.imread(str(img_path))

    if img is None:
        continue

    H_img, W_img = img.shape[:2]
    overlay = img.copy()

    frame_marker_candidate_count = 0

    for _, b in g.iterrows():
        box_id = str(b["final_box_id"])

        x1, y1, x2, y2 = clamp_box(b["x1"], b["y1"], b["x2"], b["y2"], W_img, H_img)

        crop = img[y1:y2, x1:x2].copy()
        enhanced = enhance_crop(crop)

        crop_name = f"{scan_frame_id}_{box_id}.jpg"
        crop_path = OUT_CROPS / crop_name
        enhanced_path = OUT_ENHANCED / crop_name

        cv2.imwrite(str(crop_path), crop)
        cv2.imwrite(str(enhanced_path), enhanced)

        hsv_crop = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
        lab_crop = cv2.cvtColor(crop, cv2.COLOR_BGR2LAB)

        mean_h = float(np.mean(hsv_crop[:, :, 0]))
        mean_s = float(np.mean(hsv_crop[:, :, 1]))
        mean_v = float(np.mean(hsv_crop[:, :, 2]))

        mean_lab = np.mean(lab_crop.reshape(-1, 3), axis=0)

        candidates = extract_marker_candidates(crop, x1, y1, box_id, scan_frame_id)
        frame_marker_candidate_count += len(candidates)

        for cand in candidates:
            marker_rows.append(cand)

        top_label = candidates[0]["rough_colour_label"] if candidates else "none"
        top_score = candidates[0]["candidate_score"] if candidates else 0.0

        box_feature_rows.append({
            "scan_frame_id": scan_frame_id,
            "final_box_id": box_id,
            "x1": x1,
            "y1": y1,
            "x2": x2,
            "y2": y2,
            "crop_path": str(crop_path),
            "enhanced_crop_path": str(enhanced_path),
            "mean_h": mean_h,
            "mean_s": mean_s,
            "mean_v": mean_v,
            "mean_l": float(mean_lab[0]),
            "mean_a": float(mean_lab[1]),
            "mean_lab_b": float(mean_lab[2]),
            "marker_candidate_count": len(candidates),
            "top_marker_candidate_colour": top_label,
            "top_marker_candidate_score": top_score,
        })

        # Draw final box.
        cv2.rectangle(overlay, (x1, y1), (x2, y2), (0, 220, 0), 2)
        cv2.putText(
            overlay,
            box_id.replace("scanframe_", "sf_"),
            (x1, max(18, y1 - 5)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.42,
            (0, 220, 0),
            1,
            cv2.LINE_AA,
        )

        # Draw top marker candidates.
        for cand in candidates[:3]:
            cx1 = int(cand["full_x1"])
            cy1 = int(cand["full_y1"])
            cx2 = int(cand["full_x2"])
            cy2 = int(cand["full_y2"])

            cv2.rectangle(overlay, (cx1, cy1), (cx2, cy2), (255, 180, 0), 2)
            cv2.putText(
                overlay,
                cand["rough_colour_label"],
                (cx1, max(18, cy1 - 4)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.38,
                (255, 180, 0),
                1,
                cv2.LINE_AA,
            )

    title = f"{scan_frame_id} | final boxes={len(g)} | marker candidates={frame_marker_candidate_count}"
    cv2.putText(
        overlay,
        title,
        (8, 25),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.58,
        (255, 255, 255),
        2,
        cv2.LINE_AA,
    )

    out_overlay = OUT_OVERLAYS / f"{scan_frame_id}_colour_marker_evidence_v12.jpg"
    cv2.imwrite(str(out_overlay), overlay)
    overlay_paths.append(out_overlay)

    frame_summary_rows.append({
        "scan_frame_id": scan_frame_id,
        "final_box_count": len(g),
        "marker_candidate_count": frame_marker_candidate_count,
        "overlay_path": str(out_overlay),
    })

box_features = pd.DataFrame(box_feature_rows)
marker_candidates = pd.DataFrame(marker_rows)
frame_summary = pd.DataFrame(frame_summary_rows)

safe_to_csv(box_features, OUT_BOX_FEATURES)
safe_to_csv(marker_candidates, OUT_MARKER_CANDIDATES)
safe_to_csv(frame_summary, OUT_FRAME_SUMMARY)

contact_ok = make_contact_sheet(overlay_paths, OUT_CONTACT)

cards = []

for _, r in frame_summary.sort_values("scan_frame_id").iterrows():
    p = Path(str(r["overlay_path"]))
    src = "frame_overlays/" + p.name

    cards.append(
        f"""
        <div class="frame-card">
          <h3>{r['scan_frame_id']}</h3>
          <p>final boxes={r['final_box_count']} | marker candidates={r['marker_candidate_count']}</p>
          <img src="{src}">
        </div>
        """
    )

html = f"""
<!DOCTYPE html>
<html>
<head>
  <meta charset="utf-8">
  <title>Week 7 Colour Marker Evidence v12</title>
  <style>
    body {{
      font-family: Arial, sans-serif;
      margin: 24px;
      background: #f7f7f7;
    }}
    .legend {{
      background: white;
      padding: 14px;
      border-left: 6px solid #222;
      border-radius: 8px;
      margin-bottom: 20px;
    }}
    .frame-card {{
      background: white;
      padding: 12px;
      margin: 18px 0;
      border-radius: 8px;
      box-shadow: 0 1px 4px rgba(0,0,0,0.15);
    }}
    img {{
      max-width: 100%;
      border: 1px solid #ddd;
      display: block;
    }}
  </style>
</head>
<body>
  <h1>Week 7 Colour Marker Evidence v12</h1>
  <div class="legend">
    <p>Green boxes: final corrected GT-pen pig boxes.</p>
    <p>Blue/orange small boxes: high-saturation marker candidate regions inside each pig crop.</p>
    <p>This is evidence for colour identity matching, not the final colour assignment yet.</p>
  </div>
  {''.join(cards)}
</body>
</html>
"""

OUT_HTML.write_text(html)

OUT_NOTE.write_text(
    "# Week 7 Colour Marker Evidence v12\n\n"
    "## Purpose\n\n"
    "This step extracts crops and candidate colour-marker evidence from the manually corrected GT-pen pig boxes. "
    "It prepares the input for colour identity assignment.\n\n"
    "## Inputs\n\n"
    f"- Final corrected boxes: `{FINAL_BOXES_PATH}`\n\n"
    "## Outputs\n\n"
    f"- Per-box features: `{OUT_BOX_FEATURES}`\n"
    f"- Marker candidates: `{OUT_MARKER_CANDIDATES}`\n"
    f"- Frame summary: `{OUT_FRAME_SUMMARY}`\n"
    f"- Contact sheet: `{OUT_CONTACT}`\n"
    f"- Static HTML: `{OUT_HTML}`\n\n"
    "## Interpretation\n\n"
    "The marker candidates are not final labels. They are automatically extracted visual evidence for the next colour matching step.\n"
)

print("Saved:")
print(OUT_BOX_FEATURES)
print(OUT_MARKER_CANDIDATES)
print(OUT_FRAME_SUMMARY)
print(OUT_CONTACT)
print(OUT_HTML)
print(OUT_NOTE)

print()
print("=== colour marker evidence v12 summary ===")
print({
    "final_boxes_processed": len(box_features),
    "marker_candidates": len(marker_candidates),
    "frames": frame_summary["scan_frame_id"].nunique(),
    "contact_sheet_generated": contact_ok,
})
