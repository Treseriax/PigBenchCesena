from pathlib import Path
from datetime import datetime
import csv
import math
import os

import cv2
import numpy as np
import pandas as pd


ROOT = Path.home() / "PigBench"
W7 = ROOT / "Week7_Dataset_Validation_Identity_Tracking_Behaviour_Preparation"

FINAL_BOXES_PATH = W7 / "outputs" / "final_corrected_gt_pen_boxes_v11" / "week7_final_corrected_gt_pen_boxes_v11_for_colour_matching.csv"
IMG_DIR = W7 / "outputs" / "manual_annotation_v9" / "cvat_coco_import" / "images"

SAM_CHECKPOINT = Path("/work/models/SAM/sam_vit_b_01ec64.pth")

OUT_ROOT = W7 / "outputs" / "colour_identity" / "segmented_enhanced_colour_evidence_v15"
OUT_MASKS = OUT_ROOT / "sam_or_grabcut_masks"
OUT_CROPS = OUT_ROOT / "masked_crops"
OUT_ENHANCED = OUT_ROOT / "enhanced_crops"
OUT_OVERLAYS = OUT_ROOT / "frame_overlays"

for p in [OUT_ROOT, OUT_MASKS, OUT_CROPS, OUT_ENHANCED, OUT_OVERLAYS]:
    p.mkdir(parents=True, exist_ok=True)

OUT_BOX_MASK_QA = OUT_ROOT / "week7_segmented_enhanced_colour_evidence_v15_box_mask_qa.csv"
OUT_COLOUR_SCORES = OUT_ROOT / "week7_segmented_enhanced_colour_evidence_v15_colour_scores.csv"
OUT_MARKER_CANDIDATES = OUT_ROOT / "week7_segmented_enhanced_colour_evidence_v15_marker_candidates.csv"
OUT_FRAME_SUMMARY = OUT_ROOT / "week7_segmented_enhanced_colour_evidence_v15_frame_summary.csv"
OUT_REVIEW = OUT_ROOT / "week7_segmented_enhanced_colour_evidence_v15_low_confidence_review.csv"
OUT_CONTACT = OUT_ROOT / "week7_segmented_enhanced_colour_evidence_v15_contact_sheet.jpg"
OUT_HTML = OUT_ROOT / "week7_segmented_enhanced_colour_evidence_v15_static_review.html"
OUT_NOTE = W7 / "notes" / "week7_segmented_enhanced_colour_evidence_v15_notes.md"


VALID_COLOURS = ["blue", "green", "cyan", "red", "pink", "purple"]

# OpenCV HSV hue range: 0-179.
HSV_RANGES = {
    "red": [(0, 12), (168, 179)],
    "pink": [(145, 169)],
    "purple": [(125, 155)],
    "blue": [(98, 132)],
    "cyan": [(82, 100)],
    "green": [(36, 85)],
}

COLOUR_BGR = {
    "blue": (255, 0, 0),
    "green": (0, 190, 0),
    "cyan": (255, 255, 0),
    "red": (0, 0, 255),
    "pink": (203, 120, 255),
    "purple": (180, 0, 180),
    "unknown": (160, 160, 160),
}


def safe_to_csv(df, path):
    df.to_csv(
        path,
        index=False,
        quoting=csv.QUOTE_ALL,
        escapechar="\\",
        lineterminator="\n",
    )


def clamp_box(x1, y1, x2, y2, w, h):
    x1 = max(0, min(w - 1, int(round(float(x1)))))
    y1 = max(0, min(h - 1, int(round(float(y1)))))
    x2 = max(1, min(w, int(round(float(x2)))))
    y2 = max(1, min(h, int(round(float(y2)))))

    if x2 <= x1:
        x2 = min(w, x1 + 1)
    if y2 <= y1:
        y2 = min(h, y1 + 1)

    return x1, y1, x2, y2


def load_sam_predictor():
    if not SAM_CHECKPOINT.exists():
        return None, "sam_checkpoint_missing"

    try:
        import torch
        from segment_anything import sam_model_registry, SamPredictor

        device = "cpu"
        sam = sam_model_registry["vit_b"](checkpoint=str(SAM_CHECKPOINT))
        sam.to(device=device)
        predictor = SamPredictor(sam)
        return predictor, "sam_vit_b_cpu"
    except Exception as e:
        return None, f"sam_unavailable_fallback_grabcut: {repr(e)}"


def fallback_grabcut_mask(img, box):
    h, w = img.shape[:2]
    x1, y1, x2, y2 = box

    mask = np.zeros((h, w), np.uint8)
    bgd = np.zeros((1, 65), np.float64)
    fgd = np.zeros((1, 65), np.float64)

    rect = (x1, y1, max(1, x2 - x1), max(1, y2 - y1))

    try:
        cv2.grabCut(img, mask, rect, bgd, fgd, 3, cv2.GC_INIT_WITH_RECT)
        final = np.where((mask == 2) | (mask == 0), 0, 1).astype(bool)

        if final.sum() < 30:
            raise RuntimeError("grabcut_too_small")

        return final, "grabcut"
    except Exception:
        final = np.zeros((h, w), dtype=bool)
        final[y1:y2, x1:x2] = True
        return final, "box_fill_fallback"


def enhance_lab_clahe(crop):
    lab = cv2.cvtColor(crop, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    l2 = clahe.apply(l)
    out = cv2.merge([l2, a, b])
    return cv2.cvtColor(out, cv2.COLOR_LAB2BGR)


def enhance_hsv_saturation(crop):
    hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
    h, s, v = cv2.split(hsv)

    clahe = cv2.createCLAHE(clipLimit=2.5, tileGridSize=(8, 8))
    v2 = clahe.apply(v)

    s2 = np.clip(s.astype(np.float32) * 1.55 + 15, 0, 255).astype(np.uint8)

    out = cv2.merge([h, s2, v2])
    return cv2.cvtColor(out, cv2.COLOR_HSV2BGR)


def enhance_lab_plus_sat(crop):
    lab = enhance_lab_clahe(crop)
    return enhance_hsv_saturation(lab)


def hue_mask(h, ranges):
    out = np.zeros_like(h, dtype=bool)
    for lo, hi in ranges:
        out |= (h >= lo) & (h <= hi)
    return out


def find_colour_candidates(variant_name, img_bgr, pig_mask_crop, full_x1, full_y1, scan_frame_id, final_box_id):
    hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
    H = hsv[:, :, 0]
    S = hsv[:, :, 1]
    V = hsv[:, :, 2]

    pig = pig_mask_crop.astype(np.uint8)

    if pig.sum() == 0:
        return []

    # Exclude uncertain borders of the mask so pen/floor edges are less likely to be picked.
    kernel = np.ones((3, 3), np.uint8)
    inner = cv2.erode(pig, kernel, iterations=1).astype(bool)

    if inner.sum() < max(30, 0.25 * pig.sum()):
        inner = pig.astype(bool)

    pig_area = max(1, int(inner.sum()))
    rows = []

    for colour, ranges in HSV_RANGES.items():
        cmask = hue_mask(H, ranges) & (S > 45) & (V > 35) & inner

        cmask_u8 = (cmask.astype(np.uint8) * 255)
        cmask_u8 = cv2.morphologyEx(cmask_u8, cv2.MORPH_OPEN, kernel)
        cmask_u8 = cv2.morphologyEx(cmask_u8, cv2.MORPH_CLOSE, kernel)

        contours, _ = cv2.findContours(cmask_u8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        for ci, cnt in enumerate(contours):
            area = float(cv2.contourArea(cnt))

            if area < 8:
                continue

            if area > pig_area * 0.18:
                continue

            x, y, w, h = cv2.boundingRect(cnt)

            if w < 3 or h < 3:
                continue

            comp = np.zeros_like(cmask_u8)
            cv2.drawContours(comp, [cnt], -1, 255, -1)
            pix = comp > 0

            if pix.sum() == 0:
                continue

            mean_s = float(np.mean(S[pix]))
            mean_v = float(np.mean(V[pix]))
            mean_h = float(np.mean(H[pix]))
            area_fraction = float(pix.sum() / pig_area)

            # Score rewards compact saturated patches inside the pig mask.
            score = float(area_fraction * 1000.0 * (mean_s / 255.0) * (mean_v / 255.0))

            rows.append({
                "scan_frame_id": scan_frame_id,
                "final_box_id": final_box_id,
                "variant": variant_name,
                "candidate_colour": colour,
                "candidate_score": score,
                "candidate_area_pixels": int(pix.sum()),
                "candidate_area_fraction_of_pig_mask": area_fraction,
                "mean_h": mean_h,
                "mean_s": mean_s,
                "mean_v": mean_v,
                "crop_x1": int(x),
                "crop_y1": int(y),
                "crop_x2": int(x + w),
                "crop_y2": int(y + h),
                "full_x1": int(full_x1 + x),
                "full_y1": int(full_y1 + y),
                "full_x2": int(full_x1 + x + w),
                "full_y2": int(full_y1 + y + h),
            })

    return rows


def confidence_from_scores(scores):
    if not scores:
        return "low_no_marker", "", 0.0, "", 0.0, 0.0

    sorted_scores = sorted(scores.items(), key=lambda kv: kv[1], reverse=True)
    top_colour, top_score = sorted_scores[0]
    second_colour, second_score = sorted_scores[1] if len(sorted_scores) > 1 else ("", 0.0)

    margin = top_score / max(second_score, 1e-9)

    if top_score < 4.0:
        conf = "low_no_marker"
    elif margin < 1.20:
        conf = "low_ambiguous"
    elif top_score < 10.0:
        conf = "medium"
    else:
        conf = "high"

    return conf, top_colour, top_score, second_colour, second_score, margin


def make_contact_sheet(paths, out_path):
    if not paths:
        return False

    sample = paths
    if len(sample) > 16:
        idx = np.linspace(0, len(sample) - 1, 16).round().astype(int)
        sample = [sample[i] for i in idx]

    thumbs = []
    tw, th = 380, 250

    for p in sample:
        img = cv2.imread(str(p))
        if img is None:
            continue
        thumbs.append(cv2.resize(img, (tw, th), interpolation=cv2.INTER_AREA))

    if not thumbs:
        return False

    cols = 4
    rows = math.ceil(len(thumbs) / cols)
    sheet = np.full((rows * th, cols * tw, 3), 255, dtype=np.uint8)

    for i, im in enumerate(thumbs):
        r = i // cols
        c = i % cols
        sheet[r * th:(r + 1) * th, c * tw:(c + 1) * tw] = im

    cv2.imwrite(str(out_path), sheet)
    return True


# ---------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------
boxes = pd.read_csv(FINAL_BOXES_PATH)

for c in ["x1", "y1", "x2", "y2"]:
    boxes[c] = pd.to_numeric(boxes[c], errors="coerce")

boxes = boxes.dropna(subset=["x1", "y1", "x2", "y2"]).copy()
boxes = boxes.sort_values(["scan_frame_id", "final_box_id"]).copy()

predictor, segmentation_method = load_sam_predictor()

box_qa_rows = []
candidate_rows = []
score_rows = []
frame_summary_rows = []
overlay_paths = []

for scan_frame_id, g in boxes.groupby("scan_frame_id", sort=True):
    scan_frame_id = str(scan_frame_id)
    img_path = IMG_DIR / f"{scan_frame_id}.jpg"

    if not img_path.exists():
        continue

    img = cv2.imread(str(img_path))
    if img is None:
        continue

    h_img, w_img = img.shape[:2]

    if predictor is not None:
        predictor.set_image(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))

    overlay = img.copy()
    frame_review_count = 0
    frame_high = 0
    frame_medium = 0
    frame_low = 0

    for _, b in g.iterrows():
        final_box_id = str(b["final_box_id"])
        x1, y1, x2, y2 = clamp_box(b["x1"], b["y1"], b["x2"], b["y2"], w_img, h_img)

        if predictor is not None:
            try:
                masks, scores, _ = predictor.predict(
                    box=np.array([x1, y1, x2, y2]),
                    multimask_output=True,
                )
                best_idx = int(np.argmax(scores))
                full_mask = masks[best_idx].astype(bool)
                mask_score = float(scores[best_idx])
                used_method = segmentation_method
            except Exception:
                full_mask, used_method = fallback_grabcut_mask(img, (x1, y1, x2, y2))
                mask_score = np.nan
        else:
            full_mask, used_method = fallback_grabcut_mask(img, (x1, y1, x2, y2))
            mask_score = np.nan

        crop = img[y1:y2, x1:x2].copy()
        crop_mask = full_mask[y1:y2, x1:x2].astype(bool)

        if crop_mask.sum() < 20:
            crop_mask[:, :] = True
            used_method = used_method + "_crop_mask_too_small_box_fill"

        masked_crop = crop.copy()
        masked_crop[~crop_mask] = (128, 128, 128)

        enhanced_lab = enhance_lab_clahe(masked_crop)
        enhanced_hsv = enhance_hsv_saturation(masked_crop)
        enhanced_combo = enhance_lab_plus_sat(masked_crop)

        crop_name = f"{scan_frame_id}_{final_box_id}.jpg"

        mask_path = OUT_MASKS / crop_name.replace(".jpg", "_mask.png")
        crop_path = OUT_CROPS / crop_name
        lab_path = OUT_ENHANCED / crop_name.replace(".jpg", "_lab_clahe.jpg")
        hsv_path = OUT_ENHANCED / crop_name.replace(".jpg", "_hsv_sat.jpg")
        combo_path = OUT_ENHANCED / crop_name.replace(".jpg", "_combo.jpg")

        cv2.imwrite(str(mask_path), (crop_mask.astype(np.uint8) * 255))
        cv2.imwrite(str(crop_path), masked_crop)
        cv2.imwrite(str(lab_path), enhanced_lab)
        cv2.imwrite(str(hsv_path), enhanced_hsv)
        cv2.imwrite(str(combo_path), enhanced_combo)

        variants = {
            "masked_original": masked_crop,
            "lab_clahe": enhanced_lab,
            "hsv_saturation": enhanced_hsv,
            "lab_clahe_plus_hsv_saturation": enhanced_combo,
        }

        local_candidates = []

        for vname, vim in variants.items():
            local_candidates.extend(
                find_colour_candidates(
                    vname,
                    vim,
                    crop_mask,
                    x1,
                    y1,
                    scan_frame_id,
                    final_box_id,
                )
            )

        candidate_rows.extend(local_candidates)

        # Per-colour max score over all variants/components.
        colour_scores = {}

        for colour in VALID_COLOURS:
            vals = [r["candidate_score"] for r in local_candidates if r["candidate_colour"] == colour]
            colour_scores[colour] = max(vals) if vals else 0.0

        conf, top_colour, top_score, second_colour, second_score, margin = confidence_from_scores(colour_scores)

        if conf == "high":
            frame_high += 1
        elif conf == "medium":
            frame_medium += 1
        else:
            frame_low += 1
            frame_review_count += 1

        for colour in VALID_COLOURS:
            score_rows.append({
                "scan_frame_id": scan_frame_id,
                "final_box_id": final_box_id,
                "candidate_colour": colour,
                "segmented_enhanced_score_v15": colour_scores[colour],
                "top_colour_v15": top_colour,
                "top_score_v15": top_score,
                "second_colour_v15": second_colour,
                "second_score_v15": second_score,
                "top_second_margin_v15": margin,
                "evidence_confidence_v15": conf,
                "manual_review_recommended_v15": conf not in ["high", "medium"],
            })

        mask_area = int(crop_mask.sum())
        bbox_area = int(max(1, (x2 - x1) * (y2 - y1)))

        box_qa_rows.append({
            "scan_frame_id": scan_frame_id,
            "final_box_id": final_box_id,
            "x1": x1,
            "y1": y1,
            "x2": x2,
            "y2": y2,
            "segmentation_method": used_method,
            "sam_score": mask_score,
            "mask_area_pixels": mask_area,
            "bbox_area_pixels": bbox_area,
            "mask_area_fraction_of_bbox": float(mask_area / bbox_area),
            "masked_crop_path": str(crop_path),
            "enhanced_lab_clahe_path": str(lab_path),
            "enhanced_hsv_saturation_path": str(hsv_path),
            "enhanced_combo_path": str(combo_path),
            "top_colour_v15": top_colour,
            "top_score_v15": top_score,
            "second_colour_v15": second_colour,
            "second_score_v15": second_score,
            "top_second_margin_v15": margin,
            "evidence_confidence_v15": conf,
        })

        bgr = COLOUR_BGR.get(top_colour, COLOUR_BGR["unknown"])
        cv2.rectangle(overlay, (x1, y1), (x2, y2), bgr, 3)

        label = f"{top_colour if top_colour else 'none'} {top_score:.1f}"
        if conf.startswith("low"):
            label += " ?"

        cv2.putText(
            overlay,
            label,
            (x1, max(18, y1 - 5)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.48,
            bgr,
            2,
            cv2.LINE_AA,
        )

        # Draw top few marker components for this box.
        best_components = sorted(local_candidates, key=lambda r: r["candidate_score"], reverse=True)[:3]

        for cand in best_components:
            cbgr = COLOUR_BGR.get(cand["candidate_colour"], COLOUR_BGR["unknown"])
            cx1 = int(cand["full_x1"])
            cy1 = int(cand["full_y1"])
            cx2 = int(cand["full_x2"])
            cy2 = int(cand["full_y2"])

            cv2.rectangle(overlay, (cx1, cy1), (cx2, cy2), cbgr, 2)
            cv2.putText(
                overlay,
                cand["candidate_colour"],
                (cx1, max(18, cy1 - 4)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.36,
                cbgr,
                1,
                cv2.LINE_AA,
            )

    title = f"{scan_frame_id} | high={frame_high} medium={frame_medium} low={frame_low} review={frame_review_count}"
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

    out_overlay = OUT_OVERLAYS / f"{scan_frame_id}_segmented_enhanced_colour_v15.jpg"
    cv2.imwrite(str(out_overlay), overlay)
    overlay_paths.append(out_overlay)

    frame_summary_rows.append({
        "scan_frame_id": scan_frame_id,
        "final_box_count": int(len(g)),
        "high_confidence_boxes_v15": frame_high,
        "medium_confidence_boxes_v15": frame_medium,
        "low_confidence_boxes_v15": frame_low,
        "manual_review_recommended_boxes_v15": frame_review_count,
        "frame_review_recommended_v15": frame_review_count > 0,
        "overlay_path": str(out_overlay),
    })

box_qa = pd.DataFrame(box_qa_rows)
scores = pd.DataFrame(score_rows)
candidates = pd.DataFrame(candidate_rows)
frame_summary = pd.DataFrame(frame_summary_rows)

safe_to_csv(box_qa, OUT_BOX_MASK_QA)
safe_to_csv(scores, OUT_COLOUR_SCORES)
safe_to_csv(candidates, OUT_MARKER_CANDIDATES)
safe_to_csv(frame_summary, OUT_FRAME_SUMMARY)

review = box_qa[box_qa["evidence_confidence_v15"].astype(str).str.startswith("low")].copy()
safe_to_csv(review, OUT_REVIEW)

contact_ok = make_contact_sheet(overlay_paths, OUT_CONTACT)

cards = []
for _, r in frame_summary.sort_values("scan_frame_id").iterrows():
    sid = str(r["scan_frame_id"])
    src = f"frame_overlays/{sid}_segmented_enhanced_colour_v15.jpg"
    cls = "review" if bool(r["frame_review_recommended_v15"]) else "ok"

    cards.append(
        f"""
        <div class="frame-card {cls}">
          <h3>{sid}</h3>
          <p>
            boxes={r['final_box_count']} |
            high={r['high_confidence_boxes_v15']} |
            medium={r['medium_confidence_boxes_v15']} |
            low={r['low_confidence_boxes_v15']} |
            review={r['manual_review_recommended_boxes_v15']}
          </p>
          <img src="{src}">
        </div>
        """
    )

html = f"""
<!DOCTYPE html>
<html>
<head>
  <meta charset="utf-8">
  <title>Week 7 Segmented Enhanced Colour Evidence v15</title>
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
      margin: 18px 0;
      padding: 12px;
      border-radius: 8px;
      box-shadow: 0 1px 4px rgba(0,0,0,0.15);
    }}
    .ok {{ border-left: 8px solid green; }}
    .review {{ border-left: 8px solid darkorange; }}
    img {{
      max-width: 100%;
      border: 1px solid #ddd;
      display: block;
    }}
  </style>
</head>
<body>
  <h1>Week 7 Segmented Enhanced Colour Evidence v15</h1>
  <div class="legend">
    <p>Large boxes: final corrected GT-pen pig boxes coloured by top v15 evidence.</p>
    <p>Small boxes: marker candidates found inside segmentation-masked, contrast-enhanced pig crops.</p>
    <p>This is improved evidence, not final manually verified identity yet.</p>
  </div>
  {''.join(cards)}
</body>
</html>
"""
OUT_HTML.write_text(html)

total_boxes = len(box_qa)
high = int((box_qa["evidence_confidence_v15"] == "high").sum())
medium = int((box_qa["evidence_confidence_v15"] == "medium").sum())
low = int(box_qa["evidence_confidence_v15"].astype(str).str.startswith("low").sum())
review_count = int(len(review))
frames_review = int(frame_summary["frame_review_recommended_v15"].sum())

OUT_NOTE.write_text(
    "# Week 7 Segmented Enhanced Colour Evidence v15\n\n"
    "## Purpose\n\n"
    "This step improves colour marker evidence by using segmentation masks and contrast/saturation enhancement. "
    "It is designed to reduce background false positives and reveal weak pig marker colours.\n\n"
    "## Method\n\n"
    "- Use SAM box-prompt masks when available; otherwise fallback to GrabCut/box-fill.\n"
    "- Keep only pig-mask pixels inside each final corrected GT-pen box.\n"
    "- Generate masked crop, LAB CLAHE crop, HSV saturation crop, and combined enhancement crop.\n"
    "- Search only valid marker colours: blue, green, cyan, red, pink, purple.\n"
    "- Ignore orange/yellow as invalid marker colours.\n\n"
    "## Summary\n\n"
    f"- Total boxes processed: `{total_boxes}`\n"
    f"- High confidence evidence: `{high}`\n"
    f"- Medium confidence evidence: `{medium}`\n"
    f"- Low confidence evidence: `{low}`\n"
    f"- Boxes recommended for review: `{review_count}`\n"
    f"- Frames recommended for review: `{frames_review}`\n"
    f"- Segmentation method: `{segmentation_method}`\n\n"
    "## Outputs\n\n"
    f"- Box/mask QA: `{OUT_BOX_MASK_QA}`\n"
    f"- Colour scores: `{OUT_COLOUR_SCORES}`\n"
    f"- Marker candidates: `{OUT_MARKER_CANDIDATES}`\n"
    f"- Frame summary: `{OUT_FRAME_SUMMARY}`\n"
    f"- Low-confidence review table: `{OUT_REVIEW}`\n"
    f"- Static HTML: `{OUT_HTML}`\n"
)

print("Saved:")
print(OUT_BOX_MASK_QA)
print(OUT_COLOUR_SCORES)
print(OUT_MARKER_CANDIDATES)
print(OUT_FRAME_SUMMARY)
print(OUT_REVIEW)
print(OUT_CONTACT)
print(OUT_HTML)
print(OUT_NOTE)

print()
print("=== segmented enhanced colour evidence v15 summary ===")
print({
    "total_boxes_processed": total_boxes,
    "high_confidence": high,
    "medium_confidence": medium,
    "low_confidence": low,
    "boxes_recommended_for_review": review_count,
    "frames_recommended_for_review": frames_review,
    "segmentation_method": segmentation_method,
    "contact_sheet_generated": contact_ok,
})
