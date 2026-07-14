from pathlib import Path
from datetime import datetime
import csv
import math
import re

import cv2
import numpy as np
import pandas as pd
from PIL import Image, ImageDraw, ImageFont


ROOT = Path.home() / "PigBench"
W7 = ROOT / "Week7_Dataset_Validation_Identity_Tracking_Behaviour_Preparation"

V28D_TRACKS = W7 / "outputs" / "dense_polygon_filtered_tracking_v28d" / "week7_dense_polygon_filtered_tracking_v28d_tracks.csv"
V28D_CLIP_SUMMARY = W7 / "outputs" / "dense_polygon_filtered_tracking_v28d" / "week7_dense_polygon_filtered_tracking_v28d_clip_summary.csv"

V29B_TRACKLETS = W7 / "outputs" / "colour_constrained_tracklet_linking_v29b" / "week7_colour_constrained_tracklet_linking_v29b_tracklet_identity_candidates.csv"
V29B_CLIP_SUMMARY = W7 / "outputs" / "colour_constrained_tracklet_linking_v29b" / "week7_colour_constrained_tracklet_linking_v29b_clip_summary.csv"

OUT_ROOT = W7 / "outputs" / "full_tracklet_colour_evidence_v29c"
CROP_ROOT = OUT_ROOT / "sampled_tracklet_crops"
CONTACT_ROOT = OUT_ROOT / "contact_sheets"

for p in [OUT_ROOT, CROP_ROOT, CONTACT_ROOT]:
    p.mkdir(parents=True, exist_ok=True)

OUT_DET_EVIDENCE = OUT_ROOT / "week7_full_tracklet_colour_evidence_v29c_detection_colour_evidence.csv"
OUT_TRACKLET_EVIDENCE = OUT_ROOT / "week7_full_tracklet_colour_evidence_v29c_tracklet_colour_evidence.csv"
OUT_COMPARE_V29B = OUT_ROOT / "week7_full_tracklet_colour_evidence_v29c_compare_to_v29b.csv"
OUT_CLIP_SUMMARY = OUT_ROOT / "week7_full_tracklet_colour_evidence_v29c_clip_summary.csv"
OUT_MERGE_CANDIDATES = OUT_ROOT / "week7_full_tracklet_colour_evidence_v29c_same_colour_merge_candidates.csv"
OUT_DECISION = OUT_ROOT / "week7_full_tracklet_colour_evidence_v29c_decision_summary.csv"
OUT_CONTACT_INDEX = OUT_ROOT / "week7_full_tracklet_colour_evidence_v29c_contact_sheet_index.csv"
OUT_LIMITATIONS = OUT_ROOT / "week7_full_tracklet_colour_evidence_v29c_limitations.md"
OUT_ISSUES = OUT_ROOT / "week7_full_tracklet_colour_evidence_v29c_issues.csv"
OUT_README = OUT_ROOT / "README_full_tracklet_colour_evidence_v29c.md"
OUT_NOTE = W7 / "notes" / "week7_full_tracklet_colour_evidence_v29c_notes.md"

VALID_COLOURS = ["blue", "green", "cyan", "red", "pink", "purple"]
COLOUR_TO_BEHAVIOUR_ID = {
    "blue": "blue",
    "green": "green",
    "cyan": "no_color",
    "red": "red_neck",
    "pink": "red_tail",
    "purple": "purple",
}

# We do not force a colour when evidence is weak.
MIN_EVIDENCE_FRAMES_HIGH = 5
MIN_EVIDENCE_FRAMES_MEDIUM = 3
MIN_DET_SCORE_FOR_COLOUR_FRAME = 0.0035
MIN_AGG_SCORE_MEDIUM = 0.015
MIN_AGG_SCORE_HIGH = 0.035
MIN_MARGIN_MEDIUM = 1.25
MIN_MARGIN_HIGH = 1.60
MAX_CROPS_PER_TRACKLET_FOR_CONTACT = 8


def safe_to_csv(df, path):
    df.to_csv(
        path,
        index=False,
        quoting=csv.QUOTE_ALL,
        escapechar="\\",
        lineterminator="\n",
    )


def clean(v):
    if pd.isna(v):
        return ""
    return str(v).strip()


def to_num(v, default=np.nan):
    try:
        x = pd.to_numeric(pd.Series([v]), errors="coerce").iloc[0]
        if pd.isna(x):
            return default
        return float(x)
    except Exception:
        return default


def slug(s):
    s = clean(s)
    s = re.sub(r"[^A-Za-z0-9_.-]+", "_", s)
    s = re.sub(r"_+", "_", s).strip("_")
    return s or "unknown"


def load_font(size=13):
    try:
        return ImageFont.truetype("DejaVuSans.ttf", size)
    except Exception:
        return ImageFont.load_default()


def clamp_box(x1, y1, x2, y2, w, h):
    x1 = int(max(0, min(w - 1, round(float(x1)))))
    y1 = int(max(0, min(h - 1, round(float(y1)))))
    x2 = int(max(0, min(w, round(float(x2)))))
    y2 = int(max(0, min(h, round(float(y2)))))

    if x2 <= x1:
        x2 = min(w, x1 + 1)
    if y2 <= y1:
        y2 = min(h, y1 + 1)

    return x1, y1, x2, y2


def marker_roi_from_crop(crop_bgr):
    """
    Heuristic: marker paint tends to be on upper/back body area.
    We keep the full crop but down-weight lower background by selecting the upper 70%.
    """
    h, w = crop_bgr.shape[:2]
    if h < 8 or w < 8:
        return crop_bgr

    y1 = 0
    y2 = int(round(h * 0.72))
    x1 = int(round(w * 0.05))
    x2 = int(round(w * 0.95))

    return crop_bgr[y1:y2, x1:x2]


def hsv_colour_masks(hsv):
    h = hsv[:, :, 0]
    s = hsv[:, :, 1]
    v = hsv[:, :, 2]

    sat = s > 45
    val = v > 45

    masks = {}

    # OpenCV hue: 0..179
    masks["red"] = (((h <= 10) | (h >= 170)) & (s > 55) & (v > 45))
    masks["pink"] = ((h >= 145) & (h <= 169) & (s > 45) & (v > 55))
    masks["purple"] = ((h >= 125) & (h <= 149) & (s > 45) & (v > 45))
    masks["blue"] = ((h >= 100) & (h <= 124) & (s > 45) & (v > 45))
    masks["cyan"] = ((h >= 82) & (h <= 99) & (s > 35) & (v > 50))
    masks["green"] = ((h >= 35) & (h <= 81) & (s > 40) & (v > 45))

    # Require not too dark / not grey globally.
    for k in masks:
        masks[k] = masks[k] & sat & val

    return masks


def compute_colour_evidence(crop_bgr):
    if crop_bgr is None or crop_bgr.size == 0:
        return {c: 0.0 for c in VALID_COLOURS}, "unknown", 0.0, 0.0, 0

    roi = marker_roi_from_crop(crop_bgr)
    hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)

    masks = hsv_colour_masks(hsv)
    total_pixels = max(1, hsv.shape[0] * hsv.shape[1])

    scores = {}

    for colour, mask in masks.items():
        if mask.sum() == 0:
            scores[colour] = 0.0
            continue

        # Score combines coloured-pixel ratio and saturation strength.
        s_vals = hsv[:, :, 1][mask]
        v_vals = hsv[:, :, 2][mask]
        ratio = float(mask.sum()) / float(total_pixels)
        sat_strength = float(np.mean(s_vals)) / 255.0
        val_strength = float(np.mean(v_vals)) / 255.0

        scores[colour] = ratio * (0.70 * sat_strength + 0.30 * val_strength)

    ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)
    best_colour, best_score = ranked[0]
    second_score = ranked[1][1] if len(ranked) > 1 else 0.0
    margin = best_score / (second_score + 1e-9)

    if best_score < MIN_DET_SCORE_FOR_COLOUR_FRAME:
        best_colour = "unknown"

    return scores, best_colour, float(best_score), float(margin), int(total_pixels)


def classify_tracklet(best_colour, best_score_sum, evidence_frames, margin):
    if not best_colour or best_colour == "unknown":
        return "unknown_no_strong_colour_evidence"

    if (
        evidence_frames >= MIN_EVIDENCE_FRAMES_HIGH
        and best_score_sum >= MIN_AGG_SCORE_HIGH
        and margin >= MIN_MARGIN_HIGH
    ):
        return "assigned_high_full_tracklet_colour"

    if (
        evidence_frames >= MIN_EVIDENCE_FRAMES_MEDIUM
        and best_score_sum >= MIN_AGG_SCORE_MEDIUM
        and margin >= MIN_MARGIN_MEDIUM
    ):
        return "assigned_medium_full_tracklet_colour"

    if evidence_frames >= 1:
        return "assigned_low_full_tracklet_colour_needs_review"

    return "unknown_no_evidence_frames"


def make_contact_sheet(crop_rows, out_path, title, cols=4, thumb_w=180, thumb_h=140):
    if not crop_rows:
        return False

    font = load_font(11)
    title_font = load_font(16)

    pad = 8
    title_h = 42
    cell_w = thumb_w + 2 * pad
    cell_h = thumb_h + 72 + 2 * pad
    rows = math.ceil(len(crop_rows) / cols)

    sheet = Image.new("RGB", (cols * cell_w, title_h + rows * cell_h), "white")
    draw = ImageDraw.Draw(sheet)
    draw.text((pad, pad), title, fill=(0, 0, 0), font=title_font)

    for i, row in enumerate(crop_rows):
        r = i // cols
        c = i % cols
        x0 = c * cell_w + pad
        y0 = title_h + r * cell_h + pad

        p = Path(row["crop_path"])

        try:
            im = Image.open(p).convert("RGB")
            im.thumbnail((thumb_w, thumb_h))
            bg = Image.new("RGB", (thumb_w, thumb_h), "white")
            bg.paste(im, ((thumb_w - im.width) // 2, (thumb_h - im.height) // 2))
            sheet.paste(bg, (x0, y0))
        except Exception:
            bg = Image.new("RGB", (thumb_w, thumb_h), "lightgray")
            dd = ImageDraw.Draw(bg)
            dd.text((8, 8), "LOAD ERROR", fill=(0, 0, 0), font=font)
            sheet.paste(bg, (x0, y0))

        label = (
            f"T{row.get('dense_track_id')} f={row.get('frame_index')}\n"
            f"det={row.get('frame_best_colour')} {float(row.get('frame_best_score', 0)):.4f}\n"
            f"v29b={row.get('v29b_colour', '')}"
        )
        draw.text((x0, y0 + thumb_h + 4), label, fill=(0, 0, 0), font=font)

    sheet.save(out_path, quality=95)
    return True


issues = []

required = [V28D_TRACKS, V28D_CLIP_SUMMARY, V29B_TRACKLETS, V29B_CLIP_SUMMARY]

for p in required:
    if not p.exists():
        issues.append({
            "issue_type": "missing_required_file",
            "issue_detail": str(p),
        })

if issues:
    issues_df = pd.DataFrame(issues)
    safe_to_csv(issues_df, OUT_ISSUES)
    print(issues_df.to_string(index=False))
    raise SystemExit(1)

tracks = pd.read_csv(V28D_TRACKS)
clip_summary = pd.read_csv(V28D_CLIP_SUMMARY)
v29b_tracklets = pd.read_csv(V29B_TRACKLETS)
v29b_clip = pd.read_csv(V29B_CLIP_SUMMARY)

for df in [tracks, clip_summary, v29b_tracklets, v29b_clip]:
    if "scan_frame_id" in df.columns:
        df["scan_frame_id"] = df["scan_frame_id"].fillna("").astype(str).str.strip()
    if "clip_id" in df.columns:
        df["clip_id"] = df["clip_id"].fillna("").astype(str).str.strip()

for c in ["frame_index", "dense_track_id", "x1", "y1", "x2", "y2", "score"]:
    if c in tracks.columns:
        tracks[c] = pd.to_numeric(tracks[c], errors="coerce")

v29b_key = v29b_tracklets[
    [
        "clip_id",
        "dense_track_id",
        "assigned_visual_colour",
        "assigned_behaviour_pig_id",
        "assignment_status",
        "center_match_iou_max",
    ]
].copy()

v29b_key["dense_track_id"] = pd.to_numeric(v29b_key["dense_track_id"], errors="coerce")

tracks = tracks.merge(
    v29b_key,
    on=["clip_id", "dense_track_id"],
    how="left",
    suffixes=("", "_v29b"),
)

det_rows = []
tracklet_rows = []
contact_rows = []

# Cache opened video objects by clip path would be risky; open per clip.
for clip_id, cg in tracks.groupby("clip_id", sort=True):
    cg = cg.sort_values(["frame_index", "dense_track_id"]).copy()
    sid = clean(cg["scan_frame_id"].iloc[0])
    clip_path = Path(clean(cg["clip_path"].iloc[0]))

    if not clip_path.exists():
        issues.append({
            "issue_type": "clip_missing",
            "issue_detail": f"{clip_id}: {clip_path}",
        })
        continue

    cap = cv2.VideoCapture(str(clip_path))

    if not cap.isOpened():
        issues.append({
            "issue_type": "clip_open_failed",
            "issue_detail": f"{clip_id}: {clip_path}",
        })
        continue

    frame_cache = {}

    for (tid), tg in cg.groupby("dense_track_id", sort=True):
        tid_int = int(tid)
        crop_dir = CROP_ROOT / slug(clip_id) / f"track_{tid_int:03d}"
        crop_dir.mkdir(parents=True, exist_ok=True)

        sample_for_contact = []

        for _, tr in tg.iterrows():
            frame_idx = int(tr["frame_index"])

            if frame_idx not in frame_cache:
                cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
                ok, frame = cap.read()
                if not ok or frame is None:
                    issues.append({
                        "issue_type": "frame_read_failed",
                        "issue_detail": f"{clip_id}, frame={frame_idx}",
                    })
                    continue
                frame_cache[frame_idx] = frame
            else:
                frame = frame_cache[frame_idx]

            h, w = frame.shape[:2]
            x1, y1, x2, y2 = clamp_box(tr["x1"], tr["y1"], tr["x2"], tr["y2"], w, h)
            crop = frame[y1:y2, x1:x2].copy()

            scores, best_colour, best_score, margin, total_pixels = compute_colour_evidence(crop)

            crop_path = crop_dir / f"{slug(clip_id)}__T{tid_int:03d}__frame_{frame_idx:06d}__{best_colour}.jpg"

            # Save only selected crops to keep package reasonable:
            # high evidence crops or every 10th frame for long tracklets.
            save_crop = best_colour != "unknown" or (len(sample_for_contact) < MAX_CROPS_PER_TRACKLET_FOR_CONTACT and frame_idx % 25 == 0)
            if save_crop:
                cv2.imwrite(str(crop_path), crop)
            else:
                crop_path = ""

            row = {
                "clip_id": clip_id,
                "scan_frame_id": sid,
                "dense_track_id": tid_int,
                "frame_index": frame_idx,
                "track_score": round(float(tr["score"]), 6),
                "bbox_x1": x1,
                "bbox_y1": y1,
                "bbox_x2": x2,
                "bbox_y2": y2,
                "frame_best_colour": best_colour,
                "frame_best_score": round(best_score, 8),
                "frame_colour_margin": round(margin, 4),
                "crop_pixels": total_pixels,
                "score_blue": round(scores["blue"], 8),
                "score_green": round(scores["green"], 8),
                "score_cyan": round(scores["cyan"], 8),
                "score_red": round(scores["red"], 8),
                "score_pink": round(scores["pink"], 8),
                "score_purple": round(scores["purple"], 8),
                "v29b_colour": clean(tr.get("assigned_visual_colour", "")),
                "v29b_behaviour_pig_id": clean(tr.get("assigned_behaviour_pig_id", "")),
                "v29b_assignment_status": clean(tr.get("assignment_status", "")),
                "crop_path": str(crop_path),
            }

            det_rows.append(row)

            if crop_path and len(sample_for_contact) < MAX_CROPS_PER_TRACKLET_FOR_CONTACT:
                sample_for_contact.append(row)

        if sample_for_contact:
            sheet_path = CONTACT_ROOT / f"{slug(clip_id)}__T{tid_int:03d}_colour_evidence_sheet.jpg"
            made = make_contact_sheet(
                sample_for_contact,
                sheet_path,
                f"v29c full-tracklet colour evidence | {sid} | T{tid_int}",
            )
            if made:
                contact_rows.append({
                    "clip_id": clip_id,
                    "scan_frame_id": sid,
                    "dense_track_id": tid_int,
                    "contact_sheet_path": str(sheet_path),
                    "sampled_crops": len(sample_for_contact),
                })

    cap.release()

det_evidence = pd.DataFrame(det_rows)
safe_to_csv(det_evidence, OUT_DET_EVIDENCE)

# Aggregate per tracklet.
for (clip_id, tid), tg in det_evidence.groupby(["clip_id", "dense_track_id"], sort=True):
    sid = clean(tg["scan_frame_id"].iloc[0])

    colour_sums = {
        colour: float(tg[f"score_{colour}"].sum())
        for colour in VALID_COLOURS
    }

    ranked = sorted(colour_sums.items(), key=lambda x: x[1], reverse=True)
    best_colour, best_score_sum = ranked[0]
    second_colour, second_score_sum = ranked[1] if len(ranked) > 1 else ("", 0.0)

    evidence_frames = int((tg["frame_best_colour"] == best_colour).sum())
    non_unknown_frames = int((tg["frame_best_colour"] != "unknown").sum())
    total_frames = int(tg["frame_index"].nunique())

    margin = best_score_sum / (second_score_sum + 1e-9)

    if best_score_sum <= 0 or evidence_frames == 0:
        best_colour = "unknown"

    status = classify_tracklet(best_colour, best_score_sum, evidence_frames, margin)
    behaviour_id = COLOUR_TO_BEHAVIOUR_ID.get(best_colour, "")

    v29b_colour = clean(tg["v29b_colour"].dropna().astype(str).iloc[0]) if "v29b_colour" in tg.columns and len(tg) else ""
    v29b_behaviour = clean(tg["v29b_behaviour_pig_id"].dropna().astype(str).iloc[0]) if "v29b_behaviour_pig_id" in tg.columns and len(tg) else ""
    v29b_status = clean(tg["v29b_assignment_status"].dropna().astype(str).iloc[0]) if "v29b_assignment_status" in tg.columns and len(tg) else ""

    if v29b_colour and best_colour != "unknown" and v29b_colour == best_colour:
        comparison_status = "agreement_with_v29b"
    elif not v29b_colour and best_colour != "unknown" and "high" in status or ("medium" in status and not v29b_colour):
        comparison_status = "recovered_from_v29b_unmatched"
    elif v29b_colour and best_colour == "unknown":
        comparison_status = "v29b_assigned_but_v29c_unknown"
    elif v29b_colour and best_colour != "unknown" and v29b_colour != best_colour:
        comparison_status = "conflict_with_v29b_needs_review"
    elif not v29b_colour and best_colour == "unknown":
        comparison_status = "still_unmatched"
    else:
        comparison_status = "needs_review"

    tracklet_rows.append({
        "clip_id": clip_id,
        "scan_frame_id": sid,
        "dense_track_id": int(tid),
        "tracklet_frames_seen": total_frames,
        "non_unknown_colour_frames": non_unknown_frames,
        "best_colour_evidence_frames": evidence_frames,
        "v29c_assigned_visual_colour": best_colour,
        "v29c_assigned_behaviour_pig_id": behaviour_id,
        "v29c_assignment_status": status,
        "v29c_best_score_sum": round(best_score_sum, 8),
        "v29c_second_colour": second_colour,
        "v29c_second_score_sum": round(second_score_sum, 8),
        "v29c_margin": round(margin, 4),
        "sum_blue": round(colour_sums["blue"], 8),
        "sum_green": round(colour_sums["green"], 8),
        "sum_cyan": round(colour_sums["cyan"], 8),
        "sum_red": round(colour_sums["red"], 8),
        "sum_pink": round(colour_sums["pink"], 8),
        "sum_purple": round(colour_sums["purple"], 8),
        "v29b_assigned_visual_colour": v29b_colour,
        "v29b_assigned_behaviour_pig_id": v29b_behaviour,
        "v29b_assignment_status": v29b_status,
        "v29b_v29c_comparison_status": comparison_status,
    })

tracklet_evidence = pd.DataFrame(tracklet_rows)
safe_to_csv(tracklet_evidence, OUT_TRACKLET_EVIDENCE)

compare = tracklet_evidence[
    [
        "clip_id",
        "scan_frame_id",
        "dense_track_id",
        "v29b_assigned_visual_colour",
        "v29c_assigned_visual_colour",
        "v29b_assigned_behaviour_pig_id",
        "v29c_assigned_behaviour_pig_id",
        "v29b_assignment_status",
        "v29c_assignment_status",
        "v29b_v29c_comparison_status",
        "v29c_margin",
        "best_colour_evidence_frames",
        "tracklet_frames_seen",
    ]
].copy()

safe_to_csv(compare, OUT_COMPARE_V29B)

# Merge candidates: same clip + same v29c colour with multiple tracklets.
merge_rows = []

assigned_good = tracklet_evidence[
    tracklet_evidence["v29c_assignment_status"].astype(str).str.contains("high|medium", regex=True)
    & tracklet_evidence["v29c_assigned_visual_colour"].isin(VALID_COLOURS)
].copy()

for (clip_id, colour), cg in assigned_good.groupby(["clip_id", "v29c_assigned_visual_colour"], sort=True):
    tids = sorted(cg["dense_track_id"].astype(int).tolist())
    if len(tids) <= 1:
        continue

    sid = clean(cg["scan_frame_id"].iloc[0])
    merge_rows.append({
        "clip_id": clip_id,
        "scan_frame_id": sid,
        "merge_visual_colour": colour,
        "merge_behaviour_pig_id": COLOUR_TO_BEHAVIOUR_ID.get(colour, ""),
        "candidate_track_ids": " | ".join(map(str, tids)),
        "candidate_tracklet_count": len(tids),
        "status": "same_colour_tracklet_merge_candidate",
        "reason": "Multiple high/medium full-tracklet colour evidence tracklets share the same visual colour.",
    })

merge_candidates = pd.DataFrame(merge_rows)
safe_to_csv(merge_candidates, OUT_MERGE_CANDIDATES)

# Clip summary.
clip_rows = []

for clip_id, cg in tracklet_evidence.groupby("clip_id", sort=True):
    sid = clean(cg["scan_frame_id"].iloc[0])
    raw_tracklets = int(cg["dense_track_id"].nunique())

    assigned_hm = cg[
        cg["v29c_assignment_status"].astype(str).str.contains("high|medium", regex=True)
        & cg["v29c_assigned_visual_colour"].isin(VALID_COLOURS)
    ]

    assigned_low = cg[cg["v29c_assignment_status"].astype(str).str.contains("low", regex=True)]
    unknown = cg[~cg.index.isin(assigned_hm.index) & ~cg.index.isin(assigned_low.index)]

    recovered = cg[cg["v29b_v29c_comparison_status"] == "recovered_from_v29b_unmatched"]
    conflicts = cg[cg["v29b_v29c_comparison_status"] == "conflict_with_v29b_needs_review"]
    agreements = cg[cg["v29b_v29c_comparison_status"] == "agreement_with_v29b"]

    colours = sorted(set(assigned_hm["v29c_assigned_visual_colour"].dropna().astype(str)))
    merge_clip = merge_candidates[merge_candidates["clip_id"] == clip_id]

    expected = np.nan
    old = clip_summary[clip_summary["clip_id"].astype(str).str.contains(clean(clip_id).replace("__dense_polygon", ""), regex=False)]
    if len(old):
        expected = to_num(old.iloc[0].get("expected_target_pig_rows", np.nan), np.nan)

    if pd.isna(expected):
        expected = ""

    clip_rows.append({
        "clip_id": clip_id,
        "scan_frame_id": sid,
        "expected_target_pig_rows": expected,
        "raw_tracklets": raw_tracklets,
        "v29c_high_medium_assigned_tracklets": int(len(assigned_hm)),
        "v29c_low_assigned_tracklets": int(len(assigned_low)),
        "v29c_unknown_tracklets": int(len(unknown)),
        "v29c_unique_high_medium_colours": int(len(colours)),
        "v29c_high_medium_colours": " | ".join(colours),
        "v29c_recovered_from_v29b_unmatched": int(len(recovered)),
        "v29c_agreements_with_v29b": int(len(agreements)),
        "v29c_conflicts_with_v29b": int(len(conflicts)),
        "same_colour_merge_candidate_count": int(len(merge_clip)),
        "same_colour_merge_candidate_colours": " | ".join(sorted(set(merge_clip["merge_visual_colour"].astype(str)))) if len(merge_clip) else "",
        "clip_status": (
            "promising_but_conflicts_need_review" if len(conflicts) else
            "promising_full_tracklet_colour_evidence"
        ),
    })

clip_out = pd.DataFrame(clip_rows)
safe_to_csv(clip_out, OUT_CLIP_SUMMARY)

contact_index = pd.DataFrame(contact_rows)
safe_to_csv(contact_index, OUT_CONTACT_INDEX)

issues_df = pd.DataFrame(issues, columns=["issue_type", "issue_detail"])
safe_to_csv(issues_df, OUT_ISSUES)

total_tracklets = int(len(tracklet_evidence))
hm_assigned = int(
    tracklet_evidence["v29c_assignment_status"].astype(str).str.contains("high|medium", regex=True).sum()
) if len(tracklet_evidence) else 0
low_assigned = int(
    tracklet_evidence["v29c_assignment_status"].astype(str).str.contains("low", regex=True).sum()
) if len(tracklet_evidence) else 0
unknown_count = total_tracklets - hm_assigned - low_assigned
recovered_count = int((tracklet_evidence["v29b_v29c_comparison_status"] == "recovered_from_v29b_unmatched").sum()) if len(tracklet_evidence) else 0
conflict_count = int((tracklet_evidence["v29b_v29c_comparison_status"] == "conflict_with_v29b_needs_review").sum()) if len(tracklet_evidence) else 0
agreement_count = int((tracklet_evidence["v29b_v29c_comparison_status"] == "agreement_with_v29b").sum()) if len(tracklet_evidence) else 0

decision = pd.DataFrame([{
    "v29c_decision": "full_tracklet_colour_evidence_prototype_completed",
    "total_tracklets": total_tracklets,
    "v29c_high_medium_assigned_tracklets": hm_assigned,
    "v29c_low_assigned_tracklets": low_assigned,
    "v29c_unknown_tracklets": unknown_count,
    "v29c_recovered_from_v29b_unmatched": recovered_count,
    "v29c_agreement_with_v29b": agreement_count,
    "v29c_conflicts_with_v29b": conflict_count,
    "same_colour_merge_candidates": int(len(merge_candidates)),
    "clips_evaluated": int(len(clip_out)),
    "contact_sheets_created": int(len(contact_index)),
    "issue_count": int(len(issues_df)),
    "ready_for_visual_review": bool(len(contact_index) > 0),
    "ready_for_v29d_merge_candidate_analysis": bool(total_tracklets > 0),
    "recommended_next_step": "visual review of full-tracklet colour evidence sheets, then same-colour tracklet merge candidate analysis",
    "generated_at": datetime.now().isoformat(timespec="seconds"),
}])

safe_to_csv(decision, OUT_DECISION)

OUT_LIMITATIONS.write_text(
    "# Week 7 v29c Full-tracklet Colour Evidence Limitations\n\n"
    "## What this step does\n\n"
    "This step estimates visual marker colour evidence over the full tracklet crop sequence using HSV-based colour evidence aggregation.\n\n"
    "## What this step does not do\n\n"
    "It does not yet replace manual identity validation. It does not force every tracklet into a colour class.\n\n"
    "## Important caveats\n\n"
    "1. HSV evidence can be affected by lighting, blur, occlusion and camera exposure.\n"
    "2. Unknown is a valid output when marker evidence is weak.\n"
    "3. Conflicts with v29b center-overlap assignment must be visually reviewed.\n"
    "4. Same-colour merge candidates are suggestions, not final merged identities.\n"
    "5. v29d should evaluate merge candidates and decide whether the full-tracklet colour evidence is reliable enough.\n"
)

OUT_README.write_text(
    "# Week 7 Full-tracklet Colour Evidence v29c\n\n"
    "## Purpose\n\n"
    "This step aggregates marker-colour evidence over every dense tracklet crop sequence.\n\n"
    "## Outputs\n\n"
    "- `week7_full_tracklet_colour_evidence_v29c_detection_colour_evidence.csv`\n"
    "- `week7_full_tracklet_colour_evidence_v29c_tracklet_colour_evidence.csv`\n"
    "- `week7_full_tracklet_colour_evidence_v29c_compare_to_v29b.csv`\n"
    "- `week7_full_tracklet_colour_evidence_v29c_same_colour_merge_candidates.csv`\n"
    "- `contact_sheets/`\n"
)

OUT_NOTE.write_text(
    "# Week 7 Full-tracklet Colour Evidence v29c\n\n"
    "## Purpose\n\n"
    "This step aggregates HSV-based visual marker colour evidence across full dense tracklets.\n\n"
    "## Summary\n\n"
    f"- Total tracklets: `{int(decision.iloc[0]['total_tracklets'])}`\n"
    f"- High/medium assigned tracklets: `{int(decision.iloc[0]['v29c_high_medium_assigned_tracklets'])}`\n"
    f"- Low assigned tracklets: `{int(decision.iloc[0]['v29c_low_assigned_tracklets'])}`\n"
    f"- Unknown tracklets: `{int(decision.iloc[0]['v29c_unknown_tracklets'])}`\n"
    f"- Recovered from v29b unmatched: `{int(decision.iloc[0]['v29c_recovered_from_v29b_unmatched'])}`\n"
    f"- Agreement with v29b: `{int(decision.iloc[0]['v29c_agreement_with_v29b'])}`\n"
    f"- Conflicts with v29b: `{int(decision.iloc[0]['v29c_conflicts_with_v29b'])}`\n"
    f"- Same-colour merge candidates: `{int(decision.iloc[0]['same_colour_merge_candidates'])}`\n"
    f"- Contact sheets created: `{int(decision.iloc[0]['contact_sheets_created'])}`\n"
    f"- Issue count: `{int(decision.iloc[0]['issue_count'])}`\n"
    f"- Ready for v29d merge candidate analysis: `{bool(decision.iloc[0]['ready_for_v29d_merge_candidate_analysis'])}`\n\n"
    "## Outputs\n\n"
    f"- Decision summary: `{OUT_DECISION}`\n"
    f"- Clip summary: `{OUT_CLIP_SUMMARY}`\n"
    f"- Tracklet evidence: `{OUT_TRACKLET_EVIDENCE}`\n"
    f"- Compare to v29b: `{OUT_COMPARE_V29B}`\n"
    f"- Merge candidates: `{OUT_MERGE_CANDIDATES}`\n"
    f"- Contact sheets: `{CONTACT_ROOT}`\n"
    f"- Issues: `{OUT_ISSUES}`\n"
)

print("Saved:")
print(OUT_DET_EVIDENCE)
print(OUT_TRACKLET_EVIDENCE)
print(OUT_COMPARE_V29B)
print(OUT_CLIP_SUMMARY)
print(OUT_MERGE_CANDIDATES)
print(OUT_DECISION)
print(OUT_CONTACT_INDEX)
print(OUT_LIMITATIONS)
print(OUT_ISSUES)
print(OUT_README)
print(OUT_NOTE)

print()
print("=== v29c decision ===")
print(decision.to_string(index=False))

print()
print("=== v29c clip summary ===")
if len(clip_out):
    print(clip_out.to_string(index=False))
else:
    print("No clip summary.")

print()
print("=== v29c merge candidates ===")
if len(merge_candidates):
    print(merge_candidates.to_string(index=False))
else:
    print("No merge candidates.")

print()
print("=== v29c issues ===")
if len(issues_df):
    print(issues_df.to_string(index=False))
else:
    print("No issues found.")
