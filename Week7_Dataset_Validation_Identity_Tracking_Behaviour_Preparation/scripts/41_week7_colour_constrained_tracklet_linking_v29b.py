from pathlib import Path
from datetime import datetime
import csv
import json
import math
import re

import cv2
import numpy as np
import pandas as pd
from PIL import Image, ImageDraw, ImageFont


ROOT = Path.home() / "PigBench"
W7 = ROOT / "Week7_Dataset_Validation_Identity_Tracking_Behaviour_Preparation"

V28D_ROOT = W7 / "outputs" / "dense_polygon_filtered_tracking_v28d"
V28D_TRACKS = V28D_ROOT / "week7_dense_polygon_filtered_tracking_v28d_tracks.csv"
V28D_CLIP_SUMMARY = V28D_ROOT / "week7_dense_polygon_filtered_tracking_v28d_clip_summary.csv"
V28D_FRAME_SUMMARY = V28D_ROOT / "week7_dense_polygon_filtered_tracking_v28d_frame_summary.csv"

V18C_BOX_LEVEL = W7 / "outputs" / "behaviour_label_fusion_v18c_verified_crosswalk" / "week7_behaviour_label_fusion_v18c_box_level_dataset.csv"
V26_INDEX = W7 / "outputs" / "clip_extraction_temporal_qa_v26" / "week7_clip_extraction_v26_index.csv"

OUT_ROOT = W7 / "outputs" / "colour_constrained_tracklet_linking_v29b"
CONTACT_ROOT = OUT_ROOT / "contact_sheets"
OVERLAY_ROOT = OUT_ROOT / "identity_linking_overlays"

for p in [OUT_ROOT, CONTACT_ROOT, OVERLAY_ROOT]:
    p.mkdir(parents=True, exist_ok=True)

OUT_GT_BOXES = OUT_ROOT / "week7_colour_constrained_tracklet_linking_v29b_center_gt_boxes.csv"
OUT_MATCHES = OUT_ROOT / "week7_colour_constrained_tracklet_linking_v29b_center_matches.csv"
OUT_TRACKLET_IDENTITY = OUT_ROOT / "week7_colour_constrained_tracklet_linking_v29b_tracklet_identity_candidates.csv"
OUT_LINKED_TRACKS = OUT_ROOT / "week7_colour_constrained_tracklet_linking_v29b_linked_track_detections.csv"
OUT_CLIP_SUMMARY = OUT_ROOT / "week7_colour_constrained_tracklet_linking_v29b_clip_summary.csv"
OUT_DECISION = OUT_ROOT / "week7_colour_constrained_tracklet_linking_v29b_decision_summary.csv"
OUT_CONTACT_INDEX = OUT_ROOT / "week7_colour_constrained_tracklet_linking_v29b_contact_sheet_index.csv"
OUT_LIMITATIONS = OUT_ROOT / "week7_colour_constrained_tracklet_linking_v29b_limitations.md"
OUT_ISSUES = OUT_ROOT / "week7_colour_constrained_tracklet_linking_v29b_issues.csv"
OUT_README = OUT_ROOT / "README_colour_constrained_tracklet_linking_v29b.md"
OUT_NOTE = W7 / "notes" / "week7_colour_constrained_tracklet_linking_v29b_notes.md"

CENTER_WINDOW_FRAMES = 20
MIN_IOU_FOR_CANDIDATE = 0.05
HIGH_IOU = 0.30
MEDIUM_IOU = 0.15


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


def bbox_iou(a, b):
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b

    ix1 = max(ax1, bx1)
    iy1 = max(ay1, by1)
    ix2 = min(ax2, bx2)
    iy2 = min(ay2, by2)

    iw = max(0.0, ix2 - ix1)
    ih = max(0.0, iy2 - iy1)
    inter = iw * ih

    area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)

    union = area_a + area_b - inter

    if union <= 0:
        return 0.0

    return inter / union


def bbox_from_row(row):
    direct_sets = [
        ("x1", "y1", "x2", "y2"),
        ("final_x1", "final_y1", "final_x2", "final_y2"),
        ("bbox_x1", "bbox_y1", "bbox_x2", "bbox_y2"),
        ("xmin", "ymin", "xmax", "ymax"),
    ]

    for cols in direct_sets:
        if all(c in row.index for c in cols):
            vals = [to_num(row[c]) for c in cols]
            if all(not pd.isna(v) for v in vals):
                return vals

    for c in ["bbox", "final_bbox", "box"]:
        if c in row.index:
            nums = re.findall(r"-?\d+(?:\.\d+)?", str(row[c]))
            if len(nums) >= 4:
                return [float(x) for x in nums[:4]]

    return None


def choose_col(df, candidates):
    for c in candidates:
        if c in df.columns:
            return c
    return None


def get_center_frame_map(v26):
    out = {}

    for _, r in v26.iterrows():
        sid = clean(r.get("scan_frame_id", ""))
        fps = to_num(r.get("fps_used", 25.0), 25.0)

        center_frame = to_num(r.get("center_frame_index", np.nan), np.nan)
        start_frame = to_num(r.get("start_frame_adjusted", np.nan), np.nan)

        if pd.isna(center_frame) or pd.isna(start_frame):
            center_sec = to_num(r.get("center_sec", np.nan), np.nan)
            start_sec = to_num(r.get("start_sec", np.nan), np.nan)

            if pd.isna(center_sec) or pd.isna(start_sec):
                rel_center = np.nan
            else:
                rel_center = int(round((center_sec - start_sec) * fps))
        else:
            rel_center = int(round(center_frame - start_frame))

        out[sid] = {
            "center_frame_in_clip": rel_center,
            "fps": fps,
        }

    return out


def classify_assignment(best_iou, candidates, best_colour):
    if not best_colour:
        return "unmatched_no_colour"

    if best_iou >= HIGH_IOU:
        return "assigned_high_iou"

    if best_iou >= MEDIUM_IOU:
        return "assigned_medium_iou"

    if best_iou >= MIN_IOU_FOR_CANDIDATE:
        return "assigned_low_iou_needs_review"

    return "unmatched_low_iou"


def draw_identity_overlay(base_path, tracks_for_frame, gt_boxes, title, out_path):
    try:
        im = Image.open(base_path).convert("RGB")
    except Exception:
        return False

    draw = ImageDraw.Draw(im)
    font = load_font(12)
    title_font = load_font(15)

    colour_map = {
        "blue": (0, 90, 255),
        "green": (0, 190, 0),
        "cyan": (0, 210, 230),
        "red": (255, 30, 30),
        "pink": (255, 60, 200),
        "purple": (150, 60, 255),
        "unknown": (160, 160, 160),
    }

    draw.rectangle((0, 0, im.width, 48), fill=(255, 255, 255))
    draw.text((8, 8), title, fill=(0, 0, 0), font=title_font)
    draw.text((8, 30), "solid boxes=track detections; dashed/thin boxes=GT colour boxes", fill=(0, 0, 0), font=font)

    # GT boxes.
    for _, g in gt_boxes.iterrows():
        c = clean(g.get("visual_marker_colour", "")) or "unknown"
        col = colour_map.get(c, colour_map["unknown"])
        x1, y1, x2, y2 = [int(round(float(g[k]))) for k in ["gt_x1", "gt_y1", "gt_x2", "gt_y2"]]
        draw.rectangle((x1, y1, x2, y2), outline=col, width=2)
        draw.text((x1, max(50, y1 - 14)), f"GT {c}", fill=col, font=font)

    # Track detections.
    for _, t in tracks_for_frame.iterrows():
        c = clean(t.get("assigned_visual_colour", "")) or "unknown"
        col = colour_map.get(c, colour_map["unknown"])
        x1, y1, x2, y2 = [int(round(float(t[k]))) for k in ["x1", "y1", "x2", "y2"]]
        tid = clean(t.get("dense_track_id", ""))
        bid = clean(t.get("assigned_behaviour_pig_id", ""))
        score = to_num(t.get("score", 0.0), 0.0)

        for k in range(3):
            draw.rectangle((x1-k, y1-k, x2+k, y2+k), outline=col)

        label = f"T{tid}->{c}/{bid} {score:.2f}"
        try:
            tb = draw.textbbox((x1, max(50, y1 - 18)), label, font=font)
            draw.rectangle((tb[0], tb[1], tb[2] + 4, tb[3] + 4), fill=(255, 255, 255))
        except Exception:
            pass

        draw.text((x1 + 2, max(50, y1 - 17)), label, fill=col, font=font)

    im.save(out_path, quality=95)
    return True


def make_contact_sheet(paths, out_path, title, cols=2, thumb_w=420, thumb_h=300):
    if not paths:
        return False

    font = load_font(12)
    title_font = load_font(16)

    pad = 8
    title_h = 42
    cell_w = thumb_w + 2 * pad
    cell_h = thumb_h + 40 + 2 * pad
    rows = math.ceil(len(paths) / cols)

    sheet = Image.new("RGB", (cols * cell_w, title_h + rows * cell_h), "white")
    draw = ImageDraw.Draw(sheet)
    draw.text((pad, pad), title, fill=(0, 0, 0), font=title_font)

    for i, p in enumerate(paths):
        p = Path(p)
        r = i // cols
        c = i % cols
        x0 = c * cell_w + pad
        y0 = title_h + r * cell_h + pad

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

        draw.text((x0, y0 + thumb_h + 4), p.stem, fill=(0, 0, 0), font=font)

    sheet.save(out_path, quality=95)
    return True


issues = []

required = [V28D_TRACKS, V28D_CLIP_SUMMARY, V28D_FRAME_SUMMARY, V18C_BOX_LEVEL, V26_INDEX]
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
frame_summary = pd.read_csv(V28D_FRAME_SUMMARY)
box = pd.read_csv(V18C_BOX_LEVEL)
v26 = pd.read_csv(V26_INDEX)

for df in [tracks, clip_summary, frame_summary, box, v26]:
    if "scan_frame_id" in df.columns:
        df["scan_frame_id"] = df["scan_frame_id"].fillna("").astype(str).str.strip()
    if "clip_id" in df.columns:
        df["clip_id"] = df["clip_id"].fillna("").astype(str).str.strip()

for c in ["x1", "y1", "x2", "y2", "score", "frame_index", "dense_track_id"]:
    if c in tracks.columns:
        tracks[c] = pd.to_numeric(tracks[c], errors="coerce")

center_map = get_center_frame_map(v26)

# Prepare GT boxes from v18c.
colour_col = choose_col(box, ["visual_marker_colour_v18c", "final_colour_identity_v17", "final_colour_v16"])
behaviour_pig_col = choose_col(box, ["behaviour_pig_id_v18c", "behaviour_pig_id", "pig_id"])
behaviour_code_col = choose_col(box, ["behaviour_code", "behaviour_code_label"])
final_box_id_col = choose_col(box, ["final_box_id", "box_id", "det_id"])

gt_rows = []

for _, r in box.iterrows():
    sid = clean(r.get("scan_frame_id", ""))
    b = bbox_from_row(r)

    if b is None:
        continue

    colour = clean(r.get(colour_col, "")) if colour_col else ""
    behaviour_pig_id = clean(r.get(behaviour_pig_col, "")) if behaviour_pig_col else ""
    behaviour_code = clean(r.get(behaviour_code_col, "")) if behaviour_code_col else ""
    final_box_id = clean(r.get(final_box_id_col, "")) if final_box_id_col else ""

    if not colour:
        colour = "unknown"

    x1, y1, x2, y2 = b

    gt_rows.append({
        "scan_frame_id": sid,
        "final_box_id": final_box_id,
        "visual_marker_colour": colour,
        "behaviour_pig_id": behaviour_pig_id,
        "behaviour_code": behaviour_code,
        "gt_x1": x1,
        "gt_y1": y1,
        "gt_x2": x2,
        "gt_y2": y2,
    })

gt = pd.DataFrame(gt_rows)
safe_to_csv(gt, OUT_GT_BOXES)

# Match track detections near scanpoint center to GT boxes.
match_rows = []
tracklet_rows = []

for clip_id, cg in tracks.groupby("clip_id", sort=True):
    sid = clean(cg["scan_frame_id"].iloc[0])
    center_info = center_map.get(sid, {"center_frame_in_clip": np.nan})
    center_frame = to_num(center_info.get("center_frame_in_clip", np.nan), np.nan)

    if pd.isna(center_frame):
        issues.append({
            "issue_type": "missing_center_frame",
            "issue_detail": sid,
        })
        continue

    window_min = center_frame - CENTER_WINDOW_FRAMES
    window_max = center_frame + CENTER_WINDOW_FRAMES

    near = cg[(cg["frame_index"] >= window_min) & (cg["frame_index"] <= window_max)].copy()
    ggt = gt[gt["scan_frame_id"] == sid].copy()

    if len(near) == 0:
        issues.append({
            "issue_type": "no_track_detections_near_center",
            "issue_detail": f"{sid}, center={center_frame}",
        })
        continue

    if len(ggt) == 0:
        issues.append({
            "issue_type": "no_gt_boxes_for_scanframe",
            "issue_detail": sid,
        })
        continue

    # Match every near-center track detection to every GT.
    for _, tr in near.iterrows():
        tb = [float(tr["x1"]), float(tr["y1"]), float(tr["x2"]), float(tr["y2"])]

        for _, gr in ggt.iterrows():
            gb = [float(gr["gt_x1"]), float(gr["gt_y1"]), float(gr["gt_x2"]), float(gr["gt_y2"])]
            iou = bbox_iou(tb, gb)

            if iou < MIN_IOU_FOR_CANDIDATE:
                continue

            match_rows.append({
                "clip_id": clip_id,
                "scan_frame_id": sid,
                "center_frame_in_clip": int(round(center_frame)),
                "window_min": int(round(window_min)),
                "window_max": int(round(window_max)),
                "track_frame_index": int(tr["frame_index"]),
                "dense_track_id": int(tr["dense_track_id"]),
                "track_score": round(float(tr["score"]), 6),
                "track_x1": tr["x1"],
                "track_y1": tr["y1"],
                "track_x2": tr["x2"],
                "track_y2": tr["y2"],
                "final_box_id": gr["final_box_id"],
                "visual_marker_colour": gr["visual_marker_colour"],
                "behaviour_pig_id": gr["behaviour_pig_id"],
                "behaviour_code": gr["behaviour_code"],
                "gt_x1": gr["gt_x1"],
                "gt_y1": gr["gt_y1"],
                "gt_x2": gr["gt_x2"],
                "gt_y2": gr["gt_y2"],
                "iou": round(float(iou), 6),
            })

matches = pd.DataFrame(match_rows)
safe_to_csv(matches, OUT_MATCHES)

# Tracklet identity assignment by weighted IoU consensus.
for (clip_id, tid), tg in tracks.groupby(["clip_id", "dense_track_id"], sort=True):
    sid = clean(tg["scan_frame_id"].iloc[0])

    m = matches[(matches["clip_id"] == clip_id) & (matches["dense_track_id"] == int(tid))].copy()

    if len(m):
        # Sum IoU by colour / behaviour pig ID.
        agg = (
            m.groupby(["visual_marker_colour", "behaviour_pig_id", "behaviour_code", "final_box_id"], dropna=False)
            .agg(
                match_count=("iou", "count"),
                iou_sum=("iou", "sum"),
                iou_max=("iou", "max"),
                iou_mean=("iou", "mean"),
            )
            .reset_index()
            .sort_values(["iou_sum", "iou_max", "match_count"], ascending=[False, False, False])
        )

        best = agg.iloc[0]
        best_colour = clean(best["visual_marker_colour"])
        best_behaviour_pig_id = clean(best["behaviour_pig_id"])
        best_behaviour_code = clean(best["behaviour_code"])
        best_final_box_id = clean(best["final_box_id"])
        best_iou = float(best["iou_max"])
        second_iou_sum = float(agg.iloc[1]["iou_sum"]) if len(agg) > 1 else 0.0
        ambiguity_gap = float(best["iou_sum"]) - second_iou_sum
        assignment_status = classify_assignment(best_iou, len(agg), best_colour)

        if len(agg) > 1 and ambiguity_gap < 0.05:
            assignment_status = assignment_status + "_ambiguous"
    else:
        best_colour = ""
        best_behaviour_pig_id = ""
        best_behaviour_code = ""
        best_final_box_id = ""
        best_iou = 0.0
        second_iou_sum = 0.0
        ambiguity_gap = 0.0
        assignment_status = "unmatched_no_center_overlap"

    frame_min = int(tg["frame_index"].min())
    frame_max = int(tg["frame_index"].max())
    frames_seen = int(tg["frame_index"].nunique())

    tracklet_rows.append({
        "clip_id": clip_id,
        "scan_frame_id": sid,
        "dense_track_id": int(tid),
        "tracklet_frames_seen": frames_seen,
        "tracklet_frame_min": frame_min,
        "tracklet_frame_max": frame_max,
        "tracklet_score_mean": round(float(tg["score"].mean()), 6),
        "assigned_visual_colour": best_colour,
        "assigned_behaviour_pig_id": best_behaviour_pig_id,
        "assigned_behaviour_code_at_center": best_behaviour_code,
        "assigned_final_box_id": best_final_box_id,
        "center_match_iou_max": round(best_iou, 6),
        "center_match_candidate_count": int(len(m)),
        "assignment_status": assignment_status,
        "assignment_ambiguity_gap_iou_sum": round(ambiguity_gap, 6),
    })

tracklet_identity = pd.DataFrame(tracklet_rows)
safe_to_csv(tracklet_identity, OUT_TRACKLET_IDENTITY)

# Join identity back to tracks.
linked = tracks.merge(
    tracklet_identity[
        [
            "clip_id",
            "dense_track_id",
            "assigned_visual_colour",
            "assigned_behaviour_pig_id",
            "assigned_behaviour_code_at_center",
            "assigned_final_box_id",
            "center_match_iou_max",
            "assignment_status",
        ]
    ],
    on=["clip_id", "dense_track_id"],
    how="left",
)

safe_to_csv(linked, OUT_LINKED_TRACKS)

# Clip summaries.
clip_rows = []

for clip_id, cg in linked.groupby("clip_id", sort=True):
    sid = clean(cg["scan_frame_id"].iloc[0])
    tids = tracklet_identity[tracklet_identity["clip_id"] == clip_id].copy()

    assigned = tids[tids["assigned_visual_colour"].fillna("").astype(str).str.len() > 0].copy()
    high_medium = assigned[assigned["assignment_status"].astype(str).str.contains("high|medium", regex=True)].copy()
    low = assigned[assigned["assignment_status"].astype(str).str.contains("low", regex=True)].copy()
    unmatched = tids[tids["assigned_visual_colour"].fillna("").astype(str).str.len() == 0].copy()

    unique_raw_tracklets = int(tids["dense_track_id"].nunique())
    unique_assigned_colours = int(assigned["assigned_visual_colour"].nunique()) if len(assigned) else 0
    unique_assigned_behaviour_ids = int(assigned["assigned_behaviour_pig_id"].nunique()) if len(assigned) else 0

    duplicate_colour_tracklets = 0
    duplicate_colour_detail = []

    if len(assigned):
        counts = assigned.groupby("assigned_visual_colour")["dense_track_id"].nunique()
        for colour, count in counts.items():
            if count > 1:
                duplicate_colour_tracklets += int(count - 1)
                duplicate_colour_detail.append(f"{colour}:{int(count)}")

    expected = to_num(
        clip_summary[clip_summary["clip_id"] == clip_id]["expected_target_pig_rows"].iloc[0]
        if len(clip_summary[clip_summary["clip_id"] == clip_id]) else np.nan,
        np.nan
    )

    raw_frag_ratio = unique_raw_tracklets / expected if expected and not pd.isna(expected) and expected > 0 else np.nan
    colour_linked_ratio = unique_assigned_colours / expected if expected and not pd.isna(expected) and expected > 0 else np.nan

    clip_rows.append({
        "clip_id": clip_id,
        "scan_frame_id": sid,
        "expected_target_pig_rows": int(expected) if not pd.isna(expected) else "",
        "raw_dense_tracklets": unique_raw_tracklets,
        "assigned_tracklets": int(len(assigned)),
        "high_medium_confidence_assigned_tracklets": int(len(high_medium)),
        "low_confidence_assigned_tracklets": int(len(low)),
        "unmatched_tracklets": int(len(unmatched)),
        "unique_assigned_visual_colours": unique_assigned_colours,
        "unique_assigned_behaviour_pig_ids": unique_assigned_behaviour_ids,
        "duplicate_tracklets_after_colour_linking": int(duplicate_colour_tracklets),
        "duplicate_colour_detail": " | ".join(duplicate_colour_detail),
        "raw_fragmentation_ratio_tracklets_over_expected": round(raw_frag_ratio, 4) if not pd.isna(raw_frag_ratio) else "",
        "colour_linked_ratio_colours_over_expected": round(colour_linked_ratio, 4) if not pd.isna(colour_linked_ratio) else "",
        "assigned_visual_colours": " | ".join(sorted(set(assigned["assigned_visual_colour"].dropna().astype(str)))) if len(assigned) else "",
        "assigned_behaviour_pig_ids": " | ".join(sorted(set(assigned["assigned_behaviour_pig_id"].dropna().astype(str)))) if len(assigned) else "",
        "prototype_status": (
            "promising_reduces_fragmentation"
            if unique_assigned_colours < unique_raw_tracklets and unique_assigned_colours > 0
            else "needs_review"
        ),
    })

clip_link_summary = pd.DataFrame(clip_rows)
safe_to_csv(clip_link_summary, OUT_CLIP_SUMMARY)

# Visual overlays near center frame.
contact_rows = []

for clip_id, cg in linked.groupby("clip_id", sort=True):
    sid = clean(cg["scan_frame_id"].iloc[0])
    center_info = center_map.get(sid, {"center_frame_in_clip": np.nan})
    center_frame = to_num(center_info.get("center_frame_in_clip", np.nan), np.nan)

    if pd.isna(center_frame):
        continue

    # Pick up to 4 frames closest to the center.
    fg = frame_summary[frame_summary["scan_frame_id"] == sid].copy()
    if len(fg) == 0:
        continue

    fg["center_dist"] = (pd.to_numeric(fg["frame_index"], errors="coerce") - center_frame).abs()
    chosen = fg.sort_values("center_dist").head(4)

    gt_sid = gt[gt["scan_frame_id"] == sid].copy()
    overlay_paths = []

    for _, fr in chosen.iterrows():
        fidx = int(fr["frame_index"])
        base_path = clean(fr["annotated_frame_path"])
        tracks_frame = linked[(linked["clip_id"] == clip_id) & (linked["frame_index"] == fidx)].copy()

        out_path = OVERLAY_ROOT / f"{slug(clip_id)}__frame_{fidx:06d}_identity_linking_v29b.jpg"

        made = draw_identity_overlay(
            base_path,
            tracks_frame,
            gt_sid,
            f"{sid} colour-constrained identity linking | frame {fidx}",
            out_path,
        )

        if made:
            overlay_paths.append(str(out_path))

    if overlay_paths:
        sheet_path = CONTACT_ROOT / f"{slug(clip_id)}_identity_linking_contact_sheet.jpg"
        made = make_contact_sheet(
            overlay_paths,
            sheet_path,
            f"v29b identity linking review | {sid}",
        )

        if made:
            contact_rows.append({
                "clip_id": clip_id,
                "scan_frame_id": sid,
                "contact_sheet_path": str(sheet_path),
                "overlay_count": len(overlay_paths),
            })

contact_index = pd.DataFrame(contact_rows)
safe_to_csv(contact_index, OUT_CONTACT_INDEX)

# Decision.
total_tracklets = int(len(tracklet_identity))
assigned_tracklets = int(tracklet_identity["assigned_visual_colour"].fillna("").astype(str).str.len().gt(0).sum()) if len(tracklet_identity) else 0
high_medium_assigned = int(tracklet_identity["assignment_status"].astype(str).str.contains("high|medium", regex=True).sum()) if len(tracklet_identity) else 0
unmatched_tracklets = total_tracklets - assigned_tracklets

promising_clips = int(clip_link_summary["prototype_status"].eq("promising_reduces_fragmentation").sum()) if len(clip_link_summary) else 0
problem_clips = int(len(clip_link_summary) - promising_clips)

decision = pd.DataFrame([{
    "v29b_decision": "colour_constrained_tracklet_linking_prototype_completed",
    "total_tracklets": total_tracklets,
    "assigned_tracklets": assigned_tracklets,
    "high_medium_confidence_assigned_tracklets": high_medium_assigned,
    "unmatched_tracklets": unmatched_tracklets,
    "clips_evaluated": int(len(clip_link_summary)),
    "promising_clips": promising_clips,
    "clips_needing_review": problem_clips,
    "contact_sheets_created": int(len(contact_index)),
    "issue_count": int(len(issues)),
    "ready_for_visual_review": bool(len(contact_index) > 0),
    "ready_for_v29c_refinement": bool(total_tracklets > 0),
    "recommended_next_step": "visual review of identity-linking contact sheets; then refine linking with colour evidence over full tracklet, not center-frame only",
    "generated_at": datetime.now().isoformat(timespec="seconds"),
}])

safe_to_csv(decision, OUT_DECISION)

OUT_LIMITATIONS.write_text(
    "# Week 7 v29b Colour-constrained Tracklet Linking Limitations\n\n"
    "## What this step does\n\n"
    "This step links v28d dense polygon-filtered tracklets to visual marker colour and behaviour pig ID candidates using overlap with final scanpoint GT boxes near the scanpoint center.\n\n"
    "## What this step does not do\n\n"
    "It does not yet perform full robust colour recognition across the entire 10-second clip. It uses center-frame overlap as the first prototype signal.\n\n"
    "## Important caveats\n\n"
    "1. Tracklets not visible near the scanpoint center may remain unmatched.\n"
    "2. Low-IoU assignments require visual review.\n"
    "3. Fragmented tracklets may be repairable if multiple tracklets map to the same visual colour.\n"
    "4. The next refinement should use marker-colour evidence over the whole tracklet, not only center-frame GT overlap.\n"
    "5. This step should be treated as an identity-linking prototype, not final identity tracking.\n"
)

OUT_README.write_text(
    "# Week 7 Colour-constrained Tracklet Identity Linking v29b\n\n"
    "## Purpose\n\n"
    "This step tests whether fragmented dense tracklets can be linked to marker colour and behaviour pig ID using scanpoint GT identity boxes.\n\n"
    "## Outputs\n\n"
    "- `week7_colour_constrained_tracklet_linking_v29b_tracklet_identity_candidates.csv`\n"
    "- `week7_colour_constrained_tracklet_linking_v29b_linked_track_detections.csv`\n"
    "- `week7_colour_constrained_tracklet_linking_v29b_clip_summary.csv`\n"
    "- `week7_colour_constrained_tracklet_linking_v29b_decision_summary.csv`\n"
    "- `contact_sheets/`\n"
)

OUT_NOTE.write_text(
    "# Week 7 Colour-constrained Tracklet Identity Linking v29b\n\n"
    "## Purpose\n\n"
    "This step links v28d dense polygon tracklets to final visual marker colour / behaviour pig ID candidates using center-frame GT overlap.\n\n"
    "## Summary\n\n"
    f"- Total tracklets: `{int(decision.iloc[0]['total_tracklets'])}`\n"
    f"- Assigned tracklets: `{int(decision.iloc[0]['assigned_tracklets'])}`\n"
    f"- High/medium confidence assigned tracklets: `{int(decision.iloc[0]['high_medium_confidence_assigned_tracklets'])}`\n"
    f"- Unmatched tracklets: `{int(decision.iloc[0]['unmatched_tracklets'])}`\n"
    f"- Clips evaluated: `{int(decision.iloc[0]['clips_evaluated'])}`\n"
    f"- Promising clips: `{int(decision.iloc[0]['promising_clips'])}`\n"
    f"- Clips needing review: `{int(decision.iloc[0]['clips_needing_review'])}`\n"
    f"- Contact sheets created: `{int(decision.iloc[0]['contact_sheets_created'])}`\n"
    f"- Issue count: `{int(decision.iloc[0]['issue_count'])}`\n"
    f"- Ready for v29c refinement: `{bool(decision.iloc[0]['ready_for_v29c_refinement'])}`\n\n"
    "## Outputs\n\n"
    f"- Decision summary: `{OUT_DECISION}`\n"
    f"- Clip summary: `{OUT_CLIP_SUMMARY}`\n"
    f"- Tracklet candidates: `{OUT_TRACKLET_IDENTITY}`\n"
    f"- Linked detections: `{OUT_LINKED_TRACKS}`\n"
    f"- Contact sheets: `{CONTACT_ROOT}`\n"
    f"- Issues: `{OUT_ISSUES}`\n"
)

issues_df = pd.DataFrame(issues, columns=["issue_type", "issue_detail"])
safe_to_csv(issues_df, OUT_ISSUES)

print("Saved:")
print(OUT_GT_BOXES)
print(OUT_MATCHES)
print(OUT_TRACKLET_IDENTITY)
print(OUT_LINKED_TRACKS)
print(OUT_CLIP_SUMMARY)
print(OUT_DECISION)
print(OUT_CONTACT_INDEX)
print(OUT_LIMITATIONS)
print(OUT_ISSUES)
print(OUT_README)
print(OUT_NOTE)

print()
print("=== v29b decision ===")
print(decision.to_string(index=False))

print()
print("=== v29b clip summary ===")
if len(clip_link_summary):
    print(clip_link_summary.to_string(index=False))
else:
    print("No clip summary.")

print()
print("=== v29b issues ===")
if len(issues_df):
    print(issues_df.to_string(index=False))
else:
    print("No issues found.")
