from pathlib import Path
from datetime import datetime
import csv
import re
import math

import pandas as pd
from PIL import Image, ImageDraw, ImageFont


ROOT = Path.home() / "PigBench"
W7 = ROOT / "Week7_Dataset_Validation_Identity_Tracking_Behaviour_Preparation"

V18C_ROOT = W7 / "outputs" / "behaviour_label_fusion_v18c_verified_crosswalk"
BOX_LEVEL = V18C_ROOT / "week7_behaviour_label_fusion_v18c_box_level_dataset.csv"

OUT_ROOT = W7 / "outputs" / "final_colour_coded_bbox_overlays_v23b"
OVERLAY_ROOT = OUT_ROOT / "frame_overlays"
CONTACT_ROOT = OUT_ROOT / "contact_sheets"
OUT_ROOT.mkdir(parents=True, exist_ok=True)
OVERLAY_ROOT.mkdir(parents=True, exist_ok=True)
CONTACT_ROOT.mkdir(parents=True, exist_ok=True)

OUT_INDEX = OUT_ROOT / "week7_final_colour_coded_bbox_overlays_v23b_index.csv"
OUT_FRAME_QA = OUT_ROOT / "week7_final_colour_coded_bbox_overlays_v23b_frame_qa.csv"
OUT_SUMMARY = OUT_ROOT / "week7_final_colour_coded_bbox_overlays_v23b_summary.csv"
OUT_ISSUES = OUT_ROOT / "week7_final_colour_coded_bbox_overlays_v23b_issues.csv"
OUT_CONTACT_INDEX = OUT_ROOT / "week7_final_colour_coded_bbox_overlays_v23b_contact_sheet_index.csv"
OUT_NOTE = W7 / "notes" / "week7_final_colour_coded_bbox_overlays_v23b_notes.md"

VALID_VISUAL = ["blue", "green", "cyan", "red", "pink", "purple"]

COLOURS = {
    "blue": (0, 90, 255),
    "green": (0, 190, 0),
    "cyan": (0, 210, 230),
    "red": (255, 30, 30),
    "pink": (255, 60, 200),
    "purple": (150, 60, 255),
    "unknown": (150, 150, 150),
}

LINE_WIDTH = 4


def safe_to_csv(df, path):
    df.to_csv(
        path,
        index=False,
        quoting=csv.QUOTE_ALL,
        escapechar="\\",
        lineterminator="\n",
    )


def slug(s):
    s = str(s).strip()
    s = re.sub(r"[^A-Za-z0-9_.-]+", "_", s)
    s = re.sub(r"_+", "_", s).strip("_")
    return s or "unknown"


def load_font(size=16):
    try:
        return ImageFont.truetype("DejaVuSans.ttf", size)
    except Exception:
        return ImageFont.load_default()


def clean(v):
    if pd.isna(v):
        return ""
    return str(v).strip()


def find_image_path_from_group(g):
    candidates = [
        "frame_image_path",
        "frame_image_path_label",
        "frame_image_path_identity",
        "image_path",
        "img_path",
        "path",
    ]

    for c in candidates:
        if c in g.columns:
            vals = g[c].dropna().astype(str).str.strip()
            vals = vals[~vals.str.lower().isin(["", "nan", "none", "null"])]
            for v in vals:
                if Path(v).exists():
                    return v

    return ""


def bbox_from_row(row):
    direct_sets = [
        ("x1", "y1", "x2", "y2"),
        ("final_x1", "final_y1", "final_x2", "final_y2"),
        ("bbox_x1", "bbox_y1", "bbox_x2", "bbox_y2"),
        ("xmin", "ymin", "xmax", "ymax"),
    ]

    for cols in direct_sets:
        if all(c in row.index for c in cols):
            try:
                return tuple(float(row[c]) for c in cols)
            except Exception:
                pass

    for c in ["bbox", "final_bbox", "box"]:
        if c in row.index:
            text = str(row[c])
            nums = re.findall(r"-?\d+(?:\.\d+)?", text)
            if len(nums) >= 4:
                return tuple(float(x) for x in nums[:4])

    return None


def get_visual_colour(row):
    for c in ["visual_marker_colour_v18c", "final_colour_identity_v17", "final_colour_v16"]:
        if c in row.index:
            v = clean(row.get(c, "")).lower()
            if v in VALID_VISUAL:
                return v
    return ""


def get_behaviour_pig_id(row):
    for c in ["behaviour_pig_id_v18c", "pig_id", "colour_id"]:
        if c in row.index:
            v = clean(row.get(c, ""))
            if v:
                return v
    return ""


def get_behaviour_code(row):
    for c in ["behaviour_code", "behaviour_code_label"]:
        if c in row.index:
            v = clean(row.get(c, ""))
            if v:
                return v
    return ""


def draw_label(draw, xy, text, fill, font):
    x, y = xy
    # Text background.
    try:
        bbox = draw.textbbox((x, y), text, font=font)
        bg = (bbox[0], bbox[1], bbox[2] + 4, bbox[3] + 4)
    except Exception:
        bg = (x, y, x + 260, y + 24)

    draw.rectangle(bg, fill=(255, 255, 255))
    draw.text((x + 2, y + 2), text, fill=fill, font=font)


df = pd.read_csv(BOX_LEVEL)

for c in [
    "scan_frame_id",
    "final_box_id",
    "visual_marker_colour_v18c",
    "final_colour_identity_v17",
    "behaviour_pig_id_v18c",
    "behaviour_code",
    "final_identity_status_v17",
]:
    if c in df.columns:
        df[c] = df[c].fillna("").astype(str).str.strip()

issues = []
index_rows = []
frame_rows = []

font = load_font(14)
small_font = load_font(12)

for scan_frame_id, g in df.groupby("scan_frame_id", sort=True):
    image_path = find_image_path_from_group(g)

    if not image_path:
        issues.append({
            "scan_frame_id": scan_frame_id,
            "final_box_id": "",
            "issue_type": "missing_frame_image_path",
            "issue_detail": "No valid image path found for frame group",
        })
        continue

    try:
        im = Image.open(image_path).convert("RGB")
    except Exception as e:
        issues.append({
            "scan_frame_id": scan_frame_id,
            "final_box_id": "",
            "issue_type": "frame_image_open_failed",
            "issue_detail": f"{image_path}: {repr(e)}",
        })
        continue

    w, h = im.size
    draw = ImageDraw.Draw(im)

    valid_count = 0
    unknown_count = 0
    drawn_count = 0

    for _, row in g.iterrows():
        final_box_id = clean(row.get("final_box_id", ""))
        bbox = bbox_from_row(row)

        if bbox is None:
            issues.append({
                "scan_frame_id": scan_frame_id,
                "final_box_id": final_box_id,
                "issue_type": "missing_bbox",
                "issue_detail": "Could not parse bbox",
            })
            continue

        x1, y1, x2, y2 = bbox
        x1c = max(0, min(w - 1, int(round(x1))))
        y1c = max(0, min(h - 1, int(round(y1))))
        x2c = max(0, min(w, int(round(x2))))
        y2c = max(0, min(h, int(round(y2))))

        if x2c <= x1c or y2c <= y1c:
            issues.append({
                "scan_frame_id": scan_frame_id,
                "final_box_id": final_box_id,
                "issue_type": "invalid_bbox_after_clamp",
                "issue_detail": f"{bbox} -> {(x1c, y1c, x2c, y2c)}",
            })
            continue

        visual_colour = get_visual_colour(row)
        behaviour_pig_id = get_behaviour_pig_id(row)
        behaviour_code = get_behaviour_code(row)
        identity_status = clean(row.get("final_identity_status_v17", ""))

        if visual_colour in VALID_VISUAL:
            box_colour = COLOURS[visual_colour]
            valid_count += 1
        else:
            box_colour = COLOURS["unknown"]
            unknown_count += 1

        # Draw thicker rectangle.
        for k in range(LINE_WIDTH):
            draw.rectangle(
                (x1c - k, y1c - k, x2c + k, y2c + k),
                outline=box_colour,
            )

        label_colour = visual_colour if visual_colour else "unknown"
        label = f"{label_colour}"
        if behaviour_pig_id:
            label += f"→{behaviour_pig_id}"
        if behaviour_code:
            label += f" | {behaviour_code}"
        elif identity_status:
            label += f" | {identity_status}"

        draw_label(
            draw,
            (x1c, max(0, y1c - 22)),
            label,
            box_colour,
            small_font,
        )

        drawn_count += 1

        index_rows.append({
            "scan_frame_id": scan_frame_id,
            "final_box_id": final_box_id,
            "visual_marker_colour": visual_colour,
            "behaviour_pig_id": behaviour_pig_id,
            "behaviour_code": behaviour_code,
            "final_identity_status_v17": identity_status,
            "image_path": image_path,
            "x1": x1,
            "y1": y1,
            "x2": x2,
            "y2": y2,
            "x1_clamped": x1c,
            "y1_clamped": y1c,
            "x2_clamped": x2c,
            "y2_clamped": y2c,
        })

    # Legend.
    legend_x = 10
    legend_y = 10
    draw.rectangle((legend_x - 6, legend_y - 6, legend_x + 420, legend_y + 135), fill=(255, 255, 255))
    draw.text((legend_x, legend_y), f"{scan_frame_id} | colour-coded final boxes", fill=(0, 0, 0), font=font)

    ly = legend_y + 25
    for colour_name in VALID_VISUAL + ["unknown"]:
        c = COLOURS[colour_name]
        draw.rectangle((legend_x, ly, legend_x + 18, ly + 14), fill=c)
        draw.text((legend_x + 25, ly - 2), colour_name, fill=(0, 0, 0), font=small_font)
        ly += 17

    out_path = OVERLAY_ROOT / f"{slug(scan_frame_id)}_final_colour_coded_boxes_v23b.jpg"
    im.save(out_path, quality=95)

    frame_rows.append({
        "scan_frame_id": scan_frame_id,
        "source_image_path": image_path,
        "overlay_path": str(out_path),
        "box_rows": int(len(g)),
        "drawn_boxes": int(drawn_count),
        "valid_colour_boxes": int(valid_count),
        "unknown_colour_boxes": int(unknown_count),
        "issue_count_for_frame": int(sum(1 for x in issues if x["scan_frame_id"] == scan_frame_id)),
    })


index_df = pd.DataFrame(index_rows)
frame_qa = pd.DataFrame(frame_rows)
issues_df = pd.DataFrame(issues, columns=["scan_frame_id", "final_box_id", "issue_type", "issue_detail"])

safe_to_csv(index_df, OUT_INDEX)
safe_to_csv(frame_qa, OUT_FRAME_QA)
safe_to_csv(issues_df, OUT_ISSUES)

# Contact sheets over overlays.
contact_rows = []

def make_contact_sheet(paths, out_path, title, cols=4, thumb_w=280, thumb_h=180):
    if not paths:
        return False

    title_h = 40
    pad = 8
    rows = math.ceil(len(paths) / cols)
    cell_w = thumb_w + 2 * pad
    cell_h = thumb_h + 45 + 2 * pad

    sheet = Image.new("RGB", (cols * cell_w, title_h + rows * cell_h), "white")
    d = ImageDraw.Draw(sheet)
    d.text((pad, pad), title, fill=(0, 0, 0), font=font)

    for i, p in enumerate(paths):
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
            dd.text((8, 8), "LOAD ERROR", fill=(0, 0, 0), font=small_font)
            sheet.paste(bg, (x0, y0))

        d.text((x0, y0 + thumb_h + 4), Path(p).stem.replace("_final_colour_coded_boxes_v23b", ""), fill=(0, 0, 0), font=small_font)

    sheet.save(out_path, quality=95)
    return True


if len(frame_qa):
    all_paths = frame_qa["overlay_path"].tolist()

    # Overall sheets in chunks of 24.
    for idx in range(0, len(all_paths), 24):
        chunk = all_paths[idx:idx + 24]
        out = CONTACT_ROOT / f"all_frames_part_{idx // 24 + 1:02d}.jpg"
        make_contact_sheet(chunk, out, f"Final colour-coded overlays v23b | part {idx // 24 + 1}")
        contact_rows.append({
            "sheet_type": "all_frames",
            "sheet_path": str(out),
            "frame_count": len(chunk),
        })

    # Unknown-heavy frames sheet.
    unknown_frames = frame_qa[frame_qa["unknown_colour_boxes"] > 0]["overlay_path"].tolist()
    if unknown_frames:
        out = CONTACT_ROOT / "frames_with_unknown_or_not_visible_boxes.jpg"
        make_contact_sheet(unknown_frames, out, "Frames with unknown/not-visible/uncertain boxes")
        contact_rows.append({
            "sheet_type": "unknown_frames",
            "sheet_path": str(out),
            "frame_count": len(unknown_frames),
        })

contact_index = pd.DataFrame(contact_rows)
safe_to_csv(contact_index, OUT_CONTACT_INDEX)

summary = pd.DataFrame([{
    "source_box_level_dataset": str(BOX_LEVEL),
    "frames_processed": int(len(frame_qa)),
    "total_box_rows": int(len(df)),
    "drawn_boxes": int(frame_qa["drawn_boxes"].sum()) if len(frame_qa) else 0,
    "valid_colour_boxes": int(frame_qa["valid_colour_boxes"].sum()) if len(frame_qa) else 0,
    "unknown_colour_boxes": int(frame_qa["unknown_colour_boxes"].sum()) if len(frame_qa) else 0,
    "issue_count": int(len(issues_df)),
    "overlay_root": str(OVERLAY_ROOT),
    "contact_sheet_root": str(CONTACT_ROOT),
    "ready_for_report_visuals": bool(len(issues_df) == 0 and len(frame_qa) > 0),
    "generated_at": datetime.now().isoformat(timespec="seconds"),
}])

safe_to_csv(summary, OUT_SUMMARY)

ready = bool(summary.iloc[0]["ready_for_report_visuals"])

OUT_NOTE.write_text(
    "# Week 7 Final Colour-coded BBox Overlays v23b\n\n"
    "## Purpose\n\n"
    "This step redraws final corrected bounding boxes using the final visual marker colour identity. "
    "It provides report-ready visual evidence for the `Pig → Visual Colour → Behaviour Pig ID → Behaviour Label` pipeline.\n\n"
    "## Colour policy\n\n"
    "- blue boxes: visual marker colour `blue`\n"
    "- green boxes: visual marker colour `green`\n"
    "- cyan boxes: visual marker colour `cyan`, mapped to behaviour pig ID `no_color`\n"
    "- red boxes: visual marker colour `red`, mapped to behaviour pig ID `red_neck`\n"
    "- pink/magenta boxes: visual marker colour `pink`, mapped to behaviour pig ID `red_tail`\n"
    "- purple boxes: visual marker colour `purple`\n"
    "- grey boxes: unknown / not_visible / uncertain identity\n\n"
    "## Summary\n\n"
    f"- Frames processed: `{int(summary.iloc[0]['frames_processed'])}`\n"
    f"- Total box rows: `{int(summary.iloc[0]['total_box_rows'])}`\n"
    f"- Drawn boxes: `{int(summary.iloc[0]['drawn_boxes'])}`\n"
    f"- Valid colour boxes: `{int(summary.iloc[0]['valid_colour_boxes'])}`\n"
    f"- Unknown colour boxes: `{int(summary.iloc[0]['unknown_colour_boxes'])}`\n"
    f"- Issue count: `{int(summary.iloc[0]['issue_count'])}`\n"
    f"- Ready for report visuals: `{ready}`\n\n"
    "## Outputs\n\n"
    f"- Frame overlays: `{OVERLAY_ROOT}`\n"
    f"- Contact sheets: `{CONTACT_ROOT}`\n"
    f"- Overlay index: `{OUT_INDEX}`\n"
    f"- Frame QA: `{OUT_FRAME_QA}`\n"
    f"- Contact sheet index: `{OUT_CONTACT_INDEX}`\n"
    f"- Issues: `{OUT_ISSUES}`\n"
    f"- Summary: `{OUT_SUMMARY}`\n"
)

print("Saved:")
print(OVERLAY_ROOT)
print(CONTACT_ROOT)
print(OUT_INDEX)
print(OUT_FRAME_QA)
print(OUT_CONTACT_INDEX)
print(OUT_ISSUES)
print(OUT_SUMMARY)
print(OUT_NOTE)

print()
print("=== v23b final colour-coded bbox overlay summary ===")
print(summary.to_string(index=False))

print()
print("=== v23b frame QA head ===")
print(frame_qa.head(20).to_string(index=False))

print()
print("=== v23b issues ===")
if len(issues_df):
    print(issues_df.head(50).to_string(index=False))
else:
    print("No issues found.")
