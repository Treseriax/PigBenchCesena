from pathlib import Path
from datetime import datetime
import csv
import math
import random

import pandas as pd
from PIL import Image, ImageDraw, ImageFont


ROOT = Path.home() / "PigBench"
W7 = ROOT / "Week7_Dataset_Validation_Identity_Tracking_Behaviour_Preparation"

V22_ROOT = W7 / "outputs" / "model_ready_crop_dataset_v22"
V22_INDEX = V22_ROOT / "week7_model_ready_crop_dataset_v22_index.csv"

OUT_ROOT = W7 / "outputs" / "crop_visual_qa_contact_sheets_v23"
SHEETS_ROOT = OUT_ROOT / "contact_sheets"
OUT_ROOT.mkdir(parents=True, exist_ok=True)
SHEETS_ROOT.mkdir(parents=True, exist_ok=True)

OUT_SUMMARY = OUT_ROOT / "week7_crop_visual_qa_v23_summary.csv"
OUT_ISSUES = OUT_ROOT / "week7_crop_visual_qa_v23_issues.csv"
OUT_CLASS_SAMPLE_INDEX = OUT_ROOT / "week7_crop_visual_qa_v23_class_sample_index.csv"
OUT_NOTE = W7 / "notes" / "week7_crop_visual_qa_contact_sheets_v23_notes.md"

RANDOM_SEED = 42
SAMPLES_PER_SPLIT_CLASS = 12
THUMB_W = 160
THUMB_H = 120
LABEL_H = 45
PADDING = 8
COLS = 4


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
    out = "".join(ch if ch.isalnum() or ch in "._-" else "_" for ch in s)
    while "__" in out:
        out = out.replace("__", "_")
    return out.strip("_") or "unknown"


def load_font(size=12):
    try:
        return ImageFont.truetype("DejaVuSans.ttf", size)
    except Exception:
        return ImageFont.load_default()


def make_contact_sheet(rows, out_path, title):
    font = load_font(12)
    title_font = load_font(16)

    n = len(rows)
    if n == 0:
        return False

    rows_count = math.ceil(n / COLS)
    cell_w = THUMB_W + 2 * PADDING
    cell_h = THUMB_H + LABEL_H + 2 * PADDING
    title_h = 36

    sheet_w = COLS * cell_w
    sheet_h = title_h + rows_count * cell_h

    sheet = Image.new("RGB", (sheet_w, sheet_h), "white")
    draw = ImageDraw.Draw(sheet)
    draw.text((PADDING, PADDING), title, fill="black", font=title_font)

    for idx, row in enumerate(rows):
        r = idx // COLS
        c = idx % COLS

        x0 = c * cell_w + PADDING
        y0 = title_h + r * cell_h + PADDING

        crop_path = Path(row["crop_path"])

        try:
            with Image.open(crop_path) as im:
                im = im.convert("RGB")
                im.thumbnail((THUMB_W, THUMB_H))

                bg = Image.new("RGB", (THUMB_W, THUMB_H), "white")
                px = (THUMB_W - im.width) // 2
                py = (THUMB_H - im.height) // 2
                bg.paste(im, (px, py))

                sheet.paste(bg, (x0, y0))

        except Exception:
            bg = Image.new("RGB", (THUMB_W, THUMB_H), "lightgray")
            d = ImageDraw.Draw(bg)
            d.text((8, 8), "LOAD ERROR", fill="black", font=font)
            sheet.paste(bg, (x0, y0))

        label = (
            f"{row.get('scan_frame_id', '')}\n"
            f"{row.get('visual_marker_colour_v18c', '')} → {row.get('behaviour_pig_id_v18c', '')}\n"
            f"{row.get('behaviour_code', '')}"
        )

        draw.text(
            (x0, y0 + THUMB_H + 4),
            label,
            fill="black",
            font=font,
        )

    sheet.save(out_path, quality=95)
    return True


random.seed(RANDOM_SEED)

df = pd.read_csv(V22_INDEX)

for c in [
    "split",
    "scan_frame_id",
    "final_box_id",
    "visual_marker_colour_v18c",
    "behaviour_pig_id_v18c",
    "behaviour_code",
    "behaviour_label",
    "crop_path",
]:
    if c in df.columns:
        df[c] = df[c].fillna("").astype(str).str.strip()

issues = []
sample_rows = []
sheet_rows = []

# Basic file readability QA.
for i, row in df.iterrows():
    p = Path(row["crop_path"])

    if not p.exists():
        issues.append({
            "row_index": i,
            "issue_type": "crop_file_missing",
            "issue_detail": str(p),
        })
        continue

    try:
        with Image.open(p) as im:
            w, h = im.size
            if w <= 1 or h <= 1:
                issues.append({
                    "row_index": i,
                    "issue_type": "invalid_crop_size",
                    "issue_detail": f"{p} size={w}x{h}",
                })
    except Exception as e:
        issues.append({
            "row_index": i,
            "issue_type": "crop_file_unreadable",
            "issue_detail": f"{p}: {repr(e)}",
        })

# Per split + behaviour contact sheets.
for (split, behaviour_code), g in df.groupby(["split", "behaviour_code"], sort=True):
    g = g.copy()

    if len(g) > SAMPLES_PER_SPLIT_CLASS:
        sample = g.sample(SAMPLES_PER_SPLIT_CLASS, random_state=RANDOM_SEED)
    else:
        sample = g

    sample = sample.sort_values(["scan_frame_id", "final_box_id"])

    out_name = f"{slug(split)}__{slug(behaviour_code)}__n{len(sample)}.jpg"
    out_path = SHEETS_ROOT / out_name

    title = f"{split} / {behaviour_code} / sample {len(sample)} of {len(g)}"
    made = make_contact_sheet(sample.to_dict("records"), out_path, title)

    if made:
        sheet_rows.append({
            "split": split,
            "behaviour_code": behaviour_code,
            "behaviour_label": sample["behaviour_label"].iloc[0] if len(sample) else "",
            "total_crops_in_group": int(len(g)),
            "sampled_crops": int(len(sample)),
            "contact_sheet_path": str(out_path),
        })

    for _, r in sample.iterrows():
        d = r.to_dict()
        d["contact_sheet_path"] = str(out_path)
        sample_rows.append(d)

# Overall split overview sheets.
for split, g in df.groupby("split", sort=True):
    g = g.copy()

    # Balanced-ish sample: up to 3 per behaviour.
    samples = []
    for behaviour_code, gg in g.groupby("behaviour_code", sort=True):
        n = min(3, len(gg))
        samples.append(gg.sample(n, random_state=RANDOM_SEED))
    sample = pd.concat(samples, ignore_index=True) if samples else g.head(0)
    sample = sample.sort_values(["behaviour_code", "scan_frame_id", "final_box_id"])

    out_path = SHEETS_ROOT / f"{slug(split)}__overview.jpg"
    title = f"{split} overview / {len(sample)} sampled crops"
    made = make_contact_sheet(sample.to_dict("records"), out_path, title)

    if made:
        sheet_rows.append({
            "split": split,
            "behaviour_code": "OVERVIEW",
            "behaviour_label": "overview",
            "total_crops_in_group": int(len(g)),
            "sampled_crops": int(len(sample)),
            "contact_sheet_path": str(out_path),
        })

sheet_index = pd.DataFrame(sheet_rows)
sample_index = pd.DataFrame(sample_rows)
issues_df = pd.DataFrame(issues, columns=["row_index", "issue_type", "issue_detail"])

safe_to_csv(sample_index, OUT_CLASS_SAMPLE_INDEX)
safe_to_csv(issues_df, OUT_ISSUES)

summary = pd.DataFrame([{
    "source_v22_index": str(V22_INDEX),
    "total_crops_indexed": int(len(df)),
    "contact_sheets_created": int(len(sheet_index)),
    "sampled_crop_rows": int(len(sample_index)),
    "issue_count": int(len(issues_df)),
    "sheets_root": str(SHEETS_ROOT),
    "ready_for_feature_extraction": bool(len(issues_df) == 0 and len(df) > 0),
    "generated_at": datetime.now().isoformat(timespec="seconds"),
}])

safe_to_csv(summary, OUT_SUMMARY)

# Save sheet index too.
OUT_SHEET_INDEX = OUT_ROOT / "week7_crop_visual_qa_v23_contact_sheet_index.csv"
safe_to_csv(sheet_index, OUT_SHEET_INDEX)

ready = bool(summary.iloc[0]["ready_for_feature_extraction"])

OUT_NOTE.write_text(
    "# Week 7 Crop Visual QA Contact Sheets v23\n\n"
    "## Purpose\n\n"
    "This step creates visual contact sheets from the v22 model-ready crop dataset. "
    "The goal is to quickly inspect crop quality before feature extraction or model training.\n\n"
    "## Summary\n\n"
    f"- Total crops indexed: `{int(summary.iloc[0]['total_crops_indexed'])}`\n"
    f"- Contact sheets created: `{int(summary.iloc[0]['contact_sheets_created'])}`\n"
    f"- Sampled crop rows: `{int(summary.iloc[0]['sampled_crop_rows'])}`\n"
    f"- Issue count: `{int(summary.iloc[0]['issue_count'])}`\n"
    f"- Ready for feature extraction: `{ready}`\n\n"
    "## Outputs\n\n"
    f"- Contact sheets folder: `{SHEETS_ROOT}`\n"
    f"- Contact sheet index: `{OUT_SHEET_INDEX}`\n"
    f"- Sample index: `{OUT_CLASS_SAMPLE_INDEX}`\n"
    f"- Issues: `{OUT_ISSUES}`\n"
    f"- Summary: `{OUT_SUMMARY}`\n"
)

print("Saved:")
print(SHEETS_ROOT)
print(OUT_SHEET_INDEX)
print(OUT_CLASS_SAMPLE_INDEX)
print(OUT_ISSUES)
print(OUT_SUMMARY)
print(OUT_NOTE)

print()
print("=== v23 crop visual QA summary ===")
print(summary.to_string(index=False))

print()
print("=== v23 contact sheets ===")
print(sheet_index.head(50).to_string(index=False))

print()
print("=== v23 issues ===")
if len(issues_df):
    print(issues_df.head(50).to_string(index=False))
else:
    print("No issues found.")
