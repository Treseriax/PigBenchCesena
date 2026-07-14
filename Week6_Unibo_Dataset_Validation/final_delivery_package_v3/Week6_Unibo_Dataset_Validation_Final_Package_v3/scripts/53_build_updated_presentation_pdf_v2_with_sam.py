from pathlib import Path
from datetime import datetime
import csv
import pandas as pd
from PIL import Image, ImageDraw, ImageFont, ImageOps


ROOT = Path.home() / "PigBench"
W6 = ROOT / "Week6_Unibo_Dataset_Validation"

OUT_DIR = W6 / "final_presentation"
OUT_DIR.mkdir(parents=True, exist_ok=True)

STATS = W6 / "outputs" / "dataset_statistics"
VIS = W6 / "outputs" / "visual_label_check"
NOTES = W6 / "notes"

PDF_PATH = OUT_DIR / "Week6_Unibo_Dataset_Validation_Presentation_v2_with_Segment_Anything_Model.pdf"
SUMMARY_PATH = STATS / "week6_presentation_v2_generation_summary.csv"
NOTE_PATH = NOTES / "week6_presentation_v2_with_segment_anything_model_notes.md"

SAM_CONTACT = VIS / "week6_sam_box_prompt_segmentation_contact_sheet.jpg"
CLASSICAL_SEG_CONTACT = VIS / "week6_preliminary_segmentation_contact_sheet.jpg"
BBOX_CONTACT = VIS / "bbox_count_warning_qc" / "all_bbox_count_warning_frames_contact_sheet.jpg"


def safe_to_csv(df, path):
    df.to_csv(
        path,
        index=False,
        quoting=csv.QUOTE_ALL,
        escapechar="\\",
        lineterminator="\n",
    )


def font(size, bold=False):
    candidates = [
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf" if bold else "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        "/usr/share/fonts/truetype/liberation2/LiberationSans-Bold.ttf" if bold else "/usr/share/fonts/truetype/liberation2/LiberationSans-Regular.ttf",
    ]
    for p in candidates:
        if Path(p).exists():
            return ImageFont.truetype(p, size)
    return ImageFont.load_default()


def wrap(draw, text, fnt, max_width):
    words = str(text).split()
    lines = []
    cur = ""
    for w in words:
        cand = w if not cur else cur + " " + w
        if draw.textbbox((0, 0), cand, font=fnt)[2] <= max_width:
            cur = cand
        else:
            if cur:
                lines.append(cur)
            cur = w
    if cur:
        lines.append(cur)
    return lines


def draw_wrapped(draw, text, x, y, fnt, fill, max_width, gap=8):
    for line in wrap(draw, text, fnt, max_width):
        draw.text((x, y), line, font=fnt, fill=fill)
        y += fnt.size + gap
    return y


def paste_image(slide, path, box, caption=None):
    draw = ImageDraw.Draw(slide)
    path = Path(path)

    if not path.exists():
        draw.rectangle(box, outline=(170, 170, 170), width=3)
        draw.text(
            (box[0] + 20, box[1] + 20),
            "Image not found:\n" + str(path.name),
            font=font(24),
            fill=(150, 0, 0),
        )
        return

    img = Image.open(path).convert("RGB")
    target_w = box[2] - box[0]
    target_h = box[3] - box[1]
    img = ImageOps.contain(img, (target_w, target_h))

    x = box[0] + (target_w - img.width) // 2
    y = box[1] + (target_h - img.height) // 2

    slide.paste(img, (x, y))
    draw.rectangle((x, y, x + img.width, y + img.height), outline=(210, 210, 210), width=2)

    if caption:
        draw.text((box[0], box[3] + 12), caption, font=font(21), fill=(80, 80, 80))


def make_slide(title, subtitle=None, bullets=None, image_path=None, caption=None):
    W, H = 1600, 900
    slide = Image.new("RGB", (W, H), (248, 250, 252))
    draw = ImageDraw.Draw(slide)

    navy = (24, 44, 73)
    blue = (42, 90, 150)
    dark = (30, 30, 30)

    draw.rectangle((0, 0, W, 94), fill=navy)
    draw.text((60, 26), title, font=font(38, True), fill=(255, 255, 255))

    if subtitle:
        draw.text((64, 116), subtitle, font=font(26), fill=blue)

    if image_path:
        x = 70
        y = 180
        text_w = 650
        image_box = (780, 150, 1530, 775)
    else:
        x = 115
        y = 180
        text_w = 1360
        image_box = None

    if bullets:
        bullet_font = font(30)
        for b in bullets:
            draw.ellipse((x, y + 11, x + 13, y + 24), fill=blue)
            y = draw_wrapped(draw, b, x + 30, y, bullet_font, dark, text_w)
            y += 22

    if image_path:
        paste_image(slide, image_path, image_box, caption)

    draw.line((60, 835, 1540, 835), fill=(210, 210, 210), width=2)
    draw.text((60, 852), "Week 6 Unibo Dataset Validation", font=font(20), fill=(90, 90, 90))

    return slide


slides = []

slides.append(make_slide(
    "Week 6 Unibo Dataset Validation",
    "From raw videos and scan-sampling labels to structured visual evidence",
    [
        "Objective: prepare a reliable experimental dataset for future pig behaviour classification.",
        "Main outcome: labelled scanpoint frames, detector bounding boxes, feature tables, segmentation outputs, visual quality control artifacts, and a final delivery package.",
        "The work keeps manual labels, automatic detections, automatic masks, and candidate identity evidence clearly separated."
    ]
))

slides.append(make_slide(
    "Dataset and Annotation Conversion",
    "Manual behaviour annotations were converted into unified ground truth files",
    [
        "The Excel scan-sampling annotations were parsed into a structured table.",
        "The final dataset contains 432 manual behaviour labels.",
        "The labels are linked to 72 scanpoint frames.",
        "Each scanpoint frame contains six pig colour identifiers.",
        "The unified ground truth was exported as Comma-Separated Values and JavaScript Object Notation files."
    ]
))

slides.append(make_slide(
    "Detection and Visual Alignment",
    "Automatic pig detections were linked to scanpoint frames",
    [
        "The You Only Look Once version 8 small pig detector was applied to all scanpoint frames.",
        "The detector produced 540 pig bounding boxes across 72 frames.",
        "Bounding box quality control flags and contact sheets were generated.",
        "These detections support crop extraction, learned embeddings, marker evidence, and segmentation."
    ],
    image_path=BBOX_CONTACT,
    caption="Bounding box quality control contact sheet"
))

slides.append(make_slide(
    "Feature Extraction Outputs",
    "Multiple feature families were generated",
    [
        "Bounding box geometry features describe location, size, and normalized area.",
        "Group-spatial features describe spread, spacing, overlap, and contact-related proxies.",
        "Region of interest proxy features describe coarse position relative to pen resources.",
        "Crop descriptor and colour-marker features support candidate colour identity evidence.",
        "Detector-backed learned crop embeddings were generated for all 540 detected pig crops."
    ]
))

slides.append(make_slide(
    "Detector-Backed Learned Crop Embeddings",
    "The embedding route uses the local pig detector checkpoint",
    [
        "A model capability audit was performed before generating learned embeddings.",
        "The final route used the local PigBench You Only Look Once version 8 small detector checkpoint.",
        "The method produced 896-dimensional learned crop embeddings.",
        "Embedding rows generated: 540.",
        "This is not a random or untrained embedding route."
    ]
))

slides.append(make_slide(
    "Classical Segmentation Baseline",
    "A first segmentation route was implemented before using a foundation model",
    [
        "The classical route used bounding-box-guided GrabCut with Otsu fallback.",
        "It produced segmentation-derived features for 540 detections.",
        "It generated overlay images for all 72 scanpoint frames.",
        "Manual visual quality control accepted it for preliminary feature extraction with notes."
    ],
    image_path=CLASSICAL_SEG_CONTACT,
    caption="Classical GrabCut and Otsu segmentation contact sheet"
))

slides.append(make_slide(
    "Advanced Segmentation Extension",
    "Segment Anything Model was added with bounding box prompts",
    [
        "The Segment Anything Model package was installed and verified.",
        "The Segment Anything Model checkpoint was downloaded and loaded.",
        "The smoke test passed on five sample detector bounding boxes.",
        "The full run used each detector bounding box as a box prompt.",
        "This adds a foundation-model-based segmentation route."
    ]
))

slides.append(make_slide(
    "Segment Anything Model Results",
    "The full box-prompt segmentation run completed successfully",
    [
        "Input detector bounding boxes: 540.",
        "Generated Segment Anything Model masks: 540.",
        "Failed masks: 0.",
        "Frame overlay images: 72.",
        "Mean predicted mask quality score: approximately 0.94."
    ],
    image_path=SAM_CONTACT,
    caption="Segment Anything Model box-prompt segmentation contact sheet"
))

slides.append(make_slide(
    "Classical Baseline versus Segment Anything Model",
    "Two segmentation routes are now available",
    [
        "The classical GrabCut and Otsu route provides a transparent computer vision baseline.",
        "The Segment Anything Model route provides cleaner object-shaped masks using the same detector bounding boxes.",
        "The comparison matched 540 rows between both routes.",
        "The Segment Anything Model masks had a larger average mask area, which is expected because they often cover fuller pig body shapes."
    ]
))

slides.append(make_slide(
    "Visualization Interface",
    "The outputs can be inspected through a browser-based static viewer",
    [
        "The static viewer displays scanpoint frames and visual outputs.",
        "It can be served from the Week 6 project folder using the Python Hypertext Transfer Protocol server.",
        "The raw videos are not duplicated in the package because they already exist on the server.",
        "Derived frames, overlays, tables, summaries, and scripts are included."
    ]
))

slides.append(make_slide(
    "Final Deliverables",
    "The delivery package contains the complete Week 6 evidence set",
    [
        "Unified ground truth tables, JavaScript Object Notation files, and schema files.",
        "Recommended training, validation, and test split protocol.",
        "Detection, marker, embedding, classical segmentation, and Segment Anything Model feature outputs.",
        "Visualization assets, contact sheets, quality control notes, scripts, shared tracker, and this presentation."
    ]
))

slides.append(make_slide(
    "Final Takeaway",
    "The Week 6 pipeline is ready for presentation and future modelling",
    [
        "The task sheet requirements are complete.",
        "The pipeline was strengthened with detector-backed embeddings and Segment Anything Model segmentation.",
        "The final outputs are suitable for future pig behaviour classification experiments.",
        "Automatic detections and automatic masks are clearly documented as model outputs, not manual ground truth."
    ]
))

slides[0].save(PDF_PATH, save_all=True, append_images=slides[1:], resolution=150)

summary = pd.DataFrame([
    {
        "artifact": "presentation_pdf_v2",
        "path": str(PDF_PATH),
        "exists": PDF_PATH.exists(),
        "size_mb": round(PDF_PATH.stat().st_size / (1024 * 1024), 3),
    },
    {
        "artifact": "slide_count",
        "path": "",
        "exists": True,
        "size_mb": len(slides),
    },
    {
        "artifact": "segment_anything_contact_sheet",
        "path": str(SAM_CONTACT),
        "exists": SAM_CONTACT.exists(),
        "size_mb": round(SAM_CONTACT.stat().st_size / (1024 * 1024), 3) if SAM_CONTACT.exists() else "",
    },
])
safe_to_csv(summary, SUMMARY_PATH)

NOTE_PATH.write_text(
    "# Week 6 Presentation v2 with Segment Anything Model\n\n"
    f"Generated: `{datetime.now().isoformat(timespec='seconds')}`\n\n"
    "## Output\n\n"
    f"- Presentation PDF: `{PDF_PATH}`\n\n"
    "## Main update\n\n"
    "This version adds the Segment Anything Model results:\n\n"
    "- 540 masks for 540 detector bounding boxes.\n"
    "- 72 frame overlays.\n"
    "- 0 failed rows.\n"
    "- Comparison against the classical GrabCut and Otsu baseline.\n\n"
    "## Interpretation\n\n"
    "This is the recommended presentation PDF for explaining the final Week 6 work.\n"
)

print("Saved presentation:")
print(PDF_PATH)
print()
print("Saved summary:")
print(SUMMARY_PATH)
print()
print("Saved note:")
print(NOTE_PATH)
print()
print("=== Summary ===")
print(summary.to_string(index=False))
