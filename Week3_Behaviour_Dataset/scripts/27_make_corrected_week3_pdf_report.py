from pathlib import Path
import shutil
import pandas as pd

from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER, TA_LEFT
from reportlab.lib.pagesizes import A4, landscape
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import cm
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle,
    PageBreak, Image
)
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.pdfbase import pdfmetrics


ROOT = Path("Week3_Behaviour_Dataset")
FINAL = ROOT / "final_outputs_corrected_scan_windows"
OUT_PDF = FINAL / "Week3_Corrected_Scan_Window_Report.pdf"

FIG = FINAL / "figures"
CSV = FINAL / "csv"
NOTES = FINAL / "notes"

doc = SimpleDocTemplate(
    str(OUT_PDF),
    pagesize=landscape(A4),
    rightMargin=1.2 * cm,
    leftMargin=1.2 * cm,
    topMargin=1.0 * cm,
    bottomMargin=1.0 * cm,
)

styles = getSampleStyleSheet()

styles.add(ParagraphStyle(
    name="TitleCenter",
    parent=styles["Title"],
    alignment=TA_CENTER,
    fontSize=26,
    leading=32,
    spaceAfter=18,
))

styles.add(ParagraphStyle(
    name="SectionTitle",
    parent=styles["Heading1"],
    fontSize=20,
    leading=24,
    spaceAfter=12,
))

styles.add(ParagraphStyle(
    name="BodyLarge",
    parent=styles["BodyText"],
    fontSize=11.5,
    leading=16,
    alignment=TA_LEFT,
))

styles.add(ParagraphStyle(
    name="Small",
    parent=styles["BodyText"],
    fontSize=8.5,
    leading=11,
))


story = []


def title_page():
    story.append(Spacer(1, 2.2 * cm))
    story.append(Paragraph("Week 3 Corrected Report", styles["TitleCenter"]))
    story.append(Paragraph("Behaviour Dataset Construction and Visualization", styles["TitleCenter"]))
    story.append(Spacer(1, 0.8 * cm))
    story.append(Paragraph(
        "Corrected scan-window version using manually concatenated observation intervals "
        "and segment-to-Excel-window alignment.",
        styles["BodyLarge"]
    ))
    story.append(Spacer(1, 1.0 * cm))
    story.append(Paragraph("<b>Main correction:</b> the previous no-overlap limitation between video and Excel scan-sampling windows was addressed.", styles["BodyLarge"]))
    story.append(Paragraph("<b>Remaining conservative choice:</b> track-to-colour identity mapping is kept as a manual verification step to avoid false behaviour labels.", styles["BodyLarge"]))
    story.append(PageBreak())


def add_table_from_df(df, max_rows=None, font_size=8):
    if max_rows:
        df = df.head(max_rows).copy()

    data = [list(df.columns)] + df.astype(str).values.tolist()

    table = Table(data, repeatRows=1)
    table.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#1f2937")),
        ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
        ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
        ("FONTSIZE", (0, 0), (-1, -1), font_size),
        ("GRID", (0, 0), (-1, -1), 0.25, colors.grey),
        ("VALIGN", (0, 0), (-1, -1), "TOP"),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#f3f4f6")]),
        ("LEFTPADDING", (0, 0), (-1, -1), 4),
        ("RIGHTPADDING", (0, 0), (-1, -1), 4),
        ("TOPPADDING", (0, 0), (-1, -1), 3),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 3),
    ]))
    story.append(table)


def add_image(path, width_cm=23, max_height_cm=12.5):
    path = Path(path)
    if not path.exists():
        story.append(Paragraph(f"Missing figure: {path.name}", styles["Small"]))
        return

    img = Image(str(path))

    max_w = width_cm * cm
    max_h = max_height_cm * cm

    scale = min(max_w / img.imageWidth, max_h / img.imageHeight)

    img.drawWidth = img.imageWidth * scale
    img.drawHeight = img.imageHeight * scale

    story.append(img)


def section(title):
    story.append(Paragraph(title, styles["SectionTitle"]))


# Page 1
title_page()

# Page 2
section("1. Purpose of the correction")
story.append(Paragraph(
    "The first Week 3 version successfully built the behaviour annotation structure, "
    "but the original video sample did not exactly overlap with the Excel scan-sampling observation windows. "
    "To fix this, a new clean screen-recording video was prepared by placing the relevant observation windows "
    "one after another.",
    styles["BodyLarge"]
))
story.append(Spacer(1, 0.3 * cm))
story.append(Paragraph(
    "The corrected version uses six aligned scan windows: 09:00, 09:10, 09:20, 09:30, 09:40, and 09:50. "
    "Each segment is manually mapped to its corresponding Excel interval.",
    styles["BodyLarge"]
))
story.append(Spacer(1, 0.5 * cm))
add_image(FIG / "segment_confirmation_montage.jpg", width_cm=25)
story.append(PageBreak())

# Page 3
section("2. Corrected processing pipeline")
pipeline = [
    ["Step", "Description"],
    ["1", "Check clean scan-window video metadata and sample frames."],
    ["2", "Create manual segment-to-Excel-window mapping."],
    ["3", "Extract each scan-window segment into a separate image sequence."],
    ["4", "Run YOLOv8-s detection on each segment."],
    ["5", "Run ByteTrack tracking on each segment."],
    ["6", "Attach segment ID and Excel interval information to each track row."],
    ["7", "Parse Excel behaviour labels for each scan window."],
    ["8", "Build corrected JSON annotation dataset."],
    ["9", "Generate visualization videos and package final outputs."],
]
table = Table(pipeline, colWidths=[2 * cm, 22 * cm])
table.setStyle(TableStyle([
    ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#1f2937")),
    ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
    ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
    ("FONTSIZE", (0, 0), (-1, -1), 10),
    ("GRID", (0, 0), (-1, -1), 0.25, colors.grey),
    ("VALIGN", (0, 0), (-1, -1), "TOP"),
    ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#f3f4f6")]),
]))
story.append(table)
story.append(PageBreak())

# Page 4
section("3. Corrected JSON dataset summary")
json_summary = pd.read_csv(CSV / "corrected_scan_window_json_summary.csv")
story.append(Paragraph(
    "The corrected JSON links video frame, track ID, bounding box, segment ID, Excel interval, "
    "and segment-level behaviour candidates.",
    styles["BodyLarge"]
))
story.append(Spacer(1, 0.4 * cm))
add_table_from_df(json_summary, font_size=8)
story.append(PageBreak())

# Page 5
section("4. Tracking summary")
tracking_summary = pd.read_csv(CSV / "scan_window_tracking_with_time_summary.csv")
story.append(Paragraph(
    "ByteTrack was applied separately to each extracted scan-window sequence. "
    "Some segments have more unique track IDs than the number of pigs because of ID fragmentation, "
    "partial visibility, neighbouring pen detections, and screen-recording artefacts.",
    styles["BodyLarge"]
))
story.append(Spacer(1, 0.4 * cm))
add_table_from_df(tracking_summary, font_size=8)
story.append(PageBreak())

# Page 6
section("5. Behaviour labels from Excel")
label_summary = pd.read_csv(CSV / "scan_window_behaviour_label_summary.csv")
story.append(Paragraph(
    "The Excel file provides scan-sampling behaviour labels for colour-coded pig identities. "
    "These labels are now correctly attached to the corresponding scan-window segments.",
    styles["BodyLarge"]
))
story.append(Spacer(1, 0.4 * cm))
add_table_from_df(label_summary, font_size=6.5)
story.append(PageBreak())

# Page 7
section("6. Track ID montage examples")
story.append(Paragraph(
    "Track ID montages were generated for every scan window. These figures support manual verification "
    "of track-to-colour identity mapping.",
    styles["BodyLarge"]
))
story.append(Spacer(1, 0.4 * cm))
add_image(FIG / "scan_09_00_track_id_montage.jpg", width_cm=25)
story.append(PageBreak())

# Page 8
section("7. Additional montage example")
add_image(FIG / "scan_09_50_track_id_montage.jpg", width_cm=25)
story.append(PageBreak())

# Page 9
section("8. Identity mapping policy")
story.append(Paragraph(
    "The tracker produces numerical IDs such as track_id=1 or track_id=2. "
    "However, the Excel file gives labels for colour-coded identities such as green, blue, purple, "
    "red neck, red tail, and no color. Therefore, track-to-colour mapping is required before assigning "
    "a final behaviour label to an individual track.",
    styles["BodyLarge"]
))
story.append(Spacer(1, 0.3 * cm))
story.append(Paragraph(
    "This corrected version keeps individual track identities unverified unless a human manually confirms "
    "the colour identity. This conservative approach avoids assigning incorrect behaviour labels when "
    "colour markers are unclear or when ID fragmentation occurs.",
    styles["BodyLarge"]
))
story.append(Spacer(1, 0.5 * cm))
dominant = pd.read_csv(CSV / "dominant_track_id_to_colour_mapping_table.csv")
dominant_short = dominant[[
    "segment_id", "track_id", "first_frame", "last_frame",
    "num_frames", "mean_score", "mean_cx", "mean_cy"
]].head(18)
add_table_from_df(dominant_short, font_size=7)
story.append(PageBreak())

# Page 10
section("9. Final outputs and conclusion")
story.append(Paragraph(
    "The corrected final package contains corrected JSON annotations, segment-level behaviour labels, "
    "tracking CSV files with Excel window information, identity mapping templates, montages, visualization videos, "
    "scripts, and summary notes.",
    styles["BodyLarge"]
))
story.append(Spacer(1, 0.4 * cm))
story.append(Paragraph(
    "The main previous limitation, no exact overlap between video and Excel scan windows, has been fixed. "
    "The remaining limitation is identity verification: individual track IDs still require manual confirmation "
    "before they can be safely linked to colour-coded pig identities.",
    styles["BodyLarge"]
))
story.append(Spacer(1, 0.4 * cm))
story.append(Paragraph(
    "Overall, the corrected Week 3 dataset now provides a stronger behaviour annotation foundation by linking "
    "frames, tracked pig instances, scan-window timing, and Excel behaviour labels in a structured JSON format.",
    styles["BodyLarge"]
))

doc.build(story)

# Copy report into notes too
shutil.copy2(OUT_PDF, NOTES / OUT_PDF.name)

print("Saved PDF:", OUT_PDF)
print("Copied PDF to notes:", NOTES / OUT_PDF.name)

# Refresh zip
zip_base = ROOT / "Week3_corrected_scan_window_final_outputs"
zip_path = shutil.make_archive(str(zip_base), "zip", FINAL)
print("Updated ZIP:", zip_path)
