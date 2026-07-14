from pathlib import Path
import re
from reportlab.lib import colors
from reportlab.lib.pagesizes import A4, landscape
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import cm
from reportlab.platypus import (
    SimpleDocTemplate,
    Paragraph,
    Spacer,
    PageBreak,
    Table,
    TableStyle,
    Preformatted,
)
from reportlab.pdfbase.pdfmetrics import stringWidth


ROOT = Path("Week4_5_Behaviour_Representation")
FINAL = ROOT / "final_outputs"

MD_PATH = FINAL / "Week4_5_Final_Report_Draft.md"
PDF_PATH = FINAL / "Week4_5_Final_Report_Draft.pdf"

if not MD_PATH.exists():
    raise FileNotFoundError(MD_PATH)

text = MD_PATH.read_text(errors="ignore")

# Basic cleanup for PDF rendering.
text = text.replace("–", "-")
text = text.replace("—", "-")
text = text.replace("“", '"').replace("”", '"')
text = text.replace("’", "'")

styles = getSampleStyleSheet()

styles.add(ParagraphStyle(
    name="ReportTitle",
    parent=styles["Title"],
    fontSize=20,
    leading=24,
    spaceAfter=16,
))

styles.add(ParagraphStyle(
    name="SectionHeading",
    parent=styles["Heading1"],
    fontSize=15,
    leading=18,
    spaceBefore=12,
    spaceAfter=8,
))

styles.add(ParagraphStyle(
    name="SubsectionHeading",
    parent=styles["Heading2"],
    fontSize=12,
    leading=15,
    spaceBefore=10,
    spaceAfter=6,
))

styles.add(ParagraphStyle(
    name="BodyTextSmall",
    parent=styles["BodyText"],
    fontSize=8.5,
    leading=11,
    spaceAfter=6,
))

styles.add(ParagraphStyle(
    name="TableCell",
    parent=styles["BodyText"],
    fontSize=5.8,
    leading=7,
    wordWrap="CJK",
))

styles.add(ParagraphStyle(
    name="TableHeader",
    parent=styles["BodyText"],
    fontSize=6.0,
    leading=7.2,
    textColor=colors.white,
    wordWrap="CJK",
))

styles.add(ParagraphStyle(
    name="CodeBlock",
    parent=styles["Code"],
    fontSize=7,
    leading=8,
    leftIndent=6,
    rightIndent=6,
    spaceAfter=6,
))


def escape_xml(s: str) -> str:
    return (
        str(s)
        .replace("&", "&amp;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
    )


def parse_markdown_table(lines):
    rows = []
    for line in lines:
        line = line.strip()
        if not line.startswith("|"):
            continue

        # Skip markdown separator row like |:---|---:|
        cells_raw = [c.strip() for c in line.strip("|").split("|")]
        if all(re.fullmatch(r":?-{2,}:?", c.replace(" ", "")) for c in cells_raw):
            continue

        rows.append(cells_raw)

    if not rows:
        return None

    max_len = max(len(r) for r in rows)
    rows = [r + [""] * (max_len - len(r)) for r in rows]
    return rows


def make_table(table_lines, available_width):
    rows = parse_markdown_table(table_lines)
    if not rows:
        return []

    # Limit extreme table width by wrapping cells.
    processed = []
    for r_i, row in enumerate(rows):
        processed_row = []
        for cell in row:
            cell = escape_xml(cell)
            style = styles["TableHeader"] if r_i == 0 else styles["TableCell"]
            processed_row.append(Paragraph(cell, style))
        processed.append(processed_row)

    ncols = len(processed[0])

    # Wide tables: distribute width evenly.
    # Very wide tables use smaller cells but remain readable in landscape PDF.
    col_width = available_width / max(ncols, 1)
    col_widths = [col_width] * ncols

    table = Table(processed, colWidths=col_widths, repeatRows=1)

    table.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#303030")),
        ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
        ("GRID", (0, 0), (-1, -1), 0.25, colors.HexColor("#BBBBBB")),
        ("VALIGN", (0, 0), (-1, -1), "TOP"),
        ("LEFTPADDING", (0, 0), (-1, -1), 2),
        ("RIGHTPADDING", (0, 0), (-1, -1), 2),
        ("TOPPADDING", (0, 0), (-1, -1), 2),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 2),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#F7F7F7")]),
    ]))

    return [Spacer(1, 4), table, Spacer(1, 8)]


def build_story(md_text):
    story = []
    lines = md_text.splitlines()
    i = 0

    page_width, page_height = landscape(A4)
    available_width = page_width - 2.0 * cm

    in_code = False
    code_lines = []

    while i < len(lines):
        line = lines[i]

        if line.strip().startswith("```"):
            if not in_code:
                in_code = True
                code_lines = []
            else:
                in_code = False
                code_text = "\n".join(code_lines)
                if code_text.strip():
                    story.append(Preformatted(code_text, styles["CodeBlock"]))
                    story.append(Spacer(1, 6))
            i += 1
            continue

        if in_code:
            code_lines.append(line)
            i += 1
            continue

        # Markdown table block
        if line.strip().startswith("|"):
            table_lines = []
            while i < len(lines) and lines[i].strip().startswith("|"):
                table_lines.append(lines[i])
                i += 1
            story.extend(make_table(table_lines, available_width))
            continue

        stripped = line.strip()

        if not stripped:
            i += 1
            continue

        if stripped.startswith("# "):
            title = stripped[2:].strip()
            if not story:
                story.append(Paragraph(escape_xml(title), styles["ReportTitle"]))
            else:
                story.append(PageBreak())
                story.append(Paragraph(escape_xml(title), styles["SectionHeading"]))
            i += 1
            continue

        if stripped.startswith("## "):
            story.append(Paragraph(escape_xml(stripped[3:].strip()), styles["SubsectionHeading"]))
            i += 1
            continue

        if stripped.startswith("- "):
            bullet_lines = []
            while i < len(lines) and lines[i].strip().startswith("- "):
                bullet_lines.append(lines[i].strip()[2:])
                i += 1
            for b in bullet_lines:
                story.append(Paragraph("• " + escape_xml(b), styles["BodyTextSmall"]))
            story.append(Spacer(1, 4))
            continue

        # Merge paragraph lines until blank/table/heading/code.
        para_lines = [stripped]
        i += 1
        while i < len(lines):
            nxt = lines[i].strip()
            if (
                not nxt
                or nxt.startswith("#")
                or nxt.startswith("|")
                or nxt.startswith("```")
                or nxt.startswith("- ")
            ):
                break
            para_lines.append(nxt)
            i += 1

        para = " ".join(para_lines)
        story.append(Paragraph(escape_xml(para), styles["BodyTextSmall"]))

    return story


doc = SimpleDocTemplate(
    str(PDF_PATH),
    pagesize=landscape(A4),
    rightMargin=1.0 * cm,
    leftMargin=1.0 * cm,
    topMargin=1.0 * cm,
    bottomMargin=1.0 * cm,
    title="Week 4-5 Behaviour Representation and Feature Engineering Report",
    author="Orhun Yavuz",
)

story = build_story(text)
doc.build(story)

print("Saved PDF:", PDF_PATH)
print("PDF size MB:", round(PDF_PATH.stat().st_size / (1024 * 1024), 2))
