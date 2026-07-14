from pathlib import Path
import csv
import cv2
import numpy as np
import pandas as pd


PROJECT_ROOT = Path.home() / "PigBench"
W6 = PROJECT_ROOT / "Week6_Unibo_Dataset_Validation"

OUT_STATS = W6 / "outputs/dataset_statistics"
OUT_VIS = W6 / "outputs/visual_label_check"
NOTES = W6 / "notes"

LOW = OUT_STATS / "week6_integrity_low_bbox_frames.csv"
HIGH = OUT_STATS / "week6_integrity_high_bbox_frames.csv"
OVERLAY_SUMMARY = OUT_VIS / "marker_bbox_overlay_v3_summary.csv"
DETECTION_SUMMARY = OUT_STATS / "week6_final_detection_frame_summary.csv"

QC_DIR = OUT_VIS / "bbox_count_warning_qc"
QC_DIR.mkdir(parents=True, exist_ok=True)
NOTES.mkdir(parents=True, exist_ok=True)


def safe_to_csv(df, path):
    df.to_csv(
        path,
        index=False,
        quoting=csv.QUOTE_ALL,
        escapechar="\\",
        lineterminator="\n"
    )


def read_optional(path):
    if path.exists():
        return pd.read_csv(path)
    return pd.DataFrame()


def add_header(img, lines, width=None):
    if img is None:
        img = np.full((320, 480, 3), 245, dtype=np.uint8)

    if width is not None:
        h, w = img.shape[:2]
        scale = width / w
        img = cv2.resize(img, (width, int(h * scale)), interpolation=cv2.INTER_AREA)

    h, w = img.shape[:2]
    header_h = 92
    header = np.full((header_h, w, 3), 255, dtype=np.uint8)

    y = 22
    for line in lines:
        cv2.putText(
            header,
            str(line)[:95],
            (8, y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.48,
            (0, 0, 0),
            1,
            cv2.LINE_AA,
        )
        y += 22

    return np.vstack([header, img])


def make_contact_sheet(rows, out_path, title, thumb_width=620, ncols=2):
    tiles = []

    for _, r in rows.iterrows():
        overlay_path = Path(str(r.get("output_path", "")))
        img = cv2.imread(str(overlay_path)) if overlay_path.exists() else None

        lines = [
            f"{r.get('scan_frame_id', '')} | bbox={r.get('bbox_count', '')} | labels={r.get('manual_label_count', '')}",
            f"{r.get('timestamp', '')} | {r.get('video_match_status', '')}",
            f"{r.get('qc_issue_type', '')}: {r.get('qc_hint', '')}",
        ]

        tile = add_header(img, lines, width=thumb_width)
        tiles.append(tile)

    if not tiles:
        blank = np.full((300, thumb_width * ncols, 3), 255, dtype=np.uint8)
        cv2.putText(blank, f"No frames for {title}", (20, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 2)
        cv2.imwrite(str(out_path), blank)
        return

    max_h = max(t.shape[0] for t in tiles)
    max_w = max(t.shape[1] for t in tiles)

    padded = []

    for t in tiles:
        pad_bottom = max_h - t.shape[0]
        pad_right = max_w - t.shape[1]

        if pad_bottom or pad_right:
            t = cv2.copyMakeBorder(
                t,
                0,
                pad_bottom,
                0,
                pad_right,
                cv2.BORDER_CONSTANT,
                value=(255, 255, 255),
            )

        padded.append(t)

    rows_imgs = []

    for i in range(0, len(padded), ncols):
        row_tiles = padded[i:i+ncols]

        while len(row_tiles) < ncols:
            row_tiles.append(np.full_like(padded[0], 255))

        rows_imgs.append(np.hstack(row_tiles))

    sheet = np.vstack(rows_imgs)

    title_bar = np.full((60, sheet.shape[1], 3), 255, dtype=np.uint8)
    cv2.putText(
        title_bar,
        title,
        (20, 38),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.9,
        (0, 0, 0),
        2,
        cv2.LINE_AA,
    )

    sheet = np.vstack([title_bar, sheet])
    cv2.imwrite(str(out_path), sheet)


low = read_optional(LOW)
high = read_optional(HIGH)
overlay = read_optional(OVERLAY_SUMMARY)
det_summary = read_optional(DETECTION_SUMMARY)

if overlay.empty:
    raise FileNotFoundError(OVERLAY_SUMMARY)

low["qc_issue_type"] = "low_bbox_count"
low["qc_hint"] = "possible occlusion, missed pigs, frame quality, or strict detector threshold"

high["qc_issue_type"] = "high_bbox_count"
high["qc_hint"] = "possible duplicate detections, false positives, or overlapping pig detections"

qc = pd.concat([low, high], ignore_index=True)

# Merge with overlay summary to get output images, timestamp, status.
qc = qc.merge(
    overlay,
    on="scan_frame_id",
    how="left",
    suffixes=("_audit", "")
)

# Merge detection stats if available.
if not det_summary.empty:
    qc = qc.merge(
        det_summary,
        on="scan_frame_id",
        how="left",
        suffixes=("", "_detstat")
    )

# Keep useful columns.
preferred_cols = [
    "qc_issue_type",
    "scan_frame_id",
    "bbox_count",
    "manual_label_count",
    "timestamp",
    "video_id",
    "video_match_status",
    "high_marker_candidates",
    "medium_marker_candidates",
    "low_marker_candidates",
    "no_marker",
    "mean_score",
    "max_score",
    "mean_bbox_area",
    "output_path",
    "qc_hint",
]

qc = qc[[c for c in preferred_cols if c in qc.columns]]

qc_path = QC_DIR / "bbox_count_warning_qc_table.csv"
safe_to_csv(qc, qc_path)

low_qc = qc[qc["qc_issue_type"] == "low_bbox_count"].copy()
high_qc = qc[qc["qc_issue_type"] == "high_bbox_count"].copy()

low_sheet = QC_DIR / "low_bbox_count_frames_contact_sheet.jpg"
high_sheet = QC_DIR / "high_bbox_count_frames_contact_sheet.jpg"
combined_sheet = QC_DIR / "all_bbox_count_warning_frames_contact_sheet.jpg"

make_contact_sheet(low_qc, low_sheet, "Low bbox-count frames QC: fewer than 6 detections", thumb_width=620, ncols=2)
make_contact_sheet(high_qc, high_sheet, "High bbox-count frames QC: more than 10 detections", thumb_width=620, ncols=2)
make_contact_sheet(qc, combined_sheet, "All bbox-count warning frames QC", thumb_width=620, ncols=2)

# Summary.
summary_rows = []

for issue, g in qc.groupby("qc_issue_type"):
    summary_rows.append({
        "qc_issue_type": issue,
        "frame_count": len(g),
        "mean_bbox_count": round(g["bbox_count"].mean(), 3) if "bbox_count" in g.columns else "",
        "min_bbox_count": int(g["bbox_count"].min()) if "bbox_count" in g.columns and len(g) else "",
        "max_bbox_count": int(g["bbox_count"].max()) if "bbox_count" in g.columns and len(g) else "",
        "direct_tlc_frames": int((g["video_match_status"] == "matched_tlc_hour_video").sum()) if "video_match_status" in g.columns else "",
        "candidate_ctoken_frames": int((g["video_match_status"] == "candidate_recovered_ctoken_video").sum()) if "video_match_status" in g.columns else "",
    })

summary = pd.DataFrame(summary_rows)
summary_path = QC_DIR / "bbox_count_warning_qc_summary.csv"
safe_to_csv(summary, summary_path)

note_path = NOTES / "week6_bbox_count_warning_qc_notes.md"

with open(note_path, "w") as f:
    f.write("# Week 6 Bbox Count Warning QC Notes\n\n")

    f.write("## Purpose\n\n")
    f.write(
        "This step creates visual QC contact sheets for frames flagged by the integrity audit because their detector bbox count is unusually low or high compared with the six manually labelled pigs.\n\n"
    )

    f.write("## Outputs\n\n")
    f.write(f"- QC table: `{qc_path}`\n")
    f.write(f"- Low bbox contact sheet: `{low_sheet}`\n")
    f.write(f"- High bbox contact sheet: `{high_sheet}`\n")
    f.write(f"- Combined contact sheet: `{combined_sheet}`\n")
    f.write(f"- Summary: `{summary_path}`\n\n")

    f.write("## QC summary\n\n")
    f.write(summary.to_markdown(index=False) if len(summary) else "No warning frames.")
    f.write("\n\n")

    f.write("## Warning frame table\n\n")
    f.write(qc.to_markdown(index=False) if len(qc) else "No warning frames.")
    f.write("\n\n")

    f.write("## Interpretation guide\n\n")
    f.write(
        "- Low bbox count can indicate occlusion, detector misses, frame quality issues, or too strict a threshold.\n"
        "- High bbox count can indicate duplicate detections, false positives, overlapping pigs, or detector boxes on body parts.\n"
        "- These frames do not invalidate the dataset because detector bboxes are automatic, but they should be documented as visual QC cases.\n"
        "- If many low/high cases are visually severe, we can rerun detection with adjusted thresholds or apply NMS/post-filtering.\n"
    )

print("Saved:")
print(qc_path)
print(summary_path)
print(low_sheet)
print(high_sheet)
print(combined_sheet)
print(note_path)

print()
print("=== Bbox warning QC summary ===")
print(summary.to_string(index=False) if len(summary) else "No warning frames.")

print()
print("=== Bbox warning QC table ===")
print(qc.to_string(index=False) if len(qc) else "No warning frames.")
