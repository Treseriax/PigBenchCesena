from pathlib import Path
import pandas as pd
import shutil


ROOT = Path("Week3_Behaviour_Dataset")
FINAL = ROOT / "final_outputs_corrected_scan_windows"
NOTES = FINAL / "notes"
NOTES.mkdir(parents=True, exist_ok=True)

summary_csv = FINAL / "csv/corrected_scan_window_json_summary.csv"
tracking_csv = FINAL / "csv/scan_window_tracking_with_time_summary.csv"
labels_csv = FINAL / "csv/scan_window_behaviour_label_summary.csv"
mapping_csv = FINAL / "csv/dominant_track_id_to_colour_mapping_table.csv"

json_summary = pd.read_csv(summary_csv)
tracking_summary = pd.read_csv(tracking_csv)
label_summary = pd.read_csv(labels_csv)
mapping_table = pd.read_csv(mapping_csv)


def df_to_markdown(df):
    if df.empty:
        return "_No data available._"

    cols = list(df.columns)
    rows = []

    rows.append("| " + " | ".join(cols) + " |")
    rows.append("| " + " | ".join(["---"] * len(cols)) + " |")

    for _, row in df.iterrows():
        values = []
        for col in cols:
            value = str(row[col])
            value = value.replace("\n", " ").replace("|", "/")
            values.append(value)
        rows.append("| " + " | ".join(values) + " |")

    return "\n".join(rows)


total_track_instances = int(json_summary["track_instances"].sum())
total_frames = int(json_summary["frames"].sum())
total_segments = int(json_summary["segment_id"].nunique())
total_unique_ids = int(tracking_summary["unique_track_ids"].sum())

report_path = NOTES / "corrected_week3_report.md"

with open(report_path, "w") as f:
    f.write("# Week 3 Corrected Report — Behaviour Dataset Construction and Visualization\n\n")

    f.write("## 1. Purpose of the correction\n\n")
    f.write(
        "The first Week 3 version successfully built the behaviour annotation structure, "
        "but the original video sample did not exactly overlap with the Excel scan-sampling observation windows. "
        "To fix this limitation, a new manually concatenated screen-recording video was prepared. "
        "This video contains the relevant scan-window intervals at 09:00, 09:10, 09:20, 09:30, 09:40, and 09:50.\n\n"
    )

    f.write(
        "The corrected version therefore replaces the previous no-overlap situation with a segment-level alignment strategy. "
        "Each video segment is mapped to its corresponding Excel scan-sampling window.\n\n"
    )

    f.write("## 2. Methodological note\n\n")
    f.write(
        "The corrected video is not a continuous camera recording. It is a manually concatenated screen recording "
        "of selected observation windows. For this reason, the pipeline does not treat video elapsed time as continuous real camera time. "
        "Instead, each segment is manually mapped to its Excel observation interval.\n\n"
    )

    f.write("This is the corrected time mapping used in the dataset:\n\n")
    f.write("- `scan_09_00` → `09:00:00–09:00:10`\n")
    f.write("- `scan_09_10` → `09:10:00–09:10:10`\n")
    f.write("- `scan_09_20` → `09:20:00–09:20:10`\n")
    f.write("- `scan_09_30` → `09:30:00–09:30:10`\n")
    f.write("- `scan_09_40` → `09:40:00–09:40:10`\n")
    f.write("- `scan_09_50` → `09:50:00–09:50:10`\n\n")

    f.write("## 3. Corrected processing pipeline\n\n")
    f.write("The corrected Week 3 pipeline follows these steps:\n\n")
    f.write("1. Check the clean scan-window video metadata and sample frames.\n")
    f.write("2. Create a manual segment-to-Excel-window mapping.\n")
    f.write("3. Extract each scan-window segment into a separate image sequence.\n")
    f.write("4. Run YOLOv8-s detection on each segment.\n")
    f.write("5. Run ByteTrack tracking on each segment.\n")
    f.write("6. Attach segment ID and Excel interval information to each track row.\n")
    f.write("7. Parse the Excel behaviour labels for each scan window.\n")
    f.write("8. Build the corrected JSON annotation dataset.\n")
    f.write("9. Generate visualization videos showing track IDs, segment windows, and behaviour candidates.\n")
    f.write("10. Package corrected CSV, JSON, figures, videos, scripts, and notes into a final output folder.\n\n")

    f.write("## 4. Corrected JSON dataset summary\n\n")
    f.write(
        f"The corrected JSON dataset contains {total_segments} aligned scan-window segments, "
        f"{total_frames} processed frames, and {total_track_instances} track instances.\n\n"
    )
    f.write(df_to_markdown(json_summary))
    f.write("\n\n")

    f.write("## 5. Tracking summary\n\n")
    f.write(
        "ByteTrack was applied separately to each extracted scan-window sequence. "
        "The number of unique track IDs is higher than the number of pigs in some segments because of tracking fragmentation, "
        "partial visibility, neighbouring pen detections, and screen-recording artefacts.\n\n"
    )
    f.write(df_to_markdown(tracking_summary))
    f.write("\n\n")

    f.write("## 6. Behaviour label summary from Excel\n\n")
    f.write(
        "The Excel annotation file provides scan-sampling behaviour labels for colour-coded pig identities. "
        "These labels were successfully attached at the segment level.\n\n"
    )
    f.write(df_to_markdown(label_summary))
    f.write("\n\n")

    f.write("## 7. Identity mapping policy\n\n")
    f.write(
        "The tracker produces numerical IDs such as `track_id=1`, `track_id=2`, and so on. "
        "However, the Excel file provides labels for colour-coded identities such as green, blue, purple, red neck, red tail, and no color. "
        "Therefore, a track-to-colour mapping step is needed before assigning a behaviour label to an individual track.\n\n"
    )

    f.write(
        "This corrected version keeps individual track identities unverified unless a human manually confirms the colour identity. "
        "This conservative approach prevents false behaviour assignment when colour markers are unclear or when track fragmentation occurs.\n\n"
    )

    f.write("Current status:\n\n")
    f.write("- Segment-level behaviour labels are available.\n")
    f.write("- Track-level bounding boxes and track IDs are available.\n")
    f.write("- Track-to-colour identity mapping is prepared as a manual verification table.\n")
    f.write("- Individual track behaviour labels are not forced unless the identity mapping is manually verified.\n\n")

    f.write("## 8. Final corrected outputs\n\n")
    f.write("The corrected final output package contains:\n\n")
    f.write("- Corrected JSON annotation file\n")
    f.write("- Segment-level behaviour label JSON and CSV files\n")
    f.write("- Tracking CSV files with Excel window information\n")
    f.write("- Track ID summary and manual identity mapping template\n")
    f.write("- Track ID montage figures for all scan windows\n")
    f.write("- Corrected visualization videos for all scan windows\n")
    f.write("- Python scripts used in the corrected pipeline\n")
    f.write("- Summary notes and this corrected report\n\n")

    f.write("## 9. Remaining limitation and future work\n\n")
    f.write(
        "The main previous limitation, no exact overlap between video and Excel scan windows, has been fixed. "
        "The remaining limitation is identity verification: individual track IDs still require manual confirmation before they can be safely linked to colour-coded pig identities. "
        "A future improvement would be to manually complete the track-to-colour mapping table for high-confidence tracks, or to train a colour-marker/identity recognition module.\n\n"
    )

    f.write("## 10. Conclusion\n\n")
    f.write(
        "The corrected Week 3 dataset now provides a stronger behaviour annotation foundation. "
        "It links video frames, detected/tracked pig instances, scan-window timing, and Excel behaviour labels in a structured JSON format. "
        "The dataset is intentionally conservative at the individual identity level to avoid incorrect behaviour labels, while still preserving all information needed for future manual verification and downstream behaviour analysis.\n"
    )

print("Saved corrected report:", report_path)

# Also fix the earlier summary note typo by copying report as main note reference
short_summary = NOTES / "corrected_week3_summary.md"
with open(short_summary, "w") as f:
    f.write("# Week 3 Corrected Scan-Window Behaviour Dataset\n\n")
    f.write("The corrected final package was generated successfully.\n\n")
    f.write("Main correction: the previous no-overlap limitation was addressed by using a manually concatenated scan-window video containing the relevant observation intervals.\n\n")
    f.write("Important note: the corrected video is not continuous camera time. Therefore, timestamp alignment is performed using manual segment-to-Excel-window mapping.\n\n")
    f.write("See `corrected_week3_report.md` for the full corrected report.\n")

print("Updated short summary:", short_summary)

# Refresh zip
zip_base = ROOT / "Week3_corrected_scan_window_final_outputs"
zip_path = shutil.make_archive(str(zip_base), "zip", FINAL)
print("Updated ZIP:", zip_path)
