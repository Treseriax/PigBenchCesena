from pathlib import Path
import pandas as pd


SUMMARY_CSV = Path("Week3_Behaviour_Dataset/outputs/scan_window_aligned/identity_mapping/all_segments_track_id_summary.csv")
LABEL_CSV = Path("Week3_Behaviour_Dataset/outputs/scan_window_aligned/behaviour_labels/scan_window_behaviour_labels.csv")

OUT_DIR = Path("Week3_Behaviour_Dataset/outputs/scan_window_aligned/identity_mapping")
OUT_DIR.mkdir(parents=True, exist_ok=True)

summary = pd.read_csv(SUMMARY_CSV)
labels = pd.read_csv(LABEL_CSV)

rows = []

for seg, seg_df in summary.groupby("segment_id"):
    seg_df = seg_df.sort_values(["num_frames", "mean_score"], ascending=[False, False]).copy()

    # Keep dominant IDs only:
    # - track visible for at least 30 frames, or
    # - among top 12 longest tracks of that segment.
    dominant = seg_df[
        (seg_df["num_frames"] >= 30)
    ].head(12)

    label_text = labels[labels["segment_id"] == seg].copy()

    available_labels = "; ".join(
        f"{r['colour']}={r['behaviour_code']}({r['behaviour_label']})"
        for _, r in label_text.iterrows()
    )

    for _, r in dominant.iterrows():
        rows.append({
            "segment_id": seg,
            "track_id": int(r["track_id"]),
            "first_frame": int(r["first_frame"]),
            "last_frame": int(r["last_frame"]),
            "num_frames": int(r["num_frames"]),
            "mean_score": round(float(r["mean_score"]), 4),
            "mean_cx": round(float(r["mean_cx"]), 2),
            "mean_cy": round(float(r["mean_cy"]), 2),
            "excel_interval_start": r["excel_interval_start"],
            "excel_interval_end": r["excel_interval_end"],
            "available_excel_labels": available_labels,
            "assigned_colour": "",
            "assigned_pig_id": "",
            "assigned_behaviour_code": "",
            "assigned_behaviour_label": "",
            "identity_confidence": "",
            "use_in_final_json": "",
            "notes": ""
        })

dominant_df = pd.DataFrame(rows)

out_csv = OUT_DIR / "dominant_track_id_to_colour_mapping_table.csv"
dominant_df.to_csv(out_csv, index=False)

print("Saved:", out_csv)
print()
print("=== Dominant track mapping table preview ===")
print(dominant_df.to_string(index=False))
