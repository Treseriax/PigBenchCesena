from pathlib import Path
import csv
import pandas as pd


ROOT = Path.home() / "PigBench"
W7 = ROOT / "Week7_Dataset_Validation_Identity_Tracking_Behaviour_Preparation"

IN_FRAME_RULES = W7 / "outputs" / "colour_identity" / "colour_rule_workspace_v13" / "week7_colour_rule_workspace_v13_frame_colour_rule_template.csv"

OUT_ROOT = W7 / "outputs" / "colour_identity" / "colour_rule_workspace_v13"
OUT_VIDEO_RULES = OUT_ROOT / "week7_colour_rule_workspace_v13b_video_level_rule_template.csv"
OUT_PROPAGATED_PREVIEW = OUT_ROOT / "week7_colour_rule_workspace_v13b_frame_rules_propagated_preview.csv"
OUT_NOTE = W7 / "notes" / "week7_colour_rule_workspace_v13b_video_level_rules_notes.md"


def safe_to_csv(df, path):
    df.to_csv(
        path,
        index=False,
        quoting=csv.QUOTE_ALL,
        escapechar="\\",
        lineterminator="\n",
    )


colour_cols = [f"allowed_colour_{i}" for i in range(1, 7)]

df = pd.read_csv(IN_FRAME_RULES)

video_rows = []

for video_id, g in df.groupby("video_id", sort=False):
    all_colours = []

    for c in colour_cols:
        if c in g.columns:
            all_colours.extend([str(x).strip() for x in g[c].dropna().tolist() if str(x).strip()])

    counts = pd.Series(all_colours).value_counts()

    suggested = counts.index.tolist()

    # İlk 6 rengi öneri olarak koyuyoruz ama manuel verify şart.
    row = {
        "video_id": video_id,
        "frame_count": len(g),
        "scan_frame_ids": " | ".join(g["scan_frame_id"].astype(str).tolist()),
        "suggested_colour_frequency": " | ".join([f"{k}:{v}" for k, v in counts.items()]),
        "video_rule_colour_1": suggested[0] if len(suggested) > 0 else "",
        "video_rule_colour_2": suggested[1] if len(suggested) > 1 else "",
        "video_rule_colour_3": suggested[2] if len(suggested) > 2 else "",
        "video_rule_colour_4": suggested[3] if len(suggested) > 3 else "",
        "video_rule_colour_5": suggested[4] if len(suggested) > 4 else "",
        "video_rule_colour_6": suggested[5] if len(suggested) > 5 else "",
        "manual_rule_status": "needs_manual_verification",
        "manual_notes": "",
    }

    video_rows.append(row)

video_rules = pd.DataFrame(video_rows)
safe_to_csv(video_rules, OUT_VIDEO_RULES)

# Preview: video-level suggestionları frame-level’e yayılmış gibi göster.
preview = df.copy()

for _, vr in video_rules.iterrows():
    mask = preview["video_id"].astype(str) == str(vr["video_id"])

    for i in range(1, 7):
        preview.loc[mask, f"allowed_colour_{i}"] = vr[f"video_rule_colour_{i}"]

    preview.loc[mask, "rule_status"] = "video_level_rule_preview_needs_manual_verification"
    preview.loc[mask, "rule_confidence"] = "medium"
    preview.loc[mask, "manual_notes"] = "Preview propagated from video-level colour rule template. Verify before final assignment."

safe_to_csv(preview, OUT_PROPAGATED_PREVIEW)

OUT_NOTE.write_text(
    "# Week 7 Video-Level Colour Rules v13b\n\n"
    "## Purpose\n\n"
    "Frame-level colour rules were incomplete for many frames. "
    "This step creates a compact video-level colour rule template so each video/pen session can be verified once and propagated to its scanpoint frames.\n\n"
    "## Main outputs\n\n"
    f"- Video-level rule template: `{OUT_VIDEO_RULES}`\n"
    f"- Propagated preview: `{OUT_PROPAGATED_PREVIEW}`\n\n"
    "## Manual step\n\n"
    "Open the video-level rule template and verify `video_rule_colour_1..6` for each video_id. "
    "After verification, the confirmed rules will be propagated to the frame-level rule table for colour identity assignment.\n"
)

print("Saved:")
print(OUT_VIDEO_RULES)
print(OUT_PROPAGATED_PREVIEW)
print(OUT_NOTE)

print()
print("=== video-level rule template summary ===")
print({
    "video_rule_rows": len(video_rules),
    "frame_rows_in_preview": len(preview),
})
print()
print(video_rules[[
    "video_id",
    "frame_count",
    "suggested_colour_frequency",
    "video_rule_colour_1",
    "video_rule_colour_2",
    "video_rule_colour_3",
    "video_rule_colour_4",
    "video_rule_colour_5",
    "video_rule_colour_6",
]].to_string(index=False))
