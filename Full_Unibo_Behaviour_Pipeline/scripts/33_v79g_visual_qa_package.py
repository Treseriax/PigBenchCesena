from pathlib import Path
from datetime import datetime
import csv
import shutil
import pandas as pd

ROOT = Path.home() / "PigBench"
FULL = ROOT / "Full_Unibo_Behaviour_Pipeline"

FUSION = FULL / "outputs/v79f_track_to_clip_fusion/v79f_clip_track_fusion_table.csv"
TARGET_SUMMARY_IN = FULL / "outputs/v79f_track_to_clip_fusion/v79f_target_track_coverage.csv"
CLASS_SUMMARY_IN = FULL / "outputs/v79f_track_to_clip_fusion/v79f_behaviour_class_track_coverage.csv"
VIDEO_SUMMARY_IN = FULL / "outputs/v79e_full_36_video_sampled_tracking_run/v79e_video_tracking_summary.csv"
TRACKS_IN = FULL / "outputs/v79e_full_36_video_sampled_tracking_run/v79e_all_sampled_tracks.csv"
ANNOT_SRC = FULL / "outputs/v79e_full_36_video_sampled_tracking_run/annotated_sample_frames"

OUT = FULL / "outputs/v79g_visual_qa_package"
PKG = OUT / "Full_Unibo_Week9_Visual_QA_Package"
IMG_OUT = PKG / "annotated_sample_frames"
OUT.mkdir(parents=True, exist_ok=True)
PKG.mkdir(parents=True, exist_ok=True)
IMG_OUT.mkdir(parents=True, exist_ok=True)

QA_CLIPS = PKG / "v79g_visual_qa_clip_samples.csv"
QA_VIDEOS = PKG / "v79g_visual_qa_video_samples.csv"
QA_TARGETS = PKG / "v79g_visual_qa_target_summary.csv"
QA_CLASSES = PKG / "v79g_visual_qa_class_summary.csv"
QA_HTML = PKG / "v79g_visual_qa_index.html"
DECISION = OUT / "v79g_decision_summary.csv"
ISSUES = OUT / "v79g_issues.csv"
NOTE = FULL / "notes/v79g_visual_qa_package_notes.md"
NOTE.parent.mkdir(parents=True, exist_ok=True)

def clean(x):
    if pd.isna(x):
        return ""
    s = str(x).strip()
    if s.lower() in {"nan", "none", "null"}:
        return ""
    return s

def read_csv(path):
    df = pd.read_csv(path).fillna("")
    for c in df.columns:
        if df[c].dtype == object:
            df[c] = df[c].map(clean)
    return df

def write(df, path):
    df.to_csv(path, index=False, quoting=csv.QUOTE_ALL, escapechar="\\", lineterminator="\n")

issues = []

fusion = read_csv(FUSION)
target_summary = read_csv(TARGET_SUMMARY_IN)
class_summary = read_csv(CLASS_SUMMARY_IN)
video_summary = read_csv(VIDEO_SUMMARY_IN)
tracks = read_csv(TRACKS_IN)

if len(fusion) == 0:
    issues.append({"item":"fusion","issue_type":"hard_empty_fusion_table","severity":"hard","detail":str(FUSION)})

if len(video_summary) == 0:
    issues.append({"item":"video_summary","issue_type":"hard_empty_video_summary","severity":"hard","detail":str(VIDEO_SUMMARY_IN)})

if not ANNOT_SRC.exists():
    issues.append({"item":"annotated_frames","issue_type":"hard_missing_annotated_frame_dir","severity":"hard","detail":str(ANNOT_SRC)})

# Select representative clips:
# 1 sample per behaviour class, then per target, then high-candidate examples.
sample_rows = []

if len(fusion):
    fusion["candidate_tracklet_count_num"] = pd.to_numeric(fusion["candidate_tracklet_count"], errors="coerce").fillna(0)

    for label, g in fusion.groupby("behaviour_label"):
        g = g.sort_values(["candidate_tracklet_count_num", "clip_id"], ascending=[False, True])
        r = g.iloc[0].to_dict()
        r["sample_reason"] = f"behaviour_class_sample_{label}"
        sample_rows.append(r)

    for key, g in fusion.groupby(["date", "tlc_camera", "room_pen"]):
        g = g.sort_values(["candidate_tracklet_count_num", "clip_id"], ascending=[False, True])
        r = g.iloc[0].to_dict()
        r["sample_reason"] = f"target_sample_{key[1]}_{key[2]}"
        sample_rows.append(r)

    high = fusion.sort_values(["candidate_tracklet_count_num", "clip_id"], ascending=[False, True]).head(20)
    for _, row in high.iterrows():
        r = row.to_dict()
        r["sample_reason"] = "high_candidate_count_sample"
        sample_rows.append(r)

qa_clips = pd.DataFrame(sample_rows).drop_duplicates(subset=["clip_id"]) if sample_rows else pd.DataFrame()
write(qa_clips, QA_CLIPS)

# Copy annotated frames.
copied = []
if ANNOT_SRC.exists():
    imgs = sorted([p for p in ANNOT_SRC.rglob("*.jpg")])
    for p in imgs:
        dst = IMG_OUT / p.name
        if not dst.exists():
            shutil.copy2(p, dst)
        copied.append({
            "image_name": p.name,
            "source_path": str(p),
            "package_path": str(dst),
        })

img_df = pd.DataFrame(copied)

# Video QA summary.
if len(video_summary):
    video_summary["detections_num"] = pd.to_numeric(video_summary["detections"], errors="coerce").fillna(0)
    video_summary["unique_tracklets_num"] = pd.to_numeric(video_summary["unique_tracklets"], errors="coerce").fillna(0)
    qa_videos = video_summary.sort_values(["detections_num", "unique_tracklets_num"], ascending=[False, False])
else:
    qa_videos = pd.DataFrame()

write(qa_videos, QA_VIDEOS)
write(target_summary, QA_TARGETS)
write(class_summary, QA_CLASSES)

# HTML report.
html = []
html.append("<html><head><meta charset='utf-8'>")
html.append("<title>v79g Week9 Visual QA Package</title>")
html.append("<style>body{font-family:Arial;margin:24px} table{border-collapse:collapse} td,th{border:1px solid #ccc;padding:4px;font-size:12px} img{width:330px;border:1px solid #999;margin:5px}</style>")
html.append("</head><body>")
html.append("<h1>v79g Week9 Visual QA Package</h1>")
html.append("<h2>Scope</h2>")
html.append("<p>This package reviews sampled annotation-window tracking outputs. It does not claim final colour-identity track assignment.</p>")

html.append("<h2>Core counts</h2>")
html.append("<ul>")
html.append(f"<li>Fusion rows: {len(fusion)}</li>")
html.append(f"<li>QA clip samples: {len(qa_clips)}</li>")
html.append(f"<li>Videos in summary: {len(video_summary)}</li>")
html.append(f"<li>Annotated sample frames copied: {len(img_df)}</li>")
html.append(f"<li>Track rows: {len(tracks)}</li>")
html.append("</ul>")

html.append("<h2>Target coverage</h2>")
html.append(target_summary.to_html(index=False, escape=False) if len(target_summary) else "<p>empty</p>")

html.append("<h2>Behaviour class coverage</h2>")
html.append(class_summary.to_html(index=False, escape=False) if len(class_summary) else "<p>empty</p>")

html.append("<h2>QA clip samples</h2>")
cols = [c for c in ["sample_reason","clip_id","video_id","tlc_camera","room_pen","identity_colour","behaviour_label","candidate_tracklet_count","top_candidate_tracklet","fusion_status"] if c in qa_clips.columns]
html.append(qa_clips[cols].to_html(index=False, escape=False) if len(qa_clips) else "<p>empty</p>")

html.append("<h2>Annotated sample frames</h2>")
for _, r in img_df.head(180).iterrows():
    rel = "annotated_sample_frames/" + clean(r["image_name"])
    html.append(f"<div style='display:inline-block;vertical-align:top'><img src='{rel}'><br><small>{clean(r['image_name'])}</small></div>")

html.append("</body></html>")
QA_HTML.write_text("\n".join(html), encoding="utf-8")

if len(img_df) == 0:
    issues.append({"item":"annotated_frames","issue_type":"warning_no_annotated_frames_copied","severity":"warning","detail":"No jpg frames copied."})

issues.append({
    "item":"identity_assignment",
    "issue_type":"info_no_colour_identity_assignment_yet",
    "severity":"info",
    "detail":"v79g is visual QA for candidate tracks, not final colour identity assignment."
})

issues.append({
    "item":"scope",
    "issue_type":"info_visual_qa_package_only",
    "severity":"info",
    "detail":"v79g packages QA tables and annotated frames. It does not run new tracking."
})

issues_df = pd.DataFrame(issues)
write(issues_df, ISSUES)

hard_count = int((issues_df["severity"] == "hard").sum()) if "severity" in issues_df.columns else 0
warning_count = int((issues_df["severity"] == "warning").sum()) if "severity" in issues_df.columns else 0

decision = pd.DataFrame([{
    "v79g_decision": "visual_qa_package_created" if hard_count == 0 else "visual_qa_package_has_blocking_issues",
    "fusion_rows": len(fusion),
    "qa_clip_samples": len(qa_clips),
    "video_summary_rows": len(video_summary),
    "track_rows": len(tracks),
    "annotated_frames_copied": len(img_df),
    "html_index": str(QA_HTML),
    "package_dir": str(PKG),
    "hard_issue_count": hard_count,
    "warning_count": warning_count,
    "ready_for_week9_final_package": bool(hard_count == 0),
    "ready_for_final_identity_tracking_claim": False,
    "claim_scope": "visual_qa_package_only",
    "generated_at": datetime.now().isoformat(timespec="seconds"),
}])
write(decision, DECISION)

NOTE.write_text(
    "# v79g Visual QA Package\n\n"
    f"- Decision: {decision.iloc[0]['v79g_decision']}\n"
    f"- Fusion rows: {len(fusion)}\n"
    f"- QA clip samples: {len(qa_clips)}\n"
    f"- Video summary rows: {len(video_summary)}\n"
    f"- Track rows: {len(tracks)}\n"
    f"- Annotated frames copied: {len(img_df)}\n"
    f"- Hard issues: {hard_count}\n"
    f"- Ready for Week 9 final package: {bool(hard_count == 0)}\n\n"
    "This stage creates visual QA material for sampled tracking. It does not assign colour identity to tracklets.\n",
    encoding="utf-8"
)

print("=== v79g decision ===")
print(decision.to_string(index=False))
print("\n=== qa clips sample ===")
print(qa_clips.head(40).to_string(index=False) if len(qa_clips) else "empty")
print("\n=== issues ===")
print(issues_df.to_string(index=False))
