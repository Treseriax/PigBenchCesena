from pathlib import Path
from datetime import datetime
import csv
import pandas as pd

ROOT = Path.home() / "PigBench"
FULL = ROOT / "Full_Unibo_Behaviour_Pipeline"

CLIPS = FULL / "outputs/v79a_tracking_preparation_manifest/v79a_annotation_clip_manifest.csv"
TRACKS = FULL / "outputs/v79e_full_36_video_sampled_tracking_run/v79e_all_sampled_tracks.csv"
WINDOWS = FULL / "outputs/v79e_full_36_video_sampled_tracking_run/v79e_tracking_windows.csv"
VIDEO_SUMMARY_IN = FULL / "outputs/v79e_full_36_video_sampled_tracking_run/v79e_video_tracking_summary.csv"

OUT = FULL / "outputs/v79f_track_to_clip_fusion"
OUT.mkdir(parents=True, exist_ok=True)

CLIP_FUSION = OUT / "v79f_clip_track_fusion_table.csv"
TRACKLET_SUMMARY = OUT / "v79f_tracklet_summary.csv"
WINDOW_SUMMARY = OUT / "v79f_window_track_summary.csv"
CLASS_SUMMARY = OUT / "v79f_behaviour_class_track_coverage.csv"
TARGET_SUMMARY = OUT / "v79f_target_track_coverage.csv"
DECISION = OUT / "v79f_decision_summary.csv"
ISSUES = OUT / "v79f_issues.csv"
NOTE = FULL / "notes/v79f_track_to_clip_fusion_notes.md"
NOTE.parent.mkdir(parents=True, exist_ok=True)

def clean(x):
    if pd.isna(x):
        return ""
    s = str(x).strip()
    if s.lower() in {"nan", "none", "null"}:
        return ""
    return s

def write(df, path):
    df.to_csv(path, index=False, quoting=csv.QUOTE_ALL, escapechar="\\", lineterminator="\n")

def read_csv(path):
    df = pd.read_csv(path).fillna("")
    for c in df.columns:
        if df[c].dtype == object:
            df[c] = df[c].map(clean)
    return df

issues = []

clips = read_csv(CLIPS)
tracks = read_csv(TRACKS)
windows = read_csv(WINDOWS)
video_summary = read_csv(VIDEO_SUMMARY_IN)

required_clip_cols = [
    "clip_id", "video_id", "video_filename", "date", "tlc_camera", "room_pen",
    "identity_colour", "behaviour_label", "clip_start_sec_in_video", "clip_end_sec_in_video"
]
required_track_cols = [
    "tracklet_id", "video_id", "window_id", "time_sec", "frame_index",
    "score", "x1", "y1", "x2", "y2"
]
required_window_cols = [
    "window_id", "video_id", "window_start_sec", "window_end_sec", "clip_ids"
]

for c in required_clip_cols:
    if c not in clips.columns:
        issues.append({"item": c, "issue_type": "hard_missing_clip_column", "severity": "hard", "detail": c})

for c in required_track_cols:
    if c not in tracks.columns:
        issues.append({"item": c, "issue_type": "hard_missing_track_column", "severity": "hard", "detail": c})

for c in required_window_cols:
    if c not in windows.columns:
        issues.append({"item": c, "issue_type": "hard_missing_window_column", "severity": "hard", "detail": c})

if len(clips) == 0:
    issues.append({"item": "clips", "issue_type": "hard_empty_clip_manifest", "severity": "hard", "detail": str(CLIPS)})

if len(tracks) == 0:
    issues.append({"item": "tracks", "issue_type": "hard_empty_tracking_output", "severity": "hard", "detail": str(TRACKS)})

if len(windows) == 0:
    issues.append({"item": "windows", "issue_type": "hard_empty_tracking_windows", "severity": "hard", "detail": str(WINDOWS)})

if any(i["severity"] == "hard" for i in issues):
    issues_df = pd.DataFrame(issues)
    write(issues_df, ISSUES)
    print(issues_df.to_string(index=False))
    raise SystemExit(1)

for c in ["clip_start_sec_in_video", "clip_end_sec_in_video"]:
    clips[c] = pd.to_numeric(clips[c], errors="coerce")

for c in ["window_start_sec", "window_end_sec"]:
    windows[c] = pd.to_numeric(windows[c], errors="coerce")

for c in ["time_sec", "frame_index", "score", "x1", "y1", "x2", "y2"]:
    tracks[c] = pd.to_numeric(tracks[c], errors="coerce")

# Tracklet-level summary.
tracklet_summary = (
    tracks.groupby(["video_id", "window_id", "tracklet_id"], dropna=False)
    .agg(
        track_rows=("tracklet_id", "count"),
        first_time_sec=("time_sec", "min"),
        last_time_sec=("time_sec", "max"),
        first_frame=("frame_index", "min"),
        last_frame=("frame_index", "max"),
        mean_score=("score", "mean"),
        max_score=("score", "max"),
        min_x1=("x1", "min"),
        min_y1=("y1", "min"),
        max_x2=("x2", "max"),
        max_y2=("y2", "max"),
    )
    .reset_index()
)
tracklet_summary["duration_sec_observed"] = tracklet_summary["last_time_sec"] - tracklet_summary["first_time_sec"]
write(tracklet_summary, TRACKLET_SUMMARY)

# Window-level summary.
window_track_summary = (
    tracks.groupby(["video_id", "window_id"], dropna=False)
    .agg(
        track_rows=("tracklet_id", "count"),
        unique_tracklets=("tracklet_id", "nunique"),
        sampled_frames=("frame_index", "nunique"),
        mean_score=("score", "mean"),
        max_score=("score", "max"),
    )
    .reset_index()
)

window_summary = windows.merge(window_track_summary, on=["video_id", "window_id"], how="left").fillna("")
for c in ["track_rows", "unique_tracklets", "sampled_frames"]:
    if c in window_summary.columns:
        window_summary[c] = pd.to_numeric(window_summary[c], errors="coerce").fillna(0).astype(int)

write(window_summary, WINDOW_SUMMARY)

# Clip fusion: attach each clip to its matching tracking window and candidate tracklets.
fusion_rows = []

# Build lookup from clip_id to window.
clip_to_window = {}
for _, w in windows.iterrows():
    clip_ids = [clean(x) for x in clean(w["clip_ids"]).split(";") if clean(x)]
    for cid in clip_ids:
        clip_to_window[cid] = clean(w["window_id"])

tracklet_by_window = {}
for (video_id, window_id), g in tracklet_summary.groupby(["video_id", "window_id"]):
    g = g.sort_values(["track_rows", "max_score"], ascending=[False, False])
    tracklet_by_window[(clean(video_id), clean(window_id))] = g

for _, cl in clips.iterrows():
    clip_id = clean(cl["clip_id"])
    video_id = clean(cl["video_id"])
    window_id = clip_to_window.get(clip_id, "")

    cand = tracklet_by_window.get((video_id, window_id), pd.DataFrame())

    if len(cand):
        candidate_tracklets = cand["tracklet_id"].map(clean).tolist()
        candidate_count = len(candidate_tracklets)
        track_rows = int(cand["track_rows"].sum())
        sampled_frames = int(cand[["first_frame", "last_frame"]].drop_duplicates().shape[0])
        top_tracklet = clean(cand.iloc[0]["tracklet_id"])
        top_track_rows = int(cand.iloc[0]["track_rows"])
        top_score = float(cand.iloc[0]["max_score"])
        status = "HAS_TRACK_CANDIDATES"
    else:
        candidate_tracklets = []
        candidate_count = 0
        track_rows = 0
        sampled_frames = 0
        top_tracklet = ""
        top_track_rows = 0
        top_score = 0.0
        status = "NO_TRACK_CANDIDATES"

    fusion_rows.append({
        "clip_id": clip_id,
        "annotation_window_id": clean(cl.get("annotation_window_id", "")),
        "video_id": video_id,
        "video_filename": clean(cl["video_filename"]),
        "date": clean(cl["date"]),
        "tlc_camera": clean(cl["tlc_camera"]),
        "room_pen": clean(cl["room_pen"]),
        "identity_colour": clean(cl["identity_colour"]),
        "behaviour_label": clean(cl["behaviour_label"]),
        "clip_start_sec_in_video": clean(cl["clip_start_sec_in_video"]),
        "clip_end_sec_in_video": clean(cl["clip_end_sec_in_video"]),
        "window_id": window_id,
        "candidate_tracklet_count": candidate_count,
        "candidate_tracklets": ";".join(candidate_tracklets),
        "track_rows_in_window": track_rows,
        "sampled_frame_proxy_count": sampled_frames,
        "top_candidate_tracklet": top_tracklet,
        "top_candidate_track_rows": top_track_rows,
        "top_candidate_max_score": round(top_score, 6),
        "fusion_status": status,
        "claim_boundary": "candidate_tracklet_fusion_not_identity_assignment",
    })

fusion = pd.DataFrame(fusion_rows)
write(fusion, CLIP_FUSION)

# Summaries.
class_summary = (
    fusion.groupby("behaviour_label", dropna=False)
    .agg(
        clip_rows=("clip_id", "count"),
        clips_with_track_candidates=("fusion_status", lambda x: int((x == "HAS_TRACK_CANDIDATES").sum())),
        clips_without_track_candidates=("fusion_status", lambda x: int((x != "HAS_TRACK_CANDIDATES").sum())),
        mean_candidate_tracklet_count=("candidate_tracklet_count", "mean"),
        total_track_rows=("track_rows_in_window", "sum"),
    )
    .reset_index()
)
class_summary["candidate_coverage_ratio"] = class_summary["clips_with_track_candidates"] / class_summary["clip_rows"]
write(class_summary, CLASS_SUMMARY)

target_summary = (
    fusion.groupby(["date", "tlc_camera", "room_pen"], dropna=False)
    .agg(
        clip_rows=("clip_id", "count"),
        clips_with_track_candidates=("fusion_status", lambda x: int((x == "HAS_TRACK_CANDIDATES").sum())),
        clips_without_track_candidates=("fusion_status", lambda x: int((x != "HAS_TRACK_CANDIDATES").sum())),
        mean_candidate_tracklet_count=("candidate_tracklet_count", "mean"),
        total_track_rows=("track_rows_in_window", "sum"),
    )
    .reset_index()
)
target_summary["candidate_coverage_ratio"] = target_summary["clips_with_track_candidates"] / target_summary["clip_rows"]
write(target_summary, TARGET_SUMMARY)

clips_total = len(fusion)
clips_with_candidates = int((fusion["fusion_status"] == "HAS_TRACK_CANDIDATES").sum())
clips_without_candidates = clips_total - clips_with_candidates
coverage_ratio = clips_with_candidates / clips_total if clips_total else 0

if clips_without_candidates:
    issues.append({
        "item": "clip_fusion",
        "issue_type": "warning_some_clips_without_track_candidates",
        "severity": "warning",
        "detail": f"{clips_without_candidates} clips have no candidate tracklets.",
    })

issues.append({
    "item": "identity_assignment",
    "issue_type": "info_no_colour_identity_assignment_yet",
    "severity": "info",
    "detail": "v79f attaches candidate tracklets to clips but does not decide which tracklet belongs to which colour identity.",
})

issues.append({
    "item": "scope",
    "issue_type": "info_track_clip_fusion_only",
    "severity": "info",
    "detail": "v79f fuses sampled tracking outputs with behaviour clips. It is not final identity tracking.",
})

issues_df = pd.DataFrame(issues)
write(issues_df, ISSUES)

hard_count = int((issues_df["severity"] == "hard").sum()) if "severity" in issues_df.columns else 0
warning_count = int((issues_df["severity"] == "warning").sum()) if "severity" in issues_df.columns else 0

decision = pd.DataFrame([{
    "v79f_decision": "track_to_clip_fusion_created" if hard_count == 0 else "track_to_clip_fusion_has_blocking_issues",
    "input_clip_rows": len(clips),
    "input_track_rows": len(tracks),
    "input_windows": len(windows),
    "clip_fusion_rows": clips_total,
    "clips_with_track_candidates": clips_with_candidates,
    "clips_without_track_candidates": clips_without_candidates,
    "candidate_coverage_ratio": round(coverage_ratio, 6),
    "unique_tracklets": int(tracklet_summary["tracklet_id"].nunique()) if len(tracklet_summary) else 0,
    "hard_issue_count": hard_count,
    "warning_count": warning_count,
    "ready_for_visual_qa": bool(hard_count == 0 and clips_with_candidates > 0),
    "ready_for_identity_assignment": bool(hard_count == 0 and clips_with_candidates > 0),
    "ready_for_final_identity_tracking_claim": False,
    "claim_scope": "candidate_tracklet_fusion_only",
    "generated_at": datetime.now().isoformat(timespec="seconds"),
}])
write(decision, DECISION)

NOTE.write_text(
    "# v79f Track-to-Clip Fusion\n\n"
    f"- Decision: {decision.iloc[0]['v79f_decision']}\n"
    f"- Input clip rows: {len(clips)}\n"
    f"- Input track rows: {len(tracks)}\n"
    f"- Input windows: {len(windows)}\n"
    f"- Clip fusion rows: {clips_total}\n"
    f"- Clips with candidate tracklets: {clips_with_candidates}\n"
    f"- Clips without candidate tracklets: {clips_without_candidates}\n"
    f"- Candidate coverage ratio: {coverage_ratio:.6f}\n"
    f"- Unique tracklets: {decision.iloc[0]['unique_tracklets']}\n"
    f"- Hard issues: {hard_count}\n"
    f"- Ready for visual QA: {bool(hard_count == 0 and clips_with_candidates > 0)}\n\n"
    "This stage links behaviour clips to candidate sampled tracking outputs. It does not assign pig colour identity to a specific tracklet.\n",
    encoding="utf-8"
)

print("=== v79f decision ===")
print(decision.to_string(index=False))

print("\n=== target summary ===")
print(target_summary.to_string(index=False))

print("\n=== class summary ===")
print(class_summary.to_string(index=False))

print("\n=== fusion sample ===")
cols = ["clip_id", "video_id", "tlc_camera", "room_pen", "identity_colour", "behaviour_label", "window_id", "candidate_tracklet_count", "top_candidate_tracklet", "fusion_status"]
print(fusion[cols].head(40).to_string(index=False))

print("\n=== issues ===")
print(issues_df.to_string(index=False))
