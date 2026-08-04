from pathlib import Path
from datetime import datetime
import re
import csv
import pandas as pd

ROOT = Path.home() / "PigBench"
FULL = ROOT / "Full_Unibo_Behaviour_Pipeline"

READY = FULL / "outputs/v78k9c_annotation_video_join_corrected/v78k9c_ready_annotation_video_rows.csv"

OUT = FULL / "outputs/v79a_tracking_preparation_manifest"
OUT.mkdir(parents=True, exist_ok=True)

VIDEO_MANIFEST = OUT / "v79a_unique_tracking_video_manifest.csv"
CLIP_MANIFEST = OUT / "v79a_annotation_clip_manifest.csv"
VIDEO_COUNTS = OUT / "v79a_annotation_counts_by_video.csv"
TARGET_COUNTS = OUT / "v79a_annotation_counts_by_target.csv"
CLASS_COUNTS = OUT / "v79a_behaviour_class_counts_ready_subset.csv"
DECISION = OUT / "v79a_decision_summary.csv"
ISSUES = OUT / "v79a_issues.csv"
NOTE = FULL / "notes/v79a_tracking_preparation_manifest_notes.md"
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

def parse_hms(t):
    t = clean(t)
    m = re.match(r"^(\d{2}):(\d{2}):(\d{2})$", t)
    if not m:
        return None
    return int(m.group(1))*3600 + int(m.group(2))*60 + int(m.group(3))

issues = []

ready = pd.read_csv(READY).fillna("")
for c in ready.columns:
    if ready[c].dtype == object:
        ready[c] = ready[c].map(clean)

required = [
    "annotation_window_id",
    "date",
    "tlc_camera",
    "resolved_room_pen",
    "identity_colour",
    "behaviour_label",
    "annotation_start_time",
    "matched_video_id",
    "matched_video_filename",
    "matched_video_path",
    "matched_c_code",
    "matched_video_type",
    "matched_video_start_time",
]

for c in required:
    if c not in ready.columns:
        issues.append({
            "item": c,
            "issue_type": "hard_missing_required_ready_column",
            "severity": "hard",
            "detail": f"Missing required column: {c}",
        })

if len(ready) == 0:
    issues.append({
        "item": "ready_rows",
        "issue_type": "hard_empty_ready_rows",
        "severity": "hard",
        "detail": "No ready annotation-video rows found.",
    })

clip_rows = []

if len(ready) and not issues:
    for _, r in ready.iterrows():
        ann_sec = parse_hms(r["annotation_start_time"])
        vid_sec = parse_hms(r["matched_video_start_time"])

        rel_start = ""
        rel_end = ""

        if ann_sec is not None and vid_sec is not None:
            rel_start = ann_sec - vid_sec
            rel_end = rel_start + 10

        clip_rows.append({
            "clip_id": f"clip_{len(clip_rows):06d}",
            "annotation_window_id": clean(r["annotation_window_id"]),
            "date": clean(r["date"]),
            "tlc_camera": clean(r["tlc_camera"]),
            "room_pen": clean(r["resolved_room_pen"]),
            "identity_colour": clean(r["identity_colour"]),
            "behaviour_label": clean(r["behaviour_label"]),
            "annotation_start_time": clean(r["annotation_start_time"]),
            "video_id": clean(r["matched_video_id"]),
            "video_filename": clean(r["matched_video_filename"]),
            "video_path": clean(r["matched_video_path"]),
            "video_type": clean(r["matched_video_type"]),
            "c_code": clean(r["matched_c_code"]),
            "video_start_time": clean(r["matched_video_start_time"]),
            "clip_start_sec_in_video": rel_start,
            "clip_end_sec_in_video": rel_end,
            "clip_duration_sec": 10,
            "tracking_input_status": "READY_FOR_DETECTION_TRACKING",
            "claim_boundary": "tracking_preparation_manifest_only",
        })

clips = pd.DataFrame(clip_rows)
write(clips, CLIP_MANIFEST)

if len(clips):
    bad_rel = clips[
        (clips["clip_start_sec_in_video"] == "") |
        (clips["clip_start_sec_in_video"].astype(str).str.startswith("-"))
    ]
    if len(bad_rel):
        issues.append({
            "item": "clip_relative_time",
            "issue_type": "warning_some_bad_relative_clip_times",
            "severity": "warning",
            "detail": f"{len(bad_rel)} clips have blank or negative relative start time.",
        })

    video_manifest = (
        clips.groupby(["video_id", "video_filename", "video_path", "video_type", "date", "tlc_camera", "room_pen", "c_code"], dropna=False)
        .agg(
            clip_count=("clip_id", "count"),
            behaviour_labels=("behaviour_label", lambda x: ";".join(sorted(set([clean(v) for v in x if clean(v)])))),
            identity_colours=("identity_colour", lambda x: ";".join(sorted(set([clean(v) for v in x if clean(v)])))),
            min_clip_start_sec=("clip_start_sec_in_video", "min"),
            max_clip_end_sec=("clip_end_sec_in_video", "max"),
        )
        .reset_index()
        .sort_values(["date", "tlc_camera", "room_pen", "min_clip_start_sec", "video_filename"])
    )
else:
    video_manifest = pd.DataFrame()

write(video_manifest, VIDEO_MANIFEST)

if len(clips):
    vc = (
        clips.groupby(["video_id", "video_filename", "tlc_camera", "room_pen", "c_code"], dropna=False)
        .size()
        .reset_index(name="annotation_clip_count")
        .sort_values("annotation_clip_count", ascending=False)
    )

    tc = (
        clips.groupby(["date", "tlc_camera", "room_pen"], dropna=False)
        .size()
        .reset_index(name="annotation_clip_count")
        .sort_values(["date", "tlc_camera", "room_pen"])
    )

    cc = (
        clips.groupby("behaviour_label", dropna=False)
        .size()
        .reset_index(name="clip_count")
        .sort_values("clip_count", ascending=False)
    )
else:
    vc = pd.DataFrame()
    tc = pd.DataFrame()
    cc = pd.DataFrame()

write(vc, VIDEO_COUNTS)
write(tc, TARGET_COUNTS)
write(cc, CLASS_COUNTS)

issues.append({
    "item": "scope",
    "issue_type": "info_manifest_only",
    "severity": "info",
    "detail": "v79a creates tracking preparation manifests only. It does not run detector/tracker.",
})

issues_df = pd.DataFrame(issues)
write(issues_df, ISSUES)

hard_count = int((issues_df["severity"] == "hard").sum()) if "severity" in issues_df.columns else 0
warning_count = int((issues_df["severity"] == "warning").sum()) if "severity" in issues_df.columns else 0

decision = pd.DataFrame([{
    "v79a_decision": "tracking_preparation_manifest_created" if hard_count == 0 else "tracking_preparation_manifest_has_blocking_issues",
    "input_ready_rows": len(ready),
    "clip_manifest_rows": len(clips),
    "unique_tracking_videos": len(video_manifest),
    "unique_targets": clips[["date","tlc_camera","room_pen"]].drop_duplicates().shape[0] if len(clips) else 0,
    "unique_behaviour_labels": clips["behaviour_label"].nunique() if len(clips) else 0,
    "video_manifest_csv": str(VIDEO_MANIFEST),
    "clip_manifest_csv": str(CLIP_MANIFEST),
    "hard_issue_count": hard_count,
    "warning_count": warning_count,
    "ready_for_detector_tracking_run": bool(hard_count == 0 and len(video_manifest) > 0 and len(clips) > 0),
    "ready_for_full_tracking": False,
    "claim_scope": "tracking_preparation_manifest_only",
    "generated_at": datetime.now().isoformat(timespec="seconds"),
}])

write(decision, DECISION)

NOTE.write_text(
    "# v79a Tracking Preparation Manifest\n\n"
    f"- Decision: {decision.iloc[0]['v79a_decision']}\n"
    f"- Input ready rows: {len(ready)}\n"
    f"- Clip manifest rows: {len(clips)}\n"
    f"- Unique tracking videos: {len(video_manifest)}\n"
    f"- Unique targets: {decision.iloc[0]['unique_targets']}\n"
    f"- Unique behaviour labels: {decision.iloc[0]['unique_behaviour_labels']}\n"
    f"- Hard issues: {hard_count}\n"
    f"- Ready for detector/tracking run: {bool(hard_count == 0 and len(video_manifest) > 0 and len(clips) > 0)}\n\n"
    "This stage prepares tracking input manifests from v78k9c matched annotation-video rows. It does not run detector/tracker.\n",
    encoding="utf-8"
)

print("=== v79a decision ===")
print(decision.to_string(index=False))

print("\n=== target counts ===")
print(tc.to_string(index=False) if len(tc) else "empty")

print("\n=== behaviour counts ===")
print(cc.to_string(index=False) if len(cc) else "empty")

print("\n=== video manifest sample ===")
print(video_manifest.head(50).to_string(index=False) if len(video_manifest) else "empty")

print("\n=== issues ===")
print(issues_df.to_string(index=False))
