from pathlib import Path
from datetime import datetime
import csv
import pandas as pd

ROOT = Path.home() / "PigBench"
FULL = ROOT / "Full_Unibo_Behaviour_Pipeline"

CLIPS = FULL / "outputs/v79a_tracking_preparation_manifest/v79a_annotation_clip_manifest.csv"
VIDEOS = FULL / "outputs/v79a_tracking_preparation_manifest/v79a_unique_tracking_video_manifest.csv"

OUT = FULL / "outputs/v79b_tracking_manifest_audit_and_run_plan"
OUT.mkdir(parents=True, exist_ok=True)

RUN_PLAN = OUT / "v79b_tracking_run_plan_36_videos.csv"
PILOT_PLAN = OUT / "v79b_pilot_tracking_run_plan.csv"
CLIP_AUDIT = OUT / "v79b_clip_manifest_audit.csv"
VIDEO_AUDIT = OUT / "v79b_video_manifest_audit.csv"
DECISION = OUT / "v79b_decision_summary.csv"
ISSUES = OUT / "v79b_issues.csv"
NOTE = FULL / "notes/v79b_tracking_manifest_audit_and_run_plan_notes.md"
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

clips = read_csv(CLIPS)
videos = read_csv(VIDEOS)

required_clip_cols = [
    "clip_id", "annotation_window_id", "date", "tlc_camera", "room_pen",
    "identity_colour", "behaviour_label", "video_id", "video_filename",
    "video_path", "clip_start_sec_in_video", "clip_end_sec_in_video"
]

required_video_cols = [
    "video_id", "video_filename", "video_path", "video_type", "date",
    "tlc_camera", "room_pen", "clip_count"
]

for c in required_clip_cols:
    if c not in clips.columns:
        issues.append({"item": c, "issue_type": "hard_missing_clip_column", "severity": "hard", "detail": f"Missing clip column: {c}"})

for c in required_video_cols:
    if c not in videos.columns:
        issues.append({"item": c, "issue_type": "hard_missing_video_column", "severity": "hard", "detail": f"Missing video column: {c}"})

if len(clips) == 0:
    issues.append({"item": "clip_manifest", "issue_type": "hard_empty_clip_manifest", "severity": "hard", "detail": "Clip manifest is empty."})

if len(videos) == 0:
    issues.append({"item": "video_manifest", "issue_type": "hard_empty_video_manifest", "severity": "hard", "detail": "Video manifest is empty."})

# Video path existence audit.
video_audit_rows = []
if len(videos):
    for _, r in videos.iterrows():
        vp = Path(clean(r["video_path"]))
        exists = vp.exists()
        video_audit_rows.append({
            "video_id": clean(r["video_id"]),
            "video_filename": clean(r["video_filename"]),
            "video_path": str(vp),
            "exists": exists,
            "file_size_bytes": vp.stat().st_size if exists else 0,
            "date": clean(r["date"]),
            "tlc_camera": clean(r["tlc_camera"]),
            "room_pen": clean(r["room_pen"]),
            "clip_count": clean(r["clip_count"]),
        })
        if not exists:
            issues.append({
                "item": clean(r["video_filename"]),
                "issue_type": "hard_video_path_missing",
                "severity": "hard",
                "detail": str(vp),
            })

video_audit = pd.DataFrame(video_audit_rows)
write(video_audit, VIDEO_AUDIT)

# Clip sanity.
clip_audit_rows = []
bad_time_count = 0
duplicate_clip_count = int(clips["clip_id"].duplicated().sum()) if "clip_id" in clips.columns else 0

if duplicate_clip_count:
    issues.append({
        "item": "clip_id",
        "issue_type": "hard_duplicate_clip_ids",
        "severity": "hard",
        "detail": f"{duplicate_clip_count} duplicated clip_id rows.",
    })

if len(clips):
    for _, r in clips.iterrows():
        try:
            start = float(r["clip_start_sec_in_video"])
            end = float(r["clip_end_sec_in_video"])
            good_time = start >= 0 and end > start
        except Exception:
            start = ""
            end = ""
            good_time = False

        if not good_time:
            bad_time_count += 1

        clip_audit_rows.append({
            "clip_id": clean(r["clip_id"]),
            "video_id": clean(r["video_id"]),
            "date": clean(r["date"]),
            "tlc_camera": clean(r["tlc_camera"]),
            "room_pen": clean(r["room_pen"]),
            "behaviour_label": clean(r["behaviour_label"]),
            "identity_colour": clean(r["identity_colour"]),
            "clip_start_sec_in_video": start,
            "clip_end_sec_in_video": end,
            "time_valid": good_time,
        })

if bad_time_count:
    issues.append({
        "item": "clip_time",
        "issue_type": "hard_invalid_clip_relative_times",
        "severity": "hard",
        "detail": f"{bad_time_count} clips have invalid relative time.",
    })

clip_audit = pd.DataFrame(clip_audit_rows)
write(clip_audit, CLIP_AUDIT)

# Run plan: one row per video.
run_plan_rows = []
if len(videos):
    for i, r in videos.reset_index(drop=True).iterrows():
        video_id = clean(r["video_id"])
        clip_count = int(float(clean(r["clip_count"]) or 0))

        run_plan_rows.append({
            "run_order": i,
            "video_id": video_id,
            "video_filename": clean(r["video_filename"]),
            "video_path": clean(r["video_path"]),
            "date": clean(r["date"]),
            "tlc_camera": clean(r["tlc_camera"]),
            "room_pen": clean(r["room_pen"]),
            "c_code": clean(r.get("c_code", "")),
            "clip_count": clip_count,
            "tracking_output_dir": str(OUT / "tracking_runs" / video_id),
            "run_status": "PENDING",
            "priority": "pilot" if i < 3 else "full",
        })

run_plan = pd.DataFrame(run_plan_rows)
write(run_plan, RUN_PLAN)

# Pilot: choose diverse first videos: B1, B6, C1 if available.
pilot_targets = [("TLC1", "B1"), ("TLC2", "B6"), ("TLC3", "C1")]
pilot_rows = []
for tlc, pen in pilot_targets:
    sub = run_plan[(run_plan["tlc_camera"] == tlc) & (run_plan["room_pen"] == pen)].copy()
    if len(sub):
        sub = sub.sort_values(["clip_count", "video_filename"], ascending=[False, True])
        pilot_rows.append(sub.iloc[0].to_dict())

if not pilot_rows and len(run_plan):
    pilot_rows = run_plan.head(3).to_dict("records")

pilot = pd.DataFrame(pilot_rows)
write(pilot, PILOT_PLAN)

issues.append({
    "item": "scope",
    "issue_type": "info_audit_and_plan_only",
    "severity": "info",
    "detail": "v79b validates manifests and creates tracking run plans. It does not run detector/tracker.",
})

issues_df = pd.DataFrame(issues)
write(issues_df, ISSUES)

hard_count = int((issues_df["severity"] == "hard").sum()) if "severity" in issues_df.columns else 0
warning_count = int((issues_df["severity"] == "warning").sum()) if "severity" in issues_df.columns else 0

decision = pd.DataFrame([{
    "v79b_decision": "tracking_manifest_audit_passed" if hard_count == 0 else "tracking_manifest_audit_has_blocking_issues",
    "input_clip_rows": len(clips),
    "input_video_rows": len(videos),
    "run_plan_video_rows": len(run_plan),
    "pilot_video_rows": len(pilot),
    "missing_video_paths": int((video_audit["exists"] == False).sum()) if len(video_audit) else 0,
    "invalid_clip_time_rows": bad_time_count,
    "duplicate_clip_ids": duplicate_clip_count,
    "hard_issue_count": hard_count,
    "warning_count": warning_count,
    "ready_for_pilot_tracking_run": bool(hard_count == 0 and len(pilot) > 0),
    "ready_for_full_tracking_run": bool(hard_count == 0 and len(run_plan) > 0),
    "run_plan_csv": str(RUN_PLAN),
    "pilot_plan_csv": str(PILOT_PLAN),
    "claim_scope": "tracking_manifest_audit_and_run_plan_only",
    "generated_at": datetime.now().isoformat(timespec="seconds"),
}])
write(decision, DECISION)

NOTE.write_text(
    "# v79b Tracking Manifest Audit and Run Plan\n\n"
    f"- Decision: {decision.iloc[0]['v79b_decision']}\n"
    f"- Input clip rows: {len(clips)}\n"
    f"- Input video rows: {len(videos)}\n"
    f"- Run plan videos: {len(run_plan)}\n"
    f"- Pilot videos: {len(pilot)}\n"
    f"- Missing video paths: {decision.iloc[0]['missing_video_paths']}\n"
    f"- Invalid clip times: {bad_time_count}\n"
    f"- Duplicate clip ids: {duplicate_clip_count}\n"
    f"- Hard issues: {hard_count}\n"
    f"- Ready for pilot tracking run: {bool(hard_count == 0 and len(pilot) > 0)}\n"
    f"- Ready for full tracking run: {bool(hard_count == 0 and len(run_plan) > 0)}\n\n"
    "This stage audits tracking manifests and creates pilot/full tracking run plans. It does not run detector/tracker.\n",
    encoding="utf-8"
)

print("=== v79b decision ===")
print(decision.to_string(index=False))

print("\n=== pilot plan ===")
print(pilot.to_string(index=False) if len(pilot) else "empty")

print("\n=== run plan sample ===")
print(run_plan.head(20).to_string(index=False) if len(run_plan) else "empty")

print("\n=== issues ===")
print(issues_df.to_string(index=False))
