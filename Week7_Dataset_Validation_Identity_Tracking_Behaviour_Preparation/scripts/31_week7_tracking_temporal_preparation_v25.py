from pathlib import Path
from datetime import datetime
import csv
import re
import math

import pandas as pd


ROOT = Path.home() / "PigBench"
W6 = ROOT / "Week6_Unibo_Dataset_Validation"
W7 = ROOT / "Week7_Dataset_Validation_Identity_Tracking_Behaviour_Preparation"

RAW_VIDEO_ROOT = Path("/work/pig/datasets/Unibo")

PRIMARY_SPLIT = W7 / "outputs" / "primary_split_v21" / "week7_primary_split_v21_all.csv"
BOX_LEVEL = W7 / "outputs" / "behaviour_label_fusion_v18c_verified_crosswalk" / "week7_behaviour_label_fusion_v18c_box_level_dataset.csv"
LABELS_LONG = W6 / "outputs" / "unified_ground_truth" / "week6_scanpoint_frame_labels_long.csv"
V24_SUMMARY = W7 / "outputs" / "final_audit_package_v24" / "Week7_Final_Audit_Package_v24" / "package_summary_v24.csv"

OUT_ROOT = W7 / "outputs" / "tracking_temporal_preparation_v25"
OUT_ROOT.mkdir(parents=True, exist_ok=True)

OUT_VIDEO_INVENTORY = OUT_ROOT / "week7_tracking_temporal_v25_raw_video_inventory.csv"
OUT_SCANPOINT_TIMELINE = OUT_ROOT / "week7_tracking_temporal_v25_scanpoint_timeline.csv"
OUT_VIDEO_SUMMARY = OUT_ROOT / "week7_tracking_temporal_v25_video_scanpoint_summary.csv"
OUT_FPS_ESTIMATES = OUT_ROOT / "week7_tracking_temporal_v25_fps_estimates.csv"
OUT_CLIP_PLAN = OUT_ROOT / "week7_tracking_temporal_v25_10sec_clip_window_plan.csv"
OUT_TRACKING_PLAN = OUT_ROOT / "week7_tracking_temporal_v25_tracking_run_plan.csv"
OUT_ISSUES = OUT_ROOT / "week7_tracking_temporal_v25_issues.csv"
OUT_SUMMARY = OUT_ROOT / "week7_tracking_temporal_v25_summary.csv"
OUT_README = OUT_ROOT / "README_tracking_temporal_preparation_v25.md"
OUT_NOTE = W7 / "notes" / "week7_tracking_temporal_preparation_v25_notes.md"

VIDEO_EXTS = [".mp4", ".avi", ".mov", ".mkv", ".m4v"]


def safe_to_csv(df, path):
    df.to_csv(
        path,
        index=False,
        quoting=csv.QUOTE_ALL,
        escapechar="\\",
        lineterminator="\n",
    )


def clean(v):
    if pd.isna(v):
        return ""
    return str(v).strip()


def slug_text(s):
    s = clean(s).lower()
    s = re.sub(r"[^a-z0-9]+", "", s)
    return s


def first_existing_col(df, candidates):
    for c in candidates:
        if c in df.columns:
            return c
    return None


def parse_time_col(df, col):
    if col and col in df.columns:
        return pd.to_datetime(df[col], errors="coerce")
    return pd.Series([pd.NaT] * len(df), index=df.index)


def find_video_candidates(video_id, inventory):
    if not video_id or len(inventory) == 0:
        return []

    key = slug_text(video_id)
    if not key:
        return []

    matches = []

    for _, r in inventory.iterrows():
        stem_key = slug_text(r["video_stem"])
        name_key = slug_text(r["video_filename"])

        if key in stem_key or stem_key in key or key in name_key:
            matches.append((100, r["video_path"]))

        # Softer token overlap.
        tokens = [t for t in re.split(r"[^a-zA-Z0-9]+", video_id.lower()) if len(t) >= 3]
        overlap = sum(1 for t in tokens if t in r["video_filename"].lower())
        if overlap > 0:
            matches.append((overlap * 10, r["video_path"]))

    if not matches:
        return []

    matches = sorted(set(matches), key=lambda x: (-x[0], x[1]))
    return [m[1] for m in matches[:5]]


def build_video_inventory():
    rows = []

    if RAW_VIDEO_ROOT.exists():
        for p in RAW_VIDEO_ROOT.rglob("*"):
            if p.is_file() and p.suffix.lower() in VIDEO_EXTS:
                rows.append({
                    "video_path": str(p),
                    "video_filename": p.name,
                    "video_stem": p.stem,
                    "extension": p.suffix.lower(),
                    "size_bytes": p.stat().st_size,
                    "modified_at": datetime.fromtimestamp(p.stat().st_mtime).isoformat(timespec="seconds"),
                })

    return pd.DataFrame(rows)


def estimate_fps_from_scanpoints(timeline):
    rows = []

    if len(timeline) == 0:
        return pd.DataFrame(columns=[
            "video_id", "fps_estimate", "fps_source", "valid_pair_count",
            "frame_index_min", "frame_index_max", "timestamp_min", "timestamp_max"
        ])

    for video_id, g in timeline.groupby("video_id", dropna=False):
        g = g.copy()
        g["frame_index_numeric"] = pd.to_numeric(g["frame_index"], errors="coerce")
        g["timestamp_dt"] = pd.to_datetime(g["timestamp"], errors="coerce")
        g = g.dropna(subset=["frame_index_numeric", "timestamp_dt"])
        g = g.sort_values(["timestamp_dt", "frame_index_numeric"])

        fps_values = []

        for i in range(1, len(g)):
            prev = g.iloc[i - 1]
            curr = g.iloc[i]

            df_frames = float(curr["frame_index_numeric"] - prev["frame_index_numeric"])
            dt_sec = (curr["timestamp_dt"] - prev["timestamp_dt"]).total_seconds()

            if dt_sec > 0 and df_frames > 0:
                fps_values.append(df_frames / dt_sec)

        if fps_values:
            fps = sum(fps_values) / len(fps_values)
            fps_source = "estimated_from_scanpoint_frame_index_and_timestamp"
        else:
            fps = 25.0
            fps_source = "fallback_default_25fps"

        rows.append({
            "video_id": video_id,
            "fps_estimate": round(float(fps), 4),
            "fps_source": fps_source,
            "valid_pair_count": int(len(fps_values)),
            "frame_index_min": float(g["frame_index_numeric"].min()) if len(g) else "",
            "frame_index_max": float(g["frame_index_numeric"].max()) if len(g) else "",
            "timestamp_min": g["timestamp_dt"].min().isoformat() if len(g) else "",
            "timestamp_max": g["timestamp_dt"].max().isoformat() if len(g) else "",
        })

    return pd.DataFrame(rows)


def make_clip_plan(timeline, fps_df):
    fps_map = dict(zip(fps_df["video_id"].astype(str), fps_df["fps_estimate"]))

    rows = []

    for _, r in timeline.iterrows():
        video_id = clean(r.get("video_id", ""))
        scan_frame_id = clean(r.get("scan_frame_id", ""))
        frame_index = pd.to_numeric(pd.Series([r.get("frame_index", "")]), errors="coerce").iloc[0]
        fps = float(fps_map.get(video_id, 25.0))

        if pd.isna(frame_index):
            start_frame = ""
            end_frame = ""
            center_frame = ""
            status = "missing_frame_index"
        else:
            center_frame = int(round(float(frame_index)))
            half_window = int(round(5.0 * fps))
            start_frame = max(0, center_frame - half_window)
            end_frame = center_frame + half_window
            status = "planned"

        rows.append({
            "scan_frame_id": scan_frame_id,
            "video_id": video_id,
            "timestamp": clean(r.get("timestamp", "")),
            "center_frame_index": center_frame,
            "fps_used": fps,
            "clip_duration_seconds": 10,
            "start_frame_index": start_frame,
            "end_frame_index": end_frame,
            "candidate_video_path": clean(r.get("candidate_video_path", "")),
            "behaviour_codes_present": clean(r.get("behaviour_codes_present", "")),
            "pig_rows": r.get("training_ready_rows", ""),
            "clip_plan_status": status,
        })

    return pd.DataFrame(rows)


video_inventory = build_video_inventory()
safe_to_csv(video_inventory, OUT_VIDEO_INVENTORY)

issues = []

primary = pd.read_csv(PRIMARY_SPLIT)
box = pd.read_csv(BOX_LEVEL)
labels = pd.read_csv(LABELS_LONG)

for df in [primary, box, labels]:
    if "scan_frame_id" in df.columns:
        df["scan_frame_id"] = df["scan_frame_id"].fillna("").astype(str).str.strip()

# Prefer label file for scanpoint-level metadata because it has all 72 frames and all behaviour rows.
video_col = first_existing_col(labels, ["video_id", "video", "source_video"])
timestamp_col = first_existing_col(labels, ["timestamp", "timestamp_label", "datetime"])
frame_index_col = first_existing_col(labels, ["frame_index", "frame_idx", "frame"])
image_col = first_existing_col(labels, ["frame_image_path", "image_path", "img_path"])

if video_col is None:
    issues.append({
        "issue_type": "missing_video_id_column",
        "issue_detail": "No video_id-like column found in Week6 labels.",
    })

if frame_index_col is None:
    issues.append({
        "issue_type": "missing_frame_index_column",
        "issue_detail": "No frame_index-like column found in Week6 labels.",
    })

if timestamp_col is None:
    issues.append({
        "issue_type": "missing_timestamp_column",
        "issue_detail": "No timestamp-like column found in Week6 labels.",
    })

# Aggregate primary split/training-ready info per scanpoint.
primary_agg = (
    primary.groupby("scan_frame_id")
    .agg(
        training_ready_rows=("scan_frame_id", "count"),
        splits_present=("split_v20", lambda x: " | ".join(sorted(set(map(str, x)))) if "split_v20" in primary.columns else ""),
        behaviour_codes_present=("behaviour_code", lambda x: " | ".join(sorted(set(map(str, x)))) if "behaviour_code" in primary.columns else ""),
        visual_colours_present=("visual_marker_colour_v18c", lambda x: " | ".join(sorted(set(map(str, x)))) if "visual_marker_colour_v18c" in primary.columns else ""),
        behaviour_pig_ids_present=("behaviour_pig_id_v18c", lambda x: " | ".join(sorted(set(map(str, x)))) if "behaviour_pig_id_v18c" in primary.columns else ""),
    )
    .reset_index()
)

# Aggregate box-level info per scanpoint.
box_agg = (
    box.groupby("scan_frame_id")
    .agg(
        total_box_rows=("scan_frame_id", "count"),
        known_identity_boxes=("final_colour_identity_v17", lambda x: sum(clean(v).lower() in ["blue", "green", "cyan", "red", "pink", "purple"] for v in x) if "final_colour_identity_v17" in box.columns else 0),
        unknown_identity_boxes=("final_identity_status_v17", lambda x: sum("unknown" in clean(v).lower() for v in x) if "final_identity_status_v17" in box.columns else 0),
    )
    .reset_index()
)

# Metadata from labels.
label_rows = []
for sid, g in labels.groupby("scan_frame_id", sort=True):
    video_id = clean(g[video_col].iloc[0]) if video_col else ""
    timestamp = clean(g[timestamp_col].iloc[0]) if timestamp_col else ""
    frame_index = clean(g[frame_index_col].iloc[0]) if frame_index_col else ""
    image_path = clean(g[image_col].iloc[0]) if image_col else ""

    candidates = find_video_candidates(video_id, video_inventory)
    candidate_video_path = candidates[0] if candidates else ""

    label_rows.append({
        "scan_frame_id": sid,
        "video_id": video_id,
        "timestamp": timestamp,
        "frame_index": frame_index,
        "frame_image_path": image_path,
        "label_rows": int(len(g)),
        "behaviour_label_codes": " | ".join(sorted(set(g["behaviour_code"].astype(str)))) if "behaviour_code" in g.columns else "",
        "candidate_video_path": candidate_video_path,
        "candidate_video_match_count": int(len(candidates)),
        "candidate_video_paths_top5": " | ".join(candidates),
        "raw_video_match_status": "matched" if candidate_video_path else "not_found",
    })

timeline = pd.DataFrame(label_rows)
timeline = timeline.merge(primary_agg, on="scan_frame_id", how="left")
timeline = timeline.merge(box_agg, on="scan_frame_id", how="left")

for c in ["training_ready_rows", "total_box_rows", "known_identity_boxes", "unknown_identity_boxes"]:
    if c in timeline.columns:
        timeline[c] = pd.to_numeric(timeline[c], errors="coerce").fillna(0).astype(int)

# Sort.
timeline["scan_num"] = timeline["scan_frame_id"].apply(lambda x: int(re.search(r"(\d+)$", str(x)).group(1)) if re.search(r"(\d+)$", str(x)) else 10**9)
timeline = timeline.sort_values("scan_num").drop(columns=["scan_num"])

safe_to_csv(timeline, OUT_SCANPOINT_TIMELINE)

fps_df = estimate_fps_from_scanpoints(timeline)
safe_to_csv(fps_df, OUT_FPS_ESTIMATES)

clip_plan = make_clip_plan(timeline, fps_df)
safe_to_csv(clip_plan, OUT_CLIP_PLAN)

# Video summary.
video_rows = []
for video_id, g in timeline.groupby("video_id", dropna=False):
    fps_row = fps_df[fps_df["video_id"].astype(str) == str(video_id)]
    fps = fps_row["fps_estimate"].iloc[0] if len(fps_row) else 25.0
    fps_source = fps_row["fps_source"].iloc[0] if len(fps_row) else "fallback_default_25fps"

    candidate_paths = [p for p in g["candidate_video_path"].astype(str).unique().tolist() if p]
    candidate_video_path = candidate_paths[0] if candidate_paths else ""

    video_rows.append({
        "video_id": video_id,
        "scanpoint_frames": int(g["scan_frame_id"].nunique()),
        "label_rows": int(g["label_rows"].sum()),
        "training_ready_rows": int(g["training_ready_rows"].sum()),
        "total_box_rows": int(g["total_box_rows"].sum()),
        "known_identity_boxes": int(g["known_identity_boxes"].sum()),
        "unknown_identity_boxes": int(g["unknown_identity_boxes"].sum()),
        "behaviour_codes_present": " | ".join(sorted(set(" | ".join(g["behaviour_label_codes"].astype(str)).split(" | ")))),
        "splits_present": " | ".join(sorted(set(" | ".join(g["splits_present"].fillna("").astype(str)).split(" | ")) - {""})),
        "candidate_video_path": candidate_video_path,
        "raw_video_match_status": "matched" if candidate_video_path else "not_found",
        "fps_estimate": fps,
        "fps_source": fps_source,
    })

video_summary = pd.DataFrame(video_rows)
safe_to_csv(video_summary, OUT_VIDEO_SUMMARY)

# Tracking run plan.
tracking_rows = []

for _, r in video_summary.iterrows():
    video_id = clean(r["video_id"])
    candidate_path = clean(r["candidate_video_path"])

    if candidate_path:
        tracking_status = "ready_for_tracking"
        priority = "high" if int(r["training_ready_rows"]) > 0 else "medium"
    else:
        tracking_status = "blocked_missing_raw_video"
        priority = "blocked"

    tracking_rows.append({
        "video_id": video_id,
        "candidate_video_path": candidate_path,
        "scanpoint_frames": int(r["scanpoint_frames"]),
        "training_ready_rows": int(r["training_ready_rows"]),
        "known_identity_boxes": int(r["known_identity_boxes"]),
        "unknown_identity_boxes": int(r["unknown_identity_boxes"]),
        "fps_estimate": r["fps_estimate"],
        "tracking_status": tracking_status,
        "priority": priority,
        "planned_tracking_output_dir": str(OUT_ROOT / "future_tracking_outputs" / re.sub(r"[^A-Za-z0-9_.-]+", "_", video_id)),
        "notes": "Run detector/tracker in v26 only after this plan is accepted.",
    })

tracking_plan = pd.DataFrame(tracking_rows)
safe_to_csv(tracking_plan, OUT_TRACKING_PLAN)

# Issues.
for _, r in timeline[timeline["raw_video_match_status"] != "matched"].iterrows():
    issues.append({
        "issue_type": "raw_video_not_matched",
        "issue_detail": f"{r['scan_frame_id']} video_id={r['video_id']}",
    })

issues_df = pd.DataFrame(issues, columns=["issue_type", "issue_detail"])
safe_to_csv(issues_df, OUT_ISSUES)

summary = pd.DataFrame([{
    "source_primary_split": str(PRIMARY_SPLIT),
    "source_box_level": str(BOX_LEVEL),
    "source_labels_long": str(LABELS_LONG),
    "raw_video_root": str(RAW_VIDEO_ROOT),
    "raw_videos_found": int(len(video_inventory)),
    "scanpoint_frames": int(timeline["scan_frame_id"].nunique()),
    "videos_in_annotations": int(timeline["video_id"].nunique()),
    "videos_matched_to_raw_files": int(video_summary["raw_video_match_status"].eq("matched").sum()) if len(video_summary) else 0,
    "videos_missing_raw_files": int(video_summary["raw_video_match_status"].ne("matched").sum()) if len(video_summary) else 0,
    "total_training_ready_rows": int(timeline["training_ready_rows"].sum()),
    "total_box_rows": int(timeline["total_box_rows"].sum()),
    "total_known_identity_boxes": int(timeline["known_identity_boxes"].sum()),
    "total_unknown_identity_boxes": int(timeline["unknown_identity_boxes"].sum()),
    "clip_windows_planned": int(clip_plan["clip_plan_status"].eq("planned").sum()) if len(clip_plan) else 0,
    "tracking_ready_videos": int(tracking_plan["tracking_status"].eq("ready_for_tracking").sum()) if len(tracking_plan) else 0,
    "blocked_videos": int(tracking_plan["tracking_status"].ne("ready_for_tracking").sum()) if len(tracking_plan) else 0,
    "issue_count": int(len(issues_df)),
    "ready_for_v26_tracking": bool(len(tracking_plan) > 0 and tracking_plan["tracking_status"].eq("ready_for_tracking").any()),
    "generated_at": datetime.now().isoformat(timespec="seconds"),
}])

safe_to_csv(summary, OUT_SUMMARY)

ready = bool(summary.iloc[0]["ready_for_v26_tracking"])

OUT_README.write_text(
    "# Week 7 Tracking & Temporal Preparation v25\n\n"
    "## Purpose\n\n"
    "This step prepares the next professional block: tracking, temporal consistency, and clip-level representation. "
    "It does not run tracking yet. It audits scanpoint-to-video relationships, raw video availability, FPS estimates, and 10-second clip windows.\n\n"
    "## Outputs\n\n"
    "- `week7_tracking_temporal_v25_raw_video_inventory.csv`\n"
    "- `week7_tracking_temporal_v25_scanpoint_timeline.csv`\n"
    "- `week7_tracking_temporal_v25_video_scanpoint_summary.csv`\n"
    "- `week7_tracking_temporal_v25_fps_estimates.csv`\n"
    "- `week7_tracking_temporal_v25_10sec_clip_window_plan.csv`\n"
    "- `week7_tracking_temporal_v25_tracking_run_plan.csv`\n"
    "- `week7_tracking_temporal_v25_issues.csv`\n"
    "- `week7_tracking_temporal_v25_summary.csv`\n\n"
    "## Interpretation\n\n"
    "If raw videos are matched, v26 can run detector/tracker per video. "
    "If some videos are missing, tracking is blocked only for those videos, while the scanpoint-level dataset remains valid.\n"
)

OUT_NOTE.write_text(
    "# Week 7 Tracking & Temporal Preparation v25\n\n"
    "## Purpose\n\n"
    "This step audits the temporal and video-level basis for future tracking and clip-level work.\n\n"
    "## Summary\n\n"
    f"- Raw videos found: `{int(summary.iloc[0]['raw_videos_found'])}`\n"
    f"- Scanpoint frames: `{int(summary.iloc[0]['scanpoint_frames'])}`\n"
    f"- Videos in annotations: `{int(summary.iloc[0]['videos_in_annotations'])}`\n"
    f"- Videos matched to raw files: `{int(summary.iloc[0]['videos_matched_to_raw_files'])}`\n"
    f"- Videos missing raw files: `{int(summary.iloc[0]['videos_missing_raw_files'])}`\n"
    f"- Clip windows planned: `{int(summary.iloc[0]['clip_windows_planned'])}`\n"
    f"- Tracking-ready videos: `{int(summary.iloc[0]['tracking_ready_videos'])}`\n"
    f"- Issue count: `{int(summary.iloc[0]['issue_count'])}`\n"
    f"- Ready for v26 tracking: `{ready}`\n\n"
    "## Outputs\n\n"
    f"- Summary: `{OUT_SUMMARY}`\n"
    f"- Scanpoint timeline: `{OUT_SCANPOINT_TIMELINE}`\n"
    f"- Video summary: `{OUT_VIDEO_SUMMARY}`\n"
    f"- FPS estimates: `{OUT_FPS_ESTIMATES}`\n"
    f"- 10-sec clip window plan: `{OUT_CLIP_PLAN}`\n"
    f"- Tracking run plan: `{OUT_TRACKING_PLAN}`\n"
    f"- Issues: `{OUT_ISSUES}`\n"
)

print("Saved:")
print(OUT_VIDEO_INVENTORY)
print(OUT_SCANPOINT_TIMELINE)
print(OUT_VIDEO_SUMMARY)
print(OUT_FPS_ESTIMATES)
print(OUT_CLIP_PLAN)
print(OUT_TRACKING_PLAN)
print(OUT_ISSUES)
print(OUT_SUMMARY)
print(OUT_README)
print(OUT_NOTE)

print()
print("=== v25 tracking temporal preparation summary ===")
print(summary.to_string(index=False))

print()
print("=== v25 video summary ===")
print(video_summary.to_string(index=False))

print()
print("=== v25 issues ===")
if len(issues_df):
    print(issues_df.head(50).to_string(index=False))
else:
    print("No issues found.")
