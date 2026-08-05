from pathlib import Path
from datetime import datetime
import pandas as pd
import numpy as np

F = Path.home() / "PigBench" / "Full_Unibo_Behaviour_Pipeline"
O = F / "outputs/v80_final_project_completion/04_frame_based_baseline"
O.mkdir(parents=True, exist_ok=True)

tracks_path = F / "outputs/v79g4_roi_filter_tracks/v79g4_roi_filtered_tracks_KEEP.csv"
split_a_path = F / "outputs/v80_final_project_completion/03_splits/v80e_split_A_grouped_video_level.csv"
split_b_path = F / "outputs/v80_final_project_completion/03_splits/v80e_split_B_cross_camera_pen_level.csv"

tracks = pd.read_csv(tracks_path).fillna("")
split_a = pd.read_csv(split_a_path).fillna("")
split_b = pd.read_csv(split_b_path).fillna("")

def split_candidates(x):
    x = str(x)
    if not x:
        return []
    return [p for p in x.split(";") if p]

def build_frame_proxy(df, split_col, protocol_name):
    keep_cols = [
        "clip_id",
        "video_id",
        "video_filename",
        "date",
        "tlc_camera",
        "room_pen",
        "identity_colour",
        "behaviour_label",
        "clip_start_sec_in_video",
        "clip_end_sec_in_video",
        "window_id",
        "top6_candidate_tracklets",
        "top6_candidate_count",
        "identity_dataset_status",
        "usable_for_identity_training",
        "usable_for_behaviour_clip_dataset",
        split_col,
    ]

    clip = df[keep_cols].copy()
    clip["candidate_tracklet_id"] = clip["top6_candidate_tracklets"].apply(split_candidates)
    clip = clip.explode("candidate_tracklet_id").fillna("")
    clip = clip[clip["candidate_tracklet_id"].astype(str) != ""].copy()

    clip["candidate_rank_proxy"] = (
        clip.groupby(["clip_id", "window_id"]).cumcount() + 1
    )

    merged = clip.merge(
        tracks,
        left_on=["video_id", "window_id", "candidate_tracklet_id"],
        right_on=["video_id", "window_id", "tracklet_id"],
        how="left",
        suffixes=("", "_track"),
    ).fillna("")

    for c in [
        "clip_start_sec_in_video",
        "clip_end_sec_in_video",
        "time_sec",
        "frame_index",
        "score",
        "x1",
        "y1",
        "x2",
        "y2",
        "bbox_cx",
        "bbox_cy",
    ]:
        if c in merged.columns:
            merged[c] = pd.to_numeric(merged[c], errors="coerce")

    merged = merged.dropna(subset=["time_sec", "x1", "y1", "x2", "y2"]).copy()

    merged = merged[
        (merged["time_sec"] >= merged["clip_start_sec_in_video"])
        & (merged["time_sec"] <= merged["clip_end_sec_in_video"])
    ].copy()

    merged["bbox_w"] = merged["x2"] - merged["x1"]
    merged["bbox_h"] = merged["y2"] - merged["y1"]
    merged["bbox_area"] = merged["bbox_w"] * merged["bbox_h"]
    merged["bbox_aspect"] = merged["bbox_w"] / merged["bbox_h"].replace(0, np.nan)
    merged["clip_duration"] = merged["clip_end_sec_in_video"] - merged["clip_start_sec_in_video"]
    merged["time_norm_in_clip"] = (
        merged["time_sec"] - merged["clip_start_sec_in_video"]
    ) / merged["clip_duration"].replace(0, np.nan)

    max_x = max(float(merged["x2"].max()), 1.0)
    max_y = max(float(merged["y2"].max()), 1.0)
    max_area = max(float(merged["bbox_area"].max()), 1.0)

    merged["bbox_cx_norm"] = merged["bbox_cx"] / max_x
    merged["bbox_cy_norm"] = merged["bbox_cy"] / max_y
    merged["bbox_w_norm"] = merged["bbox_w"] / max_x
    merged["bbox_h_norm"] = merged["bbox_h"] / max_y
    merged["bbox_area_norm"] = merged["bbox_area"] / max_area

    merged["split"] = merged[split_col]
    merged["split_protocol"] = protocol_name
    merged["frame_proxy_label_source"] = "clip_level_behaviour_label_expanded_to_roi_candidate_track_rows"
    merged["claim_boundary"] = "frame_proxy_dataset_not_final_colour_to_track_identity_assignment"

    out_cols = [
        "split_protocol",
        "split",
        "clip_id",
        "video_id",
        "video_filename",
        "date",
        "tlc_camera",
        "room_pen",
        "identity_colour",
        "behaviour_label",
        "window_id",
        "candidate_tracklet_id",
        "candidate_rank_proxy",
        "tracklet_id",
        "time_sec",
        "frame_index",
        "score",
        "x1",
        "y1",
        "x2",
        "y2",
        "bbox_cx",
        "bbox_cy",
        "bbox_w",
        "bbox_h",
        "bbox_area",
        "bbox_aspect",
        "bbox_cx_norm",
        "bbox_cy_norm",
        "bbox_w_norm",
        "bbox_h_norm",
        "bbox_area_norm",
        "time_norm_in_clip",
        "identity_dataset_status",
        "usable_for_identity_training",
        "usable_for_behaviour_clip_dataset",
        "frame_proxy_label_source",
        "claim_boundary",
    ]

    return merged[out_cols].copy()

A = build_frame_proxy(split_a, "split_A_grouped_video", "split_A_grouped_video_level")
B = build_frame_proxy(split_b, "split_B_cross_camera", "split_B_cross_camera_pen_level")

A.to_csv(O / "v80f2_split_A_frame_proxy_dataset.csv", index=False)
B.to_csv(O / "v80f2_split_B_frame_proxy_dataset.csv", index=False)

summary = pd.DataFrame([
    {
        "dataset": "split_A_frame_proxy",
        "rows": len(A),
        "clips": A["clip_id"].nunique(),
        "videos": A["video_id"].nunique(),
        "classes": A["behaviour_label"].nunique(),
        "train_rows": int((A["split"] == "train").sum()),
        "val_rows": int((A["split"] == "val").sum()),
        "test_rows": int((A["split"] == "test").sum()),
    },
    {
        "dataset": "split_B_frame_proxy",
        "rows": len(B),
        "clips": B["clip_id"].nunique(),
        "videos": B["video_id"].nunique(),
        "classes": B["behaviour_label"].nunique(),
        "train_rows": int((B["split"] == "train").sum()),
        "val_rows": int((B["split"] == "val").sum()),
        "test_rows": int((B["split"] == "test").sum()),
    },
])

summary.to_csv(O / "v80f2_frame_proxy_dataset_summary.csv", index=False)

issues = []

for name, data in [("split_A", A), ("split_B", B)]:
    if len(data) == 0:
        issues.append({
            "item": name,
            "issue_type": "empty_frame_proxy_dataset",
            "severity": "hard",
            "detail": "no frame proxy rows created",
        })
    missing_splits = set(["train", "val", "test"]) - set(data["split"].astype(str))
    if missing_splits:
        issues.append({
            "item": name,
            "issue_type": "missing_split",
            "severity": "hard",
            "detail": ";".join(sorted(missing_splits)),
        })

if not issues:
    issues = [{
        "item": "none",
        "issue_type": "none",
        "severity": "info",
        "detail": "frame proxy datasets created",
    }]

pd.DataFrame(issues).to_csv(O / "v80f2_issues.csv", index=False)

hard = sum(1 for x in issues if x["severity"] == "hard")

decision = pd.DataFrame([{
    "v80f2_decision": "frame_proxy_datasets_created" if hard == 0 else "frame_proxy_dataset_has_blocking_issues",
    "split_A_frame_rows": len(A),
    "split_B_frame_rows": len(B),
    "split_A_clips": A["clip_id"].nunique(),
    "split_B_clips": B["clip_id"].nunique(),
    "split_A_classes": A["behaviour_label"].nunique(),
    "split_B_classes": B["behaviour_label"].nunique(),
    "hard_issue_count": hard,
    "ready_for_v80f3_frame_based_training": hard == 0,
    "claim_scope": "frame_proxy_dataset_for_frame_based_baseline_not_final_identity_assignment",
    "generated_at": datetime.now().isoformat(timespec="seconds"),
}])

decision.to_csv(O / "v80f2_decision_summary.csv", index=False)

note = F / "notes/v80f2_frame_proxy_dataset_notes.md"
note.write_text(
    "# v80f-2 Frame Proxy Dataset\n\n"
    f"- Decision: {decision.iloc[0]['v80f2_decision']}\n"
    f"- Split A frame rows: {len(A)}\n"
    f"- Split B frame rows: {len(B)}\n"
    f"- Split A clips: {A['clip_id'].nunique()}\n"
    f"- Split B clips: {B['clip_id'].nunique()}\n"
    f"- Hard issues: {hard}\n"
    f"- Ready for v80f-3 frame-based training: {hard == 0}\n\n"
    "The frame proxy dataset expands clip-level behaviour labels over ROI-filtered candidate track rows. "
    "It is suitable for a baseline frame-based classifier, but it does not claim final colour-to-track identity assignment.\n",
    encoding="utf-8",
)

print(decision.to_string(index=False))
print("=== summary ===")
print(summary.to_string(index=False))
print("=== issues ===")
print(pd.DataFrame(issues).to_string(index=False))
