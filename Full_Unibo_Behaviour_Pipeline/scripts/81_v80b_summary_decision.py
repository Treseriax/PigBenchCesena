from pathlib import Path
from datetime import datetime
import pandas as pd

F = Path.home() / "PigBench" / "Full_Unibo_Behaviour_Pipeline"
O = F / "outputs/v80_final_project_completion/01_dataset_json_export"

df = pd.read_csv(O / "v80b_final_video_mapping_status_table.csv").fillna("")

summary_parts = [
    df["mapping_status"]
    .value_counts()
    .rename_axis("category")
    .reset_index(name="count")
    .assign(summary_type="mapping_status"),
    df["annotation_status"]
    .value_counts()
    .rename_axis("category")
    .reset_index(name="count")
    .assign(summary_type="annotation_status"),
    df["tracking_subset_status"]
    .value_counts()
    .rename_axis("category")
    .reset_index(name="count")
    .assign(summary_type="tracking_subset_status"),
]

summary = pd.concat(summary_parts, ignore_index=True)
summary = summary[["summary_type", "category", "count"]]
summary.to_csv(O / "v80b_video_mapping_status_summary.csv", index=False)

inc = df[df["inconsistency_notes"] != "ok"].copy()
inc.to_csv(O / "v80b_inconsistency_report.csv", index=False)

mapping_unresolved = int((df["mapping_status"] != "MAPPING_RESOLVED").sum())
mapping_resolved = int((df["mapping_status"] == "MAPPING_RESOLVED").sum())
with_clips = int((df["annotation_status"] == "MATCHED_BEHAVIOUR_CLIPS_AVAILABLE").sum())
without_clips = int((df["annotation_status"] != "MATCHED_BEHAVIOUR_CLIPS_AVAILABLE").sum())
tracking_subset = int((df["tracking_subset_status"] == "IN_36_VIDEO_TRACKING_SUBSET").sum())

decision = pd.DataFrame(
    [
        {
            "v80b_decision": "final_84_video_mapping_status_table_created"
            if mapping_unresolved == 0
            else "mapping_status_table_has_unresolved_videos",
            "video_rows": len(df),
            "mapping_resolved_videos": mapping_resolved,
            "mapping_unresolved_videos": mapping_unresolved,
            "videos_with_matched_behaviour_clips": with_clips,
            "videos_without_matched_behaviour_clips": without_clips,
            "videos_in_tracking_subset": tracking_subset,
            "inconsistency_rows": len(inc),
            "ready_for_v80c_json_export": mapping_unresolved == 0 and with_clips > 0,
            "claim_scope": "84_video_mapping_status_table_not_full_84_behaviour_completion_claim",
            "generated_at": datetime.now().isoformat(timespec="seconds"),
        }
    ]
)

decision.to_csv(O / "v80b_decision_summary.csv", index=False)

note = F / "notes/v80b_final_84_video_mapping_status_notes.md"
note.parent.mkdir(parents=True, exist_ok=True)
note.write_text(
    "# v80b Final 84-video Mapping Status Table\n\n"
    f"- Decision: {decision.iloc[0]['v80b_decision']}\n"
    f"- Video rows: {len(df)}\n"
    f"- Mapping resolved videos: {mapping_resolved}\n"
    f"- Mapping unresolved videos: {mapping_unresolved}\n"
    f"- Videos with matched behaviour clips: {with_clips}\n"
    f"- Videos without matched behaviour clips: {without_clips}\n"
    f"- Videos in 36-video tracking subset: {tracking_subset}\n"
    f"- Inconsistency rows: {len(inc)}\n"
    f"- Ready for v80c JSON export: {bool(decision.iloc[0]['ready_for_v80c_json_export'])}\n\n"
    "This table documents mapping and annotation status for all 84 videos. "
    "It does not claim that all 84 videos have completed behaviour annotations.\n",
    encoding="utf-8",
)

print(decision.to_string(index=False))
print("=== summary ===")
print(summary.to_string(index=False))
