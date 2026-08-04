from pathlib import Path
from datetime import datetime
import json
import pandas as pd

F = Path.home() / "PigBench" / "Full_Unibo_Behaviour_Pipeline"
O = F / "outputs/v80_final_project_completion/01_dataset_json_export"
JSON_DIR = O / "json_annotations_all_84_videos"

manifest = pd.read_csv(O / "v80c_json_export_manifest.csv").fillna("")

required_top = ["schema_version", "generated_at", "video", "annotation_file_type", "annotations", "summary", "claim_boundary"]
required_video = [
    "video_id",
    "video_filename",
    "camera_identifier",
    "pen_identifier",
    "recording_date",
    "recording_time_interval",
    "mapping_status",
    "annotation_status",
]
required_ann = [
    "clip_id",
    "annotation_window_id",
    "window_id",
    "start_sec",
    "end_sec",
    "duration_sec",
    "behaviour_label",
    "identity_colour",
    "candidate_tracklets",
    "bbox_source",
]

rows = []

for _, r in manifest.iterrows():
    p = Path(r["json_path"])
    errors = []

    if not p.exists():
        errors.append("json_file_missing")
        rows.append(
            {
                "video_id": r["video_id"],
                "video_filename": r["video_filename"],
                "json_path": str(p),
                "validation_status": "INVALID",
                "error_count": len(errors),
                "errors": ";".join(errors),
            }
        )
        continue

    try:
        data = json.loads(p.read_text(encoding="utf-8"))
    except Exception as e:
        errors.append("json_parse_error:" + str(e))
        data = {}

    for k in required_top:
        if k not in data:
            errors.append("missing_top_field:" + k)

    video = data.get("video", {})
    for k in required_video:
        if k not in video or str(video.get(k, "")) == "":
            errors.append("missing_video_field:" + k)

    annotations = data.get("annotations", [])
    file_type = data.get("annotation_file_type", "")

    if file_type == "behaviour_annotations" and len(annotations) == 0:
        errors.append("label_bearing_file_has_no_annotations")

    if file_type == "metadata_status_only_no_behaviour_labels" and len(annotations) != 0:
        errors.append("metadata_only_file_has_annotations")

    for i, ann in enumerate(annotations):
        for k in required_ann:
            if k not in ann:
                errors.append(f"annotation_{i}_missing_field:{k}")

        try:
            start = float(ann.get("start_sec"))
            end = float(ann.get("end_sec"))
            if end <= start:
                errors.append(f"annotation_{i}_invalid_time_order")
        except Exception:
            errors.append(f"annotation_{i}_invalid_time_values")

        if not str(ann.get("behaviour_label", "")):
            errors.append(f"annotation_{i}_missing_behaviour_label")

        if not str(ann.get("identity_colour", "")):
            errors.append(f"annotation_{i}_missing_identity_colour")

    rows.append(
        {
            "video_id": r["video_id"],
            "video_filename": r["video_filename"],
            "json_path": str(p),
            "annotation_file_type": file_type,
            "annotation_count": len(annotations),
            "validation_status": "VALID" if len(errors) == 0 else "INVALID",
            "error_count": len(errors),
            "errors": ";".join(errors),
        }
    )

report = pd.DataFrame(rows)
report.to_csv(O / "v80c_json_validation_report.csv", index=False)

valid_count = int((report["validation_status"] == "VALID").sum())
invalid_count = int((report["validation_status"] != "VALID").sum())
label_json = int((report["annotation_file_type"] == "behaviour_annotations").sum())
metadata_json = int((report["annotation_file_type"] == "metadata_status_only_no_behaviour_labels").sum())
total_annotations = int(report["annotation_count"].sum())

decision = pd.DataFrame(
    [
        {
            "v80c_decision": "json_export_and_validation_passed"
            if invalid_count == 0
            else "json_validation_has_invalid_files",
            "json_files": len(report),
            "valid_json_files": valid_count,
            "invalid_json_files": invalid_count,
            "label_bearing_json_files": label_json,
            "metadata_only_json_files": metadata_json,
            "total_annotation_records": total_annotations,
            "ready_for_v80d_annotation_visualizer": invalid_count == 0,
            "ready_for_model_training": label_json > 0 and invalid_count == 0,
            "claim_scope": "36_label_bearing_json_plus_48_metadata_status_json_not_full_84_behaviour_completion_claim",
            "generated_at": datetime.now().isoformat(timespec="seconds"),
        }
    ]
)

decision.to_csv(O / "v80c_decision_summary.csv", index=False)

note = F / "notes/v80c_json_export_validation_notes.md"
note.parent.mkdir(parents=True, exist_ok=True)
note.write_text(
    "# v80c JSON Export and Validation\n\n"
    f"- Decision: {decision.iloc[0]['v80c_decision']}\n"
    f"- JSON files: {len(report)}\n"
    f"- Valid JSON files: {valid_count}\n"
    f"- Invalid JSON files: {invalid_count}\n"
    f"- Label-bearing JSON files: {label_json}\n"
    f"- Metadata-only JSON files: {metadata_json}\n"
    f"- Total annotation records: {total_annotations}\n"
    f"- Ready for v80d annotation visualizer: {invalid_count == 0}\n\n"
    "The 36 label-bearing JSON files correspond to the current validated matched behaviour subset. "
    "The remaining 48 JSON files are metadata/status-only files and do not claim completed behaviour labels.\n",
    encoding="utf-8",
)

print(decision.to_string(index=False))
print("=== validation status ===")
print(report["validation_status"].value_counts().to_string())
print("=== file type ===")
print(report["annotation_file_type"].value_counts().to_string())
