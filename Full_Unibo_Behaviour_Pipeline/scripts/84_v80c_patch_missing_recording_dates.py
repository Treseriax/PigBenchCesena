from pathlib import Path
from datetime import datetime
import json
import pandas as pd

F = Path.home() / "PigBench" / "Full_Unibo_Behaviour_Pipeline"
O = F / "outputs/v80_final_project_completion/01_dataset_json_export"
JSON_DIR = O / "json_annotations_all_84_videos"

status_path = O / "v80b_final_video_mapping_status_table.csv"
manifest_path = O / "v80c_json_export_manifest.csv"

status = pd.read_csv(status_path).fillna("")
manifest = pd.read_csv(manifest_path).fillna("")

# These are friendly-named TLC1/B1 videos whose date was missing from filename metadata.
# They belong to the same validated 2021-07-22 Unibo mapping context.
patch_date = "2021-07-22"
patched_rows = []

missing_date_mask = status["recording_date"].astype(str).str.strip() == ""
missing = status[missing_date_mask].copy()

for idx, row in missing.iterrows():
    video_id = str(row["video_id"])
    filename = str(row["video_filename"])

    status.loc[idx, "recording_date"] = patch_date

    old_note = str(status.loc[idx, "inconsistency_notes"])
    add_note = "recording_date_inferred_as_2021-07-22_for_friendly_named_video"
    if old_note and old_note != "ok":
        new_note = old_note + ";" + add_note
    else:
        new_note = add_note
    status.loc[idx, "inconsistency_notes"] = new_note

    mrow = manifest[manifest["video_id"].astype(str) == video_id]
    if len(mrow) == 0:
        patched_rows.append(
            {
                "video_id": video_id,
                "video_filename": filename,
                "patch_status": "manifest_row_missing",
                "patched_recording_date": patch_date,
            }
        )
        continue

    json_path = Path(mrow.iloc[0]["json_path"])

    if not json_path.exists():
        patched_rows.append(
            {
                "video_id": video_id,
                "video_filename": filename,
                "patch_status": "json_file_missing",
                "patched_recording_date": patch_date,
            }
        )
        continue

    data = json.loads(json_path.read_text(encoding="utf-8"))
    data["video"]["recording_date"] = patch_date

    old_json_note = str(data["video"].get("inconsistency_notes", ""))
    if old_json_note and old_json_note != "ok":
        data["video"]["inconsistency_notes"] = old_json_note + ";" + add_note
    else:
        data["video"]["inconsistency_notes"] = add_note

    data["date_patch"] = {
        "patched_at": datetime.now().isoformat(timespec="seconds"),
        "patched_recording_date": patch_date,
        "reason": "friendly_named_video_missing_filename_date",
        "claim_boundary": "recording_date_inferred_for_metadata_consistency_not_new_behaviour_label",
    }

    json_path.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")

    patched_rows.append(
        {
            "video_id": video_id,
            "video_filename": filename,
            "patch_status": "patched",
            "patched_recording_date": patch_date,
            "json_path": str(json_path),
        }
    )

status.to_csv(status_path, index=False)
pd.DataFrame(patched_rows).to_csv(O / "v80c_missing_recording_date_patch_report.csv", index=False)

print("missing date rows patched", len(patched_rows))
print(pd.DataFrame(patched_rows).to_string(index=False))
