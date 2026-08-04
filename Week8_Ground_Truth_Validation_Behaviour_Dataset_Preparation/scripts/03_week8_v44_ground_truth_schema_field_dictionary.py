from pathlib import Path
from datetime import datetime
import json
import csv
import hashlib
import pandas as pd
import numpy as np


ROOT = Path.home() / "PigBench"
W8 = ROOT / "Week8_Ground_Truth_Validation_Behaviour_Dataset_Preparation"

LOCKED_JSON = W8 / "outputs" / "v43_setup_input_audit" / "week8_v43c_final_locked_inputs.json"

OUT = W8 / "outputs" / "v44_ground_truth_schema"
OUT.mkdir(parents=True, exist_ok=True)

DOCS = W8 / "docs"
DOCS.mkdir(parents=True, exist_ok=True)

NOTES = W8 / "notes"
NOTES.mkdir(parents=True, exist_ok=True)

REPORTS = W8 / "reports"
REPORTS.mkdir(parents=True, exist_ok=True)

PROGRESS = W8 / "progress"
PROGRESS.mkdir(parents=True, exist_ok=True)

OUT_SCHEMA = OUT / "week8_v44_ground_truth_json_schema_v1.json"
OUT_EXAMPLE = OUT / "week8_v44_example_annotation_record.json"
OUT_FIELD_DICT = OUT / "week8_v44_field_dictionary.csv"
OUT_BEHAVIOUR_DICT = OUT / "week8_v44_behaviour_label_dictionary.csv"
OUT_COLOUR_DICT = OUT / "week8_v44_colour_identity_dictionary.csv"
OUT_SOURCE_COLUMNS = OUT / "week8_v44_source_column_inventory.csv"
OUT_DECISION = OUT / "week8_v44_decision_summary.csv"
OUT_ISSUES = OUT / "week8_v44_issues.csv"
OUT_REPORT = REPORTS / "week8_v44_ground_truth_schema_report.md"
OUT_DOC_SCHEMA = DOCS / "week8_ground_truth_json_format_specification_v44.md"
OUT_DOC_RULES = DOCS / "week8_annotation_rules_v44.md"
OUT_NOTE = NOTES / "week8_v44_ground_truth_schema_notes.md"
OUT_PROGRESS = PROGRESS / "week8_experiment_progress_log.csv"


def safe_to_csv(df, path):
    df.to_csv(
        path,
        index=False,
        quoting=csv.QUOTE_ALL,
        escapechar="\\",
        lineterminator="\n",
    )


def clean_value(x):
    if pd.isna(x):
        return None
    if isinstance(x, np.integer):
        return int(x)
    if isinstance(x, np.floating):
        return float(x)
    if isinstance(x, np.bool_):
        return bool(x)
    return x


def clean_record(d):
    return {k: clean_value(v) for k, v in d.items()}


def read_csv(path):
    return pd.read_csv(path)


def read_locked():
    return json.loads(Path(LOCKED_JSON).read_text())


def path_for(locked, key):
    return Path(locked["locked_inputs"][key]["path"])


def unique_nonempty(df, col):
    if col not in df.columns:
        return []
    vals = []
    for x in df[col].dropna().astype(str).tolist():
        x = x.strip()
        if x and x.lower() not in ["nan", "none", "null"]:
            vals.append(x)
    return sorted(set(vals))


def count_nonempty(df, col):
    if col not in df.columns:
        return {}
    s = df[col].dropna().astype(str).str.strip()
    s = s[s.ne("") & ~s.str.lower().isin(["nan", "none", "null"])]
    return s.value_counts().to_dict()


def infer_label_columns(df):
    return sorted([c for c in df.columns if str(c).startswith("label__")])


def first_nonempty(df, col, default=None):
    if col not in df.columns:
        return default
    s = df[col].dropna().astype(str)
    s = s[s.str.strip().ne("")]
    if len(s) == 0:
        return default
    return s.iloc[0]


def markdown_table(df, max_rows=80):
    if df is None or len(df) == 0:
        return "_No rows._"
    d = df.head(max_rows)
    cols = list(d.columns)
    out = []
    out.append("| " + " | ".join(cols) + " |")
    out.append("| " + " | ".join(["---"] * len(cols)) + " |")
    for _, r in d.iterrows():
        vals = []
        for c in cols:
            x = str(r.get(c, ""))
            x = x.replace("\n", " ").replace("|", "/")
            vals.append(x)
        out.append("| " + " | ".join(vals) + " |")
    return "\n".join(out)


locked = read_locked()

paths = {
    "behaviour_fusion": path_for(locked, "behaviour_fusion_box_level_429"),
    "training_ready": path_for(locked, "training_ready_primary_split_374"),
    "clip_index": path_for(locked, "clip_extraction_index_72"),
    "clip_level": path_for(locked, "clip_level_multilabel_index_72"),
    "corrected_boxes": path_for(locked, "corrected_boxes_box_level_429"),
    "colour_frame_qa": path_for(locked, "final_colour_identity_frame_qa_72"),
    "tracking_rows": path_for(locked, "dense_polygon_tracking_rows_987"),
    "tracklet_arbitration": path_for(locked, "conservative_identity_arbitration_50"),
}

bf = read_csv(paths["behaviour_fusion"])
tr = read_csv(paths["training_ready"])
clips = read_csv(paths["clip_index"])
clip_level = read_csv(paths["clip_level"])
boxes = read_csv(paths["corrected_boxes"])
colour_qa = read_csv(paths["colour_frame_qa"])
tracking = read_csv(paths["tracking_rows"])
arb = read_csv(paths["tracklet_arbitration"])

issues = []

expected_counts = {
    "behaviour_fusion": 429,
    "training_ready": 374,
    "clip_index": 72,
    "clip_level": 72,
    "corrected_boxes": 429,
}

actual_counts = {
    "behaviour_fusion": len(bf),
    "training_ready": len(tr),
    "clip_index": len(clips),
    "clip_level": len(clip_level),
    "corrected_boxes": len(boxes),
}

for k, expected in expected_counts.items():
    if actual_counts[k] != expected:
        issues.append({
            "item": k,
            "issue_type": "hard_unexpected_input_row_count",
            "issue_detail": f"expected {expected}, got {actual_counts[k]}",
        })


# Behaviour dictionary
behaviour_codes = set(unique_nonempty(bf, "behaviour_code"))
behaviour_codes.update(unique_nonempty(tr, "behaviour_code"))

for c in infer_label_columns(clip_level):
    behaviour_codes.add(c.replace("label__", ""))

behaviour_codes = sorted(behaviour_codes)

bf_counts = count_nonempty(bf, "behaviour_code")
tr_counts = count_nonempty(tr, "behaviour_code")

clip_counts = {}
for c in infer_label_columns(clip_level):
    code = c.replace("label__", "")
    vals = pd.to_numeric(clip_level[c], errors="coerce").fillna(0)
    clip_counts[code] = int((vals > 0).sum())

behaviour_rows = []
for code in behaviour_codes:
    pig_count = int(tr_counts.get(code, 0))
    box_count = int(bf_counts.get(code, 0))
    clip_count = int(clip_counts.get(code, 0))

    if pig_count >= 30:
        recommended_use = "baseline_ready"
    elif pig_count >= 10:
        recommended_use = "limited_use"
    else:
        recommended_use = "report_only_rare"

    behaviour_rows.append({
        "behaviour_code": code,
        "behaviour_label": code,
        "box_level_count_429_source": box_count,
        "training_ready_count_374_source": pig_count,
        "clip_level_positive_count_72_source": clip_count,
        "recommended_use": recommended_use,
        "definition_note": "Behaviour code from Unibo annotation. Full semantic definition should follow the project annotation guide / Excel legend.",
    })

behaviour_dict = pd.DataFrame(behaviour_rows).sort_values(
    ["recommended_use", "training_ready_count_374_source", "behaviour_code"],
    ascending=[True, False, True],
)


# Colour dictionary
valid_visual_colours = ["blue", "green", "cyan", "red", "pink", "purple"]
status_values = ["unknown", "not_visible", "uncertain", "unassigned"]

colour_to_behaviour_pig = {
    "blue": "blue",
    "green": "green",
    "cyan": "no_color",
    "red": "red_neck",
    "pink": "red_tail",
    "purple": "purple",
}

colour_rows = []

for colour in valid_visual_colours:
    if "visual_marker_colour_v18c" in bf.columns:
        box_count = int((bf["visual_marker_colour_v18c"].astype(str) == colour).sum())
    else:
        box_count = ""

    if "visual_marker_colour_v18c" in tr.columns:
        train_count = int((tr["visual_marker_colour_v18c"].astype(str) == colour).sum())
    else:
        train_count = ""

    colour_rows.append({
        "colour_identity": colour,
        "type": "valid_visual_marker_colour",
        "behaviour_pig_id_crosswalk": colour_to_behaviour_pig.get(colour, ""),
        "box_level_count_429_source": box_count,
        "training_ready_count_374_source": train_count,
        "annotation_rule": "Valid colour-marker identity used for pig identity association.",
    })

for status in status_values:
    if "final_identity_status_v17" in bf.columns:
        box_count = int(bf["final_identity_status_v17"].astype(str).str.contains(status, case=False, na=False).sum())
    else:
        box_count = ""

    if "final_identity_status_v17" in tr.columns:
        train_count = int(tr["final_identity_status_v17"].astype(str).str.contains(status, case=False, na=False).sum())
    else:
        train_count = ""

    colour_rows.append({
        "colour_identity": status,
        "type": "identity_status_or_missing_value",
        "behaviour_pig_id_crosswalk": "",
        "box_level_count_429_source": box_count,
        "training_ready_count_374_source": train_count,
        "annotation_rule": "Not a valid identity for training unless explicitly reviewed; preserved for validation and issue tracking.",
    })

colour_dict = pd.DataFrame(colour_rows)


# Source column inventory
source_inventory_rows = []
sources = [
    ("behaviour_fusion_box_level_429", bf),
    ("training_ready_primary_split_374", tr),
    ("clip_extraction_index_72", clips),
    ("clip_level_multilabel_index_72", clip_level),
    ("corrected_boxes_box_level_429", boxes),
    ("colour_identity_frame_qa_72", colour_qa),
    ("dense_polygon_tracking_rows_987", tracking),
    ("conservative_identity_arbitration_50", arb),
]

for source_name, df in sources:
    for c in df.columns:
        source_inventory_rows.append({
            "source_table": source_name,
            "column_name": c,
            "non_null_count": int(df[c].notna().sum()),
            "dtype": str(df[c].dtype),
            "example_value": str(first_nonempty(df, c, ""))[:250],
        })

source_inventory = pd.DataFrame(source_inventory_rows)


# Field dictionary
field_rows = [
    {
        "level": "dataset",
        "field_name": "dataset_version",
        "type": "string",
        "required": True,
        "source": "generated",
        "description": "Version identifier for the Week 8 ground-truth dataset.",
        "example": "week8_v1",
    },
    {
        "level": "dataset",
        "field_name": "created_at",
        "type": "string_datetime",
        "required": True,
        "source": "generated",
        "description": "Timestamp when the JSON annotation file was generated.",
        "example": datetime.now().isoformat(timespec="seconds"),
    },
    {
        "level": "dataset",
        "field_name": "annotation_protocol",
        "type": "object",
        "required": True,
        "source": "Week 8 assignment rule",
        "description": "Defines that behaviour labels are propagated across the full annotated 10-second observation interval.",
        "example": "10_second_observation_window",
    },
    {
        "level": "clip",
        "field_name": "scan_frame_id",
        "type": "string",
        "required": True,
        "source": "clip_extraction_index_72.scan_frame_id",
        "description": "Unique scanpoint identifier used as the annotation anchor.",
        "example": first_nonempty(clips, "scan_frame_id", "scanframe_0000"),
    },
    {
        "level": "clip",
        "field_name": "video_id",
        "type": "string",
        "required": True,
        "source": "clip_extraction_index_72.video_id",
        "description": "Identifier of the source video.",
        "example": first_nonempty(clips, "video_id", ""),
    },
    {
        "level": "clip",
        "field_name": "source_video_path",
        "type": "string_path",
        "required": True,
        "source": "clip_extraction_index_72.source_video_path",
        "description": "Path to the original Unibo video.",
        "example": first_nonempty(clips, "source_video_path", ""),
    },
    {
        "level": "clip",
        "field_name": "clip_path",
        "type": "string_path",
        "required": True,
        "source": "clip_extraction_index_72.clip_path",
        "description": "Path to the extracted 10-second clip.",
        "example": first_nonempty(clips, "clip_path", ""),
    },
    {
        "level": "clip",
        "field_name": "start_sec",
        "type": "float",
        "required": True,
        "source": "clip_extraction_index_72.start_sec",
        "description": "Start time of the annotated observation interval within the source video.",
        "example": first_nonempty(clips, "start_sec", 0.0),
    },
    {
        "level": "clip",
        "field_name": "end_sec",
        "type": "float",
        "required": True,
        "source": "clip_extraction_index_72.end_sec",
        "description": "End time of the annotated observation interval within the source video.",
        "example": first_nonempty(clips, "end_sec", 10.0),
    },
    {
        "level": "clip",
        "field_name": "duration_sec",
        "type": "float",
        "required": True,
        "source": "clip_extraction_index_72.duration_sec",
        "description": "Duration of the annotated observation interval.",
        "example": first_nonempty(clips, "duration_sec", 10.0),
    },
    {
        "level": "clip",
        "field_name": "fps_used",
        "type": "float",
        "required": True,
        "source": "clip_extraction_index_72.fps_used",
        "description": "FPS used to convert timestamps and frame indices.",
        "example": first_nonempty(clips, "fps_used", 25.0),
    },
    {
        "level": "frame",
        "field_name": "frame_index_in_clip",
        "type": "integer",
        "required": True,
        "source": "generated from video frames",
        "description": "Frame index relative to the extracted 10-second clip.",
        "example": 0,
    },
    {
        "level": "frame",
        "field_name": "source_frame_index",
        "type": "integer",
        "required": True,
        "source": "clip start frame + frame index",
        "description": "Estimated absolute frame index in the source video.",
        "example": int(float(first_nonempty(clips, "start_frame_adjusted", 0))),
    },
    {
        "level": "frame",
        "field_name": "timestamp_sec",
        "type": "float",
        "required": True,
        "source": "start_sec + frame_index_in_clip / fps_used",
        "description": "Timestamp of the frame in source-video time.",
        "example": float(first_nonempty(clips, "start_sec", 0.0)),
    },
    {
        "level": "object",
        "field_name": "final_box_id",
        "type": "string",
        "required": True,
        "source": "behaviour_fusion_box_level_429.final_box_id",
        "description": "Final corrected pig box identifier.",
        "example": first_nonempty(bf, "final_box_id", ""),
    },
    {
        "level": "object",
        "field_name": "bbox_xyxy",
        "type": "array[float]",
        "required": True,
        "source": "behaviour_fusion_box_level_429.x1/y1/x2/y2 or propagated/tracked bbox",
        "description": "Bounding box in [x1, y1, x2, y2] pixel coordinates.",
        "example": "[x1, y1, x2, y2]",
    },
    {
        "level": "object",
        "field_name": "visual_marker_colour",
        "type": "string",
        "required": True,
        "source": "behaviour_fusion_box_level_429.visual_marker_colour_v18c",
        "description": "Visual marker colour assigned to the pig.",
        "example": first_nonempty(bf, "visual_marker_colour_v18c", ""),
    },
    {
        "level": "object",
        "field_name": "behaviour_pig_id",
        "type": "string",
        "required": True,
        "source": "behaviour_fusion_box_level_429.behaviour_pig_id_v18c",
        "description": "Pig identity used in the behaviour annotation file after colour crosswalk.",
        "example": first_nonempty(bf, "behaviour_pig_id_v18c", ""),
    },
    {
        "level": "object",
        "field_name": "behaviour_code",
        "type": "string",
        "required": True,
        "source": "behaviour_fusion_box_level_429.behaviour_code",
        "description": "Behaviour code assigned to this pig in the annotated observation interval.",
        "example": first_nonempty(bf, "behaviour_code", ""),
    },
    {
        "level": "object",
        "field_name": "behaviour_label",
        "type": "string",
        "required": False,
        "source": "behaviour_fusion_box_level_429.behaviour_label",
        "description": "Human-readable behaviour label if available. In the current source it may be same as behaviour_code.",
        "example": first_nonempty(bf, "behaviour_label", ""),
    },
    {
        "level": "object",
        "field_name": "label_source",
        "type": "string",
        "required": True,
        "source": "generated",
        "description": "Describes how the behaviour label was assigned to the frame/object.",
        "example": "propagated_from_10_second_observation_window",
    },
    {
        "level": "object",
        "field_name": "identity_status",
        "type": "string",
        "required": True,
        "source": "behaviour_fusion_box_level_429.final_identity_status_v17 / arbitration output",
        "description": "Identity confidence/status such as accepted, candidate, unknown, not_visible, uncertain or review_required.",
        "example": first_nonempty(bf, "final_identity_status_v17", ""),
    },
    {
        "level": "object",
        "field_name": "validation_status",
        "type": "string",
        "required": True,
        "source": "generated / validation interface",
        "description": "Manual validation status for the object annotation.",
        "example": "unchecked",
    },
    {
        "level": "object",
        "field_name": "validation_flags",
        "type": "array[string]",
        "required": False,
        "source": "generated / validation interface",
        "description": "Issue flags such as colour_error, behaviour_error, identity_switch, missing_label, occlusion, uncertain.",
        "example": "[]",
    },
]

field_dict = pd.DataFrame(field_rows)


valid_behaviours = behaviour_codes
valid_colours_plus_status = valid_visual_colours + status_values

schema = {
    "$schema": "https://json-schema.org/draft/2020-12/schema",
    "title": "Week 8 Unibo Behaviour Ground-Truth Annotation Schema",
    "type": "object",
    "required": ["dataset_version", "created_at", "annotation_protocol", "clips"],
    "properties": {
        "dataset_version": {"type": "string"},
        "created_at": {"type": "string"},
        "source_project": {"type": "string"},
        "annotation_protocol": {
            "type": "object",
            "required": ["observation_window_sec", "behaviour_label_scope", "propagation_rule"],
            "properties": {
                "observation_window_sec": {"type": "number"},
                "behaviour_label_scope": {
                    "type": "string",
                    "enum": ["full_annotated_clip_interval"],
                },
                "propagation_rule": {
                    "type": "string",
                    "enum": ["propagate_pig_behaviour_to_every_frame_in_interval"],
                },
                "bbox_rule": {"type": "string"},
                "identity_rule": {"type": "string"},
                "validation_rule": {"type": "string"},
            },
        },
        "valid_behaviour_codes": {
            "type": "array",
            "items": {"type": "string", "enum": valid_behaviours},
        },
        "valid_colour_identities": {
            "type": "array",
            "items": {"type": "string", "enum": valid_colours_plus_status},
        },
        "clips": {
            "type": "array",
            "items": {
                "type": "object",
                "required": [
                    "scan_frame_id",
                    "video_id",
                    "clip_path",
                    "start_sec",
                    "end_sec",
                    "duration_sec",
                    "fps_used",
                    "objects",
                ],
                "properties": {
                    "scan_frame_id": {"type": "string"},
                    "video_id": {"type": "string"},
                    "source_video_path": {"type": "string"},
                    "clip_path": {"type": "string"},
                    "preview_frame_path": {"type": "string"},
                    "start_sec": {"type": "number"},
                    "end_sec": {"type": "number"},
                    "duration_sec": {"type": "number"},
                    "fps_used": {"type": "number"},
                    "center_frame_index": {"type": ["integer", "number", "null"]},
                    "split": {"type": ["string", "null"], "enum": ["train", "val", "test", "unmapped", None]},
                    "behaviour_set": {"type": ["string", "null"]},
                    "objects": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "required": [
                                "final_box_id",
                                "visual_marker_colour",
                                "behaviour_pig_id",
                                "behaviour_code",
                                "bbox_xyxy",
                                "label_source",
                                "validation_status",
                            ],
                            "properties": {
                                "final_box_id": {"type": "string"},
                                "track_id": {"type": ["string", "integer", "null"]},
                                "visual_marker_colour": {"type": "string", "enum": valid_colours_plus_status},
                                "behaviour_pig_id": {"type": ["string", "null"]},
                                "behaviour_code": {"type": "string", "enum": valid_behaviours},
                                "behaviour_label": {"type": ["string", "null"]},
                                "bbox_xyxy": {
                                    "type": "array",
                                    "minItems": 4,
                                    "maxItems": 4,
                                    "items": {"type": "number"},
                                },
                                "identity_status": {"type": ["string", "null"]},
                                "identity_confidence": {"type": ["string", "null"]},
                                "label_source": {
                                    "type": "string",
                                    "enum": ["propagated_from_10_second_observation_window"],
                                },
                                "validation_status": {
                                    "type": "string",
                                    "enum": ["unchecked", "accepted", "rejected", "review_required"],
                                },
                                "validation_flags": {
                                    "type": "array",
                                    "items": {"type": "string"},
                                },
                            },
                        },
                    },
                    "frames": {
                        "type": "array",
                        "description": "Optional full frame-level expansion produced in v45.",
                        "items": {
                            "type": "object",
                            "required": [
                                "frame_index_in_clip",
                                "source_frame_index",
                                "timestamp_sec",
                                "objects",
                            ],
                            "properties": {
                                "frame_index_in_clip": {"type": "integer"},
                                "source_frame_index": {"type": "integer"},
                                "timestamp_sec": {"type": "number"},
                                "objects": {"type": "array"},
                            },
                        },
                    },
                },
            },
        },
    },
}


# Example annotation record
first_clip = clean_record(clips.iloc[0].to_dict())
scan_id = first_clip.get("scan_frame_id")
bf_scan = bf[bf["scan_frame_id"].astype(str) == str(scan_id)].head(2)

objects = []
for _, r in bf_scan.iterrows():
    rr = clean_record(r.to_dict())
    objects.append({
        "final_box_id": str(rr.get("final_box_id")),
        "track_id": None,
        "visual_marker_colour": rr.get("visual_marker_colour_v18c") or rr.get("final_colour_identity_v17") or "unknown",
        "behaviour_pig_id": rr.get("behaviour_pig_id_v18c"),
        "behaviour_code": rr.get("behaviour_code"),
        "behaviour_label": rr.get("behaviour_label"),
        "bbox_xyxy": [
            float(rr.get("x1")),
            float(rr.get("y1")),
            float(rr.get("x2")),
            float(rr.get("y2")),
        ],
        "identity_status": rr.get("final_identity_status_v17"),
        "identity_confidence": "scanpoint_locked_or_reviewed",
        "label_source": "propagated_from_10_second_observation_window",
        "validation_status": "unchecked",
        "validation_flags": [],
    })

example = {
    "dataset_version": "week8_v1",
    "created_at": datetime.now().isoformat(timespec="seconds"),
    "source_project": "PigBench / Unibo Week 8 Ground Truth Validation",
    "annotation_protocol": {
        "observation_window_sec": 10,
        "behaviour_label_scope": "full_annotated_clip_interval",
        "propagation_rule": "propagate_pig_behaviour_to_every_frame_in_interval",
        "bbox_rule": "Use corrected scanpoint boxes as identity anchors. v45 may expand frame-level boxes using tracking where reliable.",
        "identity_rule": "Use colour identity and conservative tracklet evidence. Uncertain identities remain review_required or unknown.",
        "validation_rule": "All generated annotations start as unchecked and are validated through the visualization interface.",
    },
    "valid_behaviour_codes": valid_behaviours,
    "valid_colour_identities": valid_colours_plus_status,
    "clips": [
        {
            "scan_frame_id": first_clip.get("scan_frame_id"),
            "video_id": first_clip.get("video_id"),
            "source_video_path": first_clip.get("source_video_path"),
            "clip_path": first_clip.get("clip_path"),
            "preview_frame_path": first_clip.get("preview_frame_path"),
            "start_sec": clean_value(first_clip.get("start_sec")),
            "end_sec": clean_value(first_clip.get("end_sec")),
            "duration_sec": clean_value(first_clip.get("duration_sec")),
            "fps_used": clean_value(first_clip.get("fps_used")),
            "center_frame_index": clean_value(first_clip.get("center_frame_index")),
            "split": None,
            "behaviour_set": first_clip.get("behaviour_codes_present"),
            "objects": objects,
            "frames": [
                {
                    "frame_index_in_clip": 0,
                    "source_frame_index": clean_value(first_clip.get("start_frame_adjusted")),
                    "timestamp_sec": clean_value(first_clip.get("start_sec")),
                    "objects": objects,
                }
            ],
        }
    ],
}


# Schema documentation
schema_doc = f"""# Week 8 Ground-Truth JSON Format Specification v44

## Purpose

This document defines the JSON annotation format for the Week 8 Unibo behaviour ground-truth dataset.

The dataset is designed to associate each pig with tracked pig, colour identity, and behaviour label across the annotated 10-second observation interval.

## Annotation protocol

The Week 8 protocol treats each behavioural observation as a 10-second annotated clip interval.

Behaviour labels apply to the full annotated clip interval.

Every frame belonging to the annotated interval inherits the corresponding pig-level behaviour label.

This propagation rule is used for visualization, validation, and future clip-based behaviour classification experiments.

## Main JSON hierarchy

dataset
  clips
    objects
    frames
      objects

## Required top-level fields

{markdown_table(field_dict[field_dict["level"].isin(["dataset"])])}

## Clip-level fields

{markdown_table(field_dict[field_dict["level"].isin(["clip"])])}

## Frame-level fields

{markdown_table(field_dict[field_dict["level"].isin(["frame"])])}

## Object-level fields

{markdown_table(field_dict[field_dict["level"].isin(["object"])])}

## Behaviour codes

{markdown_table(behaviour_dict)}

## Colour identities

{markdown_table(colour_dict)}

## BBox convention

Bounding boxes use pixel coordinates in xyxy format:

bbox_xyxy = [x1, y1, x2, y2]

where x1, y1 is the top-left corner and x2, y2 is the bottom-right corner.

## Timestamp convention

For a frame inside a clip:

timestamp_sec = clip.start_sec + frame_index_in_clip / fps_used

The corresponding source-video frame index is estimated from the clip start frame and the frame index inside the clip.

## Validation status

All generated annotations start as unchecked.

The visualization interface may update this to accepted, rejected, or review_required.

## Claim scope

This schema defines the ground-truth validation dataset format. It is not a final behaviour classifier and does not claim production-grade tracking.
"""

rules_doc = """# Week 8 Annotation Rules v44

## 1. Behaviour label propagation

The behaviour observation corresponds to a 10-second annotated observation window.

Therefore, the pig-level behaviour label from the annotation is propagated to every frame inside the corresponding annotated clip interval.

## 2. Identity rule

Pig identity is represented through the colour-marker identity and its behaviour-annotation pig ID crosswalk.

Valid visual marker colours:

blue, green, cyan, red, pink, purple

Verified crosswalk:

blue -> blue
green -> green
cyan -> no_color
red -> red_neck
pink -> red_tail
purple -> purple

## 3. Unknown / uncertain identities

Unknown, not-visible, uncertain, or unassigned identities must not be silently forced into a valid identity.

They must be preserved as validation states or review-required cases.

## 4. Bounding boxes

Bounding boxes use [x1, y1, x2, y2] pixel coordinates.

The scanpoint-level corrected boxes are the geometry anchor. Frame-level propagation may later use tracking boxes where reliable tracking evidence is available.

## 5. Validation flags

Supported issue flags include:

colour_error
behaviour_error
identity_switch
missing_label
wrong_bbox
occlusion
uncertain
temporal_inconsistency
review_required

## 6. Dataset claim scope

The Week 8 dataset is a validated and inspectable ground-truth preparation dataset.

It is not a final behaviour classifier and not a production-grade multi-object tracking system.
"""


if len(valid_behaviours) == 0:
    issues.append({
        "item": "behaviour_dictionary",
        "issue_type": "hard_empty_behaviour_dictionary",
        "issue_detail": "No behaviour codes were inferred.",
    })

missing_required_colours = [c for c in valid_visual_colours if c not in colour_dict["colour_identity"].tolist()]
if missing_required_colours:
    issues.append({
        "item": "colour_dictionary",
        "issue_type": "hard_missing_valid_colours",
        "issue_detail": "Missing: " + ", ".join(missing_required_colours),
    })

if len(objects) == 0:
    issues.append({
        "item": "example_annotation_record",
        "issue_type": "warning_empty_example_objects",
        "issue_detail": f"No behaviour-fusion rows found for first scan_frame_id={scan_id}",
    })


issues_df = pd.DataFrame(issues, columns=["item", "issue_type", "issue_detail"])
hard_issues = [x for x in issues if str(x["issue_type"]).startswith("hard_")]
warnings = [x for x in issues if str(x["issue_type"]).startswith("warning_")]
ready_for_v45 = len(hard_issues) == 0

decision = pd.DataFrame([{
    "v44_decision": "ground_truth_schema_created" if ready_for_v45 else "ground_truth_schema_blocked",
    "behaviour_code_count": int(len(valid_behaviours)),
    "valid_visual_colour_count": int(len(valid_visual_colours)),
    "field_dictionary_rows": int(len(field_dict)),
    "source_column_inventory_rows": int(len(source_inventory)),
    "example_clip_id": str(scan_id),
    "example_object_count": int(len(objects)),
    "behaviour_fusion_rows": int(len(bf)),
    "training_ready_rows": int(len(tr)),
    "clip_index_rows": int(len(clips)),
    "clip_level_index_rows": int(len(clip_level)),
    "hard_issue_count": int(len(hard_issues)),
    "warning_count": int(len(warnings)),
    "issue_count": int(len(issues_df)),
    "ready_for_v45_label_propagation": bool(ready_for_v45),
    "generated_at": datetime.now().isoformat(timespec="seconds"),
}])

OUT_SCHEMA.write_text(json.dumps(schema, indent=2))
OUT_EXAMPLE.write_text(json.dumps(example, indent=2))
safe_to_csv(field_dict, OUT_FIELD_DICT)
safe_to_csv(behaviour_dict, OUT_BEHAVIOUR_DICT)
safe_to_csv(colour_dict, OUT_COLOUR_DICT)
safe_to_csv(source_inventory, OUT_SOURCE_COLUMNS)
safe_to_csv(decision, OUT_DECISION)
safe_to_csv(issues_df, OUT_ISSUES)
OUT_DOC_SCHEMA.write_text(schema_doc)
OUT_DOC_RULES.write_text(rules_doc)

report = f"""# Week 8 v44 Ground-Truth Schema Report

## Decision

- v44 decision: {decision.iloc[0]["v44_decision"]}
- Behaviour code count: {len(valid_behaviours)}
- Valid visual colour count: {len(valid_visual_colours)}
- Field dictionary rows: {len(field_dict)}
- Source column inventory rows: {len(source_inventory)}
- Hard issue count: {len(hard_issues)}
- Warning count: {len(warnings)}
- Ready for v45 label propagation: {ready_for_v45}

## Core input counts

- Behaviour fusion box-level rows: {len(bf)}
- Training-ready rows: {len(tr)}
- Clip index rows: {len(clips)}
- Clip-level index rows: {len(clip_level)}
- Corrected boxes rows: {len(boxes)}

## Main outputs

- JSON schema: {OUT_SCHEMA}
- Example annotation record: {OUT_EXAMPLE}
- Field dictionary: {OUT_FIELD_DICT}
- Behaviour label dictionary: {OUT_BEHAVIOUR_DICT}
- Colour identity dictionary: {OUT_COLOUR_DICT}
- Format documentation: {OUT_DOC_SCHEMA}
- Annotation rules: {OUT_DOC_RULES}

## Next step

v45 should generate the actual propagated ground-truth annotations across the full 10-second clip intervals.
"""

OUT_REPORT.write_text(report)

OUT_NOTE.write_text(
    "# Week 8 v44 Ground-Truth Schema\n\n"
    "## Summary\n\n"
    f"- v44 decision: {decision.iloc[0]['v44_decision']}\n"
    f"- Behaviour code count: {len(valid_behaviours)}\n"
    f"- Field dictionary rows: {len(field_dict)}\n"
    f"- Hard issue count: {len(hard_issues)}\n"
    f"- Warning count: {len(warnings)}\n"
    f"- Ready for v45 label propagation: {ready_for_v45}\n\n"
    "## Key rule\n\n"
    "Behaviour labels are propagated across the full annotated 10-second observation interval.\n\n"
    "## Outputs\n\n"
    f"- JSON schema: {OUT_SCHEMA}\n"
    f"- Example annotation: {OUT_EXAMPLE}\n"
    f"- Field dictionary: {OUT_FIELD_DICT}\n"
    f"- Behaviour dictionary: {OUT_BEHAVIOUR_DICT}\n"
    f"- Colour dictionary: {OUT_COLOUR_DICT}\n"
    f"- Format documentation: {OUT_DOC_SCHEMA}\n"
    f"- Annotation rules: {OUT_DOC_RULES}\n"
)

progress_row = pd.DataFrame([{
    "date": datetime.now().date().isoformat(),
    "stage": "v44",
    "task_name": "Ground-truth JSON schema and field dictionary",
    "status": "PASS" if ready_for_v45 else "BLOCKED",
    "input_summary": str(LOCKED_JSON),
    "output_summary": str(OUT),
    "hard_issues": int(len(hard_issues)),
    "warnings": int(len(warnings)),
    "next_action": "v45 behaviour label propagation across 10-second clips" if ready_for_v45 else "Resolve schema hard issues.",
}])

if OUT_PROGRESS.exists():
    old = pd.read_csv(OUT_PROGRESS)
    progress = pd.concat([old, progress_row], ignore_index=True)
else:
    progress = progress_row

safe_to_csv(progress, OUT_PROGRESS)

print("Saved:")
print(OUT_SCHEMA)
print(OUT_EXAMPLE)
print(OUT_FIELD_DICT)
print(OUT_BEHAVIOUR_DICT)
print(OUT_COLOUR_DICT)
print(OUT_SOURCE_COLUMNS)
print(OUT_DECISION)
print(OUT_ISSUES)
print(OUT_REPORT)
print(OUT_DOC_SCHEMA)
print(OUT_DOC_RULES)
print(OUT_NOTE)

print()
print("=== v44 decision ===")
print(decision.to_string(index=False))

print()
print("=== behaviour dictionary ===")
print(behaviour_dict.to_string(index=False))

print()
print("=== colour dictionary ===")
print(colour_dict.to_string(index=False))

print()
print("=== v44 issues ===")
if len(issues_df):
    print(issues_df.to_string(index=False))
else:
    print("No issues found.")
