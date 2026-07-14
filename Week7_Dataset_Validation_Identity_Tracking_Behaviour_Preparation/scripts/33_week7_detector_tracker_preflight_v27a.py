from pathlib import Path
from datetime import datetime
import csv
import importlib.util
import shutil
import re
import subprocess

import pandas as pd


ROOT = Path.home() / "PigBench"
W7 = ROOT / "Week7_Dataset_Validation_Identity_Tracking_Behaviour_Preparation"

V26_ROOT = W7 / "outputs" / "clip_extraction_temporal_qa_v26"
V26_INDEX = V26_ROOT / "week7_clip_extraction_v26_index.csv"

OUT_ROOT = W7 / "outputs" / "detector_tracker_preflight_v27a"
OUT_ROOT.mkdir(parents=True, exist_ok=True)

OUT_ENV = OUT_ROOT / "week7_detector_tracker_preflight_v27a_environment_audit.csv"
OUT_MODELS = OUT_ROOT / "week7_detector_tracker_preflight_v27a_model_candidates.csv"
OUT_SELECTED = OUT_ROOT / "week7_detector_tracker_preflight_v27a_selected_dryrun_clips.csv"
OUT_PLAN = OUT_ROOT / "week7_detector_tracker_preflight_v27a_dryrun_plan.csv"
OUT_ISSUES = OUT_ROOT / "week7_detector_tracker_preflight_v27a_issues.csv"
OUT_SUMMARY = OUT_ROOT / "week7_detector_tracker_preflight_v27a_summary.csv"
OUT_README = OUT_ROOT / "README_detector_tracker_preflight_v27a.md"
OUT_NOTE = W7 / "notes" / "week7_detector_tracker_preflight_v27a_notes.md"


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


def command_exists(cmd):
    return shutil.which(cmd) is not None


def import_available(module_name):
    try:
        spec = importlib.util.find_spec(module_name)
        return spec is not None
    except Exception:
        return False


def run_cmd(args):
    try:
        proc = subprocess.run(
            args,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            timeout=20,
        )
        return proc.returncode, proc.stdout.strip(), proc.stderr.strip()
    except Exception as e:
        return -1, "", repr(e)


def file_size(path):
    p = Path(path)
    return p.stat().st_size if p.exists() else 0


def slug(s):
    s = clean(s)
    s = re.sub(r"[^A-Za-z0-9_.-]+", "_", s)
    s = re.sub(r"_+", "_", s).strip("_")
    return s or "unknown"


def infer_model_role(path):
    p = str(path).lower()
    if "yolov8" in p:
        return "yolov8_candidate"
    if "rt-detr" in p or "rtdetr" in p:
        return "rtdetr_candidate"
    if "co-dino" in p or "codino" in p:
        return "codino_candidate"
    if "yolox" in p:
        return "yolox_candidate"
    return "other_candidate"


def score_config(path):
    name = str(path).lower()
    score = 0
    if "pig" in name:
        score += 100
    if "yolov8" in name:
        score += 80
    if "yolo" in name:
        score += 40
    if "coco" in name:
        score += 10
    if "s_" in name or "_s" in name:
        score += 8
    if "test" in name:
        score -= 20
    return score


def score_checkpoint(path):
    name = str(path).lower()
    score = 0
    if "pig" in name:
        score += 100
    if "yolov8" in name:
        score += 80
    if "yolov8_s" in name or "yolov8-s" in name or "yolov8_s.pth" in name:
        score += 20
    if "rtdetr" in name or "rt-detr" in name:
        score += 40
    if "codino" in name or "co-dino" in name:
        score += 30
    return score


def discover_configs():
    candidates = []
    for base in [ROOT / "detection", ROOT]:
        if base.exists():
            for p in base.rglob("*.py"):
                low = str(p).lower()
                if any(k in low for k in ["config", "yolo", "yolov8", "rtdetr", "rt-detr", "co-dino", "codino", "pigdetect", "coco"]):
                    if "__pycache__" not in low and "site-packages" not in low:
                        candidates.append(p)

    rows = []
    seen = set()

    for p in candidates:
        if str(p) in seen:
            continue
        seen.add(str(p))

        text_head = ""
        try:
            text_head = p.read_text(errors="ignore")[:5000]
        except Exception:
            pass

        rows.append({
            "path": str(p),
            "filename": p.name,
            "relative_path": str(p.relative_to(ROOT)) if str(p).startswith(str(ROOT)) else str(p),
            "role": infer_model_role(p),
            "score": score_config(p),
            "has_num_classes_1_hint": "num_classes=1" in text_head.replace(" ", "") or "num_classes = 1" in text_head,
            "has_pig_hint": "pig" in text_head.lower(),
            "has_mmdet_hint": "mmdet" in text_head.lower() or "mmyolo" in text_head.lower(),
        })

    df = pd.DataFrame(rows)
    if len(df):
        df = df.sort_values(["score", "filename"], ascending=[False, True])
    return df


def discover_checkpoints():
    patterns = ["*.pth", "*.pt", "*.ckpt"]
    rows = []

    search_roots = [
        ROOT / "detection" / "data" / "pretrained_weights",
        ROOT / "detection",
        ROOT / "Week6_Unibo_Dataset_Validation",
        ROOT / "Week7_Dataset_Validation_Identity_Tracking_Behaviour_Preparation",
        ROOT,
    ]

    seen = set()

    for base in search_roots:
        if not base.exists():
            continue
        for pattern in patterns:
            for p in base.rglob(pattern):
                low = str(p).lower()
                if "site-packages" in low or "__pycache__" in low:
                    continue
                if str(p) in seen:
                    continue
                seen.add(str(p))

                rows.append({
                    "path": str(p),
                    "filename": p.name,
                    "relative_path": str(p.relative_to(ROOT)) if str(p).startswith(str(ROOT)) else str(p),
                    "role": infer_model_role(p),
                    "score": score_checkpoint(p),
                    "size_bytes": file_size(p),
                })

    df = pd.DataFrame(rows)
    if len(df):
        df = df.sort_values(["score", "filename"], ascending=[False, True])
    return df


def select_representative_clips(index_df):
    df = index_df.copy()

    df["pig_rows_numeric"] = pd.to_numeric(df.get("pig_rows", ""), errors="coerce").fillna(0).astype(int)
    df["clip_file_size_bytes"] = pd.to_numeric(df.get("clip_file_size_bytes", ""), errors="coerce").fillna(0).astype(int)

    selected = []

    def add_one(label, subset):
        if len(subset) == 0:
            return
        row = subset.iloc[0].to_dict()
        row["selection_reason"] = label
        selected.append(row)

    # 1) Edge-adjusted start clip with usable pigs.
    add_one(
        "edge_adjusted_start_clip",
        df[(df["edge_adjustment"] != "none") & (df["pig_rows_numeric"] > 0)].sort_values(["scan_frame_id"]),
    )

    # 2) Rare behaviour clip.
    rare_mask = df["behaviour_codes_present"].fillna("").astype(str).str.contains(r"\bIA\b|\bBE\b|\bDE\b", regex=True)
    add_one(
        "rare_behaviour_clip_IA_BE_DE",
        df[rare_mask & (df["pig_rows_numeric"] > 0)].sort_values(["scan_frame_id"]),
    )

    # 3) Crowded/full-label clip.
    add_one(
        "crowded_full_six_pigs_clip",
        df[(df["pig_rows_numeric"] >= 6) & (df["edge_adjustment"] == "none")].sort_values(
            ["clip_file_size_bytes"],
            ascending=[False],
        ),
    )

    # 4) Partial/occluded clip.
    add_one(
        "partial_or_occluded_clip",
        df[(df["pig_rows_numeric"] > 0) & (df["pig_rows_numeric"] < 6)].sort_values(["pig_rows_numeric", "scan_frame_id"]),
    )

    # 5) Normal representative clip.
    add_one(
        "normal_representative_clip",
        df[(df["pig_rows_numeric"] == 6) & (df["edge_adjustment"] == "none") & (~rare_mask)].sort_values(["scan_frame_id"]),
    )

    # Deduplicate by scan_frame_id while preserving order.
    out = []
    seen = set()

    for r in selected:
        sid = r.get("scan_frame_id", "")
        if sid and sid not in seen:
            seen.add(sid)
            out.append(r)

    # Fill to 5 if duplicates removed.
    for _, r in df.sort_values(["scan_frame_id"]).iterrows():
        if len(out) >= 5:
            break
        sid = r.get("scan_frame_id", "")
        if sid not in seen and int(r.get("pig_rows_numeric", 0)) > 0:
            rr = r.to_dict()
            rr["selection_reason"] = "fallback_representative_clip"
            seen.add(sid)
            out.append(rr)

    return pd.DataFrame(out)


issues = []

# Environment audit.
env_rows = []

for module in ["torch", "torchvision", "cv2", "mmcv", "mmengine", "mmdet", "mmyolo", "ultralytics", "boxmot", "numpy", "pandas"]:
    env_rows.append({
        "check_type": "python_module",
        "name": module,
        "available": import_available(module),
        "detail": "",
    })

for cmd in ["ffmpeg", "ffprobe", "nvidia-smi", "python"]:
    available = command_exists(cmd)
    detail = ""

    if available and cmd in ["python"]:
        code, out, err = run_cmd([cmd, "--version"])
        detail = out or err
    elif available and cmd == "nvidia-smi":
        code, out, err = run_cmd(["nvidia-smi", "--query-gpu=name,memory.total,memory.free", "--format=csv,noheader"])
        detail = out if code == 0 else err

    env_rows.append({
        "check_type": "command",
        "name": cmd,
        "available": available,
        "detail": detail,
    })

env_df = pd.DataFrame(env_rows)
safe_to_csv(env_df, OUT_ENV)

configs = discover_configs()
checkpoints = discover_checkpoints()

model_rows = []

for _, r in configs.head(30).iterrows():
    model_rows.append({
        "candidate_type": "config",
        **r.to_dict(),
    })

for _, r in checkpoints.head(30).iterrows():
    model_rows.append({
        "candidate_type": "checkpoint",
        **r.to_dict(),
    })

models_df = pd.DataFrame(model_rows)
safe_to_csv(models_df, OUT_MODELS)

index_df = pd.read_csv(V26_INDEX)
selected_df = select_representative_clips(index_df)
safe_to_csv(selected_df, OUT_SELECTED)

# Recommended model pair.
recommended_config = ""
recommended_checkpoint = ""
recommended_model_family = ""

if len(configs):
    recommended_config = configs.iloc[0]["path"]

if len(checkpoints):
    recommended_checkpoint = checkpoints.iloc[0]["path"]

if recommended_config and recommended_checkpoint:
    model_family_hint = (recommended_config + " " + recommended_checkpoint).lower()
    if "yolov8" in model_family_hint:
        recommended_model_family = "mmdet_mmyolo_yolov8"
    elif "rtdetr" in model_family_hint or "rt-detr" in model_family_hint:
        recommended_model_family = "mmdet_rtdetr"
    elif "co-dino" in model_family_hint or "codino" in model_family_hint:
        recommended_model_family = "mmdet_codino"
    else:
        recommended_model_family = "mmdet_generic"

if not import_available("mmdet"):
    issues.append({
        "issue_type": "mmdet_not_available",
        "issue_detail": "mmdet import was not found. Detection dry-run may require activating/correcting env.",
    })

if not import_available("mmengine"):
    issues.append({
        "issue_type": "mmengine_not_available",
        "issue_detail": "mmengine import was not found.",
    })

if not recommended_config:
    issues.append({
        "issue_type": "no_config_candidate_found",
        "issue_detail": "No detector config candidate found under PigBench.",
    })

if not recommended_checkpoint:
    issues.append({
        "issue_type": "no_checkpoint_candidate_found",
        "issue_detail": "No checkpoint candidate found under PigBench.",
    })

if len(selected_df) == 0:
    issues.append({
        "issue_type": "no_dryrun_clips_selected",
        "issue_detail": "Could not select representative clips from v26 index.",
    })

issues_df = pd.DataFrame(issues, columns=["issue_type", "issue_detail"])
safe_to_csv(issues_df, OUT_ISSUES)

plan = pd.DataFrame([{
    "recommended_model_family": recommended_model_family,
    "recommended_config": recommended_config,
    "recommended_checkpoint": recommended_checkpoint,
    "selected_clip_count": int(len(selected_df)),
    "dryrun_sampling_policy": "sample every 10th frame from each selected 10-second clip",
    "dryrun_output_policy": "save per-frame detections, simple IoU tracks, annotated sampled frames, and contact sheets",
    "confidence_threshold_initial": 0.25,
    "iou_tracking_threshold_initial": 0.35,
    "device_policy": "cuda:0 if available, otherwise cpu",
    "next_step": "v27b_detector_tracker_dryrun",
}])
safe_to_csv(plan, OUT_PLAN)

summary = pd.DataFrame([{
    "source_v26_index": str(V26_INDEX),
    "environment_checks": int(len(env_df)),
    "python_modules_available": int(env_df[(env_df["check_type"] == "python_module") & (env_df["available"] == True)].shape[0]),
    "command_checks_available": int(env_df[(env_df["check_type"] == "command") & (env_df["available"] == True)].shape[0]),
    "config_candidates_found": int(len(configs)),
    "checkpoint_candidates_found": int(len(checkpoints)),
    "selected_dryrun_clips": int(len(selected_df)),
    "recommended_model_family": recommended_model_family,
    "recommended_config": recommended_config,
    "recommended_checkpoint": recommended_checkpoint,
    "issue_count": int(len(issues_df)),
    "ready_for_v27b_dryrun": bool(
        len(selected_df) > 0
        and bool(recommended_config)
        and bool(recommended_checkpoint)
        and import_available("mmdet")
        and import_available("mmengine")
    ),
    "generated_at": datetime.now().isoformat(timespec="seconds"),
}])
safe_to_csv(summary, OUT_SUMMARY)

ready = bool(summary.iloc[0]["ready_for_v27b_dryrun"])

OUT_README.write_text(
    "# Week 7 Detector / Tracker Preflight v27a\n\n"
    "## Purpose\n\n"
    "This step audits whether the environment, model configs, checkpoints, and representative clips are ready for a small detector/tracker dry-run.\n\n"
    "## Outputs\n\n"
    "- `week7_detector_tracker_preflight_v27a_environment_audit.csv`\n"
    "- `week7_detector_tracker_preflight_v27a_model_candidates.csv`\n"
    "- `week7_detector_tracker_preflight_v27a_selected_dryrun_clips.csv`\n"
    "- `week7_detector_tracker_preflight_v27a_dryrun_plan.csv`\n"
    "- `week7_detector_tracker_preflight_v27a_summary.csv`\n"
    "- `week7_detector_tracker_preflight_v27a_issues.csv`\n\n"
    "## Next step\n\n"
    "If ready, v27b runs detector inference and simple IoU tracking on the selected clips only.\n"
)

OUT_NOTE.write_text(
    "# Week 7 Detector / Tracker Preflight v27a\n\n"
    "## Purpose\n\n"
    "This step checks detector/tracker readiness before running inference.\n\n"
    "## Summary\n\n"
    f"- Config candidates found: `{int(summary.iloc[0]['config_candidates_found'])}`\n"
    f"- Checkpoint candidates found: `{int(summary.iloc[0]['checkpoint_candidates_found'])}`\n"
    f"- Selected dry-run clips: `{int(summary.iloc[0]['selected_dryrun_clips'])}`\n"
    f"- Recommended model family: `{recommended_model_family}`\n"
    f"- Recommended config: `{recommended_config}`\n"
    f"- Recommended checkpoint: `{recommended_checkpoint}`\n"
    f"- Issue count: `{int(summary.iloc[0]['issue_count'])}`\n"
    f"- Ready for v27b dry-run: `{ready}`\n\n"
    "## Outputs\n\n"
    f"- Summary: `{OUT_SUMMARY}`\n"
    f"- Environment audit: `{OUT_ENV}`\n"
    f"- Model candidates: `{OUT_MODELS}`\n"
    f"- Selected clips: `{OUT_SELECTED}`\n"
    f"- Dry-run plan: `{OUT_PLAN}`\n"
    f"- Issues: `{OUT_ISSUES}`\n"
)

print("Saved:")
print(OUT_ENV)
print(OUT_MODELS)
print(OUT_SELECTED)
print(OUT_PLAN)
print(OUT_ISSUES)
print(OUT_SUMMARY)
print(OUT_README)
print(OUT_NOTE)

print()
print("=== v27a preflight summary ===")
print(summary.to_string(index=False))

print()
print("=== v27a selected clips ===")
print(selected_df[["scan_frame_id", "video_id", "selection_reason", "behaviour_codes_present", "pig_rows", "clip_path"]].to_string(index=False))

print()
print("=== v27a issues ===")
if len(issues_df):
    print(issues_df.to_string(index=False))
else:
    print("No issues found.")

print()
print("=== top model candidates ===")
if len(models_df):
    print(models_df.head(20).to_string(index=False))
else:
    print("No model candidates found.")
