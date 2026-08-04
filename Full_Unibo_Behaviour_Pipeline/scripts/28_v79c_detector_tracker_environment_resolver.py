from pathlib import Path
from datetime import datetime
import csv
import importlib.util
import subprocess
import sys
import os
import re
import pandas as pd

ROOT = Path.home() / "PigBench"
FULL = ROOT / "Full_Unibo_Behaviour_Pipeline"

RUN_PLAN = FULL / "outputs/v79b_tracking_manifest_audit_and_run_plan/v79b_tracking_run_plan_36_videos.csv"
PILOT_PLAN = FULL / "outputs/v79b_tracking_manifest_audit_and_run_plan/v79b_pilot_tracking_run_plan.csv"

OUT = FULL / "outputs/v79c_detector_tracker_environment_resolver"
FRAMES = OUT / "pilot_smoke_frames"
OUT.mkdir(parents=True, exist_ok=True)
FRAMES.mkdir(parents=True, exist_ok=True)

ENV_AUDIT = OUT / "v79c_environment_audit.csv"
ASSETS = OUT / "v79c_detector_assets.csv"
PILOT_FRAMES = OUT / "v79c_pilot_video_frame_smoke_test.csv"
DECISION = OUT / "v79c_decision_summary.csv"
ISSUES = OUT / "v79c_issues.csv"
NOTE = FULL / "notes/v79c_detector_tracker_environment_resolver_notes.md"
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

def has_module(name):
    return importlib.util.find_spec(name) is not None

def shell(cmd, timeout=10):
    try:
        r = subprocess.run(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            timeout=timeout,
        )
        return r.returncode, r.stdout.strip(), r.stderr.strip()
    except Exception as e:
        return -1, "", str(e)

issues = []

run_plan = pd.read_csv(RUN_PLAN).fillna("")
pilot_plan = pd.read_csv(PILOT_PLAN).fillna("")

for df in [run_plan, pilot_plan]:
    for c in df.columns:
        if df[c].dtype == object:
            df[c] = df[c].map(clean)

# 1) Package/import audit.
packages = [
    "cv2",
    "numpy",
    "pandas",
    "torch",
    "torchvision",
    "mmcv",
    "mmengine",
    "mmdet",
    "mmyolo",
    "ultralytics",
    "sklearn",
]

env_rows = []

for pkg in packages:
    ok = has_module(pkg)
    version = ""
    err = ""

    if ok:
        try:
            mod = __import__(pkg)
            version = clean(getattr(mod, "__version__", "installed"))
        except Exception as e:
            err = str(e)

    env_rows.append({
        "package": pkg,
        "import_available": ok,
        "version": version,
        "import_error": err,
    })

# Python and CUDA.
env_rows.append({
    "package": "python",
    "import_available": True,
    "version": sys.version.replace("\n", " "),
    "import_error": "",
})

torch_cuda_available = False
torch_cuda_device_count = 0
torch_cuda_name = ""

if has_module("torch"):
    try:
        import torch
        torch_cuda_available = bool(torch.cuda.is_available())
        torch_cuda_device_count = int(torch.cuda.device_count())
        if torch_cuda_device_count:
            torch_cuda_name = torch.cuda.get_device_name(0)
    except Exception as e:
        issues.append({
            "item": "torch_cuda",
            "issue_type": "warning_cuda_query_failed",
            "severity": "warning",
            "detail": str(e),
        })

rc, nvsmi_out, nvsmi_err = shell(["bash", "-lc", "nvidia-smi --query-gpu=name,memory.total,memory.free --format=csv,noheader"], timeout=10)

env_rows.append({
    "package": "torch_cuda",
    "import_available": torch_cuda_available,
    "version": f"device_count={torch_cuda_device_count}; device0={torch_cuda_name}",
    "import_error": "",
})

env_rows.append({
    "package": "nvidia_smi",
    "import_available": rc == 0,
    "version": nvsmi_out[:500],
    "import_error": nvsmi_err[:500],
})

env_df = pd.DataFrame(env_rows)
write(env_df, ENV_AUDIT)

# 2) Detector asset search.
asset_rows = []

search_roots = [
    ROOT / "detection",
    ROOT / "Full_Unibo_Behaviour_Pipeline",
    ROOT,
]

suffixes = {".pth", ".pt", ".onnx", ".py", ".yaml", ".yml", ".json"}

seen = set()

for base in search_roots:
    if not base.exists():
        continue

    for p in base.rglob("*"):
        if not p.is_file():
            continue
        if p in seen:
            continue
        seen.add(p)

        name = p.name.lower()
        rel = str(p.relative_to(ROOT)) if str(p).startswith(str(ROOT)) else str(p)

        if p.suffix.lower() not in suffixes:
            continue

        is_weight = p.suffix.lower() in {".pth", ".pt", ".onnx"}
        is_config = p.suffix.lower() in {".py", ".yaml", ".yml", ".json"}

        keyword_score = 0
        for kw in ["yolo", "yolov8", "pig", "coco", "rtdetr", "rt-detr", "co_dino", "codino", "mmdet", "mmyolo"]:
            if kw in name or kw in rel.lower():
                keyword_score += 1

        if keyword_score == 0 and not is_weight:
            continue

        asset_rows.append({
            "asset_path": str(p),
            "relative_path": rel,
            "filename": p.name,
            "suffix": p.suffix.lower(),
            "is_weight": is_weight,
            "is_config": is_config,
            "keyword_score": keyword_score,
            "file_size_bytes": p.stat().st_size,
        })

assets = pd.DataFrame(asset_rows)

if len(assets):
    assets = assets.sort_values(["is_weight", "keyword_score", "file_size_bytes"], ascending=[False, False, False])
else:
    assets = pd.DataFrame(columns=[
        "asset_path", "relative_path", "filename", "suffix", "is_weight",
        "is_config", "keyword_score", "file_size_bytes"
    ])

write(assets, ASSETS)

weight_count = int((assets["is_weight"] == True).sum()) if len(assets) else 0
config_count = int((assets["is_config"] == True).sum()) if len(assets) else 0

if weight_count == 0:
    issues.append({
        "item": "detector_weights",
        "issue_type": "warning_no_detector_weight_found",
        "severity": "warning",
        "detail": "No .pth/.pt/.onnx weight file found under PigBench search roots.",
    })

if config_count == 0:
    issues.append({
        "item": "detector_configs",
        "issue_type": "warning_no_detector_config_found",
        "severity": "warning",
        "detail": "No relevant config file found under PigBench search roots.",
    })

# 3) Pilot frame extraction smoke test.
frame_rows = []

cv2_ok = has_module("cv2")
if not cv2_ok:
    issues.append({
        "item": "cv2",
        "issue_type": "hard_cv2_not_available",
        "severity": "hard",
        "detail": "OpenCV is required for frame extraction smoke test.",
    })
else:
    import cv2

    for _, r in pilot_plan.iterrows():
        video_path = Path(clean(r["video_path"]))
        video_id = clean(r["video_id"])
        out_img = FRAMES / f"{video_id}_{re.sub(r'[^A-Za-z0-9_.-]+', '_', video_path.stem)}_frame.jpg"

        exists = video_path.exists()
        opened = False
        frame_ok = False
        fps = 0.0
        frame_count = 0
        width = 0
        height = 0
        err = ""

        if exists:
            try:
                cap = cv2.VideoCapture(str(video_path))
                opened = cap.isOpened()
                if opened:
                    fps = float(cap.get(cv2.CAP_PROP_FPS) or 0)
                    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
                    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
                    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)

                    seek = int(max(fps, 1) * 2)
                    cap.set(cv2.CAP_PROP_POS_FRAMES, seek)
                    ok, frame = cap.read()
                    if not ok or frame is None:
                        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                        ok, frame = cap.read()

                    if ok and frame is not None:
                        frame_ok = bool(cv2.imwrite(str(out_img), frame))
                cap.release()
            except Exception as e:
                err = str(e)

        frame_rows.append({
            "video_id": video_id,
            "video_filename": clean(r["video_filename"]),
            "video_path": str(video_path),
            "exists": exists,
            "cv2_opened": opened,
            "frame_extracted": frame_ok,
            "frame_output": str(out_img) if frame_ok else "",
            "fps": fps,
            "frame_count": frame_count,
            "width": width,
            "height": height,
            "error": err,
        })

frames = pd.DataFrame(frame_rows)
write(frames, PILOT_FRAMES)

if len(frames):
    failed = int((frames["frame_extracted"] == False).sum())
    if failed:
        issues.append({
            "item": "pilot_frame_extraction",
            "issue_type": "hard_pilot_frame_extraction_failed",
            "severity": "hard",
            "detail": f"{failed} pilot videos failed frame extraction.",
        })

# 4) Strategy decision.
has_mmdet = bool(env_df[(env_df["package"] == "mmdet") & (env_df["import_available"] == True)].shape[0])
has_mmengine = bool(env_df[(env_df["package"] == "mmengine") & (env_df["import_available"] == True)].shape[0])
has_torch = bool(env_df[(env_df["package"] == "torch") & (env_df["import_available"] == True)].shape[0])
has_ultralytics = bool(env_df[(env_df["package"] == "ultralytics") & (env_df["import_available"] == True)].shape[0])

recommended_strategy = "manual_select_after_asset_review"

if has_mmdet and has_mmengine and weight_count > 0 and config_count > 0:
    recommended_strategy = "mmdet_detector_plus_simple_iou_tracker"
elif has_ultralytics:
    recommended_strategy = "ultralytics_detector_plus_simple_iou_tracker"
elif has_torch and weight_count > 0:
    recommended_strategy = "torch_weight_needs_config_review"
else:
    recommended_strategy = "environment_not_ready_for_detector"

if recommended_strategy == "environment_not_ready_for_detector":
    issues.append({
        "item": "detector_strategy",
        "issue_type": "hard_no_detector_strategy_available",
        "severity": "hard",
        "detail": "No usable detector strategy found from installed packages/assets.",
    })

issues.append({
    "item": "scope",
    "issue_type": "info_environment_resolver_only",
    "severity": "info",
    "detail": "v79c audits environment/assets and extracts pilot frames. It does not run detector/tracker.",
})

issues_df = pd.DataFrame(issues)
write(issues_df, ISSUES)

hard_count = int((issues_df["severity"] == "hard").sum()) if "severity" in issues_df.columns else 0
warning_count = int((issues_df["severity"] == "warning").sum()) if "severity" in issues_df.columns else 0

decision = pd.DataFrame([{
    "v79c_decision": "detector_tracker_environment_ready" if hard_count == 0 else "detector_tracker_environment_has_blocking_issues",
    "pilot_videos": len(pilot_plan),
    "pilot_frames_extracted": int((frames["frame_extracted"] == True).sum()) if len(frames) else 0,
    "package_torch": has_torch,
    "package_mmdet": has_mmdet,
    "package_mmengine": has_mmengine,
    "package_ultralytics": has_ultralytics,
    "torch_cuda_available": torch_cuda_available,
    "torch_cuda_device_count": torch_cuda_device_count,
    "detector_weight_count": weight_count,
    "detector_config_count": config_count,
    "recommended_strategy": recommended_strategy,
    "hard_issue_count": hard_count,
    "warning_count": warning_count,
    "ready_for_pilot_detector_tracker_smoke_test": bool(hard_count == 0 and len(frames) > 0),
    "ready_for_full_tracking_run": False,
    "claim_scope": "environment_and_asset_resolver_only",
    "generated_at": datetime.now().isoformat(timespec="seconds"),
}])

write(decision, DECISION)

NOTE.write_text(
    "# v79c Detector/Tracker Environment Resolver\n\n"
    f"- Decision: {decision.iloc[0]['v79c_decision']}\n"
    f"- Pilot videos: {len(pilot_plan)}\n"
    f"- Pilot frames extracted: {decision.iloc[0]['pilot_frames_extracted']}\n"
    f"- torch: {has_torch}\n"
    f"- mmdet: {has_mmdet}\n"
    f"- mmengine: {has_mmengine}\n"
    f"- ultralytics: {has_ultralytics}\n"
    f"- CUDA available: {torch_cuda_available}\n"
    f"- Detector weights found: {weight_count}\n"
    f"- Detector configs found: {config_count}\n"
    f"- Recommended strategy: {recommended_strategy}\n"
    f"- Hard issues: {hard_count}\n"
    f"- Ready for pilot detector/tracker smoke test: {bool(hard_count == 0 and len(frames) > 0)}\n\n"
    "This stage audits detector/tracker environment and pilot video readability. It does not run detector/tracker.\n",
    encoding="utf-8"
)

print("=== v79c decision ===")
print(decision.to_string(index=False))

print("\n=== environment audit ===")
print(env_df.to_string(index=False))

print("\n=== detector assets top 40 ===")
print(assets.head(40).to_string(index=False) if len(assets) else "empty")

print("\n=== pilot frames ===")
print(frames.to_string(index=False) if len(frames) else "empty")

print("\n=== issues ===")
print(issues_df.to_string(index=False))
