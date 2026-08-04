from pathlib import Path
from datetime import datetime
import os
import csv
import json
import hashlib
import zipfile
import importlib
import pandas as pd


ROOT = Path.home() / "PigBench"
W8 = ROOT / "Week8_Ground_Truth_Validation_Behaviour_Dataset_Preparation"

V68A = W8 / "outputs" / "v68a_strict_gold_anchor_crop_materialization"
V70B = W8 / "outputs" / "v70b_modeling_decision_package"

CROP_META = V68A / "Week8_StrictGold_AnchorFrame_Crop_Dataset" / "metadata" / "week8_v68a_strict_gold_anchor_crop_metadata.csv"
V70B_DECISION = V70B / "week8_v70b_decision_summary.csv"

OUT = W8 / "outputs" / "v71_pretrained_embedding_feasibility_audit"
PKG = OUT / "Week8_Pretrained_Embedding_Feasibility_Audit"
NOTES = W8 / "notes"
REPORTS = W8 / "reports"
PROGRESS = W8 / "progress"

for p in [OUT, PKG, NOTES, REPORTS, PROGRESS]:
    p.mkdir(parents=True, exist_ok=True)

OUT_ENV = PKG / "week8_v71_environment_audit.csv"
OUT_WEIGHTS = PKG / "week8_v71_local_weight_inventory.csv"
OUT_CANDIDATES = PKG / "week8_v71_embedding_source_candidates.csv"
OUT_QA = PKG / "week8_v71_feasibility_quality_checks.csv"
OUT_README = PKG / "README_Week8_Pretrained_Embedding_Feasibility_Audit.md"
OUT_MANIFEST = PKG / "week8_v71_manifest.json"

OUT_DECISION = OUT / "week8_v71_decision_summary.csv"
OUT_ISSUES = OUT / "week8_v71_issues.csv"
OUT_ZIP = OUT / "Week8_Pretrained_Embedding_Feasibility_Audit.zip"
OUT_SHA = OUT / "Week8_Pretrained_Embedding_Feasibility_Audit.sha256"
OUT_NOTE = NOTES / "week8_v71_pretrained_embedding_feasibility_audit_notes.md"
OUT_REPORT = REPORTS / "week8_v71_pretrained_embedding_feasibility_audit_report.md"
OUT_PROGRESS = PROGRESS / "week8_experiment_progress_log.csv"


def safe_to_csv(df, path):
    df.to_csv(path, index=False, quoting=csv.QUOTE_ALL, escapechar="\\", lineterminator="\n")


def clean(x):
    if pd.isna(x):
        return ""
    s = str(x).strip()
    if s.lower() in ["nan", "none", "null"]:
        return ""
    return s


def read_csv(path):
    df = pd.read_csv(path).fillna("")
    for c in df.columns:
        if df[c].dtype == object:
            df[c] = df[c].map(clean)
    return df


def bool_true(x):
    return str(x).strip().lower() == "true"


def sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def import_status(module_name):
    try:
        mod = importlib.import_module(module_name)
        ver = getattr(mod, "__version__", "")
        return True, str(ver)
    except Exception as e:
        return False, str(e)


def classify_weight(path):
    name = path.name.lower()
    full = str(path).lower()

    if "resnet18" in name:
        return "torchvision_resnet18_cached", "direct_torchvision_backbone"
    if "resnet50" in name:
        return "torchvision_resnet50_cached", "direct_torchvision_backbone"
    if "mobilenet_v3" in name or "mobilenetv3" in name:
        return "torchvision_mobilenet_v3_cached", "direct_torchvision_backbone"
    if "efficientnet_b0" in name or "efficientnet-b0" in name:
        return "torchvision_efficientnet_b0_cached", "direct_torchvision_backbone"
    if "convnext_tiny" in name or "convnext-tiny" in name:
        return "torchvision_convnext_tiny_cached", "direct_torchvision_backbone"

    if "yolov8" in full:
        return "yolov8_detector_checkpoint", "detector_backbone_candidate"
    if "rtdetr" in full or "rt-detr" in full:
        return "rtdetr_detector_checkpoint", "detector_backbone_candidate"
    if "co-dino" in full or "codino" in full:
        return "codino_detector_checkpoint", "detector_backbone_candidate"
    if "dino" in full:
        return "dino_related_checkpoint", "detector_or_transformer_candidate"
    if "mmdet" in full or "mmyolo" in full:
        return "openmmlab_checkpoint", "detector_backbone_candidate"
    if name.endswith(".pt"):
        return "generic_pt_checkpoint", "unknown_checkpoint"
    if name.endswith(".pth"):
        return "generic_pth_checkpoint", "unknown_checkpoint"

    return "other_weight_file", "unknown_checkpoint"


def scan_weights(base, max_depth=6):
    rows = []
    base = Path(base)

    if not base.exists():
        return rows

    exts = {".pt", ".pth", ".ckpt", ".onnx"}
    skip_names = {
        ".git", "__pycache__", "images", "labels", "crops_tight", "crops_context10",
        "anchor_frames", "thumbs", "static_v66b", "static_v69d"
    }

    base_depth = len(base.resolve().parts)

    for root, dirs, files in os.walk(base):
        root_path = Path(root)

        depth = len(root_path.resolve().parts) - base_depth
        if depth > max_depth:
            dirs[:] = []
            continue

        dirs[:] = [d for d in dirs if d not in skip_names]

        for fn in files:
            p = root_path / fn
            if p.suffix.lower() not in exts:
                continue

            try:
                size_mb = p.stat().st_size / (1024 * 1024)
            except Exception:
                size_mb = 0.0

            kind, readiness = classify_weight(p)

            rows.append({
                "path": str(p),
                "filename": p.name,
                "extension": p.suffix.lower(),
                "size_mb": round(size_mb, 3),
                "checkpoint_kind": kind,
                "embedding_readiness": readiness,
                "search_base": str(base),
            })

    return rows


issues = []

for p in [CROP_META, V70B_DECISION]:
    if not p.exists():
        issues.append({
            "item": str(p),
            "issue_type": "hard_missing_required_input",
            "issue_detail": "Required input missing for v71 audit.",
            "severity": "hard",
        })

if issues:
    issues_df = pd.DataFrame(issues)
    safe_to_csv(issues_df, OUT_ISSUES)
    decision = pd.DataFrame([{
        "v71_decision": "pretrained_embedding_feasibility_audit_blocked",
        "hard_issue_count": int((issues_df["severity"] == "hard").sum()),
        "ready_for_v72_frozen_embedding_baseline": False,
        "generated_at": datetime.now().isoformat(timespec="seconds"),
    }])
    safe_to_csv(decision, OUT_DECISION)
    print(decision.to_string(index=False))
    raise SystemExit(1)


v70b_decision = read_csv(V70B_DECISION)
crop_meta = read_csv(CROP_META)

if len(v70b_decision) == 0 or not bool_true(v70b_decision.iloc[0].get("ready_for_v71_pretrained_embedding_audit", "")):
    issues.append({
        "item": str(V70B_DECISION),
        "issue_type": "hard_v70b_not_ready",
        "issue_detail": "v70b must be ready before v71.",
        "severity": "hard",
    })

env_rows = []

for mod in ["torch", "torchvision", "cv2", "numpy", "pandas", "sklearn", "PIL", "mmcv", "mmengine", "mmdet", "mmyolo", "ultralytics"]:
    ok, ver = import_status(mod)
    env_rows.append({
        "module": mod,
        "available": ok,
        "version_or_error": ver,
    })

torch_ok, _ = import_status("torch")
torchvision_ok, _ = import_status("torchvision")
mmdet_ok, _ = import_status("mmdet")
mmyolo_ok, _ = import_status("mmyolo")
ultralytics_ok, _ = import_status("ultralytics")

cuda_available = False
cuda_device_count = 0
try:
    import torch
    cuda_available = bool(torch.cuda.is_available())
    cuda_device_count = int(torch.cuda.device_count())
except Exception:
    pass

env_rows.append({
    "module": "torch.cuda",
    "available": cuda_available,
    "version_or_error": f"device_count={cuda_device_count}",
})

env_df = pd.DataFrame(env_rows)
safe_to_csv(env_df, OUT_ENV)

search_dirs = [
    Path.home() / ".cache" / "torch" / "hub" / "checkpoints",
    Path.home() / ".cache" / "ultralytics",
    ROOT / "detection" / "data" / "pretrained_weights",
    ROOT / "data" / "pretrained_weights",
    ROOT / "detection",
    Path("/work/pig/datasets/PigDetect-IoU-YOLOv8"),
]

weight_rows = []
seen = set()

for d in search_dirs:
    rows = scan_weights(d, max_depth=7)
    for r in rows:
        if r["path"] in seen:
            continue
        seen.add(r["path"])
        weight_rows.append(r)

weights = pd.DataFrame(weight_rows)

if len(weights):
    weights = weights.sort_values(["embedding_readiness", "size_mb"], ascending=[True, False]).reset_index(drop=True)
else:
    weights = pd.DataFrame(columns=["path", "filename", "extension", "size_mb", "checkpoint_kind", "embedding_readiness", "search_base"])

safe_to_csv(weights, OUT_WEIGHTS)

direct_tv = weights[weights["embedding_readiness"] == "direct_torchvision_backbone"].copy() if len(weights) else pd.DataFrame()
detector_candidates = weights[weights["embedding_readiness"] == "detector_backbone_candidate"].copy() if len(weights) else pd.DataFrame()
unknown_candidates = weights[weights["embedding_readiness"].isin(["unknown_checkpoint", "detector_or_transformer_candidate"])].copy() if len(weights) else pd.DataFrame()

candidate_rows = []

if len(direct_tv):
    for _, r in direct_tv.iterrows():
        candidate_rows.append({
            "candidate_name": r["checkpoint_kind"],
            "candidate_path": r["path"],
            "candidate_type": "direct_torchvision_backbone",
            "readiness": "high",
            "requires_internet": False,
            "requires_detector_framework": False,
            "recommended_for_v72": True,
            "note": "Best route for frozen embedding baseline if load succeeds.",
        })

if len(detector_candidates):
    for _, r in detector_candidates.head(20).iterrows():
        candidate_rows.append({
            "candidate_name": r["checkpoint_kind"],
            "candidate_path": r["path"],
            "candidate_type": "detector_backbone_candidate",
            "readiness": "medium" if (mmdet_ok or mmyolo_ok or ultralytics_ok) else "low",
            "requires_internet": False,
            "requires_detector_framework": True,
            "recommended_for_v72": False if len(direct_tv) else bool(mmdet_ok or mmyolo_ok or ultralytics_ok),
            "note": "May be usable for detector-backbone embeddings, but implementation is more fragile than torchvision.",
        })

if len(unknown_candidates):
    for _, r in unknown_candidates.head(10).iterrows():
        candidate_rows.append({
            "candidate_name": r["checkpoint_kind"],
            "candidate_path": r["path"],
            "candidate_type": "unknown_checkpoint",
            "readiness": "low",
            "requires_internet": False,
            "requires_detector_framework": False,
            "recommended_for_v72": False,
            "note": "Inventory only; not selected without architecture confirmation.",
        })

if not candidate_rows:
    candidate_rows.append({
        "candidate_name": "none",
        "candidate_path": "",
        "candidate_type": "none",
        "readiness": "none",
        "requires_internet": "",
        "requires_detector_framework": "",
        "recommended_for_v72": False,
        "note": "No local pretrained embedding source found.",
    })

candidates = pd.DataFrame(candidate_rows)
safe_to_csv(candidates, OUT_CANDIDATES)

recommended = candidates[candidates["recommended_for_v72"] == True].copy()
if len(recommended):
    selected = recommended.iloc[0]
    recommended_route = clean(selected["candidate_type"])
    recommended_path = clean(selected["candidate_path"])
    ready_v72 = True
else:
    selected = None
    recommended_route = "no_pretrained_embedding_source_ready"
    recommended_path = ""
    ready_v72 = False

qa_rows = []

def add_qa(name, expected, actual, passed, severity, detail):
    qa_rows.append({
        "check_name": name,
        "expected": str(expected),
        "actual": str(actual),
        "passed": bool(passed),
        "severity": severity,
        "detail": detail,
    })

add_qa("crop_metadata_rows", 372, len(crop_meta), len(crop_meta) == 372, "hard", "Crop metadata should contain 372 rows.")
add_qa("v70b_ready", True, bool_true(v70b_decision.iloc[0].get("ready_for_v71_pretrained_embedding_audit", "")), bool_true(v70b_decision.iloc[0].get("ready_for_v71_pretrained_embedding_audit", "")), "hard", "v70b must be ready.")
add_qa("torch_available", True, torch_ok, torch_ok, "hard", "Torch is required for frozen embeddings.")
add_qa("torchvision_available", True, torchvision_ok, torchvision_ok, "warning", "Torchvision is preferred for direct frozen embedding baseline.")
add_qa("local_weight_files_found", ">0", len(weights), len(weights) > 0, "warning", "At least one local checkpoint is useful.")
add_qa("direct_torchvision_candidates", ">0 preferred", len(direct_tv), len(direct_tv) > 0, "info", "Direct torchvision cached checkpoints are easiest for v72.")
add_qa("detector_candidates", ">0 optional", len(detector_candidates), len(detector_candidates) > 0, "info", "Detector checkpoints may be usable for embeddings.")
add_qa("recommended_v72_candidate", True, ready_v72, ready_v72, "warning", "Need at least one recommended local candidate for v72 without download.")

qa = pd.DataFrame(qa_rows)
safe_to_csv(qa, OUT_QA)

hard_quality_failures = int(((qa["severity"] == "hard") & (~qa["passed"])).sum())
warning_quality_failures = int(((qa["severity"] == "warning") & (~qa["passed"])).sum())
info_findings = int(((qa["severity"] == "info") & (~qa["passed"])).sum())

if hard_quality_failures:
    issues.append({
        "item": "v71_quality_checks",
        "issue_type": "hard_pretrained_embedding_audit_failed",
        "issue_detail": f"{hard_quality_failures} hard quality checks failed.",
        "severity": "hard",
    })

if warning_quality_failures:
    issues.append({
        "item": "v71_feasibility_warnings",
        "issue_type": "warning_embedding_feasibility_limited",
        "issue_detail": f"{warning_quality_failures} warning checks failed. See QA and candidate list.",
        "severity": "warning",
    })

issues.append({
    "item": "claim_scope",
    "issue_type": "info_feasibility_audit_only",
    "issue_detail": "v71 audits local embedding feasibility only; it does not train or evaluate a model.",
    "severity": "info",
})

issues_df = pd.DataFrame(issues, columns=["item", "issue_type", "issue_detail", "severity"])
safe_to_csv(issues_df, OUT_ISSUES)

hard_issue_count = int((issues_df["severity"] == "hard").sum()) if len(issues_df) else 0
warning_count = int((issues_df["severity"] == "warning").sum()) if len(issues_df) else 0
info_count = int((issues_df["severity"] == "info").sum()) if len(issues_df) else 0

manifest = {
    "version": "week8_v71_pretrained_embedding_feasibility_audit",
    "created_at": datetime.now().isoformat(timespec="seconds"),
    "crop_metadata_rows": int(len(crop_meta)),
    "torch_available": bool(torch_ok),
    "torchvision_available": bool(torchvision_ok),
    "cuda_available": bool(cuda_available),
    "local_weight_files_found": int(len(weights)),
    "direct_torchvision_candidate_count": int(len(direct_tv)),
    "detector_candidate_count": int(len(detector_candidates)),
    "recommended_route": recommended_route,
    "recommended_path": recommended_path,
    "ready_for_v72": bool(ready_v72 and hard_issue_count == 0),
    "claim_boundary": "feasibility audit only; no model training in v71",
}

OUT_MANIFEST.write_text(json.dumps(manifest, indent=2, ensure_ascii=False))

OUT_README.write_text(
    "# Week8 v71 Pretrained Embedding Feasibility Audit\n\n"
    f"- Torch available: {torch_ok}\n"
    f"- Torchvision available: {torchvision_ok}\n"
    f"- CUDA available: {cuda_available}\n"
    f"- Local weight files found: {len(weights)}\n"
    f"- Direct torchvision candidates: {len(direct_tv)}\n"
    f"- Detector candidates: {len(detector_candidates)}\n"
    f"- Recommended route: {recommended_route}\n"
    f"- Recommended path: {recommended_path}\n\n"
    "This audit does not train or evaluate a model.\n"
)

if OUT_ZIP.exists():
    OUT_ZIP.unlink()

with zipfile.ZipFile(OUT_ZIP, "w", compression=zipfile.ZIP_DEFLATED) as z:
    for p in sorted(PKG.rglob("*")):
        if p.is_file():
            z.write(p, p.relative_to(OUT))

zip_hash = sha256_file(OUT_ZIP)
OUT_SHA.write_text(f"{zip_hash}  {OUT_ZIP.name}\n")

decision = pd.DataFrame([{
    "v71_decision": "pretrained_embedding_feasibility_audit_completed" if hard_issue_count == 0 else "pretrained_embedding_feasibility_audit_has_blocking_issues",
    "crop_metadata_rows": int(len(crop_meta)),
    "torch_available": bool(torch_ok),
    "torchvision_available": bool(torchvision_ok),
    "cuda_available": bool(cuda_available),
    "local_weight_files_found": int(len(weights)),
    "direct_torchvision_candidate_count": int(len(direct_tv)),
    "detector_candidate_count": int(len(detector_candidates)),
    "recommended_route": recommended_route,
    "recommended_path": recommended_path,
    "zip_path": str(OUT_ZIP),
    "zip_sha256": zip_hash,
    "hard_quality_failures": hard_quality_failures,
    "warning_quality_failures": warning_quality_failures,
    "hard_issue_count": hard_issue_count,
    "warning_count": warning_count,
    "info_count": info_count,
    "issue_count": int(len(issues_df)),
    "ready_for_v72_frozen_embedding_baseline": bool(ready_v72 and hard_issue_count == 0),
    "claim_scope": "feasibility_audit_only_no_model_training",
    "generated_at": datetime.now().isoformat(timespec="seconds"),
}])

safe_to_csv(decision, OUT_DECISION)

OUT_NOTE.write_text(
    "# Week 8 v71 Pretrained Embedding Feasibility Audit\n\n"
    f"- v71 decision: {decision.iloc[0]['v71_decision']}\n"
    f"- Torch available: {torch_ok}\n"
    f"- Torchvision available: {torchvision_ok}\n"
    f"- CUDA available: {cuda_available}\n"
    f"- Local weight files found: {len(weights)}\n"
    f"- Direct torchvision candidates: {len(direct_tv)}\n"
    f"- Detector candidates: {len(detector_candidates)}\n"
    f"- Recommended route: {recommended_route}\n"
    f"- Recommended path: {recommended_path}\n"
    f"- Hard issues: {hard_issue_count}\n"
    f"- Warnings: {warning_count}\n"
    f"- Ready for v72 frozen embedding baseline: {bool(ready_v72 and hard_issue_count == 0)}\n\n"
    "v71 is only a feasibility audit. It does not train or evaluate a model.\n"
)

OUT_REPORT.write_text(
    "# Week 8 v71 Pretrained Embedding Feasibility Audit Report\n\n"
    f"Decision: {decision.iloc[0]['v71_decision']}\n\n"
    f"Recommended route: {recommended_route}\n\n"
    f"Recommended path: {recommended_path}\n\n"
    f"ZIP: {OUT_ZIP}\n\n"
    f"SHA256: {zip_hash}\n"
)

progress_row = pd.DataFrame([{
    "date": datetime.now().date().isoformat(),
    "stage": "v71",
    "task_name": "Pretrained embedding feasibility audit",
    "status": "PASS_WITH_WARNINGS" if hard_issue_count == 0 and warning_count > 0 else ("PASS" if hard_issue_count == 0 else "NEEDS_FIX"),
    "input_summary": str(CROP_META),
    "output_summary": str(PKG),
    "hard_issues": hard_issue_count,
    "warnings": warning_count,
    "next_action": "Run v72 frozen embedding baseline if ready; otherwise review candidates and choose a local source.",
}])

if OUT_PROGRESS.exists():
    old = pd.read_csv(OUT_PROGRESS)
    progress = pd.concat([old, progress_row], ignore_index=True)
else:
    progress = progress_row

safe_to_csv(progress, OUT_PROGRESS)

print("=== v71 decision ===")
print(decision.to_string(index=False))
print("\n=== candidates ===")
print(candidates.head(20).to_string(index=False))
print("\n=== QA ===")
print(qa.to_string(index=False))
print("\n=== issues ===")
print(issues_df.to_string(index=False) if len(issues_df) else "No issues found.")
