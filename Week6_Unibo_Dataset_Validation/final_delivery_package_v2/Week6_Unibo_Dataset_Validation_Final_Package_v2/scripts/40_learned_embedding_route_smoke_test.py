from pathlib import Path
import csv
import traceback
import pandas as pd
import numpy as np

import torch
import torch.nn as nn

try:
    import torchvision
    from torchvision import models, transforms
    TORCHVISION_AVAILABLE = True
except Exception:
    TORCHVISION_AVAILABLE = False

try:
    import timm
    TIMM_AVAILABLE = True
except Exception:
    TIMM_AVAILABLE = False

try:
    import cv2
    CV2_AVAILABLE = True
except Exception:
    CV2_AVAILABLE = False

try:
    from PIL import Image
    PIL_AVAILABLE = True
except Exception:
    PIL_AVAILABLE = False

try:
    from mmdet.apis import init_detector
    MMDET_AVAILABLE = True
except Exception:
    MMDET_AVAILABLE = False


ROOT = Path.home() / "PigBench"
W6 = ROOT / "Week6_Unibo_Dataset_Validation"

FEAT = W6 / "outputs/feature_extractors"
GT = W6 / "outputs/unified_ground_truth"
STATS = W6 / "outputs/dataset_statistics"
NOTES = W6 / "notes"

DETS_PATH = FEAT / "week6_yolov8s_all_scanpoint_detections_with_qc_flags.csv"
FRAME_INDEX_PATH = GT / "week6_scanpoint_frame_index.csv"

STATS.mkdir(parents=True, exist_ok=True)
NOTES.mkdir(parents=True, exist_ok=True)


def safe_to_csv(df, path):
    df.to_csv(
        path,
        index=False,
        quoting=csv.QUOTE_ALL,
        escapechar="\\",
        lineterminator="\n"
    )


def get_col(df, candidates):
    for c in candidates:
        if c in df.columns:
            return c
    return None


def resolve_image_path(p):
    if pd.isna(p):
        return None

    p = str(p)

    if not p:
        return None

    path = Path(p)

    if path.exists():
        return path

    candidate = W6 / p
    if candidate.exists():
        return candidate

    candidate = ROOT / p
    if candidate.exists():
        return candidate

    return None


def make_test_crop():
    dets = pd.read_csv(DETS_PATH)
    frames = pd.read_csv(FRAME_INDEX_PATH)

    if "scan_frame_id" in frames.columns:
        keep = ["scan_frame_id"]
        for c in ["frame_image_path", "video_id", "timestamp"]:
            if c in frames.columns:
                keep.append(c)
        dets = dets.merge(frames[keep], on="scan_frame_id", how="left")

    x1_col = get_col(dets, ["x1", "bbox_x1"])
    y1_col = get_col(dets, ["y1", "bbox_y1"])
    x2_col = get_col(dets, ["x2", "bbox_x2"])
    y2_col = get_col(dets, ["y2", "bbox_y2"])

    if not all([x1_col, y1_col, x2_col, y2_col]):
        raise ValueError("Could not find bbox columns.")

    # Pick a high score row if possible.
    if "score" in dets.columns:
        dets = dets.sort_values("score", ascending=False)

    for _, r in dets.iterrows():
        img_path = resolve_image_path(r.get("frame_image_path", ""))

        if img_path is None:
            continue

        img = cv2.imread(str(img_path))

        if img is None:
            continue

        h, w = img.shape[:2]

        x1 = max(0, min(w - 1, int(round(float(r[x1_col])))))
        y1 = max(0, min(h - 1, int(round(float(r[y1_col])))))
        x2 = max(0, min(w, int(round(float(r[x2_col])))))
        y2 = max(0, min(h, int(round(float(r[y2_col])))))

        crop = img[y1:y2, x1:x2]

        if crop.size == 0 or crop.shape[0] < 16 or crop.shape[1] < 16:
            continue

        crop_rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
        pil = Image.fromarray(crop_rgb)

        return {
            "pil": pil,
            "np_rgb": crop_rgb,
            "source_image": str(img_path),
            "scan_frame_id": r.get("scan_frame_id", ""),
            "det_id": r.get("det_id", ""),
            "bbox": [x1, y1, x2, y2],
        }

    raise RuntimeError("No usable crop found.")


device = "cuda:0" if torch.cuda.is_available() else "cpu"

rows = []

crop_info = None

try:
    crop_info = make_test_crop()
    crop_ok = True
    crop_error = ""
except Exception as e:
    crop_ok = False
    crop_error = f"{type(e).__name__}: {e}"

rows.append({
    "route": "test_crop",
    "status": "PASS" if crop_ok else "FAIL",
    "pretrained": "",
    "embedding_dim": "",
    "device": device,
    "evidence": str({k: v for k, v in (crop_info or {}).items() if k not in ["pil", "np_rgb"]}) if crop_ok else crop_error,
    "recommendation": "Proceed with embedding route tests." if crop_ok else "Fix crop loading first.",
})


def try_route(route_name, pretrained_label, fn):
    try:
        dim, evidence = fn()
        rows.append({
            "route": route_name,
            "status": "PASS",
            "pretrained": pretrained_label,
            "embedding_dim": dim,
            "device": device,
            "evidence": evidence,
            "recommendation": "Candidate for final learned crop embedding extraction.",
        })
    except Exception as e:
        rows.append({
            "route": route_name,
            "status": "FAIL",
            "pretrained": pretrained_label,
            "embedding_dim": "",
            "device": device,
            "evidence": f"{type(e).__name__}: {e}",
            "recommendation": "Do not use this route unless dependency/weights are fixed.",
        })


if crop_ok and TORCHVISION_AVAILABLE:
    preprocess = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
        ),
    ])

    x = preprocess(crop_info["pil"]).unsqueeze(0).to(device)

    def tv_resnet18_pretrained():
        weights = models.ResNet18_Weights.DEFAULT
        model = models.resnet18(weights=weights)
        model.fc = nn.Identity()
        model = model.to(device).eval()

        with torch.no_grad():
            emb = model(x)

        return int(emb.shape[1]), "torchvision resnet18 DEFAULT weights loaded successfully"

    try_route("torchvision_resnet18_default", "yes", tv_resnet18_pretrained)

    def tv_resnet50_pretrained():
        weights = models.ResNet50_Weights.DEFAULT
        model = models.resnet50(weights=weights)
        model.fc = nn.Identity()
        model = model.to(device).eval()

        with torch.no_grad():
            emb = model(x)

        return int(emb.shape[1]), "torchvision resnet50 DEFAULT weights loaded successfully"

    try_route("torchvision_resnet50_default", "yes", tv_resnet50_pretrained)

    def tv_resnet18_untrained_control():
        model = models.resnet18(weights=None)
        model.fc = nn.Identity()
        model = model.to(device).eval()

        with torch.no_grad():
            emb = model(x)

        return int(emb.shape[1]), "untrained control only; not acceptable as final learned/pretrained embedding"

    try_route("torchvision_resnet18_untrained_control", "no", tv_resnet18_untrained_control)

if crop_ok and TIMM_AVAILABLE:
    preprocess_timm = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
        ),
    ])

    x_timm = preprocess_timm(crop_info["pil"]).unsqueeze(0).to(device)

    def timm_resnet18_pretrained():
        model = timm.create_model("resnet18", pretrained=True, num_classes=0)
        model = model.to(device).eval()

        with torch.no_grad():
            emb = model(x_timm)

        return int(emb.shape[-1]), "timm resnet18 pretrained=True loaded successfully"

    try_route("timm_resnet18_pretrained", "yes", timm_resnet18_pretrained)

    def timm_convnext_tiny_pretrained():
        model = timm.create_model("convnext_tiny", pretrained=True, num_classes=0)
        model = model.to(device).eval()

        with torch.no_grad():
            emb = model(x_timm)

        return int(emb.shape[-1]), "timm convnext_tiny pretrained=True loaded successfully"

    try_route("timm_convnext_tiny_pretrained", "yes", timm_convnext_tiny_pretrained)

# Detector route smoke test: only check whether model can be loaded and has extract_feat.
def detector_backbone_smoke():
    config = ROOT / "detection/configs/yolov8/yolov8_s.py"
    checkpoint = ROOT / "detection/data/pretrained_weights/yolov8_pigs/yolov8_s.pth"

    if not config.exists():
        raise FileNotFoundError(config)

    if not checkpoint.exists():
        raise FileNotFoundError(checkpoint)

    model = init_detector(str(config), str(checkpoint), device=device)

    has_extract_feat = hasattr(model, "extract_feat")

    return "unknown", f"loaded detector model; has_extract_feat={has_extract_feat}; class={model.__class__.__name__}"

if MMDET_AVAILABLE:
    try_route("mmdet_yolov8s_detector_backbone_load", "pig_detector_checkpoint", detector_backbone_smoke)

result = pd.DataFrame(rows)

result_path = STATS / "week6_learned_embedding_route_smoke_test.csv"
safe_to_csv(result, result_path)

# Choose recommendation.
usable = result[
    (result["status"] == "PASS")
    & (result["pretrained"].isin(["yes", "pig_detector_checkpoint"]))
].copy()

if len(usable):
    # Prioritize torchvision pretrained if possible, then timm, then detector.
    priority = {
        "torchvision_resnet50_default": 1,
        "torchvision_resnet18_default": 2,
        "timm_convnext_tiny_pretrained": 3,
        "timm_resnet18_pretrained": 4,
        "mmdet_yolov8s_detector_backbone_load": 5,
    }
    usable["priority"] = usable["route"].map(priority).fillna(99)
    best = usable.sort_values("priority").iloc[0].to_dict()
else:
    best = {}

decision = pd.DataFrame([
    {
        "decision": "selected_embedding_route",
        "value": best.get("route", "none"),
        "reason": (
            "A pretrained or detector-checkpoint-backed route passed smoke test."
            if best
            else "No pretrained/detector-backed embedding route passed. Ask user for pretrained model/weights."
        ),
    },
    {
        "decision": "can_generate_final_embedding_table_now",
        "value": bool(best),
        "reason": "Only true if a pretrained or detector-checkpoint-backed route passed.",
    },
])

decision_path = STATS / "week6_learned_embedding_route_decision.csv"
safe_to_csv(decision, decision_path)

note_path = NOTES / "week6_learned_embedding_route_smoke_test_notes.md"

with open(note_path, "w") as f:
    f.write("# Week 6 Learned Embedding Route Smoke Test\n\n")

    f.write("## Purpose\n\n")
    f.write(
        "This smoke test checks which learned crop embedding route can be used without overclaiming. "
        "Untrained models are tested only as a control and should not be used as final learned semantic embeddings.\n\n"
    )

    f.write("## Smoke test results\n\n")
    f.write(result.to_markdown(index=False))
    f.write("\n\n")

    f.write("## Decision\n\n")
    f.write(decision.to_markdown(index=False))
    f.write("\n\n")

    f.write("## Interpretation\n\n")
    if best:
        f.write(
            f"The selected route is `{best.get('route')}`. "
            "Next step is to run this route over all 540 bbox crops and save the embedding table with a clear method note.\n"
        )
    else:
        f.write(
            "No acceptable pretrained/detector-backed embedding route passed. "
            "At this point, the correct next step is to provide or install a pretrained model/weight rather than generating untrained dummy embeddings.\n"
        )

print("Saved:")
print(result_path)
print(decision_path)
print(note_path)

print()
print("=== Embedding smoke test result ===")
print(result.to_string(index=False))

print()
print("=== Decision ===")
print(decision.to_string(index=False))
