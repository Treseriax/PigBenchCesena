from pathlib import Path
import csv
import json
import importlib.util
import subprocess
import pandas as pd
import os


ROOT = Path.home() / "PigBench"
W6 = ROOT / "Week6_Unibo_Dataset_Validation"

STATS = W6 / "outputs/dataset_statistics"
NOTES = W6 / "notes"

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


def package_available(name):
    return importlib.util.find_spec(name) is not None


def get_package_version(name):
    if not package_available(name):
        return ""

    try:
        mod = __import__(name)
        return getattr(mod, "__version__", "unknown")
    except Exception as e:
        return f"import_error:{type(e).__name__}"


def rel(path):
    try:
        return str(Path(path).relative_to(ROOT))
    except Exception:
        return str(path)


def file_size_mb(path):
    try:
        return round(Path(path).stat().st_size / (1024 * 1024), 3)
    except Exception:
        return ""


# ------------------------------------------------------------
# 1) Package capability audit
# ------------------------------------------------------------
packages = [
    "torch",
    "torchvision",
    "cv2",
    "numpy",
    "pandas",
    "mmdet",
    "mmcv",
    "mmengine",
    "ultralytics",
    "segment_anything",
    "sam2",
    "detectron2",
    "timm",
    "transformers",
    "sklearn",
    "PIL",
]

package_rows = []

for pkg in packages:
    package_rows.append({
        "package": pkg,
        "available": package_available(pkg),
        "version": get_package_version(pkg),
        "possible_use": {
            "torch": "embedding extraction / model inference",
            "torchvision": "ResNet/ViT style crop embeddings if weights available",
            "cv2": "segmentation baseline / image processing",
            "mmdet": "detector and possible model backbone features",
            "mmcv": "mmdet dependency",
            "mmengine": "mmdet dependency",
            "ultralytics": "YOLO-seg / YOLO embeddings if installed",
            "segment_anything": "SAM segmentation if installed with weights",
            "sam2": "SAM2 segmentation if installed with weights",
            "detectron2": "Mask R-CNN segmentation if installed",
            "timm": "ViT/DINO/ConvNeXt embeddings if installed with weights",
            "transformers": "DINOv2/CLIP/ViT if installed with weights",
            "sklearn": "feature normalization/PCA/UMAP if needed",
            "PIL": "image loading",
        }.get(pkg, ""),
    })

package_df = pd.DataFrame(package_rows)
package_path = STATS / "week6_model_package_capability_audit.csv"
safe_to_csv(package_df, package_path)

# ------------------------------------------------------------
# 2) Local model/weight inventory
# ------------------------------------------------------------
search_roots = [
    ROOT,
    Path.home(),
    Path("/work/pig/datasets"),
    Path("/work"),
]

suffixes = {
    ".pth",
    ".pt",
    ".ckpt",
    ".onnx",
    ".engine",
    ".bin",
    ".safetensors",
    ".pkl",
    ".pickle",
    ".weights",
}

keywords = [
    "sam",
    "sam2",
    "segment",
    "seg",
    "mask",
    "yolov8",
    "yolo",
    "resnet",
    "vit",
    "dino",
    "dinov2",
    "clip",
    "mae",
    "swin",
    "convnext",
    "reid",
    "bytetrack",
    "bot",
    "tracker",
    "codino",
    "co_dino",
    "rtdetr",
    "rt-detr",
    "faster",
    "maskrcnn",
    "mask_rcnn",
]

weight_rows = []
seen = set()

for root in search_roots:
    if not root.exists():
        continue

    try:
        for path in root.rglob("*"):
            if not path.is_file():
                continue

            if path in seen:
                continue

            seen.add(path)

            suffix = path.suffix.lower()
            name_l = path.name.lower()
            path_l = str(path).lower()

            if suffix not in suffixes:
                continue

            keyword_hits = [k for k in keywords if k in name_l or k in path_l]

            if not keyword_hits and suffix in [".bin", ".safetensors"]:
                # Avoid listing huge unrelated cache files unless name suggests vision model.
                continue

            size = file_size_mb(path)

            if size == "":
                continue

            # Avoid tiny non-model artifacts unless clearly named.
            if isinstance(size, (int, float)) and size < 0.05 and not keyword_hits:
                continue

            weight_rows.append({
                "path": rel(path),
                "absolute_path": str(path),
                "filename": path.name,
                "suffix": suffix,
                "size_mb": size,
                "keyword_hits": ",".join(keyword_hits),
                "candidate_type": (
                    "segmentation_candidate" if any(k in keyword_hits for k in ["sam", "sam2", "segment", "seg", "mask", "maskrcnn", "mask_rcnn"])
                    else "embedding_candidate" if any(k in keyword_hits for k in ["resnet", "vit", "dino", "dinov2", "clip", "mae", "swin", "convnext"])
                    else "detector_or_tracking_candidate" if any(k in keyword_hits for k in ["yolo", "yolov8", "rtdetr", "rt-detr", "codino", "co_dino", "faster", "reid", "tracker"])
                    else "unknown_model_candidate"
                ),
            })

    except Exception as e:
        weight_rows.append({
            "path": rel(root),
            "absolute_path": str(root),
            "filename": "",
            "suffix": "",
            "size_mb": "",
            "keyword_hits": "",
            "candidate_type": f"search_error:{type(e).__name__}",
        })

weight_df = pd.DataFrame(weight_rows)

if len(weight_df):
    weight_df = weight_df.sort_values(["candidate_type", "size_mb", "path"], ascending=[True, False, True])

weights_path = STATS / "week6_local_model_weight_inventory.csv"
safe_to_csv(weight_df, weights_path)

# ------------------------------------------------------------
# 3) Config inventory for segmentation / embedding relevant models
# ------------------------------------------------------------
config_rows = []
config_suffixes = {".py", ".yaml", ".yml", ".json"}

config_roots = [
    ROOT / "detection/configs",
    ROOT,
]

for root in config_roots:
    if not root.exists():
        continue

    try:
        for path in root.rglob("*"):
            if not path.is_file() or path.suffix.lower() not in config_suffixes:
                continue

            name_l = path.name.lower()
            path_l = str(path).lower()

            keyword_hits = [k for k in keywords if k in name_l or k in path_l]

            if not keyword_hits:
                # Search text lightly for segmentation/model keywords.
                try:
                    text = path.read_text(errors="ignore")[:50000].lower()
                    keyword_hits = [k for k in keywords if k in text]
                except Exception:
                    keyword_hits = []

            if keyword_hits:
                config_rows.append({
                    "path": rel(path),
                    "absolute_path": str(path),
                    "filename": path.name,
                    "suffix": path.suffix.lower(),
                    "keyword_hits": ",".join(sorted(set(keyword_hits))),
                    "candidate_type": (
                        "segmentation_config_candidate" if any(k in keyword_hits for k in ["sam", "sam2", "segment", "seg", "mask", "maskrcnn", "mask_rcnn"])
                        else "embedding_config_candidate" if any(k in keyword_hits for k in ["resnet", "vit", "dino", "dinov2", "clip", "mae", "swin", "convnext"])
                        else "detector_config_candidate"
                    ),
                })

    except Exception as e:
        config_rows.append({
            "path": rel(root),
            "absolute_path": str(root),
            "filename": "",
            "suffix": "",
            "keyword_hits": "",
            "candidate_type": f"search_error:{type(e).__name__}",
        })

config_df = pd.DataFrame(config_rows)

if len(config_df):
    config_df = config_df.sort_values(["candidate_type", "path"])

config_path = STATS / "week6_model_config_inventory.csv"
safe_to_csv(config_df, config_path)

# ------------------------------------------------------------
# 4) Capability decision table
# ------------------------------------------------------------
seg_pkg_available = bool(
    package_df[
        (package_df["package"].isin(["ultralytics", "segment_anything", "sam2", "detectron2"]))
        & (package_df["available"] == True)
    ].shape[0]
)

seg_weight_available = False

if len(weight_df):
    seg_weight_available = bool((weight_df["candidate_type"] == "segmentation_candidate").any())

torch_available = bool(package_df[(package_df["package"] == "torch") & (package_df["available"] == True)].shape[0])
torchvision_available = bool(package_df[(package_df["package"] == "torchvision") & (package_df["available"] == True)].shape[0])
timm_available = bool(package_df[(package_df["package"] == "timm") & (package_df["available"] == True)].shape[0])
transformers_available = bool(package_df[(package_df["package"] == "transformers") & (package_df["available"] == True)].shape[0])

embedding_weight_available = False

if len(weight_df):
    embedding_weight_available = bool((weight_df["candidate_type"] == "embedding_candidate").any())

detector_weight_available = False

if len(weight_df):
    detector_weight_available = bool((weight_df["candidate_type"] == "detector_or_tracking_candidate").any())

decision_rows = [
    {
        "capability": "ideal_segmentation_model",
        "status": "possible" if seg_pkg_available and seg_weight_available else "not_ready",
        "evidence": f"seg_pkg_available={seg_pkg_available}; seg_weight_available={seg_weight_available}",
        "recommended_next_step": (
            "Run segmentation model smoke test."
            if seg_pkg_available and seg_weight_available
            else "Use preliminary bbox-guided baseline unless user provides YOLO-seg/SAM/MaskRCNN package+weights."
        ),
    },
    {
        "capability": "torchvision_crop_embedding",
        "status": "possible" if torch_available and torchvision_available else "not_ready",
        "evidence": f"torch={torch_available}; torchvision={torchvision_available}; embedding_weight_available={embedding_weight_available}",
        "recommended_next_step": (
            "Try torchvision model feature extraction. Prefer pretrained local weights if available; otherwise do not overclaim pretrained embeddings."
            if torch_available and torchvision_available
            else "Ask user to provide pretrained embedding model or install torch/torchvision."
        ),
    },
    {
        "capability": "timm_or_transformers_embedding",
        "status": "possible" if timm_available or transformers_available else "not_ready",
        "evidence": f"timm={timm_available}; transformers={transformers_available}; embedding_weight_available={embedding_weight_available}",
        "recommended_next_step": (
            "Search local pretrained model weights and test embedding extraction."
            if timm_available or transformers_available
            else "Use torchvision/detector-based embeddings or ask user for model."
        ),
    },
    {
        "capability": "detector_backbone_or_yolo_embedding",
        "status": "possible" if detector_weight_available else "not_ready",
        "evidence": f"detector_weight_available={detector_weight_available}",
        "recommended_next_step": (
            "Investigate detector backbone feature extraction from available checkpoint."
            if detector_weight_available
            else "No detector weights found; ask user for checkpoint."
        ),
    },
]

decision_df = pd.DataFrame(decision_rows)
decision_path = STATS / "week6_model_capability_decision_table.csv"
safe_to_csv(decision_df, decision_path)

# ------------------------------------------------------------
# 5) Notes
# ------------------------------------------------------------
note_path = NOTES / "week6_model_weight_capability_audit_notes.md"

with open(note_path, "w") as f:
    f.write("# Week 6 Model / Weight Capability Audit\n\n")

    f.write("## Purpose\n\n")
    f.write(
        "This audit checks which model packages, weights, and configs are available locally before attempting ideal segmentation or learned embedding extraction.\n\n"
    )

    f.write("## Package capability\n\n")
    f.write(package_df.to_markdown(index=False))
    f.write("\n\n")

    f.write("## Local weight inventory summary\n\n")
    if len(weight_df):
        f.write(weight_df[["path", "size_mb", "keyword_hits", "candidate_type"]].head(50).to_markdown(index=False))
    else:
        f.write("No local model weights found by the audit.")
    f.write("\n\n")

    f.write("## Config inventory summary\n\n")
    if len(config_df):
        f.write(config_df[["path", "keyword_hits", "candidate_type"]].head(50).to_markdown(index=False))
    else:
        f.write("No relevant model configs found by the audit.")
    f.write("\n\n")

    f.write("## Capability decision table\n\n")
    f.write(decision_df.to_markdown(index=False))
    f.write("\n\n")

    f.write("## Interpretation\n\n")
    f.write(
        "If ideal segmentation packages/weights are unavailable, the current bbox-guided GrabCut/Otsu segmentation remains a preliminary baseline. "
        "For learned crop embeddings, we should only claim pretrained/learned semantic embeddings if a trained model or local pretrained weights are available. "
        "Otherwise we can produce non-pretrained descriptors but should not overclaim them.\n"
    )

print("Saved:")
print(package_path)
print(weights_path)
print(config_path)
print(decision_path)
print(note_path)

print()
print("=== Package capability ===")
print(package_df.to_string(index=False))

print()
print("=== Capability decision table ===")
print(decision_df.to_string(index=False))

print()
print("=== Top local weights ===")
if len(weight_df):
    print(weight_df[["path", "size_mb", "keyword_hits", "candidate_type"]].head(30).to_string(index=False))
else:
    print("No model weights found.")
