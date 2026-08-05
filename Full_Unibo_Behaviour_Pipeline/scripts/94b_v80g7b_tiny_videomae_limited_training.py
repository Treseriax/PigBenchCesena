from pathlib import Path
from datetime import datetime
import os
import json
import random

os.environ["CUDA_VISIBLE_DEVICES"] = ""

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import VideoMAEConfig, VideoMAEForVideoClassification

from sklearn.metrics import accuracy_score, f1_score, classification_report, confusion_matrix

SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.set_num_threads(4)

F = Path.home() / "PigBench" / "Full_Unibo_Behaviour_Pipeline"
G = F / "outputs/v80_final_project_completion/05_clip_based_videomae"
IN_DIR = G / "v80g4_limited_clip_tensor_extraction"
OUT = G / "v80g7_tiny_videomae_limited_training"
OUT.mkdir(parents=True, exist_ok=True)

report = pd.read_csv(IN_DIR / "v80g4_limited_extraction_report.csv").fillna("")
report = report[report["tensor_status"] == "OK"].copy()

classes = sorted(report["behaviour_label"].astype(str).unique().tolist())
label_to_idx = {c: i for i, c in enumerate(classes)}
idx_to_label = {i: c for c, i in label_to_idx.items()}

with open(OUT / "v80g7b_label_mapping.json", "w", encoding="utf-8") as f:
    json.dump(
        {
            "classes": classes,
            "label_to_idx": label_to_idx,
            "idx_to_label": {str(k): v for k, v in idx_to_label.items()},
        },
        f,
        indent=2,
    )

class ClipDataset(Dataset):
    def __init__(self, df, label_to_idx):
        self.df = df.reset_index(drop=True)
        self.label_to_idx = label_to_idx

    def __len__(self):
        return len(self.df)

    def __getitem__(self, i):
        r = self.df.iloc[i]
        arr = np.load(r["tensor_path"])["frames"].astype("float32") / 255.0

        # 224 -> 112 for CPU-safe limited VideoMAE.
        arr = arr[:, ::2, ::2, :]

        # T,H,W,C -> T,C,H,W
        x = torch.tensor(arr).permute(0, 3, 1, 2)
        y = torch.tensor(self.label_to_idx[str(r["behaviour_label"])], dtype=torch.long)

        return x, y, str(r["clip_id"])

train_df = report[report["split"] == "train"].copy()
val_df = report[report["split"] == "val"].copy()
test_df = report[report["split"] == "test"].copy()

train_loader = DataLoader(ClipDataset(train_df, label_to_idx), batch_size=2, shuffle=True, num_workers=0)
val_loader = DataLoader(ClipDataset(val_df, label_to_idx), batch_size=2, shuffle=False, num_workers=0)
test_loader = DataLoader(ClipDataset(test_df, label_to_idx), batch_size=2, shuffle=False, num_workers=0)

config = VideoMAEConfig(
    num_labels=len(classes),
    num_frames=32,
    image_size=112,
    patch_size=16,
    hidden_size=96,
    num_hidden_layers=1,
    num_attention_heads=3,
    intermediate_size=192,
)

model = VideoMAEForVideoClassification(config)
device = torch.device("cpu")
model.to(device)

optimizer = torch.optim.AdamW(model.parameters(), lr=5e-5, weight_decay=1e-4)

def evaluate(loader, split_name):
    model.eval()
    true_idx = []
    pred_idx = []
    clip_ids = []
    prob_rows = []

    with torch.no_grad():
        for x, y, ids in loader:
            x = x.to(device)
            y = y.to(device)

            out = model(pixel_values=x)
            logits = out.logits
            proba = torch.softmax(logits, dim=1)
            pred = torch.argmax(proba, dim=1)

            true_idx.extend(y.cpu().numpy().tolist())
            pred_idx.extend(pred.cpu().numpy().tolist())
            clip_ids.extend(list(ids))
            prob_rows.extend(proba.cpu().numpy().tolist())

    y_true = [idx_to_label[i] for i in true_idx]
    y_pred = [idx_to_label[i] for i in pred_idx]

    pred_df = pd.DataFrame({
        "split": split_name,
        "clip_id": clip_ids,
        "true_label": y_true,
        "pred_label": y_pred,
        "confidence": [float(max(p)) for p in prob_rows],
    })

    for j, c in enumerate(classes):
        pred_df["proba_" + c] = [float(p[j]) for p in prob_rows]

    pred_df.to_csv(OUT / f"v80g7b_{split_name}_clip_predictions.csv", index=False)

    rep = pd.DataFrame(
        classification_report(
            y_true,
            y_pred,
            labels=classes,
            output_dict=True,
            zero_division=0,
        )
    ).transpose().reset_index().rename(columns={"index": "class"})
    rep.to_csv(OUT / f"v80g7b_{split_name}_per_class_report.csv", index=False)

    cm = pd.DataFrame(
        confusion_matrix(y_true, y_pred, labels=classes),
        index=classes,
        columns=classes,
    )
    cm.to_csv(OUT / f"v80g7b_{split_name}_confusion_matrix.csv")

    return {
        "split": split_name,
        "clips": len(pred_df),
        "accuracy": accuracy_score(y_true, y_pred),
        "macro_f1": f1_score(y_true, y_pred, labels=classes, average="macro", zero_division=0),
        "weighted_f1": f1_score(y_true, y_pred, labels=classes, average="weighted", zero_division=0),
    }

def train_one_epoch():
    model.train()
    total_loss = 0.0
    total = 0

    for x, y, _ in train_loader:
        x = x.to(device)
        y = y.to(device)

        optimizer.zero_grad()
        out = model(pixel_values=x, labels=y)
        loss = out.loss
        loss.backward()
        optimizer.step()

        total_loss += float(loss.item()) * x.size(0)
        total += x.size(0)

    return total_loss / max(total, 1)

EPOCHS = 5
history = []
best_val_macro = -1.0
best_epoch = -1
best_path = OUT / "v80g7b_best_tiny_random_videomae.pt"

for epoch in range(1, EPOCHS + 1):
    train_loss = train_one_epoch()
    val_metrics = evaluate(val_loader, "val")

    row = {
        "epoch": epoch,
        "train_loss": train_loss,
        "val_accuracy": val_metrics["accuracy"],
        "val_macro_f1": val_metrics["macro_f1"],
        "val_weighted_f1": val_metrics["weighted_f1"],
    }
    history.append(row)

    if val_metrics["macro_f1"] > best_val_macro:
        best_val_macro = val_metrics["macro_f1"]
        best_epoch = epoch
        torch.save(model.state_dict(), best_path)

pd.DataFrame(history).to_csv(OUT / "v80g7b_training_history.csv", index=False)

model.load_state_dict(torch.load(best_path, map_location=device))

train_metrics = evaluate(train_loader, "train")
val_metrics = evaluate(val_loader, "val")
test_metrics = evaluate(test_loader, "test")

metrics = pd.DataFrame([train_metrics, val_metrics, test_metrics])
metrics.to_csv(OUT / "v80g7b_tiny_videomae_metrics_summary.csv", index=False)

issues = []

if len(train_df) == 0 or len(val_df) == 0 or len(test_df) == 0:
    issues.append({
        "item": "data",
        "issue_type": "empty_split",
        "severity": "hard",
        "detail": f"train={len(train_df)}, val={len(val_df)}, test={len(test_df)}",
    })

if float(test_metrics["macro_f1"]) < 0.10:
    issues.append({
        "item": "tiny_random_videomae",
        "issue_type": "low_macro_f1_expected_for_limited_random_model",
        "severity": "warning",
        "detail": "tiny random VideoMAE is trained on only 73 limited clips; use as feasibility/training-pipeline result, not final model",
    })

if not issues:
    issues = [{
        "item": "none",
        "issue_type": "none",
        "severity": "info",
        "detail": "tiny VideoMAE limited training completed",
    }]

pd.DataFrame(issues).to_csv(OUT / "v80g7b_issues.csv", index=False)

hard = sum(1 for x in issues if x["severity"] == "hard")
warning = sum(1 for x in issues if x["severity"] == "warning")

decision = pd.DataFrame([{
    "v80g7b_decision": "tiny_videomae_limited_training_completed" if hard == 0 else "tiny_videomae_limited_training_has_blocking_issues",
    "model_type": "tiny_random_videomae_video_classifier",
    "pretrained_weights_used": False,
    "device": str(device),
    "epochs": EPOCHS,
    "best_epoch": best_epoch,
    "train_clips": len(train_df),
    "val_clips": len(val_df),
    "test_clips": len(test_df),
    "test_accuracy": round(float(test_metrics["accuracy"]), 6),
    "test_macro_f1": round(float(test_metrics["macro_f1"]), 6),
    "test_weighted_f1": round(float(test_metrics["weighted_f1"]), 6),
    "hard_issue_count": hard,
    "warning_count": warning,
    "ready_for_v80g8_clip_model_comparison": hard == 0,
    "claim_scope": "limited_tiny_random_videomae_not_pretrained_not_final_not_production_classifier",
    "generated_at": datetime.now().isoformat(timespec="seconds"),
}])

decision.to_csv(OUT / "v80g7b_decision_summary.csv", index=False)

note = F / "notes/v80g7b_tiny_videomae_limited_training_notes.md"
note.write_text(
    "# v80g7-B Tiny VideoMAE Limited Training\n\n"
    f"- Decision: {decision.iloc[0]['v80g7b_decision']}\n"
    "- Model type: tiny random VideoMAE video classifier\n"
    "- Pretrained weights used: False\n"
    "- Input: 32 RGB frames per 10-second clip, resized to 112x112 for CPU-safe limited training\n"
    f"- Epochs: {EPOCHS}\n"
    f"- Best epoch: {best_epoch}\n"
    f"- Train clips: {len(train_df)}\n"
    f"- Val clips: {len(val_df)}\n"
    f"- Test clips: {len(test_df)}\n"
    f"- Test accuracy: {float(test_metrics['accuracy']):.6f}\n"
    f"- Test macro F1: {float(test_metrics['macro_f1']):.6f}\n"
    f"- Test weighted F1: {float(test_metrics['weighted_f1']):.6f}\n"
    f"- Hard issues: {hard}\n"
    f"- Warnings: {warning}\n\n"
    "This stage validates a VideoMAE-compatible training pipeline on the limited extracted clip dataset. "
    "It is not a pretrained model, not a final model, and not a production classifier.\n",
    encoding="utf-8",
)

print(decision.to_string(index=False))
print("=== metrics ===")
print(metrics.to_string(index=False))
print("=== issues ===")
print(pd.DataFrame(issues).to_string(index=False))
