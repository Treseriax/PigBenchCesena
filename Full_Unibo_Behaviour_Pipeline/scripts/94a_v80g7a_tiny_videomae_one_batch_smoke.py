from pathlib import Path
from datetime import datetime
import json

import numpy as np
import pandas as pd
import torch
from transformers import VideoMAEConfig, VideoMAEForVideoClassification

torch.set_num_threads(4)

F = Path.home() / "PigBench" / "Full_Unibo_Behaviour_Pipeline"
G = F / "outputs/v80_final_project_completion/05_clip_based_videomae"
IN_DIR = G / "v80g4_limited_clip_tensor_extraction"
OUT = G / "v80g7_tiny_videomae_limited_training"
OUT.mkdir(parents=True, exist_ok=True)

report = pd.read_csv(IN_DIR / "v80g4_limited_extraction_report.csv").fillna("")
report = report[(report["tensor_status"] == "OK") & (report["split"] == "train")].copy()

classes = sorted(report["behaviour_label"].astype(str).unique().tolist())
label_to_idx = {c: i for i, c in enumerate(classes)}

r = report.iloc[0]
arr = np.load(r["tensor_path"])["frames"].astype("float32") / 255.0

# 224x224 -> 112x112 for tiny CPU-safe smoke.
arr = arr[:, ::2, ::2, :]

# T,H,W,C -> 1,T,C,H,W
x = torch.tensor(arr).permute(0, 3, 1, 2).unsqueeze(0)
y = torch.tensor([label_to_idx[str(r["behaviour_label"])]], dtype=torch.long)

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
model.train()

optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)

optimizer.zero_grad()
out = model(pixel_values=x, labels=y)
loss = out.loss
loss.backward()
optimizer.step()

rows = [{
    "check": "one_batch_forward_backward",
    "status": "passed",
    "clip_id": str(r["clip_id"]),
    "behaviour_label": str(r["behaviour_label"]),
    "input_shape": str(tuple(x.shape)),
    "num_classes": len(classes),
    "loss": float(loss.item()),
    "model_parameters": int(sum(p.numel() for p in model.parameters())),
    "claim_scope": "tiny_random_videomae_one_batch_smoke_no_final_training_claim",
    "generated_at": datetime.now().isoformat(timespec="seconds"),
}]

pd.DataFrame(rows).to_csv(OUT / "v80g7a_one_batch_smoke_report.csv", index=False)

decision = pd.DataFrame([{
    "v80g7a_decision": "tiny_videomae_one_batch_smoke_passed",
    "input_shape": str(tuple(x.shape)),
    "loss": float(loss.item()),
    "hard_issue_count": 0,
    "ready_for_v80g7b_limited_videomae_training": True,
    "claim_scope": "one_batch_smoke_only_no_final_videomae_training_yet",
    "generated_at": datetime.now().isoformat(timespec="seconds"),
}])

decision.to_csv(OUT / "v80g7a_decision_summary.csv", index=False)

note = F / "notes/v80g7a_tiny_videomae_one_batch_smoke_notes.md"
note.write_text(
    "# v80g7-A Tiny VideoMAE One-Batch Smoke\n\n"
    "- Decision: tiny_videomae_one_batch_smoke_passed\n"
    f"- Input shape: {tuple(x.shape)}\n"
    f"- Loss: {float(loss.item()):.6f}\n"
    f"- Classes: {len(classes)}\n"
    "- Hard issues: 0\n"
    "- Ready for v80g7-B limited VideoMAE training: True\n\n"
    "This stage verifies that a tiny random VideoMAE model can consume extracted clip tensors and complete forward/backward training on one batch. "
    "It does not claim a final trained VideoMAE model.\n",
    encoding="utf-8",
)

print(decision.to_string(index=False))
print(pd.DataFrame(rows).to_string(index=False))
