# v80g7-B Tiny VideoMAE Limited Training

- Decision: tiny_videomae_limited_training_completed
- Model type: tiny random VideoMAE video classifier
- Pretrained weights used: False
- Input: 32 RGB frames per 10-second clip, resized to 112x112 for CPU-safe limited training
- Epochs: 5
- Best epoch: 5
- Train clips: 33
- Val clips: 18
- Test clips: 22
- Test accuracy: 0.136364
- Test macro F1: 0.053030
- Test weighted F1: 0.053030
- Hard issues: 0
- Warnings: 1

This stage validates a VideoMAE-compatible training pipeline on the limited extracted clip dataset. It is not a pretrained model, not a final model, and not a production classifier.
