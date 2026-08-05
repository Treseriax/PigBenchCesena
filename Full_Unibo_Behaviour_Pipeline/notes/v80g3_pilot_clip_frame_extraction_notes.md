# v80g3 Pilot Clip Frame Extraction

- Decision: pilot_clip_frame_extraction_passed
- Pilot clips: 6
- Expected frames per clip: 32
- Successful clip tensors: 6
- Failed clip tensors: 0
- Hard issues: 0
- Ready for v80g4 full or limited extraction: True

This stage extracts a small pilot set of uniformly sampled 224x224 RGB frames from the 10-second behaviour clips. It creates JPEG frames, compressed NPZ clip tensors, and contact-sheet previews. No model is trained here.
