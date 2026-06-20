"""Local computer-vision layer (Phase 2).

quality.py  -> pyiqa metric to drop blurry/low-quality photos
embed.py    -> ArcFace (insightface) + CLIP (open_clip) -> one feature vector

All run locally on the best device (MPS on Apple Silicon, else CUDA, else CPU);
see operation_love.device.best_device(). Stubs land in Phase 2 of the roadmap.
"""
