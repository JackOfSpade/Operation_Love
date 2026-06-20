"""Operation Love v2 — personal dating-app assistant.

Layers (see README):
  drivers/      element-based app control (Bumble web, Hinge emulator)
  perception/   capture all photos + profile text into a Profile
  vision/       local quality filter + ArcFace/CLIP embeddings
  ranker/       learns YOUR taste from YOUR swipes (local, private)
  opener/       Claude writes a natural, profile-specific opener
  costing.py    client-side spend tracking + per-run budget guard
  orchestrator  ties the loop together
"""

__version__ = "2.0.0"
