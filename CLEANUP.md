# Untracked Files (not in git)

These files are gitignored and live only on disk.

## Large assets
- `yolo12x.pt` — 114MB YOLOv12 weights
- `videos/` — 3.4GB, 3 raw All-22 MP4s (2019 Ravens/Chiefs, Chiefs/Packers, Chiefs/Titans)
- `data/clips/` — 3.5GB, per-play sideline/endzone clips extracted by segment_plays.py
- `.venv/` — 1.7GB Python virtual environment

## Generated output
- `output/homography_test/` — 250MB debug images from hash mark detection iterations
  - `hash_gaps/` — earlier debug runs
  - `hash_rebuild/` — current pipeline debug images (yard lines, edge masks, hash overlays, rectified frames)

## System files
- `.DS_Store` — macOS metadata
- `.claude/` — Claude Code session data and memory
