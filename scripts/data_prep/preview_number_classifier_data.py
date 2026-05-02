"""Render a grid of random crops per class for visual spot-check of the
auto-labeled classifier dataset. Mislabeled crops (wrong g_index) show up
as digit shapes that don't match the class column."""
import os
import sys

import cv2
import numpy as np

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

DATA_ROOT = os.path.join(PROJECT_ROOT, "data/number_classifier/round1")
OUT = os.path.join(PROJECT_ROOT, "output/number_classifier_preview.png")
CLASSES = ["10L", "20L", "30L", "40L", "50", "40R", "30R", "20R", "10R"]
N_PER_CLASS = 16   # crops to show per class (random sample)
TILE_SIZE = 64     # source crops are 64×64

CELL_SCALE = 2     # display each crop scaled up Nx for visibility
LABEL_HEIGHT = 24

rng = np.random.default_rng(0)
cell = TILE_SIZE * CELL_SCALE
row_h = cell + LABEL_HEIGHT
canvas_h = row_h * len(CLASSES)
canvas_w = cell * N_PER_CLASS
canvas = np.zeros((canvas_h, canvas_w, 3), dtype=np.uint8)

for r, cls in enumerate(CLASSES):
    cls_dir = os.path.join(DATA_ROOT, cls)
    if not os.path.isdir(cls_dir):
        continue
    files = sorted(f for f in os.listdir(cls_dir) if f.endswith(".png"))
    if not files:
        continue
    sample_n = min(N_PER_CLASS, len(files))
    pick = rng.choice(len(files), size=sample_n, replace=False)
    for i, idx in enumerate(pick):
        img = cv2.imread(os.path.join(cls_dir, files[int(idx)]),
                          cv2.IMREAD_GRAYSCALE)
        img_up = cv2.resize(img, (cell, cell), interpolation=cv2.INTER_NEAREST)
        img_rgb = cv2.cvtColor(img_up, cv2.COLOR_GRAY2BGR)
        y0 = r * row_h + LABEL_HEIGHT
        x0 = i * cell
        canvas[y0:y0 + cell, x0:x0 + cell] = img_rgb

    # Class label on the row
    cv2.putText(canvas, f"{cls}  (n={len(files)})", (8, r * row_h + 18),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1, cv2.LINE_AA)

os.makedirs(os.path.dirname(OUT), exist_ok=True)
cv2.imwrite(OUT, canvas)
print(f"saved {OUT}  ({canvas_w}x{canvas_h})")
