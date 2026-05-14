"""Number-crop classifier constants — shared by inference (src/pipeline)
and training (scripts/aux/training/train_number_classifier.py).

Lives in src/pipeline/ so the production pipeline doesn't depend on
scripts/aux/. The training entry point in scripts/aux/training/
re-imports from here.
"""
CLASSES = ["10L", "10R", "20L", "20R", "30L", "30R", "40L", "40R", "50"]
NUM_CLASSES = len(CLASSES)
INPUT_SIZE = 64
# For 1-channel input, timm averages 3-channel ImageNet pretrain stats. Use
# the average of the RGB mean/std as a single-channel reference.
PIXEL_MEAN = 0.456
PIXEL_STD = 0.224
