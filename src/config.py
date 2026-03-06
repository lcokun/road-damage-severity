from __future__ import annotations
from pathlib import Path

# Fixed root
PROJECT_ROOT = Path(__file__).resolve().parent.parent

SEED = 42

# Data paths
DATA_EXTRACTED = PROJECT_ROOT / "data_extracted"
DATA_PROCESSED = PROJECT_ROOT / "data_processed_bbox"
OUTPUTS_DIR = PROJECT_ROOT / "outputs"

# Splits
TRAIN_RATIO = 0.70
VAL_RATIO = 0.15
TEST_RATIO = 0.15

# Image / training defaults
IMG_SIZE = 224
BATCH_SIZE = 128
EPOCHS = 20
LR = 3e-4
NUM_WORKERS = 8

CLASSES = ["normal", "minor", "severe"]

# Severity mapping (LOCKED)
SEVERITY_MAP = {
    "D00": "minor",
    "D01": "minor",
    "D10": "minor",
    "D11": "minor",
    "D20": "severe",
    "D40": "severe",
}