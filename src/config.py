"""
Configuration module for On-Device Visual Memory.

Centralizes paths, defaults, and model settings.
"""

import os
from pathlib import Path

# Project root
PROJECT_ROOT = Path(__file__).parent.parent

# Data directories
DATA_DIR = PROJECT_ROOT / "data"
FRAMES_DIR = DATA_DIR / "frames"
INDEX_DIR = DATA_DIR / "index"

# Database and index files
DATABASE_PATH = INDEX_DIR / "meta.db"
VECTOR_INDEX_PATH = INDEX_DIR / "vector.index"

# Model configuration
CLIP_MODEL_NAME = "ViT-B-32-quickgelu"
CLIP_PRETRAINED = "laion2b_s34b_b79k"

# Camera capture defaults
DEFAULT_CAMERA_DEVICE = 0
DEFAULT_CAPTURE_INTERVAL = 2.0  # seconds
DEFAULT_FRAME_WIDTH = 1280
DEFAULT_FRAME_HEIGHT = 720
DEFAULT_MAX_FRAMES = None  # None = unlimited

# Index building defaults
DEFAULT_BATCH_SIZE = 32
DEFAULT_MAX_NEW_FRAMES = None  # None = process all unindexed frames

# Search defaults
DEFAULT_TOP_K = 10
EMBEDDING_DIM = 512  # For ViT-B-32

# Supported image extensions
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png"}

# Ensure directories exist
def ensure_directories():
    """Create necessary directories if they don't exist."""
    FRAMES_DIR.mkdir(parents=True, exist_ok=True)
    INDEX_DIR.mkdir(parents=True, exist_ok=True)


ensure_directories()
