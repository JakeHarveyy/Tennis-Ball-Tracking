import os
from pathlib import Path

# Base Directory
BASE_DIR = Path(__file__).resolve().parent.parent

# Storage Directory
UPLOAD_DIR = BASE_DIR / "uploads"
OUTPUT_DIR = BASE_DIR / "outputs"

UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Model Paths
YOLO_MODEL_PATH = os.getenv("YOLO_MODEL_PATH", str(BASE_DIR / "models" / "yolo" / "best.pt"))
TRACKNET_MODEL_PATH = os.getenv("TRACKNET_MODEL_PATH", str(BASE_DIR / "models" / "tracknet" / "best.pt"))

# API Settings
API_HOST = os.getenv("API_HOST", "localhost")
API_PORT = int(os.getenv("API_PORT", 8000))
API_URL = f"http://{API_HOST}:{API_PORT}"