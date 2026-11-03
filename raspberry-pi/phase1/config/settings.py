from pathlib import Path

# Racine du projet
BASE_DIR = Path(__file__).resolve().parents[1]

# Caméra
CAMERA_WIDTH = 1640
CAMERA_HEIGHT = 1232
CAMERA_FPS = 30

# Modèle IA
MODEL_PATH = BASE_DIR / "models" / "detection" / "yolo26n.pt"

# Détection
CONFIDENCE_THRESHOLD = 0.40
IMAGE_SIZE = 640
PERSON_CLASS_ID = 0

# HC-SR04
TRIG_PIN = 23
ECHO_PIN = 24

MAX_DISTANCE_METERS = 4.0

# Seuils provisoires
WARNING_DISTANCE_CM = 150
CRITICAL_DISTANCE_CM = 50

# Affichage
WINDOW_NAME = "Smart Security - Phase 1"