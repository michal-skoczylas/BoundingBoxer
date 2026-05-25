# Klasy gestów
CLASS_NAMES = ["closed_fist", "open_palm", "none"]

# Powiązanie folder → class_id
CLASS_MAP = {
    "closed_fist": 0,
    "open_palm": 1,
    "none": 2,
}

# Progi
MEDIAPIPE_MIN_DETECTION_CONFIDENCE = 0.5
MEDIAPIPE_MIN_TRACKING_CONFIDENCE = 0.5
COMBINED_CONFIDENCE_THRESHOLD = 0.8   # poniżej → flagowane do recenzji
BBOX_PADDING = 0.10                   # 10% marginesu na bounding box

# Model MediaPipe HandLandmarker
HAND_LANDMARKER_MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/"
    "hand_landmarker/hand_landmarker/float16/latest/hand_landmarker.task"
)

# Formaty eksportu
SUPPORTED_EXPORT_FORMATS = ["yolo", "coco"]
DEFAULT_EXPORT_FORMAT = "yolo"

# Rozszerzenia obrazów
IMAGE_EXTENSIONS = (".jpg", ".jpeg", ".png", ".webp", ".bmp")

# Ścieżki wyjściowe
OUTPUT_IMAGES_DIR = "images"
OUTPUT_LABELS_DIR = "labels"
OUTPUT_DATASET_YAML = "dataset.yaml"
OUTPUT_REPORT_JSON = "report.json"
