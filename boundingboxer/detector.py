from dataclasses import dataclass
from pathlib import Path
import urllib.request

import cv2
import mediapipe as mp
import numpy as np
from mediapipe.tasks.python import vision
from mediapipe.tasks.python.core import base_options

from .config import (
    HAND_LANDMARKER_MODEL_URL,
    MEDIAPIPE_MIN_DETECTION_CONFIDENCE,
    MEDIAPIPE_MIN_TRACKING_CONFIDENCE,
)


def _get_model_path():
    """Download hand_landmarker.task to user cache if not present."""
    cache_dir = Path.home() / ".cache" / "boundingboxer"
    model_path = cache_dir / "hand_landmarker.task"
    if not model_path.exists():
        cache_dir.mkdir(parents=True, exist_ok=True)
        urllib.request.urlretrieve(HAND_LANDMARKER_MODEL_URL, model_path)
    return str(model_path)


@dataclass
class HandDetection:
    landmarks: np.ndarray        # 21x3 (x, y, z) — normalized [0, 1]
    handedness: str              # "Left" or "Right"
    detection_score: float       # confidence from MediaPipe

    def __post_init__(self):
        if self.landmarks.shape != (21, 3):
            raise ValueError(
                f"landmarks must have shape (21, 3), got {self.landmarks.shape}"
            )
        if self.handedness not in ("Left", "Right"):
            raise ValueError(
                f"handedness must be 'Left' or 'Right', got {self.handedness!r}"
            )
        if not (0.0 <= self.detection_score <= 1.0):
            raise ValueError(
                f"detection_score must be in [0.0, 1.0], got {self.detection_score}"
            )


class HandDetector:
    def __init__(
        self,
        min_detection_confidence=MEDIAPIPE_MIN_DETECTION_CONFIDENCE,
        min_tracking_confidence=MEDIAPIPE_MIN_TRACKING_CONFIDENCE,
        max_num_hands=2,
    ):
        model_path = _get_model_path()
        options = vision.HandLandmarkerOptions(
            base_options=base_options.BaseOptions(model_asset_path=model_path),
            running_mode=vision.RunningMode.IMAGE,
            num_hands=max_num_hands,
            min_hand_detection_confidence=min_detection_confidence,
            min_tracking_confidence=min_tracking_confidence,
        )
        self._landmarker = vision.HandLandmarker.create_from_options(options)

    def detect(self, image):
        if not isinstance(image, np.ndarray):
            raise TypeError(f"Expected numpy.ndarray, got {type(image).__name__}")
        if image.ndim == 2:
            raise ValueError("Grayscale image is not supported; expected 3-channel BGR")
        if image.shape[-1] == 4:
            raise ValueError("RGBA image is not supported; expected 3-channel BGR")

        rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        result = self._landmarker.detect(mp_image)

        if not result.hand_landmarks:
            return []

        detections = []
        for landmarks, handedness_list in zip(result.hand_landmarks, result.handedness):
            lm_array = np.array(
                [[lm.x, lm.y, lm.z] for lm in landmarks],
                dtype=np.float32,
            )
            label = handedness_list[0].category_name
            score = float(handedness_list[0].score)
            detections.append(
                HandDetection(
                    landmarks=lm_array,
                    handedness=label,
                    detection_score=score,
                )
            )
        return detections

    def close(self):
        if self._landmarker is not None:
            self._landmarker.close()
            self._landmarker = None

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()
