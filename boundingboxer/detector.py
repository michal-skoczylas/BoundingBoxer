from dataclasses import dataclass

import cv2
import mediapipe as mp
import numpy as np

from .config import MEDIAPIPE_MIN_DETECTION_CONFIDENCE, MEDIAPIPE_MIN_TRACKING_CONFIDENCE


@dataclass
class HandDetection:
    landmarks: np.ndarray
    handedness: str
    detection_score: float

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
        self._hands = mp.solutions.hands.Hands(
            static_image_mode=True,
            max_num_hands=max_num_hands,
            min_detection_confidence=min_detection_confidence,
            min_tracking_confidence=min_tracking_confidence,
        )

    def detect(self, image):
        if not isinstance(image, np.ndarray):
            raise TypeError(f"Expected numpy.ndarray, got {type(image).__name__}")
        if image.ndim == 2:
            raise ValueError(
                "Grayscale image is not supported; expected 3-channel BGR"
            )
        if image.shape[-1] == 4:
            raise ValueError(
                "RGBA image is not supported; expected 3-channel BGR"
            )

        rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        results = self._hands.process(rgb)

        if results.multi_hand_landmarks is None:
            return []

        detections = []
        for hand_landmarks, handedness in zip(
            results.multi_hand_landmarks, results.multi_handedness
        ):
            lm_array = np.array(
                [[lm.x, lm.y, lm.z] for lm in hand_landmarks],
                dtype=np.float32,
            )
            label = handedness.classification[0].label
            score = float(handedness.classification[0].score)
            detections.append(
                HandDetection(
                    landmarks=lm_array,
                    handedness=label,
                    detection_score=score,
                )
            )
        return detections

    def close(self):
        if self._hands is not None:
            self._hands.close()
            self._hands = None

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()
