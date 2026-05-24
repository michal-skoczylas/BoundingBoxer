import numpy as np

from .config import CLASS_MAP

EXTENDED_RATIO = 1.3

FINGERS = [
    (5, 6, 8),    # index: MCP, PIP, TIP
    (9, 10, 12),  # middle
    (13, 14, 16), # ring
    (17, 18, 20), # pinky
]


def _distance(a, b):
    return np.sqrt((a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2)


class GestureClassifier:
    def classify(self, landmarks: np.ndarray) -> tuple[str, int, float]:
        if not isinstance(landmarks, np.ndarray):
            raise TypeError(
                f"Expected numpy.ndarray, got {type(landmarks).__name__}"
            )
        if landmarks.shape != (21, 3):
            raise ValueError(
                f"landmarks must have shape (21, 3), got {landmarks.shape}"
            )

        extended_count = 0
        for mcp, pip, tip in FINGERS:
            d_mcp_tip = _distance(landmarks[mcp], landmarks[tip])
            d_mcp_pip = _distance(landmarks[mcp], landmarks[pip])
            if d_mcp_tip > d_mcp_pip * EXTENDED_RATIO:
                extended_count += 1

        if extended_count >= 3:
            class_name = "open_palm"
            confidence = 0.9
        elif extended_count == 0:
            class_name = "closed_fist"
            confidence = 0.9
        else:
            class_name = "none"
            confidence = 0.3

        class_id = CLASS_MAP[class_name]
        return (class_name, class_id, confidence)
