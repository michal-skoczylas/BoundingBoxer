from dataclasses import dataclass

import cv2
import numpy as np

from .config import BBOX_PADDING
from .detector import HandDetection


@dataclass
class BBox:
    x: float
    y: float
    width: float
    height: float
    class_id: int = 0
    class_name: str = ""


class BBoxExtractor:
    def extract(self, detection: HandDetection, image_width: int, image_height: int,
                class_id: int = 0, class_name: str = "") -> BBox:
        lm = detection.landmarks
        xs = lm[:, 0] * image_width
        ys = lm[:, 1] * image_height

        min_x, max_x = xs.min(), xs.max()
        min_y, max_y = ys.min(), ys.max()

        w = max_x - min_x
        h = max_y - min_y

        pad_w = w * BBOX_PADDING
        pad_h = h * BBOX_PADDING

        x = min_x - pad_w
        y = min_y - pad_h
        w = w + 2 * pad_w
        h = h + 2 * pad_h

        x = max(0.0, min(float(x), float(image_width)))
        y = max(0.0, min(float(y), float(image_height)))
        w = min(w, image_width - x)
        h = min(h, image_height - y)

        return BBox(
            x=float(x), y=float(y), width=float(w), height=float(h),
            class_id=class_id, class_name=class_name,
        )

    def to_yolo(self, bbox: BBox, img_w: int, img_h: int) -> tuple[float, float, float, float]:
        cx = (bbox.x + bbox.width / 2) / img_w
        cy = (bbox.y + bbox.height / 2) / img_h
        w = bbox.width / img_w
        h = bbox.height / img_h
        return (cx, cy, w, h)

    def to_coco(self, bbox: BBox) -> dict:
        return {
            "x": round(bbox.x),
            "y": round(bbox.y),
            "width": round(bbox.width),
            "height": round(bbox.height),
        }


def crop_hand(image: np.ndarray, bbox: BBox) -> np.ndarray:
    """Crop the hand region from image using the bounding box.

    Returns a BGR crop suitable for CLIP classification.
    """
    x1 = max(0, int(bbox.x))
    y1 = max(0, int(bbox.y))
    x2 = min(image.shape[1], int(bbox.x + bbox.width))
    y2 = min(image.shape[0], int(bbox.y + bbox.height))
    return image[y1:y2, x1:x2]
