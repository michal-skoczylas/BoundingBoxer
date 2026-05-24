from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

from .config import CLASS_MAP, IMAGE_EXTENSIONS


@dataclass
class ImageRecord:
    path: Path
    class_name: str
    class_id: int


class ImageLoader:
    def __init__(self, input_dir, extensions=None):
        self.input_dir = Path(input_dir)
        if not self.input_dir.exists():
            raise FileNotFoundError(f"Directory not found: {self.input_dir}")
        self.extensions = extensions if extensions is not None else IMAGE_EXTENSIONS

    def scan(self) -> list[ImageRecord]:
        records = []
        for subdir in sorted(self.input_dir.iterdir()):
            if not subdir.is_dir():
                continue
            class_name = subdir.name
            if class_name not in CLASS_MAP:
                continue
            class_id = CLASS_MAP[class_name]
            for file in sorted(subdir.iterdir()):
                if file.is_file() and file.suffix.lower() in self.extensions:
                    records.append(ImageRecord(
                        path=file,
                        class_name=class_name,
                        class_id=class_id,
                    ))
        return records

    def load(self, path) -> np.ndarray:
        img = cv2.imread(str(path))
        if img is None:
            raise FileNotFoundError(f"Could not load image: {path}")
        return img
