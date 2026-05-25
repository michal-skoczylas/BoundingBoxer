from dataclasses import dataclass
import shutil
from pathlib import Path

from .config import (
    CLASS_MAP,
    CLASS_NAMES,
    OUTPUT_DATASET_YAML,
    OUTPUT_IMAGES_DIR,
    OUTPUT_LABELS_DIR,
)


@dataclass
class ProcessingResult:
    image_record: any
    detections: list
    bboxes: list
    detected_class: str | None
    detected_class_id: int | None
    classification_confidence: float
    mediapipe_confidence: float
    combined_confidence: float
    needs_review: bool
    image_width: int = 1000
    image_height: int = 1000
    reviewed: bool = False
    manual_override: bool = False


class Exporter:
    def export_yolo(self, results: list, output_dir) -> None:
        """Write YOLO-format label files for each result with bboxes."""
        output_dir = Path(output_dir)
        for result in results:
            if not result.bboxes:
                continue
            img_w, img_h = result.image_width, result.image_height
            class_dir = output_dir / OUTPUT_LABELS_DIR / result.image_record.class_name
            class_dir.mkdir(parents=True, exist_ok=True)
            stem = result.image_record.path.stem
            label_path = class_dir / f"{stem}.txt"
            with open(label_path, "w") as fh:
                for bbox in result.bboxes:
                    cx = (bbox.x + bbox.width / 2) / img_w
                    cy = (bbox.y + bbox.height / 2) / img_h
                    w = bbox.width / img_w
                    h = bbox.height / img_h
                    fh.write(f"{bbox.class_id} {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}\n")

    def export_coco(self, results: list, output_dir) -> dict:
        """Build a COCO-format dict from the processing results."""
        output_dir = Path(output_dir)
        images = []
        annotations = []
        image_id = 1
        ann_id = 1

        for result in results:
            img_w, img_h = result.image_width, result.image_height
            # Compute relative file name
            try:
                file_name = str(result.image_record.path.relative_to(output_dir))
            except ValueError:
                file_name = f"{result.image_record.class_name}/{result.image_record.path.name}"

            images.append({
                "id": image_id,
                "file_name": file_name,
                "width": img_w,
                "height": img_h,
            })

            for bbox in result.bboxes:
                annotations.append({
                    "id": ann_id,
                    "image_id": image_id,
                    "category_id": bbox.class_id,
                    "bbox": [round(bbox.x), round(bbox.y),
                             round(bbox.width), round(bbox.height)],
                    "area": round(bbox.width * bbox.height),
                })
                ann_id += 1

            image_id += 1

        categories = [
            {"id": CLASS_MAP[name], "name": name}
            for name in CLASS_NAMES
        ]

        return {
            "images": images,
            "annotations": annotations,
            "categories": categories,
        }

    def generate_dataset_yaml(self, output_dir, class_names: list) -> None:
        """Write a dataset.yaml file for YOLO training."""
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        yaml_path = output_dir / OUTPUT_DATASET_YAML
        with open(yaml_path, "w") as fh:
            fh.write(f"path: {output_dir}\n")
            fh.write(f"train: {OUTPUT_IMAGES_DIR}\n")
            fh.write(f"val: {OUTPUT_IMAGES_DIR}\n")
            fh.write("names:\n")
            for i, name in enumerate(class_names):
                fh.write(f"  {i}: {name}\n")

    def export_images(self, results: list, output_dir) -> None:
        """Copy source images to output_dir/images/<class_name>/<filename>."""
        output_dir = Path(output_dir)
        for result in results:
            class_name = result.image_record.class_name
            dst_dir = output_dir / OUTPUT_IMAGES_DIR / class_name
            dst_dir.mkdir(parents=True, exist_ok=True)
            dst = dst_dir / result.image_record.path.name
            if not dst.exists():
                shutil.copy2(result.image_record.path, dst)
