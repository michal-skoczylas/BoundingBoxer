import json
from dataclasses import dataclass
from pathlib import Path

from .config import COMBINED_CONFIDENCE_THRESHOLD
from .exporter import ProcessingResult


@dataclass
class ClassStats:
    total: int
    detected: int
    not_detected: int
    needs_review: int
    reviewed: int
    average_confidence: float


@dataclass
class Summary:
    total_images: int
    total_detected: int
    total_not_detected: int
    total_reviewed: int
    total_needs_review: int
    average_confidence: float
    per_class_stats: dict


class Reporter:
    def generate(self, results: list, input_dir) -> dict:
        """Generate a report dict from processing results."""
        input_dir = Path(input_dir)
        entries = []
        for result in results:
            rec = result.image_record
            expected_class = rec.class_name

            # needs_review logic
            if not result.detections:
                needs_review = expected_class != "none"
            else:
                needs_review = result.combined_confidence < COMBINED_CONFIDENCE_THRESHOLD

            # bbox in YOLO format (first bbox only)
            if result.bboxes:
                img_w, img_h = result.image_width, result.image_height
                b = result.bboxes[0]
                bbox = [
                    (b.x + b.width / 2) / img_w,
                    (b.y + b.height / 2) / img_h,
                    b.width / img_w,
                    b.height / img_h,
                ]
            else:
                bbox = None

            # Relative image path
            try:
                image = str(result.image_record.path.relative_to(input_dir))
            except ValueError:
                image = f"{rec.class_name}/{rec.path.name}"

            entries.append({
                "image": image,
                "detected": len(result.detections) > 0,
                "detected_class": result.detected_class,
                "expected_class": expected_class,
                "mediapipe_confidence": result.mediapipe_confidence,
                "classification_confidence": result.classification_confidence,
                "combined_confidence": result.combined_confidence,
                "bbox": bbox,
                "reviewed": result.reviewed,
                "needs_review": needs_review,
                "manual_override": result.manual_override,
            })

        return {"results": entries}

    def save(self, report: dict, output_path) -> None:
        """Persist a report dict as JSON."""
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w") as fh:
            json.dump(report, fh, indent=2)

    def load(self, report_path) -> dict:
        """Load a report dict from a JSON file."""
        report_path = Path(report_path)
        with open(report_path) as fh:
            return json.load(fh)

    def get_summary(self, report: dict) -> Summary:
        """Compute summary statistics from a report dict."""
        results = report["results"]

        total_images = len(results)
        total_detected = sum(1 for r in results if r["detected"])
        total_not_detected = total_images - total_detected
        total_reviewed = sum(1 for r in results if r["reviewed"])
        total_needs_review = sum(1 for r in results if r["needs_review"])

        confidences = [r["combined_confidence"] for r in results if r["detected"]]
        avg_conf = sum(confidences) / len(confidences) if confidences else 0.0

        # Per-class aggregation
        per_class_raw: dict = {}
        for r in results:
            cls = r["expected_class"]
            if cls not in per_class_raw:
                per_class_raw[cls] = {
                    "total": 0, "detected": 0, "not_detected": 0,
                    "needs_review": 0, "reviewed": 0, "confidences": [],
                }
            pc = per_class_raw[cls]
            pc["total"] += 1
            if r["detected"]:
                pc["detected"] += 1
                pc["confidences"].append(r["combined_confidence"])
            else:
                pc["not_detected"] += 1
            if r["needs_review"]:
                pc["needs_review"] += 1
            if r["reviewed"]:
                pc["reviewed"] += 1

        per_class_stats = {}
        for cls, pc in per_class_raw.items():
            cls_avg = (sum(pc["confidences"]) / len(pc["confidences"])
                       if pc["confidences"] else 0.0)
            per_class_stats[cls] = ClassStats(
                total=pc["total"],
                detected=pc["detected"],
                not_detected=pc["not_detected"],
                needs_review=pc["needs_review"],
                reviewed=pc["reviewed"],
                average_confidence=cls_avg,
            )

        return Summary(
            total_images=total_images,
            total_detected=total_detected,
            total_not_detected=total_not_detected,
            total_reviewed=total_reviewed,
            total_needs_review=total_needs_review,
            average_confidence=avg_conf,
            per_class_stats=per_class_stats,
        )
