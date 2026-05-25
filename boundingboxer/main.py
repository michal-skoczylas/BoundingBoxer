"""CLI entry point and pipeline orchestrator."""
import argparse
import sys
import traceback
from pathlib import Path

from tqdm import tqdm

from .config import CLASS_NAMES, OUTPUT_REPORT_JSON
from .classifier import GestureClassifier
from .detector import HandDetector
from .exporter import Exporter, ProcessingResult
from .extractor import BBoxExtractor
from .loader import ImageLoader
from .reporter import Reporter


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command")

    process_parser = subparsers.add_parser("process")
    process_parser.add_argument("--input", required=True)
    process_parser.add_argument("--output", required=True)
    process_parser.add_argument("--format", choices=["yolo", "coco"], default="yolo")
    process_parser.add_argument("--confidence", type=float, default=0.8)

    review_parser = subparsers.add_parser("review")
    review_parser.add_argument("--input", default=None)
    review_parser.add_argument("--port", type=int, default=8501)

    return parser


def run_pipeline(input_dir, output_dir, format="yolo", confidence_threshold=0.8,
                 detection_confidence=0.5, progress_callback=None):
    input_dir = Path(input_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    loader = ImageLoader(input_dir)
    records = loader.scan()
    total = len(records)

    classifier = GestureClassifier()
    extractor = BBoxExtractor()

    results = []
    with HandDetector(min_detection_confidence=detection_confidence) as detector:
        for i, record in enumerate(tqdm(records, desc="Processing"), start=1):
            try:
                image = loader.load(record.path)
                img_h, img_w = image.shape[:2]
                detections = detector.detect(image)

                bboxes = []
                for det in detections:
                    bbox = extractor.extract(det, img_w, img_h,
                                             class_id=record.class_id,
                                             class_name=record.class_name)
                    bboxes.append(bbox)

                if detections:
                    det = detections[0]
                    detected_class, detected_class_id, cls_conf = classifier.classify(det.landmarks)
                    mp_conf = det.detection_score
                    match = 1.0 if detected_class == record.class_name else 0.0
                    combined_conf = mp_conf * match
                else:
                    detected_class = None
                    detected_class_id = None
                    cls_conf = 0.0
                    mp_conf = 0.0
                    combined_conf = 0.0

                needs_review = (
                    (len(detections) == 0 and record.class_name != "none") or
                    (len(detections) > 0 and combined_conf < confidence_threshold)
                )

                results.append(ProcessingResult(
                    image_record=record,
                    detections=detections,
                    bboxes=bboxes,
                    detected_class=detected_class,
                    detected_class_id=detected_class_id,
                    classification_confidence=cls_conf,
                    mediapipe_confidence=mp_conf,
                    combined_confidence=combined_conf,
                    needs_review=needs_review,
                    image_width=img_w,
                    image_height=img_h,
                ))
            except Exception:
                print(f"[WARNING] Skipping {record.path}:", file=sys.stderr)
                traceback.print_exc(file=sys.stderr)

            if progress_callback:
                progress_callback(i, total)

    reporter = Reporter()
    report = reporter.generate(results, input_dir)
    reporter.save(report, output_dir / OUTPUT_REPORT_JSON)

    exporter = Exporter()
    exporter.export_images(results, output_dir)
    if format == "yolo":
        exporter.export_yolo(results, output_dir)
    elif format == "coco":
        exporter.export_coco(results, output_dir)
    exporter.generate_dataset_yaml(output_dir, CLASS_NAMES)

    return report


def main():
    parser = build_parser()
    args = parser.parse_args()

    if args.command == "process":
        run_pipeline(
            input_dir=args.input,
            output_dir=args.output,
            format=args.format,
            confidence_threshold=args.confidence,
        )
        print(f"Processing complete. Results saved to {args.output}")
    elif args.command == "review":
        streamlit_args = [
            "run", str(Path(__file__).parent / "review" / "app.py"),
            "--", "--port", str(args.port),
        ]
        if args.input:
            streamlit_args.extend(["--input", args.input])
        import streamlit.web.cli as stcli
        sys.argv = ["streamlit"] + streamlit_args
        stcli.main()
