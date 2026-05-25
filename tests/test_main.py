"""Unit tests for main.py – CLI entry point and pipeline orchestration."""

import argparse
from dataclasses import dataclass
from pathlib import Path
from unittest.mock import MagicMock, call, patch

import numpy as np
import pytest

from boundingboxer.config import (
    CLASS_NAMES,
    COMBINED_CONFIDENCE_THRESHOLD,
    DEFAULT_EXPORT_FORMAT,
    SUPPORTED_EXPORT_FORMATS,
)
from boundingboxer.exporter import ProcessingResult
from boundingboxer.loader import ImageRecord
from boundingboxer.extractor import BBox
from boundingboxer.detector import HandDetection


# ===================================================================
# Helpers
# ===================================================================


def _dummy_image(height=100, width=100) -> np.ndarray:
    """Create a dummy 3-channel BGR image."""
    return np.zeros((height, width, 3), dtype=np.uint8)


def _dummy_landmarks() -> np.ndarray:
    """Create dummy (21, 3) landmarks with random values in [0, 1]."""
    return np.random.default_rng(42).random((21, 3)).astype(np.float32)


def _dummy_detection(
    landmarks=None,
    handedness="Right",
    detection_score=0.95,
) -> HandDetection:
    """Create a dummy HandDetection with known scores."""
    if landmarks is None:
        landmarks = _dummy_landmarks()
    return HandDetection(
        landmarks=landmarks,
        handedness=handedness,
        detection_score=detection_score,
    )


def _dummy_record(
    path="/data/closed_fist/img001.jpg",
    class_name="closed_fist",
    class_id=0,
) -> ImageRecord:
    """Create a dummy ImageRecord."""
    return ImageRecord(path=Path(path), class_name=class_name, class_id=class_id)


def _dummy_bbox(
    x=50.0, y=60.0, width=80.0, height=90.0,
    class_id=0, class_name="closed_fist",
) -> BBox:
    """Create a dummy BBox."""
    return BBox(x=x, y=y, width=width, height=height,
                class_id=class_id, class_name=class_name)


# ===================================================================
# A.  run_pipeline() tests (mocked components)
# ===================================================================


class TestRunPipelineBasic:
    """run_pipeline – basic pipeline with mocked components."""

    @pytest.fixture(autouse=True)
    def setup_mocks(self, tmp_path):
        """Set up all mocks for sub-modules used by run_pipeline."""
        self.input_dir = tmp_path / "input"
        self.output_dir = tmp_path / "output"
        self.format = "yolo"
        self.threshold = 0.8

        # Create all patches at the module level
        self.patches = {
            "ImageLoader": patch("boundingboxer.main.ImageLoader"),
            "HandDetector": patch("boundingboxer.main.HandDetector"),
            "BBoxExtractor": patch("boundingboxer.main.BBoxExtractor"),
            "GestureClassifier": patch("boundingboxer.main.GestureClassifier"),
            "ClipClassifier": patch("boundingboxer.main.ClipClassifier"),
            "crop_hand": patch("boundingboxer.main.crop_hand"),
            "Exporter": patch("boundingboxer.main.Exporter"),
            "Reporter": patch("boundingboxer.main.Reporter"),
        }

        self.mocks = {}
        for name, p in self.patches.items():
            self.mocks[name] = p.start()

        from boundingboxer.main import run_pipeline
        self.run_pipeline = run_pipeline

        yield

        for p in self.patches.values():
            p.stop()

    # ------------------------------------------------------------------
    # Test 1: Basic pipeline with one image, one detection, correct classification
    # ------------------------------------------------------------------

    def test_pipeline_with_one_image_and_correct_classification(self):
        """Verifies full orchestration: scan→load→detect→extract→classify→report→export."""
        record = _dummy_record()
        detection = _dummy_detection(detection_score=0.95)
        bbox = _dummy_bbox()

        mock_loader_instance = self.mocks["ImageLoader"].return_value
        mock_loader_instance.scan.return_value = [record]
        mock_loader_instance.load.return_value = _dummy_image(100, 100)

        mock_detector_ctx = self.mocks["HandDetector"].return_value
        mock_detector_ctx.__enter__.return_value = mock_detector_ctx
        mock_detector_ctx.detect_with_flip.return_value = [detection]

        self.mocks["crop_hand"].return_value = _dummy_image(60, 80)

        mock_extractor = self.mocks["BBoxExtractor"].return_value
        mock_extractor.extract.return_value = bbox

        mock_clip = self.mocks["ClipClassifier"].return_value
        mock_clip.classify.return_value = ("closed_fist", 0, 0.9)

        # Mock Reporter
        mock_reporter = self.mocks["Reporter"].return_value
        expected_report = {"results": [{"image": "closed_fist/img001.jpg"}]}
        mock_reporter.generate.return_value = expected_report

        # Mock Exporter
        mock_exporter = self.mocks["Exporter"].return_value

        # Act
        report = self.run_pipeline(
            self.input_dir, self.output_dir, self.format, self.threshold,
        )

        # Assert ImageLoader was constructed with correct input_dir
        self.mocks["ImageLoader"].assert_called_once_with(self.input_dir)
        mock_loader_instance.scan.assert_called_once()
        mock_loader_instance.load.assert_called_once_with(record.path)

        # Assert HandDetector was used as context manager
        self.mocks["HandDetector"].assert_called_once()
        mock_detector_ctx.__enter__.assert_called_once()
        mock_detector_ctx.detect_with_flip.assert_called_once()
        mock_detector_ctx.__exit__.assert_called_once()

        # Assert BBoxExtractor.extract called with correct args
        mock_extractor.extract.assert_called_once_with(
            detection,
            100, 100,
            class_id=record.class_id,
            class_name=record.class_name,
        )

        # Assert GestureClassifier.classify was called with landmarks
        mock_clip.classify.assert_called_once()
        assert mock_clip.classify.call_args[0][0] is self.mocks["crop_hand"].return_value

        # Assert Reporter methods called
        mock_reporter.generate.assert_called_once()
        mock_reporter.save.assert_called_once_with(
            expected_report,
            self.output_dir / "report.json",
        )

        # Assert Exporter methods called for yolo format
        mock_exporter.export_yolo.assert_called_once()
        mock_exporter.generate_dataset_yaml.assert_called_once_with(
            self.output_dir, CLASS_NAMES,
        )

        # Assert the report is returned
        assert report is expected_report

    # ------------------------------------------------------------------
    # Test 2: Image with no detection
    # ------------------------------------------------------------------

    def test_image_with_no_detection(self):
        """When HandDetector returns [], result has empty bboxes and needs_review=True (for non-none class)."""
        record = _dummy_record()
        mock_loader_instance = self.mocks["ImageLoader"].return_value
        mock_loader_instance.scan.return_value = [record]
        mock_loader_instance.load.return_value = _dummy_image(100, 100)

        mock_detector_ctx = self.mocks["HandDetector"].return_value
        mock_detector_ctx.__enter__.return_value = mock_detector_ctx
        mock_detector_ctx.detect_with_flip.return_value = []

        mock_reporter = self.mocks["Reporter"].return_value
        mock_reporter.generate.return_value = {"results": []}

        self.run_pipeline(self.input_dir, self.output_dir, self.format, self.threshold)

        # Verify the ProcessingResult passed to Reporter.generate
        call_args = mock_reporter.generate.call_args
        results_passed = call_args[0][0]
        assert len(results_passed) == 1
        pr = results_passed[0]
        assert isinstance(pr, ProcessingResult)
        assert pr.detections == []
        assert pr.bboxes == []
        assert pr.detected_class is None
        assert pr.detected_class_id is None
        assert pr.classification_confidence == 0.0
        assert pr.mediapipe_confidence == 0.0
        assert pr.combined_confidence == 0.0
        assert pr.needs_review is True  # expected_class != "none" and no detections
        assert pr.image_record is record

    # ------------------------------------------------------------------
    # Test 3: Low confidence detection triggers needs_review
    # ------------------------------------------------------------------

    def test_low_confidence_detection_triggers_needs_review(self):
        """When combined_confidence < threshold, needs_review must be True."""
        record = _dummy_record()
        detection = _dummy_detection(detection_score=0.6)
        bbox = _dummy_bbox()

        mock_loader_instance = self.mocks["ImageLoader"].return_value
        mock_loader_instance.scan.return_value = [record]
        mock_loader_instance.load.return_value = _dummy_image(100, 100)

        mock_detector_ctx = self.mocks["HandDetector"].return_value
        mock_detector_ctx.__enter__.return_value = mock_detector_ctx
        mock_detector_ctx.detect_with_flip.return_value = [detection]

        self.mocks["BBoxExtractor"].return_value.extract.return_value = bbox
        # Correct classification → multiplier = 1.0
        self.mocks["ClipClassifier"].return_value.classify.return_value = ("closed_fist", 0, 1.0)

        mock_reporter = self.mocks["Reporter"].return_value
        mock_reporter.generate.return_value = {"results": []}

        self.run_pipeline(self.input_dir, self.output_dir, self.format, 0.8)

        call_args = mock_reporter.generate.call_args
        pr = call_args[0][0][0]
        assert pr.combined_confidence == pytest.approx(0.6)  # 0.6 * 1.0
        assert pr.needs_review is True  # 0.6 < 0.8

    # ------------------------------------------------------------------
    # Test 4: Confidence threshold is respected
    # ------------------------------------------------------------------

    def test_confidence_threshold_is_respected(self):
        """With threshold=0.5 and combined_confidence=0.6 → needs_review=False."""
        record = _dummy_record()
        detection = _dummy_detection(detection_score=0.6)

        mock_loader_instance = self.mocks["ImageLoader"].return_value
        mock_loader_instance.scan.return_value = [record]
        mock_loader_instance.load.return_value = _dummy_image(100, 100)

        mock_detector_ctx = self.mocks["HandDetector"].return_value
        mock_detector_ctx.__enter__.return_value = mock_detector_ctx
        mock_detector_ctx.detect_with_flip.return_value = [detection]

        self.mocks["BBoxExtractor"].return_value.extract.return_value = _dummy_bbox()
        self.mocks["ClipClassifier"].return_value.classify.return_value = ("closed_fist", 0, 1.0)

        mock_reporter = self.mocks["Reporter"].return_value
        mock_reporter.generate.return_value = {"results": []}

        # threshold = 0.5, combined_confidence = 0.6 → needs_review should be False
        self.run_pipeline(self.input_dir, self.output_dir, self.format, 0.5)

        call_args = mock_reporter.generate.call_args
        pr = call_args[0][0][0]
        assert pr.combined_confidence == pytest.approx(0.6)
        assert pr.needs_review is False  # 0.6 >= 0.5

    # ------------------------------------------------------------------
    # Test 5: Mismatched classification produces low combined_confidence
    # ------------------------------------------------------------------

    def test_mismatched_classification_zeroes_combined_confidence(self):
        """When detected class ≠ expected class, multiplier is 0 → combined=0."""
        record = _dummy_record(class_name="closed_fist", class_id=0)
        detection = _dummy_detection(detection_score=0.95)

        mock_loader_instance = self.mocks["ImageLoader"].return_value
        mock_loader_instance.scan.return_value = [record]
        mock_loader_instance.load.return_value = _dummy_image(100, 100)

        mock_detector_ctx = self.mocks["HandDetector"].return_value
        mock_detector_ctx.__enter__.return_value = mock_detector_ctx
        mock_detector_ctx.detect_with_flip.return_value = [detection]

        self.mocks["crop_hand"].return_value = _dummy_image(60, 80)
        self.mocks["BBoxExtractor"].return_value.extract.return_value = _dummy_bbox()
        # Classifier returns "open_palm" but expected is "closed_fist" → mismatch
        self.mocks["ClipClassifier"].return_value.classify.return_value = ("open_palm", 1, 0.9)

        mock_reporter = self.mocks["Reporter"].return_value
        mock_reporter.generate.return_value = {"results": []}

        self.run_pipeline(self.input_dir, self.output_dir, self.format, 0.8)

        call_args = mock_reporter.generate.call_args
        pr = call_args[0][0][0]
        assert pr.combined_confidence == pytest.approx(0.95 * 0.9)
        assert pr.needs_review is True  # mismatch triggers review
        assert pr.detected_class == "open_palm"
        assert pr.detected_class_id == 1
        assert pr.classification_confidence == 0.9
        assert pr.mediapipe_confidence == 0.95

    # ------------------------------------------------------------------
    # Test 6: "none" folder with no detection → needs_review=False
    # ------------------------------------------------------------------

    def test_none_class_with_no_detection_no_review(self):
        """Images in the 'none' folder with no detection should NOT need review."""
        record = _dummy_record(class_name="none", class_id=2)

        mock_loader_instance = self.mocks["ImageLoader"].return_value
        mock_loader_instance.scan.return_value = [record]
        mock_loader_instance.load.return_value = _dummy_image(100, 100)

        mock_detector_ctx = self.mocks["HandDetector"].return_value
        mock_detector_ctx.__enter__.return_value = mock_detector_ctx
        mock_detector_ctx.detect_with_flip.return_value = []

        mock_reporter = self.mocks["Reporter"].return_value
        mock_reporter.generate.return_value = {"results": []}

        self.run_pipeline(self.input_dir, self.output_dir, self.format, self.threshold)

        call_args = mock_reporter.generate.call_args
        pr = call_args[0][0][0]
        assert pr.detected_class is None
        assert pr.combined_confidence == 0.0
        assert pr.needs_review is False  # Expected class is "none"

    # ------------------------------------------------------------------
    # Test 7: Multiple images processed in order
    # ------------------------------------------------------------------

    def test_multiple_images_processed_in_order(self):
        """Two images → results list is in scan order, both reporter and exporter called."""
        records = [
            _dummy_record(path="/data/closed_fist/img001.jpg", class_name="closed_fist", class_id=0),
            _dummy_record(path="/data/open_palm/img002.jpg", class_name="open_palm", class_id=1),
        ]

        mock_loader_instance = self.mocks["ImageLoader"].return_value
        mock_loader_instance.scan.return_value = records
        mock_loader_instance.load.return_value = _dummy_image(100, 100)

        detection = _dummy_detection()
        mock_detector_ctx = self.mocks["HandDetector"].return_value
        mock_detector_ctx.__enter__.return_value = mock_detector_ctx
        mock_detector_ctx.detect_with_flip.return_value = [detection]

        self.mocks["BBoxExtractor"].return_value.extract.return_value = _dummy_bbox()
        self.mocks["ClipClassifier"].return_value.classify.return_value = ("closed_fist", 0, 1.0)

        mock_reporter = self.mocks["Reporter"].return_value
        expected_report = {"results": []}
        mock_reporter.generate.return_value = expected_report

        report = self.run_pipeline(self.input_dir, self.output_dir, self.format, self.threshold)

        # Verify results order
        call_args = mock_reporter.generate.call_args
        results_passed = call_args[0][0]
        assert len(results_passed) == 2
        assert results_passed[0].image_record is records[0]
        assert results_passed[1].image_record is records[1]

        # Both Exporter methods called with the same results
        mock_exporter = self.mocks["Exporter"].return_value
        mock_exporter.export_yolo.assert_called_once()
        exported_results = mock_exporter.export_yolo.call_args[0][0]
        assert len(exported_results) == 2
        assert exported_results is results_passed

        assert report is expected_report

    # ------------------------------------------------------------------
    # Test 8: COCO export format
    # ------------------------------------------------------------------

    def test_coco_export_format(self):
        """When format='coco', Exporter.export_coco is called instead of export_yolo."""
        record = _dummy_record()
        detection = _dummy_detection()

        mock_loader_instance = self.mocks["ImageLoader"].return_value
        mock_loader_instance.scan.return_value = [record]
        mock_loader_instance.load.return_value = _dummy_image(100, 100)

        mock_detector_ctx = self.mocks["HandDetector"].return_value
        mock_detector_ctx.__enter__.return_value = mock_detector_ctx
        mock_detector_ctx.detect_with_flip.return_value = [detection]

        self.mocks["BBoxExtractor"].return_value.extract.return_value = _dummy_bbox()
        self.mocks["ClipClassifier"].return_value.classify.return_value = ("closed_fist", 0, 1.0)
        self.mocks["Reporter"].return_value.generate.return_value = {"results": []}

        mock_exporter = self.mocks["Exporter"].return_value

        self.run_pipeline(self.input_dir, self.output_dir, "coco", self.threshold)

        # export_coco should be called, NOT export_yolo
        mock_exporter.export_coco.assert_called_once()
        mock_exporter.export_yolo.assert_not_called()
        # dataset.yaml is still generated
        mock_exporter.generate_dataset_yaml.assert_called_once()

    # ------------------------------------------------------------------
    # Test 9: Output directory passed to exporter/reporter
    # ------------------------------------------------------------------

    def test_pipeline_passes_output_dir_to_exporter_and_reporter(self):
        """Exporter and Reporter methods receive the correct output_dir."""
        record = _dummy_record()
        detection = _dummy_detection()

        mock_loader_instance = self.mocks["ImageLoader"].return_value
        mock_loader_instance.scan.return_value = [record]
        mock_loader_instance.load.return_value = _dummy_image(100, 100)

        mock_detector_ctx = self.mocks["HandDetector"].return_value
        mock_detector_ctx.__enter__.return_value = mock_detector_ctx
        mock_detector_ctx.detect_with_flip.return_value = [detection]

        self.mocks["BBoxExtractor"].return_value.extract.return_value = _dummy_bbox()
        self.mocks["ClipClassifier"].return_value.classify.return_value = ("closed_fist", 0, 1.0)

        mock_reporter = self.mocks["Reporter"].return_value
        mock_reporter.generate.return_value = {"results": []}
        mock_exporter = self.mocks["Exporter"].return_value

        output = self.output_dir

        self.run_pipeline(self.input_dir, output, self.format, self.threshold)

        # Reporter.save called with output_dir / report.json
        mock_reporter.save.assert_called_once()
        save_args = mock_reporter.save.call_args
        assert save_args[0][1] == output / "report.json"

        # Exporter.export_yolo called with output_dir
        mock_exporter.export_yolo.assert_called_once()
        yolo_args = mock_exporter.export_yolo.call_args
        assert yolo_args[0][1] == output

        # Exporter.generate_dataset_yaml called with output_dir
        mock_exporter.generate_dataset_yaml.assert_called_once_with(output, CLASS_NAMES)

    # ------------------------------------------------------------------
    # Test 10: HandDetector context manager usage
    # ------------------------------------------------------------------

    def test_hand_detector_used_as_context_manager(self):
        """HandDetector must be used with `with` statement (entered and exited)."""
        record = _dummy_record()

        mock_loader_instance = self.mocks["ImageLoader"].return_value
        mock_loader_instance.scan.return_value = [record]
        mock_loader_instance.load.return_value = _dummy_image(100, 100)

        mock_detector = self.mocks["HandDetector"].return_value
        mock_detector.__enter__.return_value = mock_detector
        mock_detector.detect.return_value = [_dummy_detection()]

        self.mocks["BBoxExtractor"].return_value.extract.return_value = _dummy_bbox()
        self.mocks["ClipClassifier"].return_value.classify.return_value = ("closed_fist", 0, 1.0)
        self.mocks["Reporter"].return_value.generate.return_value = {"results": []}

        self.run_pipeline(self.input_dir, self.output_dir, self.format, self.threshold)

        # Verify __enter__ and __exit__ were both called
        mock_detector.__enter__.assert_called()
        mock_detector.__exit__.assert_called()
        # detect must have been called on the context-managed instance
        mock_detector.detect_with_flip.assert_called()

    # ------------------------------------------------------------------
    # Test 11: Image dimensions stored in ProcessingResult
    # ------------------------------------------------------------------

    def test_image_dimensions_stored_in_result(self):
        """ProcessingResult.image_width and image_height must match loaded image shape."""
        record = _dummy_record()
        img_height, img_width = 200, 300
        img = _dummy_image(height=img_height, width=img_width)

        mock_loader_instance = self.mocks["ImageLoader"].return_value
        mock_loader_instance.scan.return_value = [record]
        mock_loader_instance.load.return_value = img

        mock_detector_ctx = self.mocks["HandDetector"].return_value
        mock_detector_ctx.__enter__.return_value = mock_detector_ctx
        mock_detector_ctx.detect_with_flip.return_value = [_dummy_detection()]

        self.mocks["BBoxExtractor"].return_value.extract.return_value = _dummy_bbox()
        self.mocks["ClipClassifier"].return_value.classify.return_value = ("closed_fist", 0, 1.0)

        mock_reporter = self.mocks["Reporter"].return_value
        mock_reporter.generate.return_value = {"results": []}

        self.run_pipeline(self.input_dir, self.output_dir, self.format, self.threshold)

        call_args = mock_reporter.generate.call_args
        pr = call_args[0][0][0]
        assert pr.image_width == img_width
        assert pr.image_height == img_height

    # ------------------------------------------------------------------
    # Test 12: BBox receives class_id and class_name from ImageRecord
    # ------------------------------------------------------------------

    def test_bbox_receives_class_info_from_image_record(self):
        """BBoxExtractor.extract must be called with class_id and class_name from ImageRecord."""
        record = _dummy_record(class_name="open_palm", class_id=1)
        detection = _dummy_detection()

        mock_loader_instance = self.mocks["ImageLoader"].return_value
        mock_loader_instance.scan.return_value = [record]
        mock_loader_instance.load.return_value = _dummy_image(100, 100)

        mock_detector_ctx = self.mocks["HandDetector"].return_value
        mock_detector_ctx.__enter__.return_value = mock_detector_ctx
        mock_detector_ctx.detect_with_flip.return_value = [detection]

        mock_extractor = self.mocks["BBoxExtractor"].return_value
        mock_extractor.extract.return_value = _dummy_bbox(class_id=1, class_name="open_palm")

        self.mocks["ClipClassifier"].return_value.classify.return_value = ("open_palm", 1, 1.0)
        self.mocks["Reporter"].return_value.generate.return_value = {"results": []}

        self.run_pipeline(self.input_dir, self.output_dir, self.format, self.threshold)

        # Verify extract called with record's class_id and class_name
        mock_extractor.extract.assert_called_once_with(
            detection,
            100, 100,
            class_id=record.class_id,
            class_name=record.class_name,
        )


# ===================================================================
# B.  run_pipeline() – edge cases
# ===================================================================


class TestRunPipelineEdgeCases:
    """run_pipeline – edge cases and special scenarios."""

    @pytest.fixture(autouse=True)
    def setup_mocks(self, tmp_path):
        """Set up all mocks for sub-modules and import run_pipeline."""
        self.input_dir = tmp_path / "input"
        self.output_dir = tmp_path / "output"
        self.format = "yolo"
        self.threshold = 0.8

        self.patches = {
            "ImageLoader": patch("boundingboxer.main.ImageLoader"),
            "HandDetector": patch("boundingboxer.main.HandDetector"),
            "BBoxExtractor": patch("boundingboxer.main.BBoxExtractor"),
            "GestureClassifier": patch("boundingboxer.main.GestureClassifier"),
            "ClipClassifier": patch("boundingboxer.main.ClipClassifier"),
            "crop_hand": patch("boundingboxer.main.crop_hand"),
            "Exporter": patch("boundingboxer.main.Exporter"),
            "Reporter": patch("boundingboxer.main.Reporter"),
        }

        self.mocks = {}
        for name, p in self.patches.items():
            self.mocks[name] = p.start()

        from boundingboxer.main import run_pipeline
        self.run_pipeline = run_pipeline

        yield
        for p in self.patches.values():
            p.stop()

    def test_empty_scan_produces_empty_results_list(self):
        """When ImageLoader.scan() returns [], pipeline should still complete gracefully."""
        mock_loader_instance = self.mocks["ImageLoader"].return_value
        mock_loader_instance.scan.return_value = []

        mock_reporter = self.mocks["Reporter"].return_value
        mock_reporter.generate.return_value = {"results": []}

        self.run_pipeline(self.input_dir, self.output_dir, self.format, self.threshold)

        # Reporter should be called with empty results
        mock_reporter.generate.assert_called_once()
        call_args = mock_reporter.generate.call_args
        assert call_args[0][0] == []

        # Exporter should still be called
        mock_exporter = self.mocks["Exporter"].return_value
        mock_exporter.export_yolo.assert_called_once_with([], self.output_dir)
        mock_exporter.generate_dataset_yaml.assert_called_once()

    def test_classification_confidence_from_classifier_preserved(self):
        """The classification_confidence from GestureClassifier must be stored in ProcessingResult."""
        record = _dummy_record()
        detection = _dummy_detection(detection_score=0.9)

        mock_loader_instance = self.mocks["ImageLoader"].return_value
        mock_loader_instance.scan.return_value = [record]
        mock_loader_instance.load.return_value = _dummy_image(100, 100)

        mock_detector_ctx = self.mocks["HandDetector"].return_value
        mock_detector_ctx.__enter__.return_value = mock_detector_ctx
        mock_detector_ctx.detect_with_flip.return_value = [detection]

        self.mocks["crop_hand"].return_value = _dummy_image(60, 80)
        self.mocks["BBoxExtractor"].return_value.extract.return_value = _dummy_bbox()
        self.mocks["ClipClassifier"].return_value.classify.return_value = ("closed_fist", 0, 0.85)

        mock_reporter = self.mocks["Reporter"].return_value
        mock_reporter.generate.return_value = {"results": []}

        self.run_pipeline(self.input_dir, self.output_dir, self.format, self.threshold)

        call_args = mock_reporter.generate.call_args
        pr = call_args[0][0][0]
        assert pr.classification_confidence == 0.85
        assert pr.mediapipe_confidence == 0.9
        assert pr.combined_confidence == pytest.approx(0.9 * 0.85)

    def test_detection_score_stored_as_mediapipe_confidence(self):
        """HandDetection.detection_score must appear as ProcessingResult.mediapipe_confidence."""
        record = _dummy_record()
        detection = _dummy_detection(detection_score=0.73)

        mock_loader_instance = self.mocks["ImageLoader"].return_value
        mock_loader_instance.scan.return_value = [record]
        mock_loader_instance.load.return_value = _dummy_image(100, 100)

        mock_detector_ctx = self.mocks["HandDetector"].return_value
        mock_detector_ctx.__enter__.return_value = mock_detector_ctx
        mock_detector_ctx.detect_with_flip.return_value = [detection]

        self.mocks["BBoxExtractor"].return_value.extract.return_value = _dummy_bbox()
        self.mocks["ClipClassifier"].return_value.classify.return_value = ("closed_fist", 0, 1.0)

        mock_reporter = self.mocks["Reporter"].return_value
        mock_reporter.generate.return_value = {"results": []}

        self.run_pipeline(self.input_dir, self.output_dir, self.format, self.threshold)

        call_args = mock_reporter.generate.call_args
        pr = call_args[0][0][0]
        assert pr.mediapipe_confidence == 0.73


# ===================================================================
# C.  build_parser() tests
# ===================================================================


    def test_progress_callback_called_with_correct_values(self):
        """progress_callback must be called with (current, total) for each record."""
        records = [
            _dummy_record(path=f"/data/closed_fist/img{i:03d}.jpg")
            for i in range(3)
        ]

        mock_loader_instance = self.mocks["ImageLoader"].return_value
        mock_loader_instance.scan.return_value = records
        mock_loader_instance.load.return_value = _dummy_image(100, 100)

        mock_detector_ctx = self.mocks["HandDetector"].return_value
        mock_detector_ctx.__enter__.return_value = mock_detector_ctx
        mock_detector_ctx.detect_with_flip.return_value = [_dummy_detection()]

        self.mocks["BBoxExtractor"].return_value.extract.return_value = _dummy_bbox()
        self.mocks["ClipClassifier"].return_value.classify.return_value = ("closed_fist", 0, 1.0)
        self.mocks["Reporter"].return_value.generate.return_value = {"results": []}

        call_args_list = []
        def progress_cb(current, total):
            call_args_list.append((current, total))

        self.run_pipeline(
            self.input_dir, self.output_dir, self.format, self.threshold,
            progress_callback=progress_cb,
        )

        assert len(call_args_list) == 3
        assert call_args_list == [(1, 3), (2, 3), (3, 3)]

    def test_progress_callback_none_does_not_crash(self):
        """When progress_callback is None, pipeline runs without errors."""
        record = _dummy_record()

        mock_loader_instance = self.mocks["ImageLoader"].return_value
        mock_loader_instance.scan.return_value = [record]
        mock_loader_instance.load.return_value = _dummy_image(100, 100)

        mock_detector_ctx = self.mocks["HandDetector"].return_value
        mock_detector_ctx.__enter__.return_value = mock_detector_ctx
        mock_detector_ctx.detect_with_flip.return_value = [_dummy_detection()]

        self.mocks["BBoxExtractor"].return_value.extract.return_value = _dummy_bbox()
        self.mocks["ClipClassifier"].return_value.classify.return_value = ("closed_fist", 0, 1.0)
        self.mocks["Reporter"].return_value.generate.return_value = {"results": []}

        self.run_pipeline(
            self.input_dir, self.output_dir, self.format, self.threshold,
        )

        mock_reporter = self.mocks["Reporter"].return_value
        mock_reporter.generate.assert_called_once()

    def test_progress_callback_called_after_failed_image(self):
        """progress_callback is called even when an image processing fails."""
        records = [
            _dummy_record(path="/data/closed_fist/img001.jpg"),
            _dummy_record(path="/data/closed_fist/img002.jpg"),
        ]

        mock_loader_instance = self.mocks["ImageLoader"].return_value
        mock_loader_instance.scan.return_value = records
        mock_loader_instance.load.return_value = _dummy_image(100, 100)

        mock_detector_ctx = self.mocks["HandDetector"].return_value
        mock_detector_ctx.__enter__.return_value = mock_detector_ctx
        mock_detector_ctx.detect_with_flip.return_value = [_dummy_detection()]

        mock_extractor = self.mocks["BBoxExtractor"].return_value
        mock_extractor.extract.side_effect = [RuntimeError("fail"), _dummy_bbox()]

        self.mocks["ClipClassifier"].return_value.classify.return_value = ("closed_fist", 0, 1.0)
        self.mocks["Reporter"].return_value.generate.return_value = {"results": []}

        call_args_list = []
        def progress_cb(current, total):
            call_args_list.append((current, total))

        self.run_pipeline(
            self.input_dir, self.output_dir, self.format, self.threshold,
            progress_callback=progress_cb,
        )

        assert len(call_args_list) == 2
        assert call_args_list == [(1, 2), (2, 2)]


class TestBuildParser:
    """build_parser() – argparse configuration."""

    @pytest.fixture(autouse=True)
    def setup_parser(self):
        """Import build_parser after patch to avoid import issues."""
        with patch("boundingboxer.main.ImageLoader"), \
             patch("boundingboxer.main.HandDetector"), \
             patch("boundingboxer.main.BBoxExtractor"), \
             patch("boundingboxer.main.GestureClassifier"), \
             patch("boundingboxer.main.Exporter"), \
             patch("boundingboxer.main.Reporter"):
            from boundingboxer.main import build_parser
            self.build_parser = build_parser

    # ------------------------------------------------------------------
    # Test 13: process subcommand exists
    # ------------------------------------------------------------------

    def test_process_subcommand_exists(self):
        """build_parser() must define a 'process' subcommand."""
        parser = self.build_parser()
        # Parse empty args to trigger the subparser help
        with pytest.raises(SystemExit):
            parser.parse_args(["process", "--help"])

    # ------------------------------------------------------------------
    # Test 14: process requires --input and --output
    # ------------------------------------------------------------------

    def test_process_requires_input_and_output(self):
        """process subcommand must require --input and --output arguments."""
        parser = self.build_parser()

        # Missing both
        with pytest.raises(SystemExit):
            parser.parse_args(["process"])

        # Missing --output
        with pytest.raises(SystemExit):
            parser.parse_args(["process", "--input", "/some/path"])

        # Both present → should succeed
        args = parser.parse_args([
            "process",
            "--input", "/some/input",
            "--output", "/some/output",
        ])
        assert args.input == "/some/input"
        assert args.output == "/some/output"

    # ------------------------------------------------------------------
    # Test 15: process --format defaults to "yolo"
    # ------------------------------------------------------------------

    def test_process_format_defaults_to_yolo(self):
        """The --format argument must default to 'yolo'."""
        parser = self.build_parser()
        args = parser.parse_args([
            "process",
            "--input", "/in",
            "--output", "/out",
        ])
        assert args.format == "yolo"

    # ------------------------------------------------------------------
    # Test 16: process --confidence defaults to 0.8
    # ------------------------------------------------------------------

    def test_process_confidence_defaults_to_08(self):
        """The --confidence argument must default to 0.8."""
        parser = self.build_parser()
        args = parser.parse_args([
            "process",
            "--input", "/in",
            "--output", "/out",
        ])
        assert args.confidence == pytest.approx(0.8)

    def test_process_format_accepts_coco(self):
        """The --format argument must accept 'coco'."""
        parser = self.build_parser()
        args = parser.parse_args([
            "process",
            "--input", "/in",
            "--output", "/out",
            "--format", "coco",
        ])
        assert args.format == "coco"

    def test_process_confidence_custom_value(self):
        """The --confidence argument must accept a custom float value."""
        parser = self.build_parser()
        args = parser.parse_args([
            "process",
            "--input", "/in",
            "--output", "/out",
            "--confidence", "0.5",
        ])
        assert args.confidence == pytest.approx(0.5)

    # ------------------------------------------------------------------
    # Test 17: review subcommand exists
    # ------------------------------------------------------------------

    def test_review_subcommand_exists(self):
        """build_parser() must define a 'review' subcommand."""
        parser = self.build_parser()
        with pytest.raises(SystemExit):
            parser.parse_args(["review", "--help"])

    # ------------------------------------------------------------------
    # Test 18: review requires --input
    # ------------------------------------------------------------------

    def test_review_input_is_optional(self):
        """review subcommand --input defaults to None when not provided."""
        parser = self.build_parser()

        # Without --input → success, input is None
        args = parser.parse_args(["review"])
        assert args.input is None

        # With --input → success
        args = parser.parse_args([
            "review",
            "--input", "/some/input",
        ])
        assert args.input == "/some/input"

    # ------------------------------------------------------------------
    # Test 19: review --port defaults to 8501
    # ------------------------------------------------------------------

    def test_review_port_defaults_to_8501(self):
        """The --port argument for review must default to 8501."""
        parser = self.build_parser()
        args = parser.parse_args([
            "review",
            "--input", "/in",
        ])
        assert args.port == 8501

    def test_review_port_custom_value(self):
        """The --port argument must accept a custom integer value."""
        parser = self.build_parser()
        args = parser.parse_args([
            "review",
            "--input", "/in",
            "--port", "9999",
        ])
        assert args.port == 9999
