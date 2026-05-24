"""Integration tests for the BoundingBoxer pipeline.

These tests exercise the full pipeline (``run_pipeline``) end-to-end using
synthetic image datasets written to disk.

Most components run **without mocking** (ImageLoader, GestureClassifier,
BBoxExtractor, Exporter, Reporter) -- only HandDetector is replaced with
a test double because this environment uses mediapipe >= 0.10 which does
not provide the legacy ``mp.solutions`` API.  Realistic synthetic landmarks
are fed to the classifier so the full classification→extraction→export→report
chain executes against real code.
"""

import json
from pathlib import Path
from unittest.mock import MagicMock, patch

import cv2
import numpy as np
import pytest

from boundingboxer.config import (
    CLASS_NAMES,
    OUTPUT_DATASET_YAML,
    OUTPUT_LABELS_DIR,
    OUTPUT_REPORT_JSON,
)
from boundingboxer.detector import HandDetection
from boundingboxer.main import run_pipeline


# ================================================================================
# Constants & helpers – test data
# ================================================================================

REQUIRED_ENTRY_KEYS = frozenset({
    "image",
    "detected",
    "detected_class",
    "expected_class",
    "mediapipe_confidence",
    "classification_confidence",
    "combined_confidence",
    "bbox",
    "reviewed",
    "needs_review",
    "manual_override",
})

IMAGE_WIDTH = 200
IMAGE_HEIGHT = 200


def _make_image(width=IMAGE_WIDTH, height=IMAGE_HEIGHT,
                color_bgr=(255, 0, 0)) -> np.ndarray:
    """Create a small synthetic BGR image (solid colour rectangle)."""
    img = np.zeros((height, width, 3), dtype=np.uint8)
    img[:, :] = color_bgr
    return img


def _build_dataset(root: Path, create_images: bool = True) -> Path:
    """Create a synthetic dataset at ``root/data/``.

    Args:
        root: Temporary directory (``tmp_path``).
        create_images: If ``True``, create class subdirectories with images;
            otherwise create an empty ``data/`` directory.

    Returns:
        Path to the data directory (``root/data``).

    Dataset layout::

        data/
          closed_fist/
            img001.jpg
            img002.jpg
          open_palm/
            img003.jpg
          none/
            img004.jpg
    """
    data_dir = root / "data"
    data_dir.mkdir(parents=True, exist_ok=True)

    if create_images:
        for cls_name in CLASS_NAMES:
            (data_dir / cls_name).mkdir(exist_ok=True)

        blue = _make_image(color_bgr=(255, 0, 0))    # BGR – blue channel max
        green = _make_image(color_bgr=(0, 255, 0))   # BGR – green channel max
        red = _make_image(color_bgr=(0, 0, 255))     # BGR – red channel max

        # closed_fist – 2 images
        cv2.imwrite(str(data_dir / "closed_fist" / "img001.jpg"), blue)
        cv2.imwrite(str(data_dir / "closed_fist" / "img002.jpg"), blue)

        # open_palm – 1 image
        cv2.imwrite(str(data_dir / "open_palm" / "img003.jpg"), green)

        # none – 1 image
        cv2.imwrite(str(data_dir / "none" / "img004.jpg"), red)

    return data_dir


def _load_report(path: Path) -> dict:
    """Load the report JSON file at *path* and return it as a dict."""
    with open(path) as fh:
        return json.load(fh)


# ================================================================================
# Constants & helpers – synthetic landmarks
# ================================================================================

def _closed_fist_landmarks() -> np.ndarray:
    """Landmarks where all 4 classifier-relevant fingers are curled.

    For every finger the TIP stays close to the PIP so that
    ``d(MCP→TIP) < d(MCP→PIP) * EXTENDED_RATIO`` → extended_count = 0.
    """
    lm = np.zeros((21, 3), dtype=np.float32)
    lm[0] = [0.5, 0.85, 0.0]          # wrist
    # Index(5-8), middle(9-12), ring(13-16), pinky(17-20): all folded
    for base, mcp_x in [(5, 0.45), (9, 0.50), (13, 0.55), (17, 0.60)]:
        lm[base]     = [mcp_x, 0.70, 0.0]   # MCP
        lm[base + 1] = [mcp_x, 0.55, 0.0]   # PIP
        lm[base + 2] = [mcp_x, 0.55, 0.0]   # DIP (ignored by classifier)
        lm[base + 3] = [mcp_x, 0.56, 0.0]   # TIP  (close to PIP → folded)
    return lm


def _open_palm_landmarks() -> np.ndarray:
    """Landmarks where all 4 classifier-relevant fingers are extended.

    TIP is far from MCP → ``d(MCP→TIP) > d(MCP→PIP) * EXTENDED_RATIO``
    for every finger → extended_count = 4.
    """
    lm = np.zeros((21, 3), dtype=np.float32)
    lm[0] = [0.5, 0.85, 0.0]
    for base, mcp_x in [(5, 0.45), (9, 0.50), (13, 0.55), (17, 0.60)]:
        lm[base]     = [mcp_x, 0.70, 0.0]   # MCP
        lm[base + 1] = [mcp_x, 0.55, 0.0]   # PIP
        lm[base + 2] = [mcp_x, 0.38, 0.0]   # DIP
        lm[base + 3] = [mcp_x, 0.20, 0.0]   # TIP  (far from MCP → extended)
    return lm


def _none_landmarks() -> np.ndarray:
    """Landmarks with exactly 1 finger extended (index) → classifier
    returns ``'none'`` (extended_count == 1 is neither 0 nor ≥ 3)."""
    lm = np.zeros((21, 3), dtype=np.float32)
    lm[0] = [0.5, 0.85, 0.0]
    # Index – EXTENDED
    lm[5] = [0.45, 0.70, 0.0]; lm[6] = [0.45, 0.55, 0.0]
    lm[7] = [0.45, 0.40, 0.0]; lm[8] = [0.45, 0.20, 0.0]
    # Middle, ring, pinky – FOLDED
    for base, mcp_x in [(9, 0.50), (13, 0.55), (17, 0.60)]:
        lm[base]     = [mcp_x, 0.70, 0.0]
        lm[base + 1] = [mcp_x, 0.55, 0.0]
        lm[base + 2] = [mcp_x, 0.55, 0.0]
        lm[base + 3] = [mcp_x, 0.56, 0.0]
    return lm


def _detection(landmarks: np.ndarray, handedness: str = "Right",
               score: float = 0.95) -> HandDetection:
    """Factory for a ``HandDetection`` with the given *landmarks*."""
    return HandDetection(
        landmarks=landmarks,
        handedness=handedness,
        detection_score=score,
    )


# ================================================================================
# Fixture — mock HandDetector with realistic class-specific detections
# ================================================================================


@pytest.fixture
def mock_detector():
    """Replace ``boundingboxer.main.HandDetector`` with a test double.

    The mock supports the context-manager protocol and provides a ``detect()``
    method whose ``side_effect`` can be configured per test.

    Yields the mock **instance** so tests can set ``detect.side_effect``.
    """
    with patch("boundingboxer.main.HandDetector") as mock_cls:
        mock_inst = MagicMock()
        mock_inst.__enter__ = MagicMock(return_value=mock_inst)
        mock_inst.__exit__ = MagicMock(return_value=None)
        mock_inst.close = MagicMock()
        mock_cls.return_value = mock_inst
        yield mock_inst


# ================================================================================
# Test 1 – End-to-end pipeline (YOLO format)
# ================================================================================


class TestPipelineEndToEnd:
    """Main integration test – full pipeline on a synthetic 4-image dataset,
    YOLO format, with class-appropriate hand detections."""

    def test_pipeline_end_to_end(self, tmp_path, mock_detector):
        """Run the full pipeline and verify every output file and report entry."""
        data_dir = _build_dataset(tmp_path)
        output_dir = tmp_path / "output"

        # --- wire up the mock to return class-appropriate detections ----------
        cf_det = _detection(_closed_fist_landmarks())
        op_det = _detection(_open_palm_landmarks())
        no_det = _detection(_none_landmarks())

        # scan order: sorted subdirs → closed_fist, none, open_palm
        #           → closed_fist×2, none×1, open_palm×1
        mock_detector.detect.side_effect = [
            [cf_det], [cf_det], [no_det], [op_det],
        ]

        # --- Act --------------------------------------------------------------
        report = run_pipeline(
            input_dir=str(data_dir),
            output_dir=str(output_dir),
            format="yolo",
            confidence_threshold=0.8,
        )

        # --- Assert – output files exist --------------------------------------
        report_path = output_dir / OUTPUT_REPORT_JSON
        dataset_yaml_path = output_dir / OUTPUT_DATASET_YAML

        assert report_path.is_file(), f"Missing {OUTPUT_REPORT_JSON}"
        assert dataset_yaml_path.is_file(), f"Missing {OUTPUT_DATASET_YAML}"

        # --- Assert – report.json is valid JSON -------------------------------
        report_data = _load_report(report_path)
        assert isinstance(report_data, dict)
        assert "results" in report_data
        assert isinstance(report_data["results"], list)

        # --- Assert – correct number of entries (4 images) --------------------
        results = report_data["results"]
        assert len(results) == 4, f"Expected 4 entries, got {len(results)}"

        # --- Assert – each entry has all 11 required keys ---------------------
        for i, entry in enumerate(results):
            missing = REQUIRED_ENTRY_KEYS - set(entry.keys())
            extra = set(entry.keys()) - REQUIRED_ENTRY_KEYS
            assert not missing, f"Entry {i} missing keys: {missing}"
            assert not extra, f"Entry {i} has unexpected keys: {extra}"

        # --- Assert – the returned report matches the saved file --------------
        assert report == report_data, (
            "Returned report dict must equal saved JSON"
        )

        # --- Assert – expected_class and detected_class make sense ------------
        # scan order: closed_fist×2, none×1, open_palm×1 (alphabetical sort)
        expected_classes = ["closed_fist", "closed_fist", "none", "open_palm"]
        detected_classes = ["closed_fist", "closed_fist", "none", "open_palm"]
        for i, (exp, det) in enumerate(
            zip(expected_classes, detected_classes)
        ):
            assert results[i]["expected_class"] == exp, (
                f"Entry {i}: expected_class mismatch"
            )
            assert results[i]["detected_class"] == det, (
                f"Entry {i}: detected_class mismatch"
            )
            assert results[i]["detected"] is True, (
                f"Entry {i}: detected should be True"
            )

        # --- Assert – bboxes are present (YOLO format, 4 values) --------------
        for i, entry in enumerate(results):
            assert entry["bbox"] is not None, (
                f"Entry {i}: bbox should not be None when detection exists"
            )
            assert isinstance(entry["bbox"], list), (
                f"Entry {i}: bbox should be a list"
            )
            assert len(entry["bbox"]) == 4, (
                f"Entry {i}: YOLO bbox must have 4 values"
            )

        # --- Assert – dataset.yaml contains class names -----------------------
        yaml_content = dataset_yaml_path.read_text()
        for name in CLASS_NAMES:
            assert name in yaml_content, (
                f"Missing class '{name}' in dataset.yaml"
            )


# ================================================================================
# Test 2 – COCO export format
# ================================================================================


class TestPipelineWithCocoFormat:
    """Pipeline end-to-end with COCO export format."""

    def test_pipeline_with_coco_format(self, tmp_path, mock_detector):
        """Pipeline completes without error when ``format='coco'``;
        ``report.json`` and ``dataset.yaml`` are still produced."""
        data_dir = _build_dataset(tmp_path)
        output_dir = tmp_path / "output"

        cf_det = _detection(_closed_fist_landmarks())
        op_det = _detection(_open_palm_landmarks())
        no_det = _detection(_none_landmarks())
        # scan order: closed_fist×2, none×1, open_palm×1
        mock_detector.detect.side_effect = [
            [cf_det], [cf_det], [no_det], [op_det],
        ]

        report = run_pipeline(
            input_dir=str(data_dir),
            output_dir=str(output_dir),
            format="coco",
            confidence_threshold=0.8,
        )

        # report.json must exist with all 4 entries
        report_path = output_dir / OUTPUT_REPORT_JSON
        assert report_path.is_file()
        report_data = _load_report(report_path)
        assert len(report_data["results"]) == 4

        # dataset.yaml should still be generated
        assert (output_dir / OUTPUT_DATASET_YAML).is_file()

        # Report dict must be returned
        assert report is not None


# ================================================================================
# Test 3 – Robustness: missing class folders
# ================================================================================


class TestPipelineHandlesMissingClassFolders:
    """Graceful handling of empty or malformed input directories."""

    def test_pipeline_handles_missing_class_folders(self, tmp_path, mock_detector):
        """Input dir without any class folders → empty results, no crash."""
        data_dir = _build_dataset(tmp_path, create_images=False)
        output_dir = tmp_path / "output"

        report = run_pipeline(
            input_dir=str(data_dir),
            output_dir=str(output_dir),
            format="yolo",
            confidence_threshold=0.8,
        )

        # report.json should exist with empty results list
        report_path = output_dir / OUTPUT_REPORT_JSON
        assert report_path.is_file()
        report_data = _load_report(report_path)
        assert report_data["results"] == []

        # dataset.yaml should still be generated (names, but no images)
        assert (output_dir / OUTPUT_DATASET_YAML).is_file()

        # Returned report must match
        assert report == {"results": []}


# ================================================================================
# Test 4 – Report JSON structure
# ================================================================================


class TestReportJsonStructure:
    """Detailed validation of ``report.json`` entry structure and field types."""

    def test_report_json_structure(self, tmp_path, mock_detector):
        """After pipeline, each report entry has exactly 11 required keys
        with correct types."""
        data_dir = _build_dataset(tmp_path)
        output_dir = tmp_path / "output"

        cf_det = _detection(_closed_fist_landmarks())
        op_det = _detection(_open_palm_landmarks())
        no_det = _detection(_none_landmarks())
        # scan order: closed_fist×2, none×1, open_palm×1
        mock_detector.detect.side_effect = [
            [cf_det], [cf_det], [no_det], [op_det],
        ]

        run_pipeline(
            str(data_dir), str(output_dir), format="yolo",
            confidence_threshold=0.8,
        )

        report_data = _load_report(output_dir / OUTPUT_REPORT_JSON)
        results = report_data["results"]

        for i, entry in enumerate(results):
            # Exactly 11 keys, no more, no less -------------------------------
            entry_keys = set(entry.keys())
            assert entry_keys == REQUIRED_ENTRY_KEYS, (
                f"Entry {i}: expected {sorted(REQUIRED_ENTRY_KEYS)}, "
                f"got {sorted(entry_keys)}"
            )

            # Type checks for every field --------------------------------------
            assert isinstance(entry["image"], str), (
                f"Entry {i}: image not str"
            )
            assert isinstance(entry["detected"], bool), (
                f"Entry {i}: detected not bool"
            )
            assert entry["detected_class"] is None or isinstance(
                entry["detected_class"], str
            ), f"Entry {i}: detected_class wrong type"
            assert isinstance(entry["expected_class"], str), (
                f"Entry {i}: expected_class not str"
            )
            assert isinstance(entry["mediapipe_confidence"], (int, float)), (
                f"Entry {i}: mediapipe_confidence not numeric"
            )
            assert isinstance(entry["classification_confidence"], (int, float)), (
                f"Entry {i}: classification_confidence not numeric"
            )
            assert isinstance(entry["combined_confidence"], (int, float)), (
                f"Entry {i}: combined_confidence not numeric"
            )
            assert entry["bbox"] is None or isinstance(entry["bbox"], list), (
                f"Entry {i}: bbox wrong type"
            )
            assert isinstance(entry["reviewed"], bool), (
                f"Entry {i}: reviewed not bool"
            )
            assert isinstance(entry["needs_review"], bool), (
                f"Entry {i}: needs_review not bool"
            )
            assert isinstance(entry["manual_override"], bool), (
                f"Entry {i}: manual_override not bool"
            )


# ================================================================================
# Test 5 – Output directory tree
# ================================================================================


class TestPipelineOutputStructure:
    """Verify the complete output directory structure after a pipeline run."""

    def test_pipeline_produces_expected_output_structure(self, tmp_path,
                                                         mock_detector):
        """Verify output dir contains ``report.json``, ``dataset.yaml``, and
        ``labels/`` directory with YOLO-format ``.txt`` files."""
        data_dir = _build_dataset(tmp_path)
        output_dir = tmp_path / "output"

        cf_det = _detection(_closed_fist_landmarks())
        op_det = _detection(_open_palm_landmarks())
        no_det = _detection(_none_landmarks())
        # scan order: closed_fist×2, none×1, open_palm×1
        mock_detector.detect.side_effect = [
            [cf_det], [cf_det], [no_det], [op_det],
        ]

        run_pipeline(
            str(data_dir), str(output_dir), format="yolo",
            confidence_threshold=0.8,
        )

        # Guaranteed outputs --------------------------------------------------
        assert (output_dir / OUTPUT_REPORT_JSON).is_file(), (
            f"Missing {OUTPUT_REPORT_JSON}"
        )
        assert (output_dir / OUTPUT_DATASET_YAML).is_file(), (
            f"Missing {OUTPUT_DATASET_YAML}"
        )

        # dataset.yaml structure check ----------------------------------------
        yaml_text = (output_dir / OUTPUT_DATASET_YAML).read_text()
        assert "path:" in yaml_text
        assert "train:" in yaml_text
        assert "val:" in yaml_text
        assert "names:" in yaml_text

        # labels/ directory – must exist because bboxes are present -----------
        labels_dir = output_dir / OUTPUT_LABELS_DIR
        assert labels_dir.is_dir(), (
            "labels/ directory should exist when bboxes are present"
        )

        # Each class that had bboxes should have a subdirectory with .txt files
        expected_label_dirs = {"closed_fist", "open_palm", "none"}
        found_label_dirs = {
            d.name for d in labels_dir.iterdir() if d.is_dir()
        }
        assert found_label_dirs == expected_label_dirs, (
            f"Expected label dirs {expected_label_dirs}, "
            f"got {found_label_dirs}"
        )

        for cls_name in expected_label_dirs:
            cls_dir = labels_dir / cls_name
            txt_files = sorted(cls_dir.glob("*.txt"))
            assert len(txt_files) > 0, (
                f"No label files in labels/{cls_name}/"
            )
            # Each .txt file should contain at least one non-empty line
            for txt_path in txt_files:
                content = txt_path.read_text().strip()
                assert len(content) > 0, (
                    f"Empty label file: {txt_path}"
                )
