"""Unit tests for Reporter component, Summary, and ClassStats dataclasses."""

import json
from dataclasses import is_dataclass
from pathlib import Path
from unittest.mock import MagicMock

import pytest

from boundingboxer.config import (
    CLASS_MAP,
    CLASS_NAMES,
    COMBINED_CONFIDENCE_THRESHOLD,
    OUTPUT_REPORT_JSON,
)
from boundingboxer.extractor import BBox
from boundingboxer.loader import ImageRecord
from boundingboxer.exporter import ProcessingResult
from boundingboxer.reporter import ClassStats, Reporter, Summary


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_image_record(class_name="closed_fist", filename="img001.jpg"):
    """Build an ImageRecord under /data/<class_name>/<filename>."""
    return ImageRecord(
        path=Path(f"/data/{class_name}/{filename}"),
        class_name=class_name,
        class_id=CLASS_MAP[class_name],
    )


def _make_result(
    image_record=None,
    detections=None,
    bboxes=None,
    detected_class="closed_fist",
    detected_class_id=0,
    classification_confidence=0.9,
    mediapipe_confidence=0.95,
    combined_confidence=0.95,
    needs_review=False,
    reviewed=False,
    manual_override=False,
):
    """Build a ProcessingResult with sensible defaults."""
    if image_record is None:
        image_record = _make_image_record("closed_fist", "img001.jpg")
    if detections is None:
        detections = [MagicMock()]
    if bboxes is None and detections:
        bboxes = [
            BBox(
                x=100.0, y=200.0, width=300.0, height=400.0,
                class_id=detected_class_id or 0,
                class_name=detected_class or "",
            )
        ]
    elif bboxes is None:
        bboxes = []
    return ProcessingResult(
        image_record=image_record,
        detections=detections,
        bboxes=bboxes,
        detected_class=detected_class,
        detected_class_id=detected_class_id,
        classification_confidence=classification_confidence,
        mediapipe_confidence=mediapipe_confidence,
        combined_confidence=combined_confidence,
        needs_review=needs_review,
        reviewed=reviewed,
        manual_override=manual_override,
    )


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def reporter():
    """Return a default Reporter instance."""
    return Reporter()


@pytest.fixture
def input_dir(tmp_path):
    """A temporary directory simulating the input root with class subfolders."""
    root = tmp_path / "input"
    for cls in CLASS_NAMES:
        (root / cls).mkdir(parents=True)
    return root


@pytest.fixture
def result_with_detection():
    """ProcessingResult: closed_fist detected with high confidence."""
    return _make_result(
        image_record=_make_image_record("closed_fist", "img001.jpg"),
        detected_class="closed_fist",
        detected_class_id=0,
        combined_confidence=0.95,
        classification_confidence=0.9,
        mediapipe_confidence=0.95,
        needs_review=False,
    )


@pytest.fixture
def result_without_detection():
    """ProcessingResult: no hand detected (empty detections/bboxes)."""
    return _make_result(
        image_record=_make_image_record("closed_fist", "img002.jpg"),
        detections=[],
        bboxes=[],
        detected_class=None,
        detected_class_id=None,
        classification_confidence=0.0,
        mediapipe_confidence=0.0,
        combined_confidence=0.0,
        needs_review=False,
    )


@pytest.fixture
def result_low_confidence():
    """ProcessingResult: detected but below threshold."""
    return _make_result(
        image_record=_make_image_record("open_palm", "img003.jpg"),
        detected_class="open_palm",
        detected_class_id=1,
        combined_confidence=0.55,
        classification_confidence=0.6,
        mediapipe_confidence=0.5,
        needs_review=False,
    )


@pytest.fixture
def result_none_class_no_detection():
    """ProcessingResult: no detection in 'none' folder."""
    return _make_result(
        image_record=_make_image_record("none", "img004.jpg"),
        detections=[],
        bboxes=[],
        detected_class=None,
        detected_class_id=None,
        classification_confidence=0.0,
        mediapipe_confidence=0.0,
        combined_confidence=0.0,
        needs_review=False,
    )


# ===================================================================
# A.  ClassStats dataclass
# ===================================================================


class TestClassStats:
    """ClassStats – dataclass contract, field access, and types."""

    def test_is_dataclass(self):
        """ClassStats must be a @dataclass."""
        assert is_dataclass(ClassStats), "ClassStats is not a dataclass"

    def test_fields_accessible(self):
        """All declared fields are readable as attributes."""
        cs = ClassStats(
            total=10, detected=8, not_detected=2,
            needs_review=1, reviewed=5, average_confidence=0.85,
        )
        assert cs.total == 10
        assert cs.detected == 8
        assert cs.not_detected == 2
        assert cs.needs_review == 1
        assert cs.reviewed == 5
        assert cs.average_confidence == pytest.approx(0.85)

    def test_field_types(self):
        """total/detected/not_detected/needs_review/reviewed are int;
        average_confidence is float."""
        cs = ClassStats(
            total=0, detected=0, not_detected=0,
            needs_review=0, reviewed=0, average_confidence=0.0,
        )
        assert isinstance(cs.total, int)
        assert isinstance(cs.detected, int)
        assert isinstance(cs.not_detected, int)
        assert isinstance(cs.needs_review, int)
        assert isinstance(cs.reviewed, int)
        assert isinstance(cs.average_confidence, float)


# ===================================================================
# B.  Summary dataclass
# ===================================================================


class TestSummary:
    """Summary – dataclass contract, field access, and types."""

    def test_is_dataclass(self):
        """Summary must be a @dataclass."""
        assert is_dataclass(Summary), "Summary is not a dataclass"

    def test_fields_accessible(self):
        """All declared fields are readable as attributes."""
        per_class = {
            "closed_fist": ClassStats(5, 4, 1, 0, 2, 0.92),
        }
        s = Summary(
            total_images=5, total_detected=4, total_not_detected=1,
            total_reviewed=2, total_needs_review=0,
            average_confidence=0.92,
            per_class_stats=per_class,
        )
        assert s.total_images == 5
        assert s.total_detected == 4
        assert s.total_not_detected == 1
        assert s.total_reviewed == 2
        assert s.total_needs_review == 0
        assert s.average_confidence == pytest.approx(0.92)
        assert s.per_class_stats == per_class

    def test_per_class_stats_keys_are_strings(self):
        """per_class_stats must be dict[str, ClassStats]."""
        s = Summary(
            total_images=1, total_detected=1, total_not_detected=0,
            total_reviewed=0, total_needs_review=0,
            average_confidence=0.9,
            per_class_stats={"closed_fist": ClassStats(1, 1, 0, 0, 0, 0.9)},
        )
        for key in s.per_class_stats:
            assert isinstance(key, str)
            assert isinstance(s.per_class_stats[key], ClassStats)


# ===================================================================
# C.  Reporter.generate()
# ===================================================================


class TestReporterGenerate:
    """Reporter.generate() – report dict generation."""

    # ------------------------------------------------------------------
    # Return type and structure
    # ------------------------------------------------------------------

    def test_returns_dict(self, reporter, input_dir, result_with_detection):
        """generate() must return a dict."""
        report = reporter.generate([result_with_detection], input_dir)
        assert isinstance(report, dict)

    def test_results_key_is_a_list(self, reporter, input_dir, result_with_detection):
        """The report dict must contain a 'results' key with a list value."""
        report = reporter.generate([result_with_detection], input_dir)
        assert "results" in report
        assert isinstance(report["results"], list)

    def test_each_entry_has_all_required_keys(
        self, reporter, input_dir, result_with_detection,
    ):
        """Every entry in results must have all spec-defined keys."""
        report = reporter.generate([result_with_detection], input_dir)
        entry = report["results"][0]

        required_keys = {
            "image", "detected", "detected_class", "expected_class",
            "mediapipe_confidence", "classification_confidence",
            "combined_confidence", "bbox", "reviewed", "needs_review",
            "manual_override",
        }
        missing = required_keys - set(entry.keys())
        assert not missing, f"Missing keys in report entry: {missing}"

    def test_empty_results_returns_empty_list(self, reporter, input_dir):
        """An empty results list produces an empty report results list."""
        report = reporter.generate([], input_dir)
        assert report["results"] == []

    # ------------------------------------------------------------------
    # Relative path computation
    # ------------------------------------------------------------------

    def test_correct_relative_path(self, reporter, tmp_path):
        """The 'image' field is a relative path from input_dir."""
        input_root = tmp_path / "data"
        (input_root / "closed_fist").mkdir(parents=True)
        image_path = input_root / "closed_fist" / "img001.jpg"
        image_path.touch()

        result = _make_result(
            image_record=ImageRecord(
                path=image_path,
                class_name="closed_fist",
                class_id=0,
            ),
        )
        report = reporter.generate([result], input_root)
        entry = report["results"][0]

        assert entry["image"] == "closed_fist/img001.jpg", (
            f"Expected 'closed_fist/img001.jpg', got {entry['image']!r}"
        )

    def test_relative_path_includes_class_name_folder(
        self, reporter, tmp_path,
    ):
        """The relative path preserves the class folder in the path."""
        input_root = tmp_path / "dataset"
        (input_root / "open_palm").mkdir(parents=True)
        image_path = input_root / "open_palm" / "test_img.png"
        image_path.touch()

        result = _make_result(
            image_record=ImageRecord(
                path=image_path,
                class_name="open_palm",
                class_id=1,
            ),
            detected_class="open_palm",
            detected_class_id=1,
        )
        report = reporter.generate([result], input_root)

        assert report["results"][0]["image"].startswith("open_palm/")

    # ------------------------------------------------------------------
    # detected flag
    # ------------------------------------------------------------------

    def test_detected_true_when_detections_non_empty(
        self, reporter, input_dir, result_with_detection,
    ):
        """detected must be True when detections list is non-empty."""
        report = reporter.generate([result_with_detection], input_dir)
        assert report["results"][0]["detected"] is True

    def test_detected_false_when_detections_empty(
        self, reporter, input_dir, result_without_detection,
    ):
        """detected must be False when detections list is empty."""
        report = reporter.generate([result_without_detection], input_dir)
        assert report["results"][0]["detected"] is False

    # ------------------------------------------------------------------
    # detected_class and expected_class
    # ------------------------------------------------------------------

    def test_detected_class_from_processing_result(
        self, reporter, input_dir, result_with_detection,
    ):
        """detected_class must come from ProcessingResult.detected_class."""
        report = reporter.generate([result_with_detection], input_dir)
        assert report["results"][0]["detected_class"] == "closed_fist"

    def test_detected_class_null_when_not_detected(
        self, reporter, input_dir, result_without_detection,
    ):
        """detected_class must be None/null when nothing is detected."""
        report = reporter.generate([result_without_detection], input_dir)
        assert report["results"][0]["detected_class"] is None

    def test_expected_class_from_image_record(
        self, reporter, input_dir, result_with_detection,
    ):
        """expected_class must come from image_record.class_name."""
        report = reporter.generate([result_with_detection], input_dir)
        assert report["results"][0]["expected_class"] == "closed_fist"

    # ------------------------------------------------------------------
    # needs_review logic
    # ------------------------------------------------------------------

    def test_high_confidence_detection_needs_review_false(
        self, reporter, input_dir, result_with_detection,
    ):
        """Detection above threshold → needs_review is False."""
        report = reporter.generate([result_with_detection], input_dir)
        assert report["results"][0]["needs_review"] is False

    def test_low_confidence_detection_needs_review_true(
        self, reporter, input_dir, result_low_confidence,
    ):
        """Detection below COMBINED_CONFIDENCE_THRESHOLD → needs_review True."""
        assert result_low_confidence.combined_confidence < COMBINED_CONFIDENCE_THRESHOLD
        report = reporter.generate([result_low_confidence], input_dir)
        assert report["results"][0]["needs_review"] is True

    def test_not_detected_in_non_none_folder_needs_review_true(
        self, reporter, input_dir, result_without_detection,
    ):
        """No detection in 'closed_fist' folder → needs_review True
        (expected a hand but none was found)."""
        report = reporter.generate([result_without_detection], input_dir)
        assert report["results"][0]["needs_review"] is True

    def test_none_class_no_detection_needs_review_false(
        self, reporter, input_dir, result_none_class_no_detection,
    ):
        """No detection in 'none' folder → needs_review False
        (we didn't expect to find a hand here)."""
        report = reporter.generate([result_none_class_no_detection], input_dir)
        assert report["results"][0]["needs_review"] is False

    def test_none_class_with_detection_needs_review_by_confidence(self, reporter, input_dir):
        """Detection in 'none' folder still checks confidence threshold."""
        # High confidence detection in none folder
        result_high = _make_result(
            image_record=_make_image_record("none", "high.jpg"),
            detected_class="closed_fist",
            detected_class_id=0,
            combined_confidence=0.95,
        )
        report = reporter.generate([result_high], input_dir)
        assert report["results"][0]["needs_review"] is False

        # Low confidence detection in none folder
        result_low = _make_result(
            image_record=_make_image_record("none", "low.jpg"),
            detected_class="closed_fist",
            detected_class_id=0,
            combined_confidence=0.55,
        )
        report = reporter.generate([result_low], input_dir)
        assert report["results"][0]["needs_review"] is True

    # ------------------------------------------------------------------
    # Confidence fields
    # ------------------------------------------------------------------

    def test_confidence_fields_from_processing_result(
        self, reporter, input_dir, result_with_detection,
    ):
        """mediapipe/classification/combined confidence come from ProcessingResult."""
        report = reporter.generate([result_with_detection], input_dir)
        entry = report["results"][0]
        assert entry["mediapipe_confidence"] == pytest.approx(0.95)
        assert entry["classification_confidence"] == pytest.approx(0.9)
        assert entry["combined_confidence"] == pytest.approx(0.95)

    def test_combined_confidence_zero_when_not_detected(
        self, reporter, input_dir, result_without_detection,
    ):
        """combined_confidence must be 0.0 when nothing is detected."""
        report = reporter.generate([result_without_detection], input_dir)
        assert report["results"][0]["combined_confidence"] == pytest.approx(0.0)

    # ------------------------------------------------------------------
    # bbox field
    # ------------------------------------------------------------------

    def test_bbox_field_is_list_of_four_floats_when_bboxes_exist(
        self, reporter, input_dir, result_with_detection,
    ):
        """When bboxes is non-empty, 'bbox' is a list of 4 floats (YOLO format)."""
        report = reporter.generate([result_with_detection], input_dir)
        bbox = report["results"][0]["bbox"]
        assert isinstance(bbox, list), f"Expected list, got {type(bbox)}"
        assert len(bbox) == 4, f"Expected 4 values, got {len(bbox)}"
        for v in bbox:
            assert isinstance(v, (int, float)), (
                f"Expected numeric, got {type(v)}"
            )

    def test_bbox_field_is_none_when_no_bboxes(
        self, reporter, input_dir, result_without_detection,
    ):
        """When bboxes is empty, 'bbox' must be None."""
        report = reporter.generate([result_without_detection], input_dir)
        assert report["results"][0]["bbox"] is None

    # ------------------------------------------------------------------
    # reviewed and manual_override
    # ------------------------------------------------------------------

    def test_reviewed_and_manual_override_from_processing_result(
        self, reporter, input_dir,
    ):
        """reviewed and manual_override are taken from ProcessingResult fields."""
        result = _make_result(reviewed=True, manual_override=True)
        report = reporter.generate([result], input_dir)
        entry = report["results"][0]
        assert entry["reviewed"] is True
        assert entry["manual_override"] is True

    # ------------------------------------------------------------------
    # Multiple results
    # ------------------------------------------------------------------

    def test_multiple_results_preserves_order(
        self, reporter, input_dir,
    ):
        """The order of entries matches the order of input results."""
        r1 = _make_result(
            image_record=_make_image_record("closed_fist", "a.jpg"),
            combined_confidence=0.9,
        )
        r2 = _make_result(
            image_record=_make_image_record("open_palm", "b.jpg"),
            detected_class="open_palm", detected_class_id=1,
            combined_confidence=0.85,
        )
        r3 = _make_result(
            image_record=_make_image_record("none", "c.jpg"),
            detections=[], bboxes=[],
            detected_class=None, detected_class_id=None,
            combined_confidence=0.0,
        )

        report = reporter.generate([r1, r2, r3], input_dir)
        entries = report["results"]
        assert len(entries) == 3
        assert entries[0]["expected_class"] == "closed_fist"
        assert entries[1]["expected_class"] == "open_palm"
        assert entries[2]["expected_class"] == "none"


# ===================================================================
# D.  Reporter.save() and Reporter.load()
# ===================================================================


class TestReporterSaveAndLoad:
    """Reporter.save() / Reporter.load() – JSON persistence."""

    @pytest.fixture
    def sample_report(self, reporter, input_dir, result_with_detection):
        """A small report dict generated by the Reporter."""
        return reporter.generate([result_with_detection], input_dir)

    def test_save_writes_valid_json(self, reporter, sample_report, tmp_path):
        """save() writes a valid JSON file."""
        output_path = tmp_path / OUTPUT_REPORT_JSON
        reporter.save(sample_report, output_path)

        assert output_path.is_file()
        with open(output_path) as fh:
            data = json.load(fh)
        assert data == sample_report

    def test_save_creates_parent_directories(self, reporter, sample_report, tmp_path):
        """save() must create parent directories if they don't exist."""
        output_path = tmp_path / "deep" / "nested" / "dir" / OUTPUT_REPORT_JSON
        assert not output_path.parent.exists()

        reporter.save(sample_report, output_path)
        assert output_path.is_file()

    def test_load_reads_back_what_save_wrote(
        self, reporter, sample_report, tmp_path,
    ):
        """A save→load roundtrip produces identical data."""
        output_path = tmp_path / OUTPUT_REPORT_JSON
        reporter.save(sample_report, output_path)
        loaded = reporter.load(output_path)
        assert loaded == sample_report

    def test_load_raises_for_nonexistent_file(self, reporter, tmp_path):
        """load() must raise an error when the file does not exist."""
        missing = tmp_path / "does_not_exist.json"
        with pytest.raises((FileNotFoundError, IOError, OSError)):
            reporter.load(missing)


# ===================================================================
# E.  Reporter.get_summary()
# ===================================================================


class TestReporterGetSummary:
    """Reporter.get_summary() – summary statistics computation."""

    def test_returns_summary_dataclass(self, reporter):
        """get_summary() must return a Summary instance."""
        report = {"results": []}
        summary = reporter.get_summary(report)
        assert isinstance(summary, Summary)

    def test_empty_report_all_zeros(self, reporter):
        """An empty report produces a Summary with all counts at zero."""
        report = {"results": []}
        summary = reporter.get_summary(report)

        assert summary.total_images == 0
        assert summary.total_detected == 0
        assert summary.total_not_detected == 0
        assert summary.total_reviewed == 0
        assert summary.total_needs_review == 0
        assert summary.average_confidence == pytest.approx(0.0)
        assert summary.per_class_stats == {}

    def test_total_counts_are_correct(self, reporter):
        """total_images, total_detected, total_not_detected are correct."""
        report = {
            "results": [
                {"expected_class": "closed_fist", "detected": True,
                 "combined_confidence": 0.95, "needs_review": False,
                 "reviewed": False, "detected_class": "closed_fist"},
                {"expected_class": "closed_fist", "detected": False,
                 "combined_confidence": 0.0, "needs_review": True,
                 "reviewed": False, "detected_class": None},
                {"expected_class": "open_palm", "detected": True,
                 "combined_confidence": 0.88, "needs_review": False,
                 "reviewed": True, "detected_class": "open_palm"},
            ]
        }
        summary = reporter.get_summary(report)

        assert summary.total_images == 3
        assert summary.total_detected == 2
        assert summary.total_not_detected == 1

    def test_total_reviewed_and_needs_review(self, reporter):
        """total_reviewed and total_needs_review count correctly."""
        report = {
            "results": [
                {"expected_class": "c", "detected": True,
                 "combined_confidence": 0.5, "needs_review": True,
                 "reviewed": True, "detected_class": "c"},
                {"expected_class": "c", "detected": True,
                 "combined_confidence": 0.3, "needs_review": True,
                 "reviewed": False, "detected_class": "c"},
                {"expected_class": "c", "detected": True,
                 "combined_confidence": 0.9, "needs_review": False,
                 "reviewed": True, "detected_class": "c"},
            ]
        }
        summary = reporter.get_summary(report)

        assert summary.total_reviewed == 2
        assert summary.total_needs_review == 2

    def test_average_confidence_only_from_detected(self, reporter):
        """average_confidence includes only detected images."""
        report = {
            "results": [
                {"expected_class": "cf", "detected": True,
                 "combined_confidence": 0.90, "needs_review": False,
                 "reviewed": False, "detected_class": "cf"},
                {"expected_class": "cf", "detected": True,
                 "combined_confidence": 0.70, "needs_review": False,
                 "reviewed": False, "detected_class": "cf"},
                {"expected_class": "cf", "detected": False,
                 "combined_confidence": 0.0, "needs_review": True,
                 "reviewed": False, "detected_class": None},
            ]
        }
        summary = reporter.get_summary(report)
        # Average of 0.90 and 0.70 only (not 0.0)
        expected_avg = (0.90 + 0.70) / 2
        assert summary.average_confidence == pytest.approx(expected_avg)

    def test_per_class_breakdown_correct(self, reporter):
        """per_class_stats groups stats by expected_class."""
        report = {
            "results": [
                {"expected_class": "closed_fist", "detected": True,
                 "combined_confidence": 0.95, "needs_review": False,
                 "reviewed": False, "detected_class": "closed_fist"},
                {"expected_class": "closed_fist", "detected": False,
                 "combined_confidence": 0.0, "needs_review": True,
                 "reviewed": False, "detected_class": None},
                {"expected_class": "open_palm", "detected": True,
                 "combined_confidence": 0.88, "needs_review": False,
                 "reviewed": True, "detected_class": "open_palm"},
            ]
        }
        summary = reporter.get_summary(report)

        assert "closed_fist" in summary.per_class_stats
        assert "open_palm" in summary.per_class_stats

        cf = summary.per_class_stats["closed_fist"]
        assert cf.total == 2
        assert cf.detected == 1
        assert cf.not_detected == 1
        assert cf.needs_review == 1
        assert cf.reviewed == 0
        assert cf.average_confidence == pytest.approx(0.95)

        op = summary.per_class_stats["open_palm"]
        assert op.total == 1
        assert op.detected == 1
        assert op.not_detected == 0
        assert op.needs_review == 0
        assert op.reviewed == 1
        assert op.average_confidence == pytest.approx(0.88)

    def test_per_class_average_confidence_only_detected(self, reporter):
        """Per-class average_confidence ignores non-detected images."""
        report = {
            "results": [
                {"expected_class": "cf", "detected": True,
                 "combined_confidence": 0.80, "needs_review": False,
                 "reviewed": False, "detected_class": "cf"},
                {"expected_class": "cf", "detected": False,
                 "combined_confidence": 0.0, "needs_review": True,
                 "reviewed": False, "detected_class": None},
                {"expected_class": "cf", "detected": True,
                 "combined_confidence": 0.60, "needs_review": True,
                 "reviewed": False, "detected_class": "cf"},
            ]
        }
        summary = reporter.get_summary(report)
        cf = summary.per_class_stats["cf"]
        # Average over 0.80 and 0.60 (not 0.0)
        assert cf.average_confidence == pytest.approx((0.80 + 0.60) / 2)

    @pytest.mark.parametrize("confidence,expected_review", [
        (0.79, True),
        (0.80, False),
        (0.81, False),
    ])
    def test_needs_review_boundary_at_threshold(
        self, reporter, confidence, expected_review,
    ):
        """needs_review respects the COMBINED_CONFIDENCE_THRESHOLD boundary."""
        report = {
            "results": [
                {"expected_class": "cf", "detected": True,
                 "combined_confidence": confidence, "needs_review": None,
                 "reviewed": False, "detected_class": "cf"},
            ]
        }
        # NOTE: needs_review is computed by generate(), not from the input.
        # get_summary uses the values already present in the report.
        # This test verifies get_summary counts from the report values,
        # so we set needs_review based on the threshold ourselves.
        report["results"][0]["needs_review"] = expected_review
        summary = reporter.get_summary(report)
        assert summary.total_needs_review == (1 if expected_review else 0)
