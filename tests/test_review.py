"""Unit tests for the Review UI – logic.py pure functions and app.py imports."""

import json
import sys
from pathlib import Path

import pytest

from boundingboxer.config import OUTPUT_REPORT_JSON
from boundingboxer.review.logic import (
    bbox_pixels_to_yolo,
    bbox_yolo_to_pixels,
    build_image_path,
    filter_results,
    load_report,
    save_report,
)


# =========================================================================
# Helpers – sample report entries that match the report.json structure
# =========================================================================


def _make_entry(
    expected_class="closed_fist",
    detected=True,
    detected_class="closed_fist",
    mediapipe_confidence=0.95,
    classification_confidence=0.90,
    combined_confidence=0.95,
    bbox=None,
    reviewed=False,
    needs_review=False,
    manual_override=False,
    image="closed_fist/img001.jpg",
):
    """Build a single report entry dict matching the report.json schema."""
    if bbox is None and detected:
        bbox = [0.45, 0.50, 0.30, 0.40]
    return {
        "image": image,
        "detected": detected,
        "detected_class": detected_class,
        "expected_class": expected_class,
        "mediapipe_confidence": mediapipe_confidence,
        "classification_confidence": classification_confidence,
        "combined_confidence": combined_confidence,
        "bbox": bbox,
        "reviewed": reviewed,
        "needs_review": needs_review,
        "manual_override": manual_override,
    }


def _make_report(entries):
    """Build a full report dict from a list of entries."""
    return {"results": entries}


# =========================================================================
# A.  filter_results()
# =========================================================================


class TestFilterResults:
    """filter_results() – filtering report entries by various criteria."""

    # ------------------------------------------------------------------
    # Empty / no-op
    # ------------------------------------------------------------------

    def test_empty_list_returns_empty(self):
        """An empty entry list produces an empty filtered list."""
        assert filter_results([]) == []

    def test_no_filters_returns_all(self):
        """With all filters off, every entry is returned."""
        entries = [
            _make_entry(expected_class="closed_fist", needs_review=False, reviewed=False),
            _make_entry(expected_class="open_palm", needs_review=True, reviewed=True),
            _make_entry(expected_class="none", needs_review=False, reviewed=True),
        ]
        result = filter_results(entries)
        assert len(result) == 3

    # ------------------------------------------------------------------
    # only_needs_review
    # ------------------------------------------------------------------

    def test_only_needs_review_true(self):
        """only_needs_review=True returns only entries with needs_review=True."""
        entries = [
            _make_entry(image="a", needs_review=True),
            _make_entry(image="b", needs_review=False),
            _make_entry(image="c", needs_review=True),
        ]
        result = filter_results(entries, only_needs_review=True)
        assert len(result) == 2
        assert all(r["needs_review"] for r in result)

    def test_only_needs_review_false_is_noop(self):
        """only_needs_review=False should not filter anything."""
        entries = [
            _make_entry(image="a", needs_review=True),
            _make_entry(image="b", needs_review=False),
        ]
        result = filter_results(entries, only_needs_review=False)
        assert len(result) == 2

    # ------------------------------------------------------------------
    # only_unreviewed
    # ------------------------------------------------------------------

    def test_only_unreviewed_true(self):
        """only_unreviewed=True returns only entries with reviewed=False."""
        entries = [
            _make_entry(image="a", reviewed=False),
            _make_entry(image="b", reviewed=True),
            _make_entry(image="c", reviewed=False),
        ]
        result = filter_results(entries, only_unreviewed=True)
        assert len(result) == 2
        assert all(not r["reviewed"] for r in result)

    def test_only_unreviewed_false_is_noop(self):
        """only_unreviewed=False should not filter anything."""
        entries = [
            _make_entry(image="a", reviewed=False),
            _make_entry(image="b", reviewed=True),
        ]
        result = filter_results(entries, only_unreviewed=False)
        assert len(result) == 2

    # ------------------------------------------------------------------
    # min_confidence
    # ------------------------------------------------------------------

    def test_min_confidence_filters_low_confidence_entries(self):
        """Entries with combined_confidence below min_confidence are excluded."""
        entries = [
            _make_entry(image="a", combined_confidence=0.95),
            _make_entry(image="b", combined_confidence=0.55),
            _make_entry(image="c", combined_confidence=0.60),
        ]
        result = filter_results(entries, min_confidence=0.60)
        assert len(result) == 2
        assert all(r["combined_confidence"] >= 0.60 for r in result)

    def test_min_confidence_zero_passes_all(self):
        """min_confidence=0.0 keeps every entry."""
        entries = [
            _make_entry(image="a", combined_confidence=0.0),
            _make_entry(image="b", combined_confidence=0.95),
        ]
        result = filter_results(entries, min_confidence=0.0)
        assert len(result) == 2

    def test_min_confidence_one_excludes_all_but_perfect(self):
        """min_confidence=1.0 keeps only entries with combined_confidence >= 1.0."""
        entries = [
            _make_entry(image="a", combined_confidence=0.99),
            _make_entry(image="b", combined_confidence=1.00),
        ]
        result = filter_results(entries, min_confidence=1.0)
        assert len(result) == 1
        assert result[0]["combined_confidence"] == 1.0

    # ------------------------------------------------------------------
    # class_filter
    # ------------------------------------------------------------------

    def test_class_filter_exact_match(self):
        """class_filter matches the expected_class field exactly."""
        entries = [
            _make_entry(image="a", expected_class="closed_fist"),
            _make_entry(image="b", expected_class="open_palm"),
            _make_entry(image="c", expected_class="closed_fist"),
        ]
        result = filter_results(entries, class_filter="closed_fist")
        assert len(result) == 2
        assert all(r["expected_class"] == "closed_fist" for r in result)

    def test_class_filter_none_ignored(self):
        """class_filter=None means no class filtering."""
        entries = [
            _make_entry(image="a", expected_class="closed_fist"),
            _make_entry(image="b", expected_class="open_palm"),
        ]
        result = filter_results(entries, class_filter=None)
        assert len(result) == 2

    def test_class_filter_all_ignored(self):
        """class_filter='all' means no class filtering."""
        entries = [
            _make_entry(image="a", expected_class="closed_fist"),
            _make_entry(image="b", expected_class="open_palm"),
        ]
        result = filter_results(entries, class_filter="all")
        assert len(result) == 2

    # ------------------------------------------------------------------
    # Combined filters
    # ------------------------------------------------------------------

    def test_combined_filters_needs_review_and_class(self):
        """Combining only_needs_review and class_filter narrows correctly."""
        entries = [
            _make_entry(image="a", needs_review=True, expected_class="closed_fist"),
            _make_entry(image="b", needs_review=False, expected_class="closed_fist"),
            _make_entry(image="c", needs_review=True, expected_class="open_palm"),
        ]
        result = filter_results(
            entries,
            only_needs_review=True,
            class_filter="closed_fist",
        )
        assert len(result) == 1
        assert result[0]["image"] == "a"

    def test_combined_filters_all_active(self):
        """All filters active simultaneously narrow results correctly."""
        entries = [
            # passes all
            _make_entry(
                image="a", needs_review=True, reviewed=False,
                combined_confidence=0.90, expected_class="closed_fist",
            ),
            # wrong class
            _make_entry(
                image="b", needs_review=True, reviewed=False,
                combined_confidence=0.90, expected_class="open_palm",
            ),
            # low confidence
            _make_entry(
                image="c", needs_review=True, reviewed=False,
                combined_confidence=0.50, expected_class="closed_fist",
            ),
            # already reviewed
            _make_entry(
                image="d", needs_review=True, reviewed=True,
                combined_confidence=0.90, expected_class="closed_fist",
            ),
        ]
        result = filter_results(
            entries,
            min_confidence=0.80,
            only_needs_review=True,
            only_unreviewed=True,
            class_filter="closed_fist",
        )
        assert len(result) == 1
        assert result[0]["image"] == "a"


# =========================================================================
# B.  bbox_yolo_to_pixels()
# =========================================================================


class TestBboxYoloToPixels:
    """bbox_yolo_to_pixels() – YOLO-normalized coordinates to pixel coordinates."""

    def test_normal_conversion(self):
        """Standard YOLO bbox converts to correct pixel coordinates."""
        result = bbox_yolo_to_pixels(
            bbox_yolo=[0.4, 0.5, 0.3, 0.4],
            img_w=640,
            img_h=480,
        )
        assert isinstance(result, dict)
        # cx=0.4*640=256, cy=0.5*480=240, w=0.3*640=192, h=0.4*480=192
        # x = cx - w/2 = 256 - 96 = 160
        # y = cy - h/2 = 240 - 96 = 144
        assert result["x"] == 160
        assert result["y"] == 144
        assert result["width"] == 192
        assert result["height"] == 192

    def test_none_input_returns_none(self):
        """A None bbox returns None."""
        assert bbox_yolo_to_pixels(None, 640, 480) is None

    def test_full_image_bbox(self):
        """A bbox covering the entire image [0.0, 0.0, 1.0, 1.0]."""
        result = bbox_yolo_to_pixels(
            bbox_yolo=[0.0, 0.0, 1.0, 1.0],
            img_w=800,
            img_h=600,
        )
        # cx=0, cy=0 → x = -400, y = -300 but here w=800, h=600 so bbox fills
        # Actually: x = 0*800 - 800/2 = -400, width = 800
        # But that would mean the center is at (0,0) which means half outside
        # Correction: center at (0.0, 0.0) means x = -400, y = -300 with w=800, h=600
        assert result["x"] == -400
        assert result["y"] == -300
        assert result["width"] == 800
        assert result["height"] == 600

    def test_all_values_are_int(self):
        """All returned coordinate values are integers."""
        result = bbox_yolo_to_pixels(
            bbox_yolo=[0.33, 0.67, 0.25, 0.50],
            img_w=640,
            img_h=480,
        )
        assert isinstance(result["x"], int)
        assert isinstance(result["y"], int)
        assert isinstance(result["width"], int)
        assert isinstance(result["height"], int)


# =========================================================================
# C.  bbox_pixels_to_yolo()
# =========================================================================


class TestBboxPixelsToYolo:
    """bbox_pixels_to_yolo() – pixel coordinates to YOLO-normalized format."""

    def test_normal_conversion(self):
        """Standard pixel coordinates convert to normalized YOLO format."""
        result = bbox_pixels_to_yolo(
            x=160, y=144, width=192, height=192,
            img_w=640, img_h=480,
        )
        assert isinstance(result, list)
        assert len(result) == 4
        # cx = (160 + 192/2)/640 = (160+96)/640 = 256/640 = 0.4
        # cy = (144 + 192/2)/480 = (144+96)/480 = 240/480 = 0.5
        # w = 192/640 = 0.3
        # h = 192/480 = 0.4
        assert result == pytest.approx([0.4, 0.5, 0.3, 0.4])

    def test_roundtrip_yolo_pixels_yolo(self):
        """Converting yolo→pixels→yolo returns the original normalized values."""
        original = [0.35, 0.55, 0.25, 0.35]
        pixels = bbox_yolo_to_pixels(original, img_w=800, img_h=600)
        recovered = bbox_pixels_to_yolo(
            x=pixels["x"],
            y=pixels["y"],
            width=pixels["width"],
            height=pixels["height"],
            img_w=800,
            img_h=600,
        )
        assert recovered == pytest.approx(original)

    def test_all_values_normalized_zero_to_one(self):
        """All values in the result must be between 0.0 and 1.0."""
        result = bbox_pixels_to_yolo(
            x=100, y=50, width=300, height=200,
            img_w=400, img_h=300,
        )
        for v in result:
            assert 0.0 <= v <= 1.0, f"Value {v} is outside [0, 1]"


# =========================================================================
# D.  load_report() / save_report()
# =========================================================================


class TestLoadSaveReport:
    """load_report() / save_report() – report.json I/O via Reporter."""

    def _write_report_json(self, path, entries):
        """Helper to write a valid report.json to a temporary path."""
        report = _make_report(entries)
        path.write_text(json.dumps(report))
        return report

    def test_load_report_reads_valid_file(self, tmp_path):
        """load_report() returns the parsed report dict."""
        report_dir = tmp_path / "output"
        report_dir.mkdir()
        self._write_report_json(
            report_dir / OUTPUT_REPORT_JSON,
            [_make_entry(image="a.jpg")],
        )
        report = load_report(report_dir)
        assert isinstance(report, dict)
        assert "results" in report
        assert len(report["results"]) == 1
        assert report["results"][0]["image"] == "a.jpg"

    def test_save_report_roundtrip(self, tmp_path):
        """A report saved and then re-loaded produces identical data."""
        report_dir = tmp_path / "output"
        report_dir.mkdir()
        expected = _make_report([
            _make_entry(image="a.jpg"),
            _make_entry(image="b.jpg", expected_class="open_palm"),
        ])
        save_report(expected, report_dir)
        loaded = load_report(report_dir)
        assert loaded == expected

    def test_load_report_raises_for_missing_file(self, tmp_path):
        """load_report() raises an error when report.json does not exist."""
        empty_dir = tmp_path / "empty"
        empty_dir.mkdir()
        with pytest.raises((FileNotFoundError, IOError, OSError)):
            load_report(empty_dir)


# =========================================================================
# E.  build_image_path()
# =========================================================================


class TestBuildImagePath:
    """build_image_path() – constructing full paths to image files."""

    def test_normal_path_construction(self):
        """Builds a path from input_dir, 'images', and the relative image path."""
        result = build_image_path(
            Path("/data/output"),
            "closed_fist/img001.jpg",
        )
        assert isinstance(result, Path)
        assert result == Path("/data/output/images/closed_fist/img001.jpg")

    def test_nested_subdirectories(self):
        """Works with deeply nested relative image paths."""
        result = build_image_path(
            Path("/data/output"),
            "closed_fist/subdir/img001.jpg",
        )
        assert result == Path("/data/output/images/closed_fist/subdir/img001.jpg")

    def test_returns_path_object(self):
        """Return value is a pathlib.Path instance."""
        result = build_image_path(Path("/tmp/out"), "a.jpg")
        assert isinstance(result, Path)


# =========================================================================
# G.  App imports (smoke tests)
# =========================================================================


class TestAppImports:
    """Smoke tests ensuring review/app.py is importable and wired correctly."""

    def test_app_module_imports_without_error(self):
        """The review.app module can be imported without raising exceptions."""
        # Reload needed because streamlit may already be imported
        if "boundingboxer.review.app" in sys.modules:
            del sys.modules["boundingboxer.review.app"]
        try:
            from boundingboxer.review import app  # noqa: F401
        except Exception as exc:
            pytest.fail(f"Importing review.app raised {type(exc).__name__}: {exc}")

    def test_app_has_streamlit_attributes(self):
        """The app module exposes Streamlit functions after import."""
        from boundingboxer.review import app

        assert hasattr(app, "st"), "app module is missing 'st' (streamlit)"
        # Verify that st is indeed the streamlit module (duck-typing check)
        st_mod = app.st
        assert hasattr(st_mod, "title"), "streamlit module lacks 'title'"
        assert hasattr(st_mod, "write"), "streamlit module lacks 'write'"
