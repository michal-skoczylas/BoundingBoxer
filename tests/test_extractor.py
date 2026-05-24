"""Unit tests for BBoxExtractor component."""

from dataclasses import is_dataclass

import numpy as np
import pytest

from boundingboxer.config import BBOX_PADDING, CLASS_NAMES
from boundingboxer.detector import HandDetection
from boundingboxer.extractor import BBox, BBoxExtractor


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def extractor():
    """Return a default BBoxExtractor instance."""
    return BBoxExtractor()


@pytest.fixture
def detection_center():
    """HandDetection with landmarks concentrated in the center of the image.

    x coords: 0.40 – 0.60, y coords: 0.35 – 0.65 (normalized).
    """
    landmarks = np.zeros((21, 3), dtype=np.float32)
    landmarks[:, 0] = np.linspace(0.40, 0.60, 21)
    landmarks[:, 1] = np.linspace(0.35, 0.65, 21)
    return HandDetection(
        landmarks=landmarks, handedness="Right", detection_score=0.95
    )


@pytest.fixture
def detection_edge():
    """HandDetection with landmarks near image edges.

    x coords: 0.01 – 0.99, y coords: 0.01 – 0.99 (normalized).
    """
    landmarks = np.zeros((21, 3), dtype=np.float32)
    landmarks[:, 0] = np.linspace(0.01, 0.99, 21)
    landmarks[:, 1] = np.linspace(0.01, 0.99, 21)
    return HandDetection(
        landmarks=landmarks, handedness="Left", detection_score=0.88
    )


@pytest.fixture
def detection_single_point():
    """HandDetection where all 21 landmarks share the exact same coordinates."""
    landmarks = np.full((21, 3), [0.5, 0.5, 0.0], dtype=np.float32)
    return HandDetection(
        landmarks=landmarks, handedness="Right", detection_score=0.90
    )


# ===================================================================
# A.  BBox dataclass
# ===================================================================


class TestBBox:
    """BBox – dataclass contract, field access, and types."""

    def test_is_dataclass(self):
        """BBox must be a @dataclass."""
        assert is_dataclass(BBox), "BBox is not a dataclass"

    def test_fields_accessible(self):
        """x, y, width, height, class_id, class_name are direct attributes."""
        bbox = BBox(
            x=10.5, y=20.3, width=100.0, height=200.0,
            class_id=1, class_name="open_palm",
        )
        assert bbox.x == pytest.approx(10.5)
        assert bbox.y == pytest.approx(20.3)
        assert bbox.width == pytest.approx(100.0)
        assert bbox.height == pytest.approx(200.0)
        assert bbox.class_id == 1
        assert bbox.class_name == "open_palm"

    def test_field_types(self):
        """x/y/width/height are float; class_id is int; class_name is str."""
        bbox = BBox(x=0.0, y=0.0, width=0.0, height=0.0,
                    class_id=0, class_name="")
        assert isinstance(bbox.x, float)
        assert isinstance(bbox.y, float)
        assert isinstance(bbox.width, float)
        assert isinstance(bbox.height, float)
        assert isinstance(bbox.class_id, int)
        assert isinstance(bbox.class_name, str)

    def test_default_values(self):
        """class_id defaults to 0 and class_name defaults to ''."""
        bbox = BBox(x=1.0, y=2.0, width=3.0, height=4.0)
        assert bbox.class_id == 0
        assert bbox.class_name == ""


# ===================================================================
# B.  BBoxExtractor.extract()
# ===================================================================


class TestBBoxExtractorExtract:
    """BBoxExtractor.extract() – bounding box computation from landmarks."""

    def test_returns_bbox(self, extractor, detection_center):
        """extract() must return a BBox instance."""
        result = extractor.extract(detection_center, 640, 480)
        assert isinstance(result, BBox)

    def test_correct_bounds_from_center_landmarks(self, extractor, detection_center):
        """Center landmarks (x: 0.4–0.6, y: 0.35–0.65) produce correct pixel bounds."""
        bbox = extractor.extract(detection_center, 640, 480)

        # Raw bounds (before padding)
        raw_w = (0.60 - 0.40) * 640  # 128.0
        raw_h = (0.65 - 0.35) * 480  # 144.0
        pad_w = raw_w * BBOX_PADDING  # 12.8
        pad_h = raw_h * BBOX_PADDING  # 14.4

        expected_x = 0.40 * 640 - pad_w   # 243.2
        expected_y = 0.35 * 480 - pad_h   # 153.6
        expected_w = raw_w + 2 * pad_w    # 153.6
        expected_h = raw_h + 2 * pad_h    # 172.8

        assert bbox.x == pytest.approx(expected_x, rel=1e-5)
        assert bbox.y == pytest.approx(expected_y, rel=1e-5)
        assert bbox.width == pytest.approx(expected_w, rel=1e-5)
        assert bbox.height == pytest.approx(expected_h, rel=1e-5)

    def test_clamps_to_image_boundaries(self, extractor, detection_edge):
        """Landmarks near edges produce a box clamped to [0, img_w] × [0, img_h]."""
        bbox = extractor.extract(detection_edge, 640, 480)

        assert bbox.x >= 0.0, f"x={bbox.x} must be >= 0"
        assert bbox.y >= 0.0, f"y={bbox.y} must be >= 0"
        assert bbox.x + bbox.width <= 640, (
            f"x+width={bbox.x + bbox.width} must be <= 640"
        )
        assert bbox.y + bbox.height <= 480, (
            f"y+height={bbox.y + bbox.height} must be <= 480"
        )

    def test_uses_bbox_padding(self, extractor, detection_center):
        """The padded box must be larger than the raw (min/max) bounding box."""
        bbox = extractor.extract(detection_center, 640, 480)

        raw_w = (0.60 - 0.40) * 640  # 128.0
        raw_h = (0.65 - 0.35) * 480  # 144.0

        assert bbox.width > raw_w, (
            f"Padded width {bbox.width} must exceed raw width {raw_w}"
        )
        assert bbox.height > raw_h, (
            f"Padded height {bbox.height} must exceed raw height {raw_h}"
        )

    @pytest.mark.parametrize("img_w,img_h", [
        (640, 480),
        (1920, 1080),
        (256, 256),
    ])
    def test_works_with_different_image_sizes(
        self, extractor, detection_center, img_w, img_h
    ):
        """extract() scales correctly for various image dimensions."""
        bbox = extractor.extract(detection_center, img_w, img_h)

        assert 0 <= bbox.x <= img_w
        assert 0 <= bbox.y <= img_h
        assert bbox.width > 0
        assert bbox.height > 0
        assert bbox.x + bbox.width <= img_w
        assert bbox.y + bbox.height <= img_h

    def test_degenerate_case_single_point(self, extractor, detection_single_point):
        """All landmarks at the same point → box computed without error.

        Width and height may be zero (no spatial spread to enlarge).
        """
        bbox = extractor.extract(detection_single_point, 640, 480)

        assert bbox.x >= 0
        assert bbox.y >= 0
        assert bbox.width >= 0
        assert bbox.height >= 0
        # x, y should be near 0.5 scaled to pixels
        assert bbox.x == pytest.approx(0.5 * 640, abs=5)
        assert bbox.y == pytest.approx(0.5 * 480, abs=5)

    def test_preserves_class_id_and_class_name(self, extractor, detection_center):
        """class_id and class_name passed to extract() are stored on the BBox."""
        bbox = extractor.extract(
            detection_center, 640, 480, class_id=1, class_name="open_palm",
        )
        assert bbox.class_id == 1
        assert bbox.class_name == "open_palm"

    def test_default_class_id_and_name_are_zero_and_empty(
        self, extractor, detection_center
    ):
        """When class_id/class_name are omitted, defaults are 0 and ''."""
        bbox = extractor.extract(detection_center, 640, 480)
        assert bbox.class_id == 0
        assert bbox.class_name == ""


# ===================================================================
# C.  BBoxExtractor.to_yolo()
# ===================================================================


class TestBBoxExtractorToYolo:
    """BBoxExtractor.to_yolo() – YOLO-format normalised box conversion."""

    def test_returns_tuple_of_four_floats(self, extractor):
        """to_yolo() must return a tuple of 4 floats."""
        bbox = BBox(x=100, y=200, width=300, height=400)
        result = extractor.to_yolo(bbox, 640, 480)
        assert isinstance(result, tuple)
        assert len(result) == 4
        assert all(isinstance(v, float) for v in result)

    def test_all_values_normalized_zero_to_one(self, extractor):
        """Every value in the YOLO tuple must be in [0, 1]."""
        bbox = BBox(x=100, y=200, width=300, height=400)
        cx, cy, w, h = extractor.to_yolo(bbox, 640, 480)
        for name, val in [("cx", cx), ("cy", cy), ("w", w), ("h", h)]:
            assert 0.0 <= val <= 1.0, f"{name}={val} out of range [0,1]"

    def test_correct_center_and_dims(self, extractor):
        """Box (100, 200, 300, 400) in 640×480 image."""
        bbox = BBox(x=100, y=200, width=300, height=400)
        cx, cy, w, h = extractor.to_yolo(bbox, 640, 480)

        assert cx == pytest.approx(250 / 640)   # (100 + 150) / 640
        assert cy == pytest.approx(400 / 480)   # (200 + 200) / 480
        assert w == pytest.approx(300 / 640)
        assert h == pytest.approx(400 / 480)

    def test_box_at_origin(self, extractor):
        """Box at (0, 0, 320, 240) in 640×480 image."""
        bbox = BBox(x=0, y=0, width=320, height=240)
        cx, cy, w, h = extractor.to_yolo(bbox, 640, 480)

        assert cx == pytest.approx(160 / 640)
        assert cy == pytest.approx(120 / 480)
        assert w == pytest.approx(0.5)
        assert h == pytest.approx(0.5)

    def test_box_filling_entire_image(self, extractor):
        """Box covering the whole image (0, 0, img_w, img_h)."""
        bbox = BBox(x=0, y=0, width=1024, height=768)
        cx, cy, w, h = extractor.to_yolo(bbox, 1024, 768)

        assert cx == pytest.approx(0.5)
        assert cy == pytest.approx(0.5)
        assert w == pytest.approx(1.0)
        assert h == pytest.approx(1.0)

    @pytest.mark.parametrize("img_w,img_h", [
        (640, 480),
        (1920, 1080),
        (1280, 720),
    ])
    def test_scales_with_different_image_sizes(self, extractor, img_w, img_h):
        """YOLO normalisation works for various image dimensions."""
        bbox = BBox(x=100, y=50, width=200, height=100)
        cx, cy, w, h = extractor.to_yolo(bbox, img_w, img_h)

        expected_cx = (100 + 200 / 2) / img_w
        expected_cy = (50 + 100 / 2) / img_h
        expected_w = 200 / img_w
        expected_h = 100 / img_h

        assert cx == pytest.approx(expected_cx)
        assert cy == pytest.approx(expected_cy)
        assert w == pytest.approx(expected_w)
        assert h == pytest.approx(expected_h)


# ===================================================================
# D.  BBoxExtractor.to_coco()
# ===================================================================


class TestBBoxExtractorToCoco:
    """BBoxExtractor.to_coco() – COCO-format pixel box conversion."""

    def test_returns_dict_with_required_keys(self, extractor):
        """to_coco() must return a dict with keys x, y, width, height."""
        bbox = BBox(x=100, y=200, width=300, height=400)
        result = extractor.to_coco(bbox)
        assert isinstance(result, dict)
        assert set(result.keys()) == {"x", "y", "width", "height"}

    def test_values_are_int(self, extractor):
        """All COCO dict values must be Python int."""
        bbox = BBox(x=100.0, y=200.0, width=300.0, height=400.0)
        result = extractor.to_coco(bbox)
        for key in ("x", "y", "width", "height"):
            assert isinstance(result[key], int), (
                f"result['{key}'] is {type(result[key]).__name__}, expected int"
            )

    def test_values_are_absolute_pixels(self, extractor):
        """COCO values are absolute pixel coordinates (not normalized)."""
        bbox = BBox(x=100.0, y=200.0, width=300.0, height=400.0)
        result = extractor.to_coco(bbox)
        assert result["x"] == 100
        assert result["y"] == 200
        assert result["width"] == 300
        assert result["height"] == 400

    def test_rounding_to_nearest_int(self, extractor):
        """Float values are rounded to nearest int (not truncated)."""
        bbox = BBox(x=100.3, y=200.7, width=300.2, height=400.8)
        result = extractor.to_coco(bbox)
        assert result["x"] == 100       # round(100.3) = 100
        assert result["y"] == 201       # round(200.7) = 201
        assert result["width"] == 300   # round(300.2) = 300
        assert result["height"] == 401  # round(400.8) = 401

    def test_rounding_at_point_five(self, extractor):
        """round(0.5) uses banker's rounding – verify behaviour is consistent."""
        bbox = BBox(x=5.5, y=10.5, width=15.5, height=20.5)
        result = extractor.to_coco(bbox)
        # Python's round uses banker's rounding (round half to even):
        #   round(5.5)=6, round(10.5)=10, round(15.5)=16, round(20.5)=20
        assert isinstance(result["x"], int)
        assert isinstance(result["y"], int)
        assert isinstance(result["width"], int)
        assert isinstance(result["height"], int)
        assert result["x"] in (5, 6)
        assert result["y"] in (10, 11)
        assert result["width"] in (15, 16)
        assert result["height"] in (20, 21)
