"""Unit tests for Exporter component and ProcessingResult dataclass."""

import json
from dataclasses import is_dataclass
from pathlib import Path
from unittest.mock import MagicMock

import pytest
import yaml

from boundingboxer.config import (
    CLASS_MAP,
    CLASS_NAMES,
    OUTPUT_DATASET_YAML,
    OUTPUT_IMAGES_DIR,
    OUTPUT_LABELS_DIR,
    SUPPORTED_EXPORT_FORMATS,
)
from boundingboxer.extractor import BBox
from boundingboxer.loader import ImageRecord
from boundingboxer.exporter import Exporter, ProcessingResult


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def exporter():
    """Return a default Exporter instance."""
    return Exporter()


@pytest.fixture
def sample_image_record():
    """An ImageRecord for closed_fist/img001.jpg."""
    return ImageRecord(
        path=Path("/data/closed_fist/img001.jpg"),
        class_name="closed_fist",
        class_id=0,
    )


@pytest.fixture
def sample_image_record_open_palm():
    """An ImageRecord for open_palm/img002.png."""
    return ImageRecord(
        path=Path("/data/open_palm/img002.png"),
        class_name="open_palm",
        class_id=1,
    )


@pytest.fixture
def sample_bbox():
    """A BBox with known pixel coordinates."""
    return BBox(
        x=100.0, y=200.0, width=300.0, height=400.0,
        class_id=0, class_name="closed_fist",
    )


@pytest.fixture
def sample_bbox_2():
    """A second BBox with different coordinates."""
    return BBox(
        x=50.0, y=60.0, width=150.0, height=200.0,
        class_id=1, class_name="open_palm",
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
    """Factory helper: build a ProcessingResult with sensible defaults."""
    if image_record is None:
        image_record = ImageRecord(
            path=Path("/data/closed_fist/img001.jpg"),
            class_name="closed_fist",
            class_id=0,
        )
    if detections is None:
        detections = [MagicMock()]
    if bboxes is None:
        bboxes = [
            BBox(
                x=100.0, y=200.0, width=300.0, height=400.0,
                class_id=detected_class_id, class_name=detected_class,
            )
        ]
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


# ===================================================================
# A.  ProcessingResult dataclass
# ===================================================================


class TestProcessingResult:
    """ProcessingResult – dataclass contract, field access, and types."""

    def test_is_dataclass(self):
        """ProcessingResult must be a @dataclass."""
        assert is_dataclass(ProcessingResult), (
            "ProcessingResult is not a dataclass"
        )

    def test_all_fields_accessible(self):
        """Every declared field is readable as an attribute."""
        pr = _make_result()
        assert pr.image_record is not None
        assert isinstance(pr.detections, list)
        assert isinstance(pr.bboxes, list)
        assert pr.detected_class == "closed_fist"
        assert pr.detected_class_id == 0
        assert pr.classification_confidence == pytest.approx(0.9)
        assert pr.mediapipe_confidence == pytest.approx(0.95)
        assert pr.combined_confidence == pytest.approx(0.95)
        assert pr.needs_review is False
        assert pr.reviewed is False
        assert pr.manual_override is False

    def test_detected_class_can_be_none(self):
        """When nothing is detected, detected_class and detected_class_id are None."""
        pr = _make_result(
            detections=[],
            bboxes=[],
            detected_class=None,
            detected_class_id=None,
        )
        assert pr.detected_class is None
        assert pr.detected_class_id is None

    def test_default_values(self):
        """reviewed and manual_override default to False."""
        # Rely on the dataclass defaults – build with only required fields.
        pr = ProcessingResult(
            image_record=MagicMock(),
            detections=[MagicMock()],
            bboxes=[MagicMock()],
            detected_class="closed_fist",
            detected_class_id=0,
            classification_confidence=0.9,
            mediapipe_confidence=0.95,
            combined_confidence=0.95,
            needs_review=False,
        )
        assert pr.reviewed is False
        assert pr.manual_override is False


# ===================================================================
# B.  Exporter.export_yolo()
# ===================================================================


class TestExporterExportYolo:
    """Exporter.export_yolo() – YOLO-format label file generation."""

    # ------------------------------------------------------------------
    # Directory structure
    # ------------------------------------------------------------------

    def test_creates_labels_directory_structure(self, exporter, tmp_path):
        """export_yolo must create output_dir/labels/<class_name>/."""
        output_dir = tmp_path / "export"
        result = _make_result()
        exporter.export_yolo([result], output_dir)

        expected_dir = output_dir / OUTPUT_LABELS_DIR / "closed_fist"
        assert expected_dir.is_dir(), (
            f"Expected directory {expected_dir} was not created"
        )

    def test_creates_parent_directories(self, exporter, tmp_path):
        """export_yolo must create parent directories if they do not exist."""
        output_dir = tmp_path / "nested" / "deep" / "export"
        assert not output_dir.exists()

        result = _make_result()
        exporter.export_yolo([result], output_dir)

        expected_label_dir = output_dir / OUTPUT_LABELS_DIR / "closed_fist"
        assert expected_label_dir.is_dir(), (
            "Parent directories were not created"
        )

    # ------------------------------------------------------------------
    # File naming
    # ------------------------------------------------------------------

    def test_filename_matches_image_stem(self, exporter, tmp_path):
        """The .txt file name is the image file's stem (without extension)."""
        output_dir = tmp_path / "export"
        result = _make_result(
            image_record=ImageRecord(
                path=Path("/data/closed_fist/img001.jpg"),
                class_name="closed_fist",
                class_id=0,
            )
        )
        exporter.export_yolo([result], output_dir)

        label_file = (
            output_dir / OUTPUT_LABELS_DIR / "closed_fist" / "img001.txt"
        )
        assert label_file.is_file(), f"Expected {label_file} to exist"

    @pytest.mark.parametrize("ext", [".jpg", ".png", ".jpeg", ".webp", ".bmp"])
    def test_handles_various_image_extensions(self, exporter, tmp_path, ext):
        """The stem-based naming works for different image extensions."""
        output_dir = tmp_path / "export"
        image_path = Path(f"/data/closed_fist/image_name{ext}")
        result = _make_result(
            image_record=ImageRecord(
                path=image_path,
                class_name="closed_fist",
                class_id=0,
            )
        )
        exporter.export_yolo([result], output_dir)

        label_file = (
            output_dir / OUTPUT_LABELS_DIR / "closed_fist" / "image_name.txt"
        )
        assert label_file.is_file()

    # ------------------------------------------------------------------
    # Content format
    # ------------------------------------------------------------------

    def test_writes_class_id_and_four_normalized_floats(self, exporter, tmp_path):
        """Each line must be: class_id cx cy w h (space-separated)."""
        output_dir = tmp_path / "export"
        result = _make_result()
        exporter.export_yolo([result], output_dir)

        label_file = (
            output_dir / OUTPUT_LABELS_DIR / "closed_fist" / "img001.txt"
        )
        content = label_file.read_text().strip()
        parts = content.split()

        assert len(parts) == 5, f"Expected 5 values, got {len(parts)}: {content}"
        # class_id is an integer
        assert parts[0] == str(result.detected_class_id)
        # The remaining four values are floats in [0, 1] (normalized)
        for i, val in enumerate(parts[1:], start=1):
            f = float(val)
            assert 0.0 <= f <= 1.0, (
                f"Value at position {i} ({f}) is out of [0, 1] range"
            )

    # ------------------------------------------------------------------
    # Skipping empty bboxes
    # ------------------------------------------------------------------

    def test_skips_results_with_empty_bboxes(self, exporter, tmp_path):
        """ProcessingResult with zero bboxes must not produce a label file."""
        output_dir = tmp_path / "export"
        result_no_bbox = _make_result(
            detections=[],
            bboxes=[],
            detected_class=None,
            detected_class_id=None,
            image_record=ImageRecord(
                path=Path("/data/closed_fist/img_empty.jpg"),
                class_name="closed_fist",
                class_id=0,
            ),
        )
        exporter.export_yolo([result_no_bbox], output_dir)

        label_file = (
            output_dir / OUTPUT_LABELS_DIR / "closed_fist" / "img_empty.txt"
        )
        assert not label_file.exists(), (
            "Label file should NOT be created for results with zero bboxes"
        )

    def test_mixed_results_skip_empty_write_nonempty(self, exporter, tmp_path):
        """When mixing empty and non-empty results, only non-empty get files."""
        output_dir = tmp_path / "export"
        results = [
            _make_result(
                image_record=ImageRecord(
                    path=Path("/data/closed_fist/img_empty.jpg"),
                    class_name="closed_fist",
                    class_id=0,
                ),
                detections=[],
                bboxes=[],
                detected_class=None,
                detected_class_id=None,
            ),
            _make_result(
                image_record=ImageRecord(
                    path=Path("/data/closed_fist/img001.jpg"),
                    class_name="closed_fist",
                    class_id=0,
                ),
            ),
        ]
        exporter.export_yolo(results, output_dir)

        assert not (
            output_dir / OUTPUT_LABELS_DIR / "closed_fist" / "img_empty.txt"
        ).exists()
        assert (
            output_dir / OUTPUT_LABELS_DIR / "closed_fist" / "img001.txt"
        ).is_file()

    # ------------------------------------------------------------------
    # Multiple bboxes
    # ------------------------------------------------------------------

    def test_handles_multiple_bboxes_per_image(self, exporter, tmp_path):
        """Multiple bboxes produce one line per bbox in the .txt file."""
        output_dir = tmp_path / "export"
        bbox_a = BBox(
            x=10, y=20, width=30, height=40, class_id=0, class_name="closed_fist",
        )
        bbox_b = BBox(
            x=100, y=200, width=50, height=60, class_id=1, class_name="open_palm",
        )
        result = _make_result(bboxes=[bbox_a, bbox_b])

        exporter.export_yolo([result], output_dir)

        label_file = (
            output_dir / OUTPUT_LABELS_DIR / "closed_fist" / "img001.txt"
        )
        lines = label_file.read_text().strip().splitlines()
        assert len(lines) == 2, (
            f"Expected 2 lines for 2 bboxes, got {len(lines)}"
        )
        for line in lines:
            parts = line.split()
            assert len(parts) == 5, f"Expected 5 values per line, got: {line}"

    # ------------------------------------------------------------------
    # Multiple results with different classes
    # ------------------------------------------------------------------

    def test_multiple_results_different_classes(
        self, exporter, sample_image_record_open_palm, tmp_path,
    ):
        """Files are placed in the correct per-class subdirectory."""
        output_dir = tmp_path / "export"
        r1 = _make_result(
            image_record=ImageRecord(
                path=Path("/data/closed_fist/cf.jpg"),
                class_name="closed_fist",
                class_id=0,
            ),
            detected_class="closed_fist",
            detected_class_id=0,
            bboxes=[BBox(x=10, y=20, width=30, height=40, class_id=0, class_name="closed_fist")],
        )
        r2 = _make_result(
            image_record=ImageRecord(
                path=Path("/data/open_palm/op.jpg"),
                class_name="open_palm",
                class_id=1,
            ),
            detected_class="open_palm",
            detected_class_id=1,
            bboxes=[BBox(x=50, y=60, width=70, height=80, class_id=1, class_name="open_palm")],
        )

        exporter.export_yolo([r1, r2], output_dir)

        cf_file = output_dir / OUTPUT_LABELS_DIR / "closed_fist" / "cf.txt"
        op_file = output_dir / OUTPUT_LABELS_DIR / "open_palm" / "op.txt"
        assert cf_file.is_file()
        assert op_file.is_file()


# ===================================================================
# C.  Exporter.export_coco()
# ===================================================================


class TestExporterExportCoco:
    """Exporter.export_coco() – COCO-format JSON dict generation."""

    def test_returns_dict_with_required_top_level_keys(self, exporter):
        """export_coco must return a dict with 'images', 'annotations', 'categories'."""
        result = _make_result()
        coco = exporter.export_coco([result], Path("/fake"))

        assert isinstance(coco, dict)
        for key in ("images", "annotations", "categories"):
            assert key in coco, f"Missing top-level key: {key}"

    def test_images_have_required_fields(self, exporter):
        """Each image entry has id, file_name, width, height."""
        result = _make_result()
        coco = exporter.export_coco([result], Path("/fake"))

        images = coco["images"]
        assert len(images) == 1
        img = images[0]
        for field in ("id", "file_name", "width", "height"):
            assert field in img, f"Missing field in image entry: {field}"
        assert isinstance(img["id"], int)
        assert isinstance(img["file_name"], str)
        assert isinstance(img["width"], (int, float))
        assert isinstance(img["height"], (int, float))

    def test_image_id_auto_increments(self, exporter):
        """Each image gets a unique, incrementing id starting from 1."""
        r1 = _make_result(
            image_record=ImageRecord(
                path=Path("/data/closed_fist/a.jpg"),
                class_name="closed_fist", class_id=0,
            ),
        )
        r2 = _make_result(
            image_record=ImageRecord(
                path=Path("/data/open_palm/b.jpg"),
                class_name="open_palm", class_id=1,
            ),
            detected_class="open_palm", detected_class_id=1,
        )
        coco = exporter.export_coco([r1, r2], Path("/fake"))

        ids = [img["id"] for img in coco["images"]]
        assert ids == [1, 2], f"Expected [1, 2], got {ids}"
        assert len(set(ids)) == len(ids), "Image IDs must be unique"

    def test_annotations_have_required_fields(self, exporter):
        """Each annotation has id, image_id, category_id, bbox, area."""
        result = _make_result()
        coco = exporter.export_coco([result], Path("/fake"))

        annotations = coco["annotations"]
        assert len(annotations) == 1
        ann = annotations[0]
        for field in ("id", "image_id", "category_id", "bbox", "area"):
            assert field in ann, f"Missing field in annotation: {field}"
        assert isinstance(ann["id"], int)
        assert isinstance(ann["image_id"], int)
        assert isinstance(ann["category_id"], int)
        assert isinstance(ann["bbox"], list)
        assert len(ann["bbox"]) == 4
        assert isinstance(ann["area"], (int, float))

    def test_annotation_id_auto_increments(self, exporter):
        """Multiple bboxes produce annotations with unique, incrementing ids."""
        result = _make_result(bboxes=[
            BBox(x=10, y=20, width=30, height=40, class_id=0, class_name="closed_fist"),
            BBox(x=50, y=60, width=70, height=80, class_id=1, class_name="open_palm"),
        ])
        coco = exporter.export_coco([result], Path("/fake"))

        ann_ids = [ann["id"] for ann in coco["annotations"]]
        assert ann_ids == [1, 2], f"Expected [1, 2], got {ann_ids}"

    def test_annotation_refers_to_correct_image(self, exporter):
        """Each annotation's image_id matches its corresponding image id."""
        r1 = _make_result(
            image_record=ImageRecord(
                path=Path("/data/closed_fist/a.jpg"),
                class_name="closed_fist", class_id=0,
            ),
        )
        r2 = _make_result(
            image_record=ImageRecord(
                path=Path("/data/open_palm/b.jpg"),
                class_name="open_palm", class_id=1,
            ),
            detected_class="open_palm", detected_class_id=1,
        )
        coco = exporter.export_coco([r1, r2], Path("/fake"))

        # r1 → image id 1, r2 → image id 2
        for ann in coco["annotations"]:
            assert ann["image_id"] in (1, 2), (
                f"Annotation image_id {ann['image_id']} not in (1, 2)"
            )

    def test_categories_match_class_names(self, exporter):
        """Categories list must match CLASS_NAMES with id+name."""
        result = _make_result()
        coco = exporter.export_coco([result], Path("/fake"))

        categories = coco["categories"]
        assert len(categories) == len(CLASS_NAMES)
        for cat in categories:
            assert "id" in cat
            assert "name" in cat
            assert cat["id"] in CLASS_MAP.values()
            assert cat["name"] in CLASS_NAMES

    def test_skips_results_without_bboxes_in_annotations(self, exporter):
        """Results with zero bboxes must not produce annotation entries."""
        r1 = _make_result()
        r2 = _make_result(
            image_record=ImageRecord(
                path=Path("/data/closed_fist/empty.jpg"),
                class_name="closed_fist", class_id=0,
            ),
            detections=[], bboxes=[],
            detected_class=None, detected_class_id=None,
        )
        coco = exporter.export_coco([r1, r2], Path("/fake"))

        # r2 has no bboxes, so only r1's bbox produces an annotation
        assert len(coco["annotations"]) == 1
        # But both images should be in the images list
        assert len(coco["images"]) == 2

    def test_empty_results_list(self, exporter):
        """Empty results list → dict with empty images/annotations lists."""
        coco = exporter.export_coco([], Path("/fake"))
        assert coco["images"] == []
        assert coco["annotations"] == []
        assert len(coco["categories"]) == len(CLASS_NAMES)

    def test_file_name_is_relative_path_string(self, exporter):
        """file_name in images should be a relative path string."""
        result = _make_result(
            image_record=ImageRecord(
                path=Path("/data/closed_fist/sub/img.jpg"),
                class_name="closed_fist", class_id=0,
            ),
        )
        coco = exporter.export_coco([result], Path("/data"))

        file_name = coco["images"][0]["file_name"]
        assert isinstance(file_name, str)
        assert file_name == "closed_fist/sub/img.jpg" or "closed_fist" in file_name


# ===================================================================
# D.  Exporter.generate_dataset_yaml()
# ===================================================================


class TestExporterGenerateDatasetYaml:
    """Exporter.generate_dataset_yaml() – dataset.yaml creation."""

    def test_creates_yaml_file(self, exporter, tmp_path):
        """generate_dataset_yaml must create the dataset.yaml file."""
        output_dir = tmp_path / "export"
        output_dir.mkdir()
        exporter.generate_dataset_yaml(output_dir, CLASS_NAMES)

        yaml_path = output_dir / OUTPUT_DATASET_YAML
        assert yaml_path.is_file(), f"{yaml_path} was not created"

    def test_yaml_has_required_keys(self, exporter, tmp_path):
        """The YAML file must contain path, train, val, and names."""
        output_dir = tmp_path / "export"
        output_dir.mkdir()
        exporter.generate_dataset_yaml(output_dir, CLASS_NAMES)

        yaml_path = output_dir / OUTPUT_DATASET_YAML
        with open(yaml_path) as fh:
            data = yaml.safe_load(fh)

        for key in ("path", "train", "val", "names"):
            assert key in data, f"Missing key in dataset.yaml: {key}"

    def test_path_points_to_output_dir(self, exporter, tmp_path):
        """The 'path' key should reference the output directory."""
        output_dir = tmp_path / "export"
        output_dir.mkdir()
        exporter.generate_dataset_yaml(output_dir, CLASS_NAMES)

        yaml_path = output_dir / OUTPUT_DATASET_YAML
        with open(yaml_path) as fh:
            data = yaml.safe_load(fh)

        assert data["path"] == str(output_dir), (
            f"Expected path {output_dir}, got {data['path']}"
        )

    def test_train_and_val_are_images_dir(self, exporter, tmp_path):
        """train and val must both equal OUTPUT_IMAGES_DIR."""
        output_dir = tmp_path / "export"
        output_dir.mkdir()
        exporter.generate_dataset_yaml(output_dir, CLASS_NAMES)

        yaml_path = output_dir / OUTPUT_DATASET_YAML
        with open(yaml_path) as fh:
            data = yaml.safe_load(fh)

        assert data["train"] == OUTPUT_IMAGES_DIR
        assert data["val"] == OUTPUT_IMAGES_DIR

    def test_names_map_matches_class_names_parameter(self, exporter, tmp_path):
        """The 'names' dict maps class_id → class_name matching the parameter."""
        output_dir = tmp_path / "export"
        output_dir.mkdir()
        exporter.generate_dataset_yaml(output_dir, CLASS_NAMES)

        yaml_path = output_dir / OUTPUT_DATASET_YAML
        with open(yaml_path) as fh:
            data = yaml.safe_load(fh)

        names = data["names"]
        assert isinstance(names, dict)
        for i, name in enumerate(CLASS_NAMES):
            assert names[i] == name, (
                f"names[{i}] = {names.get(i)!r}, expected {name!r}"
            )

    def test_creates_parent_directories_if_needed(self, exporter, tmp_path):
        """generate_dataset_yaml must create parent directories automatically."""
        output_dir = tmp_path / "deeply" / "nested" / "export"
        assert not output_dir.exists()

        exporter.generate_dataset_yaml(output_dir, CLASS_NAMES)

        yaml_path = output_dir / OUTPUT_DATASET_YAML
        assert yaml_path.is_file()
