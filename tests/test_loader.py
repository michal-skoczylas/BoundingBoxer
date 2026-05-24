"""Unit tests for ImageLoader component."""

import os
from collections import Counter
from dataclasses import is_dataclass
from pathlib import Path

import cv2
import numpy as np
import pytest

from boundingboxer.config import CLASS_MAP, IMAGE_EXTENSIONS
from boundingboxer.loader import ImageLoader, ImageRecord


# ---------------------------------------------------------------------------
# Fixture
# ---------------------------------------------------------------------------

@pytest.fixture
def sample_input_dir(tmp_path):
    """
    Create a temporary directory structure simulating input_dir.

    Structure:
        tmp_path/
          closed_fist/
            img1.jpg           ← valid (class_id=0)
            img2.png           ← valid (class_id=0)
            not_image.txt      ← IGNORED (not an image extension)
          open_palm/
            img3.jpg           ← valid (class_id=1)
          none/
            img4.jpg           ← valid (class_id=2)
            img5.jpeg          ← valid (class_id=2)
          unknown_folder/      ← IGNORED (not in CLASS_MAP)
            img6.jpg           ← IGNORED (parent skipped)
    """
    # --- Create directories ---
    closed_fist = tmp_path / "closed_fist"
    open_palm = tmp_path / "open_palm"
    none_dir = tmp_path / "none"
    unknown_folder = tmp_path / "unknown_folder"

    for d in [closed_fist, open_palm, none_dir, unknown_folder]:
        d.mkdir()

    # --- Create a small test image (10x15, red in BGR) ---
    img = np.zeros((10, 15, 3), dtype=np.uint8)
    img[:, :, :] = [0, 0, 255]  # BGR – pure red

    # closed_fist
    cv2.imwrite(str(closed_fist / "img1.jpg"), img)
    cv2.imwrite(str(closed_fist / "img2.png"), img)
    (closed_fist / "not_image.txt").write_text("this is not an image")

    # open_palm
    cv2.imwrite(str(open_palm / "img3.jpg"), img)

    # none
    cv2.imwrite(str(none_dir / "img4.jpg"), img)
    cv2.imwrite(str(none_dir / "img5.jpeg"), img)

    # unknown_folder  (should be fully skipped)
    cv2.imwrite(str(unknown_folder / "img6.jpg"), img)

    return tmp_path


@pytest.fixture
def loader(sample_input_dir):
    """Return an ImageLoader pointing at the sample directory."""
    return ImageLoader(input_dir=sample_input_dir)


# ===================================================================
# 1. Constructor
# ===================================================================

class TestImageLoaderConstruction:
    """ImageLoader.__init__  –  input_dir, extensions, error handling."""

    def test_accepts_path_object(self, sample_input_dir):
        """Constructor must accept a pathlib.Path."""
        instance = ImageLoader(input_dir=sample_input_dir)
        assert instance is not None

    def test_accepts_string_path(self, sample_input_dir):
        """Constructor must accept a plain string."""
        instance = ImageLoader(input_dir=str(sample_input_dir))
        assert instance is not None

    def test_raises_error_for_nonexistent_dir(self):
        """Raises FileNotFoundError when input_dir does not exist."""
        with pytest.raises(FileNotFoundError):
            ImageLoader(input_dir="/nonexistent/path/xyz123")

    def test_default_extensions_from_config(self, sample_input_dir):
        """When extensions is omitted, IMAGE_EXTENSIONS from config is used."""
        loader = ImageLoader(input_dir=sample_input_dir)
        records = loader.scan()
        # .txt is not an IMAGE_EXTENSIONS member → excluded
        assert len(records) == 5

    def test_custom_extensions(self, sample_input_dir):
        """Custom extensions override the default set."""
        loader = ImageLoader(input_dir=sample_input_dir, extensions=(".jpg",))
        records = loader.scan()
        # Only .jpg: img1.jpg, img3.jpg, img4.jpg  (img2.png out, img5.jpeg out)
        assert len(records) == 3


# ===================================================================
# 2. scan()
# ===================================================================

class TestScan:
    """ImageLoader.scan() –  directory traversal, filtering, records."""

    # ---- Return type & structure ----

    def test_returns_list_of_image_records(self, loader):
        """scan() must return a list of ImageRecord objects."""
        records = loader.scan()
        assert isinstance(records, list)
        assert all(isinstance(r, ImageRecord) for r in records)

    def test_correct_number_of_records(self, loader):
        """Exactly 5 valid images across 3 known classes."""
        records = loader.scan()
        assert len(records) == 5

    def test_each_record_has_correct_field_types(self, loader):
        """path→Path, class_name→str, class_id→int."""
        records = loader.scan()
        for r in records:
            assert isinstance(r.path, Path), f"{r.path=}"
            assert isinstance(r.class_name, str), f"{r.class_name=}"
            assert isinstance(r.class_id, int), f"{r.class_id=}"

    def test_class_name_matches_parent_folder(self, loader):
        """class_name must equal the name of the immediate parent directory."""
        records = loader.scan()
        for r in records:
            assert r.path.parent.name == r.class_name, (
                f"Expected {r.path.parent.name}, got {r.class_name}"
            )

    def test_class_id_matches_class_map(self, loader):
        """class_id must match CLASS_MAP[class_name]."""
        records = loader.scan()
        for r in records:
            assert r.class_id == CLASS_MAP[r.class_name], (
                f"{r.class_name} → {r.class_id} != {CLASS_MAP[r.class_name]}"
            )

    def test_all_expected_paths_present(self, loader, sample_input_dir):
        """Every expected image file must appear in the results."""
        records = loader.scan()
        paths = set(r.path for r in records)

        expected = {
            sample_input_dir / "closed_fist" / "img1.jpg",
            sample_input_dir / "closed_fist" / "img2.png",
            sample_input_dir / "open_palm" / "img3.jpg",
            sample_input_dir / "none" / "img4.jpg",
            sample_input_dir / "none" / "img5.jpeg",
        }
        assert paths == expected, f"Missing: {expected - paths}, Extra: {paths - expected}"

    # ---- Filtering ----

    def test_skips_unknown_folder(self, loader, sample_input_dir):
        """Folders not present in CLASS_MAP are completely ignored."""
        records = loader.scan()
        paths = [r.path for r in records]
        unknown_img = sample_input_dir / "unknown_folder" / "img6.jpg"
        assert unknown_img not in paths

    def test_skips_non_image_extensions(self, loader, sample_input_dir):
        """Files whose extension is not in IMAGE_EXTENSIONS are ignored."""
        records = loader.scan()
        paths = [r.path for r in records]
        txt_file = sample_input_dir / "closed_fist" / "not_image.txt"
        assert txt_file not in paths

    def test_per_class_counts(self, loader):
        """Correct subclass counts: closed_fist=2, open_palm=1, none=2."""
        records = loader.scan()
        counts = Counter(r.class_name for r in records)
        assert counts == {"closed_fist": 2, "open_palm": 1, "none": 2}

    # ---- Edge cases ----

    def test_empty_directory(self, tmp_path):
        """An input_dir with no subdirectories → empty list."""
        empty_dir = tmp_path / "empty"
        empty_dir.mkdir()
        loader = ImageLoader(input_dir=empty_dir)
        assert loader.scan() == []

    def test_class_folders_without_images(self, tmp_path):
        """Class folders exist but contain no image files → empty list."""
        input_dir = tmp_path / "no_images"
        input_dir.mkdir()
        (input_dir / "closed_fist").mkdir()
        (input_dir / "open_palm").mkdir()
        loader = ImageLoader(input_dir=input_dir)
        assert loader.scan() == []


# ===================================================================
# 3. load()
# ===================================================================

class TestLoad:
    """ImageLoader.load(path) –  image reading, return type/shape/format."""

    def test_returns_ndarray(self, loader):
        """Must return a numpy.ndarray."""
        records = loader.scan()
        img = loader.load(records[0].path)
        assert isinstance(img, np.ndarray)

    def test_loads_correct_shape(self, loader):
        """Loaded image shape must match the original (10, 15, 3)."""
        records = loader.scan()
        img = loader.load(records[0].path)
        assert img.shape == (10, 15, 3), f"Got shape {img.shape}"

    def test_loads_bgr_format(self, loader):
        """
        Image must be in BGR format (OpenCV default).
        Our test image is pure red → B = 0, G = 0, R = 255.
        Use PNG (lossless) to avoid JPEG compression artifacts.
        """
        records = loader.scan()
        png_records = [r for r in records if r.path.suffix.lower() == ".png"]
        img = loader.load(png_records[0].path)
        assert img[0, 0, 0] == 0    # Blue channel
        assert img[0, 0, 1] == 0    # Green channel
        assert img[0, 0, 2] == 255  # Red channel

    def test_accepts_string_path(self, loader):
        """load() must accept a string argument."""
        records = loader.scan()
        img = loader.load(str(records[0].path))
        assert isinstance(img, np.ndarray)

    def test_accepts_path_object(self, loader):
        """load() must accept a pathlib.Path argument."""
        records = loader.scan()
        img = loader.load(records[0].path)
        assert isinstance(img, np.ndarray)

    def test_nonexistent_file_raises_error(self, loader):
        """Loading a file that does not exist should raise an exception."""
        with pytest.raises((FileNotFoundError, OSError, Exception)):
            loader.load("/nonexistent/image.jpg")


# ===================================================================
# 4. ImageRecord dataclass
# ===================================================================

class TestImageRecord:
    """ImageRecord – dataclass contract."""

    def test_is_dataclass(self):
        """ImageRecord must be a @dataclass."""
        assert is_dataclass(ImageRecord), "ImageRecord is not a dataclass"

    def test_fields_accessible(self):
        """path, class_name, class_id must be direct attributes."""
        record = ImageRecord(
            path=Path("/tmp/test.jpg"),
            class_name="closed_fist",
            class_id=0,
        )
        assert record.path == Path("/tmp/test.jpg")
        assert record.class_name == "closed_fist"
        assert record.class_id == 0
