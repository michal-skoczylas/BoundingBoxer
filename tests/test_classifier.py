"""Unit tests for GestureClassifier component."""

import numpy as np
import pytest

from boundingboxer.config import CLASS_MAP, CLASS_NAMES
from boundingboxer.classifier import GestureClassifier


# ---------------------------------------------------------------------------
# Landmark topology helpers
# ---------------------------------------------------------------------------

# MediaPipe hand landmark indices
WRIST = 0
THUMB_CMC = 1
THUMB_MCP = 2
THUMB_IP = 3
THUMB_TIP = 4
INDEX_MCP = 5
INDEX_PIP = 6
INDEX_DIP = 7
INDEX_TIP = 8
MIDDLE_MCP = 9
MIDDLE_PIP = 10
MIDDLE_DIP = 11
MIDDLE_TIP = 12
RING_MCP = 13
RING_PIP = 14
RING_DIP = 15
RING_TIP = 16
PINKY_MCP = 17
PINKY_PIP = 18
PINKY_DIP = 19
PINKY_TIP = 20

# Finger definitions: (mcp, pip, tip) — DIP joints are unused by the algorithm
FINGERS = [
    (INDEX_MCP, INDEX_PIP, INDEX_TIP),
    (MIDDLE_MCP, MIDDLE_PIP, MIDDLE_TIP),
    (RING_MCP, RING_PIP, RING_TIP),
    (PINKY_MCP, PINKY_PIP, PINKY_TIP),
]


def _make_open_palm_landmarks():
    """Return (21, 3) landmarks for an open palm (all 4 fingers extended).

    Each extended finger has MCP→TIP distance > 1.3 × MCP→PIP distance.
    """
    lm = np.full((21, 3), 0.5, dtype=np.float32)  # neutral default

    # Index finger (extended upward)
    lm[INDEX_MCP] = [0.40, 0.60, 0.0]
    lm[INDEX_PIP] = [0.40, 0.48, 0.0]
    lm[INDEX_DIP] = [0.40, 0.34, 0.0]
    lm[INDEX_TIP] = [0.40, 0.20, 0.0]
    # d(MCP,PIP) = 0.12, d(MCP,TIP) = 0.40, ratio ≈ 3.33 > 1.3

    # Middle finger (extended upward)
    lm[MIDDLE_MCP] = [0.45, 0.58, 0.0]
    lm[MIDDLE_PIP] = [0.45, 0.44, 0.0]
    lm[MIDDLE_DIP] = [0.45, 0.30, 0.0]
    lm[MIDDLE_TIP] = [0.45, 0.18, 0.0]

    # Ring finger (extended upward)
    lm[RING_MCP] = [0.50, 0.56, 0.0]
    lm[RING_PIP] = [0.50, 0.42, 0.0]
    lm[RING_DIP] = [0.50, 0.30, 0.0]
    lm[RING_TIP] = [0.50, 0.17, 0.0]

    # Pinky finger (extended upward)
    lm[PINKY_MCP] = [0.55, 0.54, 0.0]
    lm[PINKY_PIP] = [0.55, 0.44, 0.0]
    lm[PINKY_DIP] = [0.55, 0.33, 0.0]
    lm[PINKY_TIP] = [0.55, 0.23, 0.0]

    return lm


def _make_closed_fist_landmarks():
    """Return (21, 3) landmarks for a closed fist (all 4 fingers bent).

    Each bent finger has MCP→TIP distance ≤ 1.3 × MCP→PIP distance.
    """
    lm = np.full((21, 3), 0.5, dtype=np.float32)

    # Index finger (bent — tip curled near MCP)
    lm[INDEX_MCP] = [0.40, 0.60, 0.0]
    lm[INDEX_PIP] = [0.34, 0.55, 0.0]
    lm[INDEX_DIP] = [0.38, 0.53, 0.0]
    lm[INDEX_TIP] = [0.45, 0.57, 0.0]
    # d(MCP,PIP) ≈ 0.078, d(MCP,TIP) ≈ 0.058, ratio ≈ 0.74 < 1.3

    # Middle finger (bent)
    lm[MIDDLE_MCP] = [0.45, 0.58, 0.0]
    lm[MIDDLE_PIP] = [0.38, 0.53, 0.0]
    lm[MIDDLE_DIP] = [0.42, 0.51, 0.0]
    lm[MIDDLE_TIP] = [0.50, 0.55, 0.0]

    # Ring finger (bent)
    lm[RING_MCP] = [0.50, 0.56, 0.0]
    lm[RING_PIP] = [0.43, 0.52, 0.0]
    lm[RING_DIP] = [0.46, 0.50, 0.0]
    lm[RING_TIP] = [0.54, 0.54, 0.0]

    # Pinky finger (bent)
    lm[PINKY_MCP] = [0.55, 0.54, 0.0]
    lm[PINKY_PIP] = [0.48, 0.51, 0.0]
    lm[PINKY_DIP] = [0.51, 0.49, 0.0]
    lm[PINKY_TIP] = [0.59, 0.53, 0.0]

    return lm


def _make_three_extended_landmarks():
    """Return landmarks with exactly 3 fingers extended (index bent, rest extended).

    Should classify as OPEN_PALM (extended_count ≥ 3).
    """
    lm = np.full((21, 3), 0.5, dtype=np.float32)

    # Index: BENT
    lm[INDEX_MCP] = [0.40, 0.60, 0.0]
    lm[INDEX_PIP] = [0.34, 0.55, 0.0]
    lm[INDEX_DIP] = [0.38, 0.53, 0.0]
    lm[INDEX_TIP] = [0.45, 0.57, 0.0]

    # Middle: EXTENDED
    lm[MIDDLE_MCP] = [0.45, 0.58, 0.0]
    lm[MIDDLE_PIP] = [0.45, 0.44, 0.0]
    lm[MIDDLE_DIP] = [0.45, 0.30, 0.0]
    lm[MIDDLE_TIP] = [0.45, 0.18, 0.0]

    # Ring: EXTENDED
    lm[RING_MCP] = [0.50, 0.56, 0.0]
    lm[RING_PIP] = [0.50, 0.42, 0.0]
    lm[RING_DIP] = [0.50, 0.30, 0.0]
    lm[RING_TIP] = [0.50, 0.17, 0.0]

    # Pinky: EXTENDED
    lm[PINKY_MCP] = [0.55, 0.54, 0.0]
    lm[PINKY_PIP] = [0.55, 0.44, 0.0]
    lm[PINKY_DIP] = [0.55, 0.33, 0.0]
    lm[PINKY_TIP] = [0.55, 0.23, 0.0]

    return lm


def _make_two_extended_landmarks():
    """Return landmarks with exactly 2 fingers extended (index + middle).

    Should classify as 'none' (uncertain).
    """
    lm = np.full((21, 3), 0.5, dtype=np.float32)

    # Index: EXTENDED
    lm[INDEX_MCP] = [0.40, 0.60, 0.0]
    lm[INDEX_PIP] = [0.40, 0.48, 0.0]
    lm[INDEX_DIP] = [0.40, 0.34, 0.0]
    lm[INDEX_TIP] = [0.40, 0.20, 0.0]

    # Middle: EXTENDED
    lm[MIDDLE_MCP] = [0.45, 0.58, 0.0]
    lm[MIDDLE_PIP] = [0.45, 0.44, 0.0]
    lm[MIDDLE_DIP] = [0.45, 0.30, 0.0]
    lm[MIDDLE_TIP] = [0.45, 0.18, 0.0]

    # Ring: BENT
    lm[RING_MCP] = [0.50, 0.56, 0.0]
    lm[RING_PIP] = [0.43, 0.52, 0.0]
    lm[RING_DIP] = [0.46, 0.50, 0.0]
    lm[RING_TIP] = [0.54, 0.54, 0.0]

    # Pinky: BENT
    lm[PINKY_MCP] = [0.55, 0.54, 0.0]
    lm[PINKY_PIP] = [0.48, 0.51, 0.0]
    lm[PINKY_DIP] = [0.51, 0.49, 0.0]
    lm[PINKY_TIP] = [0.59, 0.53, 0.0]

    return lm


def _make_one_extended_landmarks():
    """Return landmarks with exactly 1 finger extended (index only).

    Should classify as 'none' (uncertain).
    """
    lm = np.full((21, 3), 0.5, dtype=np.float32)

    # Index: EXTENDED
    lm[INDEX_MCP] = [0.40, 0.60, 0.0]
    lm[INDEX_PIP] = [0.40, 0.48, 0.0]
    lm[INDEX_DIP] = [0.40, 0.34, 0.0]
    lm[INDEX_TIP] = [0.40, 0.20, 0.0]

    # Middle: BENT
    lm[MIDDLE_MCP] = [0.45, 0.58, 0.0]
    lm[MIDDLE_PIP] = [0.38, 0.53, 0.0]
    lm[MIDDLE_DIP] = [0.42, 0.51, 0.0]
    lm[MIDDLE_TIP] = [0.50, 0.55, 0.0]

    # Ring: BENT
    lm[RING_MCP] = [0.50, 0.56, 0.0]
    lm[RING_PIP] = [0.43, 0.52, 0.0]
    lm[RING_DIP] = [0.46, 0.50, 0.0]
    lm[RING_TIP] = [0.54, 0.54, 0.0]

    # Pinky: BENT
    lm[PINKY_MCP] = [0.55, 0.54, 0.0]
    lm[PINKY_PIP] = [0.48, 0.51, 0.0]
    lm[PINKY_DIP] = [0.51, 0.49, 0.0]
    lm[PINKY_TIP] = [0.59, 0.53, 0.0]

    return lm


# ---------------------------------------------------------------------------
# Fixture
# ---------------------------------------------------------------------------


@pytest.fixture
def classifier():
    """Return a default GestureClassifier instance."""
    return GestureClassifier()


# ===================================================================
# A.  GestureClassifier construction
# ===================================================================


class TestGestureClassifierConstruction:
    """GestureClassifier.__init__ – default construction."""

    def test_default_construction_succeeds(self):
        """GestureClassifier can be instantiated with no arguments."""
        instance = GestureClassifier()
        assert instance is not None


# ===================================================================
# B.  classify() – return type and structure
# ===================================================================


class TestGestureClassifierClassify:
    """GestureClassifier.classify() – return tuple structure."""

    def test_returns_tuple_of_correct_types(self, classifier):
        """classify() must return (str, int, float)."""
        lm = _make_open_palm_landmarks()
        class_name, class_id, confidence = classifier.classify(lm)
        assert isinstance(class_name, str)
        assert isinstance(class_id, int)
        assert isinstance(confidence, float)

    def test_class_name_in_class_names(self, classifier):
        """class_name must be one of CLASS_NAMES."""
        results = [
            classifier.classify(_make_open_palm_landmarks())[0],
            classifier.classify(_make_closed_fist_landmarks())[0],
            classifier.classify(_make_two_extended_landmarks())[0],
        ]
        for name in results:
            assert name in CLASS_NAMES, f"{name!r} not in {CLASS_NAMES}"

    def test_class_id_matches_class_map(self, classifier):
        """class_id must match CLASS_MAP[class_name]."""
        lm = _make_open_palm_landmarks()
        class_name, class_id, _ = classifier.classify(lm)
        assert class_id == CLASS_MAP[class_name], (
            f"{class_id} != CLASS_MAP[{class_name!r}] = {CLASS_MAP[class_name]}"
        )

    def test_confidence_in_range(self, classifier):
        """confidence must be a float in [0, 1]."""
        results = [
            classifier.classify(_make_open_palm_landmarks()),
            classifier.classify(_make_closed_fist_landmarks()),
            classifier.classify(_make_two_extended_landmarks()),
        ]
        for _, _, conf in results:
            assert 0.0 <= conf <= 1.0, f"confidence {conf} out of range"


# ===================================================================
# C.  Classification – closed_fist
# ===================================================================


class TestGestureClassifierClosedFist:
    """GestureClassifier.classify() – closed fist (all fingers bent)."""

    def test_classifies_closed_fist(self, classifier):
        """All 4 fingers bent → ('closed_fist', 0, 0.9)."""
        lm = _make_closed_fist_landmarks()
        class_name, class_id, confidence = classifier.classify(lm)
        assert class_name == "closed_fist"
        assert class_id == CLASS_MAP["closed_fist"]
        assert confidence == pytest.approx(0.9)


# ===================================================================
# D.  Classification – open_palm
# ===================================================================


class TestGestureClassifierOpenPalm:
    """GestureClassifier.classify() – open palm (extended fingers)."""

    def test_all_four_extended_is_open_palm(self, classifier):
        """All 4 fingers extended → ('open_palm', 1, 0.9)."""
        lm = _make_open_palm_landmarks()
        class_name, class_id, confidence = classifier.classify(lm)
        assert class_name == "open_palm"
        assert class_id == CLASS_MAP["open_palm"]
        assert confidence == pytest.approx(0.9)

    def test_three_extended_is_open_palm(self, classifier):
        """3 fingers extended (≥ 3) → open_palm."""
        lm = _make_three_extended_landmarks()
        class_name, class_id, confidence = classifier.classify(lm)
        assert class_name == "open_palm", (
            f"Expected open_palm for 3 extended fingers, got {class_name}"
        )
        assert class_id == CLASS_MAP["open_palm"]
        assert confidence == pytest.approx(0.9)


# ===================================================================
# E.  Classification – uncertain (none)
# ===================================================================


class TestGestureClassifierUncertain:
    """GestureClassifier.classify() – uncertain gestures map to 'none'."""

    def test_two_extended_is_uncertain(self, classifier):
        """2 fingers extended → ('none', 2, 0.3)."""
        lm = _make_two_extended_landmarks()
        class_name, class_id, confidence = classifier.classify(lm)
        assert class_name == "none"
        assert class_id == CLASS_MAP["none"]
        assert confidence == pytest.approx(0.3)

    def test_one_extended_is_uncertain(self, classifier):
        """1 finger extended → ('none', 2, 0.3)."""
        lm = _make_one_extended_landmarks()
        class_name, class_id, confidence = classifier.classify(lm)
        assert class_name == "none"
        assert class_id == CLASS_MAP["none"]
        assert confidence == pytest.approx(0.3)


# ===================================================================
# F.  Edge cases
# ===================================================================


class TestGestureClassifierEdgeCases:
    """GestureClassifier.classify() – degenerate and boundary inputs."""

    def test_all_landmarks_same_position(self, classifier):
        """All 21 landmarks at identical coordinates → should not crash.

        With zero distances everywhere:
          d(MCP,TIP) = d(MCP,PIP) = 0 → 0 > 0 is False
        So extended_count == 0 → CLOSED_FIST.
        """
        lm = np.full((21, 3), 0.5, dtype=np.float32)
        class_name, class_id, confidence = classifier.classify(lm)
        assert class_name == "closed_fist"
        assert class_id == CLASS_MAP["closed_fist"]

    def test_landmarks_outside_zero_one(self, classifier):
        """Landmarks outside [0, 1] range should still be processed correctly."""
        lm = np.full((21, 3), 100.0, dtype=np.float32)
        # Override with open palm pattern (shifted to large coordinates)
        lm[INDEX_MCP] = [100.0, 100.0, 0.0]
        lm[INDEX_PIP] = [100.0, 99.0, 0.0]
        lm[INDEX_TIP] = [100.0, 95.0, 0.0]
        lm[MIDDLE_MCP] = [100.0, 100.0, 0.0]
        lm[MIDDLE_PIP] = [100.0, 99.0, 0.0]
        lm[MIDDLE_TIP] = [100.0, 95.0, 0.0]
        lm[RING_MCP] = [100.0, 100.0, 0.0]
        lm[RING_PIP] = [100.0, 99.0, 0.0]
        lm[RING_TIP] = [100.0, 95.0, 0.0]
        lm[PINKY_MCP] = [100.0, 100.0, 0.0]
        lm[PINKY_PIP] = [100.0, 99.0, 0.0]
        lm[PINKY_TIP] = [100.0, 95.0, 0.0]
        # All extended → open_palm
        class_name, class_id, confidence = classifier.classify(lm)
        assert class_name in CLASS_NAMES


# ===================================================================
# G.  Input validation
# ===================================================================


class TestGestureClassifierInputValidation:
    """GestureClassifier.classify() – rejects invalid inputs."""

    @pytest.mark.parametrize("shape", [
        (20, 3),
        (22, 3),
        (21, 2),
        (21, 4),
        (21,),
        (42, 3),
    ])
    def test_rejects_wrong_shape(self, classifier, shape):
        """landmarks with shape != (21, 3) must raise ValueError."""
        lm = np.zeros(shape, dtype=np.float32)
        with pytest.raises(ValueError, match="shape|21.*3"):
            classifier.classify(lm)

    def test_rejects_non_numpy_input(self, classifier):
        """Passing a plain Python list must raise TypeError."""
        with pytest.raises(TypeError, match="numpy|ndarray|array"):
            classifier.classify([[0.5] * 3] * 21)
