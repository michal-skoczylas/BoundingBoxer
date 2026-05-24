"""Unit tests for HandDetector component."""

from dataclasses import is_dataclass
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from boundingboxer.config import (
    MEDIAPIPE_MIN_DETECTION_CONFIDENCE,
    MEDIAPIPE_MIN_TRACKING_CONFIDENCE,
)
from boundingboxer.detector import HandDetection, HandDetector


# ---------------------------------------------------------------------------
# Mock helpers – simulate MediaPipe internals
# ---------------------------------------------------------------------------

def _mock_landmark(x=0.1, y=0.2, z=-0.05):
    """Return a MagicMock that mimics a MediaPipe NormalizedLandmark."""
    lm = MagicMock()
    lm.x = x
    lm.y = y
    lm.z = z
    return lm


def _mock_landmarks(n=21):
    """Return a list of *n* mock NormalizedLandmark objects."""
    return [_mock_landmark(x=i / n, y=(i % 7) / 7, z=0.0) for i in range(n)]


def _mock_classification(label="Left", score=0.95):
    """Return a mock classification entry with label and score."""
    cls = MagicMock()
    cls.label = label
    cls.score = score
    return cls


def _mock_handedness(label="Left", score=0.95):
    """Return a list containing one mock handedness classification."""
    h = MagicMock()
    h.classification = [_mock_classification(label, score)]
    return [h]


def _mock_process_result(landmarks_list, handedness_list):
    """
    Build a mock return value for hands.process().

    Parameters
    ----------
    landmarks_list : list[list[Mock]]
        One list of 21 landmarks per detected hand.
    handedness_list : list[list[Mock]]
        One handedness classification list per detected hand.
    """
    result = MagicMock()
    result.multi_hand_landmarks = landmarks_list
    result.multi_handedness = handedness_list
    return result


def _mock_empty_result():
    """Return a mock result with no hands detected (both attributes None)."""
    result = MagicMock()
    result.multi_hand_landmarks = None
    result.multi_handedness = None
    return result


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def sample_image():
    """Return a dummy 3-channel BGR image (10×15, filled with blue)."""
    img = np.zeros((10, 15, 3), dtype=np.uint8)
    img[:, :, 0] = 255  # blue channel
    return img


@pytest.fixture
def grayscale_image():
    """Return a dummy grayscale image (2D array, 10×15)."""
    return np.zeros((10, 15), dtype=np.uint8)


@pytest.fixture
def rgba_image():
    """Return a dummy 4-channel RGBA image (10×15×4)."""
    return np.zeros((10, 15, 4), dtype=np.uint8)


# ===================================================================
# A.  HandDetection dataclass
# ===================================================================


class TestHandDetection:
    """HandDetection – dataclass contract, field access, and validation."""

    # ---- basic dataclass contract ------------------------------------

    def test_is_dataclass(self):
        """HandDetection must be a @dataclass."""
        assert is_dataclass(HandDetection), "HandDetection is not a dataclass"

    def test_fields_accessible(self):
        """landmarks, handedness, detection_score are direct attributes."""
        landmarks = np.zeros((21, 3), dtype=np.float32)
        detection = HandDetection(
            landmarks=landmarks,
            handedness="Left",
            detection_score=0.95,
        )
        assert np.array_equal(detection.landmarks, landmarks)
        assert detection.handedness == "Left"
        assert detection.detection_score == pytest.approx(0.95)

    # ---- landmarks shape / type --------------------------------------

    def test_landmarks_is_ndarray(self):
        """landmarks must be a numpy.ndarray."""
        landmarks = np.zeros((21, 3), dtype=np.float32)
        detection = HandDetection(
            landmarks=landmarks,
            handedness="Left",
            detection_score=0.95,
        )
        assert isinstance(detection.landmarks, np.ndarray)

    def test_landmarks_correct_shape(self):
        """landmarks must have shape (21, 3)."""
        landmarks = np.zeros((21, 3), dtype=np.float32)
        detection = HandDetection(
            landmarks=landmarks,
            handedness="Right",
            detection_score=0.8,
        )
        assert detection.landmarks.shape == (21, 3)

    @pytest.mark.parametrize("shape", [
        (20, 3),   # too few landmarks
        (22, 3),   # too many landmarks
        (21, 2),   # wrong coordinate count (2 instead of 3)
        (21, 4),   # wrong coordinate count (4 instead of 3)
        (21,),     # 1D array
    ])
    def test_landmarks_rejects_wrong_shape(self, shape):
        """landmarks with shape != (21, 3) must raise ValueError."""
        landmarks = np.zeros(shape, dtype=np.float32)
        with pytest.raises(ValueError, match="landmarks.*shape.*21.*3"):
            HandDetection(
                landmarks=landmarks,
                handedness="Left",
                detection_score=0.5,
            )

    # ---- handedness validation ---------------------------------------

    def test_handedness_accepts_left(self):
        """handedness='Left' is valid."""
        landmarks = np.zeros((21, 3), dtype=np.float32)
        detection = HandDetection(
            landmarks=landmarks,
            handedness="Left",
            detection_score=0.5,
        )
        assert detection.handedness == "Left"

    def test_handedness_accepts_right(self):
        """handedness='Right' is valid."""
        landmarks = np.zeros((21, 3), dtype=np.float32)
        detection = HandDetection(
            landmarks=landmarks,
            handedness="Right",
            detection_score=0.5,
        )
        assert detection.handedness == "Right"

    @pytest.mark.parametrize("invalid", ["left", "RIGHT", "Unknown", "", None, 123])
    def test_handedness_rejects_invalid_values(self, invalid):
        """handedness must be exactly 'Left' or 'Right'; everything else raises."""
        landmarks = np.zeros((21, 3), dtype=np.float32)
        with pytest.raises(ValueError, match="[Hh]andedness"):
            HandDetection(
                landmarks=landmarks,
                handedness=invalid,
                detection_score=0.5,
            )

    # ---- detection_score validation ----------------------------------

    def test_detection_score_must_be_float(self):
        """detection_score must be a float (or float-compatible)."""
        landmarks = np.zeros((21, 3), dtype=np.float32)
        detection = HandDetection(
            landmarks=landmarks,
            handedness="Left",
            detection_score=0.95,
        )
        assert isinstance(detection.detection_score, float)

    @pytest.mark.parametrize("score", [0.0, 0.5, 1.0])
    def test_detection_score_valid_range(self, score):
        """detection_score in [0, 1] is accepted."""
        landmarks = np.zeros((21, 3), dtype=np.float32)
        detection = HandDetection(
            landmarks=landmarks,
            handedness="Left",
            detection_score=score,
        )
        assert detection.detection_score == pytest.approx(score)

    @pytest.mark.parametrize("score", [-0.1, 1.1, 2.0, -1.0])
    def test_detection_score_rejects_out_of_range(self, score):
        """detection_score outside [0, 1] must raise ValueError."""
        landmarks = np.zeros((21, 3), dtype=np.float32)
        with pytest.raises(ValueError, match="detection_score"):
            HandDetection(
                landmarks=landmarks,
                handedness="Left",
                detection_score=score,
            )


# ===================================================================
# B.  HandDetector construction
# ===================================================================


class TestHandDetectorConstruction:
    """HandDetector.__init__ – default/custom parameters, config integration."""

    # ---- default construction ----------------------------------------

    def test_default_construction_succeeds(self):
        """HandDetector can be instantiated with no arguments."""
        with patch("mediapipe.solutions.hands.Hands") as MockHands:
            MockHands.return_value = MagicMock()
            detector = HandDetector()
            assert detector is not None

    def test_default_uses_config_detection_confidence(self):
        """When no value is given, MEDIAPIPE_MIN_DETECTION_CONFIDENCE is used."""
        with patch("mediapipe.solutions.hands.Hands") as MockHands:
            MockHands.return_value = MagicMock()
            HandDetector()
            call_kwargs = MockHands.call_args.kwargs
            assert call_kwargs.get(
                "min_detection_confidence"
            ) == MEDIAPIPE_MIN_DETECTION_CONFIDENCE

    def test_default_uses_config_tracking_confidence(self):
        """When no value is given, MEDIAPIPE_MIN_TRACKING_CONFIDENCE is used."""
        with patch("mediapipe.solutions.hands.Hands") as MockHands:
            MockHands.return_value = MagicMock()
            HandDetector()
            call_kwargs = MockHands.call_args.kwargs
            assert call_kwargs.get(
                "min_tracking_confidence"
            ) == MEDIAPIPE_MIN_TRACKING_CONFIDENCE

    def test_default_max_num_hands(self):
        """Default max_num_hands should be 2 (both hands detectable)."""
        with patch("mediapipe.solutions.hands.Hands") as MockHands:
            MockHands.return_value = MagicMock()
            HandDetector()
            call_kwargs = MockHands.call_args.kwargs
            assert call_kwargs.get("max_num_hands") == 2

    # ---- custom parameters -------------------------------------------

    def test_custom_detection_confidence(self):
        """Custom min_detection_confidence is forwarded to MediaPipe."""
        with patch("mediapipe.solutions.hands.Hands") as MockHands:
            MockHands.return_value = MagicMock()
            HandDetector(min_detection_confidence=0.7)
            call_kwargs = MockHands.call_args.kwargs
            assert call_kwargs.get("min_detection_confidence") == 0.7

    def test_custom_tracking_confidence(self):
        """Custom min_tracking_confidence is forwarded to MediaPipe."""
        with patch("mediapipe.solutions.hands.Hands") as MockHands:
            MockHands.return_value = MagicMock()
            HandDetector(min_tracking_confidence=0.3)
            call_kwargs = MockHands.call_args.kwargs
            assert call_kwargs.get("min_tracking_confidence") == 0.3

    def test_custom_max_num_hands(self):
        """Custom max_num_hands is forwarded to MediaPipe."""
        with patch("mediapipe.solutions.hands.Hands") as MockHands:
            MockHands.return_value = MagicMock()
            HandDetector(max_num_hands=1)
            call_kwargs = MockHands.call_args.kwargs
            assert call_kwargs.get("max_num_hands") == 1

    def test_static_image_mode_is_enabled(self):
        """HandDetector must request static_image_mode=True for individual images."""
        with patch("mediapipe.solutions.hands.Hands") as MockHands:
            MockHands.return_value = MagicMock()
            HandDetector()
            call_kwargs = MockHands.call_args.kwargs
            assert call_kwargs.get("static_image_mode") is True, (
                "Expected static_image_mode=True for single-image processing"
            )


# ===================================================================
# C.  detect() – input validation
# ===================================================================


class TestDetectInputValidation:
    """HandDetector.detect() – rejects invalid inputs before calling MediaPipe."""

    @pytest.fixture
    def detector(self):
        """HandDetector with mocked MediaPipe backend."""
        with patch("mediapipe.solutions.hands.Hands") as MockHands:
            MockHands.return_value = MagicMock()
            yield HandDetector()

    def test_raises_for_non_numpy_list(self, detector):
        """Passing a plain Python list instead of ndarray must raise TypeError."""
        with pytest.raises(TypeError, match="numpy|ndarray|array"):
            detector.detect([1, 2, 3])  # type: ignore[arg-type]

    def test_raises_for_grayscale_image(self, detector, grayscale_image):
        """A 2D grayscale image must raise ValueError."""
        with pytest.raises(ValueError, match="grayscale|2D|dimension|channel"):
            detector.detect(grayscale_image)

    def test_raises_for_rgba_image(self, detector, rgba_image):
        """A 4-channel RGBA image must raise ValueError (expected 3-channel BGR)."""
        with pytest.raises(ValueError, match="channel|RGBA|4"):
            detector.detect(rgba_image)

    def test_accepts_valid_bgr_image(self, detector, sample_image):
        """A valid 3-channel uint8 ndarray is accepted — no exception raised."""
        # We need to mock process return so it doesn't crash on real MediaPipe
        detector._hands.process.return_value = _mock_empty_result()
        try:
            detector.detect(sample_image)
        except Exception as exc:
            pytest.fail(f"Valid BGR image raised unexpected exception: {exc}")


# ===================================================================
# D.  detect() – integration with MediaPipe (mocked)
# ===================================================================


class TestDetectIntegration:
    """HandDetector.detect() – correctly transforms MediaPipe results."""

    @staticmethod
    def _make_detector():
        """Create a HandDetector with a fresh mock Hands instance."""
        with patch("mediapipe.solutions.hands.Hands") as MockHands:
            mock_instance = MagicMock()
            MockHands.return_value = mock_instance
            detector = HandDetector()
            return detector, mock_instance

    # ---- no hands ----------------------------------------------------

    def test_detect_no_hands(self, sample_image):
        """When MediaPipe finds no hands, an empty list is returned."""
        detector, mock_hands = self._make_detector()
        mock_hands.process.return_value = _mock_empty_result()

        results = detector.detect(sample_image)

        assert isinstance(results, list)
        assert len(results) == 0

    def test_detect_none_landmarks_returns_empty(self, sample_image):
        """When multi_hand_landmarks is None, return empty list."""
        detector, mock_hands = self._make_detector()
        result = MagicMock()
        result.multi_hand_landmarks = None
        result.multi_handedness = None
        mock_hands.process.return_value = result

        results = detector.detect(sample_image)

        assert results == []

    # ---- one hand ----------------------------------------------------

    def test_detect_one_hand_left(self, sample_image):
        """Single left hand → one HandDetection with handedness='Left'."""
        detector, mock_hands = self._make_detector()
        mock_result = _mock_process_result(
            landmarks_list=[_mock_landmarks(21)],
            handedness_list=_mock_handedness(label="Left", score=0.95),
        )
        mock_hands.process.return_value = mock_result

        results = detector.detect(sample_image)

        assert len(results) == 1
        det = results[0]
        assert isinstance(det, HandDetection)
        assert det.handedness == "Left"

    def test_detect_one_hand_right(self, sample_image):
        """Single right hand → one HandDetection with handedness='Right'."""
        detector, mock_hands = self._make_detector()
        mock_result = _mock_process_result(
            landmarks_list=[_mock_landmarks(21)],
            handedness_list=_mock_handedness(label="Right", score=0.88),
        )
        mock_hands.process.return_value = mock_result

        results = detector.detect(sample_image)

        assert len(results) == 1
        det = results[0]
        assert isinstance(det, HandDetection)
        assert det.handedness == "Right"

    # ---- two hands ---------------------------------------------------

    def test_detect_two_hands(self, sample_image):
        """Both hands detected → two HandDetection objects (Left + Right)."""
        detector, mock_hands = self._make_detector()
        mock_result = _mock_process_result(
            landmarks_list=[_mock_landmarks(21), _mock_landmarks(21)],
            handedness_list=[
                _mock_handedness(label="Left", score=0.95)[0],
                _mock_handedness(label="Right", score=0.91)[0],
            ],
        )
        mock_hands.process.return_value = mock_result

        results = detector.detect(sample_image)

        assert len(results) == 2
        assert results[0].handedness == "Left"
        assert results[1].handedness == "Right"

    # ---- landmarks extraction ----------------------------------------

    def test_landmarks_extracted_as_ndarray_21x3(self, sample_image):
        """Each detection.landmarks must be a numpy array of shape (21, 3)."""
        detector, mock_hands = self._make_detector()
        mock_result = _mock_process_result(
            landmarks_list=[_mock_landmarks(21)],
            handedness_list=_mock_handedness(label="Left", score=0.95),
        )
        mock_hands.process.return_value = mock_result

        results = detector.detect(sample_image)

        det = results[0]
        assert isinstance(det.landmarks, np.ndarray)
        assert det.landmarks.shape == (21, 3)
        assert det.landmarks.dtype == np.float32

    def test_landmarks_contain_xyz_coordinates(self, sample_image):
        """Landmarks values are extracted from NormalizedLandmark x, y, z."""
        detector, mock_hands = self._make_detector()
        lm_list = [_mock_landmark(x=0.1, y=0.2, z=0.3) for _ in range(21)]
        mock_result = _mock_process_result(
            landmarks_list=[lm_list],
            handedness_list=_mock_handedness(label="Right", score=0.8),
        )
        mock_hands.process.return_value = mock_result

        results = detector.detect(sample_image)

        det = results[0]
        # Verify specific coordinates from the mock landmarks
        assert det.landmarks[0, 0] == pytest.approx(0.1)  # x
        assert det.landmarks[0, 1] == pytest.approx(0.2)  # y
        assert det.landmarks[0, 2] == pytest.approx(0.3)  # z
        # All 21 landmarks should be present
        assert det.landmarks.shape[0] == 21

    def test_landmarks_normalized_to_zero_one(self, sample_image):
        """Landmark x, y values must be within [0, 1] (MediaPipe convention)."""
        detector, mock_hands = self._make_detector()
        # Create landmarks with boundary values
        lm_list = [
            _mock_landmark(x=0.0, y=1.0, z=0.0),
        ] + [_mock_landmark(x=0.5, y=0.5, z=0.0) for _ in range(20)]
        mock_result = _mock_process_result(
            landmarks_list=[lm_list],
            handedness_list=_mock_handedness(label="Left", score=0.9),
        )
        mock_hands.process.return_value = mock_result

        results = detector.detect(sample_image)

        det = results[0]
        assert np.all(det.landmarks[:, 0] >= 0.0) and np.all(det.landmarks[:, 0] <= 1.0)
        assert np.all(det.landmarks[:, 1] >= 0.0) and np.all(det.landmarks[:, 1] <= 1.0)

    # ---- detection score extraction ----------------------------------

    def test_detection_score_extracted_correctly(self, sample_image):
        """detection_score comes from the handedness classification score."""
        detector, mock_hands = self._make_detector()
        mock_result = _mock_process_result(
            landmarks_list=[_mock_landmarks(21)],
            handedness_list=_mock_handedness(label="Left", score=0.73),
        )
        mock_hands.process.return_value = mock_result

        results = detector.detect(sample_image)

        det = results[0]
        assert det.detection_score == pytest.approx(0.73)

    def test_detection_score_is_float(self, sample_image):
        """detection_score must be a float, not numpy scalar or int."""
        detector, mock_hands = self._make_detector()
        mock_result = _mock_process_result(
            landmarks_list=[_mock_landmarks(21)],
            handedness_list=_mock_handedness(label="Right", score=0.5),
        )
        mock_hands.process.return_value = mock_result

        results = detector.detect(sample_image)

        det = results[0]
        assert isinstance(det.detection_score, float)

    # ---- hands.process is called with RGB image ----------------------

    def test_process_receives_rgb_image(self, sample_image):
        """MediaPipe expects RGB; BGR input must be converted before process()."""
        detector, mock_hands = self._make_detector()
        mock_hands.process.return_value = _mock_empty_result()

        detector.detect(sample_image)

        # The image passed to process() should be the RGB-converted version
        call_arg = mock_hands.process.call_args[0][0]
        assert call_arg is not sample_image, (
            "Expected a copy/converted image, not the original BGR array"
        )


# ===================================================================
# E.  Resource cleanup
# ===================================================================


class TestHandDetectorResourceCleanup:
    """HandDetector – close() and context-manager protocol."""

    def test_close_calls_hands_close(self):
        """close() must delegate to the underlying MediaPipe Hands.close()."""
        with patch("mediapipe.solutions.hands.Hands") as MockHands:
            mock_instance = MagicMock()
            MockHands.return_value = mock_instance
            detector = HandDetector()

            detector.close()

            mock_instance.close.assert_called_once()

    def test_context_manager_enter(self):
        """__enter__ must return self (standard context manager protocol)."""
        with patch("mediapipe.solutions.hands.Hands") as MockHands:
            MockHands.return_value = MagicMock()
            detector = HandDetector()

            ctx_obj = detector.__enter__()

            assert ctx_obj is detector

    def test_context_manager_exit_calls_close(self):
        """__exit__ must call close() (even on exception)."""
        with patch("mediapipe.solutions.hands.Hands") as MockHands:
            mock_instance = MagicMock()
            MockHands.return_value = mock_instance
            detector = HandDetector()

            detector.__exit__(None, None, None)

            mock_instance.close.assert_called_once()

    def test_context_manager_with_statement(self):
        """HandDetector works with the 'with' statement."""
        with patch("mediapipe.solutions.hands.Hands") as MockHands:
            mock_instance = MagicMock()
            MockHands.return_value = mock_instance

            with HandDetector() as detector:
                assert detector is not None
                # Inside the block, Hands should be open
                mock_instance.close.assert_not_called()

            # After the block, close() must have been called
            mock_instance.close.assert_called_once()

    def test_close_is_idempotent(self):
        """Calling close() multiple times should not raise errors."""
        with patch("mediapipe.solutions.hands.Hands") as MockHands:
            mock_instance = MagicMock()
            MockHands.return_value = mock_instance
            detector = HandDetector()

            detector.close()
            detector.close()  # second call must not crash

            assert mock_instance.close.call_count >= 1
