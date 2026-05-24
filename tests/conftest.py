"""Ensure mediapipe has the legacy solutions API available for mocking."""
import mediapipe
from types import SimpleNamespace

if not hasattr(mediapipe, 'solutions'):
    hands_ns = SimpleNamespace()
    hands_ns.Hands = None
    solutions_ns = SimpleNamespace()
    solutions_ns.hands = hands_ns
    mediapipe.solutions = solutions_ns
