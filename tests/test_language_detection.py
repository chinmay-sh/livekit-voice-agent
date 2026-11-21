"""
Unit tests for _detect_language heuristic.

These tests are pure Python and do NOT require LiveKit credentials or a live LLM.
They test the language-detection logic used by GatekeeperAgent to identify
Hindi / Hinglish speaker language from transcribed text.

Run with: uv run pytest tests/test_language_detection.py -v
"""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from agent import _detect_language


def test_detect_language_none_for_english() -> None:
    assert _detect_language("Hello, I'd like to know about Alex.") is None
    assert _detect_language("What is the capital of France?") is None
    assert _detect_language("I am a recruiter looking for candidates.") is None


def test_detect_language_hindi_devanagari() -> None:
    assert (
        _detect_language("namaste main Alex ke baare mein jaanna chahta hoon")
        == "Hinglish"
    )
    assert _detect_language("aap kaisa feel karte hain") == "Hinglish"


def test_detect_language_devanagari_script() -> None:
    assert (
        _detect_language("namaste, main Alex ke baare mein jaanna chahta hoon")
        == "Hinglish"
    )


def test_detect_language_english_not_misclassified() -> None:
    # "main" is 1 out of 8 words = 12.5% < 25% threshold
    result = _detect_language("This is my main concern about the project")
    assert result is None


def test_detect_language_empty_string() -> None:
    assert _detect_language("") is None
