"""Pytest configuration and shared fixtures for scanner tests."""

import os
import sys

import pytest

try:
    from src.detectors.patterns import PatternMatcher  # type: ignore # noqa: F401
except ModuleNotFoundError:
    ROOT = os.path.dirname(os.path.dirname(__file__))
    if ROOT not in sys.path:
        sys.path.insert(0, ROOT)

from src.detectors.patterns import PatternMatcher


@pytest.fixture
def pattern_matcher() -> PatternMatcher:
    """Provide a PatternMatcher instance for tests using the repo config."""
    return PatternMatcher.from_file()


@pytest.fixture
def custom_matcher() -> PatternMatcher:
    """Provide a PatternMatcher with a deterministic keyword set for unit tests."""
    return PatternMatcher(
        content_keywords=[
            "openai",
            "google.generativeai",
            "OPENAI_API_KEY",
            "langchain",
            "llm",
        ],
        content_whole_words=["ai", "agent", "prompt"],
    )


@pytest.fixture
def assert_detection_with_details():
    """
    Assertion helper for behavioural agent detection tests.

    This helper calls the detector and provides a rich assertion message including detected locations and code context.
    The `expected` argument supports:
      - int: exact match required
      - dict with "min": count must be >= min
      - dict with "exact": exact match required
    """

    def _assert_detection_with_details(detector, code: str, expected, test_name: str) -> None:
        count = detector.count_agents_in_text(code)

        expectation_text = ""
        ok = False

        if isinstance(expected, int):
            ok = count == expected
            expectation_text = f"Expected: {expected}"
        elif isinstance(expected, dict) and "min" in expected:
            ok = count >= expected["min"]
            expectation_text = f"Expected: >= {expected['min']}"
        elif isinstance(expected, dict) and "exact" in expected:
            ok = count == expected["exact"]
            expectation_text = f"Expected: {expected['exact']}"
        else:
            raise TypeError("Unsupported expected type. Use an int, {'min': int}, or {'exact': int}.")

        if ok:
            return

        locations = detector.get_agent_locations(code)
        raise AssertionError(
            f"Test '{test_name}' failed:\n"
            f"{expectation_text}\n"
            f"Got: {count}\n"
            f"Detected at: {locations}\n"
            f"Code:\n{code}"
        )

    return _assert_detection_with_details
