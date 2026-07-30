"""Tests for file utility functions."""

import pytest

from src.utils.file_utils import file_depth, is_code_file, sample_evenly_by_depth


def make_files(paths: list[str]) -> list[dict[str, str]]:
    """Create a list of file dicts from path strings for testing."""
    return [{"path": p} for p in paths]


@pytest.mark.parametrize(
    "path,expected",
    [
        ("README.md", 0),
        ("src/main.py", 1),
        ("a/b/c/d.txt", 3),
        ("", 0),
    ],
)
def test_file_depth(path: str, expected: int) -> None:
    """Depth calculation returns the expected integer for several paths."""
    assert file_depth(path) == expected


@pytest.mark.parametrize(
    "path,is_code",
    [
        ("src/main.py", True),
        ("image.png", False),
        ("Dockerfile", True),
    ],
)
def test_is_code_file(path: str, is_code: bool) -> None:
    """Detect whether a path should be considered a code/text file."""
    assert is_code_file(path) is is_code


def test_sample_evenly_by_depth_returns_limit():
    """Sampling returns the requested number of files when available."""
    files = make_files(["a.py", "b/c.py", "d/e/f.py", "g/h.py"])
    sampled = sample_evenly_by_depth(files, 3)
    assert len(sampled) == 3


def test_sample_evenly_by_depth_zero_limit() -> None:
    """Sampling with zero limit returns an empty list."""
    files = make_files(["a.py", "b/c.py"])

    sampled = sample_evenly_by_depth(files, 0)

    assert sampled == []


@pytest.mark.parametrize("path,expected", [("image.PNG", False), ("LICENSE", True)])
def test_is_code_file_binary_and_no_ext_parametrised(path, expected):
    """Detect code vs binary and files without an extension."""
    assert is_code_file(path) is expected


def test_sample_even_distribution():
    """Sample evenly returns the expected number of files across depths."""
    files = [
        {"path": "root.py"},
        {"path": "a/b.py"},
        {"path": "a/b/c.py"},
        {"path": "d/e/f.py"},
    ]
    sampled = sample_evenly_by_depth(files, 3)
    assert len(sampled) == 3


def test_sample_evenly_by_depth_negative_limit() -> None:
    """Sampling with a negative limit raises a ValueError."""
    files = make_files(["a.py"])

    with pytest.raises(ValueError):
        sample_evenly_by_depth(files, -1)


def test_sample_evenly_by_depth_more_files_than_limit():
    """Sampling returns limit when more files available across multiple depths."""
    files = make_files(["a.py", "b.py", "c/d.py", "e/f.py", "g/h/i.py", "j/k/l.py"])
    sampled = sample_evenly_by_depth(files, 4)
    assert len(sampled) == 4


def test_sample_evenly_by_depth_exhausts_shallower_depths_first():
    """Sampling continues correctly when some depths are exhausted."""
    files = make_files(["root.py", "a/b.py", "a/c.py", "d/e/f.py", "d/e/g.py", "d/e/h.py"])
    sampled = sample_evenly_by_depth(files, 5)
    assert len(sampled) == 5


def test_sample_evenly_by_depth_empty_files_list():
    """Sampling with empty files list returns empty list."""
    sampled = sample_evenly_by_depth([], 10)
    assert sampled == []
