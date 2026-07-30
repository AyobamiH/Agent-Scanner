"""Concurrency and Stage 3 behaviours for Scanner."""

from concurrent.futures import Future
from unittest.mock import MagicMock

import pytest

from src.exceptions import ScannerError
from src.scanner import scanner as scanner_module
from src.scanner.scanner import Scanner


def make_future(result=None, exc: Exception | None = None):
    fut: Future = Future()
    if exc:
        fut.set_exception(exc)
    else:
        fut.set_result(result)
    return fut


def test_scan_file_contents_fail_fast_timeout(monkeypatch):
    """_scan_file_contents should raise when future times out and fail_fast is True."""

    github_client = MagicMock()
    github_client.max_workers = 1
    github_client.max_file_size = 1_000_000

    scanner = Scanner(github_client)

    meta = {"path": "file.py", "size": 10}

    def fake_submit(fn, *a, **k):
        fut = MagicMock()
        fut.result.side_effect = TimeoutError()
        fut.done.return_value = False
        return fut

    def fake_as_completed(futs, timeout=None):
        return futs

    mock_executor = MagicMock()
    mock_executor.submit.side_effect = fake_submit
    mock_executor.__enter__.return_value = mock_executor

    monkeypatch.setattr(scanner_module, "ThreadPoolExecutor", lambda max_workers=1: mock_executor)
    monkeypatch.setattr(scanner_module, "as_completed", fake_as_completed)

    with pytest.raises(ScannerError):
        scanner._scan_file_contents("owner", "repo", [meta], fail_fast=True)


def test_scan_file_contents_cancels_after_threshold(monkeypatch):
    """When threshold met, remaining futures are cancelled."""

    github_client = MagicMock()
    github_client.max_workers = 2
    github_client.max_file_size = 1_000_000

    scanner = Scanner(github_client)
    scanner.matcher = MagicMock()
    scanner.matcher.score_content.return_value = 3
    scanner.matcher._tokenise_text.return_value = []
    scanner.agent_detector = MagicMock()
    scanner.agent_detector.get_agent_locations.return_value = []

    files = [{"path": "a.py", "size": 1}, {"path": "b.py", "size": 1}]

    completed = MagicMock()
    completed.result.return_value = "content-a"
    completed.done.return_value = True

    pending = MagicMock()
    pending.result.return_value = "content-b"
    pending.cancel = MagicMock(return_value=True)
    pending.done.return_value = False

    submitted = []

    def fake_submit(fn, *a, **k):
        submitted.append((fn, a, k))
        return completed if len(submitted) == 1 else pending

    mock_executor = MagicMock()
    mock_executor.submit.side_effect = fake_submit
    mock_executor.__enter__.return_value = mock_executor

    def fake_as_completed(futs, timeout=None):
        yield from futs

    monkeypatch.setattr(scanner_module, "ThreadPoolExecutor", lambda max_workers=2: mock_executor)
    monkeypatch.setattr(scanner_module, "as_completed", fake_as_completed)

    matched, contributing = scanner._scan_file_contents("owner", "repo", files, required_score=3)

    assert matched is True
    assert contributing == ["a.py"]
    assert pending.cancel.called  # type: ignore[attr-defined]


def test_stage3_sampling_uses_unseen_files(monkeypatch):
    """Stage 3 should sample only unseen files after Stage 2."""

    github_client = MagicMock()
    github_client._api_url = "https://api.github.com"
    github_client.max_workers = 1
    github_client.max_file_size = 1_000_000

    pattern_matcher = MagicMock()
    pattern_matcher.score_path.return_value = 0
    pattern_matcher.score_content.return_value = 0
    pattern_matcher._tokenise_text.return_value = []

    scanner = Scanner(github_client, pattern_matcher=pattern_matcher)
    scanner.agent_detector = MagicMock()
    scanner.agent_detector.get_agent_locations.return_value = []

    metadata = {"default_branch": "main", "head_sha": "abc", "html_url": "https://github.com/o/r"}
    blobs = [
        {"type": "blob", "path": "file1.py"},
        {"type": "blob", "path": "file2.py"},
        {"type": "blob", "path": "file3.py"},
    ]

    github_client.get_repo_tree.return_value = (blobs, metadata)
    github_client.get_file_content.side_effect = ["x", "y", "z"]

    result = scanner.scan("owner/repo", branch="main")

    assert result is None
    assert github_client.get_file_content.call_count == 3
