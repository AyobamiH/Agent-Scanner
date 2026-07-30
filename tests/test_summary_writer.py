"""Tests for summary writer utilities."""

import json
import logging
from pathlib import Path
from unittest.mock import patch

import pytest

from src.exceptions import SummaryWriteError
from src.models.results import RepoScanResult
from src.utils.summary_writer import write_summary_file


def _assert_single_output_with_prefix(output_dir: Path, prefix: str) -> Path:
    matches = list(output_dir.glob(f"{prefix}*.json"))
    assert matches, f"No output files found matching prefix {prefix}"
    assert len(matches) == 1, f"Expected one output file for prefix {prefix}, found {len(matches)}"
    return matches[0]


@pytest.fixture
def sample_repo_result() -> RepoScanResult:
    """Create a sample RepoScanResult for testing."""
    return RepoScanResult(
        repo_name="test-org/test-repo",
        org="test-org",
        provider="github",
        repo_url="https://github.com/test-org/test-repo",
        default_branch="main",
        scanned_branch="main",
        current_commit_hash="aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa",
        scan_id="00000000-0000-0000-0000-000000000000",
        scan_timestamp="2026-01-01T00:00:00Z",
        agentic_signals_detected=False,
    )


def test_write_summary_file_with_explicit_path(tmp_path: Path, sample_repo_result: RepoScanResult) -> None:
    """Write summary to an explicitly provided file path."""
    output_file = tmp_path / "custom_summary.json"

    write_summary_file(str(output_file), sample_repo_result)

    assert output_file.exists(), f"File was not created at {output_file}"
    content = json.loads(output_file.read_text())

    assert "schema_version" in content, "Missing schema_version"
    assert "repo" in content, "Missing repo section"
    assert "scan" in content, "Missing scan section"
    assert "detected" in content, "Missing detected section"

    assert content["repo"]["org"] == "test-org"
    assert content["repo"]["repo_name"] == "test-org/test-repo"
    assert content["repo"]["default_branch"] == "main"

    assert content["scan"]["scanned_branch"] == "main"
    assert content["scan"]["current_commit_hash"] == "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa"

    assert content["detected"]["signals"]["agentic"] is False


def test_write_summary_file_creates_parent_directories(tmp_path: Path, sample_repo_result: RepoScanResult) -> None:
    """Write summary creates parent directories if they do not exist."""
    nested_path = tmp_path / "nested" / "deep" / "summary.json"

    write_summary_file(str(nested_path), sample_repo_result)

    assert nested_path.exists()
    assert nested_path.parent.exists()


def test_write_summary_file_with_auto_generated_filename(
    tmp_path: Path,
    sample_repo_result: RepoScanResult,
    monkeypatch,
) -> None:
    """Write summary with auto-generated filename when path is None."""
    monkeypatch.chdir(tmp_path)

    write_summary_file(None, sample_repo_result)

    output_dir = tmp_path / "output"
    _assert_single_output_with_prefix(output_dir, "test-org-test-repo_main_")


def test_write_summary_file_with_empty_string_path(
    tmp_path: Path,
    sample_repo_result: RepoScanResult,
    monkeypatch,
) -> None:
    """Write summary auto-generates filename when path is empty string."""
    monkeypatch.chdir(tmp_path)

    write_summary_file("", sample_repo_result)

    output_dir = tmp_path / "output"
    _assert_single_output_with_prefix(output_dir, "test-org-test-repo_main_")


def test_write_summary_file_accepts_path_object(tmp_path: Path, sample_repo_result: RepoScanResult) -> None:
    """Write summary accepts a Path object for the output path."""
    output_file = tmp_path / "path_object_summary.json"

    write_summary_file(output_file, sample_repo_result)

    assert output_file.exists()


def test_write_summary_file_rejects_invalid_path_type(sample_repo_result: RepoScanResult) -> None:
    """Write summary raises ValueError when provided path type is invalid."""

    with pytest.raises(ValueError) as exc:
        write_summary_file(1234, sample_repo_result)

    assert "summary_file_path must be a string, Path, or None" in str(exc.value)


def test_write_summary_file_sanitises_repo_name(tmp_path: Path, monkeypatch) -> None:
    """Write summary sanitises slashes and backslashes in repo name."""
    result = RepoScanResult(
        repo_name="org/repo-name",
        org="org",
        provider="github",
        repo_url="https://github.com/org/repo-name",
        default_branch="feature/branch",
        scanned_branch="feature/branch",
        current_commit_hash="bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb",
        scan_id="11111111-1111-1111-1111-111111111111",
        scan_timestamp="2026-01-01T00:00:00Z",
        agentic_signals_detected=False,
    )
    monkeypatch.chdir(tmp_path)

    write_summary_file("", result)

    output_dir = tmp_path / "output"
    _assert_single_output_with_prefix(output_dir, "org-repo-name_feature-branch_")


def test_write_summary_file_raises_on_none_result() -> None:
    """Write summary raises ValueError when repo_result is None."""
    with pytest.raises(ValueError) as exc:
        write_summary_file("output.json", None)

    assert "repo_result cannot be None" in str(exc.value)


def test_write_summary_file_raises_on_os_error(tmp_path: Path, sample_repo_result: RepoScanResult) -> None:
    """Write summary raises SummaryWriteError on OS errors."""
    readonly_path = tmp_path / "readonly_dir" / "summary.json"
    readonly_path.parent.mkdir()

    with patch("builtins.open", side_effect=OSError("Permission denied")):
        with pytest.raises(SummaryWriteError) as exc:
            write_summary_file(str(readonly_path), sample_repo_result)

        assert "Failed to write summary file" in str(exc.value)


def test_write_summary_file_raises_on_serialisation_error(tmp_path: Path, sample_repo_result: RepoScanResult) -> None:
    """Write summary raises SummaryWriteError on JSON serialisation errors."""
    output_file = tmp_path / "bad_output.json"

    with patch.object(sample_repo_result, "to_dict", side_effect=TypeError("Cannot serialise")):
        with pytest.raises(SummaryWriteError) as exc:
            write_summary_file(str(output_file), sample_repo_result)

        assert "Failed to serialise summary data" in str(exc.value)


def test_write_summary_file_raises_on_json_dump_error(tmp_path: Path, sample_repo_result: RepoScanResult) -> None:
    """Write summary raises SummaryWriteError when json.dump fails during file write."""
    output_file = tmp_path / "json_dump_error.json"

    with patch("src.utils.summary_writer.json.dump", side_effect=TypeError("JSON dump failed")):
        with pytest.raises(SummaryWriteError) as exc:
            write_summary_file(str(output_file), sample_repo_result)

        assert "Failed to serialise summary data" in str(exc.value)


def test_write_summary_file_logs_success(tmp_path: Path, sample_repo_result: RepoScanResult, caplog) -> None:
    """Write summary logs info message on successful write."""
    output_file = tmp_path / "logged_summary.json"

    caplog.set_level(logging.INFO)
    write_summary_file(str(output_file), sample_repo_result)

    assert "Wrote summary to" in caplog.text
    assert str(output_file) in caplog.text


def test_write_summary_file_uses_atomic_write(tmp_path: Path, sample_repo_result: RepoScanResult) -> None:
    """Write summary should leave only the final file without temporary remnants."""
    output_file = tmp_path / "atomic_summary.json"

    write_summary_file(str(output_file), sample_repo_result)

    assert output_file.exists()
    temp_files = [path for path in tmp_path.iterdir() if path.suffix == ".tmp"]
    assert not temp_files


def test_write_summary_file_cleans_temp_on_dump_failure(tmp_path: Path, sample_repo_result: RepoScanResult) -> None:
    """Write summary should remove temporary files when json dump fails."""
    output_file = tmp_path / "failed_summary.json"

    with patch("src.utils.summary_writer.json.dump", side_effect=TypeError("JSON dump failed")):
        with pytest.raises(SummaryWriteError):
            write_summary_file(str(output_file), sample_repo_result)

    assert not output_file.exists()
    temp_files = [path for path in tmp_path.iterdir() if path.suffix == ".tmp"]
    assert not temp_files


def test_write_summary_file_schema_validation_failure(
    tmp_path: Path, sample_repo_result: RepoScanResult, monkeypatch
) -> None:
    """Write summary should fail when schema validation fails."""
    invalid_schema = {
        "type": "object",
        "required": ["nonexistent"],
        "properties": {"nonexistent": {"type": "string"}},
    }
    schema_path = tmp_path / "invalid.schema.json"
    schema_path.write_text(json.dumps(invalid_schema), encoding="utf-8")

    monkeypatch.setenv("AGENT_SCANNER_SCHEMA_PATH", str(schema_path))

    with pytest.raises(SummaryWriteError) as exc:
        write_summary_file(str(tmp_path / "out.json"), sample_repo_result)

    assert "Schema validation failed" in str(exc.value)
    assert not (tmp_path / "out.json").exists()


def test_write_summary_file_schema_path_override(
    tmp_path: Path, sample_repo_result: RepoScanResult, monkeypatch
) -> None:
    """Write summary should respect schema path override from environment."""
    schema_path = tmp_path / "schema.json"
    schema_path.write_text(json.dumps({"type": "object"}), encoding="utf-8")

    monkeypatch.setenv("AGENT_SCANNER_SCHEMA_PATH", str(schema_path))

    write_summary_file(str(tmp_path / "out.json"), sample_repo_result)

    assert (tmp_path / "out.json").exists()


def test_write_summary_file_logs_os_error(tmp_path: Path, sample_repo_result: RepoScanResult, caplog) -> None:
    """Write summary logs exception details on OS error."""
    output_file = tmp_path / "failing_summary.json"

    caplog.set_level(logging.ERROR)

    with patch("builtins.open", side_effect=OSError("Disk full")):
        with pytest.raises(SummaryWriteError):
            write_summary_file(str(output_file), sample_repo_result)

    assert "Failed writing summary" in caplog.text


def test_write_summary_file_with_special_characters_in_branch(tmp_path: Path, monkeypatch) -> None:
    """Write summary sanitises special characters in branch names."""
    result = RepoScanResult(
        repo_name="org/repo",
        org="org",
        provider="github",
        repo_url="https://github.com/org/repo",
        default_branch="main",
        scanned_branch="feature/JIRA-123/fix",
        current_commit_hash="cccccccccccccccccccccccccccccccccccccccc",
        scan_id="22222222-2222-2222-2222-222222222222",
        scan_timestamp="2026-01-01T00:00:00Z",
        agentic_signals_detected=False,
    )
    monkeypatch.chdir(tmp_path)

    write_summary_file("", result)

    output_dir = tmp_path / "output"
    _assert_single_output_with_prefix(output_dir, "org-repo_feature-JIRA-123-fix_")


@pytest.mark.parametrize(
    "repo_name,expected_safe",
    [
        ("org/repo", "org-repo"),
        ("org\\repo", "org-repo"),
        ("simple-repo", "simple-repo"),
    ],
)
def test_write_summary_file_sanitises_various_repo_names(
    repo_name: str,
    expected_safe: str,
    tmp_path: Path,
    monkeypatch,
) -> None:
    """Write summary correctly sanitises various repository name formats."""
    org_name = repo_name.split("/")[0] if "/" in repo_name else repo_name
    result = RepoScanResult(
        repo_name=repo_name,
        org=org_name,
        provider="github",
        repo_url=f"https://github.com/{org_name}/{repo_name.split('/')[-1]}",
        default_branch="main",
        scanned_branch="main",
        current_commit_hash="dddddddddddddddddddddddddddddddddddddddd",
        scan_id="33333333-3333-3333-3333-333333333333",
        scan_timestamp="2026-01-01T00:00:00Z",
        agentic_signals_detected=False,
    )
    monkeypatch.chdir(tmp_path)

    write_summary_file("", result)

    output_dir = tmp_path / "output"
    _assert_single_output_with_prefix(output_dir, f"{expected_safe}_main_")
