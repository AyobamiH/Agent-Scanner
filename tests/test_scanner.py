"""Tests for Scanner initialization and core functionality."""

import time
from unittest.mock import Mock

import pytest

from src.exceptions import GitHubClientError, ScannerError
from src.models.results import RepoScanResult
from src.scanner.scanner import Scanner


class TestScannerInit:
    """Tests for Scanner initialisation."""

    @staticmethod
    def test_init_with_github_client():
        """Test Scanner initialisation with a GitHub client."""
        github_client = Mock()
        scanner = Scanner(github_client)
        assert scanner.github == github_client
        assert scanner.matcher is not None
        assert scanner.agent_detector is not None
        assert scanner._file_matches == {}
        assert scanner._detection["matched_stage"] is None

    @staticmethod
    def test_init_with_custom_pattern_matcher():
        """Test Scanner initialisation with custom pattern matcher."""
        github_client = Mock()
        pattern_matcher = Mock()
        scanner = Scanner(github_client, pattern_matcher=pattern_matcher)
        assert scanner.matcher == pattern_matcher


class TestScanValidation:
    """Tests for scan method input validation."""

    @staticmethod
    def test_scan_invalid_repo_name_none():
        """Test scan with None repo_full_name."""
        github_client = Mock()
        scanner = Scanner(github_client)
        with pytest.raises(ScannerError, match="repo_full_name must be a non-empty string"):
            scanner.scan(None)  # NOSONAR

    @staticmethod
    def test_scan_invalid_repo_name_empty():
        """Test scan with empty repo_full_name."""
        github_client = Mock()
        scanner = Scanner(github_client)
        with pytest.raises(ScannerError, match="repo_full_name must be a non-empty string"):
            scanner.scan("")

    @staticmethod
    def test_scan_invalid_repo_format_no_slash():
        """Test scan with invalid format (no slash)."""
        github_client = Mock()
        scanner = Scanner(github_client)
        with pytest.raises(ScannerError, match="repo_full_name must be in owner/repo format"):
            scanner.scan("invalid-format")

    @staticmethod
    def test_scan_invalid_repo_format_not_string():
        """Test scan with non-string repo_full_name."""
        github_client = Mock()
        scanner = Scanner(github_client)
        with pytest.raises(ScannerError, match="repo_full_name must be a non-empty string"):
            scanner.scan(123)  # NOSONAR


class TestScanStage1:
    """Tests for Stage 1 (path scanning) of scan method."""

    @staticmethod
    def test_stage1_path_match_returns_repo_name():
        """Test Stage 1 positive match returns repo name."""
        github_client = Mock()
        github_client._api_url = "https://api.github.com"
        pattern_matcher = Mock()
        metadata = {"default_branch": "main", "head_sha": "abc123", "html_url": "https://github.com/owner/repo"}
        github_client.get_repo_tree.return_value = (
            [
                {"type": "blob", "path": "agent_framework.py"},
            ],
            metadata,
        )
        pattern_matcher.score_path.return_value = 1

        scanner = Scanner(github_client, pattern_matcher=pattern_matcher)
        result = scanner.scan("owner/repo")

        assert result == "owner/repo"
        assert scanner._repo_result.agentic_signals_detected is True
        assert scanner._repo_result.matched_stage == 1

    @staticmethod
    def test_stage1_no_match_continues_to_stage2():
        """Test Stage 1 no match continues to Stage 2."""
        github_client = Mock()
        github_client.max_workers = 2
        github_client.max_file_size = 1000000
        pattern_matcher = Mock()
        metadata = {"default_branch": "main", "head_sha": "abc123", "html_url": "https://github.com/owner/repo"}
        github_client.get_repo_tree.return_value = (
            [
                {"type": "blob", "path": "normal_file.py"},
            ],
            metadata,
        )
        pattern_matcher.score_path.return_value = 0
        github_client.get_file_content.return_value = "normal python code"
        pattern_matcher.score_content.return_value = 0
        pattern_matcher._tokenise_text.return_value = []

        agent_detector = Mock()
        agent_detector.count_agents_in_text.return_value = {}

        scanner = Scanner(github_client, pattern_matcher=pattern_matcher)
        scanner.agent_detector = agent_detector

        result = scanner.scan("owner/repo")

        assert result is None
        assert not hasattr(scanner, "_repo_result") or scanner._repo_result is None

    @staticmethod
    def test_scan_ignores_top_level_tests_folder():
        """Ensure paths under tests/ are ignored for detection."""
        github_client = Mock()
        github_client._api_url = "https://api.github.com"
        metadata = {"default_branch": "main", "head_sha": "abc123", "html_url": "https://github.com/owner/repo"}
        github_client.get_repo_tree.return_value = (
            [
                {"type": "blob", "path": "tests/agent_framework.py"},
            ],
            metadata,
        )

        pattern_matcher = Mock()

        scanner = Scanner(github_client, pattern_matcher=pattern_matcher)
        result = scanner.scan("owner/repo")

        assert result is None
        pattern_matcher.score_path.assert_not_called()


def test_scan_respects_ignore_paths_env(monkeypatch):  # NOSONAR S2325
    """Ignore paths from environment should be skipped during detection."""
    monkeypatch.setenv("AGENT_SCANNER_IGNORE_PATHS", "src/ignored")

    github_client = Mock()
    github_client._api_url = "https://api.github.com"
    metadata = {"default_branch": "main", "head_sha": "abc123", "html_url": "https://github.com/owner/repo"}
    github_client.get_repo_tree.return_value = (
        [
            {"type": "blob", "path": "src/ignored/agent.py"},
        ],
        metadata,
    )

    pattern_matcher = Mock()
    pattern_matcher.score_path.return_value = 1

    scanner = Scanner(github_client, pattern_matcher=pattern_matcher)
    result = scanner.scan("owner/repo")

    assert result is None
    pattern_matcher.score_path.assert_not_called()


def test_scan_times_out_when_deadline_exceeded(monkeypatch):  # NOSONAR S2325
    """Scanner should raise when deadline has already passed."""
    monkeypatch.setenv("AGENT_SCANNER_DEADLINE_EPOCH", str(time.time() - 5))

    github_client = Mock()
    scanner = Scanner(github_client)

    with pytest.raises(ScannerError, match="Scan exceeded timeout"):
        scanner.scan("owner/repo")


def test_dependency_files_are_sorted():
    """Dependency file paths should be sorted deterministically."""
    github_client = Mock()
    github_client.get_file_content.return_value = ""

    scanner = Scanner(github_client, pattern_matcher=Mock())
    result = RepoScanResult(repo_name="repo", org="org")

    tree = [
        {"type": "blob", "path": "pyproject.toml"},
        {"type": "blob", "path": "requirements.txt"},
    ]

    scanner._extract_dependencies("org", "repo", tree, result)

    assert result.dependency_files == ["pyproject.toml", "requirements.txt"]


class TestScanFileContents:
    """Tests for _scan_file_contents method."""

    @staticmethod
    def test_scan_file_contents_empty_files():
        """Test _scan_file_contents with empty files list."""
        github_client = Mock()
        scanner = Scanner(github_client)
        result = scanner._scan_file_contents("owner", "repo", [])

        assert result == (False, [])

    @staticmethod
    def test_scan_file_contents_reaches_threshold():
        """Test _scan_file_contents reaches required score threshold."""
        github_client = Mock()
        github_client.max_workers = 2
        github_client.max_file_size = 1000000
        pattern_matcher = Mock()
        github_client.get_file_content.return_value = "agentic content"
        pattern_matcher.score_content.return_value = 2
        pattern_matcher._tokenise_text.return_value = []

        agent_detector = Mock()
        agent_detector.count_agents_in_text.return_value = {}

        scanner = Scanner(github_client, pattern_matcher=pattern_matcher)
        scanner.agent_detector = agent_detector

        files = [
            {"path": "file1.py", "size": 100},
            {"path": "file2.py", "size": 100},
        ]

        result = scanner._scan_file_contents("owner", "repo", files, required_score=3)

        assert result == (True, ["file1.py", "file2.py"])

    @staticmethod
    def test_scan_file_contents_fail_fast_on_error():
        """Test _scan_file_contents with fail_fast raises on error."""
        github_client = Mock()
        github_client.max_workers = 2
        github_client.max_file_size = 1000000
        github_client.get_file_content.side_effect = GitHubClientError("API Error")

        scanner = Scanner(github_client)
        files = [{"path": "file1.py", "size": 100}]

        with pytest.raises(ScannerError, match="Failed to fetch"):
            scanner._scan_file_contents("owner", "repo", files, fail_fast=True)

        with pytest.raises(ScannerError, match="Failed to fetch"):
            scanner._scan_file_contents("owner", "repo", files, fail_fast=True)

    @staticmethod
    def test_scan_file_contents_skips_large_files():
        """Test _scan_file_contents skips files exceeding max size."""
        github_client = Mock()
        github_client.max_workers = 2
        github_client.max_file_size = 100
        github_client.get_file_content.return_value = "content"

        pattern_matcher = Mock()
        pattern_matcher.score_content.return_value = 0
        pattern_matcher._tokenise_text.return_value = []

        scanner = Scanner(github_client, pattern_matcher=pattern_matcher)
        scanner.agent_detector = Mock()

        files = [{"path": "large_file.py", "size": 200}]
        result = scanner._scan_file_contents("owner", "repo", files)

        assert result == (False, [])
        github_client.get_file_content.assert_not_called()

    @staticmethod
    def test_scan_file_contents_deduplicates_paths():
        """Test _scan_file_contents deduplicates paths using seen_paths."""
        github_client = Mock()
        github_client.max_workers = 2

        pattern_matcher = Mock()
        pattern_matcher.score_content.return_value = 0
        pattern_matcher._tokenise_text.return_value = []

        scanner = Scanner(github_client, pattern_matcher=pattern_matcher)
        scanner.agent_detector = Mock()
        scanner.agent_detector.count_agents_in_text.return_value = {}

        files = [{"path": "file1.py", "size": 100}]
        seen_paths = {"file1.py"}

        result = scanner._scan_file_contents("owner", "repo", files, seen_paths=seen_paths)

        assert result == (False, [])
        github_client.get_file_content.assert_not_called()


class TestExtractDependencies:
    """Tests for _extract_dependencies method."""

    @staticmethod
    def test_extract_dependencies_no_files():
        """Test _extract_dependencies with no dependency files."""
        github_client = Mock()
        scanner = Scanner(github_client)
        result = RepoScanResult(
            repo_name="repo",
            org="owner",
            agentic_signals_detected=True,
            matched_stage=1,
            matched_paths=None,
            dependency_files=[],
            ai_dependencies=[],
            agent_counts=[],
            agent_instances=[],
            parse_errors={},
        )

        scanner._extract_dependencies("owner", "repo", [], result)
        assert result.ai_dependencies == []


class TestExtractAgents:
    """Tests for _extract_agents method."""

    @staticmethod
    def test_extract_agents_from_file_matches():
        """Test _extract_agents uses cached file matches."""
        github_client = Mock()
        scanner = Scanner(github_client)
        scanner._file_matches = {
            "file.py": {
                "agent_locations": [
                    {"line": 10, "name": "LLMAgent", "detection_type": "call"},
                    {"line": 20, "name": "ChatAgent", "detection_type": "class"},
                ]
            }
        }

        agent_detector = Mock()
        agent_detector.aggregate_counts.return_value = {"LLMAgent": 1}
        agent_detector._normalise_agent_name = Mock(side_effect=lambda x: x.lower() if x else "")
        agent_detector.get_detection_priority = Mock(side_effect=lambda dt: {"class": 100, "call": 85}.get(dt, 50))
        scanner.agent_detector = agent_detector

        result = RepoScanResult(
            repo_name="repo",
            org="owner",
            agentic_signals_detected=True,
            matched_stage=1,
            matched_paths=None,
            dependency_files=[],
            ai_dependencies=[],
            agent_counts=[],
            agent_instances=[],
            parse_errors={},
            repo_url="https://example.com/owner/repo",
            current_commit_hash="abc123",
        )

        scanner._extract_agents("owner", "repo", [], result)

        assert len(result.agent_counts) > 0
        assert result.agent_counts[0]["count"] == 2
        assert len(result.agent_instances) == 1
        assert result.agent_instances[0]["file"] == "file.py"
        assert result.agent_instances[0]["count"] == 2
        assert len(result.agent_unique) == 1
        assert len(result.agent_unique[0]["agents"]) == 2
        assert result.agent_counts_unique[0]["count"] == 2


class TestIsIgnoredPath:
    """Tests for _is_ignored_path method."""

    @staticmethod
    def test_is_ignored_path_tests_folder():
        """Test _is_ignored_path returns True for paths under tests/."""
        github_client = Mock()
        scanner = Scanner(github_client)

        assert scanner._is_ignored_path("tests/agent.py") is True
        assert scanner._is_ignored_path("tests") is True
        assert scanner._is_ignored_path("tests/nested/file.py") is True

    @staticmethod
    def test_is_ignored_path_non_tests_folder():
        """Test _is_ignored_path returns False for normal paths."""
        github_client = Mock()
        scanner = Scanner(github_client)

        assert scanner._is_ignored_path("src/scanner.py") is False
        assert scanner._is_ignored_path("README.md") is False
        assert scanner._is_ignored_path("test_file.py") is False

    @staticmethod
    def test_is_ignored_path_non_string():
        """Test _is_ignored_path returns False for non-string input."""
        github_client = Mock()
        scanner = Scanner(github_client)

        assert scanner._is_ignored_path(123) is False  # NOSONAR
        assert scanner._is_ignored_path(None) is False  # NOSONAR
        assert scanner._is_ignored_path([]) is False  # NOSONAR

    @staticmethod
    def test_is_ignored_path_case_insensitive():
        """Test _is_ignored_path is case insensitive."""
        github_client = Mock()
        scanner = Scanner(github_client)

        assert scanner._is_ignored_path("TESTS/file.py") is True
        assert scanner._is_ignored_path("Tests/file.py") is True


class TestPrepareFetchTasks:
    """Tests for _prepare_fetch_tasks method."""

    @staticmethod
    def test_prepare_fetch_tasks_filters_seen_paths():
        """Test _prepare_fetch_tasks excludes already seen paths."""
        github_client = Mock()
        github_client.max_file_size = 1000000
        scanner = Scanner(github_client)

        files = [{"path": "file1.py", "size": 100}, {"path": "file2.py", "size": 100}]
        seen_paths = {"file1.py"}

        result = scanner._prepare_fetch_tasks(files, seen_paths)

        assert "file1.py" not in result
        assert "file2.py" in result

    @staticmethod
    def test_prepare_fetch_tasks_filters_large_files():
        """Test _prepare_fetch_tasks excludes oversized files."""
        github_client = Mock()
        github_client.max_file_size = 100
        scanner = Scanner(github_client)

        files = [
            {"path": "small.py", "size": 50},
            {"path": "large.py", "size": 200},
        ]
        seen_paths = set()

        result = scanner._prepare_fetch_tasks(files, seen_paths)

        assert "small.py" in result
        assert "large.py" not in result

    @staticmethod
    def test_prepare_fetch_tasks_deduplicates_files():
        """Test _prepare_fetch_tasks deduplicates file paths."""
        github_client = Mock()
        github_client.max_file_size = 1000000
        scanner = Scanner(github_client)

        files = [
            {"path": "file.py", "size": 100},
            {"path": "file.py", "size": 100},
        ]
        seen_paths = set()

        result = scanner._prepare_fetch_tasks(files, seen_paths)

        assert len(result) == 1
        assert "file.py" in result

    @staticmethod
    def test_prepare_fetch_tasks_skips_missing_path():
        """Test _prepare_fetch_tasks skips entries without path."""
        github_client = Mock()
        github_client.max_file_size = 1000000
        scanner = Scanner(github_client)

        files = [
            {"size": 100},
            {"path": "file.py", "size": 100},
        ]
        seen_paths = set()

        result = scanner._prepare_fetch_tasks(files, seen_paths)

        assert len(result) == 1
        assert "file.py" in result


class TestProcessSingleFileResult:
    """Tests for _process_single_file_result method."""

    @staticmethod
    def test_process_single_file_result_python_file_with_agents():
        """Test _process_single_file_result extracts agents from Python file."""
        github_client = Mock()
        pattern_matcher = Mock()
        pattern_matcher.score_content.return_value = 5
        pattern_matcher._tokenise_text.return_value = ["agent", "framework"]

        agent_detector = Mock()
        agent_detector._get_framework_imports.return_value = {"langchain"}
        agent_detector.get_agent_locations.return_value = [{"line": 10, "name": "LLMAgent"}]

        scanner = Scanner(github_client, pattern_matcher=pattern_matcher)
        scanner.agent_detector = agent_detector

        score, tokens, locations, imports = scanner._process_single_file_result(
            "file.py",
            "import langchain\nagent = LLMAgent()",
        )

        assert score == 5
        assert "agent" in tokens
        assert "framework" in tokens
        assert len(locations) == 1
        assert locations[0]["name"] == "LLMAgent"
        assert imports == {"langchain"}

    @staticmethod
    def test_process_single_file_result_yaml_file_with_agents():
        """Test _process_single_file_result extracts agents from YAML file."""
        github_client = Mock()
        pattern_matcher = Mock()
        pattern_matcher.score_content.return_value = 3
        pattern_matcher._tokenise_text.return_value = []

        agent_detector = Mock()
        agent_detector.get_structured_agent_locations.return_value = [{"line": 5, "agent_type": "LLMAgent"}]

        scanner = Scanner(github_client, pattern_matcher=pattern_matcher)
        scanner.agent_detector = agent_detector

        score, _, locations, imports = scanner._process_single_file_result(
            "config.yaml",
            "agents:\n  - type: LLMAgent",
        )

        assert score == 3
        assert len(locations) == 1
        assert imports == set()

    @staticmethod
    def test_process_single_file_result_syntax_error():
        """Test _process_single_file_result handles syntax errors gracefully."""
        github_client = Mock()
        pattern_matcher = Mock()
        pattern_matcher.score_content.return_value = 2
        pattern_matcher._tokenise_text.return_value = []

        agent_detector = Mock()
        agent_detector._get_framework_imports.side_effect = SyntaxError("Invalid syntax")
        agent_detector.get_agent_locations.return_value = []

        scanner = Scanner(github_client, pattern_matcher=pattern_matcher)
        scanner.agent_detector = agent_detector

        score, _, locations, imports = scanner._process_single_file_result(
            "broken.py",
            "this is not valid python !!!",
        )

        assert score == 2
        assert locations == []
        assert imports == set()

    @staticmethod
    def test_process_single_file_result_tokenisation_failure():
        """Test _process_single_file_result handles tokenisation failures."""
        github_client = Mock()
        pattern_matcher = Mock()
        pattern_matcher.score_content.return_value = 1
        pattern_matcher._tokenise_text.side_effect = Exception("Tokenisation failed")

        agent_detector = Mock()
        scanner = Scanner(github_client, pattern_matcher=pattern_matcher)
        scanner.agent_detector = agent_detector

        score, tokens, _, _ = scanner._process_single_file_result(
            "file.py",
            "content",
        )

        assert score == 1
        assert tokens == []

    @staticmethod
    def test_process_single_file_result_non_string_path():
        """Test _process_single_file_result handles non-string paths."""
        github_client = Mock()
        pattern_matcher = Mock()
        pattern_matcher.score_content.return_value = 1
        pattern_matcher._tokenise_text.return_value = []

        scanner = Scanner(github_client, pattern_matcher=pattern_matcher)

        score, _, locations, _ = scanner._process_single_file_result(
            123,  # NOSONAR
            "content",
        )

        assert score == 1
        assert locations == []


class TestLogScanSummary:
    """Tests for _log_scan_summary method."""

    def test_log_scan_summary_with_scores_and_errors(self, caplog):  # NOSONAR S2325
        """Test _log_scan_summary logs file scores and errors."""
        import logging

        github_client = Mock()
        scanner = Scanner(github_client)

        top_scores = [(5, "file1.py"), (3, "file2.py")]
        error_counts = {"API Error": 2, "Timeout": 1}
        total_score = 8
        required_score = 10

        with caplog.at_level(logging.INFO):
            scanner._log_scan_summary(top_scores, error_counts, total_score, required_score)

        assert "Top file scores" in caplog.text
        assert "Fetch failures summary" in caplog.text
        assert "did not reach required" in caplog.text

    def test_log_scan_summary_empty_scores_and_errors(self, caplog):  # NOSONAR S2325
        """Test _log_scan_summary handles empty scores and errors."""
        import logging

        github_client = Mock()
        scanner = Scanner(github_client)

        with caplog.at_level(logging.INFO):
            scanner._log_scan_summary([], {}, 0, 10)

        assert "did not reach required" in caplog.text


class TestExtractDependenciesEdgeCases:
    """Additional tests for _extract_dependencies method."""

    @staticmethod
    def test_extract_dependencies_with_parse_errors():
        """Test _extract_dependencies handles parse errors gracefully."""
        github_client = Mock()
        github_client.get_file_content.return_value = "invalid content"

        scanner = Scanner(github_client)

        result = RepoScanResult(
            repo_name="repo",
            org="owner",
            agentic_signals_detected=True,
            matched_stage=1,
            matched_paths=None,
            dependency_files=["requirements.txt"],
            ai_dependencies=[],
            agent_counts=[],
            agent_instances=[],
            parse_errors={},
        )

        scanner._extract_dependencies("owner", "repo", [], result)

        assert isinstance(result.parse_errors, dict)


class TestExtractAgentsEdgeCases:
    """Additional tests for _extract_agents method."""

    @staticmethod
    def test_extract_agents_no_cached_files_samples_repository():
        """Test _extract_agents samples files when no cache available."""
        github_client = Mock()
        github_client.get_file_content.return_value = "python code"

        pattern_matcher = Mock()
        scanner = Scanner(github_client, pattern_matcher=pattern_matcher)
        scanner._file_matches = {}

        agent_detector = Mock()
        agent_detector.get_agent_locations.return_value = []
        agent_detector._get_framework_imports.return_value = set()
        scanner.agent_detector = agent_detector

        language_detector = Mock()
        language_detector._get_language_from_path.return_value = "python"
        scanner.language_detector = language_detector

        result = RepoScanResult(
            repo_name="repo",
            org="owner",
            agentic_signals_detected=True,
            matched_stage=1,
            matched_paths=None,
            dependency_files=[],
            ai_dependencies=[],
            agent_counts=[],
            agent_instances=[],
            parse_errors={},
        )

        tree = [{"type": "blob", "path": "file.py", "size": 100}]

        scanner._extract_agents("owner", "repo", tree, result)

        assert result.agent_counts == []
        assert result.agent_instances == []


class TestScanWithBranch:
    """Tests for scan method with branch parameter."""

    @staticmethod
    def test_scan_with_explicit_branch():
        """Test scan method with explicit branch parameter."""
        github_client = Mock()
        github_client._api_url = "https://api.github.com"
        pattern_matcher = Mock()
        metadata = {"default_branch": "main", "head_sha": "abc123", "html_url": "https://github.com/owner/repo"}
        github_client.get_repo_tree.return_value = (
            [
                {"type": "blob", "path": "agent_framework.py"},
            ],
            metadata,
        )
        pattern_matcher.score_path.return_value = 1

        scanner = Scanner(github_client, pattern_matcher=pattern_matcher)
        result = scanner.scan("owner/repo", branch="develop")

        assert result == "owner/repo"
        github_client.get_repo_tree.assert_called_once()
        call_args = github_client.get_repo_tree.call_args
        assert call_args[0] == ("owner", "repo", "develop")


class TestScanWithVerbose:
    """Tests for scan method with verbose logging."""

    def test_scan_with_verbose_logging(self, caplog):  # NOSONAR S2325
        """Test scan method enables debug logging when verbose=True."""
        import logging

        github_client = Mock()
        github_client._api_url = "https://api.github.com"
        pattern_matcher = Mock()
        metadata = {"default_branch": "main", "head_sha": "abc123", "html_url": "https://github.com/owner/repo"}
        github_client.get_repo_tree.return_value = (
            [
                {"type": "blob", "path": "agent_framework.py"},
            ],
            metadata,
        )
        pattern_matcher.score_path.return_value = 1

        scanner = Scanner(github_client, pattern_matcher=pattern_matcher)

        with caplog.at_level(logging.DEBUG):
            result = scanner.scan("owner/repo", verbose=True)

        assert result == "owner/repo"


class TestExtractRepositoryInfo:
    """Tests for _extract_repository_info method."""

    @staticmethod
    def test_extract_repository_info_success():
        """Test _extract_repository_info extracts owner information."""
        github_client = Mock()
        scanner = Scanner(github_client)

        result = RepoScanResult(
            repo_name="repo",
            org="owner",
            agentic_signals_detected=True,
            matched_stage=1,
            matched_paths=None,
            dependency_files=[],
            ai_dependencies=[],
            agent_counts=[],
            agent_instances=[],
            parse_errors={},
        )

        scanner._extract_repository_info("owner", "repo", result)

        assert hasattr(result, "owner_detected")

    @staticmethod
    def test_extract_repository_info_failure():
        """Test _extract_repository_info handles detection failures gracefully."""
        github_client = Mock()
        scanner = Scanner(github_client)

        result = RepoScanResult(
            repo_name="repo",
            org="owner",
            agentic_signals_detected=True,
            matched_stage=1,
            matched_paths=None,
            dependency_files=[],
            ai_dependencies=[],
            agent_counts=[],
            agent_instances=[],
            parse_errors={},
        )

        scanner._extract_repository_info("owner", "repo", result)

        assert result.owner_detected is False
        assert result.detected_owner_name is None
        assert result.detected_owner_email is None


class TestScanStage2Stage3:
    """Tests for Stage 2 and Stage 3 scanning methods."""

    @staticmethod
    def test_run_stage_1_path_scan_no_match():
        """Test _run_stage_1_path_scan returns None when no paths match."""
        github_client = Mock()
        pattern_matcher = Mock()
        pattern_matcher.score_path.return_value = 0
        scanner = Scanner(github_client, pattern_matcher=pattern_matcher)

        blobs = [{"type": "blob", "path": "normal_file.py"}]
        result = scanner._run_stage_1_path_scan(blobs)

        assert result is None

    @staticmethod
    def test_run_stage_1_path_scan_non_string_path():
        """Test _run_stage_1_path_scan skips non-string paths."""
        github_client = Mock()
        pattern_matcher = Mock()
        pattern_matcher.score_path.return_value = 1
        scanner = Scanner(github_client, pattern_matcher=pattern_matcher)

        blobs = [
            {"type": "blob", "path": None},
            {"type": "blob", "path": 123},
        ]
        result = scanner._run_stage_1_path_scan(blobs)

        assert result is None
        pattern_matcher.score_path.assert_not_called()

    @staticmethod
    def test_scan_file_contents_no_files_to_fetch():
        """Test _scan_file_contents returns early when all files filtered."""
        github_client = Mock()
        github_client.max_file_size = 100
        pattern_matcher = Mock()
        scanner = Scanner(github_client, pattern_matcher=pattern_matcher)

        files = [{"path": "large.py", "size": 200}]
        result = scanner._scan_file_contents("owner", "repo", files)

        assert result == (False, [])
        github_client.get_file_content.assert_not_called()


class TestScanErrorPaths:
    """Tests for error handling in scan method."""

    @staticmethod
    def test_scan_with_github_enterprise_url():
        """Test scan correctly identifies GitHub Enterprise provider."""
        github_client = Mock()
        github_client._api_url = "https://github.enterprise.com/api/v3"
        pattern_matcher = Mock()
        metadata = {
            "default_branch": "main",
            "head_sha": "abc123",
            "html_url": "https://github.enterprise.com/owner/repo",
        }
        github_client.get_repo_tree.return_value = (
            [
                {"type": "blob", "path": "agent_framework.py"},
            ],
            metadata,
        )
        pattern_matcher.score_path.return_value = 1

        scanner = Scanner(github_client, pattern_matcher=pattern_matcher)
        result = scanner.scan("owner/repo")

        assert result == "owner/repo"
        assert scanner._repo_result.provider == "github-enterprise"


class TestScanStage2EdgeCases:
    """Tests for Stage 2 edge cases."""

    @staticmethod
    def test_run_stage_2_integration():
        """Test _run_stage_2_content_scan integration with file scanning."""
        github_client = Mock()
        github_client.max_workers = 1
        github_client.max_file_size = 1000000
        github_client.get_file_content.return_value = "python code"

        pattern_matcher = Mock()
        pattern_matcher.score_content.return_value = 0
        pattern_matcher._tokenise_text.return_value = []

        scanner = Scanner(github_client, pattern_matcher=pattern_matcher)
        scanner.agent_detector = Mock()

        blobs = [
            {"type": "blob", "path": "file1.py"},
            {"type": "blob", "path": "file2.py"},
        ]
        seen_paths = set()

        result = scanner._run_stage_2_content_scan("owner", "repo", blobs, False, seen_paths)

        assert result is None


class TestScanStage3EdgeCases:
    """Tests for Stage 3 edge cases."""

    @staticmethod
    def test_run_stage_3_with_no_unseen_files():
        """Test _run_stage_3_extended_scan when all files have been seen."""
        github_client = Mock()
        github_client.max_workers = 2
        pattern_matcher = Mock()
        scanner = Scanner(github_client, pattern_matcher=pattern_matcher)

        blobs = [
            {"type": "blob", "path": "file1.py"},
            {"type": "blob", "path": "file2.py"},
        ]
        seen_paths = {"file1.py", "file2.py"}

        result = scanner._run_stage_3_extended_scan("owner", "repo", blobs, False, seen_paths)

        assert result is None


class TestProcessSingleFileResultEdgeCases:
    """Additional edge case tests for _process_single_file_result."""

    @staticmethod
    def test_process_single_file_result_json_file():
        """Test _process_single_file_result processes JSON files."""
        github_client = Mock()
        pattern_matcher = Mock()
        pattern_matcher.score_content.return_value = 2
        pattern_matcher._tokenise_text.return_value = []

        agent_detector = Mock()
        agent_detector.get_structured_agent_locations.return_value = [{"line": 1}]

        scanner = Scanner(github_client, pattern_matcher=pattern_matcher)
        scanner.agent_detector = agent_detector

        score, _, locations, _ = scanner._process_single_file_result(
            "config.json",
            '{"agents": []}',
        )

        assert score == 2
        assert len(locations) == 1


class TestLocalFilesystemDeterminism:
    """Tests for deterministic local filesystem traversal."""

    def test_local_filesystem_tree_is_deterministic(self, tmp_path):  # NOSONAR S2325
        """Local filesystem tree ordering should be deterministic."""
        from src.github.filesystem_client import LocalFilesystemClient

        repo_root = tmp_path / "repo"
        (repo_root / "b").mkdir(parents=True, exist_ok=True)
        (repo_root / "a").mkdir(parents=True, exist_ok=True)
        (repo_root / "b" / "file.txt").write_text("b", encoding="utf-8")
        (repo_root / "a" / "file.txt").write_text("a", encoding="utf-8")

        github_client = Mock()
        client = LocalFilesystemClient(str(repo_root), github_client)
        tree_entries, _ = client.get_repo_tree("owner", "repo")
        paths = [entry["path"] for entry in tree_entries if "path" in entry]

        assert paths == sorted(paths)
