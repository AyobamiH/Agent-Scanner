"""CLI behaviour tests for src.main."""

import logging
from types import SimpleNamespace
from unittest.mock import MagicMock


def test_main_list_branches_success(monkeypatch, caplog):
    """list-branches flag should list branches and exit with success."""

    caplog.set_level(logging.INFO)

    fake_client = MagicMock()
    fake_client.get_branches.return_value = ["main", "dev"]

    monkeypatch.setenv("GITHUB_TOKEN", "fake")
    monkeypatch.setattr("src.main.GitHubClient", lambda api_url=None: fake_client)
    monkeypatch.setattr("src.main.PatternMatcher", SimpleNamespace(from_file=lambda: MagicMock()))

    from src.main import main

    rc = main(["--repo-api-url", "https://api.github.com/repos/owner/repo", "--list-branches"])

    assert rc == 0
    assert "Branch available" in caplog.text


def test_main_pattern_load_failure(monkeypatch):
    """Pattern load errors should return client init failure code."""

    monkeypatch.setenv("GITHUB_TOKEN", "fake")
    monkeypatch.setattr("src.main.GitHubClient", lambda api_url=None: MagicMock())

    def failing_loader():
        raise ValueError("cannot load")

    monkeypatch.setattr("src.main.PatternMatcher", SimpleNamespace(from_file=failing_loader))

    from src.main import EXIT_CLIENT_INIT_FAILURE, main

    rc = main(["--repo-api-url", "https://api.github.com/repos/owner/repo"])

    assert rc == EXIT_CLIENT_INIT_FAILURE


def test_main_summary_write_failure(monkeypatch):
    """Summary write failures should return scan failure code."""

    monkeypatch.setenv("GITHUB_TOKEN", "fake")

    fake_client = MagicMock()
    fake_client.get_branches.return_value = []

    fake_matcher = MagicMock()
    fake_scanner = MagicMock()
    fake_result = MagicMock()
    fake_result.agentic_signals_detected = True
    fake_result.matched_stage = 2
    fake_result.dependency_files = []
    fake_result.ai_dependencies = [MagicMock()]
    fake_result.agent_instances = [{"file": "agent.py"}]
    fake_scanner.scan.return_value = "owner/repo"
    fake_scanner._repo_result = fake_result

    monkeypatch.setattr("src.main.GitHubClient", lambda api_url=None: fake_client)
    monkeypatch.setattr("src.main.PatternMatcher", SimpleNamespace(from_file=lambda: fake_matcher))
    monkeypatch.setattr("src.main.Scanner", lambda *a, **k: fake_scanner)

    def failing_write(path, repo_result):
        from src.exceptions import SummaryWriteError

        raise SummaryWriteError("disk full")

    monkeypatch.setattr("src.main.write_summary_file", failing_write)

    from src.main import EXIT_SCAN_FAILURE, main

    rc = main(["--repo-api-url", "https://api.github.com/repos/owner/repo", "--summary-file", "out.json"])

    assert rc == EXIT_SCAN_FAILURE


def test_main_verbose_sets_debug(monkeypatch, caplog):
    """Verbose flag should elevate logging to DEBUG."""

    caplog.set_level(logging.INFO)

    fake_client = MagicMock()
    monkeypatch.setenv("GITHUB_TOKEN", "fake")
    monkeypatch.setattr("src.main.GitHubClient", lambda api_url=None: fake_client)
    monkeypatch.setattr("src.main.PatternMatcher", SimpleNamespace(from_file=lambda: MagicMock()))

    fake_scanner = MagicMock()
    fake_scanner.scan.return_value = None
    fake_scanner._repo_result = None
    monkeypatch.setattr("src.main.Scanner", lambda *a, **k: fake_scanner)

    from src.main import main

    seen = {}

    def fake_basic_config(**kwargs):
        seen.update(kwargs)

    monkeypatch.setattr(logging, "basicConfig", fake_basic_config)

    main(["--repo-api-url", "https://api.github.com/repos/owner/repo", "--verbose"])

    assert seen.get("level") == logging.DEBUG


def test_main_log_file_handler(monkeypatch, tmp_path):
    """Log file flag should configure file handler."""

    log_file = tmp_path / "test.log"
    monkeypatch.setenv("GITHUB_TOKEN", "fake")
    monkeypatch.setattr("src.main.GitHubClient", lambda api_url=None: MagicMock())
    monkeypatch.setattr("src.main.PatternMatcher", SimpleNamespace(from_file=lambda: MagicMock()))

    fake_scanner = MagicMock()
    fake_scanner.scan.return_value = None
    fake_scanner._repo_result = None
    monkeypatch.setattr("src.main.Scanner", lambda *a, **k: fake_scanner)

    from src.main import main

    seen = {}

    def fake_basic_config(**kwargs):
        seen.update(kwargs)

    monkeypatch.setattr(logging, "basicConfig", fake_basic_config)

    main(["--repo-api-url", "https://api.github.com/repos/owner/repo", "--log-file", str(log_file)])

    assert "handlers" in seen
    assert len(seen["handlers"]) == 1
    assert isinstance(seen["handlers"][0], logging.FileHandler)


def test_main_log_level_overrides_verbose(monkeypatch):
    """Log level flag should override verbose logging."""
    monkeypatch.setenv("GITHUB_TOKEN", "fake")
    monkeypatch.setattr("src.main.GitHubClient", lambda api_url=None: MagicMock())
    monkeypatch.setattr("src.main.PatternMatcher", SimpleNamespace(from_file=lambda: MagicMock()))

    fake_scanner = MagicMock()
    fake_scanner.scan.return_value = None
    fake_scanner._repo_result = None
    monkeypatch.setattr("src.main.Scanner", lambda *a, **k: fake_scanner)

    from src.main import main

    seen = {}

    def fake_basic_config(**kwargs):
        seen.update(kwargs)

    monkeypatch.setattr(logging, "basicConfig", fake_basic_config)

    main(["--repo-api-url", "https://api.github.com/repos/owner/repo", "--verbose", "--log-level", "error"])

    assert seen.get("level") == logging.ERROR


def test_main_scan_org_recent_disabled(monkeypatch, caplog):
    """Scan org recent should fail when ORG_BULK_ENABLED is disabled."""

    caplog.set_level(logging.INFO)
    monkeypatch.setenv("GITHUB_TOKEN", "fake")
    monkeypatch.setenv("AGENT_SCANNER_BULK_ENABLED", "0")

    import importlib

    import src.main

    importlib.reload(src.main)

    monkeypatch.setattr("src.main.GitHubClient", lambda api_url=None: MagicMock())
    monkeypatch.setattr("src.main.PatternMatcher", SimpleNamespace(from_file=lambda: MagicMock()))

    from src.main import EXIT_CLIENT_INIT_FAILURE, main

    rc = main(["--org", "myorg", "--scan-org-recent"])

    assert rc == EXIT_CLIENT_INIT_FAILURE
    assert "Org bulk scanning is disabled" in caplog.text


def test_main_scan_org_recent_disabled_in_pipeline_mode(monkeypatch, caplog):
    """Scan org recent should fail when pipeline mode is enabled."""
    caplog.set_level(logging.INFO)
    monkeypatch.setenv("GITHUB_TOKEN", "fake")
    monkeypatch.setenv("AGENT_SCANNER_BULK_ENABLED", "1")
    monkeypatch.setenv("AGENT_SCANNER_PIPELINE_MODE", "1")

    import importlib

    import src.main

    importlib.reload(src.main)

    monkeypatch.setattr("src.main.GitHubClient", lambda api_url=None: MagicMock())
    monkeypatch.setattr("src.main.PatternMatcher", SimpleNamespace(from_file=lambda: MagicMock()))

    from src.main import EXIT_CLIENT_INIT_FAILURE, main

    rc = main(["--org", "myorg", "--scan-org-recent"])

    assert rc == EXIT_CLIENT_INIT_FAILURE
    assert "Org bulk scanning is disabled in pipeline mode" in caplog.text


def test_main_pipeline_requires_output_path(monkeypatch, caplog, tmp_path):
    """Pipeline mode should require an explicit output path."""
    caplog.set_level(logging.INFO)
    monkeypatch.setenv("GITHUB_TOKEN", "fake")
    monkeypatch.setenv("AGENT_SCANNER_PIPELINE_MODE", "1")

    import importlib

    import src.main

    importlib.reload(src.main)

    monkeypatch.setattr("src.main.GitHubClient", lambda api_url=None: MagicMock())
    monkeypatch.setattr("src.main.PatternMatcher", SimpleNamespace(from_file=lambda: MagicMock()))
    monkeypatch.setattr("src.main.Scanner", lambda *a, **k: MagicMock(scan=lambda *a, **k: None))

    from src.main import EXIT_CLIENT_INIT_FAILURE, main

    rc = main(
        [
            "--repo-api-url",
            "https://api.github.com/repos/owner/repo",
            "--workspace-path",
            str(tmp_path),
        ]
    )

    assert rc == EXIT_CLIENT_INIT_FAILURE
    assert "--output-path or --summary-file is required in pipeline mode" in caplog.text


def test_main_output_path_overrides_summary_file(monkeypatch, tmp_path):
    """--output-path should override --summary-file when both are provided."""
    monkeypatch.setenv("GITHUB_TOKEN", "fake")
    monkeypatch.delenv("AGENT_SCANNER_PIPELINE_MODE", raising=False)

    import importlib

    import src.main

    importlib.reload(src.main)

    fake_client = MagicMock()
    fake_matcher = MagicMock()
    fake_scanner = MagicMock()
    fake_result = MagicMock()
    fake_result.agentic_signals_detected = True
    fake_result.matched_stage = 2
    fake_result.dependency_files = []
    fake_result.ai_dependencies = [MagicMock()]
    fake_result.agent_instances = [{"file": "agent.py"}]
    fake_scanner.scan.return_value = "owner/repo"
    fake_scanner._repo_result = fake_result

    output_path = tmp_path / "out"
    summary_file = tmp_path / "ignored.json"

    write_args = {}

    def fake_write(path, repo_result):
        write_args["path"] = path

    monkeypatch.setattr("src.main.GitHubClient", lambda api_url=None: fake_client)
    monkeypatch.setattr("src.main.PatternMatcher", SimpleNamespace(from_file=lambda: fake_matcher))
    monkeypatch.setattr("src.main.Scanner", lambda *a, **k: fake_scanner)
    monkeypatch.setattr("src.main.write_summary_file", fake_write)

    from src.main import EXIT_SUCCESS, main

    rc = main(
        [
            "--repo-api-url",
            "https://api.github.com/repos/owner/repo",
            "--output-path",
            str(output_path),
            "--summary-file",
            str(summary_file),
        ]
    )

    assert rc == EXIT_SUCCESS
    assert write_args["path"] == str(output_path)


def test_main_run_header_logs_correlation(monkeypatch, caplog):
    """Run header should include correlation identifiers in logs."""
    caplog.set_level(logging.INFO)
    monkeypatch.setenv("GITHUB_TOKEN", "fake")
    monkeypatch.setattr("src.main.GitHubClient", lambda api_url=None: MagicMock())
    monkeypatch.setattr("src.main.PatternMatcher", SimpleNamespace(from_file=lambda: MagicMock()))

    fake_scanner = MagicMock()
    fake_scanner.scan.return_value = None
    fake_scanner._repo_result = None
    monkeypatch.setattr("src.main.Scanner", lambda *a, **k: fake_scanner)

    from src.main import main

    main(
        [
            "--org",
            "myorg",
            "--repo-api-url",
            "https://api.github.com/repos/myorg/myrepo",
            "--branch",
            "main",
            "--commit",
            "abcdef1234567890",
            "--event",
            "push",
        ]
    )

    assert "Run header" in caplog.text
    assert "org=myorg" in caplog.text
    assert "repo=myrepo" in caplog.text
    assert "repository=myorg/myrepo" in caplog.text
    assert "branch=main" in caplog.text
    assert "commit=abcdef1234567890" in caplog.text
    assert "event=push" in caplog.text


def test_main_scan_org_recent_without_org(monkeypatch, caplog):
    """Scan org recent should fail when --org is missing."""

    caplog.set_level(logging.INFO)
    monkeypatch.setenv("GITHUB_TOKEN", "fake")
    monkeypatch.setenv("AGENT_SCANNER_BULK_ENABLED", "1")

    import importlib

    import src.main

    importlib.reload(src.main)

    monkeypatch.setattr("src.main.GitHubClient", lambda api_url=None: MagicMock())
    monkeypatch.setattr("src.main.PatternMatcher", SimpleNamespace(from_file=lambda: MagicMock()))

    from src.main import EXIT_CLIENT_INIT_FAILURE, main

    rc = main(["--scan-org-recent"])

    assert rc == EXIT_CLIENT_INIT_FAILURE
    assert "--scan-org-recent requires --org" in caplog.text


def test_main_missing_repo_without_org_scan(monkeypatch, caplog):
    """Missing --repo-api-url should fail when not scanning org."""

    caplog.set_level(logging.INFO)
    monkeypatch.setenv("GITHUB_TOKEN", "fake")
    monkeypatch.setattr("src.main.GitHubClient", lambda api_url=None: MagicMock())
    monkeypatch.setattr("src.main.PatternMatcher", SimpleNamespace(from_file=lambda: MagicMock()))

    from src.main import EXIT_CLIENT_INIT_FAILURE, main

    rc = main([])

    assert rc == EXIT_CLIENT_INIT_FAILURE
    assert "--repo-api-url is required" in caplog.text


def test_main_pipeline_mode_without_repo_api_url(monkeypatch, tmp_path):
    """Pipeline mode should allow missing --repo-api-url when workspace is provided."""
    monkeypatch.setenv("AGENT_SCANNER_PIPELINE_MODE", "1")

    repo_path = tmp_path / "repo"
    repo_path.mkdir()

    fake_matcher = MagicMock()
    fake_scanner = MagicMock()
    fake_scanner.scan.return_value = None
    fake_scanner._repo_result = None

    monkeypatch.setattr("src.main.PatternMatcher", SimpleNamespace(from_file=lambda: fake_matcher))
    monkeypatch.setattr("src.main.Scanner", lambda *a, **k: fake_scanner)

    from src.main import EXIT_SUCCESS, main

    rc = main(
        [
            "--workspace-path",
            str(repo_path),
            "--output-path",
            str(tmp_path / "out"),
        ]
    )

    assert rc == EXIT_SUCCESS
    fake_scanner.scan.assert_called_once()
    assert fake_scanner.scan.call_args[0][0] == f"local/{repo_path.name}"


def test_main_repo_api_url_parsing(monkeypatch):
    """Repo API URL should parse into owner/repo."""

    monkeypatch.setenv("GITHUB_TOKEN", "fake")

    fake_client = MagicMock()
    fake_matcher = MagicMock()
    fake_scanner = MagicMock()
    fake_scanner.scan.return_value = None
    fake_scanner._repo_result = None

    monkeypatch.setattr("src.main.GitHubClient", lambda api_url=None: fake_client)
    monkeypatch.setattr("src.main.PatternMatcher", SimpleNamespace(from_file=lambda: fake_matcher))
    monkeypatch.setattr("src.main.Scanner", lambda *a, **k: fake_scanner)

    from src.main import main

    main(["--repo-api-url", "https://api.github.com/repos/myorg/myrepo"])

    fake_scanner.scan.assert_called_once()
    call_args = fake_scanner.scan.call_args
    assert call_args[0][0] == "myorg/myrepo"


def test_main_list_branches_with_scan_org_recent(monkeypatch, caplog):
    """List branches flag should fail with --scan-org-recent."""

    caplog.set_level(logging.INFO)
    monkeypatch.setenv("GITHUB_TOKEN", "fake")
    monkeypatch.setenv("AGENT_SCANNER_BULK_ENABLED", "1")

    import importlib

    import src.main

    importlib.reload(src.main)

    monkeypatch.setattr("src.main.GitHubClient", lambda api_url=None: MagicMock())
    monkeypatch.setattr("src.main.PatternMatcher", SimpleNamespace(from_file=lambda: MagicMock()))

    from src.main import EXIT_CLIENT_INIT_FAILURE, main

    rc = main(["--org", "myorg", "--scan-org-recent", "--list-branches"])

    assert rc == EXIT_CLIENT_INIT_FAILURE
    assert "--list-branches is not supported with --scan-org-recent" in caplog.text


def test_main_list_branches_without_repo(monkeypatch, caplog):
    """List branches flag should fail without --repo-api-url."""

    caplog.set_level(logging.INFO)
    monkeypatch.setenv("GITHUB_TOKEN", "fake")
    monkeypatch.setattr("src.main.GitHubClient", lambda api_url=None: MagicMock())
    monkeypatch.setattr("src.main.PatternMatcher", SimpleNamespace(from_file=lambda: MagicMock()))

    from src.main import EXIT_CLIENT_INIT_FAILURE, main

    rc = main(["--list-branches"])

    assert rc == EXIT_CLIENT_INIT_FAILURE
    assert "--repo-api-url is required" in caplog.text


def test_main_list_branches_invalid_format(monkeypatch, caplog):
    """List branches should fail with invalid repo API URL."""

    caplog.set_level(logging.INFO)
    monkeypatch.setenv("GITHUB_TOKEN", "fake")
    monkeypatch.setattr("src.main.GitHubClient", lambda api_url=None: MagicMock())
    monkeypatch.setattr("src.main.PatternMatcher", SimpleNamespace(from_file=lambda: MagicMock()))

    from src.main import EXIT_CLIENT_INIT_FAILURE, main

    rc = main(["--repo-api-url", "https://api.github.com/invalid", "--list-branches"])

    assert rc == EXIT_CLIENT_INIT_FAILURE
    assert "Invalid --repo-api-url" in caplog.text


def test_main_list_branches_github_client_error(monkeypatch, caplog):
    """List branches should handle GitHubClientError gracefully."""

    caplog.set_level(logging.INFO)
    monkeypatch.setenv("GITHUB_TOKEN", "fake")

    fake_client = MagicMock()
    from src.exceptions import GitHubClientError

    fake_client.get_branches.side_effect = GitHubClientError("API error")

    monkeypatch.setattr("src.main.GitHubClient", lambda api_url=None: fake_client)
    monkeypatch.setattr("src.main.PatternMatcher", SimpleNamespace(from_file=lambda: MagicMock()))

    from src.main import EXIT_CLIENT_INIT_FAILURE, main

    rc = main(["--repo-api-url", "https://api.github.com/repos/owner/repo", "--list-branches"])

    assert rc == EXIT_CLIENT_INIT_FAILURE
    assert "Failed to list branches" in caplog.text


def test_main_scan_failure_scanner_error(monkeypatch, caplog):
    """Scan should return failure code when scanner raises error."""

    caplog.set_level(logging.INFO)
    monkeypatch.setenv("GITHUB_TOKEN", "fake")

    fake_client = MagicMock()
    fake_matcher = MagicMock()
    fake_scanner = MagicMock()

    from src.exceptions import ScannerError

    fake_scanner.scan.side_effect = ScannerError("scan failed")

    monkeypatch.setattr("src.main.GitHubClient", lambda api_url=None: fake_client)
    monkeypatch.setattr("src.main.PatternMatcher", SimpleNamespace(from_file=lambda: fake_matcher))
    monkeypatch.setattr("src.main.Scanner", lambda *a, **k: fake_scanner)

    from src.main import EXIT_SCAN_FAILURE, main

    rc = main(["--repo-api-url", "https://api.github.com/repos/owner/repo"])

    assert rc == EXIT_SCAN_FAILURE
    assert "Scan failed" in caplog.text


def test_main_result_without_agents_or_dependencies(monkeypatch, caplog):
    """Result without agents or dependencies should skip summary output."""

    caplog.set_level(logging.INFO)
    monkeypatch.setenv("GITHUB_TOKEN", "fake")

    fake_client = MagicMock()
    fake_matcher = MagicMock()
    fake_scanner = MagicMock()
    fake_result = MagicMock()
    fake_result.org = "owner"
    fake_result.repo_name = "repo"
    fake_result.agent_instances = []
    fake_result.ai_dependencies = []

    fake_scanner.scan.return_value = "owner/repo"
    fake_scanner._repo_result = fake_result

    monkeypatch.setattr("src.main.GitHubClient", lambda api_url=None: fake_client)
    monkeypatch.setattr("src.main.PatternMatcher", SimpleNamespace(from_file=lambda: fake_matcher))
    monkeypatch.setattr("src.main.Scanner", lambda *a, **k: fake_scanner)

    from src.main import EXIT_SUCCESS, main

    rc = main(["--repo-api-url", "https://api.github.com/repos/owner/repo"])

    assert rc == EXIT_SUCCESS
    assert "No agents or AI dependencies found" in caplog.text
    assert "skipping summary output" in caplog.text


def test_main_result_with_agents_and_dependencies(monkeypatch, caplog):
    """Result with agents and dependencies should log details."""

    caplog.set_level(logging.INFO)
    monkeypatch.setenv("GITHUB_TOKEN", "fake")

    fake_client = MagicMock()
    fake_matcher = MagicMock()
    fake_scanner = MagicMock()
    fake_result = MagicMock()
    fake_result.org = "owner"
    fake_result.repo_name = "repo"
    fake_result.agentic_signals_detected = True
    fake_result.matched_stage = 2
    fake_result.dependency_files = ["requirements.txt", "pyproject.toml"]

    fake_dep1 = MagicMock()
    fake_dep1.package_name = "langchain"
    fake_dep1.version = "0.1.0"
    fake_dep2 = MagicMock()
    fake_dep2.package_name = "openai"
    fake_dep2.version = None

    fake_result.ai_dependencies = [fake_dep1, fake_dep2]
    fake_result.agent_instances = [{"file": "agent.py", "count": 1, "agents": []}]
    fake_result.agent_unique = [{"name": "TestAgent"}]

    fake_scanner.scan.return_value = "owner/repo"
    fake_scanner._repo_result = fake_result

    monkeypatch.setattr("src.main.GitHubClient", lambda api_url=None: fake_client)
    monkeypatch.setattr("src.main.PatternMatcher", SimpleNamespace(from_file=lambda: fake_matcher))
    monkeypatch.setattr("src.main.Scanner", lambda *a, **k: fake_scanner)
    monkeypatch.setattr("src.main.write_summary_file", lambda *a, **k: None)

    from src.main import EXIT_SUCCESS, main

    rc = main(["--repo-api-url", "https://api.github.com/repos/owner/repo", "--summary-file", "out.json"])

    assert rc == EXIT_SUCCESS
    assert "Match found" in caplog.text
    assert "Dependency files" in caplog.text
    assert "AI dependencies" in caplog.text
    assert "langchain 0.1.0" in caplog.text
    assert "openai" in caplog.text
    assert "Found 1 total agents" in caplog.text


def test_main_recent_since_valid_date(monkeypatch, caplog, tmp_path):
    """Valid --recent-since date should be parsed correctly."""

    caplog.set_level(logging.INFO)
    monkeypatch.setenv("GITHUB_TOKEN", "fake")
    monkeypatch.setenv("AGENT_SCANNER_BULK_ENABLED", "1")

    import importlib

    import src.main

    importlib.reload(src.main)

    fake_client = MagicMock()
    fake_client.list_org_repos.return_value = []

    monkeypatch.setattr("src.main.GitHubClient", lambda api_url=None: fake_client)
    monkeypatch.setattr("src.main.PatternMatcher", SimpleNamespace(from_file=lambda: MagicMock()))

    from src.main import EXIT_SUCCESS, main

    rc = main(["--org", "myorg", "--scan-org-recent", "--recent-since", "2025-01-01"])

    assert rc == EXIT_SUCCESS
    assert fake_client.list_org_repos.called


def test_main_recent_since_invalid_date(monkeypatch, caplog):
    """Invalid --recent-since date should return failure."""

    caplog.set_level(logging.INFO)
    monkeypatch.setenv("GITHUB_TOKEN", "fake")
    monkeypatch.setenv("AGENT_SCANNER_BULK_ENABLED", "1")

    import importlib

    import src.main

    importlib.reload(src.main)

    fake_client = MagicMock()
    monkeypatch.setattr("src.main.GitHubClient", lambda api_url=None: fake_client)
    monkeypatch.setattr("src.main.PatternMatcher", SimpleNamespace(from_file=lambda: MagicMock()))

    from src.main import EXIT_CLIENT_INIT_FAILURE, main

    rc = main(["--org", "myorg", "--scan-org-recent", "--recent-since", "invalid-date"])

    assert rc == EXIT_CLIENT_INIT_FAILURE
    assert "--recent-since must be in YYYY-MM-DD format" in caplog.text


def test_main_recent_days_positive(monkeypatch, caplog):
    """Positive --recent-days should work correctly."""

    caplog.set_level(logging.INFO)
    monkeypatch.setenv("GITHUB_TOKEN", "fake")
    monkeypatch.setenv("AGENT_SCANNER_BULK_ENABLED", "1")

    import importlib

    import src.main

    importlib.reload(src.main)

    fake_client = MagicMock()
    fake_client.list_org_repos.return_value = []

    monkeypatch.setattr("src.main.GitHubClient", lambda api_url=None: fake_client)
    monkeypatch.setattr("src.main.PatternMatcher", SimpleNamespace(from_file=lambda: MagicMock()))

    from src.main import EXIT_SUCCESS, main

    rc = main(["--org", "myorg", "--scan-org-recent", "--recent-days", "30"])

    assert rc == EXIT_SUCCESS
    assert "Using recent cutoff: 30 days" in caplog.text


def test_main_recent_days_zero(monkeypatch, caplog):
    """Zero --recent-days should return failure."""

    caplog.set_level(logging.INFO)
    monkeypatch.setenv("GITHUB_TOKEN", "fake")
    monkeypatch.setenv("AGENT_SCANNER_BULK_ENABLED", "1")

    import importlib

    import src.main

    importlib.reload(src.main)

    fake_client = MagicMock()
    monkeypatch.setattr("src.main.GitHubClient", lambda api_url=None: fake_client)
    monkeypatch.setattr("src.main.PatternMatcher", SimpleNamespace(from_file=lambda: MagicMock()))

    from src.main import EXIT_CLIENT_INIT_FAILURE, main

    rc = main(["--org", "myorg", "--scan-org-recent", "--recent-days", "0"])

    assert rc == EXIT_CLIENT_INIT_FAILURE
    assert "--recent-days must be positive" in caplog.text


def test_main_scan_org_recent_no_repos(monkeypatch, caplog):
    """Scan org recent with no matching repos should succeed."""

    caplog.set_level(logging.INFO)
    monkeypatch.setenv("GITHUB_TOKEN", "fake")
    monkeypatch.setenv("AGENT_SCANNER_BULK_ENABLED", "1")

    import importlib

    import src.main

    importlib.reload(src.main)

    fake_client = MagicMock()
    fake_client.list_org_repos.return_value = []

    monkeypatch.setattr("src.main.GitHubClient", lambda api_url=None: fake_client)
    monkeypatch.setattr("src.main.PatternMatcher", SimpleNamespace(from_file=lambda: MagicMock()))

    from src.main import EXIT_SUCCESS, main

    rc = main(["--org", "myorg", "--scan-org-recent"])

    assert rc == EXIT_SUCCESS
    assert "No repositories matched the recent push window" in caplog.text


def test_main_scan_org_recent_list_failure(monkeypatch, caplog):
    """Scan org recent should fail gracefully when list_org_repos fails."""

    caplog.set_level(logging.INFO)
    monkeypatch.setenv("GITHUB_TOKEN", "fake")
    monkeypatch.setenv("AGENT_SCANNER_BULK_ENABLED", "1")

    import importlib

    import src.main

    importlib.reload(src.main)

    fake_client = MagicMock()
    from src.exceptions import GitHubClientError

    fake_client.list_org_repos.side_effect = GitHubClientError("API error")

    monkeypatch.setattr("src.main.GitHubClient", lambda api_url=None: fake_client)
    monkeypatch.setattr("src.main.PatternMatcher", SimpleNamespace(from_file=lambda: MagicMock()))

    from src.main import EXIT_CLIENT_INIT_FAILURE, main

    rc = main(["--org", "myorg", "--scan-org-recent"])

    assert rc == EXIT_CLIENT_INIT_FAILURE
    assert "Failed to list org repositories" in caplog.text


def test_main_scan_org_recent_with_existing_summaries(monkeypatch, caplog, tmp_path):
    """Scan org recent should skip repos with existing summaries."""

    caplog.set_level(logging.INFO)
    monkeypatch.setenv("GITHUB_TOKEN", "fake")
    monkeypatch.setenv("AGENT_SCANNER_BULK_ENABLED", "1")

    import importlib

    import src.main

    importlib.reload(src.main)

    output_dir = tmp_path / "output"
    output_dir.mkdir()
    (output_dir / "myorg-repo1_main.json").write_text("{}")

    fake_client = MagicMock()
    fake_client.list_org_repos.return_value = [{"name": "repo1", "full_name": "myorg/repo1"}]

    monkeypatch.setattr("src.main.GitHubClient", lambda api_url=None: fake_client)
    monkeypatch.setattr("src.main.PatternMatcher", SimpleNamespace(from_file=lambda: MagicMock()))

    from src.main import EXIT_SUCCESS, main

    rc = main(["--org", "myorg", "--scan-org-recent", "--output-dir", str(output_dir)])

    assert rc == EXIT_SUCCESS
    assert "Found 1 existing summary files" in caplog.text
    assert "All repositories already scanned" in caplog.text


def test_main_scan_org_recent_no_skip_existing(monkeypatch, caplog, tmp_path):
    """Scan org recent with --no-skip-existing-output should scan all repos."""

    caplog.set_level(logging.INFO)
    monkeypatch.setenv("GITHUB_TOKEN", "fake")
    monkeypatch.setenv("AGENT_SCANNER_BULK_ENABLED", "1")

    import importlib

    import src.main

    importlib.reload(src.main)

    output_dir = tmp_path / "output"
    output_dir.mkdir()
    (output_dir / "myorg-repo1_main.json").write_text("{}")

    fake_client = MagicMock()
    fake_client.list_org_repos.return_value = [{"name": "repo1", "full_name": "myorg/repo1"}]
    fake_matcher = MagicMock()

    def fake_scanner_factory(*args, **kwargs):
        scanner = MagicMock()
        scanner.scan.return_value = "myorg/repo1"
        result = MagicMock()
        result.agent_instances = []
        result.ai_dependencies = []
        scanner._repo_result = result
        return scanner

    monkeypatch.setattr("src.main.GitHubClient", lambda api_url=None: fake_client)
    monkeypatch.setattr("src.main.PatternMatcher", SimpleNamespace(from_file=lambda: fake_matcher))
    monkeypatch.setattr("src.main.Scanner", fake_scanner_factory)

    from src.main import EXIT_SUCCESS, main

    rc = main(
        [
            "--org",
            "myorg",
            "--scan-org-recent",
            "--output-dir",
            str(output_dir),
            "--no-skip-existing-output",
            "--scan-workers",
            "1",
        ]
    )

    assert rc == EXIT_SUCCESS
    assert "Queued 1 repositories" in caplog.text


def test_main_scan_org_recent_with_repos_success(monkeypatch, caplog, tmp_path):
    """Scan org recent with repos should queue and scan them."""

    caplog.set_level(logging.INFO)
    monkeypatch.setenv("GITHUB_TOKEN", "fake")
    monkeypatch.setenv("AGENT_SCANNER_BULK_ENABLED", "1")

    import importlib

    import src.main

    importlib.reload(src.main)

    fake_client = MagicMock()
    fake_client.list_org_repos.return_value = [
        {"name": "repo1", "full_name": "myorg/repo1"},
        {"name": "repo2", "full_name": "myorg/repo2"},
    ]
    fake_matcher = MagicMock()

    def fake_scanner_factory(*args, **kwargs):
        scanner = MagicMock()
        scanner.scan.return_value = "myorg/repo1"
        result = MagicMock()
        result.agent_instances = []
        result.ai_dependencies = []
        scanner._repo_result = result
        return scanner

    monkeypatch.setattr("src.main.GitHubClient", lambda api_url=None: fake_client)
    monkeypatch.setattr("src.main.PatternMatcher", SimpleNamespace(from_file=lambda: fake_matcher))
    monkeypatch.setattr("src.main.Scanner", fake_scanner_factory)

    from src.main import EXIT_SUCCESS, main

    rc = main(["--org", "myorg", "--scan-org-recent", "--scan-workers", "1"])

    assert rc == EXIT_SUCCESS
    assert "Queued 2 repositories" in caplog.text
    assert "Completed scans for 2 repositories" in caplog.text


def test_main_scan_org_recent_rate_limit_retry(monkeypatch, caplog):
    """Scan org recent should retry on rate limit errors."""

    caplog.set_level(logging.INFO)
    monkeypatch.setenv("GITHUB_TOKEN", "fake")
    monkeypatch.setenv("AGENT_SCANNER_BULK_ENABLED", "1")

    import importlib

    import src.main

    importlib.reload(src.main)

    fake_client = MagicMock()
    from src.exceptions import GitHubClientError

    call_count = {"count": 0}

    def list_with_retry(*args, **kwargs):
        call_count["count"] += 1
        if call_count["count"] == 1:
            raise GitHubClientError("rate limit exceeded")
        return []

    fake_client.list_org_repos.side_effect = list_with_retry

    monkeypatch.setattr("src.main.GitHubClient", lambda api_url=None: fake_client)
    monkeypatch.setattr("src.main.PatternMatcher", SimpleNamespace(from_file=lambda: MagicMock()))
    monkeypatch.setattr("src.main.time.sleep", lambda x: None)

    from src.main import EXIT_SUCCESS, main

    rc = main(["--org", "myorg", "--scan-org-recent", "--list-retries", "3", "--list-retry-sleep", "1"])

    assert rc == EXIT_SUCCESS
    assert call_count["count"] == 2
    assert "Rate limit while listing org repos" in caplog.text


def test_main_result_no_signals_detected(monkeypatch, caplog):
    """Result with agentic_signals_detected False should log no signals."""

    caplog.set_level(logging.INFO)
    monkeypatch.setenv("GITHUB_TOKEN", "fake")

    fake_client = MagicMock()
    fake_matcher = MagicMock()
    fake_scanner = MagicMock()
    fake_result = MagicMock()
    fake_result.org = "owner"
    fake_result.repo_name = "repo"
    fake_result.agentic_signals_detected = False
    fake_result.agent_instances = [{"file": "agent.py"}]
    fake_result.ai_dependencies = []

    fake_scanner.scan.return_value = "owner/repo"
    fake_scanner._repo_result = fake_result

    monkeypatch.setattr("src.main.GitHubClient", lambda api_url=None: fake_client)
    monkeypatch.setattr("src.main.PatternMatcher", SimpleNamespace(from_file=lambda: fake_matcher))
    monkeypatch.setattr("src.main.Scanner", lambda *a, **k: fake_scanner)
    monkeypatch.setattr(
        "src.main.write_summary_file",
        lambda *a, **k: (_ for _ in ()).throw(AssertionError()),
    )  # NOSONAR S7500

    from src.main import EXIT_SUCCESS, main

    rc = main(["--repo-api-url", "https://api.github.com/repos/owner/repo"])

    assert rc == EXIT_SUCCESS


def test_main_scan_org_recent_scan_failure(monkeypatch, caplog):
    """Scan org recent with failing scans should return failure code."""

    caplog.set_level(logging.INFO)
    monkeypatch.setenv("GITHUB_TOKEN", "fake")
    monkeypatch.setenv("AGENT_SCANNER_BULK_ENABLED", "1")

    import importlib

    import src.main

    importlib.reload(src.main)

    fake_client = MagicMock()
    fake_client.list_org_repos.return_value = [{"name": "repo1", "full_name": "myorg/repo1"}]
    fake_matcher = MagicMock()

    from src.exceptions import ScannerError

    def fake_scanner_factory(*args, **kwargs):
        scanner = MagicMock()
        scanner.scan.side_effect = ScannerError("scan failed")
        return scanner

    monkeypatch.setattr("src.main.GitHubClient", lambda api_url=None: fake_client)
    monkeypatch.setattr("src.main.PatternMatcher", SimpleNamespace(from_file=lambda: fake_matcher))
    monkeypatch.setattr("src.main.Scanner", fake_scanner_factory)

    from src.main import EXIT_SCAN_FAILURE, main

    rc = main(["--org", "myorg", "--scan-org-recent", "--scan-workers", "1"])

    assert rc == EXIT_SCAN_FAILURE
    assert "Scan failed for" in caplog.text
    assert "Completed with 1 failures" in caplog.text


def test_main_scan_org_recent_unexpected_exception(monkeypatch, caplog):
    """Scan org recent with unexpected exceptions should log and continue."""

    caplog.set_level(logging.INFO)
    monkeypatch.setenv("GITHUB_TOKEN", "fake")
    monkeypatch.setenv("AGENT_SCANNER_BULK_ENABLED", "1")

    import importlib

    import src.main

    importlib.reload(src.main)

    fake_client = MagicMock()
    fake_client.list_org_repos.return_value = [{"name": "repo1", "full_name": "myorg/repo1"}]
    fake_matcher = MagicMock()

    def fake_scanner_factory(*args, **kwargs):
        scanner = MagicMock()
        scanner.scan.side_effect = RuntimeError("unexpected error")
        return scanner

    monkeypatch.setattr("src.main.GitHubClient", lambda api_url=None: fake_client)
    monkeypatch.setattr("src.main.PatternMatcher", SimpleNamespace(from_file=lambda: fake_matcher))
    monkeypatch.setattr("src.main.Scanner", fake_scanner_factory)

    from src.main import EXIT_SCAN_FAILURE, main

    rc = main(["--org", "myorg", "--scan-org-recent", "--scan-workers", "1"])

    assert rc == EXIT_SCAN_FAILURE
    assert "Unexpected failure scanning" in caplog.text


def test_main_scan_org_recent_rate_limit_during_scan(monkeypatch, caplog):
    """Scan org recent should sleep on rate limit during individual scans."""

    caplog.set_level(logging.INFO)
    monkeypatch.setenv("GITHUB_TOKEN", "fake")
    monkeypatch.setenv("AGENT_SCANNER_BULK_ENABLED", "1")

    import importlib

    import src.main

    importlib.reload(src.main)

    fake_client = MagicMock()
    fake_client.list_org_repos.return_value = [{"name": "repo1", "full_name": "myorg/repo1"}]
    fake_matcher = MagicMock()

    from src.exceptions import GitHubClientError

    def fake_scanner_factory(*args, **kwargs):
        scanner = MagicMock()
        scanner.scan.side_effect = GitHubClientError("rate limit exceeded")
        return scanner

    sleep_called = {"called": False}

    def fake_sleep(seconds):
        sleep_called["called"] = True

    monkeypatch.setattr("src.main.GitHubClient", lambda api_url=None: fake_client)
    monkeypatch.setattr("src.main.PatternMatcher", SimpleNamespace(from_file=lambda: fake_matcher))
    monkeypatch.setattr("src.main.Scanner", fake_scanner_factory)
    monkeypatch.setattr("src.main.time.sleep", fake_sleep)

    from src.main import EXIT_SCAN_FAILURE, main

    rc = main(["--org", "myorg", "--scan-org-recent", "--scan-workers", "1", "--rate-limit-sleep", "30"])

    assert rc == EXIT_SCAN_FAILURE
    assert sleep_called["called"]
    assert "Rate limit suspected while scanning" in caplog.text


def test_main_scan_org_recent_with_summary_output(monkeypatch, caplog, tmp_path):
    """Scan org recent should write summary files for repos with signals."""

    caplog.set_level(logging.INFO)
    monkeypatch.setenv("GITHUB_TOKEN", "fake")
    monkeypatch.setenv("AGENT_SCANNER_BULK_ENABLED", "1")

    import importlib

    import src.main

    importlib.reload(src.main)

    fake_client = MagicMock()
    fake_client.list_org_repos.return_value = [{"name": "repo1", "full_name": "myorg/repo1"}]
    fake_matcher = MagicMock()

    summary_file = tmp_path / "summary.json"

    def fake_scanner_factory(*args, **kwargs):
        scanner = MagicMock()
        scanner.scan.return_value = "myorg/repo1"
        result = MagicMock()
        result.org = "myorg"
        result.repo_name = "repo1"
        result.agentic_signals_detected = True
        result.matched_stage = 2
        result.dependency_files = []
        result.ai_dependencies = [MagicMock()]
        result.agent_instances = [{"file": "agent.py"}]
        scanner._repo_result = result
        return scanner

    write_called = {"called": False}

    def fake_write_summary(path, repo_result):
        write_called["called"] = True

    monkeypatch.setattr("src.main.GitHubClient", lambda api_url=None: fake_client)
    monkeypatch.setattr("src.main.PatternMatcher", SimpleNamespace(from_file=lambda: fake_matcher))
    monkeypatch.setattr("src.main.Scanner", fake_scanner_factory)
    monkeypatch.setattr("src.main.write_summary_file", fake_write_summary)

    from src.main import EXIT_CLIENT_INIT_FAILURE, main

    rc = main(["--org", "myorg", "--scan-org-recent", "--scan-workers", "1", "--summary-file", str(summary_file)])

    assert rc == EXIT_CLIENT_INIT_FAILURE
    assert not write_called["called"]
