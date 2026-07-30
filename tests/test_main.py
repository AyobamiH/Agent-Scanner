"""Tests for main CLI entry point and command execution."""

import pytest


def test_main_exits_when_github_token_missing(monkeypatch):  # NOSONAR S2325
    """Running the CLI without a token exits with SystemExit."""
    monkeypatch.delenv("GITHUB_TOKEN", raising=False)
    with pytest.raises(SystemExit):
        import runpy
        import sys

        sys.modules.pop("src.main", None)
        sys.modules.pop("src", None)
        runpy.run_module("src.main", run_name="__main__")


def test_main_returns_zero_on_successful_run(monkeypatch):  # NOSONAR S2325
    """CLI returns 0 when GitHubClient initialisation succeeds and no match is found."""
    monkeypatch.setenv("GITHUB_TOKEN", "fake")
    from src.main import main

    class Dummy:
        def __init__(self, *a, **k):  # NOSONAR S1186 - intentionally empty mock for testing
            # Mock class for testing; intentionally accepts any arguments without processing
            pass

        def get_repo_tree(self, owner, repo, branch=None):
            metadata = {
                "default_branch": "main",
                "head_sha": "abc123",
                "html_url": f"https://github.com/{owner}/{repo}",
            }
            return ([], metadata)

    monkeypatch.setattr("src.main.GitHubClient", Dummy)

    rc = main(["--repo-api-url", "https://api.github.com/repos/owner/repo"])
    assert rc == 0
