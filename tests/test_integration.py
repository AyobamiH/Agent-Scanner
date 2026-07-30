"""Integration tests for Scanner with multiple components."""

from src.scanner.scanner import Scanner


def make_client(tree, contents=None):  # NOSONAR S2325
    class C:
        api_url = "https://api.github.com"
        max_workers = 2
        max_file_size = 1000000

        def get_repo_tree(self, owner, repo, branch=None):
            metadata = {
                "default_branch": "main",
                "head_sha": "abc123",
                "html_url": f"https://github.com/{owner}/{repo}",
            }
            return (tree, metadata)

        def get_file_content(self, owner, repo, path, branch=None):
            return (contents or {}).get(path, "")

    return C()


def test_integration_stage1_reports_on_path_token(pattern_matcher):  # NOSONAR S2325
    """Stage 1 reports when path contains a configured path token."""
    tree = [{"path": "agents/agent.py", "type": "blob"}]

    client = make_client(tree, {"agents/agent.py": "class MyAgent:\n    pass\n"})
    scanner = Scanner(client, pattern_matcher)
    assert scanner.scan("o/r") == "o/r"


def test_integration_stage2_reports_on_content_signals(pattern_matcher):  # NOSONAR S2325
    """Stage 2 aggregates content signals across sampled files."""
    tree = [{"path": "a.py", "type": "blob"}, {"path": "b/c.py", "type": "blob"}]
    contents = {"b/c.py": "openai langchain llm"}

    contents["a.py"] = "class WorkerAgent:\n    pass\n"
    client = make_client(tree, contents)
    scanner = Scanner(client, pattern_matcher)

    assert scanner.scan("owner/repo") == "owner/repo"
