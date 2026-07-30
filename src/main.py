"""CLI entry point for the agent scanner.

Provides command-line interface for scanning GitHub repositories (or local workspaces)
for agentic patterns. Supports:
    - Single repository scanning via repo API URL or local path
    - Bulk organisation scanning (optional, when ORG_BULK_ENABLED)
    - Output to JSON summary files with schema validation
    - Correlation tracking via scan IDs and commit hashes
    - Configurable logging and timeout behavior

Exit Codes:
    0 (EXIT_SUCCESS): Scan completed successfully
    1 (EXIT_CLIENT_INIT_FAILURE): Failed to initialise GitHub/file client
    2 (EXIT_SCAN_FAILURE): Scan operation failed (fail-fast mode)

Environment Variables:
    See CorrelationFilter and Scanner classes for configuration options.
"""

from __future__ import annotations

import argparse
import logging
import os
import time
import uuid
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any, cast
from urllib.parse import urlparse

from src.detectors.patterns import PatternMatcher
from src.exceptions import GitHubClientError, ScannerError, SummaryWriteError
from src.github.client import GitHubClient
from src.github.filesystem_client import LocalFilesystemClient
from src.models.results import RepoScanResult
from src.scanner.scanner import Scanner
from src.utils.summary_writer import write_summary_file

EXIT_SUCCESS = 0
EXIT_CLIENT_INIT_FAILURE = 1
EXIT_SCAN_FAILURE = 2

logger = logging.getLogger(__name__)
ORG_BULK_ENABLED = os.getenv("AGENT_SCANNER_BULK_ENABLED", "1") != "0"
PIPELINE_MODE = os.getenv("AGENT_SCANNER_PIPELINE_MODE", "0") == "1"
DEFAULT_SCHEMA_PATH = "scanner-payload.schema.json"


class CorrelationFilter(logging.Filter):
    """Inject correlation identifiers into log records for distributed tracing.

    Adds contextual metadata fields to all log records:
        - org: Repository organisation/owner
        - repo: Repository name
        - branch: Git branch being scanned
        - commit: Git commit hash
        - event: Event name (e.g., "push", "schedule")
        - scan_id: Unique scan identifier (generated during scan)

    This enables correlation of log messages across pipeline runs and
    makes it easy to filter logs by repository or scan ID in aggregation systems.

    Example log output: org=MyOrg repo=MyRepo scan_id=abc-123 Scan started...
    """

    def __init__(
        self,
        organisation: str | None,
        repository: str | None,
        branch: str | None,
        commit: str | None,
        event: str | None,
        scan_id: str | None = None,
    ) -> None:
        super().__init__()
        self.organisation = organisation or ""
        self.repository = repository or ""
        self.branch = branch or ""
        self.commit = commit or ""
        self.event = event or ""
        self.scan_id = scan_id or ""

    def filter(self, record: logging.LogRecord) -> bool:
        record.org = self.organisation
        record.repo = self.repository
        record.branch = self.branch
        record.commit = self.commit
        record.event = self.event
        record.scan_id = self.scan_id
        return True

    def set_scan_id(self, scan_id: str) -> None:
        """Update the scan_id after it's generated."""
        self.scan_id = scan_id


def _resolve_log_level(log_level: str | None, verbose: bool) -> int:
    """Resolve the logging level for the CLI run.

    Precedence: explicit --log-level > --verbose flag > default INFO level

    Args:
        log_level: Optional explicit log level (debug/warning/error/info).
        verbose: Whether the --verbose flag is enabled (implies DEBUG level).

    Returns:
        Logging level constant (logging.DEBUG, logging.INFO, etc).
    """
    if log_level:
        resolved = log_level.lower()
        if resolved == "debug":
            return logging.DEBUG
        if resolved == "warning":
            return logging.WARNING
        if resolved == "error":
            return logging.ERROR
        return logging.INFO
    if verbose:
        return logging.DEBUG
    return logging.INFO


def _parse_repo_api_url(repo_api_url: str) -> tuple[str, str]:
    """Parse a repository API URL into base API URL and owner/repo identifier.

    Handles both GitHub.com and GitHub Enterprise API URL formats:

    GitHub.com examples:
        - https://api.github.com/repos/microsoft/vscode
        - https://api.github.com/repos/microsoft/vscode/issues

    GitHub Enterprise examples:
        - https://github.mycompany.com/api/v3/repos/owner/repo
        - https://github.mycompany.com/api/v3/owner/repo

    Args:
        repo_api_url: Full repository API URL from GitHub or GitHub Enterprise.

    Returns:
        Tuple of (base_api_url, "owner/repo"):
            - base_api_url: e.g., "https://api.github.com" or "https://github.com/api/v3"
            - owner/repo: e.g., "microsoft/vscode"

    Raises:
        ValueError: If the URL is invalid or missing required components (owner/repo).

    """

    parsed = urlparse(repo_api_url)
    if not parsed.scheme or not parsed.netloc:
        raise ValueError("repo_api_url must include scheme and host")

    path = (parsed.path or "").rstrip("/")

    marker = "/repos/"
    if marker in path:
        prefix, suffix = path.split(marker, 1)
        parts = [segment for segment in suffix.split("/") if segment]
        if len(parts) < 2:
            raise ValueError("repo_api_url must include owner and repo")
        owner, repo = parts[0], parts[1]
        base_url = f"{parsed.scheme}://{parsed.netloc}{prefix}".rstrip("/")
    else:
        parts = [segment for segment in path.split("/") if segment]
        if len(parts) < 2:
            raise ValueError("repo_api_url must include owner and repo (e.g., https://host/api/v3/owner/repo)")
        owner, repo = parts[-2], parts[-1]
        base_url = f"{parsed.scheme}://{parsed.netloc}/{'/'.join(parts[:-2])}".rstrip("/")

    return base_url, f"{owner}/{repo}"


def _handle_list_branches(
    parsed_args: argparse.Namespace,
    repository_full_name: str | None,
    github_client: Any | None,
    pipeline_mode: bool,
) -> int | None:
    """Handle --list-branches option if requested.

    Returns:
        EXIT_SUCCESS if branches listed, None if not requested, EXIT_* if error.
    """
    if not parsed_args.list_branches:
        return None

    if pipeline_mode:
        logger.error("--list-branches is not supported in pipeline mode")
        return EXIT_CLIENT_INIT_FAILURE
    if github_client is None:
        logger.error("Cannot list branches: GitHub client initialization failed")
        return EXIT_CLIENT_INIT_FAILURE
    if parsed_args.scan_org_recent:
        logger.error("--list-branches is not supported with --scan-org-recent")
        return EXIT_CLIENT_INIT_FAILURE
    if not repository_full_name:
        logger.error("--list-branches requires --repo-api-url")
        return EXIT_CLIENT_INIT_FAILURE

    try:
        owner, repository = repository_full_name.split("/", 1)
        branches = github_client.get_branches(owner, repository)
        for branch in branches:
            logger.info("Branch available: %s", branch)
        return EXIT_SUCCESS
    except ValueError:
        logger.error("Invalid repo format extracted from --repo-api-url")
        return EXIT_CLIENT_INIT_FAILURE
    except GitHubClientError as exc:
        logger.exception("Failed to list branches: %s", exc)
        return EXIT_CLIENT_INIT_FAILURE


def _initialise_github_clients(
    parsed_args: argparse.Namespace,
    pipeline_mode: bool,
    resolved_base_url: str,
) -> tuple[Any | None, Any | None, int | None]:
    """initialise GitHub and file source clients based on configuration.

    Returns:
        (github_client, file_source_client, error_code or None)
    """

    class _WorkspaceStubClient:
        def __init__(self, max_file_size: int = 1_000_000) -> None:
            self.max_file_size = max_file_size
            self.api_stats: dict[str, int] = {}

        @staticmethod
        def fetch_all_commits(*_args: Any, **_kwargs: Any) -> None:
            raise GitHubClientError("GitHub token missing; owner detection disabled")

        @staticmethod
        def get_branches(*_args: Any, **_kwargs: Any) -> None:
            raise GitHubClientError("get_branches not available in workspace stub")

        @staticmethod
        def list_org_repos(*_args: Any, **_kwargs: Any) -> None:
            raise GitHubClientError("list_org_repos not available in workspace stub")

    github_client: Any | None = None
    file_source_client = None

    if pipeline_mode and parsed_args.workspace_path:
        workspace_path_obj = Path(parsed_args.workspace_path)
        if not workspace_path_obj.exists():
            logger.error("Workspace path does not exist: %s", parsed_args.workspace_path)
            return None, None, EXIT_CLIENT_INIT_FAILURE
        if not workspace_path_obj.is_dir():
            logger.error("Workspace path is not a directory: %s", parsed_args.workspace_path)
            return None, None, EXIT_CLIENT_INIT_FAILURE
        logger.info("Pipeline mode: using local git history for owner detection")
        logger.info("Using local filesystem scanning mode with workspace: %s", parsed_args.workspace_path)
        local_client = LocalFilesystemClient(
            str(workspace_path_obj),
            None,
            api_url=resolved_base_url,
        )
        github_client = local_client
        file_source_client = local_client
    else:
        try:
            github_client = GitHubClient(api_url=resolved_base_url)
        except (GitHubClientError, ValueError) as exc:
            if parsed_args.workspace_path and "GITHUB_TOKEN" in str(exc):
                logger.warning("GitHub token missing; disabling owner detection for workspace scan")
                os.environ["AGENT_SCANNER_OWNER_DETECTION_ENABLED"] = "0"
                github_client = _WorkspaceStubClient()
            else:
                logger.exception("Failed to initialise GitHub client: %s", exc)
                return None, None, EXIT_CLIENT_INIT_FAILURE

    return github_client, file_source_client, None


def _load_pattern_matcher() -> tuple[Any | None, int | None]:
    """Load pattern matcher from configuration file.

    Returns:
        (matcher, error_code or None)
    """
    try:
        matcher = PatternMatcher.from_file()
        return matcher, None
    except (OSError, ValueError) as exc:
        logger.exception("Failed to load pattern matcher: %s", exc)
        return None, EXIT_CLIENT_INIT_FAILURE


def _initialise_workspace_client(
    parsed_args: argparse.Namespace,
    github_client: Any | None,
    file_source_client: Any | None,
    resolved_base_url: str,
) -> tuple[Any | None, int | None]:
    """initialise workspace file source client if needed.

    Returns:
        (file_source_client, error_code or None)
    """
    if parsed_args.workspace_path and file_source_client is None:
        workspace_path_obj = Path(parsed_args.workspace_path)
        if not workspace_path_obj.exists():
            logger.error("Workspace path does not exist: %s", parsed_args.workspace_path)
            return None, EXIT_CLIENT_INIT_FAILURE
        if not workspace_path_obj.is_dir():
            logger.error("Workspace path is not a directory: %s", parsed_args.workspace_path)
            return None, EXIT_CLIENT_INIT_FAILURE
        logger.info("Using local filesystem scanning mode with workspace: %s", parsed_args.workspace_path)
        file_source_client = LocalFilesystemClient(
            str(workspace_path_obj),
            github_client,
            api_url=resolved_base_url,
        )

    return file_source_client, None


def _setup_logging_handlers(
    parsed_args: argparse.Namespace,
    log_level: int,
    correlation_filter: CorrelationFilter,
) -> None:
    """Configure logging handlers with correlation filter.

    Args:
        parsed_args: Parsed command-line arguments.
        log_level: Log level to use.
        correlation_filter: Filter to add correlation context to logs.
    """
    handlers: list[logging.Handler] = []
    if parsed_args.log_file:
        handlers.append(logging.FileHandler(parsed_args.log_file))
    else:
        handlers.append(logging.StreamHandler())

    for handler in handlers:
        handler.addFilter(correlation_filter)

    logging.basicConfig(
        level=log_level,
        format=(
            "%(asctime)s %(name)s %(levelname)s %(message)s "
            "[org=%(org)s repo=%(repo)s branch=%(branch)s commit=%(commit)s event=%(event)s scan_id=%(scan_id)s]"
        ),
        handlers=handlers,
    )


def main(argv: list[str] | None = None) -> int:
    """Run the agent scanner CLI.

    Args:
        argv: Command line arguments. If None, uses sys.argv.

    Returns:
        Exit code: 0 for success, 1 for client initialisation failure, 2 for scan failure.
    """
    parser = argparse.ArgumentParser(description="Scan a GitHub repo for AI/agentic indicators")
    parser.add_argument("--base-url", help="GitHub API base URL (e.g., https://api.github.com)")
    parser.add_argument("--org", "--orgs", dest="org", help="Repository organisation/owner")
    parser.add_argument(
        "--repo-api-url",
        help="Repository API URL (e.g., https://api.github.com/repos/owner/repo)",
    )
    parser.add_argument("--branch", help="Specific branch to scan (default: repository's default branch)")
    parser.add_argument("--list-branches", action="store_true", help="List all branches in the repository and exit")
    parser.add_argument(
        "--workspace-path",
        help="Path to local repository directory for scanning local files instead of GitHub API",
    )
    parser.add_argument("--verbose", action="store_true", help="Enable debug logging")
    parser.add_argument("--fail-fast", action="store_true", help="Stop and error on first fetch failure")
    parser.add_argument("--log-file", help="Path to write logs to")
    parser.add_argument("--summary-file", help="Path to write scan summary JSON")
    parser.add_argument("--schema-path", default=DEFAULT_SCHEMA_PATH, help="Path to the scanner JSON schema")
    parser.add_argument("--timeout-seconds", type=int, help="Maximum scan duration before timing out")
    parser.add_argument(
        "--ignore-paths",
        help="Comma separated list of repository paths to ignore during scanning",
    )
    parser.add_argument(
        "--output-path",
        dest="output_path",
        help="Path to write scan summary JSON (preferred; overrides --summary-file)",
    )
    parser.add_argument(
        "--log-level",
        choices=["debug", "info", "warning", "error"],
        help="Set the log level for this run",
    )
    parser.add_argument("--commit", help="Commit hash for correlation metadata")
    parser.add_argument("--event", help="Event name for correlation metadata")
    parser.add_argument(
        "--scan-org-recent",
        action="store_true",
        help="Scan all repos in an org that have pushed updates within a recent window",
    )
    parser.add_argument(
        "--recent-days",
        type=int,
        help="Include repos pushed within the last N days (defaults to 180 when bulk scanning)",
    )
    parser.add_argument(
        "--recent-since",
        help="Include repos pushed since YYYY-MM-DD (UTC). Overrides --recent-days when provided.",
    )
    parser.add_argument(
        "--max-repos",
        type=int,
        help="Optional cap on number of repos to scan in bulk mode (after filtering by push date)",
    )
    parser.add_argument(
        "--scan-workers",
        type=int,
        default=4,
        help="Number of concurrent workers when bulk scanning org repos",
    )
    parser.add_argument(
        "--output-dir",
        default="output",
        help="Directory containing existing summaries to use when skipping already scanned repos",
    )
    skip_group = parser.add_mutually_exclusive_group()
    skip_group.add_argument(
        "--skip-existing-output",
        dest="skip_existing_output",
        action="store_true",
        default=True,
        help="Skip scanning repos that already have a summary file in the output directory",
    )
    skip_group.add_argument(
        "--no-skip-existing-output",
        dest="skip_existing_output",
        action="store_false",
        help="Force scanning even if a summary file already exists",
    )
    parser.add_argument(
        "--rate-limit-sleep",
        type=int,
        default=60,
        help="Seconds to sleep after a rate limit error before continuing bulk scans",
    )
    parser.add_argument(
        "--list-retries",
        type=int,
        default=3,
        help="Number of retry cycles when listing org repositories after rate limits",
    )
    parser.add_argument(
        "--list-retry-sleep",
        type=int,
        default=60,
        help="Seconds to sleep between org list retries when rate limited",
    )
    parsed_args = parser.parse_args(argv)

    repository_full_name = None
    resolved_base_url = parsed_args.base_url
    extracted_org = None
    extracted_repo_name = None

    if parsed_args.repo_api_url:
        try:
            resolved_base_url, repository_full_name = _parse_repo_api_url(parsed_args.repo_api_url)
        except ValueError as exc:
            logger.error("Invalid --repo-api-url: %s", exc)
            return EXIT_CLIENT_INIT_FAILURE

        if repository_full_name and "/" in repository_full_name:
            extracted_org, extracted_repo_name = repository_full_name.split("/", 1)

        if parsed_args.org:
            logger.info("--repo-api-url provided; ignoring --org")

    pipeline_env = os.getenv("AGENT_SCANNER_PIPELINE_MODE", "0") == "1"

    if not repository_full_name and pipeline_env and parsed_args.workspace_path:
        workspace_path_obj = Path(parsed_args.workspace_path)
        repo_dir_name = workspace_path_obj.name or "workspace"
        repository_full_name = f"local/{repo_dir_name}"
        extracted_org = "local"
        extracted_repo_name = repo_dir_name

    correlation_org = extracted_org or parsed_args.org
    correlation_repo = extracted_repo_name or (
        repository_full_name.split("/", 1)[-1] if repository_full_name and "/" in repository_full_name else None
    )

    correlation_commit = parsed_args.commit

    if parsed_args.timeout_seconds is not None and parsed_args.timeout_seconds <= 0:
        logger.error("--timeout-seconds must be a positive number")
        return EXIT_CLIENT_INIT_FAILURE

    resolved_output_path = parsed_args.output_path or parsed_args.summary_file
    if parsed_args.output_path and parsed_args.summary_file:
        logger.info("--output-path provided; overriding --summary-file")

    pre_generated_scan_id = str(uuid.uuid4())

    log_level = _resolve_log_level(parsed_args.log_level, parsed_args.verbose)
    correlation_filter = CorrelationFilter(
        correlation_org,
        correlation_repo,
        parsed_args.branch,
        correlation_commit,
        parsed_args.event,
        pre_generated_scan_id,
    )
    _setup_logging_handlers(parsed_args, log_level, correlation_filter)

    logger.info(
        "Run header: org=%s repo=%s branch=%s commit=%s event=%s repository=%s workspace=%s schema=%s "
        "timeout_seconds=%s ignore_paths=%s log_level=%s output_path=%s",
        correlation_org,
        correlation_repo,
        parsed_args.branch or "",
        correlation_commit or "",
        parsed_args.event or "",
        repository_full_name,
        parsed_args.workspace_path,
        parsed_args.schema_path,
        parsed_args.timeout_seconds,
        parsed_args.ignore_paths,
        parsed_args.log_level or ("debug" if parsed_args.verbose else "info"),
        resolved_output_path,
    )

    os.environ["AGENT_SCANNER_SCHEMA_PATH"] = parsed_args.schema_path
    os.environ["AGENT_SCANNER_IGNORE_PATHS"] = parsed_args.ignore_paths or ""

    if parsed_args.timeout_seconds is not None:
        deadline_epoch = time.time() + parsed_args.timeout_seconds
        os.environ["AGENT_SCANNER_DEADLINE_EPOCH"] = str(deadline_epoch)
        logger.info("Timeout deadline set for %s", datetime.fromtimestamp(deadline_epoch, tz=UTC).isoformat())

    if parsed_args.workspace_path and parsed_args.scan_org_recent:
        logger.error("--workspace-path is not compatible with --scan-org-recent")
        return EXIT_CLIENT_INIT_FAILURE

    pipeline_mode = pipeline_env and bool(parsed_args.workspace_path or parsed_args.scan_org_recent)
    if pipeline_env and not pipeline_mode:
        logger.warning("Pipeline mode enabled but --workspace-path missing; falling back to API mode")

    if pipeline_mode:
        os.environ.setdefault("SCANNER_MAX_WORKERS", "1")

        if parsed_args.skip_existing_output:
            logger.info("Pipeline mode enabled: disabling skip-existing-output to ensure reproducible scans")
            parsed_args.skip_existing_output = False
        if not parsed_args.scan_org_recent:
            if not parsed_args.workspace_path:
                logger.error("--workspace-path is required in pipeline mode")
                return EXIT_CLIENT_INIT_FAILURE
            if not resolved_output_path:
                logger.error("--output-path or --summary-file is required in pipeline mode")
                return EXIT_CLIENT_INIT_FAILURE
    if parsed_args.scan_org_recent:
        if resolved_output_path:
            logger.error("--output-path/--summary-file is not supported with --scan-org-recent")
            return EXIT_CLIENT_INIT_FAILURE
        if pipeline_mode:
            logger.error("Org bulk scanning is disabled in pipeline mode")
            return EXIT_CLIENT_INIT_FAILURE
        if not ORG_BULK_ENABLED:
            logger.error("Org bulk scanning is disabled via AGENT_SCANNER_BULK_ENABLED=0")
            return EXIT_CLIENT_INIT_FAILURE
        if not parsed_args.org:
            logger.error("--scan-org-recent requires --org")
            return EXIT_CLIENT_INIT_FAILURE
    elif not repository_full_name:
        logger.error("--repo-api-url is required when not scanning an org's recent repositories")
        return EXIT_CLIENT_INIT_FAILURE

    github_client, file_source_client, init_error = _initialise_github_clients(
        parsed_args, pipeline_mode, resolved_base_url
    )
    if init_error is not None:
        return init_error

    if parsed_args.scan_org_recent and github_client is None:
        logger.error("Failed to initialise GitHub client")
        return EXIT_CLIENT_INIT_FAILURE

    list_branches_result = _handle_list_branches(parsed_args, repository_full_name, github_client, pipeline_mode)
    if list_branches_result is not None:
        return list_branches_result

    matcher, matcher_error = _load_pattern_matcher()
    if matcher_error is not None:
        return matcher_error

    file_source_client, workspace_error = _initialise_workspace_client(
        parsed_args, github_client, file_source_client, resolved_base_url
    )
    if workspace_error is not None:
        return workspace_error

    def _resolve_pushed_after() -> datetime | None:
        if parsed_args.recent_since:
            try:
                parsed_since_date = datetime.strptime(parsed_args.recent_since, "%Y-%m-%d").replace(tzinfo=UTC)
                return parsed_since_date
            except ValueError:
                logger.error("--recent-since must be in YYYY-MM-DD format")
                raise

        default_days = 180 if parsed_args.scan_org_recent else None
        days = parsed_args.recent_days if parsed_args.recent_days is not None else default_days
        if days is None:
            return None
        if days <= 0:
            logger.error("--recent-days must be positive")
            raise ValueError("recent-days must be positive")
        cutoff = datetime.now(UTC) - timedelta(days=days)
        logger.info("Using recent cutoff: %s days -> %s", days, cutoff.isoformat())
        return cutoff

    def _log_repo_result(repo_result: RepoScanResult) -> None:
        if repo_result.agentic_signals_detected:
            name = f"{repo_result.org}/{repo_result.repo_name}"
            logger.info("Match found: %s (stage=%s)", name, repo_result.matched_stage)
            if repo_result.dependency_files:
                logger.info("Dependency files: %s", ", ".join(repo_result.dependency_files))
            if repo_result.ai_dependencies:
                dependencies_list = [
                    f"{dependency.package_name}{(' ' + dependency.version) if dependency.version else ''}"
                    for dependency in repo_result.ai_dependencies
                ]
                logger.info("AI dependencies: %s", ", ".join(dependencies_list))
            if repo_result.agent_instances:
                total_detections = sum(
                    cast(int, file_agents.get("count", 0)) for file_agents in repo_result.agent_instances
                )
                unique_count = (
                    repo_result.agent_counts_unique[0]["count"]
                    if (
                        repo_result.agent_counts_unique
                        and isinstance(
                            repo_result.agent_counts_unique[0].get("count"),
                            int,
                        )
                    )
                    else 0
                )
                logger.info(
                    "Found %d total agents across %d files (%d unique)",
                    total_detections,
                    len(repo_result.agent_instances),
                    unique_count,
                )
        else:
            logger.info(
                "No agentic signals detected in %s/%s - no output generated",
                repo_result.org,
                repo_result.repo_name,
            )

    def _load_existing_repo_stems(output_dir: str) -> set[str]:
        output_path = Path(output_dir)
        if not output_path.exists():
            return set()
        return {p.stem.lower() for p in output_path.glob("*.json") if p.is_file()}

    def _should_skip_repo(repo_full_name: str, existing_stems: set[str]) -> bool:
        if not parsed_args.skip_existing_output:
            return False
        repo_name_only = repo_full_name.split("/", 1)[-1].lower()
        sanitised_full = repo_full_name.replace("/", "-").lower()
        for candidate in (repo_name_only, sanitised_full):
            if any(candidate in stem for stem in existing_stems):
                logger.info("Skipping %s because a summary already exists", repo_full_name)
                return True
        return False

    if parsed_args.scan_org_recent:
        try:
            pushed_after = _resolve_pushed_after()
        except ValueError:
            return EXIT_CLIENT_INIT_FAILURE

        def _fetch_org_repos_with_retry() -> list[dict[str, Any]]:
            attempts = max(parsed_args.list_retries, 1)
            for attempt in range(1, attempts + 1):
                try:
                    org_client = github_client
                    if org_client is None:
                        raise RuntimeError("GitHub client was not initialised")
                    return org_client.list_org_repos(
                        parsed_args.org,
                        pushed_after=pushed_after,
                        max_repos=parsed_args.max_repos,
                    )
                except GitHubClientError as exc:
                    message = str(exc).lower()
                    is_rate_limited = "rate limit" in message or "http 403" in message or "http 429" in message
                    if not is_rate_limited or attempt >= attempts:
                        raise
                    logger.warning(
                        "Rate limit while listing org repos (attempt %s/%s); sleeping %s seconds",
                        attempt,
                        attempts,
                        parsed_args.list_retry_sleep,
                    )
                    time.sleep(max(parsed_args.list_retry_sleep, 0))

            return []

        try:
            repos = _fetch_org_repos_with_retry()
        except GitHubClientError as exc:
            logger.exception("Failed to list org repositories: %s", exc)
            return EXIT_CLIENT_INIT_FAILURE

        if not repos:
            logger.info("No repositories matched the recent push window for org %s", parsed_args.org)
            return EXIT_SUCCESS

        repo_full_names = [
            repo.get("full_name") or f"{parsed_args.org}/{repo.get('name')}" for repo in repos if repo.get("name")
        ]
        repo_full_names = [repo_name for repo_name in repo_full_names if repo_name]

        existing_stems = _load_existing_repo_stems(parsed_args.output_dir)
        if existing_stems:
            logger.info("Found %d existing summary files in %s", len(existing_stems), parsed_args.output_dir)

        repo_full_names = [repo for repo in repo_full_names if not _should_skip_repo(repo, existing_stems)]

        if not repo_full_names:
            logger.info("All repositories already scanned; nothing to do")
            return EXIT_SUCCESS

        logger.info(
            "Queued %d repositories from org %s for scanning%s",
            len(repo_full_names),
            parsed_args.org,
            f" (limit {parsed_args.max_repos})" if parsed_args.max_repos else "",
        )

        def _scan_single_repo(repo_full_name: str) -> tuple[str, bool]:
            logger.info("Starting scan for %s", repo_full_name)
            try:
                client = GitHubClient(api_url=parsed_args.base_url)
                scanner = Scanner(
                    client,
                    matcher,
                    file_source_client=file_source_client,
                    correlation_filter=correlation_filter,
                    scan_id=pre_generated_scan_id,
                    commit_hash=correlation_commit,
                )
                result = scanner.scan(
                    repo_full_name,
                    branch=parsed_args.branch,
                    verbose=parsed_args.verbose,
                    fail_fast=parsed_args.fail_fast,
                )
                repo_result = getattr(scanner, "_repo_result", None)
                if result and repo_result is not None:
                    has_agents = bool(repo_result.agent_instances)
                    has_ai_dependencies = bool(repo_result.ai_dependencies)
                    if not has_agents and not has_ai_dependencies:
                        logger.info(
                            "No agents or AI dependencies found in %s/%s - skipping summary output",
                            repo_result.org,
                            repo_result.repo_name,
                        )
                        return repo_full_name, True

                    _log_repo_result(repo_result)
                    if repo_result.agentic_signals_detected:
                        try:
                            write_summary_file(resolved_output_path, repo_result)
                        except (SummaryWriteError, ValueError) as exc:
                            logger.exception("Failed to write summary file: %s", exc)
                            return repo_full_name, False
                logger.info("Finished scan for %s", repo_full_name)
                return repo_full_name, True
            except ScannerError as exc:
                logger.exception("Scan failed for %s: %s", repo_full_name, exc)
                message = str(exc).lower()
                if "rate limit" in message or "http 403" in message or "http 429" in message:
                    logger.warning(
                        "Rate limit suspected while scanning %s; sleeping for %s seconds",
                        repo_full_name,
                        parsed_args.rate_limit_sleep,
                    )
                    time.sleep(max(parsed_args.rate_limit_sleep, 0))
                return repo_full_name, False
            except Exception as exc:
                logger.exception("Unexpected failure scanning %s: %s", repo_full_name, exc)
                return repo_full_name, False

        failures = 0
        workers = max(parsed_args.scan_workers, 1)
        logger.info("Launching thread pool with %d workers", workers)
        with ThreadPoolExecutor(max_workers=workers) as executor:
            future_map = {executor.submit(_scan_single_repo, repo): repo for repo in repo_full_names}
            for future in as_completed(future_map):
                repo_name = future_map[future]
                try:
                    _, is_successful = future.result()
                    if not is_successful:
                        failures += 1
                        logger.error("Scan failed for %s (current failures=%d)", repo_name, failures)
                except Exception as exc:
                    logger.exception("Unhandled exception scanning %s: %s", repo_name, exc)
                    failures += 1
                    logger.error("Scan failed for %s (current failures=%d)", repo_name, failures)

        if failures:
            logger.error("Completed with %d failures out of %d repositories", failures, len(repo_full_names))
            return EXIT_SCAN_FAILURE

        logger.info("Completed scans for %d repositories", len(repo_full_names))
        return EXIT_SUCCESS

    if repository_full_name is None:
        raise RuntimeError("repository_full_name should not be None")
    if github_client is None:
        raise RuntimeError("github_client should not be None")

    scanner = Scanner(
        github_client,
        matcher,
        file_source_client=file_source_client,
        correlation_filter=correlation_filter,
        scan_id=pre_generated_scan_id,
        commit_hash=correlation_commit,
    )
    try:
        result = scanner.scan(
            repository_full_name,
            branch=parsed_args.branch,
            verbose=parsed_args.verbose,
            fail_fast=parsed_args.fail_fast,
        )
    except ScannerError as exc:
        logger.exception("Scan failed: %s", exc)
        return EXIT_SCAN_FAILURE

    repository_result = getattr(scanner, "_repo_result", None)

    if result and repository_result is not None:
        has_agents = bool(repository_result.agent_instances)
        has_ai_dependencies = bool(repository_result.ai_dependencies)
        if not has_agents and not has_ai_dependencies:
            logger.info(
                "No agents or AI dependencies found in %s/%s - skipping summary output",
                repository_result.org,
                repository_result.repo_name,
            )
            return EXIT_SUCCESS

        _log_repo_result(repository_result)
        if repository_result.agentic_signals_detected:
            try:
                write_summary_file(resolved_output_path, repository_result)
            except (SummaryWriteError, ValueError) as exc:
                logger.exception("Failed to write summary file: %s", exc)
                return EXIT_SCAN_FAILURE

    return EXIT_SUCCESS


if __name__ == "__main__":
    raise SystemExit(main())
