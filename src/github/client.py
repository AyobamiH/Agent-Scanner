"""
Lightweight GitHub client with retry/backoff.
"""

from __future__ import annotations

import logging
import os
import time
from collections.abc import Callable
from datetime import UTC, datetime
from typing import Any, TypedDict

import requests

from src.exceptions import GitHubClientError
from src.utils.cache import FileCache
from src.utils.rate_limiter import (
    CircuitBreaker,
    RateLimitAwareSleeper,
    TokenBucket,
)

ERROR_OWNER_EMPTY = "owner must be a non-empty string"
ERROR_REPO_EMPTY = "repo must be a non-empty string"


class RepoTreeEntry(TypedDict, total=False):
    path: str
    mode: str
    sha: str
    type: str
    size: int


logger = logging.getLogger(__name__)

DEFAULT_MAX_RETRY_ATTEMPTS = 4
GRAPHQL_MAX_RETRY_ATTEMPTS = 3
INITIAL_BACKOFF_SECONDS = 1.0
BACKOFF_MULTIPLIER = 2
REQUEST_TIMEOUT_SECONDS = 10
RATE_LIMIT_STATUS_CODES = frozenset({403, 429})
DEFAULT_CACHE_MAX_ITEMS = 2000
DEFAULT_CACHE_TTL_SECONDS = 3600
MAX_REPO_SIZE_ENV = "AGENT_SCANNER_MAX_REPO_BYTES"

_load_dotenv: Callable[..., bool] | None = None
try:
    from dotenv import load_dotenv as _load_dotenv_import

    _load_dotenv = _load_dotenv_import
except ImportError as exception:
    logger.debug("python-dotenv not available: %s", exception)


class GitHubClient:
    """GitHub REST and GraphQL API client with automatic retry and caching.

    This client handles authentication, retry logic for rate limits, and optional
    persistent caching of repository trees and file contents. Supports both
    public GitHub (api.github.com) and GitHub Enterprise instances.

    The client uses environment variables for configuration:
        GITHUB_TOKEN: Required authentication token
        GITHUB_API_URL: Optional custom API URL for GitHub Enterprise
        GITHUB_RAW_URL: Optional custom raw content URL
        GITHUB_CACHE_PATH: Optional path for persistent cache file
        GITHUB_CACHE_MAX_ITEMS: Optional cache size limit (default 2000)
        GITHUB_CACHE_TTL: Optional cache TTL in seconds (default 3600)
    """

    API_URL = "https://api.github.com"

    def __init__(
        self,
        token: str | None = None,
        max_file_size: int = 1_000_000,
        max_workers: int = 8,
        api_url: str | None = None,
        raw_url: str | None = None,
    ) -> None:
        """Initialise the GitHub client.

        Args:
            token: GitHub personal access token. If None, reads from GITHUB_TOKEN environment variable.
            max_file_size: Maximum file size in bytes to fetch (default 1MB).
            max_workers: Maximum concurrent workers for file fetching (default 8).
            api_url: Custom GitHub API URL for Enterprise instances. Reads from GITHUB_API_URL if not provided.
            raw_url: Custom raw content URL. Reads from GITHUB_RAW_URL if not provided.

        Raises:
            GitHubClientError: If GITHUB_TOKEN is not provided or found in environment.
        """
        if callable(_load_dotenv):
            _load_dotenv()
        self.token = token or os.getenv("GITHUB_TOKEN")

        self._api_url = (api_url or os.getenv("GITHUB_API_URL") or self.API_URL).rstrip("/")

        self.raw_url = raw_url or os.getenv("GITHUB_RAW_URL")
        if not self.token:
            raise GitHubClientError("GITHUB_TOKEN not provided")
        if not isinstance(self.token, str) or len(self.token.strip()) == 0:
            raise GitHubClientError("GITHUB_TOKEN must be a non-empty string")

        self.session = requests.Session()
        self.session.headers.update(
            {
                "Authorization": f"token {self.token}",
                "Accept": "application/vnd.github.v3+json",
            }
        )

        self.max_file_size = max_file_size
        self.max_workers = max_workers

        logger.debug("GitHub client initialised for API URL: %s", self._api_url)

        self.api_stats: dict[str, int] = {
            "api_gets": 0,
            "contents_gets": 0,
            "raw_gets": 0,
            "bytes": 0,
        }
        self._cache = None
        self._repo_tree_cache: dict[str, list[dict[str, Any]]] = {}
        self._file_content_cache: dict[str, str] = {}
        self._404_cache: set[str] = set()
        requests_per_second = float(os.getenv("GITHUB_RATE_LIMIT_RPS", "5.0"))
        self._rate_limiter = TokenBucket(capacity=requests_per_second * 2, refill_rate=requests_per_second)
        self._circuit_breaker = CircuitBreaker(
            failure_threshold=int(os.getenv("GITHUB_CIRCUIT_BREAKER_THRESHOLD", "10")),
            recovery_timeout=float(os.getenv("GITHUB_CIRCUIT_BREAKER_TIMEOUT", "120.0")),
            success_threshold=2,
        )
        recent_window = int(os.getenv("GITHUB_RATE_LIMIT_RECENT_WINDOW", "10"))
        recent_threshold = float(os.getenv("GITHUB_RATE_LIMIT_RECENT_THRESHOLD", "120.0"))
        self._sleep_helper = RateLimitAwareSleeper(
            use_jitter=True,
            recent_window=recent_window,
            recent_threshold_seconds=recent_threshold,
        )

        pipeline_mode = os.getenv("AGENT_SCANNER_PIPELINE_MODE", "0") == "1"
        persistent = os.getenv("GITHUB_CACHE_PATH") or os.getenv("GITHUB_PERSISTENT_CACHE") == "1"
        if pipeline_mode:
            persistent = False
        if persistent:
            cache_path = os.getenv("GITHUB_CACHE_PATH") or os.path.join(os.getcwd(), ".cache", "github_cache.json")
            try:
                self._cache = FileCache(
                    cache_path,
                    max_items=int(os.getenv("GITHUB_CACHE_MAX_ITEMS", str(DEFAULT_CACHE_MAX_ITEMS))),
                    default_ttl=int(os.getenv("GITHUB_CACHE_TTL", str(DEFAULT_CACHE_TTL_SECONDS))),
                )
            except (OSError, ValueError) as exception:
                logger.debug("Failed to initialise FileCache at %s: %s", cache_path, exception)
                self._cache = None

    @staticmethod
    def _resolve_repo_size_limit() -> int | None:
        configured = os.getenv(MAX_REPO_SIZE_ENV)
        if not configured:
            return None
        try:
            limit = int(configured)
        except ValueError:
            logger.warning("Invalid %s value: %s", MAX_REPO_SIZE_ENV, configured)
            return None
        if limit <= 0:
            return None
        return limit

    def _enforce_repo_size_limit(self, entries: list[dict[str, Any]]) -> None:
        limit = self._resolve_repo_size_limit()
        if limit is None:
            return
        total_size = 0
        for entry in entries:
            if entry.get("type") != "blob":
                continue
            size = entry.get("size")
            if isinstance(size, int):
                total_size += size
                if total_size > limit:
                    raise GitHubClientError(f"Repository size {total_size} exceeds limit {limit} bytes")

    def _get_from_cache(self, cache_key: str) -> Any | None:
        """Retrieve a value from cache (persistent or in-memory).

        Args:
            cache_key: Cache key to look up.

        Returns:
            Cached value if found and not expired, None otherwise.
        """
        try:
            if self._cache:
                cached = self._cache.get(cache_key)
                if cached is not None:
                    logger.debug("Cache hit for %s", cache_key)
                    return cached
            else:
                if cache_key.startswith("repo_tree:"):
                    if cache_key in self._repo_tree_cache:
                        logger.debug("In-memory cache hit for %s", cache_key)
                        return self._repo_tree_cache[cache_key]
                elif cache_key.startswith("file:"):
                    if cache_key in self._file_content_cache:
                        logger.debug("In-memory cache hit for %s", cache_key)
                        return self._file_content_cache[cache_key]
        except (OSError, TypeError, ValueError) as exception:
            logger.debug("Cache lookup failed for %s: %s", cache_key, exception)
        return None

    def _set_cache(self, cache_key: str, value: Any, context: str = "") -> None:
        """Store a value in cache (persistent or in-memory).

        Args:
            cache_key: Cache key to store under.
            value: Value to cache.
            context: Optional context string for logging (e.g., GraphQL, REST).
        """
        try:
            if self._cache:
                self._cache.set(cache_key, value)
                context_str = f" ({context})" if context else ""
                logger.debug("Cached %s%s", cache_key, context_str)
            else:
                if cache_key.startswith("repo_tree:"):
                    self._repo_tree_cache[cache_key] = value
                    context_str = f" ({context})" if context else ""
                    logger.debug("Cached in memory %s%s", cache_key, context_str)
                elif cache_key.startswith("file:"):
                    self._file_content_cache[cache_key] = value
                    context_str = f" ({context})" if context else ""
                    logger.debug("Cached in memory %s%s", cache_key, context_str)
        except (OSError, TypeError, ValueError) as exception:
            logger.debug("Failed to cache %s: %s", cache_key, exception)

    def _execute_request_with_retry(
        self,
        url: str,
        method: str = "GET",
        max_attempts: int = DEFAULT_MAX_RETRY_ATTEMPTS,
        timeout: int = REQUEST_TIMEOUT_SECONDS,
        params: dict[str, Any] | None = None,
        json_data: dict[str, Any] | None = None,
        headers: dict[str, str] | None = None,
        stat_key: str = "api_gets",
    ) -> requests.Response:
        """Execute an HTTP request with exponential backoff retry logic.

        Args:
            url: Full URL to request.
            method: HTTP method (GET or POST).
            max_attempts: Maximum number of retry attempts.
            timeout: Request timeout in seconds.
            params: Optional query parameters for GET requests.
            json_data: Optional JSON body for POST requests.
            headers: Optional additional headers.
            stat_key: Key for tracking API statistics.

        Returns:
            Response object if successful.

        Raises:
            GitHubClientError: If request fails after all retries.
        """
        for _attempt in range(max_attempts):

            if not self._rate_limiter.consume(tokens=1.0, timeout=30.0):
                raise GitHubClientError("Rate limiter timeout: could not acquire tokens")

            start = time.time()
            try:
                if method == "POST":
                    if headers:
                        response = self.session.post(url, json=json_data, timeout=timeout, headers=headers)
                    else:
                        response = self.session.post(url, json=json_data, timeout=timeout)
                else:
                    if headers:
                        response = self.session.get(url, params=params, timeout=timeout, headers=headers)
                    else:
                        response = self.session.get(url, params=params, timeout=timeout)
            except requests.RequestException as exception:
                elapsed = time.time() - start
                logger.warning("Request failed for %s: %s (elapsed=%.3fs)", url, exception, elapsed)
                self._circuit_breaker.record_failure()
                if _attempt >= max_attempts - 1:
                    raise GitHubClientError(f"Request failed: {exception}") from exception

                sleep_time = self._sleep_helper.calculate_sleep_time(
                    0, {}, _attempt, INITIAL_BACKOFF_SECONDS, BACKOFF_MULTIPLIER
                )
                self._sleep_helper.sleep_and_log(sleep_time, "request_exception", url)
                continue
            except Exception as exception:
                elapsed = time.time() - start
                logger.warning("Unexpected request failure for %s: %s (elapsed=%.3fs)", url, exception, elapsed)
                self._circuit_breaker.record_failure()
                if _attempt >= max_attempts - 1:
                    raise GitHubClientError(f"Request failed: {exception}") from exception

                sleep_time = self._sleep_helper.calculate_sleep_time(
                    0, {}, _attempt, INITIAL_BACKOFF_SECONDS, BACKOFF_MULTIPLIER
                )
                self._sleep_helper.sleep_and_log(sleep_time, "unexpected_exception", url)
                continue

            elapsed = time.time() - start
            content_length = response.headers.get("Content-Length") if hasattr(response, "headers") else None
            logger.debug(
                "%s %s -> status=%s elapsed=%.3fs content-length=%s",
                method,
                url,
                response.status_code,
                elapsed,
                content_length,
            )

            self.api_stats[stat_key] = self.api_stats.get(stat_key, 0) + 1
            if content_length:
                try:
                    self.api_stats["bytes"] += int(content_length)
                except ValueError:
                    pass

            if response.status_code == 200:
                self._circuit_breaker.record_success()
                return response

            if response.status_code in RATE_LIMIT_STATUS_CODES:
                self._circuit_breaker.record_failure()

                sleep_seconds = self._sleep_helper.calculate_sleep_time(
                    response.status_code,
                    dict(response.headers) if hasattr(response, "headers") else {},
                    _attempt,
                    INITIAL_BACKOFF_SECONDS,
                    BACKOFF_MULTIPLIER,
                )

                self._sleep_helper.sleep_and_log(sleep_seconds, "rate_limited", url)
                continue

            error_message = f"HTTP {response.status_code}"
            logger.error("Request failed: %s for %s", error_message, url)
            self._circuit_breaker.record_failure()
            raise GitHubClientError(error_message)

        raise GitHubClientError(f"Exceeded {max_attempts} retry attempts")

    def _get_github_api(self, path: str, params: dict[str, Any] | None = None) -> dict[str, Any]:
        """Perform a GET request against the GitHub API with automatic retry.

        Implements exponential backoff for rate-limited responses (403, 429).
        Retries up to 4 times with increasing delays.

        Args:
            path: API endpoint path (e.g., /repos/owner/repo).
            params: Optional query parameters.

        Returns:
            Parsed JSON response as a dictionary.

        Raises:
            GitHubClientError: If the request fails after all retries or receives an unexpected status code.
        """
        url = f"{self._api_url}{path}"
        response = self._execute_request_with_retry(
            url=url,
            method="GET",
            max_attempts=DEFAULT_MAX_RETRY_ATTEMPTS,
            timeout=REQUEST_TIMEOUT_SECONDS,
            params=params,
            stat_key="api_gets",
        )
        return response.json()

    def fetch_all_commits(
        self, repo_full_name: str, max_commits: int = 100, branch: str | None = None
    ) -> list[dict[str, Any]]:
        """Fetch commit history from a repository.

        Args:
            repo_full_name: Repository in owner/repo format.
            max_commits: Maximum number of commits to fetch.
            branch: Optional branch name to fetch commits from. If None, uses default branch.

        Returns:
            List of commit dictionaries with author, committer, and email information.

        Raises:
            GitHubClientError: If commit fetching fails.
        """
        try:
            owner, repo = repo_full_name.split("/", 1)
        except ValueError as exception:
            raise GitHubClientError("repo_full_name must be in owner/repo format") from exception

        params: dict[str, Any] = {"per_page": min(max_commits, 100)}
        if branch:
            params["sha"] = branch

        commits_url = f"/repos/{owner}/{repo}/commits"

        try:
            response = self._get_github_api(commits_url, params=params)

            normalised_commits: list[dict[str, Any]] = []
            if isinstance(response, list):
                for commit_data in response:
                    commit_object = commit_data.get("commit", {})
                    author_object = commit_object.get("author", {})
                    committer_object = commit_object.get("committer", {})

                    normalised_commits.append(
                        {
                            "author": author_object.get("name", ""),
                            "committer": committer_object.get("name", ""),
                            "author_email": author_object.get("email", ""),
                            "committer_email": committer_object.get("email", ""),
                        }
                    )

            return normalised_commits[:max_commits]
        except GitHubClientError:
            raise
        except (KeyError, TypeError, requests.RequestException) as exception:
            raise GitHubClientError(f"Failed to fetch commits: {exception}") from exception

    def _get_graphql(self, query: str, variables: dict[str, Any] | None = None) -> dict[str, Any]:
        """Execute a GraphQL query against the GitHub API with automatic retry.

        Implements exponential backoff for rate-limited responses (403, 429).
        Retries up to 3 times with increasing delays.

        Args:
            query: GraphQL query string.
            variables: Optional query variables dictionary.

        Returns:
            Parsed JSON response body.

        Raises:
            GitHubClientError: If the request fails after all retries.
        """
        url = f"{self._api_url.rstrip('/')}/graphql"
        headers = {"Accept": "application/vnd.github.v3+json"}
        resp = self._execute_request_with_retry(
            url=url,
            method="POST",
            max_attempts=GRAPHQL_MAX_RETRY_ATTEMPTS,
            timeout=REQUEST_TIMEOUT_SECONDS,
            json_data={"query": query, "variables": variables or {}},
            headers=headers,
            stat_key="api_gets",
        )
        return resp.json()

    def get_branches(self, owner: str, repo: str) -> list[str]:
        """Retrieve all branch names for a repository.

        Fetches branch names using pagination to handle repositories with many branches.
        Results are cached to avoid redundant API calls.

        Args:
            owner: Repository owner (user or organisation name).
            repo: Repository name.

        Returns:
            List of branch name strings.

        Raises:
            GitHubClientError: If branches cannot be fetched or validation fails.
        """
        if not owner or not isinstance(owner, str):
            raise GitHubClientError(ERROR_OWNER_EMPTY)
        if not repo or not isinstance(repo, str):
            raise GitHubClientError(ERROR_REPO_EMPTY)

        cache_key = f"branches:{owner}/{repo}"
        cached = self._get_from_cache(cache_key)
        if cached is not None:
            return cached

        logger.debug("Cache miss for %s, fetching from API", cache_key)

        branches: list[str] = []
        page = 1
        per_page = 100

        try:
            while True:
                params = {"page": page, "per_page": per_page}
                data = self._get_github_api(f"/repos/{owner}/{repo}/branches", params=params)

                if not data or not isinstance(data, list):
                    break

                for branch in data:
                    branch_name = branch.get("name")
                    if branch_name:
                        branches.append(branch_name)

                if len(data) < per_page:
                    break

                page += 1

            self._set_cache(cache_key, branches)
            logger.debug("Fetched %d branches for %s/%s", len(branches), owner, repo)
            return branches

        except GitHubClientError:
            raise
        except (KeyError, TypeError, requests.RequestException) as exception:
            raise GitHubClientError("Failed to fetch repository branches") from exception

    def list_org_repos(
        self,
        org: str,
        pushed_after: datetime | None = None,
        per_page: int = 100,
        max_pages: int | None = None,
        max_repos: int | None = None,
    ) -> list[dict[str, Any]]:
        """List repositories in an organisation, optionally filtered by push date.

        Args:
            org: Organisation name.
            pushed_after: If provided, only include repos with pushed_at >= this UTC datetime.
            per_page: Page size for GitHub pagination (max 100).
            max_pages: Optional limit on number of pages to fetch.
            max_repos: Optional cap on number of repos to return.

        Returns:
            List of repository metadata dictionaries as returned by the GitHub API.

        Raises:
            GitHubClientError: On validation or API failures.
        """
        if not org or not isinstance(org, str):
            raise GitHubClientError("org must be a non-empty string")

        repos: list[dict[str, Any]] = []
        page = 1
        pages_fetched = 0

        logger.info(
            "Listing org repos (org=%s, per_page=%s, max_pages=%s, max_repos=%s, pushed_after=%s)",
            org,
            per_page,
            max_pages,
            max_repos,
            pushed_after.isoformat() if pushed_after else None,
        )

        while True:
            if max_pages is not None and pages_fetched >= max_pages:
                break

            params = {
                "type": "all",
                "sort": "pushed",
                "direction": "desc",
                "per_page": min(per_page, 100),
                "page": page,
            }

            data = self._get_github_api(f"/orgs/{org}/repos", params=params)
            pages_fetched += 1

            if not data or not isinstance(data, list):
                break

            for repo in data:
                pushed_at_raw = repo.get("pushed_at")
                pushed_dt: datetime | None = None
                if pushed_at_raw:
                    try:
                        pushed_dt = datetime.strptime(pushed_at_raw, "%Y-%m-%dT%H:%M:%SZ").replace(tzinfo=UTC)
                    except (TypeError, ValueError):
                        pushed_dt = None

                if pushed_after and pushed_dt and pushed_dt < pushed_after:
                    logger.info(
                        "Stopping at page %s repo %s due to pushed_at cutoff (%s < %s)",
                        page,
                        repo.get("full_name") or repo.get("name"),
                        pushed_dt,
                        pushed_after,
                    )
                    logger.info("Collected %d repos for org %s", len(repos), org)
                    return repos

                repos.append(repo)

                if max_repos is not None and len(repos) >= max_repos:
                    logger.info("Hit max_repos=%s after page %s", max_repos, page)
                    return repos

            if len(data) < params["per_page"]:
                break

            page += 1

        logger.info("Collected %d repos for org %s", len(repos), org)
        return repos

    def get_repo_tree(
        self, owner: str, repo: str, branch: str | None = None
    ) -> tuple[list[dict[str, Any]], dict[str, Any]]:
        """Retrieve the complete file tree for a repository.

        Fetches the repository's file tree recursively for a specific branch or the default branch.
        Results are cached to avoid redundant API calls. Attempts to use GraphQL API first for
        efficiency, falling back to REST API if needed.

        Args:
            owner: Repository owner (user or organisation name).
            repo: Repository name.
            branch: Optional branch name. If None, uses the repository's default branch.

        Returns:
            Tuple of (tree_entries, metadata) where:
            - tree_entries: List of file tree entries with keys: path, mode, sha, type, size (optional)
            - metadata: Dict with keys: default_branch, head_sha, html_url

        Raises:
            GitHubClientError: If the repository tree cannot be fetched.
        """
        if not owner or not isinstance(owner, str):
            raise GitHubClientError(ERROR_OWNER_EMPTY)
        if not repo or not isinstance(repo, str):
            raise GitHubClientError(ERROR_REPO_EMPTY)

        cache_key = f"repo_tree:{owner}/{repo}:{branch or 'default'}"
        cached = self._get_from_cache(cache_key)
        if cached is not None:
            return cached

        logger.debug("Cache miss for %s, fetching from API", cache_key)

        try:
            if self._api_url.rstrip("/") in (self.API_URL, "https://api.github.com"):
                result = self._get_repo_tree_graphql(owner, repo, branch)
                if result is not None:
                    self._set_cache(cache_key, result, "GraphQL")
                    return result
        except (GitHubClientError, requests.RequestException) as exception:
            logger.debug("GraphQL repo tree attempt failed, falling back to REST: %s", exception)

        try:
            repository_response = self._get_github_api(f"/repos/{owner}/{repo}")
            default_branch = repository_response.get("default_branch", "main")
            html_url = repository_response.get("html_url", f"https://github.com/{owner}/{repo}")

            target_branch = branch or default_branch

            commit = self._get_github_api(f"/repos/{owner}/{repo}/git/refs/heads/{target_branch}")
            sha = commit.get("object", {}).get("sha")
            if not sha:
                raise GitHubClientError(f"Could not determine sha for branch {target_branch}")
            tree = self._get_github_api(f"/repos/{owner}/{repo}/git/trees/{sha}", params={"recursive": "1"})
            entries = tree.get("tree", [])
            entries = sorted(entries, key=lambda entry: (entry.get("path", ""), entry.get("type", "")))
            self._enforce_repo_size_limit(entries)

            metadata = {
                "default_branch": default_branch,
                "head_sha": sha,
                "html_url": html_url,
            }
            result = (entries, metadata)
            self._set_cache(cache_key, result, "REST")
            return result
        except GitHubClientError:
            raise
        except (KeyError, TypeError, requests.RequestException) as exception:
            raise GitHubClientError("Failed to fetch repository tree") from exception

    def _get_repo_tree_graphql(
        self, owner: str, repo: str, branch: str | None = None
    ) -> tuple[list[dict[str, Any]], dict[str, Any]] | None:
        """Attempt to retrieve a repository file tree using GraphQL API.

        This method provides more efficient tree retrieval than REST API pagination.
        Returns normalised entries matching the REST API format for compatibility.

        Args:
            owner: Repository owner name.
            repo: Repository name.
            branch: Optional branch name. If None, uses the repository's default branch.

        Returns:
            Tuple of (tree_entries, metadata) or None if GraphQL is unavailable.
        """
        if branch is None:
            query = """
            query($owner: String!, $name: String!) {
                repository(owner: $owner, name: $name) {
                    url
                    defaultBranchRef {
                        name
                        target {
                            ... on Commit {
                                oid
                                tree {
                                    entries {
                                        path
                                        mode
                                        oid
                                        type
                                        object {
                                            ... on Blob { byteSize }
                                        }
                                    }
                                }
                            }
                        }
                    }
                }
            }
            """
            variables = {"owner": owner, "name": repo}
        else:
            query = """
            query($owner: String!, $name: String!, $branch: String!) {
                repository(owner: $owner, name: $name) {
                    url
                    defaultBranchRef {
                        name
                    }
                    ref(qualifiedName: $branch) {
                        name
                        target {
                            ... on Commit {
                                oid
                                tree {
                                    entries {
                                        path
                                        mode
                                        oid
                                        type
                                        object {
                                            ... on Blob { byteSize }
                                        }
                                    }
                                }
                            }
                        }
                    }
                }
            }
            """
            variables = {"owner": owner, "name": repo, "branch": f"refs/heads/{branch}"}

        try:
            data = self._get_graphql(query, variables)
            repository = data.get("data", {}).get("repository")
            if not repository:
                return None

            if branch is None:
                default_branch_reference = repository.get("defaultBranchRef")
                if not default_branch_reference:
                    return None
                target_object = default_branch_reference.get("target", {})
                tree_object = target_object.get("tree", {})
                entries = tree_object.get("entries", [])
                branch_name = default_branch_reference.get("name", "main")
                default_branch_name = branch_name
            else:
                branch_reference = repository.get("ref")
                if not branch_reference:
                    return None
                target_object = branch_reference.get("target", {})
                tree_object = target_object.get("tree", {})
                entries = tree_object.get("entries", [])
                branch_name = branch_reference.get("name", branch)

                default_branch_reference = repository.get("defaultBranchRef")
                default_branch_name = (
                    default_branch_reference.get("name", "main") if default_branch_reference else "main"
                )

            normalised_entries: list[dict[str, Any]] = []
            for entry in entries:
                normalised_entry = {
                    "path": entry.get("path"),
                    "mode": entry.get("mode"),
                    "sha": entry.get("oid"),
                    "type": entry.get("type").lower() if entry.get("type") else None,
                }
                entry_object = entry.get("object") or {}
                if entry_object and isinstance(entry_object, dict):
                    normalised_entry["size"] = entry_object.get("byteSize")
                normalised_entries.append(normalised_entry)

            normalised_entries.sort(key=lambda entry: (entry.get("path", ""), entry.get("type", "")))
            self._enforce_repo_size_limit(normalised_entries)

            metadata = {
                "default_branch": default_branch_name if branch else branch_name,
                "head_sha": target_object.get("oid"),
                "html_url": repository.get("url", f"https://github.com/{owner}/{repo}"),
            }
            return (normalised_entries, metadata)
        except (KeyError, TypeError, AttributeError) as exception:
            logger.debug("GraphQL tree parsing failed: %s", exception)
            return None

    def _fetch_via_contents_api(self, owner: str, repo: str, path: str, branch: str | None = None) -> str | None:
        """Attempt to fetch file content via GitHub Contents API.

        Args:
            owner: Repository owner.
            repo: Repository name.
            path: File path within repository.
            branch: Optional branch name. If None, uses repository default.

        Returns:
            File content as UTF-8 string if successful, None if failed.

        Raises:
            GitHubClientError: If file exceeds max_file_size.
        """
        contents_url = f"{self._api_url}/repos/{owner}/{repo}/contents/{path}"
        logger.debug("Attempting contents API url: %s", contents_url)
        headers = {"Accept": "application/vnd.github.v3.raw"}
        backoff = INITIAL_BACKOFF_SECONDS

        for _attempt in range(2):
            logger.debug("Contents API attempt %d for %s", _attempt + 1, path)
            previous_accept_header = self.session.headers.get("Accept")
            self.session.headers.update(headers)
            params = {"ref": branch} if branch else None

            start = time.time()
            try:
                response = self.session.get(contents_url, params=params, timeout=REQUEST_TIMEOUT_SECONDS)
            except requests.RequestException as request_exception:
                elapsed = time.time() - start
                logger.warning(
                    "Contents API request failed for %s: %s (elapsed=%.3fs)",
                    path,
                    request_exception,
                    elapsed,
                )
                if previous_accept_header is None:
                    self.session.headers.pop("Accept", None)
                else:
                    self.session.headers["Accept"] = previous_accept_header
                return None

            elapsed = time.time() - start
            content_length = response.headers.get("Content-Length")
            logger.debug(
                "Contents API GET %s -> status=%s elapsed=%.3fs content-length=%s",
                contents_url,
                response.status_code,
                elapsed,
                content_length,
            )

            self.api_stats["contents_gets"] = self.api_stats.get("contents_gets", 0) + 1
            if content_length:
                try:
                    self.api_stats["bytes"] += int(content_length)
                except ValueError:
                    pass

            if previous_accept_header is None:
                self.session.headers.pop("Accept", None)
            else:
                self.session.headers["Accept"] = previous_accept_header

            if response.status_code == 200:
                if content_length is not None:
                    try:
                        if int(content_length) > self.max_file_size:
                            raise GitHubClientError("File skipped due to size")
                    except ValueError:
                        pass

                logger.debug("Reading response text for %s (content-length=%s)", path, content_length)
                response.encoding = response.encoding or "utf-8"
                text_content = response.text
                logger.debug("Finished reading response text for %s (length=%d)", path, len(text_content))
                return text_content

            if response.status_code in RATE_LIMIT_STATUS_CODES:
                logger.warning(
                    "Contents API rate limited, status=%s, retrying in %.1fs (%s)",
                    response.status_code,
                    backoff,
                    contents_url,
                )
                time.sleep(backoff)
                backoff *= BACKOFF_MULTIPLIER
                continue

            logger.debug("Contents API returned %s for %s", response.status_code, contents_url)
            break

        return None

    def _fetch_via_raw_url(self, owner: str, repo: str, path: str, cache_key: str, branch: str | None = None) -> str:
        """Fetch file content via raw.githubusercontent.com.

        Args:
            owner: Repository owner.
            repo: Repository name.
            path: File path within repository.
            cache_key: Cache key for 404 tracking.
            branch: Optional branch name. If None, uses HEAD.

        Returns:
            File content as UTF-8 string.

        Raises:
            GitHubClientError: If fetch fails or file exceeds max_file_size.
        """
        branch_ref = branch or "HEAD"

        if self.raw_url:
            raw_url = f"{self.raw_url.rstrip('/')}/{owner}/{repo}/{branch_ref}/{path}"
        else:
            raw_url = f"https://raw.githubusercontent.com/{owner}/{repo}/{branch_ref}/{path}"

        logger.debug("Fetching raw file url: %s", raw_url)
        backoff = INITIAL_BACKOFF_SECONDS

        for _attempt in range(DEFAULT_MAX_RETRY_ATTEMPTS):
            logger.debug("Raw URL attempt %d for %s (timeout=%ds)", _attempt + 1, path, REQUEST_TIMEOUT_SECONDS)
            start = time.time()

            try:
                response = self.session.get(raw_url, timeout=REQUEST_TIMEOUT_SECONDS)
            except requests.RequestException as request_exception:
                elapsed = time.time() - start
                logger.warning(
                    "Raw URL request failed for %s (attempt %d): %s (elapsed=%.3fs)",
                    path,
                    _attempt + 1,
                    request_exception,
                    elapsed,
                )
                if _attempt >= DEFAULT_MAX_RETRY_ATTEMPTS - 1:
                    raise GitHubClientError(f"All raw URL attempts failed: {request_exception}") from request_exception
                time.sleep(backoff)
                backoff *= BACKOFF_MULTIPLIER
                continue

            elapsed = time.time() - start
            content_length = response.headers.get("Content-Length")
            logger.debug(
                "Raw GET %s -> status=%s elapsed=%.3fs content-length=%s",
                raw_url,
                response.status_code,
                elapsed,
                content_length,
            )

            self.api_stats["raw_gets"] = self.api_stats.get("raw_gets", 0) + 1
            if content_length:
                try:
                    self.api_stats["bytes"] += int(content_length)
                except ValueError:
                    pass

            if response.status_code == 200:
                if content_length is not None:
                    try:
                        if int(content_length) > self.max_file_size:
                            raise GitHubClientError("File skipped due to size")
                    except ValueError:
                        pass

                logger.debug("Reading raw response text for %s (content-length=%s)", path, content_length)
                response.encoding = response.encoding or "utf-8"
                text_content = response.text
                logger.debug("Finished reading raw response text for %s (length=%d)", path, len(text_content))
                return text_content

            if response.status_code in RATE_LIMIT_STATUS_CODES:
                logger.warning(
                    "Raw content rate limited, status=%s, retrying in %.1fs (%s)",
                    response.status_code,
                    backoff,
                    raw_url,
                )
                time.sleep(backoff)
                backoff *= BACKOFF_MULTIPLIER
                continue

            if response.status_code == 404:
                self._404_cache.add(cache_key)
                logger.debug("Added %s to 404 cache", path)

            logger.debug("Raising error for %s: status=%s", path, response.status_code)
            raise GitHubClientError(f"Failed to fetch file content: {response.status_code} {raw_url}")

        raise GitHubClientError("Exceeded retries fetching file content")

    def get_file_content(self, owner: str, repo: str, path: str, branch: str | None = None) -> str:
        """Fetch raw file content from a repository.

        Attempts to fetch file content using the GitHub Contents API first,
        falling back to raw.githubusercontent.com if that fails. Enforces
        max_file_size limit. Results are cached.

        Args:
            owner: Repository owner (user or organisation name).
            repo: Repository name.
            path: File path within the repository.
            branch: Optional branch name. If None, uses HEAD.

        Returns:
            File contents as a decoded UTF-8 string.

        Raises:
            GitHubClientError: If the file cannot be fetched or exceeds size limits.
        """
        if not owner or not isinstance(owner, str):
            raise GitHubClientError(ERROR_OWNER_EMPTY)
        if not repo or not isinstance(repo, str):
            raise GitHubClientError(ERROR_REPO_EMPTY)
        if not path or not isinstance(path, str):
            raise GitHubClientError("path must be a non-empty string")

        cache_key = f"file:{owner}/{repo}:{branch or 'HEAD'}:{path}"
        cached = self._get_from_cache(cache_key)
        if cached is not None:
            return cached

        if cache_key in self._404_cache:
            logger.debug("File %s is in 404 cache, skipping fetch", path)
            raise GitHubClientError(f"File not found (cached 404): {path}")

        logger.debug("Cache miss for file %s, fetching from API", path)

        content = self._fetch_via_contents_api(owner, repo, path, branch)
        if content is not None:
            self._set_cache(cache_key, content, "Contents API")
            return content

        logger.debug("Contents API failed, falling back to raw URL for %s", path)
        content = self._fetch_via_raw_url(owner, repo, path, cache_key, branch)
        self._set_cache(cache_key, content, "Raw URL")
        return content
