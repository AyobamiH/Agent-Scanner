"""
Scanner orchestration for three-stage repo analysis.
"""

import ast
import logging
import os
import time
import uuid
from concurrent.futures import ThreadPoolExecutor, TimeoutError, as_completed
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, cast

from src.detectors.agents.agents import AgentDetector
from src.detectors.dependencies import COMMON_FILES, DependencyParser
from src.detectors.framework_detector import FrameworkDetector
from src.detectors.language_detector import LanguageDetector
from src.detectors.patterns import PatternMatcher
from src.detectors.repository_info import detect_likely_owner
from src.exceptions import GitHubClientError, ScannerError
from src.models.results import DependencyInfo, RepoScanResult
from src.utils.file_utils import is_code_file, sample_evenly_by_depth

logger = logging.getLogger(__name__)

STAGE_1_PATH_SCORE_THRESHOLD = 1
STAGE_2_SAMPLE_SIZE = 50
STAGE_3_SAMPLE_SIZE = 100
CONTENT_SCORE_THRESHOLD = 3
ERROR_FETCH_MESSAGE = "Failed to fetch %s: %s"
EXECUTOR_TIMEOUT_SECONDS = 300
FUTURE_RESULT_TIMEOUT_SECONDS = 30
PROGRESS_LOG_INTERVAL_SECONDS = 5.0
MAX_TOP_SCORES_TRACKED = 5
IGNORED_ROOTS = ("tests",)

DEFAULT_MAX_WORKERS = 3
DEFAULT_BATCH_SIZE = 10
DEFAULT_BATCH_DELAY_SECONDS = 1.0
IGNORE_PATHS_ENV = "AGENT_SCANNER_IGNORE_PATHS"
DEADLINE_ENV = "AGENT_SCANNER_DEADLINE_EPOCH"


def _resolve_max_workers() -> int:
    """Resolve the maximum number of concurrent workers for file fetching.

    Reads from SCANNER_MAX_WORKERS environment variable with DEFAULT_MAX_WORKERS as fallback.
    Ensures value is at least 1 to prevent zero-worker scenarios.

    Returns:
        Positive integer representing max concurrent workers.
    """
    try:
        return max(int(os.getenv("SCANNER_MAX_WORKERS", str(DEFAULT_MAX_WORKERS))), 1)
    except (TypeError, ValueError):
        return DEFAULT_MAX_WORKERS


def _resolve_batch_size() -> int:
    """Resolve the batch size for batched API requests.

    Reads from SCANNER_BATCH_SIZE environment variable with DEFAULT_BATCH_SIZE as fallback.
    Used to limit concurrent requests to avoid overwhelming rate limits.
    Ensures value is at least 1 to prevent zero-batch scenarios.

    Returns:
        Positive integer representing batch size.
    """
    try:
        return max(int(os.getenv("SCANNER_BATCH_SIZE", str(DEFAULT_BATCH_SIZE))), 1)
    except (TypeError, ValueError):
        return DEFAULT_BATCH_SIZE


def _resolve_batch_delay_seconds() -> float:
    """Resolve the delay (in seconds) between batch submissions.

    Reads from SCANNER_BATCH_DELAY environment variable with DEFAULT_BATCH_DELAY_SECONDS
    as fallback. Used for rate limiting and API throttling.

    Returns:
        Non-negative float representing delay in seconds (0.0 means no delay).
    """
    try:
        return max(float(os.getenv("SCANNER_BATCH_DELAY", str(DEFAULT_BATCH_DELAY_SECONDS))), 0.0)
    except (TypeError, ValueError):
        return DEFAULT_BATCH_DELAY_SECONDS


class Scanner:
    """Orchestrates multi-stage scanning of a repository for agentic patterns.

    Implements a progressive detection strategy with early exit:
        Stage 1: Scan file/folder paths for keywords (fast, no content fetch)
        Stage 2: Sample and score 50 code files
        Stage 3: Sample and score 100 additional files

    If any stage reaches threshold, exits early and extracts dependencies and agent metadata.
    Uses concurrent file fetching, AST parsing for Python, and regex fallback for other formats.
    """

    def __init__(
        self,
        github_client: Any,
        pattern_matcher: PatternMatcher | None = None,
        file_source_client: Any | None = None,
        correlation_filter: Any | None = None,
        scan_id: str | None = None,
        commit_hash: str | None = None,
    ) -> None:
        """Initialise the scanner with a GitHub client and pattern matcher.

        Args:
            github_client: Configured GitHub API client.
            pattern_matcher: Optional pattern matcher. If None, loads default from configuration file.
            file_source_client: Optional file source client (GitHubClient or LocalFilesystemClient).
                If None, uses github_client for file operations. Used for local filesystem scanning.
            correlation_filter: Optional correlation filter to update with scan_id when scan starts.
            scan_id: Optional pre-generated scan ID. If None, a new one will be generated during scan().
            commit_hash: Optional commit hash from pipeline/CLI. Used instead of git commit if provided.
        """
        self.github = github_client
        self._file_source = file_source_client or github_client
        self.matcher = pattern_matcher or PatternMatcher.from_file()
        self.agent_detector = AgentDetector()
        self.framework_detector = FrameworkDetector()
        self.language_detector = LanguageDetector()
        self.correlation_filter = correlation_filter
        self._scan_id = scan_id
        self._commit_hash = commit_hash

        self._file_matches: dict[str, dict[str, Any]] = {}
        self._detection: dict[str, Any] = {
            "matched_stage": None,
            "matched_paths": None,
            "top_files": [],
        }
        self._file_cache = None

    def _is_ignored_path(self, path: str) -> bool:
        """Check if a repository path should be skipped during scanning.

        Paths in IGNORED_ROOTS (typically 'tests/') are excluded to avoid
        false positives from test code. Additional paths can be configured
        via AGENT_SCANNER_IGNORE_PATHS environment variable.

        Args:
            path: Repository file path to check (e.g., 'tests/test_foo.py').

        Returns:
            True if path should be ignored, False if it should be scanned.
        """
        if not isinstance(path, str):
            return False
        normalised = path.replace("\\", "/").lstrip("/").lower()
        ignored = self._resolve_ignored_roots()
        return any(normalised == root or normalised.startswith(f"{root}/") for root in ignored)

    @staticmethod
    def _resolve_ignored_roots() -> tuple[str, ...]:
        """Resolve the list of repository paths to ignore during scanning.

        Combines hardcoded IGNORED_ROOTS (e.g., 'tests/') with paths from
        AGENT_SCANNER_IGNORE_PATHS environment variable (comma-separated).
        All paths are normalised to lowercase for case-insensitive matching.

        Returns:
            Tuple of normalised path strings to skip during scanning.
        """
        configured = os.getenv(IGNORE_PATHS_ENV, "")
        if not configured:
            return IGNORED_ROOTS
        entries = [item.strip() for item in configured.split(",")]
        cleaned = [item.replace("\\", "/").lstrip("/").lower() for item in entries if item]
        return tuple(dict.fromkeys([*IGNORED_ROOTS, *cleaned]))

    @staticmethod
    def _resolve_deadline_epoch() -> float | None:
        """Resolve the scan deadline as Unix epoch timestamp.

        Reads from AGENT_SCANNER_DEADLINE_EPOCH environment variable.
        Used to enforce maximum scan duration across all stages.

        Returns:
            Float Unix epoch timestamp if configured, None if not set or invalid.
        """
        configured = os.getenv(DEADLINE_ENV)
        if not configured:
            return None
        try:
            return float(configured)
        except ValueError:
            logger.warning("Invalid %s value: %s", DEADLINE_ENV, configured)
            return None

    def _check_deadline(self, context: str) -> None:
        deadline = self._resolve_deadline_epoch()
        if deadline is None:
            return
        now = time.time()
        if now >= deadline:
            logger.error("Timeout exceeded during %s (deadline: %s)", context, context)
            raise ScannerError(f"Scan exceeded timeout during {context} (deadline epoch: {deadline:.0f})")

    def _prepare_fetch_tasks(self, files: list[dict[str, Any]], seen_paths: set[str]) -> dict[str, dict[str, Any]]:
        """Filter and prepare files for fetching, excluding oversized and duplicate files.

        Removes files that:
        - Have already been processed (in seen_paths)
        - Exceed max_file_size limit (skipped to avoid large downloads)
        - Are duplicates in the current batch

        Args:
            files: List of file metadata dictionaries from GitHub/filesystem API.
            seen_paths: Set of already processed paths to avoid re-fetching.

        Returns:
            Dictionary mapping file paths to metadata for files to fetch.
        """
        unique: dict[str, dict[str, Any]] = {}
        for f in files:
            p = f.get("path")
            if not p:
                continue
            if p in seen_paths:
                logger.debug("Skipping already fetched path: %s", p)
                continue
            if p not in unique:
                unique[p] = f

        to_fetch = {}
        for p, meta in unique.items():
            size = meta.get("size")
            if isinstance(size, int) and getattr(self.github, "max_file_size", None) is not None:
                if size > self.github.max_file_size:
                    logger.debug(
                        "Skipping %s due to size %d > max %d",
                        p,
                        size,
                        self.github.max_file_size,
                    )
                    seen_paths.add(p)
                    continue
            to_fetch[p] = meta

        return to_fetch

    def _process_single_file_result(
        self, path: str, content: str
    ) -> tuple[int, list[str], list[dict[str, Any]], set[str]]:
        """Process content from a single file to extract score, tokens, agent locations, and framework imports.

        Args:
            path: File path.
            content: File content.

        Returns:
            Tuple of (score, tokens, agent_locations, framework_imports).
        """
        score = self.matcher.score_content(content)
        logger.debug("File %s scored %d", path, score)

        tokens: list[str] = []
        framework_imports: set[str] = set()
        try:
            tokens = list(self.matcher._tokenise_text(content))
        except Exception as exc:
            logger.debug("Tokenisation failed for %s: %s", path, exc)

        if isinstance(path, str):
            path_lower = path.lower()
            try:
                if path_lower.endswith(".py"):
                    try:
                        tree = ast.parse(content)
                        framework_imports = self.agent_detector._get_framework_imports(tree)
                    except SyntaxError:
                        framework_imports = set()
                    agent_locations = self.agent_detector.get_agent_locations(content)
                elif path_lower.endswith((".yaml", ".yml", ".json")):
                    agent_locations = self.agent_detector.get_structured_agent_locations(content, path)
                else:
                    agent_locations = []
            except Exception as exc:
                logger.debug("Agent detection failed for %s: %s", path, exc)
                agent_locations = []
        else:
            agent_locations = []

        return score, tokens, agent_locations, framework_imports

    @staticmethod
    def _log_scan_summary(
        top_scores: list[tuple[int, str]], error_counts: dict[str, int], total_score: int, required_score: int
    ) -> None:
        """Log summary of file scan results.

        Args:
            top_scores: List of (score, path) tuples for top scoring files.
            error_counts: Dictionary mapping error types to counts.
            total_score: Total aggregated score.
            required_score: Required score threshold.
        """
        if top_scores:
            scores_summary = ", ".join([f"{p}={s}" for s, p in top_scores])
            logger.info("Top file scores: %s", scores_summary)

        if error_counts:
            summary = ", ".join([f"{k}: {v}" for k, v in error_counts.items()])
            logger.info("Fetch failures summary: %s", summary)

        logger.info("Aggregated score %d did not reach required %d", total_score, required_score)

    def _run_stage_1_path_scan(self, blobs: list[dict[str, Any]]) -> tuple[int, list[str]] | None:
        """Stage 1: Scan file and folder paths for agentic patterns.

        Args:
            blobs: List of blob entries from repository tree.

        Returns:
            Tuple of (stage_number, list_of_matched_paths) if stage matched, None otherwise.
        """
        logger.info("Stage 1: Scanning file and folder paths")
        for entry in blobs:
            path = entry.get("path")
            if not isinstance(path, str):
                continue

            score = self.matcher.score_path(path)
            if score >= STAGE_1_PATH_SCORE_THRESHOLD:
                logger.info("Stage 1 returned AI matches - threshold met on: %s", path)
                return (1, [path])
        return None

    def _run_stage_2_content_scan(
        self, owner: str, repo: str, blobs: list[dict[str, Any]], fail_fast: bool, seen_paths: set[str]
    ) -> tuple[int, list[str]] | None:
        """Stage 2: Sample 50 code files for content scoring.

        Returns:
            Tuple of (stage_number, list_of_matched_paths) if stage matched, None otherwise.
        """
        logger.info("Stage 1 returned no AI matches. Starting Stage 2: Sampling %d code files", STAGE_2_SAMPLE_SIZE)
        code_files = [f for f in blobs if is_code_file(f.get("path", ""))]
        sampled_50 = sample_evenly_by_depth(code_files, STAGE_2_SAMPLE_SIZE)

        if sampled_50:
            matched, paths = self._scan_file_contents(
                owner=owner,
                repo=repo,
                files=[dict(f) for f in sampled_50],
                required_score=CONTENT_SCORE_THRESHOLD,
                fail_fast=fail_fast,
                seen_paths=seen_paths,
            )

            if matched:
                logger.info("Stage 2 returned AI matches - aggregated score >= %d", CONTENT_SCORE_THRESHOLD)
                return (2, paths)
        return None

    def _run_stage_3_extended_scan(
        self, owner: str, repo: str, blobs: list[dict[str, Any]], fail_fast: bool, seen_paths: set[str]
    ) -> tuple[int, list[str]] | None:
        """Stage 3: Sample 100 additional code files for broader coverage.

        Returns:
            Tuple of (stage_number, list_of_matched_paths) if stage matched, None otherwise.
        """
        logger.info(
            "Stage 2 returned no AI matches. Starting Stage 3: Sampling %d additional code files", STAGE_3_SAMPLE_SIZE
        )
        code_files = [f for f in blobs if is_code_file(f.get("path", ""))]
        unseen_files = [f for f in code_files if f.get("path") not in seen_paths]
        sampled_100 = sample_evenly_by_depth(unseen_files, STAGE_3_SAMPLE_SIZE)

        if sampled_100:
            matched, paths = self._scan_file_contents(
                owner=owner,
                repo=repo,
                files=[dict(f) for f in sampled_100],
                required_score=CONTENT_SCORE_THRESHOLD,
                fail_fast=fail_fast,
                seen_paths=seen_paths,
            )

            if matched:
                logger.info("Stage 3 returned AI matches - aggregated score >= %d", CONTENT_SCORE_THRESHOLD)
                return (3, paths)
        return None

    @staticmethod
    def _create_positive_result(owner: str, repo: str, matched_stage: int) -> RepoScanResult:
        """Create scan result for repositories with detected agentic signals.

        Args:
            owner: Repository owner.
            repo: Repository name.
            matched_stage: Stage number where match occurred (1-3).

        Returns:
            RepoScanResult with initial positive detection data.
        """
        return RepoScanResult(
            repo_name=repo,
            org=owner,
            agentic_signals_detected=True,
            matched_stage=matched_stage,
            matched_paths=[],
            dependency_files=[],
            ai_dependencies=[],
            agent_counts=[],
            agent_instances=[],
            parse_errors={},
        )

    def _scan_file_contents(
        self,
        owner: str,
        repo: str,
        files: list[dict[str, Any]],
        required_score: int = 3,
        fail_fast: bool = False,
        seen_paths: set[str] | None = None,
    ) -> tuple[bool, list[str]]:
        """Fetch and scan file contents concurrently for pattern matches.

        Returns:
            Tuple of (matched, list_of_contributing_paths) where contributing_paths
            are files with score > 0 that helped reach the threshold.

        Raises:
            ScannerError: If file fetching times out or fail_fast triggers on fetch errors.
        """
        if not files:
            return (False, [])

        seen_paths = seen_paths if seen_paths is not None else set()
        to_fetch = self._prepare_fetch_tasks(files, seen_paths)

        if not to_fetch:
            return (False, [])

        total_score = 0
        top_scores: list[tuple[int, str]] = []
        contributing_paths: list[str] = []
        error_counts: dict[str, int] = {}

        try:
            batch_size = _resolve_batch_size()
            batch_delay = _resolve_batch_delay_seconds()
            max_workers = _resolve_max_workers()

            file_paths = list(to_fetch.keys())
            branch = getattr(self, "_current_branch", None)

            for batch_idx in range(0, len(file_paths), batch_size):
                self._check_deadline("file content scanning")
                batch = file_paths[batch_idx : batch_idx + batch_size]
                logger.debug("Processing batch %d with %d files", batch_idx // batch_size + 1, len(batch))

                with ThreadPoolExecutor(max_workers=max_workers) as ex:
                    futures = {
                        ex.submit(self._file_source.get_file_content, owner, repo, p, branch): to_fetch[p]
                        for p in batch
                    }
                    logger.debug("Submitted %d file fetch tasks to ThreadPoolExecutor", len(futures))

                    failed_paths: set[str] = set()
                    completed_count = 0
                    last_log_time = time.time()

                    for fut in as_completed(futures, timeout=EXECUTOR_TIMEOUT_SECONDS):
                        completed_count += 1
                        if completed_count % 5 == 0:
                            self._check_deadline("file content scanning")
                        current_time = time.time()

                        if current_time - last_log_time >= PROGRESS_LOG_INTERVAL_SECONDS:
                            logger.debug(
                                "Progress: Completed %d/%d file fetch tasks (%.1f%%)",
                                completed_count,
                                len(futures),
                                100 * completed_count / len(futures),
                            )
                            last_log_time = current_time

                        meta = futures[fut]
                        path = meta.get("path")
                        if not path or not isinstance(path, str):
                            continue

                        try:
                            content = fut.result(timeout=FUTURE_RESULT_TIMEOUT_SECONDS)
                            seen_paths.add(path)
                        except TimeoutError:
                            failed_paths.add(path)
                            logger.error(
                                "Timeout waiting for result of %s after %d seconds",
                                path,
                                FUTURE_RESULT_TIMEOUT_SECONDS,
                            )
                            error_counts[f"Future timeout ({FUTURE_RESULT_TIMEOUT_SECONDS}s)"] = (
                                error_counts.get(f"Future timeout ({FUTURE_RESULT_TIMEOUT_SECONDS}s)", 0) + 1
                            )
                            if fail_fast:
                                error_message = f"Timeout fetching {path}"
                                raise ScannerError(error_message) from None
                            continue
                        except GitHubClientError as exc:
                            failed_paths.add(path)
                            if fail_fast:
                                logger.error(ERROR_FETCH_MESSAGE, path, exc)
                                error_message = f"Failed to fetch {path}: {exc}"
                                raise ScannerError(error_message) from exc
                            error_counts[str(exc)] = error_counts.get(str(exc), 0) + 1
                            logger.debug(ERROR_FETCH_MESSAGE, path, exc)
                            continue
                        except Exception as exc:
                            failed_paths.add(path)
                            if fail_fast:
                                logger.error("Unexpected error fetching %s: %s", path, exc)
                                error_message = f"Unexpected error fetching {path}: {exc}"
                                raise ScannerError(error_message) from exc
                            error_counts[f"Unexpected: {type(exc).__name__}"] = (
                                error_counts.get(f"Unexpected: {type(exc).__name__}", 0) + 1
                            )
                            logger.debug(ERROR_FETCH_MESSAGE, path, exc)
                            continue

                        (
                            score,
                            tokens,
                            agent_locations,
                            framework_imports,
                        ) = self._process_single_file_result(path, content)

                        self._file_matches[path] = {
                            "score": score,
                            "tokens": tokens,
                            "agent_locations": agent_locations,
                            "framework_imports": framework_imports,
                        }

                        if score > 0:
                            contributing_paths.append(path)

                        top_scores.append((score, path))
                        top_scores.sort(reverse=True)
                        if len(top_scores) > MAX_TOP_SCORES_TRACKED:
                            top_scores = top_scores[:MAX_TOP_SCORES_TRACKED]

                        total_score += score

                        if total_score >= required_score:
                            logger.info("Aggregated score %d reached required %d", total_score, required_score)
                            contributing_files_display = ", ".join(contributing_paths[:10])
                            logger.info(
                                "Contributing files (%d): %s",
                                len(contributing_paths),
                                contributing_files_display,
                            )
                            for pending in futures:
                                if not pending.done():
                                    try:
                                        pending.cancel()
                                    except Exception as exc:
                                        logger.debug("Failed to cancel pending future: %s", exc)
                            return (True, contributing_paths)

                if batch_idx + batch_size < len(file_paths):
                    logger.debug("Waiting %.1f seconds before next batch", batch_delay)
                    time.sleep(batch_delay)

        except TimeoutError:
            logger.error("Timeout waiting for file fetch tasks to complete after %d seconds", EXECUTOR_TIMEOUT_SECONDS)
            logger.error("Completed %d/%d tasks before timeout", completed_count, len(futures))
            still_running = [futures[f].get("path", "") for f in futures if not f.done() and futures[f].get("path")]
            if still_running:
                logger.error("Files still being fetched: %s", ", ".join(still_running[:10]))
            raise ScannerError("File fetching timed out") from None

        for p in failed_paths:
            seen_paths.add(p)

        self._log_scan_summary(top_scores, error_counts, total_score, required_score)
        return (False, [])

    def _extract_dependencies(self, owner: str, repo: str, tree: list[dict[str, Any]], result: RepoScanResult) -> None:
        """Extract and parse dependency files from the repository tree.

        Locates common dependency manifest files (requirements.txt, package.json, etc.),
        fetches their contents, parses them for AI-related dependencies, and populates
        the result object. Parse errors are logged but do not stop execution.

        Args:
            owner: Repository owner.
            repo: Repository name.
            tree: Repository file tree entries.
            result: RepoScanResult object to populate with dependency information.
        """
        logger.info("Extracting dependencies for %s/%s", owner, repo)
        parser = DependencyParser()
        dep_entries = [e for e in tree if e.get("type") == "blob" and Path(e.get("path", "")).name in COMMON_FILES]
        result.dependency_files = sorted(p for p in (e.get("path") for e in dep_entries) if isinstance(p, str))
        ai_deps: list[DependencyInfo] = []
        parse_errors: dict[str, str] = {}

        branch = getattr(self, "_current_branch", None)
        for path in result.dependency_files:
            if not isinstance(path, str):
                continue
            self._check_deadline("dependency extraction")
            try:
                content = self._file_source.get_file_content(owner, repo, path, branch)
                found, err = parser.extract_ai_dependencies(path, content)
                if err:
                    parse_errors[path] = err
                for name, ver in found:
                    di = DependencyInfo(package_name=name, version=ver, source_file=path)
                    ai_deps.append(di)
            except Exception as exc:
                logger.debug("Failed to parse dependency file %s: %s", path, exc)
                parse_errors[path] = str(exc)

        dedup: dict[str, DependencyInfo] = {}
        for d in ai_deps:
            key = d.package_name.lower()
            if key not in dedup:
                dedup[key] = d
        result.ai_dependencies = list(dedup.values())
        result.parse_errors = parse_errors

        logger.info(
            "Found %d AI dependencies across %d files",
            len(result.ai_dependencies),
            len(result.dependency_files),
        )
        if parse_errors:
            logger.warning("Encountered %d parse errors in dependency files", len(parse_errors))

        self._repo_result = result

    def _extract_agents(self, owner: str, repo: str, tree: list[dict[str, Any]], result: RepoScanResult) -> list[str]:
        """Extract agent instances from Python files using AST detection.

        Returns raw agent locations without enrichment or template processing.

        Args:
            owner: Repository owner name.
            repo: Repository name.
            tree: Repository file tree entries.
            result: RepoScanResult object to populate with agent information.

        Returns:
            List of framework import strings detected across scanned files.
        """
        logger.info("Extracting agent instances for %s/%s", owner, repo)
        try:
            scanned_files = {p: v for p, v in self._file_matches.items() if v.get("agent_locations")}

            processed_paths: set[str] = set(scanned_files.keys())
            # In Stage B (agent extraction), scan ALL files, not just code files
            # This ensures we capture agents in all formats (Bru, JSON, YAML, etc.)
            all_files = [
                f
                for f in tree
                if f.get("type") == "blob" and isinstance(f.get("path"), str) and not self._is_ignored_path(f["path"])
            ]

            branch = getattr(self, "_current_branch", None)
            to_fetch = self._prepare_fetch_tasks(all_files, processed_paths)

            for path, _meta in to_fetch.items():
                self._check_deadline("agent extraction")

                if not isinstance(path, str):
                    continue

                try:
                    path_lower = path.lower()
                    content = self._file_source.get_file_content(owner, repo, path, branch)
                    framework_imports: set[str] = set()

                    if path_lower.endswith(".py"):
                        try:
                            py_ast = ast.parse(content)
                            framework_imports = self.agent_detector._get_framework_imports(py_ast)
                        except SyntaxError:
                            framework_imports = set()
                        locations = self.agent_detector.get_agent_locations(content)
                    elif path_lower.endswith((".yaml", ".yml", ".json", ".bru")):
                        locations = self.agent_detector.get_structured_agent_locations(content, path_lower)
                    else:
                        locations = []

                    if locations:
                        scanned_files[path] = {
                            "agent_locations": locations,
                            "framework_imports": framework_imports,
                        }
                except Exception as exc:
                    logger.debug("Failed to process %s: %s", path, exc)
                finally:
                    processed_paths.add(path)

            agent_instances_agg = []
            total_agents = 0

            collected_imports: set[str] = set()

            for file_path in sorted(scanned_files.keys()):
                data = scanned_files[file_path]
                locations = data.get("agent_locations", [])
                if not locations:
                    continue

                collected_imports.update(data.get("framework_imports", set()))

                agent_count = len(locations)
                total_agents += agent_count

                file_language = self.language_detector._get_language_from_path(file_path)

                enriched_agents = []
                for agent_loc in locations:
                    agent_scan_id = str(uuid.uuid4())
                    line_num = agent_loc.get("line")
                    agent_url = None

                    if result.repo_url and result.current_commit_hash and line_num:
                        agent_url = f"{result.repo_url}/blob/{result.current_commit_hash}/{file_path}#L{line_num}"

                    enriched_agent = {**agent_loc, "agent_scan_id": agent_scan_id, "agent_url": agent_url}
                    if file_language:
                        enriched_agent["language"] = file_language
                    enriched_agents.append(enriched_agent)

                agent_instances_agg.append(
                    {
                        "file": file_path,
                        "count": agent_count,
                        "agents": enriched_agents,
                        "imports": sorted(data.get("framework_imports", set())),
                    }
                )

            result.agent_counts = [{"count": total_agents}] if total_agents > 0 else []
            result.agent_instances = agent_instances_agg

            unique_agents, unique_count = self._build_unique_agents(agent_instances_agg)
            result.agent_unique = unique_agents
            result.agent_counts_unique = [{"count": unique_count}] if unique_count > 0 else []

            if total_agents > 0:
                logger.info(
                    "Found %d total agents across %d files (%d unique)",
                    total_agents,
                    len(agent_instances_agg),
                    unique_count,
                )
            else:
                logger.info("No agents detected in scanned files")

            agentic_imports = list(collected_imports)
            result.agentic_imports = agentic_imports
            return agentic_imports

        except Exception as exc:
            logger.exception("Failed to extract agents: %s", exc)
            result.agent_counts = []
            result.agent_instances = []
            result.agentic_imports = []
            return []

    def _build_unique_agents(self, agent_instances: list[dict[str, object]]) -> tuple[list[dict[str, object]], int]:
        """Build deduplicated unique agents structure grouped by normalised agent name.

        Returns file-grouped structure identical to instances, but with duplicate agents removed
        (one canonical detection per unique agent name, with others as usages).
        Preserves file grouping and imports structure for direct pipeline consumption.

        Args:
            agent_instances: List of agent instance dictionaries from files

        Returns:
            Tuple of (unique_agents_list, unique_agent_count)
        """
        from collections import defaultdict

        agents_by_normalised: dict[str, list[dict[str, Any]]] = defaultdict(list)
        generic_name_guard = {
            "agent",
            "createagent",
            "buildagent",
            "makeagent",
            "initagent",
            "agentfactory",
            "generateagent",
            "instantiateagent",
        }
        file_to_imports: dict[str, set] = defaultdict(set)

        for file_entry in agent_instances:
            file_path = file_entry.get("file")
            if isinstance(file_path, str):
                imports = file_entry.get("imports")
                if isinstance(imports, (list, set)):
                    for imp in imports:
                        if isinstance(imp, str):
                            file_to_imports[file_path].add(imp)

            agents_list = file_entry.get("agents")
            if not isinstance(agents_list, list):
                continue

            for agent in agents_list:
                if not isinstance(agent, dict):
                    continue
                agent_name = agent.get("name", "")
                if not isinstance(agent_name, str) or not agent_name:
                    continue

                normalised_name = self.agent_detector._normalise_agent_name(agent_name)
                if not normalised_name:
                    continue

                group_key = normalised_name
                if normalised_name in generic_name_guard:
                    file_key = file_path or "unknown_file"
                    group_key = f"{normalised_name}::{file_key}"

                detection = {
                    **agent,
                    "name": agent_name,
                    "file": file_path,
                }

                agents_by_normalised[group_key].append(detection)

        canonical_map: dict[str, dict[str, Any]] = {}
        usages_map: dict[str, list[dict[str, Any]]] = defaultdict(list)

        for normalised_name, detections in agents_by_normalised.items():
            if not detections:
                continue

            def _priority_key(d: dict[str, Any]) -> int:
                v = d.get("detection_type")
                return int(self.agent_detector.get_detection_priority(v))

            detections_sorted = sorted(detections, key=_priority_key, reverse=True)

            canonical_map[normalised_name] = detections_sorted[0]

            if len(detections_sorted) > 1:
                usages_map[normalised_name] = detections_sorted[1:]

        unique_agents_by_file: dict[str, dict[str, Any]] = {}
        total_unique = 0

        for file_entry in agent_instances:
            file_path = file_entry.get("file")
            if not isinstance(file_path, str):
                continue

            unique_agents_in_file = []

            agents_list = file_entry.get("agents")
            if not isinstance(agents_list, list):
                continue

            for agent in agents_list:
                if not isinstance(agent, dict):
                    continue
                agent_name = agent.get("name", "")
                if not isinstance(agent_name, str) or not agent_name:
                    continue
                normalised_name = self.agent_detector._normalise_agent_name(agent_name)
                if not normalised_name:
                    continue

                group_key = normalised_name
                if normalised_name in {"agent"}:
                    group_key = f"{normalised_name}::{file_path}"

                canonical = canonical_map.get(group_key)
                if canonical and canonical.get("agent_scan_id") == agent.get("agent_scan_id"):
                    agent_entry = dict(agent)

                    if group_key in usages_map:
                        agent_entry["usages"] = [
                            {k: v for k, v in usage.items() if k != "name"} for usage in usages_map[group_key]
                        ]

                    unique_names = sorted(
                        [
                            cast(str, d.get("name"))
                            for d in agents_by_normalised.get(group_key, [])
                            if isinstance(d.get("name"), str)
                        ]
                    )
                    if len(unique_names) > 1:
                        aliases = [n for n in unique_names if n != agent_name]
                        if aliases:
                            agent_entry["aliases"] = aliases

                    unique_agents_in_file.append(agent_entry)
                    total_unique += 1

            if unique_agents_in_file:
                file_path_str = cast(str, file_path)
                unique_agents_by_file[file_path_str] = {
                    "file": file_path_str,
                    "count": len(unique_agents_in_file),
                    "agents": unique_agents_in_file,
                    "imports": sorted(file_to_imports.get(file_path_str, [])),
                }

        unique_agents_list = []
        for file_entry in agent_instances:
            file_path = file_entry.get("file")
            if not isinstance(file_path, str):
                continue
            if file_path in unique_agents_by_file:
                unique_agents_list.append(unique_agents_by_file[file_path])

        return unique_agents_list, total_unique

    def _extract_repository_info(self, owner: str, repo: str, result: RepoScanResult) -> None:
        """Extract repository metadata and owner information.

        Args:
            owner: Repository owner.
            repo: Repository name.
            result: RepoScanResult object to populate.
        """
        logger.info("Extracting repository information for %s/%s", owner, repo)

        repo_full_name = f"{owner}/{repo}"

        branch_for_owner = getattr(self, "_current_branch", None)

        owner_detection_enabled = os.getenv("AGENT_SCANNER_OWNER_DETECTION_ENABLED", "1") != "0"
        if not owner_detection_enabled:
            logger.info("Owner detection disabled via AGENT_SCANNER_OWNER_DETECTION_ENABLED=0")
            result.owner_detected = False
            result.detected_owner_name = None
            result.detected_owner_email = None
            return

        try:
            self._check_deadline("owner detection")
            owner_info = detect_likely_owner(repo_full_name, self.github, branch=branch_for_owner)
            result.detected_owner_name = owner_info.get("detectedOwnerName")
            result.detected_owner_email = owner_info.get("detectedOwnerEmail")
            result.owner_detected = bool(result.detected_owner_name or result.detected_owner_email)
            if result.owner_detected:
                logger.info("Detected likely owner: %s <%s>", result.detected_owner_name, result.detected_owner_email)
        except Exception as exc:
            logger.warning("Failed to detect repository owner: %s", exc)
            result.owner_detected = False
            result.detected_owner_name = None
            result.detected_owner_email = None

    def scan(
        self,
        repo_full_name: str,
        branch: str | None = None,
        verbose: bool = False,
        fail_fast: bool = False,
    ) -> str | None:
        """Perform three-stage progressive scan to detect agentic patterns in repository.

        Returns repo name if agentic signals detected (Stage 1: path analysis,
        Stage 2: sample 50 files, Stage 3: sample 100 files). On positive match,
        extracts dependencies and agent metadata. Populates self._repo_result.

        Args:
            repo_full_name: Repository in owner/repo format.
            branch: Optional branch name. Defaults to repository's default branch.
            verbose: Enable debug logging.
            fail_fast: Stop on first fetch failure.

        Returns:
            repo_full_name if agentic patterns detected, None otherwise.

        Raises:
            ScannerError: If repo_full_name invalid or fail_fast triggers on errors.
        """
        if verbose:
            logging.getLogger().setLevel(logging.DEBUG)

        if not repo_full_name or not isinstance(repo_full_name, str):
            raise ScannerError("repo_full_name must be a non-empty string")

        try:
            owner, repo = repo_full_name.split("/", 1)
        except ValueError as exc:
            raise ScannerError("repo_full_name must be in owner/repo format") from exc

        scan_id = self._scan_id or str(uuid.uuid4())
        scan_timestamp = datetime.now(UTC).isoformat()

        if self.correlation_filter and hasattr(self.correlation_filter, "set_scan_id"):
            self.correlation_filter.set_scan_id(scan_id)
        branch_info = f" (branch: {branch})" if branch else " (default branch)"
        logger.info("Starting scan %s for %s/%s%s", scan_id, owner, repo, branch_info)

        self._check_deadline("scan start")

        self._current_branch = branch

        file_source_name = "local filesystem" if getattr(self._file_source, "workspace_path", None) else "GitHub API"
        logger.info("Scanning mode: %s for file access", file_source_name)
        logger.info("Fetching repository tree for %s/%s%s", owner, repo, branch_info)
        repo_tree, repo_metadata = self._file_source.get_repo_tree(owner, repo, branch)

        self._check_deadline("repository tree fetch")

        repo_tree_sorted = sorted(repo_tree, key=lambda entry: (entry.get("path", ""), entry.get("type", "")))

        filtered_tree = [entry for entry in repo_tree_sorted if not self._is_ignored_path(entry.get("path", ""))]
        ignored_count = len(repo_tree_sorted) - len(filtered_tree)
        if ignored_count:
            logger.info("Ignoring %d repository entries under tests/", ignored_count)

        self._file_matches = {}
        blobs = [entry for entry in filtered_tree if entry.get("type") == "blob"]
        seen_paths: set[str] = set()

        matched_stage = None
        matched_paths: list[str] = []

        stage1_result = self._run_stage_1_path_scan(blobs)
        if stage1_result:
            matched_stage, matched_paths = stage1_result

        if matched_stage is None:
            self._check_deadline("stage 2 scan")
            stage2_result = self._run_stage_2_content_scan(owner, repo, blobs, fail_fast, seen_paths)
            if stage2_result:
                matched_stage, matched_paths = stage2_result

        if matched_stage is None:
            self._check_deadline("stage 3 scan")
            stage3_result = self._run_stage_3_extended_scan(owner, repo, blobs, fail_fast, seen_paths)
            if stage3_result:
                matched_stage, matched_paths = stage3_result

        if matched_stage is None:
            logger.info("STAGE A FAILED - Repository NOT agentic")
            return None

        logger.info("Repository classified as POSSIBLY AGENTIC (stage=%d)", matched_stage)
        logger.info("Matched paths (%d): %s", len(matched_paths), ", ".join(matched_paths[:10]))

        result = self._create_positive_result(owner, repo, matched_stage)

        result.matched_paths = sorted(matched_paths)

        result.default_branch = repo_metadata.get("default_branch")
        result.scanned_branch = branch or result.default_branch

        result.current_commit_hash = self._commit_hash or repo_metadata.get("head_sha")

        api_url_str = str(self.github.api_url) if hasattr(self.github, "api_url") else "https://api.github.com"
        web_base_url = api_url_str.replace("/api/v3", "").rstrip("/")
        if web_base_url.startswith("https://api."):
            web_base_url = web_base_url.replace("https://api.", "https://", 1)
        result.repo_url = f"{web_base_url}/{owner}/{repo}"
        result.provider = "github" if "api.github.com" in api_url_str.lower() else "github-enterprise"
        result.scan_id = scan_id
        result.scan_timestamp = scan_timestamp

        logger.info("Extracting repository information")
        self._extract_repository_info(owner, repo, result)

        logger.info("Extracting dependencies")
        self._extract_dependencies(owner, repo, filtered_tree, result)

        logger.info("Extracting agent counts and instances")
        framework_imports = self._extract_agents(owner, repo, filtered_tree, result)
        result.agentic_imports = framework_imports

        has_agents = bool(result.agent_instances) or bool(result.agent_counts)
        result.agentic_signals_detected = result.agentic_signals_detected or has_agents

        if has_agents:
            logger.info("Detecting primary framework")
            fw_result = self.framework_detector.detect_frameworks(
                imports=framework_imports,
                dependencies=result.ai_dependencies,
            )
            result.main_framework = fw_result.get("main_framework")
            result.supporting_infrastructure = fw_result.get("supporting_infrastructure", [])
            result.framework_scores = fw_result.get("framework_scores", {})
            result.multi_framework = bool(fw_result.get("multi_framework", False))
        else:
            logger.info("No agents detected")

        self._repo_result = result
        logger.info("STAGE B COMPLETE - Repo: %s", repo_full_name)

        return repo_full_name
