"""Utilities for writing scan summary files.

Provides functionality to generate and write JSON summary files containing
scan results, including repository information, agent locations, and dependencies.
"""

import json
import logging
import os
import uuid
from datetime import UTC, datetime
from pathlib import Path

from jsonschema import Draft202012Validator, ValidationError

from src.exceptions import SummaryWriteError
from src.models.results import RepoScanResult

logger = logging.getLogger(__name__)
DEFAULT_SCHEMA_PATH = Path(__file__).resolve().parents[2] / "scanner-payload.schema.json"
SCHEMA_PATH_ENV = "AGENT_SCANNER_SCHEMA_PATH"


def _sanitise_path_component(component: str) -> str:
    """Convert repository and branch names to safe filesystem components.

    Args:
        component: Raw repository or branch string.

    Returns:
        A sanitised string safe for filesystem paths.
    """

    return component.replace("/", "-").replace("\\", "-")


def _build_output_filename(repo_result: RepoScanResult) -> str:
    """Build an output filename for a repository scan result including a UTC timestamp."""
    safe_repo_name = _sanitise_path_component(repo_result.repo_name)
    branch = repo_result.scanned_branch or repo_result.default_branch or "unknown"
    branch_name = _sanitise_path_component(branch)
    timestamp = datetime.now(UTC).strftime("%Y%m%d_%H%M%S")
    return f"{safe_repo_name}_{branch_name}_{timestamp}.json"


def _generate_default_output_path(repo_result: RepoScanResult) -> Path:
    """Generate a default output path using repository and branch metadata.

    Args:
        repo_result: Repository scan result to use for path generation.

    Returns:
        Path object pointing to the generated output file location.
    """

    filename = _build_output_filename(repo_result)
    return Path("output") / filename


def _resolve_output_path(summary_file_path: str | Path | None, repo_result: RepoScanResult) -> Path:
    """Resolve the target output path, generating one when none or blank is provided.

    Args:
        summary_file_path: Optional path provided by the caller.
        repo_result: Repository scan result to use for generated filenames.

    Returns:
        Path object to write the summary to.

    Raises:
        ValueError: If summary_file_path is not None, str, or Path.
    """

    if summary_file_path is None:
        return _generate_default_output_path(repo_result)

    if isinstance(summary_file_path, Path):
        provided_path = str(summary_file_path)
    elif isinstance(summary_file_path, str):
        provided_path = summary_file_path
    else:
        raise ValueError("summary_file_path must be a string, Path, or None")

    if not provided_path.strip():
        return _generate_default_output_path(repo_result)

    resolved = Path(provided_path)

    if resolved.exists() and resolved.is_dir():
        filename = _build_output_filename(repo_result)
        return resolved / filename

    if not resolved.suffix and provided_path.rstrip("/\\") == provided_path:
        filename = _build_output_filename(repo_result)
        return resolved / filename

    if provided_path.rstrip("/\\") != provided_path:
        filename = _build_output_filename(repo_result)
        return resolved / filename

    return resolved


def _resolve_temp_path(output_path: Path) -> Path:
    """Resolve a temporary output path for atomic writes.

    Args:
        output_path: Final output path.

    Returns:
        Path to a temporary file in the same directory.
    """
    token = uuid.uuid4().hex
    return output_path.with_name(f"{output_path.name}.{token}.tmp")


def _resolve_schema_path() -> Path:
    """Resolve the schema path for validation.

    Returns:
        Path to the schema JSON file.

    Raises:
        SummaryWriteError: When the schema path is invalid.
    """
    configured = os.getenv(SCHEMA_PATH_ENV)
    if configured:
        candidate = Path(configured)
        if candidate.is_absolute():
            return candidate
        resolved = (DEFAULT_SCHEMA_PATH.parent / candidate).resolve()
        if resolved.exists():
            return resolved
        return candidate
    return DEFAULT_SCHEMA_PATH


def _load_schema(schema_path: Path) -> dict[str, object]:
    """Load JSON schema from disk.

    Args:
        schema_path: Path to the schema file.

    Returns:
        Parsed schema data.

    Raises:
        SummaryWriteError: When the schema cannot be loaded.
    """
    if not schema_path.exists():
        raise SummaryWriteError(f"Schema file not found: {schema_path}")
    try:
        with schema_path.open("r", encoding="utf-8") as file_handle:
            return json.load(file_handle)
    except (OSError, json.JSONDecodeError) as exc:
        raise SummaryWriteError(f"Failed to load schema file: {exc}") from exc


def _validate_output(output: dict[str, object]) -> None:
    """Validate output JSON against the schema.

    Args:
        output: Output payload to validate.

    Raises:
        SummaryWriteError: When schema validation fails.
    """
    schema_path = _resolve_schema_path()
    schema_data = _load_schema(schema_path)
    try:
        Draft202012Validator(schema_data).validate(output)
    except ValidationError as exc:
        raise SummaryWriteError(f"Schema validation failed: {exc.message}") from exc


def _cleanup_temp_file(temp_path: Path) -> None:
    """Remove temporary files created during output writing.

    Args:
        temp_path: Temporary file path to remove.
    """
    try:
        if temp_path.exists():
            temp_path.unlink()
    except OSError as exc:
        logger.warning("Failed to clean up temporary file %s: %s", temp_path, exc)


def write_summary_file(summary_file_path: str | Path | None, repo_result: RepoScanResult) -> None:
    """Write scan summary to a JSON file.

    If summary_file_path is None or blank, automatically generates a filename in the output/ directory
    using the pattern: output/{repo_name}_{branch}_{datetime}.json

    Args:
        summary_file_path: Optional path where summary JSON should be written. Accepts string, Path, or None.
        repo_result: Repository scan result object containing all scan data.

    Raises:
        SummaryWriteError: If writing the summary file fails.
        ValueError: If repo_result is None or summary_file_path is an invalid type.
    """

    if repo_result is None:
        raise ValueError("repo_result cannot be None")

    try:
        output = repo_result.to_dict()
    except (TypeError, ValueError) as exc:
        logger.exception("Failed serialising summary data: %s", exc)
        error_message = f"Failed to serialise summary data: {exc}"
        raise SummaryWriteError(error_message) from exc

    try:
        _validate_output(output)
    except SummaryWriteError as exc:
        logger.exception("Schema validation failed: %s", exc)
        raise

    output_path = _resolve_output_path(summary_file_path, repo_result)
    temp_path = _resolve_temp_path(output_path)

    try:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(temp_path, "w", encoding="utf-8") as fh:
            json.dump(output, fh, indent=2)
        os.replace(temp_path, output_path)
    except OSError as exc:
        _cleanup_temp_file(temp_path)
        logger.exception("Failed writing summary to %s: %s", output_path, exc)
        error_message = f"Failed to write summary file: {exc}"
        raise SummaryWriteError(error_message) from exc
    except (TypeError, ValueError) as exc:
        _cleanup_temp_file(temp_path)
        logger.exception("Failed serialising summary data: %s", exc)
        error_message = f"Failed to serialise summary data: {exc}"
        raise SummaryWriteError(error_message) from exc

    logger.info("Wrote summary to %s", output_path)
