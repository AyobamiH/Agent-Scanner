"""File utility helpers for filtering and sampling repository files.

Provides functions for:
- Determining file depth in repository hierarchy
- Filtering code files from binary/excluded types
- Grouping files by path depth
- Even sampling across directory depths for representative coverage

Extensions are hardcoded for simplicity but can be made configurable if needed.
"""

from __future__ import annotations

import os
from collections.abc import Iterable, Mapping

CODE_EXTENSIONS = {
    ".py",
    ".js",
    ".ts",
    ".jsx",
    ".tsx",
    ".java",
    ".go",
    ".rs",
    ".rb",
    ".php",
    ".scala",
    ".kt",
    ".swift",
    ".c",
    ".cpp",
    ".cs",
    ".yaml",
    ".yml",
    ".json",
    ".toml",
    ".md",
    ".txt",
    ".ipynb",
    ".tf",
    ".bru",
}

SKIP_EXTENSIONS = {".png", ".jpg", ".jpeg", ".gif", ".pdf", ".zip", ".exe", ".dll"}

FileMetadata = Mapping[str, object]


def file_depth(path: str) -> int:
    """Calculate the depth of a file path from the repository root.

    Depth is the number of directory separators in the path. Root-level files
    have depth 0, files in immediate subdirectories have depth 1, etc.

    Args:
        path: File path (forward or backward slashes supported).

    Returns:
        Non-negative integer representing path depth. Empty paths return 0.
    """
    if not path:
        return 0
    normalised_path = path.replace("\\", "/")
    return len([part for part in normalised_path.split("/") if part]) - 1


def is_code_file(path: str) -> bool:
    """Determine if a file path represents scannable code or text content.

    Filters by file extension, excluding common binary types (images, archives, etc.)
    and including known code, configuration, and documentation formats.

    Args:
        path: File path to evaluate.

    Returns:
        True if the file should be scanned, False for binary or excluded types.
    """
    _, extension = os.path.splitext(path.lower())
    if not extension:
        return True
    if extension in SKIP_EXTENSIONS:
        return False
    return extension in CODE_EXTENSIONS


def group_files_by_depth(files: Iterable[FileMetadata]) -> dict[int, list[FileMetadata]]:
    """Group file metadata dictionaries by their path depth.

    Args:
        files: Iterable of file dictionaries, each containing a path key.

    Returns:
        Dictionary mapping depth (int) to list of file dictionaries at that depth.
        Files with missing or empty paths are assigned to depth 0.
    """
    groups: dict[int, list[FileMetadata]] = {}
    for file_item in files:
        path_value = file_item.get("path", "")
        depth = file_depth(path_value if isinstance(path_value, str) else "")
        groups.setdefault(depth, []).append(file_item)
    return groups


def sample_evenly_by_depth(files: Iterable[FileMetadata], limit: int) -> list[FileMetadata]:
    """Sample files evenly across different path depths.

    Implements round-robin sampling across depth groups, prioritising shallower
    files first. This ensures representative sampling of the repository structure
    rather than biasing towards deeply nested files.

    Args:
        files: Iterable of file dictionaries, each containing a path key.
        limit: Maximum number of files to sample. Must be non-negative.

    Returns:
        List of up to limit file dictionaries, sampled evenly across depths.
        If files contains fewer than limit items, all are returned.

    Raises:
        ValueError: If limit is negative.
    """
    if limit < 0:
        raise ValueError("limit must be non-negative")
    if limit == 0:
        return []
    groups = group_files_by_depth(files)

    sorted_depths = sorted(groups.keys())
    samples: list[FileMetadata] = []
    if not sorted_depths:
        return samples

    iterators = {depth_value: iter(groups[depth_value]) for depth_value in sorted_depths}
    depth_index = 0
    while len(samples) < limit:
        depth = sorted_depths[depth_index % len(sorted_depths)]
        try:
            candidate = next(iterators[depth])
            samples.append(candidate)
        except StopIteration:
            sorted_depths.remove(depth)
            if not sorted_depths:
                break
            iterators = {depth_value: iter(groups[depth_value]) for depth_value in sorted_depths}
            depth_index = 0
            continue
        depth_index += 1

    return samples[:limit]
