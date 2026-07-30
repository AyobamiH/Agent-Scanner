"""Result data models for repository scans.

Provides dataclass models for scan results, dependency information, and agent locations.
"""

from __future__ import annotations

import importlib.metadata
from dataclasses import dataclass, field


def _get_scanner_version() -> str:
    """Get the scanner package version from metadata or return default.

    Returns:
        Version string from package metadata, or "1.0.0" if not found.
    """
    try:
        return importlib.metadata.version("agent-scanner")
    except importlib.metadata.PackageNotFoundError:
        return "1.0.0"


@dataclass
class DependencyInfo:
    """Information about a detected AI or agent framework dependency.

    Attributes:
        package_name: Name of the package or dependency.
        version: Optional version string or version specifier.
        source_file: Path to the dependency file where this was found.
    """

    package_name: str
    version: str | None
    source_file: str


@dataclass
class AgentCount:
    """Agent count information.

    Attributes:
        count: Number of detected agents.
    """

    count: int


@dataclass
class RepoScanResult:
    """Complete results from a repository scan for agentic patterns.

    Attributes:
        repo_name: Repository name.
        org: Repository owner or organisation.
        agentic_signals_detected: Whether agentic patterns were found.
        matched_stage: Which scan stage triggered detection (1, 2, or 3).
        matched_paths: File path that triggered the detection.
        dependency_files: List of dependency manifest file paths found.
        ai_dependencies: List of detected AI/agent framework dependencies.
        agent_counts: List of agent count dictionaries per framework.
        agent_instances: List of detected agent instance details.
        parse_errors: Dictionary mapping file paths to parse error messages.
    """

    repo_name: str
    org: str
    provider: str | None = None
    repo_url: str | None = None
    default_branch: str | None = None
    scanned_branch: str | None = None
    current_commit_hash: str | None = None
    scan_id: str | None = None
    scan_timestamp: str | None = None
    owner_detected: bool = False
    detected_owner_name: str | None = None
    detected_owner_email: str | None = None
    agentic_signals_detected: bool = False
    matched_stage: int | None = None
    matched_paths: list[str] = field(default_factory=list)
    dependency_files: list[str] = field(default_factory=list)
    ai_dependencies: list[DependencyInfo] = field(default_factory=list)
    agent_counts: list[dict[str, object]] = field(default_factory=list)
    agent_counts_unique: list[dict[str, object]] = field(default_factory=list)
    agent_instances: list[dict[str, object]] = field(default_factory=list)
    agent_unique: list[dict[str, object]] = field(default_factory=list)
    main_framework: str | None = None
    supporting_infrastructure: list[str] = field(default_factory=list)
    framework_scores: dict[str, int] = field(default_factory=dict)
    multi_framework: bool = False
    agentic_imports: list[str] = field(default_factory=list)
    parse_errors: dict[str, str] = field(default_factory=dict)

    def to_dict(self) -> dict[str, object]:
        """Convert the result to a dictionary for serialisation.

        Returns:
            Dictionary representation matching the scanner-payload.schema.json structure.
            Organises fields into repo, scan, and detected sections.
        """
        ai_dependencies = [
            {
                "package_name": dep.package_name,
                "version": dep.version or "",
                "source_file": dep.source_file,
            }
            for dep in self.ai_dependencies
        ]

        signals: dict[str, object] = {
            "agentic": self.agentic_signals_detected,
            "matched_paths": self.matched_paths,
        }
        if self.matched_stage is not None:
            signals["matched_stage"] = self.matched_stage

        return {
            "schema_version": "1.0.0",
            "repo": {
                "provider": self.provider,
                "org": self.org,
                "repo_name": self.repo_name,
                "repo_url": self.repo_url,
                "default_branch": self.default_branch,
            },
            "scan": {
                "scanned_branch": self.scanned_branch,
                "scan_id": self.scan_id,
                "scan_timestamp": self.scan_timestamp,
                "current_commit_hash": self.current_commit_hash,
                "scanner_version": _get_scanner_version(),
            },
            "detected": {
                "owner": {
                    "detected": self.owner_detected,
                    "name": self.detected_owner_name,
                    "email": self.detected_owner_email,
                },
                "signals": signals,
                "dependencies": {
                    "dependency_files": self.dependency_files,
                    "ai_dependencies": ai_dependencies,
                },
                "agents": {
                    "counts": self.agent_counts,
                    "counts_unique": self.agent_counts_unique,
                    "instances": self.agent_instances,
                    "unique": self.agent_unique,
                    "frameworks": {
                        "main_framework": self.main_framework,
                        "supporting_infrastructure": self.supporting_infrastructure,
                        "framework_scores": self.framework_scores,
                        "multi_framework": self.multi_framework,
                    },
                    "agentic_imports": self.agentic_imports,
                },
                "parse_errors": self.parse_errors,
            },
        }
