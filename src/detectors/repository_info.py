"""Repository information detection utilities.

Provides functions to detect repository owners from commit history and
build repository identity blocks for scan results.
"""

from __future__ import annotations

import logging
import os
import re
from collections import Counter
from typing import Any

from src.exceptions import RepositoryOwnerDetectionError

logger = logging.getLogger(__name__)


def _parse_allowed_owner_domains(allowed_domains: str | None) -> list[str]:
    """Parse allowed owner email domains from environment configuration.

    Args:
        allowed_domains: Raw comma-separated domain list from the environment.

    Returns:
        A list of validated lower-case domain strings.
    """
    if not allowed_domains:
        return []

    parsed_domains: list[str] = []
    for domain_value in allowed_domains.split(","):
        candidate_domain = domain_value.strip().lower()
        if not candidate_domain:
            continue
        if re.match(r"^[a-z0-9.-]+$", candidate_domain):
            parsed_domains.append(candidate_domain)
            continue

        logger.debug("Skipping malformed domain from ALLOWED_OWNER_EMAIL_DOMAINS: %s", candidate_domain)

    return parsed_domains


def _is_valid_owner_email(email: str, name: str, domain_whitelist: list[str], is_human: bool) -> bool:
    """Check if an email/name pair represents a valid repository owner.

    Args:
        email: Email address to validate.
        name: Committer/author name.
        domain_whitelist: List of allowed email domains.
        is_human: Whether the email appears to be from a human (not a bot).

    Returns:
        True if the email/name pair is valid for owner detection.
    """
    if not (email and name and is_human):
        return False

    if "@" not in email:
        logger.debug("Malformed email (missing @): %s", email)
        return False

    if not domain_whitelist:
        return "bot" not in name.lower()

    try:
        email_domain = email.split("@", 1)[1].lower()
        if not email_domain:
            logger.debug("Malformed email (empty domain): %s", email)
            return False
        domain_matches = any(
            email_domain == domain or email_domain.endswith("." + domain) for domain in domain_whitelist
        )
    except (IndexError, ValueError):
        logger.debug("Failed to parse email domain from: %s", email)
        return False

    return domain_matches and "bot" not in name.lower()


def _is_human_email(email: str) -> bool:
    """Check if an email appears to be from a human (not automated).

    Args:
        email: Email address to check.

    Returns:
        True if the email appears to be from a human user.
    """
    if not email:
        return False
    email_lower = email.lower()
    return not ("noreply" in email_lower or "bot" in email_lower)


def _extract_fuller_name(name: str, email: str) -> str:
    """Extract a fuller name from email if the provided name is incomplete.

    If the name is only one word (likely a surname), try to extract the first name
    from the email local part (before @ symbol).

    Args:
        name: Committer/author name from commit.
        email: Email address to extract name parts from.

    Returns:
        Fuller name with first and last name if possible, otherwise original name.
    """
    if not name or not email:
        return name

    if " " in name.strip():
        return name

    try:
        local_part = email.split("@")[0]
        if not local_part:
            return name

        parts = re.split(r"[.\-_]", local_part)
        if len(parts) >= 2:
            first_part = parts[0].capitalize()
            last_part = parts[-1].capitalize()
            fuller_name = f"{first_part} {last_part}"
            if fuller_name.lower() != name.lower():
                return fuller_name
    except (IndexError, ValueError, AttributeError):
        pass

    return name


def detect_likely_owner(repository_full_name: str, service: Any, branch: str | None = None) -> dict[str, str | None]:
    """Detect the most likely owner of a repository from commit history.

    Fetches up to 100 commits and counts (name, email) pairs, filtering for
    human emails from whitelisted domains (if configured). Returns the most
    frequently occurring valid pair.

    Args:
        repository_full_name: Repository in owner/repository format.
        service: GitHub client instance with fetch_all_commits method.
        branch: Optional branch name to fetch commits from. If None, uses default branch.

    Returns:
        Dictionary with detectedOwnerName and detectedOwnerEmail keys.

    Raises:
        RepositoryOwnerDetectionError: If repository input is invalid, service lacks commit fetching, or commit
        retrieval fails.
    """
    if not repository_full_name or not repository_full_name.strip():
        raise RepositoryOwnerDetectionError("Repository full name cannot be empty")

    fetch_commits = getattr(service, "fetch_all_commits", None)
    if not callable(fetch_commits):
        raise RepositoryOwnerDetectionError("Service must implement fetch_all_commits")

    allowed_domains_env = os.environ.get("ALLOWED_OWNER_EMAIL_DOMAINS")
    domain_whitelist = _parse_allowed_owner_domains(allowed_domains_env)

    owner_pair_counter: Counter[tuple[str, str]] = Counter()

    try:
        commits = fetch_commits(repository_full_name, max_commits=100, branch=branch)
    except Exception as exc:
        logger.warning("Failed to fetch commits for %s: %s", repository_full_name, exc)
        raise RepositoryOwnerDetectionError(f"Failed to fetch commits for {repository_full_name}") from exc

    for commit in commits:
        committer_email = commit.get("committer_email", "").strip() or commit.get("email", "").strip()
        author_email = commit.get("author_email", "").strip() or commit.get("email", "").strip()

        name_email_pairs = [
            (commit.get("committer", "").strip(), committer_email),
            (commit.get("author", "").strip(), author_email),
        ]

        for name, email in name_email_pairs:
            if name and email:
                is_human = _is_human_email(email)
                if _is_valid_owner_email(email, name, domain_whitelist, is_human):
                    fuller_name = _extract_fuller_name(name, email)
                    owner_pair_counter[(fuller_name, email)] += 1

    if owner_pair_counter:
        likely_name, likely_email = owner_pair_counter.most_common(1)[0][0]
        return {
            "detectedOwnerName": likely_name,
            "detectedOwnerEmail": likely_email,
        }

    return {
        "detectedOwnerName": None,
        "detectedOwnerEmail": None,
    }


def build_repo_identity(
    repository_metadata: dict[str, Any], service: Any
) -> tuple[dict[str, str | None | bool], dict[str, str | None]]:
    """Build repository identity block and detect owner information.

    Args:
        repository_metadata: Repository metadata dictionary from GitHub API.
        service: GitHub client instance.

    Returns:
        Tuple of (identity_dict, owner_dict).
    """
    full_name = repository_metadata.get("full_name", "")
    html_url = repository_metadata.get("html_url", "")
    default_branch, head_commit_sha = service.get_repo_branch_and_head(full_name.split("/")[-1])

    organisation_name = full_name.split("/")[0] if "/" in full_name else None
    repository_name = full_name.split("/")[1] if "/" in full_name else full_name

    owner_guess: dict[str, str | None] = {}
    if full_name:
        try:
            owner_guess = detect_likely_owner(full_name, service)
        except RepositoryOwnerDetectionError as e:
            logger.debug("Owner detection failed for %s, proceeding with empty owner: %s", full_name, e)

    html_base = ""
    if html_url:
        if "/api/v3/" in html_url:
            html_base = html_url.replace("/api/v3/", "/")
        else:
            html_base = html_url

    identity: dict[str, str | None | bool] = {
        "provider": None,
        "org": organisation_name,
        "repo": repository_name,
        "repoUrl": html_base,
        "defaultBranch": default_branch,
        "currentCommitHash": head_commit_sha,
        "releaseTag": None,
        "ownerDetected": bool(owner_guess.get("detectedOwnerName") or owner_guess.get("detectedOwnerEmail")),
        "detectedOwnerName": owner_guess.get("detectedOwnerName"),
        "detectedOwnerEmail": owner_guess.get("detectedOwnerEmail"),
    }

    return identity, owner_guess
