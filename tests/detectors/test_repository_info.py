"""Tests for repository information detection utilities."""

import logging
import os
from unittest.mock import MagicMock, patch

import pytest

from src.detectors.repository_info import (
    _is_human_email,
    _is_valid_owner_email,
    build_repo_identity,
    detect_likely_owner,
)
from src.exceptions import RepositoryOwnerDetectionError


@pytest.mark.parametrize(
    "email,name,domain_whitelist,is_human,expected",
    [
        ("user@example.com", "John Doe", [], True, True),
        ("bot@example.com", "Bot User", [], True, False),
        ("user@company.com", "Jane Smith", ["company.com"], True, True),
        ("user@other.com", "Jane Smith", ["company.com"], True, False),
        ("", "John Doe", [], True, False),
        ("user@example.com", "", [], True, False),
        ("user@example.com", "John Doe", [], False, False),
        ("invalid-email", "John Doe", [], True, False),
    ],
)
def test_is_valid_owner_email(email: str, name: str, domain_whitelist: list, is_human: bool, expected: bool) -> None:
    """Validate owner email checks email format, name, domain whitelist, and human status."""
    result = _is_valid_owner_email(email, name, domain_whitelist, is_human)

    assert result is expected


def test_is_valid_owner_email_logs_malformed_email(caplog) -> None:
    """Valid owner email logs debug for malformed email."""
    caplog.set_level(logging.DEBUG)

    _is_valid_owner_email("invalid", "John Doe", [], True)

    assert "Malformed email" in caplog.text


def test_is_valid_owner_email_with_subdomain_match() -> None:
    """Valid owner email matches subdomains correctly."""
    result = _is_valid_owner_email("user@sub.company.com", "John Doe", ["company.com"], True)

    assert result is True


def test_is_valid_owner_email_with_exact_domain_match() -> None:
    """Valid owner email matches exact domain correctly."""
    result = _is_valid_owner_email("user@company.com", "John Doe", ["company.com"], True)

    assert result is True


def test_is_valid_owner_email_filters_bot_in_name() -> None:
    """Valid owner email filters out bot in name even with whitelist."""
    result = _is_valid_owner_email("user@company.com", "GitHub Bot", ["company.com"], True)

    assert result is False


@pytest.mark.parametrize(
    "email,expected",
    [
        ("john.doe@example.com", True),
        ("noreply@example.com", False),
        ("bot@example.com", False),
        ("", False),
        ("NOREPLY@example.com", False),
        ("valid.user@company.com", True),
    ],
)
def test_is_human_email(email: str, expected: bool) -> None:
    """Check if email appears to be from a human user."""
    result = _is_human_email(email)

    assert result is expected


def test_detect_likely_owner_with_single_committer() -> None:
    """Detect likely owner returns most common committer."""
    mock_service = MagicMock()
    mock_service.fetch_all_commits.return_value = [
        {"committer": "Alice", "committer_email": "alice@example.com"},
        {"committer": "Alice", "committer_email": "alice@example.com"},
        {"committer": "Alice", "committer_email": "alice@example.com"},
    ]

    result = detect_likely_owner("org/repo", mock_service)

    assert result["detectedOwnerName"] == "Alice"
    assert result["detectedOwnerEmail"] == "alice@example.com"


def test_detect_likely_owner_requires_repo_full_name() -> None:
    """Detect likely owner raises when repository name is empty."""
    mock_service = MagicMock()
    mock_service.fetch_all_commits.return_value = []

    with pytest.raises(RepositoryOwnerDetectionError):
        detect_likely_owner("", mock_service)


def test_detect_likely_owner_requires_fetch_method() -> None:
    """Detect likely owner raises when service lacks fetch_all_commits implementation."""

    class IncompleteService:
        """Service without the required fetch_all_commits method."""

    with pytest.raises(RepositoryOwnerDetectionError):
        detect_likely_owner("org/repo", IncompleteService())


def test_detect_likely_owner_with_multiple_committers() -> None:
    """Detect likely owner returns most frequent committer."""
    mock_service = MagicMock()
    mock_service.fetch_all_commits.return_value = [
        {"committer": "Alice", "committer_email": "alice@example.com"},
        {"committer": "Bob", "committer_email": "bob@example.com"},
        {"committer": "Alice", "committer_email": "alice@example.com"},
    ]

    result = detect_likely_owner("org/repo", mock_service)

    assert result["detectedOwnerName"] == "Alice"
    assert result["detectedOwnerEmail"] == "alice@example.com"


def test_detect_likely_owner_filters_bot_emails() -> None:
    """Detect likely owner filters out bot emails."""
    mock_service = MagicMock()
    mock_service.fetch_all_commits.return_value = [
        {"committer": "Bot", "committer_email": "bot@example.com"},
        {"committer": "Alice", "committer_email": "alice@example.com"},
        {"committer": "NoReply", "committer_email": "noreply@example.com"},
    ]

    result = detect_likely_owner("org/repo", mock_service)

    assert result["detectedOwnerName"] == "Alice"
    assert result["detectedOwnerEmail"] == "alice@example.com"


def test_detect_likely_owner_with_no_valid_commits() -> None:
    """Detect likely owner returns None values when no valid commits."""
    mock_service = MagicMock()
    mock_service.fetch_all_commits.return_value = [
        {"committer": "Bot", "committer_email": "bot@example.com"},
    ]

    result = detect_likely_owner("org/repo", mock_service)

    assert result["detectedOwnerName"] is None
    assert result["detectedOwnerEmail"] is None


def test_detect_likely_owner_with_empty_commits() -> None:
    """Detect likely owner returns None values with empty commit list."""
    mock_service = MagicMock()
    mock_service.fetch_all_commits.return_value = []

    result = detect_likely_owner("org/repo", mock_service)

    assert result["detectedOwnerName"] is None
    assert result["detectedOwnerEmail"] is None


def test_detect_likely_owner_with_domain_whitelist() -> None:
    """Detect likely owner respects ALLOWED_OWNER_EMAIL_DOMAINS environment variable."""
    mock_service = MagicMock()
    mock_service.fetch_all_commits.return_value = [
        {"committer": "Alice", "committer_email": "alice@company.com"},
        {"committer": "Bob", "committer_email": "bob@other.com"},
    ]

    with patch.dict(os.environ, {"ALLOWED_OWNER_EMAIL_DOMAINS": "company.com"}):
        result = detect_likely_owner("org/repo", mock_service)

    assert result["detectedOwnerName"] == "Alice"
    assert result["detectedOwnerEmail"] == "alice@company.com"


def test_detect_likely_owner_with_multiple_whitelisted_domains() -> None:
    """Detect likely owner handles comma-separated domain whitelist."""
    mock_service = MagicMock()
    mock_service.fetch_all_commits.return_value = [
        {"committer": "Alice", "committer_email": "alice@company1.com"},
        {"committer": "Bob", "committer_email": "bob@company2.com"},
    ]

    with patch.dict(os.environ, {"ALLOWED_OWNER_EMAIL_DOMAINS": "company1.com, company2.com"}):
        result = detect_likely_owner("org/repo", mock_service)

    assert result["detectedOwnerName"] in ["Alice", "Bob"]


def test_detect_likely_owner_filters_malformed_domains(caplog) -> None:
    """Detect likely owner logs and skips malformed domains in whitelist."""
    mock_service = MagicMock()
    mock_service.fetch_all_commits.return_value = [
        {"committer": "Alice", "committer_email": "alice@company.com"},
    ]

    caplog.set_level(logging.DEBUG)

    with patch.dict(os.environ, {"ALLOWED_OWNER_EMAIL_DOMAINS": "company.com, invalid domain!, good.com"}):
        detect_likely_owner("org/repo", mock_service)

    assert "Skipping malformed domain" in caplog.text


def test_detect_likely_owner_uses_author_fallback() -> None:
    """Detect likely owner uses author when committer not present."""
    mock_service = MagicMock()
    mock_service.fetch_all_commits.return_value = [
        {"author": "Alice", "author_email": "alice@example.com"},
    ]

    result = detect_likely_owner("org/repo", mock_service)

    assert result["detectedOwnerName"] == "Alice"
    assert result["detectedOwnerEmail"] == "alice@example.com"


def test_detect_likely_owner_with_branch_parameter() -> None:
    """Detect likely owner passes branch parameter to fetch commits."""
    mock_service = MagicMock()
    mock_service.fetch_all_commits.return_value = [
        {"committer": "Alice", "committer_email": "alice@example.com"},
    ]

    detect_likely_owner("org/repo", mock_service, branch="feature-branch")

    mock_service.fetch_all_commits.assert_called_once_with("org/repo", max_commits=100, branch="feature-branch")


def test_detect_likely_owner_raises_on_fetch_failure() -> None:
    """Detect likely owner raises RepositoryOwnerDetectionError on fetch failure."""
    mock_service = MagicMock()
    mock_service.fetch_all_commits.side_effect = Exception("API Error")

    with pytest.raises(RepositoryOwnerDetectionError) as exc:
        detect_likely_owner("org/repo", mock_service)

    assert "Failed to fetch commits" in str(exc.value)


def test_detect_likely_owner_logs_fetch_failure(caplog) -> None:
    """Detect likely owner logs warning on fetch failure."""
    mock_service = MagicMock()
    mock_service.fetch_all_commits.side_effect = Exception("API Error")

    caplog.set_level(logging.WARNING)

    with pytest.raises(RepositoryOwnerDetectionError):
        detect_likely_owner("org/repo", mock_service)

    assert "Failed to fetch commits" in caplog.text


def test_build_repo_identity_with_full_metadata() -> None:
    """Build repository identity with complete metadata."""
    repository_metadata = {
        "full_name": "org/repo",
        "html_url": "https://github.com/org/repo",
    }

    mock_service = MagicMock()
    mock_service.get_repo_branch_and_head.return_value = ("main", "abc123")
    mock_service.fetch_all_commits.return_value = [
        {"committer": "Alice", "committer_email": "alice@example.com"},
    ]

    identity, owner = build_repo_identity(repository_metadata, mock_service)

    assert identity["org"] == "org"
    assert identity["repo"] == "repo"
    assert identity["repoUrl"] == "https://github.com/org/repo"
    assert identity["defaultBranch"] == "main"
    assert identity["currentCommitHash"] == "abc123"
    assert identity["ownerDetected"] is True
    assert owner["detectedOwnerName"] == "Alice"


def test_build_repo_identity_without_owner_detection() -> None:
    """Build repository identity when owner detection fails."""
    repository_metadata = {
        "full_name": "org/repo",
        "html_url": "https://github.com/org/repo",
    }

    mock_service = MagicMock()
    mock_service.get_repo_branch_and_head.return_value = ("main", "xyz789")
    mock_service.fetch_all_commits.return_value = []

    identity, _ = build_repo_identity(repository_metadata, mock_service)

    assert identity["ownerDetected"] is False
    assert identity["detectedOwnerName"] is None
    assert identity["detectedOwnerEmail"] is None


def test_build_repo_identity_handles_ghe_url() -> None:
    """Build repository identity correctly handles GitHub Enterprise URLs."""
    repository_metadata = {
        "full_name": "org/repo",
        "html_url": "https://github.company.com/api/v3/repos/org/repo",
    }

    mock_service = MagicMock()
    mock_service.get_repo_branch_and_head.return_value = ("develop", "def456")
    mock_service.fetch_all_commits.return_value = []

    identity, _ = build_repo_identity(repository_metadata, mock_service)

    assert identity["repoUrl"] == "https://github.company.com/repos/org/repo"


def test_build_repo_identity_with_single_name_repo() -> None:
    """Build repository identity handles repo name without org."""
    repository_metadata = {
        "full_name": "standalone-repo",
        "html_url": "https://github.com/standalone-repo",
    }

    mock_service = MagicMock()
    mock_service.get_repo_branch_and_head.return_value = ("main", "ghi789")
    mock_service.fetch_all_commits.return_value = []

    identity, _ = build_repo_identity(repository_metadata, mock_service)

    assert identity["org"] is None
    assert identity["repo"] == "standalone-repo"


def test_build_repo_identity_logs_owner_detection_failure(caplog) -> None:
    """Build repository identity logs debug when owner detection fails."""
    repository_metadata = {
        "full_name": "org/repo",
        "html_url": "https://github.com/org/repo",
    }

    mock_service = MagicMock()
    mock_service.get_repo_branch_and_head.return_value = ("main", "jkl012")
    mock_service.fetch_all_commits.side_effect = Exception("Fetch failed")

    caplog.set_level(logging.DEBUG)

    _, _ = build_repo_identity(repository_metadata, mock_service)

    assert "Owner detection failed" in caplog.text


def test_build_repo_identity_sets_provider_none() -> None:
    """Build repository identity sets provider to None by default."""
    repository_metadata = {
        "full_name": "org/repo",
        "html_url": "https://github.com/org/repo",
    }

    mock_service = MagicMock()
    mock_service.get_repo_branch_and_head.return_value = ("main", "mno345")
    mock_service.fetch_all_commits.return_value = []

    identity, _ = build_repo_identity(repository_metadata, mock_service)

    assert identity["provider"] is None


def test_build_repo_identity_sets_release_tag_none() -> None:
    """Build repository identity sets releaseTag to None."""
    repository_metadata = {
        "full_name": "org/repo",
        "html_url": "https://github.com/org/repo",
    }

    mock_service = MagicMock()
    mock_service.get_repo_branch_and_head.return_value = ("main", "pqr678")
    mock_service.fetch_all_commits.return_value = []

    identity, _ = build_repo_identity(repository_metadata, mock_service)

    assert identity["releaseTag"] is None


def test_detect_likely_owner_handles_email_key_variants() -> None:
    """Detect likely owner checks multiple email key variants."""
    mock_service = MagicMock()
    mock_service.fetch_all_commits.return_value = [
        {"committer": "Alice", "email": "alice@example.com"},
    ]

    result = detect_likely_owner("org/repo", mock_service)

    assert result["detectedOwnerEmail"] == "alice@example.com"


def test_detect_likely_owner_strips_whitespace() -> None:
    """Detect likely owner strips whitespace from name and email."""
    mock_service = MagicMock()
    mock_service.fetch_all_commits.return_value = [
        {"committer": "  Alice  ", "committer_email": "  alice@example.com  "},
    ]

    result = detect_likely_owner("org/repo", mock_service)

    assert result["detectedOwnerName"] == "Alice"
    assert result["detectedOwnerEmail"] == "alice@example.com"


def test_parse_allowed_owner_domains_with_malformed_domain(caplog) -> None:
    """Parse allowed owner domains logs and skips malformed domain format."""
    from src.detectors.repository_info import _parse_allowed_owner_domains

    caplog.set_level(logging.DEBUG)

    result = _parse_allowed_owner_domains("valid.com, invalid@domain!, another.org")

    assert "valid.com" in result
    assert "another.org" in result
    assert "invalid@domain!" not in result
    assert "Skipping malformed domain" in caplog.text


def test_parse_allowed_owner_domains_empty_string() -> None:
    """Parse allowed owner domains returns empty list for empty string."""
    from src.detectors.repository_info import _parse_allowed_owner_domains

    result = _parse_allowed_owner_domains("")

    assert result == []


def test_parse_allowed_owner_domains_whitespace_only() -> None:
    """Parse allowed owner domains handles whitespace-only values."""
    from src.detectors.repository_info import _parse_allowed_owner_domains

    result = _parse_allowed_owner_domains("valid.com,   ,  another.org  ")

    assert result == ["valid.com", "another.org"]


def test_parse_allowed_owner_domains_case_insensitive() -> None:
    """Parse allowed owner domains converts to lowercase."""
    from src.detectors.repository_info import _parse_allowed_owner_domains

    result = _parse_allowed_owner_domains("EXAMPLE.COM, MixedCase.Org")

    assert result == ["example.com", "mixedcase.org"]


def test_is_valid_owner_email_with_empty_domain(caplog) -> None:
    """Valid owner email logs debug when domain parsing yields empty string."""
    caplog.set_level(logging.DEBUG)

    result = _is_valid_owner_email("user@", "John Doe", ["company.com"], True)

    assert result is False
    assert "Malformed email (empty domain)" in caplog.text


def test_is_valid_owner_email_with_subdomain_and_whitelist() -> None:
    """Valid owner email correctly validates subdomain against whitelist."""
    result = _is_valid_owner_email("user@sub.domain.company.com", "John Doe", ["company.com"], True)

    assert result is True


def test_is_valid_owner_email_with_numeric_domain() -> None:
    """Valid owner email accepts numeric domains."""
    result = _is_valid_owner_email("user@192.168.1.1", "John Doe", ["192.168.1.1"], True)  # NOSONAR S1313

    assert result is True


def test_is_valid_owner_email_with_hyphenated_domain() -> None:
    """Valid owner email accepts hyphenated domains."""
    result = _is_valid_owner_email("user@my-company.com", "John Doe", ["my-company.com"], True)

    assert result is True


def test_detect_likely_owner_with_mixed_valid_invalid_emails() -> None:
    """Detect likely owner handles mix of valid and invalid emails in commits."""
    mock_service = MagicMock()
    mock_service.fetch_all_commits.return_value = [
        {"committer": "Alice", "committer_email": "alice@example.com"},
        {"committer": "Bot", "committer_email": "bot@example.com"},
        {"committer": "Alice", "committer_email": "alice@example.com"},
        {"author": "NoReply", "author_email": "noreply@example.com"},
    ]

    result = detect_likely_owner("org/repo", mock_service)

    assert result["detectedOwnerName"] == "Alice"
    assert result["detectedOwnerEmail"] == "alice@example.com"


def test_detect_likely_owner_with_whitespace_handling() -> None:
    """Detect likely owner correctly handles whitespace in email fields."""
    mock_service = MagicMock()
    mock_service.fetch_all_commits.return_value = [
        {"committer": "Alice", "committer_email": "  alice@example.com  "},
        {"committer": "Alice", "committer_email": "alice@example.com"},
    ]

    result = detect_likely_owner("org/repo", mock_service)

    assert result["detectedOwnerName"] == "Alice"
    assert result["detectedOwnerEmail"] == "alice@example.com"


def test_build_repo_identity_with_empty_metadata() -> None:
    """Build repository identity handles empty metadata gracefully."""
    repository_metadata = {
        "full_name": "",
        "html_url": "",
    }

    mock_service = MagicMock()
    mock_service.get_repo_branch_and_head.return_value = ("main", None)
    mock_service.fetch_all_commits.return_value = []

    identity, _ = build_repo_identity(repository_metadata, mock_service)

    assert identity["org"] is None
    assert identity["repo"] == ""
    assert identity["repoUrl"] == ""
    assert identity["ownerDetected"] is False


def test_build_repo_identity_with_missing_html_url() -> None:
    """Build repository identity handles missing html_url field."""
    repository_metadata = {
        "full_name": "org/repo",
    }

    mock_service = MagicMock()
    mock_service.get_repo_branch_and_head.return_value = ("main", "abc123")
    mock_service.fetch_all_commits.return_value = []

    identity, _ = build_repo_identity(repository_metadata, mock_service)

    assert identity["repoUrl"] == ""
    assert identity["org"] == "org"
    assert identity["repo"] == "repo"
