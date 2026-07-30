"""Custom exceptions for the agent scanner.

Provides a hierarchy of exception types for different failure modes in the
scanning pipeline, from GitHub API errors to pattern matching and file processing
failures. All custom exceptions inherit from ScannerError for easy catching.
"""


class ScannerError(Exception):
    """Base class for scanner-related errors.

    Raised for general scanning failures that do not fit into more specific
    error categories. Also serves as the base class for all custom scanner
    exceptions, enabling catch-all error handling when needed.

    This exception indicates a failure in the scanning process that prevents
    the scanner from completing its analysis of a repository.
    """

    pass


class RepositoryOwnerDetectionError(Exception):
    """Raised when detecting repository owner from commit history fails."""

    pass


class GitHubClientError(ScannerError):
    """Raised when GitHub client encounters an error.

    This exception indicates failures in GitHub API communication, including:
    - Missing or invalid authentication tokens
    - Rate limit errors (403, 429 status codes)
    - Network connectivity issues
    - Repository or file not found (404)
    - API request timeouts
    - Invalid repository identifiers
    - File size limit violations

    Raised by GitHubClient methods when unable to fetch repository data.
    """

    pass


class PatternMatcherError(ScannerError):
    """Raised when pattern matcher fails.

    This exception indicates failures in keyword pattern matching operations,
    including:
    - Invalid or malformed regular expression patterns
    - Tokenisation failures for path or content analysis
    - Unexpected errors during scoring or matching operations

    Raised by PatternMatcher methods when unable to evaluate patterns against
    repository paths or file contents.
    """

    pass


class FileUtilsError(ScannerError):
    """Raised for file utilities related errors.

    This exception indicates failures in file processing utilities, including:
    - Invalid file path formats
    - Unsupported file types or extensions
    - File sampling or grouping failures
    - Path depth calculation errors

    Raised by utility functions in src.utils.file_utils when unable to process
    file metadata or perform file selection operations.
    """

    pass


class CacheError(ScannerError):
    """Raised for cache-related errors.

    This exception indicates failures in cache operations, including:
    - Unable to read or write cache files to disk
    - JSON serialisation or deserialisation failures
    - Invalid cache configuration (negative TTL, invalid path)
    - Cache corruption or format errors

    Raised by FileCache methods when cache persistence or retrieval operations
    fail. Note that cache operations are often best-effort, so this exception
    may be caught and logged rather than propagated.
    """

    pass


class ConfigurationError(ScannerError):
    """Raised when configuration or keywords cannot be loaded or parsed.

    This exception indicates failures in loading or validating configuration,
    including:
    - Missing required configuration files (keywords.json)
    - Malformed JSON in configuration files
    - Invalid or empty keyword lists
    - Missing required configuration sections
    - Type mismatches in configuration values

    Raised by load_keywords() and PatternMatcher.from_file() when unable to
    initialise pattern matching configuration. This is typically a fatal error
    requiring user intervention to fix the configuration file.
    """

    pass


class ContentMatchError(ScannerError):
    """Raised for errors encountered while matching content.

    This exception indicates failures during content analysis operations,
    including:
    - Errors in text encoding or decoding
    - Failures in content tokenisation
    - Regular expression matching errors on malformed input
    - AST parsing failures for non-Python files processed as Python

    Raised by pattern matching and agent detection code when content cannot
    be reliably analysed. Callers may fall back to simpler regex-based
    detection when this occurs.
    """

    pass


class SummaryWriteError(ScannerError):
    """Raised when writing summary files fails.

    This exception indicates failures in summary file generation, including:
    - Unable to write to the specified file path
    - JSON serialisation errors for scan results
    - Invalid or missing data in scan results
    - Insufficient permissions for output directory

    Raised by summary writing utilities when unable to persist scan results
    to disk. This is typically a non-fatal error that should be logged but
    not prevent the scanner from completing.
    """

    pass
