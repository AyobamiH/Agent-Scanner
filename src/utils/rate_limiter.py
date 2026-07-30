"""
Rate limiting and circuit breaker utilities for API request throttling.
"""

from __future__ import annotations

import logging
import random
import time
from collections import deque
from collections.abc import Iterator
from contextlib import contextmanager
from threading import Lock
from typing import Any

logger = logging.getLogger(__name__)


class TokenBucket:
    """Token bucket rate limiter for throttling requests.

    Uses a token bucket algorithm to limit request rates. Tokens are added
    at a constant rate, and each request consumes a token. When no tokens
    are available, requests are blocked.
    """

    def __init__(self, capacity: float, refill_rate: float) -> None:
        """Initialise token bucket.

        Args:
            capacity: Maximum number of tokens the bucket can hold.
            refill_rate: Number of tokens added per second.
        """
        self.capacity = capacity
        self.refill_rate = refill_rate
        self.tokens = capacity
        self.last_refill = time.time()
        self.lock = Lock()

    def _refill(self) -> None:
        """Refill tokens based on elapsed time."""
        now = time.time()
        elapsed = now - self.last_refill
        tokens_to_add = elapsed * self.refill_rate
        self.tokens = min(self.capacity, self.tokens + tokens_to_add)
        self.last_refill = now

    def consume(self, tokens: float = 1.0, timeout: float = 30.0) -> bool:
        """Attempt to consume tokens.

        Args:
            tokens: Number of tokens to consume.
            timeout: Maximum time to wait for tokens in seconds.

        Returns:
            True if tokens were consumed, False if timeout exceeded.
        """
        start_time = time.time()

        while True:
            with self.lock:
                self._refill()
                if self.tokens >= tokens:
                    self.tokens -= tokens
                    return True

            elapsed = time.time() - start_time
            if elapsed >= timeout:
                logger.warning("Token bucket timeout after %.1fs waiting for %.1f tokens", elapsed, tokens)
                return False

            sleep_time = min(0.1, timeout - elapsed)
            if sleep_time > 0:
                time.sleep(sleep_time)

    def available(self) -> float:
        """Get currently available tokens."""
        with self.lock:
            self._refill()
            return self.tokens


class CircuitBreaker:
    """Circuit breaker for failing fast when service is unhealthy.

    Opens circuit after threshold of failures, temporarily blocking requests.
    Supports half-open state to test if service has recovered.
    """

    STATE_CLOSED = "closed"
    STATE_OPEN = "open"
    STATE_HALF_OPEN = "half_open"

    def __init__(
        self,
        failure_threshold: int = 5,
        recovery_timeout: float = 60.0,
        success_threshold: int = 2,
    ) -> None:
        """Initialise circuit breaker.

        Args:
            failure_threshold: Number of failures before opening circuit.
            recovery_timeout: Seconds to wait before attempting recovery (half-open).
            success_threshold: Successes in half-open state needed to close circuit.
        """
        self.failure_threshold = failure_threshold
        self.recovery_timeout = recovery_timeout
        self.success_threshold = success_threshold

        self.state = self.STATE_CLOSED
        self.failure_count = 0
        self.success_count = 0
        self.last_failure_time = 0.0
        self.lock = Lock()

    def record_failure(self) -> None:
        """Record a failure."""
        with self.lock:
            self.failure_count += 1
            self.last_failure_time = time.time()

            if self.state == self.STATE_HALF_OPEN:
                self.state = self.STATE_OPEN
                self.success_count = 0
                logger.warning("Circuit breaker opened after failure in half-open state")
            elif self.failure_count >= self.failure_threshold and self.state == self.STATE_CLOSED:
                self.state = self.STATE_OPEN
                logger.warning("Circuit breaker opened: %d failures threshold reached", self.failure_threshold)

    def record_success(self) -> None:
        """Record a success."""
        with self.lock:
            if self.state == self.STATE_HALF_OPEN:
                self.success_count += 1
                if self.success_count >= self.success_threshold:
                    self.state = self.STATE_CLOSED
                    self.failure_count = 0
                    logger.info("Circuit breaker closed: recovered from failures")
            elif self.state == self.STATE_CLOSED:
                self.failure_count = 0

    def call(self, function: Any, *args: Any, **kwargs: Any) -> Any:
        """Execute function with circuit breaker protection.

        Args:
            function: Callable to execute.
            *args: Positional arguments.
            **kwargs: Keyword arguments.

        Returns:
            Function result if successful.

        Raises:
            CircuitBreakerOpenError: If circuit is open.
        """
        with self.lock:
            if self.state == self.STATE_OPEN:
                elapsed = time.time() - self.last_failure_time
                if elapsed >= self.recovery_timeout:
                    self.state = self.STATE_HALF_OPEN
                    self.success_count = 0
                    logger.info("Circuit breaker transitioning to half-open state")
                else:
                    raise CircuitBreakerOpenError(
                        f"Circuit breaker is open. Will retry in {self.recovery_timeout - elapsed:.1f}s"
                    )

        try:
            result = function(*args, **kwargs)
            self.record_success()
            return result
        except Exception:
            self.record_failure()
            raise

    def is_available(self) -> bool:
        """Check if circuit is available for requests."""
        with self.lock:
            if self.state in (self.STATE_CLOSED, self.STATE_HALF_OPEN):
                return True

            elapsed = time.time() - self.last_failure_time
            if elapsed >= self.recovery_timeout:
                self.state = self.STATE_HALF_OPEN
                self.success_count = 0
                return True
            return False


class CircuitBreakerOpenError(Exception):
    """Raised when the circuit breaker is open and requests are blocked."""

    pass


class RateLimitAwareSleeper:
    """Intelligently sleep based on rate limit headers.

    Respects Retry-After before X-RateLimit-Reset to minimise delays while avoiding burst requests.
    Falls back to exponential backoff with optional jitter when headers are absent or invalid.
    """

    def __init__(
        self,
        use_jitter: bool = True,
        recent_window: int = 10,
        recent_threshold_seconds: float = 120.0,
    ) -> None:
        """Initialise sleeper.

        Args:
            use_jitter: Whether to add random jitter to backoff times.
            recent_window: Number of recent sleep times to track.
            recent_threshold_seconds: Total seconds threshold to log an error.
        """
        self.use_jitter = use_jitter
        self._recent_sleep_times: deque[float] = deque(maxlen=recent_window)
        self._recent_threshold_seconds = recent_threshold_seconds

    def calculate_sleep_time(
        self, status_code: int, headers: dict[str, str], attempt: int, base_backoff: float, backoff_multiplier: float
    ) -> float:
        """Calculate sleep time based on response headers and attempt number.

        Args:
            status_code: HTTP status code (403, 429, etc.).
            headers: Response headers dictionary.
            attempt: Current attempt number (0-indexed).
            base_backoff: Base backoff time in seconds.
            backoff_multiplier: Multiplier for exponential backoff.

        Returns:
            Sleep time in seconds.
        """
        retry_after = headers.get("Retry-After") or headers.get("retry-after")
        if retry_after:
            try:
                return float(retry_after)
            except (ValueError, TypeError):
                logger.debug("Could not parse Retry-After header: %s", retry_after)

        reset_epoch = headers.get("X-RateLimit-Reset") or headers.get("x-ratelimit-reset")
        if reset_epoch:
            try:
                reset_time = float(reset_epoch)
                sleep_time = max(0.5, reset_time - time.time())
                return sleep_time
            except (ValueError, TypeError):
                logger.debug("Could not parse X-RateLimit-Reset header: %s", reset_epoch)

        sleep_time = base_backoff * (backoff_multiplier**attempt)

        if self.use_jitter:
            jitter = random.uniform(0, sleep_time * 0.1)
            sleep_time += jitter

        logger.debug(
            "Using exponential backoff of %.2fs for status %s on attempt %d",
            sleep_time,
            status_code,
            attempt,
        )

        return sleep_time

    def sleep_and_log(self, sleep_time: float, reason: str, url: str = "") -> None:
        """Sleep and log the delay.

        Args:
            sleep_time: Time to sleep in seconds.
            reason: Reason for sleeping (e.g., "rate_limited", "circuit_open").
            url: Optional URL for context in logs.
        """
        self._recent_sleep_times.append(sleep_time)

        total_recent = sum(self._recent_sleep_times)
        logger.warning(
            "Rate limit sleep: %.2fs (%s) for %s. Recent total: %.1fs",
            sleep_time,
            reason,
            url or "request",
            total_recent,
        )
        if total_recent > self._recent_threshold_seconds:
            logger.error(
                "Total rate limit sleep exceeded %.0f seconds in recent requests",
                self._recent_threshold_seconds,
            )

        time.sleep(sleep_time)

    @property
    def recent_total(self) -> float:
        """Return the sum of recent sleep times currently tracked."""
        return sum(self._recent_sleep_times)


@contextmanager
def rate_limited_context(token_bucket: TokenBucket, circuit_breaker: CircuitBreaker | None = None) -> Iterator[None]:
    """Context manager for rate-limited operations.

    Args:
        token_bucket: Token bucket limiter.
        circuit_breaker: Optional circuit breaker.

    Yields:
        None

    Raises:
        CircuitBreakerOpenError: If circuit breaker is open.
        RuntimeError: If token acquisition times out.
    """
    if circuit_breaker and not circuit_breaker.is_available():
        raise CircuitBreakerOpenError("Circuit breaker is open")

    if not token_bucket.consume(timeout=30.0):
        raise RuntimeError("Timeout waiting for rate limit tokens")

    try:
        yield
    except Exception:
        if circuit_breaker:
            circuit_breaker.record_failure()
        raise
    else:
        if circuit_breaker:
            circuit_breaker.record_success()
