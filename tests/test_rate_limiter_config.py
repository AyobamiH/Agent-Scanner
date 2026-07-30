"""Tests for rate limiter configuration and GitHub API throttling."""

import time
from unittest.mock import MagicMock

import pytest

from src.github.client import GitHubClient
from src.utils.rate_limiter import (
    CircuitBreaker,
    CircuitBreakerOpenError,
    RateLimitAwareSleeper,
    TokenBucket,
    rate_limited_context,
)


def test_githubclient_sleeper_config_from_env(monkeypatch, tmp_path):  # NOSONAR S2325
    """GitHubClient respects rate limiter configuration from environment variables."""
    monkeypatch.setenv("GITHUB_TOKEN", "dummy-token-for-tests")
    monkeypatch.setenv("GITHUB_RATE_LIMIT_RECENT_WINDOW", "5")
    monkeypatch.setenv("GITHUB_RATE_LIMIT_RECENT_THRESHOLD", "42.0")

    client = GitHubClient()

    assert isinstance(client._sleep_helper, RateLimitAwareSleeper)
    assert client._sleep_helper._recent_sleep_times.maxlen == 5
    assert client._sleep_helper._recent_threshold_seconds == pytest.approx(42.0)
    assert client._sleep_helper.recent_total == pytest.approx(0.0)


def test_token_bucket_initialisation():
    """TokenBucket initialises with correct capacity and refill rate."""
    bucket = TokenBucket(capacity=10.0, refill_rate=2.0)

    assert bucket.capacity == pytest.approx(10.0)
    assert bucket.refill_rate == pytest.approx(2.0)
    assert bucket.tokens == pytest.approx(10.0)


def test_token_bucket_consume_success():
    """TokenBucket consume succeeds when tokens are available."""
    bucket = TokenBucket(capacity=10.0, refill_rate=2.0)

    result = bucket.consume(tokens=5.0)

    assert result is True
    assert bucket.tokens == pytest.approx(5.0)


def test_token_bucket_consume_timeout():
    """TokenBucket consume returns False when timeout is exceeded."""
    bucket = TokenBucket(capacity=1.0, refill_rate=0.1)
    bucket.tokens = 0.0

    result = bucket.consume(tokens=10.0, timeout=0.05)

    assert result is False


def test_token_bucket_available():
    """TokenBucket available returns current token count."""
    bucket = TokenBucket(capacity=10.0, refill_rate=2.0)

    available = bucket.available()

    assert available == pytest.approx(10.0)


def test_token_bucket_refill():
    """TokenBucket refills tokens over time."""
    bucket = TokenBucket(capacity=10.0, refill_rate=10.0)
    bucket.tokens = 5.0
    bucket.last_refill = time.time() - 0.5

    bucket._refill()

    assert bucket.tokens >= 9.0


def test_circuit_breaker_initialisation():
    """CircuitBreaker initialises with correct thresholds."""
    breaker = CircuitBreaker(failure_threshold=3, recovery_timeout=30.0, success_threshold=2)

    assert breaker.failure_threshold == 3
    assert breaker.recovery_timeout == pytest.approx(30.0)
    assert breaker.success_threshold == 2
    assert breaker.state == CircuitBreaker.STATE_CLOSED


def test_circuit_breaker_opens_after_threshold():
    """CircuitBreaker opens after reaching failure threshold."""
    breaker = CircuitBreaker(failure_threshold=3)

    for _ in range(3):
        breaker.record_failure()

    assert breaker.state == CircuitBreaker.STATE_OPEN


def test_circuit_breaker_half_open_transition():
    """CircuitBreaker transitions to half-open after recovery timeout."""
    breaker = CircuitBreaker(failure_threshold=1, recovery_timeout=0.1)

    breaker.record_failure()
    assert breaker.state == CircuitBreaker.STATE_OPEN

    time.sleep(0.15)
    is_available = breaker.is_available()

    assert is_available is True
    assert breaker.state == CircuitBreaker.STATE_HALF_OPEN


def test_circuit_breaker_closes_after_success_threshold():
    """CircuitBreaker closes after enough successes in half-open state."""
    breaker = CircuitBreaker(failure_threshold=1, recovery_timeout=0.1, success_threshold=2)

    breaker.record_failure()
    time.sleep(0.15)
    breaker.is_available()

    breaker.record_success()
    breaker.record_success()

    assert breaker.state == CircuitBreaker.STATE_CLOSED


def test_circuit_breaker_reopens_on_failure_in_half_open():
    """CircuitBreaker reopens on failure in half-open state."""
    breaker = CircuitBreaker(failure_threshold=1, recovery_timeout=0.1)

    breaker.record_failure()
    time.sleep(0.15)
    breaker.is_available()

    breaker.record_failure()

    assert breaker.state == CircuitBreaker.STATE_OPEN


def test_circuit_breaker_call_success():
    """CircuitBreaker call executes function and records success."""
    breaker = CircuitBreaker()

    def test_function(x, y):
        return x + y

    result = breaker.call(test_function, 2, 3)

    assert result == 5


def test_circuit_breaker_call_failure():
    """CircuitBreaker call records failure when function raises exception."""
    breaker = CircuitBreaker(failure_threshold=2)

    def failing_function():
        raise ValueError("Test error")

    with pytest.raises(ValueError):
        breaker.call(failing_function)

    assert breaker.failure_count == 1


def test_circuit_breaker_call_raises_when_open():
    """CircuitBreaker call raises CircuitBreakerOpenError when circuit is open."""
    breaker = CircuitBreaker(failure_threshold=1)

    breaker.record_failure()

    with pytest.raises(CircuitBreakerOpenError):
        breaker.call(lambda: None)


def test_circuit_breaker_resets_failure_count_on_success_when_closed():
    """CircuitBreaker resets failure count on success when in closed state."""
    breaker = CircuitBreaker(failure_threshold=5)

    breaker.record_failure()
    breaker.record_failure()
    assert breaker.failure_count == 2

    breaker.record_success()

    assert breaker.failure_count == 0


def test_rate_limit_aware_sleeper_initialisation():
    """RateLimitAwareSleeper initialises with correct configuration."""
    sleeper = RateLimitAwareSleeper(use_jitter=True, recent_window=5, recent_threshold_seconds=100.0)

    assert sleeper.use_jitter is True
    assert sleeper._recent_sleep_times.maxlen == 5
    assert sleeper._recent_threshold_seconds == pytest.approx(100.0)


def test_rate_limit_aware_sleeper_calculate_sleep_time_retry_after():
    """RateLimitAwareSleeper respects Retry-After header."""
    sleeper = RateLimitAwareSleeper()

    sleep_time = sleeper.calculate_sleep_time(
        status_code=429, headers={"Retry-After": "10"}, attempt=0, base_backoff=1.0, backoff_multiplier=2.0
    )

    assert sleep_time == pytest.approx(10.0)


def test_rate_limit_aware_sleeper_calculate_sleep_time_rate_limit_reset():
    """RateLimitAwareSleeper respects X-RateLimit-Reset header."""
    sleeper = RateLimitAwareSleeper(use_jitter=False)
    future_time = time.time() + 15.0

    sleep_time = sleeper.calculate_sleep_time(
        status_code=429,
        headers={"X-RateLimit-Reset": str(future_time)},
        attempt=0,
        base_backoff=1.0,
        backoff_multiplier=2.0,
    )

    assert sleep_time >= 14.0
    assert sleep_time <= 16.0


def test_rate_limit_aware_sleeper_calculate_sleep_time_exponential_backoff():
    """RateLimitAwareSleeper uses exponential backoff when no headers present."""
    sleeper = RateLimitAwareSleeper(use_jitter=False)

    sleep_time = sleeper.calculate_sleep_time(
        status_code=429, headers={}, attempt=2, base_backoff=1.0, backoff_multiplier=2.0
    )

    assert sleep_time == pytest.approx(4.0)


def test_rate_limit_aware_sleeper_calculate_sleep_time_with_jitter():
    """RateLimitAwareSleeper adds jitter when enabled."""
    sleeper = RateLimitAwareSleeper(use_jitter=True)

    sleep_time = sleeper.calculate_sleep_time(
        status_code=429, headers={}, attempt=1, base_backoff=2.0, backoff_multiplier=2.0
    )

    assert sleep_time >= 4.0
    assert sleep_time <= 4.5


def test_rate_limit_aware_sleeper_recent_total():
    """RateLimitAwareSleeper tracks recent total sleep time."""
    sleeper = RateLimitAwareSleeper(recent_window=3)

    sleeper._recent_sleep_times.append(5.0)
    sleeper._recent_sleep_times.append(10.0)
    sleeper._recent_sleep_times.append(3.0)

    assert sleeper.recent_total == pytest.approx(18.0)


def test_rate_limit_aware_sleeper_sleep_and_log(caplog, monkeypatch):  # NOSONAR S2325
    """RateLimitAwareSleeper logs sleep events and tracks totals."""
    sleeper = RateLimitAwareSleeper(recent_window=2, recent_threshold_seconds=5.0)
    sleep_mock = MagicMock()
    monkeypatch.setattr("time.sleep", sleep_mock)

    sleeper.sleep_and_log(2.0, "test_reason", "http://test.url")  # NOSONAR S5332 - test URL only

    assert sleep_mock.called
    assert sleeper.recent_total == pytest.approx(2.0)
    assert "Rate limit sleep" in caplog.text


def test_rate_limit_aware_sleeper_sleep_and_log_threshold_exceeded(caplog, monkeypatch):  # NOSONAR S2325
    """RateLimitAwareSleeper logs error when threshold exceeded."""
    sleeper = RateLimitAwareSleeper(recent_window=2, recent_threshold_seconds=5.0)
    sleep_mock = MagicMock()
    monkeypatch.setattr("time.sleep", sleep_mock)

    sleeper.sleep_and_log(3.0, "reason1")
    sleeper.sleep_and_log(3.0, "reason2")

    assert "exceeded" in caplog.text


def test_rate_limited_context_success():
    """rate_limited_context allows execution when tokens available."""
    bucket = TokenBucket(capacity=10.0, refill_rate=1.0)

    with rate_limited_context(bucket):
        pass  # NOSONAR S108 - intentional empty block, testing context side effects

    assert bucket.tokens == pytest.approx(9.0)


def test_rate_limited_context_with_circuit_breaker():
    """rate_limited_context records success in circuit breaker."""
    bucket = TokenBucket(capacity=10.0, refill_rate=1.0)
    breaker = CircuitBreaker()

    with rate_limited_context(bucket, breaker):
        pass  # NOSONAR S108 - intentional empty block, testing context side effects

    assert breaker.failure_count == 0


def test_rate_limited_context_raises_when_circuit_open():
    """rate_limited_context raises CircuitBreakerOpenError when circuit is open."""
    bucket = TokenBucket(capacity=10.0, refill_rate=1.0)
    breaker = CircuitBreaker(failure_threshold=1)
    breaker.record_failure()

    with pytest.raises(CircuitBreakerOpenError):
        with rate_limited_context(bucket, breaker):
            pass  # NOSONAR S108 - intentional empty block, testing exception on context entry


def test_rate_limited_context_timeout_raises_runtime_error(monkeypatch):  # NOSONAR S2325
    """rate_limited_context raises RuntimeError when token acquisition times out."""
    bucket = TokenBucket(capacity=1.0, refill_rate=0.01)

    def mock_consume(tokens=1.0, timeout=30.0):
        return False

    monkeypatch.setattr(bucket, "consume", mock_consume)

    with pytest.raises(RuntimeError):
        with rate_limited_context(bucket):
            pass  # NOSONAR S108 - intentional empty block, testing exception on context entry


def test_rate_limited_context_records_failure_on_exception():
    """rate_limited_context records failure in circuit breaker when exception occurs."""
    bucket = TokenBucket(capacity=10.0, refill_rate=1.0)
    breaker = CircuitBreaker()

    with pytest.raises(ValueError):
        with rate_limited_context(bucket, breaker):
            raise ValueError("Test error")

    assert breaker.failure_count == 1
