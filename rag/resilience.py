import logging
import random
import threading
import time
from dataclasses import dataclass
from typing import Any, Callable, Dict, Optional, Sequence, Tuple, Type

from django.db import connections

logger = logging.getLogger(__name__)


class CircuitBreakerOpenError(RuntimeError):
    """Raised when a circuit breaker is open and calls should fail fast."""


@dataclass
class CircuitBreakerState:
    failures: int = 0
    last_failure_ts: Optional[float] = None
    open_until_ts: Optional[float] = None
    half_open_calls: int = 0
    half_open: bool = False


class CircuitBreaker:
    def __init__(
        self,
        name: str,
        failure_threshold: int = 5,
        recovery_timeout: float = 30.0,
        half_open_max_calls: int = 2,
    ) -> None:
        self.name = name
        self.failure_threshold = failure_threshold
        self.recovery_timeout = recovery_timeout
        self.half_open_max_calls = half_open_max_calls
        self._state = CircuitBreakerState()
        self._lock = threading.Lock()

    def _now(self) -> float:
        return time.monotonic()

    def allow_call(self) -> None:
        with self._lock:
            if self._state.open_until_ts is not None:
                if self._state.open_until_ts > self._now():
                    raise CircuitBreakerOpenError(
                        f"Circuit breaker '{self.name}' is open; skipping call."
                    )
                self._state.open_until_ts = None
                self._state.failures = 0
                self._state.half_open_calls = 0
                self._state.half_open = True

            if self._state.half_open:
                if self._state.half_open_calls >= self.half_open_max_calls:
                    raise CircuitBreakerOpenError(
                        f"Circuit breaker '{self.name}' is half-open and at capacity."
                    )
                self._state.half_open_calls += 1
                return

            return

    def record_success(self) -> None:
        with self._lock:
            self._state.failures = 0
            self._state.last_failure_ts = None
            self._state.open_until_ts = None
            self._state.half_open_calls = 0
            self._state.half_open = False

    def record_failure(self) -> None:
        with self._lock:
            if self._state.half_open:
                self._state.open_until_ts = self._now() + self.recovery_timeout
                self._state.last_failure_ts = self._now()
                self._state.half_open_calls = 0
                self._state.half_open = False
                logger.warning(
                    "Circuit breaker '%s' re-opened for %.1fs after half-open failure.",
                    self.name,
                    self.recovery_timeout,
                )
                return
            self._state.failures += 1
            self._state.last_failure_ts = self._now()
            if self._state.failures >= self.failure_threshold:
                self._state.open_until_ts = self._state.last_failure_ts + self.recovery_timeout
                self._state.half_open = False
                logger.warning(
                    "Circuit breaker '%s' opened for %.1fs after %s failures.",
                    self.name,
                    self.recovery_timeout,
                    self._state.failures,
                )


_BREAKERS: Dict[str, CircuitBreaker] = {}


def get_circuit_breaker(
    name: str,
    failure_threshold: int = 5,
    recovery_timeout: float = 30.0,
    half_open_max_calls: int = 2,
) -> CircuitBreaker:
    breaker = _BREAKERS.get(name)
    if breaker is None:
        breaker = CircuitBreaker(
            name=name,
            failure_threshold=failure_threshold,
            recovery_timeout=recovery_timeout,
            half_open_max_calls=half_open_max_calls,
        )
        _BREAKERS[name] = breaker
    return breaker


def close_connections_before_io(reason: str) -> None:
    logger.debug("Closing DB connections before %s.", reason)
    connections.close_all()


def call_with_resilience(
    func: Callable[[], Any],
    service: str,
    exceptions: Sequence[Type[BaseException]] = (Exception,),
    max_attempts: int = 3,
    backoff_base: float = 0.5,
    backoff_max: float = 8.0,
    jitter: float = 0.2,
    breaker_failure_threshold: int = 5,
    breaker_recovery_timeout: float = 30.0,
) -> Any:
    breaker = get_circuit_breaker(
        service,
        failure_threshold=breaker_failure_threshold,
        recovery_timeout=breaker_recovery_timeout,
    )

    attempt = 0
    while True:
        breaker.allow_call()
        try:
            result = func()
            breaker.record_success()
            return result
        except tuple(exceptions) as exc:
            breaker.record_failure()
            attempt += 1
            if attempt >= max_attempts:
                logger.error(
                    "Service '%s' failed after %s attempts: %s",
                    service,
                    attempt,
                    exc,
                )
                raise
            sleep_for = min(backoff_max, backoff_base * (2 ** (attempt - 1)))
            sleep_for += random.uniform(0, jitter)
            logger.warning(
                "Service '%s' attempt %s failed: %s. Retrying in %.2fs.",
                service,
                attempt,
                exc,
                sleep_for,
            )
            time.sleep(sleep_for)
