# What: Postpone evaluation of type annotations.
# Why: Helps with forward references and keeps import-time behavior predictable.
from __future__ import annotations

# What: Import dataclass helper.
# Why: Used to define a compact config object with defaults.
from dataclasses import dataclass
# What: Import datetime utilities.
# Why: Circuit breaker needs time-based recovery logic.
from datetime import datetime, timedelta
# What: Import Enum base type.
# Why: Models explicit finite circuit states.
from enum import Enum
# What: Import callable and generic type variable support.
# Why: Keeps `call()` type-safe for wrapped function return values.
from typing import Callable, TypeVar

# What: Generic return type placeholder for wrapped function calls.
# Why: Preserves original function return type through circuit wrapper.
T = TypeVar("T")


# What: Enumerates valid circuit states.
# Why: Makes state transitions explicit and less error-prone than raw strings.
class CircuitState(str, Enum):
    # What: Normal operating mode.
    # Why: Calls are allowed and failure counters are tracked.
    CLOSED = "closed"
    # What: Failure-protection mode.
    # Why: Calls are blocked until recovery timeout passes.
    OPEN = "open"
    # What: Probe mode after timeout.
    # Why: Allows limited trial calls to test dependency recovery.
    HALF_OPEN = "half_open"


# What: Domain-specific exception for open-circuit blocks.
# Why: Callers can distinguish circuit-policy failures from dependency failures.
class CircuitBreakerOpenError(RuntimeError):
    """Raised when a call is blocked due to an open circuit."""


# What: Configuration container for circuit breaker behavior.
# Why: Makes thresholds/timings explicit and easy to tune per dependency.
@dataclass
class CircuitBreakerConfig:
    # What: Number of consecutive failures before opening circuit.
    # Why: Prevents hammering unstable dependencies.
    failure_threshold: int = 5
    # What: Seconds to wait before trying half-open probe.
    # Why: Gives dependency time to recover.
    recovery_timeout_seconds: int = 30
    # What: Max probe calls allowed while half-open.
    # Why: Limits risk during recovery testing.
    half_open_max_calls: int = 1


# What: Reusable circuit breaker implementation.
# Why: Encapsulates resilience policy around external function calls.
class CircuitBreaker:
    # What: Initialize breaker with name and optional config.
    # Why: Name improves observability; config controls behavior.
    def __init__(self, name: str, config: CircuitBreakerConfig | None = None) -> None:
        # What: Human-readable breaker identifier.
        # Why: Included in raised messages and logs.
        self.name = name
        # What: Use provided config or default config.
        # Why: Enables sane defaults with optional customization.
        self.config = config or CircuitBreakerConfig()
        # What: Start in closed state.
        # Why: Fresh dependencies should accept traffic initially.
        self.state = CircuitState.CLOSED
        # What: Track consecutive failures.
        # Why: Drives transition from closed to open.
        self.failure_count = 0
        # What: Timestamp of most recent failure.
        # Why: Used to determine when open can move to half-open.
        self.last_failure_at: datetime | None = None
        # What: Counter of probe calls in half-open state.
        # Why: Enforces configured half-open trial capacity.
        self.half_open_calls = 0

    # What: Check whether open circuit can transition to half-open.
    # Why: Implements time-based recovery gate.
    def _should_transition_to_half_open(self) -> bool:
        # What: Transition allowed only from OPEN with known failure time.
        # Why: Prevents invalid state transitions.
        if self.state != CircuitState.OPEN or self.last_failure_at is None:
            return False

        # What: Compute earliest recovery timestamp.
        # Why: Determines when probe traffic may resume.
        recovery_time = self.last_failure_at + timedelta(seconds=self.config.recovery_timeout_seconds)
        # What: Compare current UTC time against recovery threshold.
        # Why: Return True only when timeout window has elapsed.
        return datetime.utcnow() >= recovery_time

    # What: Success handler.
    # Why: Resets breaker to healthy baseline after successful call.
    def _on_success(self) -> None:
        # What: Move breaker to closed state.
        # Why: Dependency appears healthy again.
        self.state = CircuitState.CLOSED
        # What: Reset failure count.
        # Why: Past failures no longer relevant after success.
        self.failure_count = 0
        # What: Clear last failure timestamp.
        # Why: Not needed while healthy.
        self.last_failure_at = None
        # What: Reset half-open probe counter.
        # Why: Future half-open phases should start fresh.
        self.half_open_calls = 0

    # What: Failure handler.
    # Why: Tracks failure progression and opens circuit at threshold.
    def _on_failure(self) -> None:
        # What: Increment consecutive failures.
        # Why: Used for threshold comparison.
        self.failure_count += 1
        # What: Record failure timestamp.
        # Why: Starts/refreshes open-state recovery timer.
        self.last_failure_at = datetime.utcnow()

        # What: Open circuit after reaching configured failure threshold.
        # Why: Stop forwarding calls to failing dependency.
        if self.failure_count >= self.config.failure_threshold:
            self.state = CircuitState.OPEN

    # What: Execute a dependency call under circuit breaker policy.
    # Why: Central method that enforces state transitions and failure protection.
    def call(self, func: Callable[..., T], *args, **kwargs) -> T:
        # What: Handle OPEN state.
        # Why: Either block immediately or transition to half-open if timeout elapsed.
        if self.state == CircuitState.OPEN:
            if not self._should_transition_to_half_open():
                raise CircuitBreakerOpenError(f"Circuit '{self.name}' is OPEN.")

            # What: Enter half-open trial state.
            # Why: Allow limited probes to test recovery.
            self.state = CircuitState.HALF_OPEN
            # What: Reset trial call counter for this half-open cycle.
            # Why: Enforces max probe calls correctly.
            self.half_open_calls = 0

        # What: Handle HALF_OPEN trial capacity.
        # Why: Avoid flooding dependency during recovery probes.
        if self.state == CircuitState.HALF_OPEN:
            if self.half_open_calls >= self.config.half_open_max_calls:
                raise CircuitBreakerOpenError(f"Circuit '{self.name}' is HALF_OPEN and trial capacity is exhausted.")

            # What: Count this probe attempt.
            # Why: Enforces half-open limit.
            self.half_open_calls += 1

        # What: Execute wrapped function.
        # Why: Actual dependency call occurs here.
        try:
            result = func(*args, **kwargs)
            # What: Mark success on completion.
            # Why: Healthy call should close/reset breaker.
            self._on_success()
            # What: Return original function result.
            # Why: Preserve wrapped call contract.
            return result
        # What: Catch all dependency errors.
        # Why: Any failure should update breaker state before propagating.
        except Exception:
            self._on_failure()
            raise