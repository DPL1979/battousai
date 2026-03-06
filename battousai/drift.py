"""
battousai/drift.py
==================
Agent Behavioral Drift Detection Module

Monitors agent behavior over time and detects when an agent's actions diverge
from its established baseline — a key safety mechanism for autonomous AI systems.

Classes
-------
BehaviorProfile
    Captures an agent's behavioral fingerprint over a window of observations.
BehaviorEvent
    A single behavioral observation recorded from an agent.
DriftScore
    Quantified drift measurement with per-dimension breakdown.
DriftDetector
    Main detection engine: builds baselines, computes drift scores.
DriftPolicy
    Enumeration of configurable response actions.
DriftPolicyConfig
    Maps severity levels to DriftPolicy actions.
DriftMonitor
    Continuous monitoring orchestrator that feeds events and fires alerts.
DriftAlert
    Record of a triggered alert event.
DriftReport
    Generates comprehensive drift analysis reports.

Math Utilities (internal)
-------------------------
_cosine_similarity
_z_score
_normalize_histogram
_chi_squared_divergence
"""

from __future__ import annotations

import math
import time
import threading
import statistics
from collections import defaultdict, deque
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Tuple

# ---------------------------------------------------------------------------
# Internal Math Utilities
# ---------------------------------------------------------------------------


def _cosine_similarity(a: Dict[str, int], b: Dict[str, int]) -> float:
    """Compute cosine similarity between two histogram dictionaries.

    Both dictionaries are treated as sparse vectors in the space of all
    keys that appear in either dictionary.  Returns a value in [0.0, 1.0].
    If either vector has zero magnitude the function returns 0.0 (no
    similarity can be established).

    Parameters
    ----------
    a:
        First histogram mapping label → count.
    b:
        Second histogram mapping label → count.

    Returns
    -------
    float
        Cosine similarity in [0.0, 1.0].
    """
    if not a or not b:
        return 0.0

    all_keys = set(a.keys()) | set(b.keys())
    dot = 0.0
    mag_a = 0.0
    mag_b = 0.0

    for k in all_keys:
        va = a.get(k, 0)
        vb = b.get(k, 0)
        dot += va * vb
        mag_a += va * va
        mag_b += vb * vb

    if mag_a == 0.0 or mag_b == 0.0:
        return 0.0

    return dot / (math.sqrt(mag_a) * math.sqrt(mag_b))


def _z_score(value: float, mean: float, stddev: float) -> float:
    """Compute the z-score of *value* relative to a normal distribution.

    Parameters
    ----------
    value:
        The observed value.
    mean:
        Mean of the reference distribution.
    stddev:
        Standard deviation of the reference distribution.

    Returns
    -------
    float
        The z-score.  Returns 0.0 when *stddev* is 0 and *value* equals
        *mean*; returns a large sentinel (10.0) when *stddev* is 0 but
        *value* differs from *mean*.
    """
    if stddev == 0.0:
        return 0.0 if value == mean else 10.0
    return (value - mean) / stddev


def _normalize_histogram(hist: Dict[str, int]) -> Dict[str, float]:
    """Convert a count histogram into a probability distribution.

    Parameters
    ----------
    hist:
        Histogram mapping label → integer count.

    Returns
    -------
    Dict[str, float]
        Mapping of label → probability (values sum to 1.0).  Returns an
        empty dict when *hist* is empty or all counts are zero.
    """
    total = sum(hist.values())
    if total == 0:
        return {}
    return {k: v / total for k, v in hist.items()}


def _chi_squared_divergence(
    observed: Dict[str, float], expected: Dict[str, float]
) -> float:
    """Compute the chi-squared divergence between two distributions.

    Uses the formula:  Σ (O - E)² / E  over the union of keys.
    Missing keys in *expected* are treated as having a small epsilon
    probability to avoid division by zero.

    Parameters
    ----------
    observed:
        Observed probability distribution (values should sum to ~1).
    expected:
        Expected probability distribution (values should sum to ~1).

    Returns
    -------
    float
        Chi-squared divergence ≥ 0.  Returns 0.0 if both dicts are empty.
    """
    if not observed and not expected:
        return 0.0

    epsilon = 1e-10
    all_keys = set(observed.keys()) | set(expected.keys())
    result = 0.0

    for k in all_keys:
        o = observed.get(k, 0.0)
        e = expected.get(k, epsilon)
        result += (o - e) ** 2 / e

    return result


# ---------------------------------------------------------------------------
# Data Classes
# ---------------------------------------------------------------------------


@dataclass
class BehaviorEvent:
    """A single behavioral observation recorded from an agent.

    Attributes
    ----------
    agent_id:
        Identifier of the agent that produced this event.
    tick:
        Logical clock tick at which the event occurred.
    event_type:
        Category of the event.  One of: ``"syscall"``, ``"resource_access"``,
        ``"message_sent"``, ``"tool_use"``, ``"memory_write"``,
        ``"memory_read"``, ``"error"``, ``"think_complete"``.
    detail:
        The specific label (syscall name, resource path, tool name, …).
    timestamp:
        Wall-clock time (``time.time()``) when the event was recorded.
    metadata:
        Optional free-form dictionary for additional context.
    """

    agent_id: str
    tick: int
    event_type: str
    detail: str
    timestamp: float = field(default_factory=time.time)
    metadata: Optional[Dict[str, Any]] = field(default=None)


@dataclass
class BehaviorProfile:
    """Captures an agent's behavioral fingerprint over an observation window.

    Attributes
    ----------
    agent_id:
        Identifier of the agent being profiled.
    window_size:
        Number of ticks that were used to build this profile.
    syscall_histogram:
        Counts of each syscall type observed.
    resource_access_patterns:
        Resource → access count mapping.
    message_type_distribution:
        IPC message type → usage count mapping.
    tool_usage_histogram:
        Tool name → invocation count mapping.
    memory_write_rate:
        Average memory-write operations per tick.
    memory_read_rate:
        Average memory-read operations per tick.
    error_rate:
        Average error events per tick.
    avg_think_duration_ms:
        Average ``think_complete`` event duration in milliseconds.
    total_observations:
        Total number of events recorded into this profile.
    created_at:
        Wall-clock timestamp when the profile was first created.
    updated_at:
        Wall-clock timestamp when the profile was last updated.
    """

    agent_id: str
    window_size: int = 100
    syscall_histogram: Dict[str, int] = field(default_factory=dict)
    resource_access_patterns: Dict[str, int] = field(default_factory=dict)
    message_type_distribution: Dict[str, int] = field(default_factory=dict)
    tool_usage_histogram: Dict[str, int] = field(default_factory=dict)
    memory_write_rate: float = 0.0
    memory_read_rate: float = 0.0
    error_rate: float = 0.0
    avg_think_duration_ms: float = 0.0
    total_observations: int = 0
    created_at: float = field(default_factory=time.time)
    updated_at: float = field(default_factory=time.time)

    def update(self, events: List[BehaviorEvent]) -> None:
        """Recompute all profile statistics from the supplied event list.

        This method replaces the profile's current statistics with values
        derived entirely from *events*.  It is intended for full profile
        rebuilds rather than incremental updates.

        Parameters
        ----------
        events:
            List of :class:`BehaviorEvent` objects for this agent.
        """
        self.syscall_histogram = {}
        self.resource_access_patterns = {}
        self.message_type_distribution = {}
        self.tool_usage_histogram = {}
        self.total_observations = len(events)
        self.updated_at = time.time()

        if not events:
            self.memory_write_rate = 0.0
            self.memory_read_rate = 0.0
            self.error_rate = 0.0
            self.avg_think_duration_ms = 0.0
            return

        # Determine tick span to compute rates
        ticks = {e.tick for e in events}
        tick_count = max(len(ticks), 1)

        memory_writes = 0
        memory_reads = 0
        errors = 0
        think_durations: List[float] = []

        for ev in events:
            et = ev.event_type
            if et == "syscall":
                self.syscall_histogram[ev.detail] = (
                    self.syscall_histogram.get(ev.detail, 0) + 1
                )
            elif et == "resource_access":
                self.resource_access_patterns[ev.detail] = (
                    self.resource_access_patterns.get(ev.detail, 0) + 1
                )
            elif et == "message_sent":
                self.message_type_distribution[ev.detail] = (
                    self.message_type_distribution.get(ev.detail, 0) + 1
                )
            elif et == "tool_use":
                self.tool_usage_histogram[ev.detail] = (
                    self.tool_usage_histogram.get(ev.detail, 0) + 1
                )
            elif et == "memory_write":
                memory_writes += 1
            elif et == "memory_read":
                memory_reads += 1
            elif et == "error":
                errors += 1
            elif et == "think_complete":
                if ev.metadata and "duration_ms" in ev.metadata:
                    think_durations.append(float(ev.metadata["duration_ms"]))

        self.memory_write_rate = memory_writes / tick_count
        self.memory_read_rate = memory_reads / tick_count
        self.error_rate = errors / tick_count
        self.avg_think_duration_ms = (
            statistics.mean(think_durations) if think_durations else 0.0
        )


@dataclass
class DriftScore:
    """Quantified drift measurement with per-dimension breakdown.

    Attributes
    ----------
    agent_id:
        Identifier of the agent being assessed.
    score:
        Composite drift score in [0.0, 1.0].  0.0 means no drift; 1.0
        means maximum possible drift.
    components:
        Per-dimension drift scores (each in [0.0, 1.0]).
    threshold:
        The configured detection threshold at time of computation.
    is_drifting:
        ``True`` when ``score > threshold``.
    severity:
        Human-readable severity label: ``"none"``, ``"low"``, ``"medium"``,
        ``"high"``, or ``"critical"``.
    explanation:
        Human-readable description of what drifted.
    tick:
        Logical clock tick at which the score was computed.
    timestamp:
        Wall-clock time when the score was computed.
    """

    agent_id: str
    score: float
    components: Dict[str, float]
    threshold: float
    is_drifting: bool
    severity: str
    explanation: str
    tick: int
    timestamp: float = field(default_factory=time.time)

    @staticmethod
    def classify_severity(score: float, threshold: float) -> str:
        """Classify a drift score into a severity label.

        Parameters
        ----------
        score:
            Composite drift score in [0.0, 1.0].
        threshold:
            Detection threshold.

        Returns
        -------
        str
            One of ``"none"``, ``"low"``, ``"medium"``, ``"high"``,
            ``"critical"``.
        """
        if score <= threshold:
            return "none"
        excess = score - threshold
        band = 1.0 - threshold  # total space above threshold
        if band <= 0:
            return "critical"
        ratio = excess / band
        if ratio < 0.25:
            return "low"
        elif ratio < 0.50:
            return "medium"
        elif ratio < 0.75:
            return "high"
        else:
            return "critical"


# ---------------------------------------------------------------------------
# DriftPolicy
# ---------------------------------------------------------------------------


class DriftPolicy:
    """Enumeration of configurable response actions when drift is detected.

    Attributes
    ----------
    MONITOR:
        Log drift events but take no further action.
    ALERT:
        Emit an alert event via registered callbacks.
    THROTTLE:
        Reduce the agent's scheduling priority.
    QUARANTINE:
        Suspend the agent pending human review.
    """

    MONITOR = "MONITOR"
    ALERT = "ALERT"
    THROTTLE = "THROTTLE"
    QUARANTINE = "QUARANTINE"

    # Ordered list from least to most severe
    _ORDERED: List[str] = [MONITOR, ALERT, THROTTLE, QUARANTINE]

    @classmethod
    def severity_for_action(cls, action: str) -> int:
        """Return an integer severity rank for a given policy action.

        Higher values indicate more severe responses.

        Parameters
        ----------
        action:
            One of the class constants (``MONITOR``, ``ALERT``, ``THROTTLE``,
            ``QUARANTINE``).

        Returns
        -------
        int
            0-based rank.  Raises :class:`ValueError` for unknown actions.
        """
        try:
            return cls._ORDERED.index(action)
        except ValueError:
            raise ValueError(f"Unknown DriftPolicy action: {action!r}")


@dataclass
class DriftPolicyConfig:
    """Maps severity levels to :class:`DriftPolicy` actions.

    Attributes
    ----------
    low:
        Policy applied when severity is ``"low"``.  Default: ``MONITOR``.
    medium:
        Policy applied when severity is ``"medium"``.  Default: ``ALERT``.
    high:
        Policy applied when severity is ``"high"``.  Default: ``THROTTLE``.
    critical:
        Policy applied when severity is ``"critical"``.  Default: ``QUARANTINE``.
    """

    low: str = DriftPolicy.MONITOR
    medium: str = DriftPolicy.ALERT
    high: str = DriftPolicy.THROTTLE
    critical: str = DriftPolicy.QUARANTINE

    def action_for_severity(self, severity: str) -> str:
        """Return the configured action for a given severity label.

        Parameters
        ----------
        severity:
            One of ``"none"``, ``"low"``, ``"medium"``, ``"high"``,
            ``"critical"``.

        Returns
        -------
        str
            The corresponding :class:`DriftPolicy` constant.  Returns
            ``DriftPolicy.MONITOR`` for ``"none"`` or unknown values.
        """
        return {
            "low": self.low,
            "medium": self.medium,
            "high": self.high,
            "critical": self.critical,
        }.get(severity, DriftPolicy.MONITOR)


# ---------------------------------------------------------------------------
# DriftAlert
# ---------------------------------------------------------------------------


@dataclass
class DriftAlert:
    """Record of a triggered alert event.

    Attributes
    ----------
    agent_id:
        Identifier of the agent that triggered the alert.
    drift_score:
        The :class:`DriftScore` that caused the alert.
    policy_action:
        The :class:`DriftPolicy` action that was taken.
    timestamp:
        Wall-clock time when the alert was created.
    acknowledged:
        Whether the alert has been acknowledged by an operator.
    """

    agent_id: str
    drift_score: DriftScore
    policy_action: str
    timestamp: float = field(default_factory=time.time)
    acknowledged: bool = False

    def acknowledge(self) -> None:
        """Mark this alert as acknowledged."""
        self.acknowledged = True


# ---------------------------------------------------------------------------
# DriftDetector
# ---------------------------------------------------------------------------

# Dimension weights (must sum to 1.0)
_WEIGHTS: Dict[str, float] = {
    "syscall_histogram": 0.25,
    "tool_usage": 0.20,
    "resource_access": 0.15,
    "error_rate": 0.15,
    "memory_rates": 0.10,
    "message_types": 0.10,
    "think_duration": 0.05,
}


class DriftDetector:
    """Main detection engine: builds baselines and computes drift scores.

    The detector maintains two sliding windows per agent:

    * **Baseline window** — the first ``baseline_window`` ticks of events
      used to establish "normal" behavior.
    * **Detection window** — the most recent ``detection_window`` ticks of
      events compared against the baseline.

    Parameters
    ----------
    baseline_window:
        Number of ticks used to build the baseline profile.  Default: 100.
    detection_window:
        Number of recent ticks used when computing drift.  Default: 20.
    threshold:
        Drift score threshold above which an agent is considered drifting.
        Default: 0.4.
    sensitivity:
        Multiplier that scales z-score contributions.  Values > 1.0 make
        the detector more sensitive to scalar metric changes.  Default: 1.0.
    """

    def __init__(
        self,
        baseline_window: int = 100,
        detection_window: int = 20,
        threshold: float = 0.4,
        sensitivity: float = 1.0,
    ) -> None:
        self._baseline_window = baseline_window
        self._detection_window = detection_window
        self._threshold = threshold
        self._sensitivity = sensitivity

        # agent_id → list of all recorded events (ordered by arrival)
        self._events: Dict[str, List[BehaviorEvent]] = defaultdict(list)
        # agent_id → baseline BehaviorProfile
        self._baselines: Dict[str, BehaviorProfile] = {}
        # Baseline statistical summaries for scalar metrics (per agent)
        self._baseline_stats: Dict[str, Dict[str, Tuple[float, float]]] = {}
        # Lock for thread safety
        self._lock = threading.Lock()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def record_event(self, event: BehaviorEvent) -> None:
        """Record a single behavioral observation.

        Parameters
        ----------
        event:
            The :class:`BehaviorEvent` to record.
        """
        with self._lock:
            self._events[event.agent_id].append(event)

    def build_baseline(self, agent_id: str) -> BehaviorProfile:
        """Build (or rebuild) a baseline profile from recorded events.

        Uses the first ``baseline_window`` events for the given agent.
        If fewer events exist the entire event list is used.

        Parameters
        ----------
        agent_id:
            The agent whose baseline should be computed.

        Returns
        -------
        BehaviorProfile
            The newly built baseline profile.
        """
        with self._lock:
            events = self._events.get(agent_id, [])
            baseline_events = events[: self._baseline_window]
            profile = BehaviorProfile(
                agent_id=agent_id, window_size=len(baseline_events)
            )
            profile.update(baseline_events)

            self._baselines[agent_id] = profile
            self._baseline_stats[agent_id] = self._compute_baseline_stats(
                agent_id, baseline_events
            )
            return profile

    def get_baseline(self, agent_id: str) -> Optional[BehaviorProfile]:
        """Return the baseline profile for an agent, or ``None`` if absent.

        Parameters
        ----------
        agent_id:
            The agent whose baseline to retrieve.

        Returns
        -------
        Optional[BehaviorProfile]
        """
        with self._lock:
            return self._baselines.get(agent_id)

    def compute_drift(self, agent_id: str) -> DriftScore:
        """Compare the agent's recent behavior against its baseline.

        If no baseline exists one is built automatically.  The drift score
        is computed as a weighted average of per-dimension scores (see
        module-level ``_WEIGHTS``).

        Parameters
        ----------
        agent_id:
            The agent to assess.

        Returns
        -------
        DriftScore
            The computed drift measurement.

        Raises
        ------
        ValueError
            If no events have been recorded for ``agent_id``.
        """
        with self._lock:
            events = self._events.get(agent_id, [])

        if not events:
            # Return a zero-drift sentinel score for agents with no events
            return DriftScore(
                agent_id=agent_id,
                score=0.0,
                components={},
                threshold=self._threshold,
                is_drifting=False,
                severity="none",
                explanation="No events recorded for this agent.",
                tick=0,
            )

        # Ensure a baseline exists
        with self._lock:
            if agent_id not in self._baselines:
                self._lock.release()
                self.build_baseline(agent_id)
                self._lock.acquire()
            baseline = self._baselines[agent_id]
            stats = self._baseline_stats.get(agent_id, {})

        # Gather recent window events (most recent detection_window ticks)
        recent_events = self._recent_events(agent_id, events)

        # Build a temporary profile from recent events
        recent_profile = BehaviorProfile(
            agent_id=agent_id, window_size=len(recent_events)
        )
        recent_profile.update(recent_events)

        # Compute per-dimension scores
        components = self._compute_dimension_scores(
            baseline, recent_profile, stats
        )

        # Weighted composite
        composite = sum(
            _WEIGHTS.get(dim, 0.0) * score for dim, score in components.items()
        )
        composite = min(max(composite, 0.0), 1.0)

        # Determine tick (max tick in recent events or overall)
        current_tick = max((e.tick for e in events), default=0)

        is_drifting = composite > self._threshold
        severity = DriftScore.classify_severity(composite, self._threshold)
        explanation = self._build_explanation(components, composite, is_drifting)

        return DriftScore(
            agent_id=agent_id,
            score=composite,
            components=components,
            threshold=self._threshold,
            is_drifting=is_drifting,
            severity=severity,
            explanation=explanation,
            tick=current_tick,
        )

    def is_drifting(self, agent_id: str) -> bool:
        """Convenience method: return ``True`` if the agent is currently drifting.

        Parameters
        ----------
        agent_id:
            The agent to check.

        Returns
        -------
        bool
        """
        return self.compute_drift(agent_id).is_drifting

    def reset_baseline(self, agent_id: str) -> BehaviorProfile:
        """Rebuild the baseline from the agent's most recent events.

        Useful when an intentional behavioral change has been made and the
        old baseline is no longer appropriate.

        Parameters
        ----------
        agent_id:
            The agent whose baseline should be reset.

        Returns
        -------
        BehaviorProfile
            The newly built baseline profile.
        """
        with self._lock:
            events = self._events.get(agent_id, [])
            # Use the most recent baseline_window events
            recent = events[-self._baseline_window :] if events else []
            self._events[agent_id] = list(recent)

        return self.build_baseline(agent_id)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _recent_events(
        self, agent_id: str, events: List[BehaviorEvent]
    ) -> List[BehaviorEvent]:
        """Return events from the most recent detection_window ticks.

        Parameters
        ----------
        agent_id:
            Agent identifier (unused; kept for signature clarity).
        events:
            All recorded events for the agent.

        Returns
        -------
        List[BehaviorEvent]
        """
        if not events:
            return []
        all_ticks = sorted({e.tick for e in events})
        if len(all_ticks) <= self._detection_window:
            return list(events)
        cutoff_tick = all_ticks[-self._detection_window]
        return [e for e in events if e.tick >= cutoff_tick]

    def _compute_baseline_stats(
        self, agent_id: str, events: List[BehaviorEvent]
    ) -> Dict[str, Tuple[float, float]]:
        """Compute per-tick mean and stddev for scalar metrics.

        Returns a dict mapping metric name → (mean, stddev).

        Parameters
        ----------
        agent_id:
            Agent identifier.
        events:
            Baseline events.

        Returns
        -------
        Dict[str, Tuple[float, float]]
        """
        if not events:
            return {}

        ticks = sorted({e.tick for e in events})
        if not ticks:
            return {}

        # Build per-tick counts
        per_tick: Dict[int, Dict[str, float]] = {
            t: {"memory_write": 0, "memory_read": 0, "error": 0, "think_ms": 0.0}
            for t in ticks
        }
        think_counts: Dict[int, int] = {t: 0 for t in ticks}

        for ev in events:
            t = ev.tick
            if ev.event_type == "memory_write":
                per_tick[t]["memory_write"] += 1
            elif ev.event_type == "memory_read":
                per_tick[t]["memory_read"] += 1
            elif ev.event_type == "error":
                per_tick[t]["error"] += 1
            elif ev.event_type == "think_complete":
                if ev.metadata and "duration_ms" in ev.metadata:
                    per_tick[t]["think_ms"] += float(ev.metadata["duration_ms"])
                    think_counts[t] += 1

        # Convert think_ms totals to averages per tick
        for t in ticks:
            c = think_counts[t]
            if c > 0:
                per_tick[t]["think_ms"] /= c

        def _stats(key: str) -> Tuple[float, float]:
            vals = [per_tick[t][key] for t in ticks]
            m = statistics.mean(vals) if vals else 0.0
            s = statistics.pstdev(vals) if len(vals) > 1 else 0.0
            return m, s

        return {
            "memory_write_rate": _stats("memory_write"),
            "memory_read_rate": _stats("memory_read"),
            "error_rate": _stats("error"),
            "think_duration_ms": _stats("think_ms"),
        }

    def _dimension_score_from_cosine(self, baseline_hist: Dict[str, int],
                                     recent_hist: Dict[str, int]) -> float:
        """Convert cosine similarity to a [0, 1] drift score.

        High similarity → low drift score.  Returns 1.0 when there is zero
        similarity (completely different behavior) and 0.0 when behavior is
        identical.

        If the baseline is empty but the recent histogram is not, the agent
        has started a wholly new category of activity → maximum drift (1.0).
        If both are empty → no drift (0.0).

        Parameters
        ----------
        baseline_hist:
            Baseline histogram.
        recent_hist:
            Recent observation histogram.

        Returns
        -------
        float
            Drift score in [0.0, 1.0].
        """
        if not baseline_hist and not recent_hist:
            return 0.0
        if not baseline_hist and recent_hist:
            return 1.0
        if not recent_hist:
            # Agent has stopped doing something it used to do
            return 0.5
        return 1.0 - _cosine_similarity(baseline_hist, recent_hist)

    def _dimension_score_from_z_score(
        self, value: float, stats: Tuple[float, float]
    ) -> float:
        """Convert a z-score into a [0, 1] drift score.

        Uses a sigmoid-like clamping so that extreme deviations (|z| > 5)
        saturate at 1.0.

        Parameters
        ----------
        value:
            Observed metric value.
        stats:
            (mean, stddev) from baseline.

        Returns
        -------
        float
            Drift score in [0.0, 1.0].
        """
        mean, stddev = stats
        z = abs(_z_score(value, mean, stddev)) * self._sensitivity
        # Map |z| → [0, 1] via a soft cap at z=5
        return min(z / 5.0, 1.0)

    def _compute_dimension_scores(
        self,
        baseline: BehaviorProfile,
        recent: BehaviorProfile,
        stats: Dict[str, Tuple[float, float]],
    ) -> Dict[str, float]:
        """Compute a drift score for each weighted dimension.

        Parameters
        ----------
        baseline:
            The established baseline profile.
        recent:
            Profile built from the recent detection window.
        stats:
            Per-metric (mean, stddev) from baseline.

        Returns
        -------
        Dict[str, float]
            Mapping of dimension name → score in [0.0, 1.0].
        """
        scores: Dict[str, float] = {}

        # Histogram-based dimensions (cosine distance)
        scores["syscall_histogram"] = self._dimension_score_from_cosine(
            baseline.syscall_histogram, recent.syscall_histogram
        )
        scores["tool_usage"] = self._dimension_score_from_cosine(
            baseline.tool_usage_histogram, recent.tool_usage_histogram
        )
        scores["resource_access"] = self._dimension_score_from_cosine(
            baseline.resource_access_patterns, recent.resource_access_patterns
        )
        scores["message_types"] = self._dimension_score_from_cosine(
            baseline.message_type_distribution, recent.message_type_distribution
        )

        # Scalar dimensions (z-score based)
        def _scalar(key: str, recent_val: float) -> float:
            if key in stats:
                return self._dimension_score_from_z_score(recent_val, stats[key])
            # No baseline stats → fall back to relative comparison
            return 0.0

        # Combine memory_write and memory_read into one memory_rates score
        write_score = _scalar("memory_write_rate", recent.memory_write_rate)
        read_score = _scalar("memory_read_rate", recent.memory_read_rate)
        scores["memory_rates"] = (write_score + read_score) / 2.0

        scores["error_rate"] = _scalar("error_rate", recent.error_rate)
        scores["think_duration"] = _scalar(
            "think_duration_ms", recent.avg_think_duration_ms
        )

        return scores

    @staticmethod
    def _build_explanation(
        components: Dict[str, float], composite: float, is_drifting: bool
    ) -> str:
        """Build a human-readable explanation of the drift assessment.

        Parameters
        ----------
        components:
            Per-dimension drift scores.
        composite:
            Overall composite score.
        is_drifting:
            Whether the composite exceeds the threshold.

        Returns
        -------
        str
        """
        if not is_drifting:
            return (
                f"Agent behavior is within normal parameters "
                f"(composite score: {composite:.3f})."
            )

        # Sort dimensions by their contribution (weight × score)
        ranked = sorted(
            components.items(), key=lambda kv: _WEIGHTS.get(kv[0], 0) * kv[1],
            reverse=True,
        )

        top_dims = [
            f"{dim} ({score:.3f})"
            for dim, score in ranked
            if score > 0.1
        ]

        if not top_dims:
            top_dims = [f"{ranked[0][0]} ({ranked[0][1]:.3f})"] if ranked else ["unknown"]

        return (
            f"Drift detected (score: {composite:.3f}). "
            f"Top contributing dimensions: {', '.join(top_dims[:3])}."
        )


# ---------------------------------------------------------------------------
# DriftMonitor
# ---------------------------------------------------------------------------


class DriftMonitor:
    """Continuous monitoring orchestrator that feeds events and fires alerts.

    The monitor wraps a :class:`DriftDetector` and adds:

    * Automatic drift checks every ``check_interval`` ticks per agent.
    * Alert history and callback notification.
    * Agent registration/unregistration.

    Parameters
    ----------
    detector:
        The :class:`DriftDetector` instance to use.
    policy_config:
        :class:`DriftPolicyConfig` mapping severities to actions.
    check_interval:
        Number of ticks between automatic drift checks.  Default: 10.
    """

    def __init__(
        self,
        detector: DriftDetector,
        policy_config: DriftPolicyConfig,
        check_interval: int = 10,
    ) -> None:
        self._detector = detector
        self._policy_config = policy_config
        self._check_interval = check_interval

        self._registered_agents: set = set()
        self._tick_counters: Dict[str, int] = defaultdict(int)
        self._last_check_tick: Dict[str, int] = defaultdict(int)
        self.alerts: List[DriftAlert] = []
        self._alert_callbacks: List[Callable[[DriftAlert], None]] = []
        self._lock = threading.Lock()

        # Track agent states for policy enforcement
        self._agent_states: Dict[str, str] = {}  # agent_id → current state

    # ------------------------------------------------------------------
    # Agent registration
    # ------------------------------------------------------------------

    def register_agent(self, agent_id: str) -> None:
        """Register an agent for drift monitoring.

        Parameters
        ----------
        agent_id:
            Identifier of the agent to monitor.
        """
        with self._lock:
            self._registered_agents.add(agent_id)
            self._agent_states[agent_id] = "active"

    def unregister_agent(self, agent_id: str) -> None:
        """Remove an agent from drift monitoring.

        Parameters
        ----------
        agent_id:
            Identifier of the agent to remove.
        """
        with self._lock:
            self._registered_agents.discard(agent_id)
            self._agent_states.pop(agent_id, None)

    # ------------------------------------------------------------------
    # Event ingestion
    # ------------------------------------------------------------------

    def on_event(self, event: BehaviorEvent) -> None:
        """Feed a single event into the monitor.

        Records the event in the detector and auto-checks drift every
        ``check_interval`` ticks for the event's agent.

        Parameters
        ----------
        event:
            The :class:`BehaviorEvent` to process.
        """
        self._detector.record_event(event)

        if event.agent_id not in self._registered_agents:
            return

        with self._lock:
            self._tick_counters[event.agent_id] = max(
                self._tick_counters[event.agent_id], event.tick
            )
            current_tick = self._tick_counters[event.agent_id]
            last_check = self._last_check_tick[event.agent_id]

        if current_tick - last_check >= self._check_interval:
            self._run_check(event.agent_id, current_tick)

    # ------------------------------------------------------------------
    # Manual checks
    # ------------------------------------------------------------------

    def check_all(self) -> List[DriftScore]:
        """Run drift checks on all registered agents.

        Returns
        -------
        List[DriftScore]
            One :class:`DriftScore` per registered agent.
        """
        scores: List[DriftScore] = []
        with self._lock:
            agents = list(self._registered_agents)

        for agent_id in agents:
            score = self._detector.compute_drift(agent_id)
            scores.append(score)
            if score.is_drifting:
                self._handle_drift(score)
        return scores

    def get_status(self, agent_id: str) -> Dict[str, Any]:
        """Return a status summary for a specific agent.

        Parameters
        ----------
        agent_id:
            The agent to query.

        Returns
        -------
        dict
            Keys: ``agent_id``, ``registered``, ``state``, ``drift_score``,
            ``baseline_info``, ``recent_alerts``.
        """
        score = self._detector.compute_drift(agent_id)
        baseline = self._detector.get_baseline(agent_id)

        baseline_info: Dict[str, Any] = {}
        if baseline:
            baseline_info = {
                "window_size": baseline.window_size,
                "total_observations": baseline.total_observations,
                "created_at": baseline.created_at,
                "updated_at": baseline.updated_at,
            }

        with self._lock:
            registered = agent_id in self._registered_agents
            state = self._agent_states.get(agent_id, "unknown")
            recent_alerts = [
                a for a in self.alerts if a.agent_id == agent_id
            ][-5:]

        return {
            "agent_id": agent_id,
            "registered": registered,
            "state": state,
            "drift_score": score,
            "baseline_info": baseline_info,
            "recent_alerts": recent_alerts,
        }

    # ------------------------------------------------------------------
    # Alert management
    # ------------------------------------------------------------------

    def add_alert_callback(self, callback: Callable[[DriftAlert], None]) -> None:
        """Register a callback to be invoked when an alert is triggered.

        Parameters
        ----------
        callback:
            A callable that accepts a single :class:`DriftAlert` argument.
        """
        with self._lock:
            self._alert_callbacks.append(callback)

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _run_check(self, agent_id: str, current_tick: int) -> None:
        """Run a drift check for one agent and handle the result.

        Parameters
        ----------
        agent_id:
            The agent to check.
        current_tick:
            Current tick counter.
        """
        with self._lock:
            self._last_check_tick[agent_id] = current_tick

        score = self._detector.compute_drift(agent_id)
        if score.is_drifting:
            self._handle_drift(score)

    def _handle_drift(self, score: DriftScore) -> None:
        """Apply the appropriate policy action and record an alert.

        Parameters
        ----------
        score:
            The drift score that triggered the policy.
        """
        action = self._policy_config.action_for_severity(score.severity)
        alert = DriftAlert(
            agent_id=score.agent_id,
            drift_score=score,
            policy_action=action,
        )

        with self._lock:
            self.alerts.append(alert)
            # Apply state changes based on policy
            if action == DriftPolicy.QUARANTINE:
                self._agent_states[score.agent_id] = "quarantined"
            elif action == DriftPolicy.THROTTLE:
                self._agent_states[score.agent_id] = "throttled"
            callbacks = list(self._alert_callbacks)

        # Fire callbacks outside the lock to avoid potential deadlocks
        for cb in callbacks:
            try:
                cb(alert)
            except Exception:
                pass  # Never let a bad callback crash the monitor


# ---------------------------------------------------------------------------
# DriftReport
# ---------------------------------------------------------------------------


class DriftReport:
    """Generates comprehensive drift analysis reports for an agent.

    Parameters
    ----------
    detector:
        The :class:`DriftDetector` providing baseline and drift data.
    monitor:
        Optional :class:`DriftMonitor` providing alert history.
    """

    def __init__(
        self,
        detector: DriftDetector,
        monitor: Optional[DriftMonitor] = None,
    ) -> None:
        self._detector = detector
        self._monitor = monitor

    def generate(self, agent_id: str) -> Dict[str, Any]:
        """Generate a comprehensive drift analysis for the given agent.

        The returned dictionary contains the following top-level keys:

        * ``agent_id`` — the agent identifier.
        * ``generated_at`` — wall-clock timestamp.
        * ``baseline_profile`` — baseline :class:`BehaviorProfile` or
          ``None``.
        * ``current_drift_score`` — :class:`DriftScore` from the latest
          assessment.
        * ``top_drifting_dimensions`` — list of (dimension, score) tuples
          sorted by weighted contribution, descending.
        * ``recommendations`` — list of human-readable recommendation
          strings.
        * ``alert_history`` — list of recent :class:`DriftAlert` objects
          (empty if no monitor is attached).

        Parameters
        ----------
        agent_id:
            The agent to analyse.

        Returns
        -------
        dict
        """
        baseline = self._detector.get_baseline(agent_id)
        if baseline is None:
            self._detector.build_baseline(agent_id)
            baseline = self._detector.get_baseline(agent_id)

        score = self._detector.compute_drift(agent_id)

        # Rank drifting dimensions by weighted contribution
        top_drifting = sorted(
            [
                (dim, s, _WEIGHTS.get(dim, 0.0) * s)
                for dim, s in score.components.items()
            ],
            key=lambda x: x[2],
            reverse=True,
        )

        recommendations = self._build_recommendations(score, top_drifting)

        alert_history: List[DriftAlert] = []
        if self._monitor is not None:
            alert_history = [
                a for a in self._monitor.alerts if a.agent_id == agent_id
            ]

        return {
            "agent_id": agent_id,
            "generated_at": time.time(),
            "baseline_profile": baseline,
            "current_drift_score": score,
            "top_drifting_dimensions": [
                {"dimension": dim, "score": s, "weighted_contribution": w}
                for dim, s, w in top_drifting
            ],
            "recommendations": recommendations,
            "alert_history": alert_history,
        }

    @staticmethod
    def _build_recommendations(
        score: DriftScore,
        top_drifting: List[Tuple[str, float, float]],
    ) -> List[str]:
        """Build a list of human-readable recommendations.

        Parameters
        ----------
        score:
            Current drift score.
        top_drifting:
            Sorted list of (dimension, score, weighted_contribution).

        Returns
        -------
        List[str]
        """
        recs: List[str] = []

        if not score.is_drifting:
            recs.append("No action required — agent behavior is within normal parameters.")
            return recs

        severity = score.severity

        if severity == "low":
            recs.append("Monitor the agent closely for continued drift.")
        elif severity == "medium":
            recs.append("Investigate the root cause of behavioral drift.")
            recs.append("Consider reviewing recent prompt history for injection attempts.")
        elif severity == "high":
            recs.append("Reduce agent scheduling priority (throttle) pending investigation.")
            recs.append("Capture a full state snapshot for forensic analysis.")
        elif severity == "critical":
            recs.append("Immediately quarantine the agent — suspend all execution.")
            recs.append("Conduct a full security audit before re-enabling the agent.")
            recs.append("Notify the security team.")

        # Dimension-specific recommendations
        for dim, s, _w in top_drifting[:3]:
            if s < 0.2:
                continue
            if dim == "syscall_histogram":
                recs.append(
                    "Syscall patterns have shifted — check for new or unexpected system calls."
                )
            elif dim == "tool_usage":
                recs.append(
                    "Tool usage distribution has changed — verify the agent is not using unauthorized tools."
                )
            elif dim == "error_rate":
                recs.append(
                    "Error rate has increased significantly — check for model degradation or hallucination loops."
                )
            elif dim == "resource_access":
                recs.append(
                    "Resource access patterns have changed — review accessed paths for data exfiltration risk."
                )
            elif dim == "memory_rates":
                recs.append(
                    "Memory access rates are anomalous — possible thrashing or unauthorized memory inspection."
                )
            elif dim == "think_duration":
                recs.append(
                    "Think duration has changed significantly — possible model swap or reasoning loop."
                )
            elif dim == "message_types":
                recs.append(
                    "IPC message distribution has shifted — check for unexpected inter-agent communication."
                )

        return recs


# ---------------------------------------------------------------------------
# Convenience factory helpers
# ---------------------------------------------------------------------------


def make_detector(
    baseline_window: int = 100,
    detection_window: int = 20,
    threshold: float = 0.4,
    sensitivity: float = 1.0,
) -> DriftDetector:
    """Create and return a :class:`DriftDetector` with the given parameters.

    Parameters
    ----------
    baseline_window:
        Number of ticks for baseline.
    detection_window:
        Number of recent ticks for drift detection.
    threshold:
        Drift threshold.
    sensitivity:
        Z-score multiplier.

    Returns
    -------
    DriftDetector
    """
    return DriftDetector(
        baseline_window=baseline_window,
        detection_window=detection_window,
        threshold=threshold,
        sensitivity=sensitivity,
    )


def make_monitor(
    detector: Optional[DriftDetector] = None,
    policy_config: Optional[DriftPolicyConfig] = None,
    check_interval: int = 10,
) -> DriftMonitor:
    """Create and return a :class:`DriftMonitor`.

    Parameters
    ----------
    detector:
        Optional :class:`DriftDetector`.  A new one is created if not supplied.
    policy_config:
        Optional :class:`DriftPolicyConfig`.  Defaults to standard config if
        not supplied.
    check_interval:
        Tick interval between auto-checks.

    Returns
    -------
    DriftMonitor
    """
    if detector is None:
        detector = DriftDetector()
    if policy_config is None:
        policy_config = DriftPolicyConfig()
    return DriftMonitor(
        detector=detector,
        policy_config=policy_config,
        check_interval=check_interval,
    )


def make_event(
    agent_id: str,
    tick: int,
    event_type: str,
    detail: str,
    metadata: Optional[Dict[str, Any]] = None,
) -> BehaviorEvent:
    """Convenience factory for :class:`BehaviorEvent`.

    Parameters
    ----------
    agent_id:
        Agent identifier.
    tick:
        Logical clock tick.
    event_type:
        Event category.
    detail:
        Specific label.
    metadata:
        Optional extra data.

    Returns
    -------
    BehaviorEvent
    """
    return BehaviorEvent(
        agent_id=agent_id,
        tick=tick,
        event_type=event_type,
        detail=detail,
        metadata=metadata,
    )
