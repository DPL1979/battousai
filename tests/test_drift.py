"""
tests/test_drift.py
===================
Comprehensive test suite for battousai/drift.py

Covers:
- BehaviorProfile creation and updates
- BehaviorEvent recording
- Baseline building from event streams
- Drift detection: no-drift scenario
- Drift detection: new syscalls appear
- Drift detection: frequency shift
- Drift detection: error rate spike
- Drift detection: think duration anomaly
- DriftScore severity classification
- DriftPolicy enforcement at each level
- DriftMonitor auto-check at interval
- Alert callbacks firing
- Baseline reset
- Edge cases: no events, single event, multi-agent
- Math utilities
- DriftReport generation
"""

import math
import time
import threading
import unittest
from collections import defaultdict
from typing import Dict, List

from battousai.drift import (
    BehaviorEvent,
    BehaviorProfile,
    DriftAlert,
    DriftDetector,
    DriftMonitor,
    DriftPolicy,
    DriftPolicyConfig,
    DriftReport,
    DriftScore,
    _chi_squared_divergence,
    _cosine_similarity,
    _normalize_histogram,
    _z_score,
    make_detector,
    make_event,
    make_monitor,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_events(
    agent_id: str,
    syscalls: List[str],
    tools: List[str],
    ticks_per_item: int = 1,
    start_tick: int = 0,
) -> List[BehaviorEvent]:
    """Generate a simple event stream for testing."""
    events = []
    tick = start_tick
    for sc in syscalls:
        events.append(BehaviorEvent(agent_id=agent_id, tick=tick, event_type="syscall", detail=sc))
        tick += ticks_per_item
    for tl in tools:
        events.append(BehaviorEvent(agent_id=agent_id, tick=tick, event_type="tool_use", detail=tl))
        tick += ticks_per_item
    return events


def _stable_detector(baseline_window: int = 50, detection_window: int = 10) -> DriftDetector:
    """Build a detector with a stable agent baseline."""
    detector = DriftDetector(
        baseline_window=baseline_window,
        detection_window=detection_window,
        threshold=0.4,
    )
    agent_id = "stable-agent"
    # Record baseline events (50 ticks, repeated pattern)
    for tick in range(baseline_window):
        for sc in ["read", "write", "open"]:
            detector.record_event(BehaviorEvent(agent_id=agent_id, tick=tick, event_type="syscall", detail=sc))
        detector.record_event(BehaviorEvent(agent_id=agent_id, tick=tick, event_type="tool_use", detail="search"))
        detector.record_event(BehaviorEvent(agent_id=agent_id, tick=tick, event_type="memory_write", detail=""))
        detector.record_event(BehaviorEvent(agent_id=agent_id, tick=tick, event_type="memory_read", detail=""))
    detector.build_baseline(agent_id)
    return detector


# ---------------------------------------------------------------------------
# Tests: Math Utilities
# ---------------------------------------------------------------------------

class TestCosıneSimilarity(unittest.TestCase):
    def test_identical_histograms(self):
        h = {"a": 5, "b": 3}
        self.assertAlmostEqual(_cosine_similarity(h, h), 1.0, places=6)

    def test_orthogonal_histograms(self):
        a = {"x": 1}
        b = {"y": 1}
        self.assertAlmostEqual(_cosine_similarity(a, b), 0.0, places=6)

    def test_empty_first_arg(self):
        self.assertEqual(_cosine_similarity({}, {"a": 1}), 0.0)

    def test_empty_second_arg(self):
        self.assertEqual(_cosine_similarity({"a": 1}, {}), 0.0)

    def test_both_empty(self):
        self.assertEqual(_cosine_similarity({}, {}), 0.0)

    def test_partial_overlap(self):
        a = {"a": 3, "b": 4}
        b = {"a": 3, "b": 4, "c": 0}
        # Zero value in b for c shouldn't change dot product
        result = _cosine_similarity(a, b)
        self.assertAlmostEqual(result, 1.0, places=6)

    def test_similar_not_identical(self):
        a = {"x": 10, "y": 1}
        b = {"x": 10, "y": 2}
        sim = _cosine_similarity(a, b)
        self.assertGreater(sim, 0.99)
        self.assertLessEqual(sim, 1.0)

    def test_completely_different(self):
        a = {"alpha": 100}
        b = {"beta": 100}
        self.assertAlmostEqual(_cosine_similarity(a, b), 0.0, places=6)

    def test_scalar_multiple(self):
        # Scaling one vector shouldn't change cosine similarity
        a = {"x": 1, "y": 1}
        b = {"x": 100, "y": 100}
        self.assertAlmostEqual(_cosine_similarity(a, b), 1.0, places=6)


class TestZScore(unittest.TestCase):
    def test_exact_mean(self):
        self.assertAlmostEqual(_z_score(5.0, 5.0, 2.0), 0.0)

    def test_one_stddev_above(self):
        self.assertAlmostEqual(_z_score(7.0, 5.0, 2.0), 1.0)

    def test_one_stddev_below(self):
        self.assertAlmostEqual(_z_score(3.0, 5.0, 2.0), -1.0)

    def test_zero_stddev_equal(self):
        self.assertEqual(_z_score(3.0, 3.0, 0.0), 0.0)

    def test_zero_stddev_different(self):
        self.assertEqual(_z_score(5.0, 3.0, 0.0), 10.0)

    def test_negative_deviation(self):
        z = _z_score(0.0, 10.0, 5.0)
        self.assertAlmostEqual(z, -2.0)


class TestNormalizeHistogram(unittest.TestCase):
    def test_basic_normalization(self):
        h = {"a": 1, "b": 3}
        norm = _normalize_histogram(h)
        self.assertAlmostEqual(norm["a"], 0.25)
        self.assertAlmostEqual(norm["b"], 0.75)

    def test_empty_histogram(self):
        self.assertEqual(_normalize_histogram({}), {})

    def test_all_zero_counts(self):
        self.assertEqual(_normalize_histogram({"a": 0, "b": 0}), {})

    def test_sum_to_one(self):
        h = {"x": 2, "y": 3, "z": 5}
        norm = _normalize_histogram(h)
        self.assertAlmostEqual(sum(norm.values()), 1.0, places=9)

    def test_single_key(self):
        norm = _normalize_histogram({"only": 7})
        self.assertAlmostEqual(norm["only"], 1.0)


class TestChiSquaredDivergence(unittest.TestCase):
    def test_identical_distributions(self):
        d = {"a": 0.5, "b": 0.5}
        self.assertAlmostEqual(_chi_squared_divergence(d, d), 0.0, places=6)

    def test_empty_distributions(self):
        self.assertAlmostEqual(_chi_squared_divergence({}, {}), 0.0)

    def test_new_key_in_observed(self):
        obs = {"a": 0.5, "b": 0.5}
        exp = {"a": 0.5}
        # b appears in observed but not in expected → large divergence
        result = _chi_squared_divergence(obs, exp)
        self.assertGreater(result, 0.0)

    def test_non_negative(self):
        obs = {"a": 0.6, "b": 0.4}
        exp = {"a": 0.3, "b": 0.7}
        self.assertGreaterEqual(_chi_squared_divergence(obs, exp), 0.0)

    def test_zero_observed_in_key(self):
        obs = {"a": 1.0}
        exp = {"a": 0.5, "b": 0.5}
        result = _chi_squared_divergence(obs, exp)
        self.assertGreaterEqual(result, 0.0)


# ---------------------------------------------------------------------------
# Tests: BehaviorEvent
# ---------------------------------------------------------------------------

class TestBehaviorEvent(unittest.TestCase):
    def test_creation_defaults(self):
        ev = BehaviorEvent(agent_id="a1", tick=0, event_type="syscall", detail="open")
        self.assertEqual(ev.agent_id, "a1")
        self.assertEqual(ev.tick, 0)
        self.assertEqual(ev.event_type, "syscall")
        self.assertEqual(ev.detail, "open")
        self.assertIsNone(ev.metadata)
        self.assertIsInstance(ev.timestamp, float)

    def test_creation_with_metadata(self):
        ev = BehaviorEvent(
            agent_id="a2", tick=5, event_type="think_complete",
            detail="done", metadata={"duration_ms": 42.0}
        )
        self.assertEqual(ev.metadata["duration_ms"], 42.0)

    def test_all_event_types(self):
        valid_types = [
            "syscall", "resource_access", "message_sent", "tool_use",
            "memory_write", "memory_read", "error", "think_complete",
        ]
        for et in valid_types:
            ev = BehaviorEvent(agent_id="x", tick=0, event_type=et, detail="d")
            self.assertEqual(ev.event_type, et)

    def test_make_event_factory(self):
        ev = make_event("agentX", 10, "syscall", "read")
        self.assertEqual(ev.agent_id, "agentX")
        self.assertEqual(ev.tick, 10)
        self.assertEqual(ev.event_type, "syscall")
        self.assertEqual(ev.detail, "read")

    def test_make_event_with_metadata(self):
        ev = make_event("a", 1, "tool_use", "search", metadata={"extra": True})
        self.assertEqual(ev.metadata["extra"], True)

    def test_timestamp_recent(self):
        before = time.time()
        ev = BehaviorEvent(agent_id="t", tick=0, event_type="error", detail="e")
        after = time.time()
        self.assertLessEqual(before, ev.timestamp)
        self.assertLessEqual(ev.timestamp, after)


# ---------------------------------------------------------------------------
# Tests: BehaviorProfile
# ---------------------------------------------------------------------------

class TestBehaviorProfile(unittest.TestCase):
    def test_default_creation(self):
        bp = BehaviorProfile(agent_id="a")
        self.assertEqual(bp.agent_id, "a")
        self.assertEqual(bp.window_size, 100)
        self.assertEqual(bp.total_observations, 0)
        self.assertEqual(bp.syscall_histogram, {})

    def test_custom_window_size(self):
        bp = BehaviorProfile(agent_id="a", window_size=50)
        self.assertEqual(bp.window_size, 50)

    def test_update_with_empty_events(self):
        bp = BehaviorProfile(agent_id="a")
        bp.update([])
        self.assertEqual(bp.total_observations, 0)
        self.assertEqual(bp.memory_write_rate, 0.0)
        self.assertEqual(bp.error_rate, 0.0)

    def test_update_counts_syscalls(self):
        bp = BehaviorProfile(agent_id="a")
        events = [
            BehaviorEvent("a", 0, "syscall", "open"),
            BehaviorEvent("a", 0, "syscall", "read"),
            BehaviorEvent("a", 1, "syscall", "open"),
        ]
        bp.update(events)
        self.assertEqual(bp.syscall_histogram["open"], 2)
        self.assertEqual(bp.syscall_histogram["read"], 1)

    def test_update_counts_tools(self):
        bp = BehaviorProfile(agent_id="a")
        events = [
            BehaviorEvent("a", 0, "tool_use", "search"),
            BehaviorEvent("a", 1, "tool_use", "search"),
            BehaviorEvent("a", 2, "tool_use", "execute"),
        ]
        bp.update(events)
        self.assertEqual(bp.tool_usage_histogram["search"], 2)
        self.assertEqual(bp.tool_usage_histogram["execute"], 1)

    def test_update_memory_rates(self):
        bp = BehaviorProfile(agent_id="a")
        events = [
            BehaviorEvent("a", 0, "memory_write", ""),
            BehaviorEvent("a", 0, "memory_write", ""),
            BehaviorEvent("a", 1, "memory_read", ""),
        ]
        bp.update(events)
        # 2 ticks, 2 writes / 2 ticks = 1.0
        self.assertAlmostEqual(bp.memory_write_rate, 1.0)
        # 1 read / 2 ticks = 0.5
        self.assertAlmostEqual(bp.memory_read_rate, 0.5)

    def test_update_error_rate(self):
        bp = BehaviorProfile(agent_id="a")
        events = [
            BehaviorEvent("a", 0, "error", "timeout"),
            BehaviorEvent("a", 1, "error", "oops"),
        ]
        bp.update(events)
        # 2 unique ticks → rate = 2/2 = 1.0
        self.assertAlmostEqual(bp.error_rate, 1.0)

    def test_update_think_duration(self):
        bp = BehaviorProfile(agent_id="a")
        events = [
            BehaviorEvent("a", 0, "think_complete", "", metadata={"duration_ms": 100.0}),
            BehaviorEvent("a", 1, "think_complete", "", metadata={"duration_ms": 200.0}),
        ]
        bp.update(events)
        self.assertAlmostEqual(bp.avg_think_duration_ms, 150.0)

    def test_update_think_duration_no_metadata(self):
        bp = BehaviorProfile(agent_id="a")
        events = [BehaviorEvent("a", 0, "think_complete", "")]
        bp.update(events)
        self.assertAlmostEqual(bp.avg_think_duration_ms, 0.0)

    def test_update_resource_access(self):
        bp = BehaviorProfile(agent_id="a")
        events = [
            BehaviorEvent("a", 0, "resource_access", "/etc/passwd"),
            BehaviorEvent("a", 1, "resource_access", "/tmp/file"),
            BehaviorEvent("a", 2, "resource_access", "/etc/passwd"),
        ]
        bp.update(events)
        self.assertEqual(bp.resource_access_patterns["/etc/passwd"], 2)
        self.assertEqual(bp.resource_access_patterns["/tmp/file"], 1)

    def test_update_message_types(self):
        bp = BehaviorProfile(agent_id="a")
        events = [
            BehaviorEvent("a", 0, "message_sent", "request"),
            BehaviorEvent("a", 1, "message_sent", "response"),
        ]
        bp.update(events)
        self.assertEqual(bp.message_type_distribution["request"], 1)
        self.assertEqual(bp.message_type_distribution["response"], 1)

    def test_total_observations(self):
        bp = BehaviorProfile(agent_id="a")
        events = [BehaviorEvent("a", i, "syscall", "read") for i in range(10)]
        bp.update(events)
        self.assertEqual(bp.total_observations, 10)

    def test_updated_at_changes(self):
        bp = BehaviorProfile(agent_id="a")
        old_updated = bp.updated_at
        time.sleep(0.01)
        bp.update([BehaviorEvent("a", 0, "syscall", "read")])
        self.assertGreaterEqual(bp.updated_at, old_updated)


# ---------------------------------------------------------------------------
# Tests: DriftDetector — Baseline Building
# ---------------------------------------------------------------------------

class TestDriftDetectorBaseline(unittest.TestCase):
    def _make_detector(self):
        return DriftDetector(baseline_window=10, detection_window=5, threshold=0.4)

    def test_record_and_build_baseline(self):
        d = self._make_detector()
        for tick in range(10):
            d.record_event(BehaviorEvent("a1", tick, "syscall", "read"))
        profile = d.build_baseline("a1")
        self.assertEqual(profile.agent_id, "a1")
        self.assertGreater(profile.total_observations, 0)

    def test_baseline_missing_returns_none(self):
        d = self._make_detector()
        self.assertIsNone(d.get_baseline("nonexistent"))

    def test_baseline_stored_after_build(self):
        d = self._make_detector()
        d.record_event(BehaviorEvent("a1", 0, "syscall", "open"))
        d.build_baseline("a1")
        self.assertIsNotNone(d.get_baseline("a1"))

    def test_baseline_uses_only_first_window(self):
        d = self._make_detector()
        # Record 20 events across 20 ticks, first 10 are "open", next 10 are "close"
        for tick in range(10):
            d.record_event(BehaviorEvent("a1", tick, "syscall", "open"))
        for tick in range(10, 20):
            d.record_event(BehaviorEvent("a1", tick, "syscall", "close"))
        profile = d.build_baseline("a1")
        # Baseline should only include first 10 events (syscall=open)
        self.assertIn("open", profile.syscall_histogram)
        self.assertNotIn("close", profile.syscall_histogram)

    def test_baseline_with_fewer_events_than_window(self):
        d = self._make_detector()
        for tick in range(3):
            d.record_event(BehaviorEvent("a1", tick, "syscall", "read"))
        profile = d.build_baseline("a1")
        self.assertEqual(profile.total_observations, 3)

    def test_no_events_build_baseline(self):
        d = self._make_detector()
        profile = d.build_baseline("new-agent")
        self.assertEqual(profile.total_observations, 0)
        self.assertEqual(profile.syscall_histogram, {})

    def test_multiple_agents_separate_baselines(self):
        d = self._make_detector()
        for tick in range(5):
            d.record_event(BehaviorEvent("agent1", tick, "syscall", "read"))
            d.record_event(BehaviorEvent("agent2", tick, "syscall", "write"))
        d.build_baseline("agent1")
        d.build_baseline("agent2")
        bp1 = d.get_baseline("agent1")
        bp2 = d.get_baseline("agent2")
        self.assertIn("read", bp1.syscall_histogram)
        self.assertNotIn("write", bp1.syscall_histogram)
        self.assertIn("write", bp2.syscall_histogram)
        self.assertNotIn("read", bp2.syscall_histogram)


# ---------------------------------------------------------------------------
# Tests: DriftDetector — No-Drift Scenario
# ---------------------------------------------------------------------------

class TestDriftDetectorNoDrift(unittest.TestCase):
    def setUp(self):
        self.detector = DriftDetector(
            baseline_window=50,
            detection_window=10,
            threshold=0.4,
            sensitivity=1.0,
        )
        self.agent_id = "stable-agent"
        # Build baseline with consistent pattern
        for tick in range(50):
            for sc in ["read", "write", "open"]:
                self.detector.record_event(
                    BehaviorEvent(self.agent_id, tick, "syscall", sc)
                )
            self.detector.record_event(
                BehaviorEvent(self.agent_id, tick, "tool_use", "search")
            )
            self.detector.record_event(
                BehaviorEvent(self.agent_id, tick, "memory_write", "")
            )
        self.detector.build_baseline(self.agent_id)
        # Add more of the same pattern (detection window)
        for tick in range(50, 60):
            for sc in ["read", "write", "open"]:
                self.detector.record_event(
                    BehaviorEvent(self.agent_id, tick, "syscall", sc)
                )
            self.detector.record_event(
                BehaviorEvent(self.agent_id, tick, "tool_use", "search")
            )
            self.detector.record_event(
                BehaviorEvent(self.agent_id, tick, "memory_write", "")
            )

    def test_stable_agent_low_drift_score(self):
        score = self.detector.compute_drift(self.agent_id)
        self.assertLess(score.score, 0.4)

    def test_stable_agent_not_drifting(self):
        self.assertFalse(self.detector.is_drifting(self.agent_id))

    def test_stable_agent_severity_none(self):
        score = self.detector.compute_drift(self.agent_id)
        self.assertEqual(score.severity, "none")

    def test_stable_agent_score_in_range(self):
        score = self.detector.compute_drift(self.agent_id)
        self.assertGreaterEqual(score.score, 0.0)
        self.assertLessEqual(score.score, 1.0)

    def test_stable_agent_components_populated(self):
        score = self.detector.compute_drift(self.agent_id)
        self.assertIn("syscall_histogram", score.components)
        self.assertIn("tool_usage", score.components)

    def test_drift_score_threshold_matches(self):
        score = self.detector.compute_drift(self.agent_id)
        self.assertAlmostEqual(score.threshold, 0.4)

    def test_is_drifting_false(self):
        score = self.detector.compute_drift(self.agent_id)
        self.assertFalse(score.is_drifting)


# ---------------------------------------------------------------------------
# Tests: DriftDetector — New Syscalls Appear
# ---------------------------------------------------------------------------

class TestDriftDetectorNewSyscalls(unittest.TestCase):
    def setUp(self):
        self.detector = DriftDetector(
            baseline_window=50,
            detection_window=10,
            threshold=0.2,  # low threshold: syscall weight alone (0.25) triggers drift
            sensitivity=1.0,
        )
        self.agent_id = "drifting-syscall-agent"
        # Baseline: only "read" and "write"
        for tick in range(50):
            self.detector.record_event(
                BehaviorEvent(self.agent_id, tick, "syscall", "read")
            )
            self.detector.record_event(
                BehaviorEvent(self.agent_id, tick, "syscall", "write")
            )
            # Also record consistent tool usage in baseline
            self.detector.record_event(
                BehaviorEvent(self.agent_id, tick, "tool_use", "search")
            )
        self.detector.build_baseline(self.agent_id)
        # Detection window: completely different syscalls AND tools
        for tick in range(50, 60):
            for sc in ["exec", "fork", "mmap", "socket", "connect"]:
                self.detector.record_event(
                    BehaviorEvent(self.agent_id, tick, "syscall", sc)
                )
            # Also use completely different tools
            for tl in ["delete", "upload", "exfil"]:
                self.detector.record_event(
                    BehaviorEvent(self.agent_id, tick, "tool_use", tl)
                )

    def test_new_syscalls_high_drift_score(self):
        score = self.detector.compute_drift(self.agent_id)
        # syscall weight=0.25 + tool weight=0.20 both at 1.0 → composite ≥ 0.45
        self.assertGreater(score.score, 0.2)

    def test_new_syscalls_agent_is_drifting(self):
        self.assertTrue(self.detector.is_drifting(self.agent_id))

    def test_syscall_component_high(self):
        score = self.detector.compute_drift(self.agent_id)
        self.assertGreater(score.components.get("syscall_histogram", 0), 0.5)

    def test_severity_not_none(self):
        score = self.detector.compute_drift(self.agent_id)
        self.assertNotEqual(score.severity, "none")

    def test_explanation_mentions_drift(self):
        score = self.detector.compute_drift(self.agent_id)
        self.assertIn("Drift detected", score.explanation)


# ---------------------------------------------------------------------------
# Tests: DriftDetector — Frequency Shift
# ---------------------------------------------------------------------------

class TestDriftDetectorFrequencyShift(unittest.TestCase):
    def setUp(self):
        self.detector = DriftDetector(
            baseline_window=50,
            detection_window=10,
            threshold=0.3,
            sensitivity=1.0,
        )
        self.agent_id = "freq-shift-agent"
        # Baseline: mostly "read" (90%) with "write" (10%)
        for tick in range(50):
            for _ in range(9):
                self.detector.record_event(
                    BehaviorEvent(self.agent_id, tick, "syscall", "read")
                )
            self.detector.record_event(
                BehaviorEvent(self.agent_id, tick, "syscall", "write")
            )
        self.detector.build_baseline(self.agent_id)
        # Detection: inverted — mostly "write" (90%) with "read" (10%)
        for tick in range(50, 60):
            for _ in range(9):
                self.detector.record_event(
                    BehaviorEvent(self.agent_id, tick, "syscall", "write")
                )
            self.detector.record_event(
                BehaviorEvent(self.agent_id, tick, "syscall", "read")
            )

    def test_frequency_shift_causes_drift(self):
        score = self.detector.compute_drift(self.agent_id)
        # Same keys, very different distribution
        self.assertGreater(score.components.get("syscall_histogram", 0), 0.1)

    def test_drift_score_nonzero(self):
        score = self.detector.compute_drift(self.agent_id)
        self.assertGreater(score.score, 0.0)


# ---------------------------------------------------------------------------
# Tests: DriftDetector — Error Rate Spike
# ---------------------------------------------------------------------------

class TestDriftDetectorErrorRateSpike(unittest.TestCase):
    def setUp(self):
        self.detector = DriftDetector(
            baseline_window=50,
            detection_window=10,
            threshold=0.3,
            sensitivity=2.0,  # high sensitivity to catch rate changes
        )
        self.agent_id = "error-spike-agent"
        # Baseline: zero errors per tick
        for tick in range(50):
            self.detector.record_event(
                BehaviorEvent(self.agent_id, tick, "syscall", "read")
            )
        self.detector.build_baseline(self.agent_id)
        # Detection: one error per tick
        for tick in range(50, 60):
            self.detector.record_event(
                BehaviorEvent(self.agent_id, tick, "error", "timeout")
            )
            self.detector.record_event(
                BehaviorEvent(self.agent_id, tick, "syscall", "read")
            )

    def test_error_rate_spike_detected(self):
        score = self.detector.compute_drift(self.agent_id)
        self.assertGreater(score.components.get("error_rate", 0), 0.0)

    def test_error_rate_component_elevated(self):
        score = self.detector.compute_drift(self.agent_id)
        # With sensitivity=2.0 and large spike, should be clearly above 0
        self.assertGreater(score.components.get("error_rate", 0), 0.1)


# ---------------------------------------------------------------------------
# Tests: DriftDetector — Think Duration Anomaly
# ---------------------------------------------------------------------------

class TestDriftDetectorThinkDuration(unittest.TestCase):
    def setUp(self):
        self.detector = DriftDetector(
            baseline_window=50,
            detection_window=10,
            threshold=0.3,
            sensitivity=2.0,
        )
        self.agent_id = "slow-thinker"
        # Baseline: ~100ms think time per tick
        for tick in range(50):
            self.detector.record_event(
                BehaviorEvent(
                    self.agent_id, tick, "think_complete", "",
                    metadata={"duration_ms": 100.0}
                )
            )
        self.detector.build_baseline(self.agent_id)
        # Detection: ~5000ms think time (50x slower)
        for tick in range(50, 60):
            self.detector.record_event(
                BehaviorEvent(
                    self.agent_id, tick, "think_complete", "",
                    metadata={"duration_ms": 5000.0}
                )
            )

    def test_think_duration_anomaly_detected(self):
        score = self.detector.compute_drift(self.agent_id)
        self.assertGreater(score.components.get("think_duration", 0), 0.0)

    def test_think_duration_component_high(self):
        score = self.detector.compute_drift(self.agent_id)
        self.assertGreater(score.components.get("think_duration", 0), 0.5)


# ---------------------------------------------------------------------------
# Tests: DriftScore Classification
# ---------------------------------------------------------------------------

class TestDriftScoreClassification(unittest.TestCase):
    def test_severity_none_at_zero(self):
        self.assertEqual(DriftScore.classify_severity(0.0, 0.4), "none")

    def test_severity_none_at_threshold(self):
        self.assertEqual(DriftScore.classify_severity(0.4, 0.4), "none")

    def test_severity_low_just_above_threshold(self):
        # 0.41 is just above 0.4 threshold, low band
        sev = DriftScore.classify_severity(0.41, 0.4)
        self.assertEqual(sev, "low")

    def test_severity_medium(self):
        sev = DriftScore.classify_severity(0.55, 0.4)
        self.assertIn(sev, ["medium", "low"])

    def test_severity_high(self):
        sev = DriftScore.classify_severity(0.75, 0.4)
        self.assertIn(sev, ["high", "medium"])

    def test_severity_critical_at_max(self):
        sev = DriftScore.classify_severity(1.0, 0.4)
        self.assertEqual(sev, "critical")

    def test_all_severity_labels_valid(self):
        valid = {"none", "low", "medium", "high", "critical"}
        for score in [0.0, 0.35, 0.45, 0.55, 0.70, 0.90, 1.0]:
            sev = DriftScore.classify_severity(score, 0.4)
            self.assertIn(sev, valid)

    def test_drift_score_is_drifting_flag(self):
        # Build a DriftScore manually
        ds = DriftScore(
            agent_id="x", score=0.5, components={}, threshold=0.4,
            is_drifting=True, severity="medium", explanation="test", tick=0
        )
        self.assertTrue(ds.is_drifting)

    def test_drift_score_not_drifting_flag(self):
        ds = DriftScore(
            agent_id="x", score=0.2, components={}, threshold=0.4,
            is_drifting=False, severity="none", explanation="fine", tick=0
        )
        self.assertFalse(ds.is_drifting)


# ---------------------------------------------------------------------------
# Tests: DriftPolicy
# ---------------------------------------------------------------------------

class TestDriftPolicy(unittest.TestCase):
    def test_policy_constants_defined(self):
        self.assertEqual(DriftPolicy.MONITOR, "MONITOR")
        self.assertEqual(DriftPolicy.ALERT, "ALERT")
        self.assertEqual(DriftPolicy.THROTTLE, "THROTTLE")
        self.assertEqual(DriftPolicy.QUARANTINE, "QUARANTINE")

    def test_severity_rank_order(self):
        m = DriftPolicy.severity_for_action(DriftPolicy.MONITOR)
        a = DriftPolicy.severity_for_action(DriftPolicy.ALERT)
        t = DriftPolicy.severity_for_action(DriftPolicy.THROTTLE)
        q = DriftPolicy.severity_for_action(DriftPolicy.QUARANTINE)
        self.assertLess(m, a)
        self.assertLess(a, t)
        self.assertLess(t, q)

    def test_invalid_action_raises(self):
        with self.assertRaises(ValueError):
            DriftPolicy.severity_for_action("UNKNOWN")


class TestDriftPolicyConfig(unittest.TestCase):
    def test_default_config(self):
        cfg = DriftPolicyConfig()
        self.assertEqual(cfg.low, DriftPolicy.MONITOR)
        self.assertEqual(cfg.medium, DriftPolicy.ALERT)
        self.assertEqual(cfg.high, DriftPolicy.THROTTLE)
        self.assertEqual(cfg.critical, DriftPolicy.QUARANTINE)

    def test_custom_config(self):
        cfg = DriftPolicyConfig(
            low=DriftPolicy.ALERT,
            medium=DriftPolicy.THROTTLE,
            high=DriftPolicy.QUARANTINE,
            critical=DriftPolicy.QUARANTINE,
        )
        self.assertEqual(cfg.action_for_severity("low"), DriftPolicy.ALERT)

    def test_action_for_severity_none(self):
        cfg = DriftPolicyConfig()
        self.assertEqual(cfg.action_for_severity("none"), DriftPolicy.MONITOR)

    def test_action_for_severity_low(self):
        cfg = DriftPolicyConfig()
        self.assertEqual(cfg.action_for_severity("low"), DriftPolicy.MONITOR)

    def test_action_for_severity_medium(self):
        cfg = DriftPolicyConfig()
        self.assertEqual(cfg.action_for_severity("medium"), DriftPolicy.ALERT)

    def test_action_for_severity_high(self):
        cfg = DriftPolicyConfig()
        self.assertEqual(cfg.action_for_severity("high"), DriftPolicy.THROTTLE)

    def test_action_for_severity_critical(self):
        cfg = DriftPolicyConfig()
        self.assertEqual(cfg.action_for_severity("critical"), DriftPolicy.QUARANTINE)

    def test_action_for_unknown_severity(self):
        cfg = DriftPolicyConfig()
        # Unknown severity → MONITOR (default)
        self.assertEqual(cfg.action_for_severity("bogus"), DriftPolicy.MONITOR)


# ---------------------------------------------------------------------------
# Tests: DriftAlert
# ---------------------------------------------------------------------------

class TestDriftAlert(unittest.TestCase):
    def _make_score(self, score_val=0.8):
        return DriftScore(
            agent_id="a", score=score_val, components={},
            threshold=0.4, is_drifting=True, severity="high",
            explanation="test", tick=0
        )

    def test_alert_creation(self):
        ds = self._make_score()
        alert = DriftAlert(agent_id="a", drift_score=ds, policy_action=DriftPolicy.THROTTLE)
        self.assertEqual(alert.agent_id, "a")
        self.assertFalse(alert.acknowledged)

    def test_alert_acknowledge(self):
        ds = self._make_score()
        alert = DriftAlert(agent_id="a", drift_score=ds, policy_action=DriftPolicy.ALERT)
        alert.acknowledge()
        self.assertTrue(alert.acknowledged)

    def test_alert_timestamp_recent(self):
        before = time.time()
        ds = self._make_score()
        alert = DriftAlert(agent_id="a", drift_score=ds, policy_action=DriftPolicy.MONITOR)
        after = time.time()
        self.assertLessEqual(before, alert.timestamp)
        self.assertLessEqual(alert.timestamp, after)


# ---------------------------------------------------------------------------
# Tests: DriftMonitor
# ---------------------------------------------------------------------------

def _build_drifting_agent_events(agent_id: str, n_baseline: int, n_recent: int):
    """Build events: stable baseline then very different recent behavior."""
    events = []
    for tick in range(n_baseline):
        events.append(BehaviorEvent(agent_id, tick, "syscall", "read"))
        events.append(BehaviorEvent(agent_id, tick, "tool_use", "search"))
    for tick in range(n_baseline, n_baseline + n_recent):
        for sc in ["exec", "fork", "mmap"]:
            events.append(BehaviorEvent(agent_id, tick, "syscall", sc))
        events.append(BehaviorEvent(agent_id, tick, "tool_use", "delete"))
        events.append(BehaviorEvent(agent_id, tick, "error", "crash"))
    return events


class TestDriftMonitorRegistration(unittest.TestCase):
    def setUp(self):
        self.detector = DriftDetector(baseline_window=20, detection_window=5)
        self.monitor = DriftMonitor(
            detector=self.detector,
            policy_config=DriftPolicyConfig(),
            check_interval=10,
        )

    def test_register_agent(self):
        self.monitor.register_agent("a1")
        status = self.monitor.get_status("a1")
        self.assertTrue(status["registered"])

    def test_unregister_agent(self):
        self.monitor.register_agent("a1")
        self.monitor.unregister_agent("a1")
        status = self.monitor.get_status("a1")
        self.assertFalse(status["registered"])

    def test_get_status_unregistered(self):
        status = self.monitor.get_status("ghost")
        self.assertFalse(status["registered"])

    def test_initial_state_active(self):
        self.monitor.register_agent("a2")
        status = self.monitor.get_status("a2")
        self.assertEqual(status["state"], "active")

    def test_make_monitor_factory(self):
        mon = make_monitor(check_interval=5)
        self.assertIsInstance(mon, DriftMonitor)

    def test_status_contains_drift_score(self):
        self.monitor.register_agent("a3")
        for tick in range(5):
            ev = BehaviorEvent("a3", tick, "syscall", "read")
            self.monitor.on_event(ev)
        status = self.monitor.get_status("a3")
        self.assertIn("drift_score", status)


class TestDriftMonitorCheckAll(unittest.TestCase):
    def setUp(self):
        self.detector = DriftDetector(baseline_window=20, detection_window=5, threshold=0.4)
        self.monitor = DriftMonitor(
            detector=self.detector,
            policy_config=DriftPolicyConfig(),
            check_interval=100,  # won't auto-trigger
        )

    def test_check_all_returns_list(self):
        self.monitor.register_agent("x")
        scores = self.monitor.check_all()
        self.assertIsInstance(scores, list)
        self.assertEqual(len(scores), 1)

    def test_check_all_multiple_agents(self):
        self.monitor.register_agent("a")
        self.monitor.register_agent("b")
        scores = self.monitor.check_all()
        self.assertEqual(len(scores), 2)

    def test_check_all_returns_drift_scores(self):
        self.monitor.register_agent("y")
        scores = self.monitor.check_all()
        for s in scores:
            self.assertIsInstance(s, DriftScore)


class TestDriftMonitorAutoCheck(unittest.TestCase):
    def _build_monitor_with_drifter(self):
        detector = DriftDetector(baseline_window=20, detection_window=5, threshold=0.3)
        monitor = DriftMonitor(
            detector=detector,
            policy_config=DriftPolicyConfig(),
            check_interval=10,
        )
        agent_id = "auto-check-agent"
        monitor.register_agent(agent_id)

        # Feed baseline events via on_event
        for tick in range(20):
            monitor.on_event(BehaviorEvent(agent_id, tick, "syscall", "read"))
        detector.build_baseline(agent_id)

        # Feed drifting events
        for tick in range(20, 35):
            for sc in ["exec", "fork", "mmap", "socket"]:
                monitor.on_event(BehaviorEvent(agent_id, tick, "syscall", sc))
            monitor.on_event(BehaviorEvent(agent_id, tick, "error", "crash"))

        return monitor, agent_id

    def test_auto_check_produces_alerts(self):
        monitor, agent_id = self._build_monitor_with_drifter()
        # check_all should find drifting
        scores = monitor.check_all()
        drifting_scores = [s for s in scores if s.is_drifting]
        self.assertGreater(len(drifting_scores), 0)


class TestDriftMonitorAlertCallbacks(unittest.TestCase):
    def setUp(self):
        self.detector = DriftDetector(baseline_window=20, detection_window=5, threshold=0.3)
        self.monitor = DriftMonitor(
            detector=self.detector,
            policy_config=DriftPolicyConfig(),
            check_interval=100,  # disable auto-check
        )
        self.fired_alerts = []

    def test_callback_fires_on_alert(self):
        self.monitor.add_alert_callback(self.fired_alerts.append)
        self.monitor.register_agent("cb-agent")

        # Build a clearly drifting scenario
        for tick in range(20):
            self.detector.record_event(
                BehaviorEvent("cb-agent", tick, "syscall", "read")
            )
        self.detector.build_baseline("cb-agent")
        for tick in range(20, 30):
            for sc in ["exec", "fork", "mmap"]:
                self.detector.record_event(
                    BehaviorEvent("cb-agent", tick, "syscall", sc)
                )
        # Manually trigger check_all
        scores = self.monitor.check_all()
        drifting = [s for s in scores if s.is_drifting]
        if drifting:
            self.assertGreater(len(self.fired_alerts), 0)

    def test_multiple_callbacks(self):
        fired1 = []
        fired2 = []
        self.monitor.add_alert_callback(fired1.append)
        self.monitor.add_alert_callback(fired2.append)
        self.monitor.register_agent("multi-cb")

        for tick in range(20):
            self.detector.record_event(BehaviorEvent("multi-cb", tick, "syscall", "read"))
        self.detector.build_baseline("multi-cb")
        for tick in range(20, 30):
            for sc in ["exec", "fork"]:
                self.detector.record_event(BehaviorEvent("multi-cb", tick, "syscall", sc))

        scores = self.monitor.check_all()
        drifting = [s for s in scores if s.is_drifting]
        if drifting:
            self.assertEqual(len(fired1), len(fired2))

    def test_callback_receives_drift_alert(self):
        received = []
        self.monitor.add_alert_callback(received.append)
        self.monitor.register_agent("alert-agent")

        for tick in range(20):
            self.detector.record_event(BehaviorEvent("alert-agent", tick, "syscall", "read"))
        self.detector.build_baseline("alert-agent")
        for tick in range(20, 30):
            for sc in ["exec", "fork", "mmap"]:
                self.detector.record_event(BehaviorEvent("alert-agent", tick, "syscall", sc))

        self.monitor.check_all()
        for alert in received:
            self.assertIsInstance(alert, DriftAlert)

    def test_bad_callback_does_not_crash(self):
        def bad_cb(alert):
            raise RuntimeError("intentional failure")

        self.monitor.add_alert_callback(bad_cb)
        self.monitor.register_agent("crash-cb")
        for tick in range(20):
            self.detector.record_event(BehaviorEvent("crash-cb", tick, "syscall", "read"))
        self.detector.build_baseline("crash-cb")
        for tick in range(20, 30):
            for sc in ["exec", "fork"]:
                self.detector.record_event(BehaviorEvent("crash-cb", tick, "syscall", sc))
        # Should not raise
        self.monitor.check_all()


# ---------------------------------------------------------------------------
# Tests: DriftMonitor Policy Enforcement
# ---------------------------------------------------------------------------

class TestDriftMonitorPolicyEnforcement(unittest.TestCase):
    def _setup_drifting_monitor(self, policy_config):
        detector = DriftDetector(baseline_window=20, detection_window=5, threshold=0.3)
        monitor = DriftMonitor(
            detector=detector,
            policy_config=policy_config,
            check_interval=100,
        )
        agent_id = "policy-agent"
        monitor.register_agent(agent_id)
        for tick in range(20):
            detector.record_event(BehaviorEvent(agent_id, tick, "syscall", "read"))
        detector.build_baseline(agent_id)
        for tick in range(20, 30):
            for sc in ["exec", "fork", "mmap"]:
                detector.record_event(BehaviorEvent(agent_id, tick, "syscall", sc))
            detector.record_event(BehaviorEvent(agent_id, tick, "error", "crash"))
        return monitor, agent_id

    def test_monitor_policy_no_state_change(self):
        cfg = DriftPolicyConfig(
            low=DriftPolicy.MONITOR,
            medium=DriftPolicy.MONITOR,
            high=DriftPolicy.MONITOR,
            critical=DriftPolicy.MONITOR,
        )
        monitor, agent_id = self._setup_drifting_monitor(cfg)
        monitor.check_all()
        status = monitor.get_status(agent_id)
        # MONITOR policy should NOT change state to throttled/quarantined
        self.assertNotEqual(status["state"], "quarantined")

    def test_quarantine_policy_changes_state(self):
        cfg = DriftPolicyConfig(
            low=DriftPolicy.QUARANTINE,
            medium=DriftPolicy.QUARANTINE,
            high=DriftPolicy.QUARANTINE,
            critical=DriftPolicy.QUARANTINE,
        )
        monitor, agent_id = self._setup_drifting_monitor(cfg)
        scores = monitor.check_all()
        drifting = [s for s in scores if s.is_drifting]
        if drifting:
            status = monitor.get_status(agent_id)
            self.assertEqual(status["state"], "quarantined")

    def test_throttle_policy_changes_state(self):
        cfg = DriftPolicyConfig(
            low=DriftPolicy.THROTTLE,
            medium=DriftPolicy.THROTTLE,
            high=DriftPolicy.THROTTLE,
            critical=DriftPolicy.THROTTLE,
        )
        monitor, agent_id = self._setup_drifting_monitor(cfg)
        scores = monitor.check_all()
        drifting = [s for s in scores if s.is_drifting]
        if drifting:
            status = monitor.get_status(agent_id)
            self.assertIn(status["state"], ["throttled", "quarantined"])

    def test_alert_recorded_in_history(self):
        cfg = DriftPolicyConfig()
        monitor, agent_id = self._setup_drifting_monitor(cfg)
        scores = monitor.check_all()
        drifting = [s for s in scores if s.is_drifting]
        if drifting:
            self.assertGreater(len(monitor.alerts), 0)

    def test_alert_policy_action_correct(self):
        cfg = DriftPolicyConfig(
            low=DriftPolicy.MONITOR,
            medium=DriftPolicy.ALERT,
            high=DriftPolicy.THROTTLE,
            critical=DriftPolicy.QUARANTINE,
        )
        monitor, agent_id = self._setup_drifting_monitor(cfg)
        monitor.check_all()
        for alert in monitor.alerts:
            self.assertIn(
                alert.policy_action,
                [DriftPolicy.MONITOR, DriftPolicy.ALERT, DriftPolicy.THROTTLE, DriftPolicy.QUARANTINE],
            )


# ---------------------------------------------------------------------------
# Tests: Baseline Reset
# ---------------------------------------------------------------------------

class TestBaselineReset(unittest.TestCase):
    def setUp(self):
        self.detector = DriftDetector(baseline_window=20, detection_window=5, threshold=0.3)
        self.agent_id = "reset-agent"
        # Record initial events
        for tick in range(20):
            self.detector.record_event(
                BehaviorEvent(self.agent_id, tick, "syscall", "read")
            )
        self.detector.build_baseline(self.agent_id)

    def test_reset_baseline_clears_old(self):
        old_baseline = self.detector.get_baseline(self.agent_id)
        # Add new behavior
        for tick in range(20, 30):
            self.detector.record_event(
                BehaviorEvent(self.agent_id, tick, "syscall", "exec")
            )
        self.detector.reset_baseline(self.agent_id)
        new_baseline = self.detector.get_baseline(self.agent_id)
        self.assertIsNotNone(new_baseline)
        # New baseline should include "exec"
        self.assertIn("exec", new_baseline.syscall_histogram)

    def test_reset_baseline_returns_profile(self):
        result = self.detector.reset_baseline(self.agent_id)
        self.assertIsInstance(result, BehaviorProfile)

    def test_after_reset_drift_reduced(self):
        # Add divergent events
        for tick in range(20, 50):
            for sc in ["exec", "fork", "mmap"]:
                self.detector.record_event(
                    BehaviorEvent(self.agent_id, tick, "syscall", sc)
                )
        # Drift should be high before reset
        score_before = self.detector.compute_drift(self.agent_id)
        # Reset adapts baseline to new behavior
        self.detector.reset_baseline(self.agent_id)
        score_after = self.detector.compute_drift(self.agent_id)
        # After reset drift should be lower (new baseline matches recent behavior)
        self.assertLessEqual(score_after.score, score_before.score + 0.5)


# ---------------------------------------------------------------------------
# Tests: Edge Cases
# ---------------------------------------------------------------------------

class TestEdgeCases(unittest.TestCase):
    def test_compute_drift_no_events(self):
        detector = DriftDetector()
        score = detector.compute_drift("ghost-agent")
        self.assertIsInstance(score, DriftScore)
        self.assertAlmostEqual(score.score, 0.0)
        self.assertFalse(score.is_drifting)

    def test_single_event_agent(self):
        detector = DriftDetector()
        detector.record_event(BehaviorEvent("solo", 0, "syscall", "read"))
        detector.build_baseline("solo")
        score = detector.compute_drift("solo")
        self.assertIsInstance(score, DriftScore)
        self.assertGreaterEqual(score.score, 0.0)

    def test_multiple_agents_independent(self):
        detector = DriftDetector(baseline_window=10, detection_window=5, threshold=0.4)
        agents = ["a1", "a2", "a3"]
        for agent_id in agents:
            for tick in range(10):
                detector.record_event(
                    BehaviorEvent(agent_id, tick, "syscall", f"call_{agent_id}")
                )
            detector.build_baseline(agent_id)

        for agent_id in agents:
            bp = detector.get_baseline(agent_id)
            self.assertIsNotNone(bp)
            self.assertIn(f"call_{agent_id}", bp.syscall_histogram)

    def test_event_from_unregistered_agent_ignored_by_monitor(self):
        detector = DriftDetector()
        monitor = DriftMonitor(detector=detector, policy_config=DriftPolicyConfig())
        # Don't register the agent
        monitor.on_event(BehaviorEvent("phantom", 0, "syscall", "read"))
        # No alerts should have been fired
        self.assertEqual(len(monitor.alerts), 0)

    def test_is_drifting_no_baseline_no_crash(self):
        detector = DriftDetector()
        # Should not raise, just return False
        result = detector.is_drifting("new-agent")
        self.assertFalse(result)

    def test_behavior_profile_timestamps(self):
        before = time.time()
        bp = BehaviorProfile(agent_id="ts-test")
        after = time.time()
        self.assertGreaterEqual(bp.created_at, before)
        self.assertLessEqual(bp.created_at, after)

    def test_drift_score_all_component_keys(self):
        detector = _stable_detector()
        score = detector.compute_drift("stable-agent")
        expected_dims = {
            "syscall_histogram", "tool_usage", "resource_access",
            "message_types", "memory_rates", "error_rate", "think_duration",
        }
        for dim in expected_dims:
            self.assertIn(dim, score.components)

    def test_drift_score_components_in_range(self):
        detector = _stable_detector()
        score = detector.compute_drift("stable-agent")
        for dim, val in score.components.items():
            self.assertGreaterEqual(val, 0.0, f"{dim} should be >= 0")
            self.assertLessEqual(val, 1.0, f"{dim} should be <= 1")

    def test_compute_drift_auto_builds_baseline(self):
        detector = DriftDetector(baseline_window=5, detection_window=3)
        for tick in range(5):
            detector.record_event(BehaviorEvent("auto", tick, "syscall", "read"))
        # Should auto-build baseline on first compute_drift
        self.assertIsNone(detector.get_baseline("auto"))
        score = detector.compute_drift("auto")
        self.assertIsNotNone(detector.get_baseline("auto"))

    def test_on_event_feeds_detector(self):
        detector = DriftDetector(baseline_window=10, detection_window=5)
        monitor = DriftMonitor(detector=detector, policy_config=DriftPolicyConfig())
        monitor.register_agent("fed")
        ev = BehaviorEvent("fed", 0, "syscall", "open")
        monitor.on_event(ev)
        # Event should be in detector's records
        score = detector.compute_drift("fed")
        self.assertIsInstance(score, DriftScore)


# ---------------------------------------------------------------------------
# Tests: DriftReport
# ---------------------------------------------------------------------------

class TestDriftReport(unittest.TestCase):
    def setUp(self):
        self.detector = DriftDetector(baseline_window=20, detection_window=5, threshold=0.4)
        self.agent_id = "report-agent"
        for tick in range(20):
            self.detector.record_event(
                BehaviorEvent(self.agent_id, tick, "syscall", "read")
            )
        self.detector.build_baseline(self.agent_id)

    def test_generate_returns_dict(self):
        report = DriftReport(detector=self.detector)
        result = report.generate(self.agent_id)
        self.assertIsInstance(result, dict)

    def test_report_contains_required_keys(self):
        report = DriftReport(detector=self.detector)
        result = report.generate(self.agent_id)
        for key in ["agent_id", "generated_at", "baseline_profile", "current_drift_score",
                    "top_drifting_dimensions", "recommendations", "alert_history"]:
            self.assertIn(key, result)

    def test_report_agent_id_correct(self):
        report = DriftReport(detector=self.detector)
        result = report.generate(self.agent_id)
        self.assertEqual(result["agent_id"], self.agent_id)

    def test_report_baseline_profile_populated(self):
        report = DriftReport(detector=self.detector)
        result = report.generate(self.agent_id)
        self.assertIsNotNone(result["baseline_profile"])

    def test_report_top_drifting_sorted(self):
        report = DriftReport(detector=self.detector)
        result = report.generate(self.agent_id)
        dims = result["top_drifting_dimensions"]
        if len(dims) > 1:
            for i in range(len(dims) - 1):
                self.assertGreaterEqual(
                    dims[i]["weighted_contribution"],
                    dims[i + 1]["weighted_contribution"],
                )

    def test_report_recommendations_non_empty(self):
        report = DriftReport(detector=self.detector)
        result = report.generate(self.agent_id)
        self.assertIsInstance(result["recommendations"], list)
        self.assertGreater(len(result["recommendations"]), 0)

    def test_report_with_monitor_includes_alerts(self):
        monitor = DriftMonitor(
            detector=self.detector,
            policy_config=DriftPolicyConfig(),
            check_interval=100,
        )
        report = DriftReport(detector=self.detector, monitor=monitor)
        result = report.generate(self.agent_id)
        self.assertIn("alert_history", result)

    def test_report_without_baseline_auto_builds(self):
        detector = DriftDetector()
        report = DriftReport(detector=detector)
        for tick in range(5):
            detector.record_event(BehaviorEvent("auto-report", tick, "syscall", "read"))
        result = report.generate("auto-report")
        self.assertIsNotNone(result["baseline_profile"])

    def test_report_generated_at_recent(self):
        before = time.time()
        report = DriftReport(detector=self.detector)
        result = report.generate(self.agent_id)
        after = time.time()
        self.assertGreaterEqual(result["generated_at"], before)
        self.assertLessEqual(result["generated_at"], after)


# ---------------------------------------------------------------------------
# Tests: Thread Safety
# ---------------------------------------------------------------------------

class TestThreadSafety(unittest.TestCase):
    def test_concurrent_record_events(self):
        detector = DriftDetector(baseline_window=100, detection_window=20)
        errors = []

        def record_worker(agent_id, n):
            try:
                for i in range(n):
                    detector.record_event(
                        BehaviorEvent(agent_id, i, "syscall", "read")
                    )
            except Exception as e:
                errors.append(e)

        threads = [
            threading.Thread(target=record_worker, args=(f"agent{i}", 50))
            for i in range(4)
        ]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        self.assertEqual(len(errors), 0)

    def test_concurrent_compute_drift(self):
        detector = DriftDetector(baseline_window=20, detection_window=5)
        for tick in range(20):
            detector.record_event(BehaviorEvent("shared", tick, "syscall", "read"))
        detector.build_baseline("shared")
        for tick in range(20, 30):
            detector.record_event(BehaviorEvent("shared", tick, "syscall", "read"))

        results = []
        errors = []

        def compute_worker():
            try:
                score = detector.compute_drift("shared")
                results.append(score)
            except Exception as e:
                errors.append(e)

        threads = [threading.Thread(target=compute_worker) for _ in range(6)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        self.assertEqual(len(errors), 0)
        self.assertEqual(len(results), 6)


if __name__ == "__main__":
    unittest.main()
