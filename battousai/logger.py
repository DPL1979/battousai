"""
logger.py — Battousai Structured Logger
====================================
System-wide structured logging for the Autonomous Intelligence Operating System.

All components and agents log through this module. Logs are stored both
in-memory (ring buffer) and in the virtual filesystem at /system/logs/.

Log Levels (ascending severity):
    DEBUG   — Verbose internal state, used during development
    INFO    — Normal operational events
    WARN    — Unexpected-but-recoverable situations
    ERROR   — Failures requiring attention
    SYSTEM  — OS-level lifecycle events (boot, shutdown, agent spawn/kill)
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from enum import IntEnum
from typing import Optional, List, Dict, Any


class LogLevel(IntEnum):
    """Severity levels for log entries."""
    DEBUG = 0
    INFO = 1
    WARN = 2
    ERROR = 3
    SYSTEM = 4


# ANSI color codes for terminal output
_LEVEL_COLORS: Dict[LogLevel, str] = {
    LogLevel.DEBUG:  "\033[90m",    # dark gray
    LogLevel.INFO:   "\033[36m",    # cyan
    LogLevel.WARN:   "\033[33m",    # yellow
    LogLevel.ERROR:  "\033[31m",    # red
    LogLevel.SYSTEM: "\033[35m",    # magenta
}
_RESET = "\033[0m"
_BOLD  = "\033[1m"

_LEVEL_LABELS: Dict[LogLevel, str] = {
    LogLevel.DEBUG:  "DEBUG ",
    LogLevel.INFO:   "INFO  ",
    LogLevel.WARN:   "WARN  ",
    LogLevel.ERROR:  "ERROR ",
    LogLevel.SYSTEM: "SYSTEM",
}


@dataclass
class LogEntry:
    """A single structured log record."""
    tick: int
    level: LogLevel
    source: str          # agent_id or "kernel", "scheduler", etc.
    message: str
    data: Optional[Dict[str, Any]] = field(default=None)
    wall_time: float = field(default_factory=time.time)

    def __str__(self) -> str:
        color = _LEVEL_COLORS.get(self.level, "")
        label = _LEVEL_LABELS.get(self.level, "UNKNWN")
        data_str = f" | {self.data}" if self.data else ""
        return (
            f"{color}[tick={self.tick:04d}] {label} "
            f"{_BOLD}{self.source:<20}{_RESET}{color} "
            f"{self.message}{data_str}{_RESET}"
        )

    def to_dict(self) -> Dict[str, Any]:
        return {
            "tick": self.tick,
            "level": self.level.name,
            "source": self.source,
            "message": self.message,
            "data": self.data,
            "wall_time": self.wall_time,
        }


class Logger:
    """
    System-wide structured logger for Battousai.

    Features:
    - Configurable minimum log level (filters out lower-severity entries)
    - In-memory ring buffer (capped at max_entries)
    - Console output with ANSI color coding
    - Entry retrieval by level, source, or tick range
    - Integration with the virtual filesystem (injected after FS is ready)
    """

    def __init__(
        self,
        min_level: LogLevel = LogLevel.INFO,
        max_entries: int = 10_000,
        console_output: bool = True,
    ) -> None:
        self.min_level = min_level
        self.max_entries = max_entries
        self.console_output = console_output
        self._entries: List[LogEntry] = []
        self._counts: Dict[LogLevel, int] = {lvl: 0 for lvl in LogLevel}
        self._filesystem = None  # Injected by kernel after FS init
        self._current_tick: int = 0

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _set_tick(self, tick: int) -> None:
        """Called by the kernel each tick to update the running clock."""
        self._current_tick = tick

    def _inject_filesystem(self, fs) -> None:
        """Inject the virtual filesystem once it is initialised."""
        self._filesystem = fs

    def _persist(self, entry: LogEntry) -> None:
        """Write the entry to /system/logs/ if the FS is available."""
        if self._filesystem is None:
            return
        try:
            path = f"/system/logs/tick_{entry.tick:04d}.log"
            # Append mode: read existing content then overwrite
            existing = ""
            try:
                existing = self._filesystem.read_file("kernel", path) + "\n"
            except Exception:
                pass
            self._filesystem.write_file(
                "kernel", path, existing + str(entry), create_parents=True
            )
        except Exception:
            pass  # Never let logging crash the OS

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def log(
        self,
        level: LogLevel,
        source: str,
        message: str,
        data: Optional[Dict[str, Any]] = None,
    ) -> Optional[LogEntry]:
        """Create and store a log entry. Returns None if filtered by min_level."""
        if level < self.min_level:
            return None

        entry = LogEntry(
            tick=self._current_tick,
            level=level,
            source=source,
            message=message,
            data=data,
        )

        # Maintain ring buffer
        if len(self._entries) >= self.max_entries:
            self._entries.pop(0)
        self._entries.append(entry)
        self._counts[level] += 1

        if self.console_output:
            print(str(entry))

        self._persist(entry)
        return entry

    # Convenience methods
    def debug(self, source: str, message: str, data: Optional[Dict] = None) -> Optional[LogEntry]:
        return self.log(LogLevel.DEBUG, source, message, data)

    def info(self, source: str, message: str, data: Optional[Dict] = None) -> Optional[LogEntry]:
        return self.log(LogLevel.INFO, source, message, data)

    def warn(self, source: str, message: str, data: Optional[Dict] = None) -> Optional[LogEntry]:
        return self.log(LogLevel.WARN, source, message, data)

    def error(self, source: str, message: str, data: Optional[Dict] = None) -> Optional[LogEntry]:
        return self.log(LogLevel.ERROR, source, message, data)

    def system(self, source: str, message: str, data: Optional[Dict] = None) -> Optional[LogEntry]:
        return self.log(LogLevel.SYSTEM, source, message, data)

    # ------------------------------------------------------------------
    # Query interface
    # ------------------------------------------------------------------

    def get_entries(
        self,
        min_level: Optional[LogLevel] = None,
        source: Optional[str] = None,
        tick_start: Optional[int] = None,
        tick_end: Optional[int] = None,
    ) -> List[LogEntry]:
        """Return filtered log entries."""
        results = self._entries
        if min_level is not None:
            results = [e for e in results if e.level >= min_level]
        if source is not None:
            results = [e for e in results if e.source == source]
        if tick_start is not None:
            results = [e for e in results if e.tick >= tick_start]
        if tick_end is not None:
            results = [e for e in results if e.tick <= tick_end]
        return results

    def get_counts(self) -> Dict[str, int]:
        """Return a summary of log entries per level."""
        return {lvl.name: count for lvl, count in self._counts.items()}

    def get_summary(self) -> str:
        """Return a human-readable summary string."""
        counts = self.get_counts()
        parts = [f"{k}={v}" for k, v in counts.items() if v > 0]
        return "LogSummary(" + ", ".join(parts) + ")"
