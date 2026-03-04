"""
test_logger.py — Tests for battousai.logger
"""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import unittest

from battousai.logger import Logger, LogLevel, LogEntry


class TestLogLevel(unittest.TestCase):

    def test_log_level_has_required_values(self):
        names = [l.name for l in LogLevel]
        for expected in ["DEBUG", "INFO", "WARN", "ERROR", "SYSTEM"]:
            self.assertIn(expected, names)

    def test_log_level_ordering(self):
        """DEBUG < INFO < WARN < ERROR (lower value = lower severity)."""
        self.assertLess(LogLevel.DEBUG.value, LogLevel.INFO.value)
        self.assertLess(LogLevel.INFO.value, LogLevel.WARN.value)
        self.assertLess(LogLevel.WARN.value, LogLevel.ERROR.value)


class TestLogger(unittest.TestCase):

    def setUp(self):
        self.logger = Logger(min_level=LogLevel.DEBUG, max_entries=100,
                             console_output=False)

    def test_debug_message_logged(self):
        self.logger.debug("test_module", "debug message")
        entries = self.logger.get_entries()
        self.assertGreater(len(entries), 0)

    def test_info_message_logged(self):
        self.logger.info("test_module", "info message")
        entries = self.logger.get_entries(min_level=LogLevel.INFO)
        self.assertGreater(len(entries), 0)

    def test_warn_message_logged(self):
        self.logger.warn("test_module", "warning message")
        entries = self.logger.get_entries(min_level=LogLevel.WARN)
        self.assertGreater(len(entries), 0)

    def test_error_message_logged(self):
        self.logger.error("test_module", "error message")
        entries = self.logger.get_entries(min_level=LogLevel.ERROR)
        self.assertGreater(len(entries), 0)

    def test_system_message_logged(self):
        self.logger.system("kernel", "system boot")
        entries = self.logger.get_entries(min_level=LogLevel.SYSTEM)
        self.assertGreater(len(entries), 0)

    def test_min_level_filters_lower_levels(self):
        """Logger with min_level=WARN should not store DEBUG entries."""
        logger = Logger(min_level=LogLevel.WARN, max_entries=100,
                        console_output=False)
        logger.debug("m", "this should be filtered")
        entries = logger.get_entries()
        debug_entries = [e for e in entries if e.level == LogLevel.DEBUG]
        self.assertEqual(len(debug_entries), 0)

    def test_get_entries_returns_list_of_log_entries(self):
        self.logger.info("m", "test")
        entries = self.logger.get_entries()
        self.assertIsInstance(entries, list)
        for entry in entries:
            self.assertIsInstance(entry, LogEntry)

    def test_get_entries_filtered_by_level(self):
        """get_entries(min_level=ERROR) returns only ERROR and above."""
        self.logger.info("m", "info msg")
        self.logger.error("m", "error msg")
        error_entries = self.logger.get_entries(min_level=LogLevel.ERROR)
        for entry in error_entries:
            self.assertGreaterEqual(entry.level.value, LogLevel.ERROR.value)

    def test_max_entries_limit_respected(self):
        """Logger should not grow beyond max_entries."""
        logger = Logger(min_level=LogLevel.DEBUG, max_entries=10,
                        console_output=False)
        for i in range(20):
            logger.info("m", f"message {i}")
        entries = logger.get_entries()
        self.assertLessEqual(len(entries), 10)

    def test_get_summary_returns_string(self):
        self.logger.info("m", "test")
        self.logger.error("m", "error")
        summary = self.logger.get_summary()
        self.assertIsInstance(summary, str)
        self.assertGreater(len(summary), 0)

    def test_log_entry_has_source_and_message(self):
        self.logger.info("my_module", "my message")
        entries = self.logger.get_entries()
        entry = entries[-1]
        # LogEntry uses 'source' not 'module'
        self.assertIn("my_module", str(getattr(entry, 'source', '') or getattr(entry, 'module', '')))
        self.assertIn("my message", entry.message)

    def test_multiple_modules_tracked(self):
        self.logger.info("module_a", "msg from a")
        self.logger.info("module_b", "msg from b")
        entries = self.logger.get_entries()
        self.assertGreaterEqual(len(entries), 2)


if __name__ == "__main__":
    unittest.main()
