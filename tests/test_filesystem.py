"""
test_filesystem.py — Tests for battousai.filesystem (VirtualFilesystem)
"""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import unittest

from battousai.filesystem import (
    VirtualFilesystem, FSError,
)
# Import subclasses if present, or use the base FSError
try:
    from battousai.filesystem import FileNotFoundError as VFSFileNotFoundError
    from battousai.filesystem import PermissionError as VFSPermissionError
    from battousai.filesystem import NotADirectoryError as VFSNotADirectoryError
except ImportError:
    VFSFileNotFoundError = FSError
    VFSPermissionError = FSError
    VFSNotADirectoryError = FSError


class TestVirtualFilesystemInit(unittest.TestCase):

    def setUp(self):
        self.fs = VirtualFilesystem()
        self.fs._init_standard_dirs()  # Required: not called by __init__

    def test_standard_dirs_created_on_init(self):
        """VFS must create /agents, /shared, /system on init."""
        listing = self.fs.list_dir("anyone", "/")
        self.assertIn("agents", listing)
        self.assertIn("shared", listing)
        self.assertIn("system", listing)

    def test_shared_results_subdir_exists(self):
        listing = self.fs.list_dir("anyone", "/shared")
        self.assertIn("results", listing)

    def test_system_logs_subdir_exists(self):
        listing = self.fs.list_dir("anyone", "/system")
        self.assertIn("logs", listing)


class TestVirtualFilesystemWriteRead(unittest.TestCase):

    def setUp(self):
        self.fs = VirtualFilesystem()
        self.fs._init_standard_dirs()

    def test_write_and_read_file_round_trip(self):
        self.fs.write_file("agent_a", "/agents/test.txt", "hello world")
        data = self.fs.read_file("agent_a", "/agents/test.txt")
        self.assertEqual(data, "hello world")

    def test_read_nonexistent_file_raises(self):
        with self.assertRaises(FSError):
            self.fs.read_file("agent_a", "/agents/no_such_file.txt")

    def test_overwrite_file_with_overwrite_flag(self):
        self.fs.write_file("agent_a", "/agents/file.txt", "v1")
        self.fs.write_file("agent_a", "/agents/file.txt", "v2", overwrite=True)
        data = self.fs.read_file("agent_a", "/agents/file.txt")
        self.assertEqual(data, "v2")

    def test_write_without_overwrite_flag_raises_on_existing_file(self):
        self.fs.write_file("agent_a", "/agents/existing.txt", "original")
        with self.assertRaises(FSError):
            self.fs.write_file(
                "agent_a", "/agents/existing.txt", "new", overwrite=False
            )

    def test_write_file_with_create_parents(self):
        """Writing to a deep path with create_parents=True must create dirs."""
        self.fs.write_file(
            "agent_a", "/agents/a/b/c/deep.txt", "deep content",
            create_parents=True
        )
        data = self.fs.read_file("agent_a", "/agents/a/b/c/deep.txt")
        self.assertEqual(data, "deep content")

    def test_world_readable_file_can_be_read_by_anyone(self):
        self.fs.write_file(
            "agent_a", "/shared/public.txt", "public data",
            world_readable=True
        )
        data = self.fs.read_file("agent_b", "/shared/public.txt")
        self.assertEqual(data, "public data")

    def test_private_file_cannot_be_read_by_other_agent(self):
        """A non-world-readable file should be inaccessible to other agents."""
        self.fs.write_file(
            "agent_a", "/agents/private.txt", "secret",
            world_readable=False, overwrite=True
        )
        with self.assertRaises(FSError):
            self.fs.read_file("agent_b", "/agents/private.txt")


class TestVirtualFilesystemListing(unittest.TestCase):

    def setUp(self):
        self.fs = VirtualFilesystem()
        self.fs._init_standard_dirs()

    def test_list_dir_returns_list(self):
        listing = self.fs.list_dir("agent_a", "/")
        self.assertIsInstance(listing, list)

    def test_list_dir_shows_written_files(self):
        self.fs.write_file("agent_a", "/agents/report.txt", "data")
        listing = self.fs.list_dir("agent_a", "/agents")
        self.assertIn("report.txt", listing)

    def test_mkdir_creates_directory(self):
        self.fs.mkdir("/agents/mydir")
        listing = self.fs.list_dir("agent_a", "/agents")
        self.assertIn("mydir", listing)

    def test_tree_returns_string(self):
        result = self.fs.tree("agent_a", "/")
        self.assertIsInstance(result, str)
        self.assertGreater(len(result), 0)


class TestVirtualFilesystemMetadata(unittest.TestCase):

    def setUp(self):
        self.fs = VirtualFilesystem()
        self.fs._init_standard_dirs()

    def test_get_metadata_returns_file_metadata(self):
        self.fs.write_file("agent_a", "/agents/meta.txt", "data")
        meta = self.fs.get_metadata("/agents/meta.txt")
        self.assertIsNotNone(meta)
        self.assertEqual(meta.created_by, "agent_a")

    def test_get_metadata_records_file_size(self):
        content = "hello world"
        self.fs.write_file("agent_a", "/agents/sized.txt", content)
        meta = self.fs.get_metadata("/agents/sized.txt")
        self.assertGreater(meta.size, 0)

    def test_stats_returns_dict(self):
        stats = self.fs.stats()
        self.assertIsInstance(stats, dict)


if __name__ == "__main__":
    unittest.main()
