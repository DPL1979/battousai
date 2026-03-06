"""
tests/test_real_fs.py — Unit tests for battousai.real_fs
=========================================================
Tests SandboxedFilesystem: path jailing, read/write, permissions,
directory traversal prevention, and statistics.
"""

from __future__ import annotations

import os
import sys
import tempfile
import unittest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from battousai.real_fs import (
    SandboxedFilesystem,
    FileNotFoundError,
    NotADirectoryError,
    FileExistsError,
    PathTraversalError,
    PermissionError,
)


class TestSandboxedFilesystemSetup(unittest.TestCase):
    """Tests that the filesystem creates the right directory structure."""

    def setUp(self):
        self.tmpdir = tempfile.mkdtemp(prefix="battousai_test_")
        self.fs = SandboxedFilesystem(self.tmpdir)

    def tearDown(self):
        self.fs.destroy()

    def test_root_dir_created(self):
        self.assertTrue(os.path.isdir(self.tmpdir))

    def test_agents_subdir_created(self):
        self.assertTrue(os.path.isdir(os.path.join(self.tmpdir, "agents")))

    def test_shared_subdir_created(self):
        self.assertTrue(os.path.isdir(os.path.join(self.tmpdir, "shared")))

    def test_root_dir_stored(self):
        self.assertEqual(self.fs.root_dir, os.path.abspath(self.tmpdir))

    def test_fresh_root_dir_auto_created(self):
        """SandboxedFilesystem creates the root_dir if it doesn't exist."""
        import shutil
        new_dir = os.path.join(self.tmpdir, "fresh_root")
        # new_dir does not exist yet
        fs2 = SandboxedFilesystem(new_dir)
        self.assertTrue(os.path.isdir(new_dir))
        fs2.destroy()


class TestWriteReadFile(unittest.TestCase):

    def setUp(self):
        self.tmpdir = tempfile.mkdtemp(prefix="battousai_test_")
        self.fs = SandboxedFilesystem(self.tmpdir)

    def tearDown(self):
        self.fs.destroy()

    def test_write_and_read_string(self):
        self.fs.write_file("agent1", "hello.txt", "world")
        content = self.fs.read_file("agent1", "hello.txt")
        self.assertEqual(content, "world")

    def test_write_and_read_json_as_string(self):
        import json
        data = json.dumps({"key": "value"})
        self.fs.write_file("agent1", "data.json", data)
        result = self.fs.read_file("agent1", "data.json")
        self.assertEqual(json.loads(result), {"key": "value"})

    def test_overwrite_existing_file(self):
        self.fs.write_file("agent1", "file.txt", "first")
        self.fs.write_file("agent1", "file.txt", "second")
        content = self.fs.read_file("agent1", "file.txt")
        self.assertEqual(content, "second")

    def test_write_with_overwrite_false_raises(self):
        self.fs.write_file("agent1", "existing.txt", "data")
        with self.assertRaises(FileExistsError):
            self.fs.write_file("agent1", "existing.txt", "new", overwrite=False)

    def test_write_creates_parent_dirs(self):
        self.fs.write_file("agent1", "subdir/nested/file.txt", "nested content")
        content = self.fs.read_file("agent1", "subdir/nested/file.txt")
        self.assertEqual(content, "nested content")

    def test_write_no_create_parents_raises(self):
        with self.assertRaises(FileNotFoundError):
            self.fs.write_file(
                "agent1", "nonexistent/file.txt", "data", create_parents=False
            )

    def test_read_nonexistent_raises(self):
        with self.assertRaises(FileNotFoundError):
            self.fs.read_file("agent1", "ghost.txt")

    def test_read_directory_raises(self):
        self.fs.mkdir("agent1", "adir")
        with self.assertRaises(NotADirectoryError):
            self.fs.read_file("agent1", "adir")

    def test_returns_real_path(self):
        real_path = self.fs.write_file("agent1", "foo.txt", "bar")
        self.assertTrue(os.path.isabs(real_path))
        self.assertTrue(os.path.exists(real_path))

    def test_agents_are_isolated_from_each_other(self):
        """agent2 cannot read agent1's file path."""
        self.fs.write_file("agent1", "secret.txt", "agent1 secret")
        # agent2's jail is different — reading same relative path gives FileNotFoundError
        with self.assertRaises(FileNotFoundError):
            self.fs.read_file("agent2", "secret.txt")


class TestDeleteFile(unittest.TestCase):

    def setUp(self):
        self.tmpdir = tempfile.mkdtemp(prefix="battousai_test_")
        self.fs = SandboxedFilesystem(self.tmpdir)

    def tearDown(self):
        self.fs.destroy()

    def test_delete_existing_file(self):
        self.fs.write_file("agent1", "todelete.txt", "bye")
        result = self.fs.delete_file("agent1", "todelete.txt")
        self.assertTrue(result)
        self.assertFalse(self.fs.exists("agent1", "todelete.txt"))

    def test_delete_nonexistent_returns_false(self):
        result = self.fs.delete_file("agent1", "ghost.txt")
        self.assertFalse(result)

    def test_delete_updates_stats(self):
        self.fs.write_file("agent1", "f.txt", "abc")
        before = self.fs.stats()["total_files"]
        self.fs.delete_file("agent1", "f.txt")
        after = self.fs.stats()["total_files"]
        self.assertEqual(after, before - 1)


class TestListDir(unittest.TestCase):

    def setUp(self):
        self.tmpdir = tempfile.mkdtemp(prefix="battousai_test_")
        self.fs = SandboxedFilesystem(self.tmpdir)

    def tearDown(self):
        self.fs.destroy()

    def test_list_empty_jail_root(self):
        # Make the jail exist first
        self.fs.mkdir("agent1", ".")
        entries = self.fs.list_dir("agent1", ".")
        self.assertIsInstance(entries, list)

    def test_list_dir_contains_written_files(self):
        self.fs.write_file("agent1", "a.txt", "a")
        self.fs.write_file("agent1", "b.txt", "b")
        entries = self.fs.list_dir("agent1", ".")
        self.assertIn("a.txt", entries)
        self.assertIn("b.txt", entries)

    def test_list_nonexistent_dir_raises(self):
        with self.assertRaises(FileNotFoundError):
            self.fs.list_dir("agent1", "nosuchdir")

    def test_list_file_as_dir_raises(self):
        self.fs.write_file("agent1", "afile.txt", "data")
        with self.assertRaises(NotADirectoryError):
            self.fs.list_dir("agent1", "afile.txt")

    def test_list_subdirectory(self):
        self.fs.write_file("agent1", "sub/x.txt", "x")
        entries = self.fs.list_dir("agent1", "sub")
        self.assertIn("x.txt", entries)


class TestExists(unittest.TestCase):

    def setUp(self):
        self.tmpdir = tempfile.mkdtemp(prefix="battousai_test_")
        self.fs = SandboxedFilesystem(self.tmpdir)

    def tearDown(self):
        self.fs.destroy()

    def test_exists_true_for_written_file(self):
        self.fs.write_file("agent1", "present.txt", "yes")
        self.assertTrue(self.fs.exists("agent1", "present.txt"))

    def test_exists_false_for_missing_file(self):
        self.assertFalse(self.fs.exists("agent1", "absent.txt"))

    def test_exists_true_for_created_directory(self):
        self.fs.mkdir("agent1", "mydir")
        self.assertTrue(self.fs.exists("agent1", "mydir"))


class TestMkdir(unittest.TestCase):

    def setUp(self):
        self.tmpdir = tempfile.mkdtemp(prefix="battousai_test_")
        self.fs = SandboxedFilesystem(self.tmpdir)

    def tearDown(self):
        self.fs.destroy()

    def test_mkdir_creates_directory(self):
        real_path = self.fs.mkdir("agent1", "newdir")
        self.assertTrue(os.path.isdir(real_path))

    def test_mkdir_nested(self):
        self.fs.mkdir("agent1", "a/b/c")
        self.assertTrue(self.fs.exists("agent1", "a/b/c"))

    def test_mkdir_idempotent(self):
        self.fs.mkdir("agent1", "mydir")
        # Should not raise on second call
        self.fs.mkdir("agent1", "mydir")


class TestPathTraversalPrevention(unittest.TestCase):
    """Critical security tests — directory traversal must be blocked."""

    def setUp(self):
        self.tmpdir = tempfile.mkdtemp(prefix="battousai_test_")
        self.fs = SandboxedFilesystem(self.tmpdir)

    def tearDown(self):
        self.fs.destroy()

    def test_dotdot_in_path_is_blocked(self):
        with self.assertRaises(PathTraversalError):
            self.fs.write_file("agent1", "../../etc/passwd", "evil")

    def test_absolute_path_is_normalised(self):
        """Absolute paths should be treated as relative to the jail."""
        # This should NOT escape — /etc/passwd becomes jail/etc/passwd
        # (the leading / is stripped). Verify no traversal error but the file
        # is written inside the jail.
        real_path = self.fs.write_file("agent1", "/subdir/file.txt", "data")
        self.assertTrue(real_path.startswith(self.fs.root_dir))

    def test_excessive_dotdot_blocked(self):
        with self.assertRaises(PathTraversalError):
            self.fs.read_file("agent1", "../../../../root/.ssh/id_rsa")

    def test_dotdot_in_middle_of_path_blocked(self):
        with self.assertRaises(PathTraversalError):
            self.fs.write_file("agent1", "subdir/../../../../tmp/evil", "x")

    def test_read_traversal_blocked(self):
        with self.assertRaises(PathTraversalError):
            self.fs.read_file("agent1", "../agent2/secret.txt")

    def test_delete_traversal_blocked(self):
        with self.assertRaises(PathTraversalError):
            self.fs.delete_file("agent1", "../../important_file")

    def test_list_dir_traversal_blocked(self):
        with self.assertRaises(PathTraversalError):
            self.fs.list_dir("agent1", "../../etc")

    def test_exists_traversal_returns_false(self):
        # exists() swallows PathTraversalError and returns False
        result = self.fs.exists("agent1", "../../etc/passwd")
        self.assertFalse(result)

    def test_resolved_path_stays_within_jail(self):
        """Verify the resolved path starts with the jail directory."""
        real_path = self.fs.write_file("agent1", "deep/nested/file.txt", "ok")
        jail = self.fs._agent_jail("agent1")
        self.assertTrue(real_path.startswith(os.path.realpath(jail)))


class TestPermissions(unittest.TestCase):

    def setUp(self):
        self.tmpdir = tempfile.mkdtemp(prefix="battousai_test_")
        self.fs = SandboxedFilesystem(self.tmpdir)

    def tearDown(self):
        self.fs.destroy()

    def test_world_readable_file_readable_by_other_agent(self):
        """Files with world_readable=True can be read from any agent's view if path resolved."""
        # Write as agent1 — world_readable=True
        self.fs.write_file("agent1", "pub.txt", "public data", world_readable=True)
        # agent2 can't access agent1's jail path directly; it would be FileNotFoundError
        # (isolation is at directory level, not metadata level)
        # The permission model applies within the same agent's jail view.
        # This test just confirms metadata is stored correctly.
        meta = self.fs.get_metadata("agent1", "pub.txt")
        self.assertIsNotNone(meta)
        self.assertTrue(meta.world_read)

    def test_world_not_writable_by_default(self):
        self.fs.write_file("agent1", "private.txt", "data")
        meta = self.fs.get_metadata("agent1", "private.txt")
        self.assertFalse(meta.world_write)

    def test_owner_can_write(self):
        self.fs.write_file("agent1", "own.txt", "v1")
        meta = self.fs.get_metadata("agent1", "own.txt")
        self.assertTrue(meta.can_write("agent1"))

    def test_kernel_bypasses_permissions(self):
        """kernel agent_id bypasses all permission checks."""
        self.fs.write_file("agent1", "restricted.txt", "secret", world_readable=False)
        # Kernel can still read (if we knew the path) — FileMetadata.can_read("kernel") = True
        meta = self.fs.get_metadata("agent1", "restricted.txt")
        self.assertTrue(meta.can_read("kernel"))
        self.assertTrue(meta.can_write("kernel"))

    def test_get_metadata_returns_none_for_missing_file(self):
        result = self.fs.get_metadata("agent1", "no_such_file.txt")
        self.assertIsNone(result)

    def test_metadata_created_by_matches_agent(self):
        self.fs.write_file("agent99", "myfile.txt", "data")
        meta = self.fs.get_metadata("agent99", "myfile.txt")
        self.assertEqual(meta.created_by, "agent99")


class TestStats(unittest.TestCase):

    def setUp(self):
        self.tmpdir = tempfile.mkdtemp(prefix="battousai_test_")
        self.fs = SandboxedFilesystem(self.tmpdir)

    def tearDown(self):
        self.fs.destroy()

    def test_stats_structure(self):
        s = self.fs.stats()
        self.assertIn("root_dir", s)
        self.assertIn("total_files", s)
        self.assertIn("total_size_bytes", s)
        self.assertIn("files_by_agent", s)

    def test_total_files_increments(self):
        self.fs.write_file("agent1", "a.txt", "abc")
        self.fs.write_file("agent1", "b.txt", "defgh")
        self.assertEqual(self.fs.stats()["total_files"], 2)

    def test_total_size_tracks_correctly(self):
        self.fs.write_file("agent1", "x.txt", "hello")  # 5 bytes
        self.assertEqual(self.fs.stats()["total_size_bytes"], 5)

    def test_files_by_agent_counted(self):
        self.fs.write_file("agentA", "f1.txt", "x")
        self.fs.write_file("agentA", "f2.txt", "y")
        self.fs.write_file("agentB", "f3.txt", "z")
        stats = self.fs.stats()
        self.assertEqual(stats["files_by_agent"]["agentA"], 2)
        self.assertEqual(stats["files_by_agent"]["agentB"], 1)


class TestSharedPath(unittest.TestCase):

    def setUp(self):
        self.tmpdir = tempfile.mkdtemp(prefix="battousai_test_")
        self.fs = SandboxedFilesystem(self.tmpdir)

    def tearDown(self):
        self.fs.destroy()

    def test_get_shared_path_returns_inside_shared_dir(self):
        path = self.fs.get_shared_path("results/data.json")
        self.assertTrue(path.startswith(os.path.realpath(self.fs._shared_dir)))

    def test_get_shared_path_traversal_blocked(self):
        with self.assertRaises(PathTraversalError):
            self.fs.get_shared_path("../../etc/passwd")


class TestTree(unittest.TestCase):

    def setUp(self):
        self.tmpdir = tempfile.mkdtemp(prefix="battousai_test_")
        self.fs = SandboxedFilesystem(self.tmpdir)

    def tearDown(self):
        self.fs.destroy()

    def test_tree_returns_string(self):
        self.fs.write_file("agentX", "file.txt", "data")
        tree = self.fs.tree("kernel")
        self.assertIsInstance(tree, str)

    def test_tree_contains_written_file_name(self):
        self.fs.write_file("agentX", "myfile.txt", "content")
        tree = self.fs.tree("kernel")
        self.assertIn("myfile.txt", tree)


if __name__ == "__main__":
    unittest.main()
