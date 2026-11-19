"""Verify bounded hash reuse and detection of files changing during inspection."""

import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from ailias.common.file_checksums import FileChecksums
from ailias.storage import digest


class FileChecksumsTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.path = Path(temporary.name) / "output.bin"
        self.path.write_bytes(b"first")

    def test_unchanged_files_reuse_hash_but_same_size_edits_invalidate_it(self):
        checksums = FileChecksums()
        with patch("ailias.common.file_checksums.digest", wraps=digest) as calculate:
            original = checksums(self.path)
            self.assertEqual(checksums(self.path), original)
            self.assertEqual(calculate.call_count, 1)
            modified = self.path.stat().st_mtime_ns + 1_000_000_000
            self.path.write_bytes(b"other")
            os.utime(self.path, ns=(modified, modified))
            self.assertNotEqual(checksums(self.path), original)
            self.assertEqual(calculate.call_count, 2)

    def test_file_changing_during_hash_is_rejected_without_caching_bad_result(self):
        checksums = FileChecksums()

        def mutate(path):
            checksum = digest(path)
            path.write_bytes(b"changed while reading")
            return checksum

        with patch("ailias.common.file_checksums.digest", side_effect=mutate):
            with self.assertRaises(ValueError):
                checksums(self.path)
        self.assertEqual(checksums(self.path), digest(self.path))
