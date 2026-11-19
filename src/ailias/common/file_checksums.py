"""Reuse hashes only within one inspection while file identity remains unchanged."""

from ailias.storage import digest


class FileChecksums:
    def __init__(self):
        self._values = {}

    @staticmethod
    def _identity(path):
        stat = path.stat()
        return (
            stat.st_dev,
            stat.st_ino,
            stat.st_size,
            stat.st_mtime_ns,
            stat.st_ctime_ns,
        )

    def __call__(self, path):
        before = self._identity(path)
        if path in self._values and self._values[path][0] == before:
            return self._values[path][1]
        checksum = digest(path)
        if self._identity(path) != before:
            raise ValueError("File changed while computing its checksum")
        self._values[path] = before, checksum
        return checksum
