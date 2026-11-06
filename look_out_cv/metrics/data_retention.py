"""Data retention management for LookOutCV segment-based log files.

Works on *whole immutable segment files* — it never loads or rewrites row data.
Row counts are read from Parquet metadata (no data scan needed).

The row count after retention can be slightly under ``max_rows`` because the
boundary falls on a segment boundary, not a row boundary.  This is documented
behaviour.

Retention policies
------------------
- ``KEEP_LATEST``      — delete the oldest segments until ≤ max_rows remain.
- ``ARCHIVE_AND_TRIM`` — move excess segments to ``<model>/archive/``.

The *archive* is the ``<model>/archive/`` sub-directory.  :meth:`read_log`
(and therefore rollup / drift / insights) ignores the archive sub-directory by
design.
"""
from __future__ import annotations

import glob
import os
import shutil
from enum import Enum, auto
from typing import List, Optional

import pyarrow.parquet as pq

from look_out_cv._utils import _live_segments as _get_live_segments


class RetentionPolicy(Enum):
    """Available retention policies."""

    KEEP_LATEST = auto()
    """Keep only the N most-recent rows (whole segments); discard older ones."""

    ARCHIVE_AND_TRIM = auto()
    """Move excess segments to ``<model>/archive/`` instead of deleting them."""


class DataRetentionManager:
    """Manage the size of parquet log directories by applying a retention policy.

    Parameters
    ----------
    model_name:
        Name of the model whose logs should be managed.
    max_rows:
        Maximum number of rows to keep in the live log directory.
        Because retention works on whole segments, the actual row count after
        a trim may be slightly under ``max_rows``.
    policy:
        ``RetentionPolicy.KEEP_LATEST`` (default) — excess segments are deleted.
        ``RetentionPolicy.ARCHIVE_AND_TRIM`` — excess segments are moved to
        ``<model>/archive/``.
    logs_dir:
        Root directory where the model's log directories live.
    max_bytes:
        Optional additional constraint.  When set, oldest segments are also
        removed until the total size of live segments is ≤ ``max_bytes``.
        Applied *after* the ``max_rows`` constraint.

    Note
    ----
    ``read_log`` loads the retained log into RAM. On edge devices (such as a
    1 GB Raspberry Pi 3), an unbounded log cannot fit in memory, so retention
    must actually be run periodically. Retention is not auto-run inside the
    logger.
    """

    def __init__(
        self,
        model_name: str,
        max_rows: int = 100_000,
        policy: RetentionPolicy = RetentionPolicy.KEEP_LATEST,
        logs_dir: str = "lookout_cv_logs",
        max_bytes: Optional[int] = None,
        archive_suffix: str = "_archive",
    ) -> None:
        if max_rows < 1:
            raise ValueError("``max_rows`` must be at least 1.")

        self.model_name = model_name
        self.max_rows = max_rows
        self.policy = policy
        self.logs_dir = logs_dir
        self.max_bytes = max_bytes

        self._model_dir = os.path.join(logs_dir, model_name)
        self._archive_dir = os.path.join(self._model_dir, "archive")

    def _live_segments(self) -> List[str]:
        """Return all live (non-archive, non-tmp) segment files, sorted."""
        return _get_live_segments(self._model_dir)

    @staticmethod
    def _segment_row_count(path: str) -> int:
        """Read row count from Parquet metadata — no data loaded."""
        return pq.ParquetFile(path).metadata.num_rows

    @staticmethod
    def _segment_bytes(path: str) -> int:
        return os.path.getsize(path)

    def _dispose(self, path: str) -> None:
        """Delete or archive a segment according to ``self.policy``."""
        if self.policy == RetentionPolicy.ARCHIVE_AND_TRIM:
            os.makedirs(self._archive_dir, exist_ok=True)
            shutil.move(path, os.path.join(self._archive_dir, os.path.basename(path)))
        else:
            os.remove(path)

    def current_row_count(self) -> int:
        """Return the total number of rows currently in the live log."""
        total = 0
        for f in self._live_segments():
            total += self._segment_row_count(f)
        return total

    def apply(self) -> dict:
        """Apply the configured retention policy.

        Returns
        -------
        dict
            Summary with keys: ``rows_before``, ``rows_after``,
            ``rows_removed``, ``rows_archived``, ``policy``.
        """
        segments = self._live_segments()
        if not segments:
            raise FileNotFoundError(
                f"No parquet segment files found in '{self._model_dir}'."
            )

        counts = [self._segment_row_count(f) for f in segments]
        rows_before = sum(counts)

        if rows_before <= self.max_rows:
            rows_archived = 0
            rows_removed = 0
            if self.max_bytes is not None:
                ar, rr = self._apply_max_bytes(segments, counts)
                rows_archived += ar
                rows_removed += rr
            rows_after = self.current_row_count()
            return {
                "rows_before":  rows_before,
                "rows_after":   rows_after,
                "rows_removed": rows_removed,
                "rows_archived": rows_archived,
                "policy": self.policy.name,
            }

        rows_archived = 0
        rows_removed = 0
        running_total = rows_before

        i = 0
        while i < len(segments) and running_total > self.max_rows:
            seg, cnt = segments[i], counts[i]
            self._dispose(seg)
            running_total -= cnt
            if self.policy == RetentionPolicy.ARCHIVE_AND_TRIM:
                rows_archived += cnt
            else:
                rows_removed += cnt
            i += 1

        remaining_segs = segments[i:]
        remaining_counts = counts[i:]
        if self.max_bytes is not None:
            ar, rr = self._apply_max_bytes(remaining_segs, remaining_counts)
            rows_archived += ar
            rows_removed += rr

        rows_after = self.current_row_count()

        return {
            "rows_before":   rows_before,
            "rows_after":    rows_after,
            "rows_removed":  rows_removed,
            "rows_archived": rows_archived,
            "policy":        self.policy.name,
        }

    def _apply_max_bytes(
        self, segments: List[str], counts: List[int]
    ) -> tuple:
        """Trim oldest segments until total bytes ≤ max_bytes.

        Returns (rows_archived, rows_removed) for the disposed segments.
        """
        assert self.max_bytes is not None
        rows_archived = rows_removed = 0
        total_bytes = sum(self._segment_bytes(f) for f in segments)
        i = 0
        while i < len(segments) and total_bytes > self.max_bytes:
            seg = segments[i]
            total_bytes -= self._segment_bytes(seg)
            self._dispose(seg)
            if self.policy == RetentionPolicy.ARCHIVE_AND_TRIM:
                rows_archived += counts[i]
            else:
                rows_removed += counts[i]
            i += 1
        return rows_archived, rows_removed

    def purge_archive(self) -> bool:
        """Delete all files in the archive directory. Returns True if any deleted."""
        if not os.path.isdir(self._archive_dir):
            return False
        files = glob.glob(os.path.join(self._archive_dir, "*.parquet"))
        if not files:
            return False
        for f in files:
            os.remove(f)
        return True

    def archive_exists(self) -> bool:
        """Return True if the archive directory contains at least one parquet file."""
        if not os.path.isdir(self._archive_dir):
            return False
        return bool(glob.glob(os.path.join(self._archive_dir, "*.parquet")))

    def archive_row_count(self) -> int:
        """Return the total number of rows in the archive directory (0 if empty)."""
        if not os.path.isdir(self._archive_dir):
            return 0
        total = 0
        for f in glob.glob(os.path.join(self._archive_dir, "*.parquet")):
            total += self._segment_row_count(f)
        return total
