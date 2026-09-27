"""Data retention management for LookOutCV log files.

Provides policies to archive or delete old prediction log rows from parquet
files, keeping storage usage bounded without losing all historical data.

Since the logs don't store timestamps, retention is row-count based.
"""

from __future__ import annotations

import glob
import os
from enum import Enum, auto
from typing import List, Optional

import pyarrow as pa
import pyarrow.parquet as pq


class RetentionPolicy(Enum):
    """Available retention policies."""

    KEEP_LATEST = auto()
    """Keep only the N most-recent rows; discard the rest."""

    ARCHIVE_AND_TRIM = auto()
    """Move excess rows to a separate archive parquet file, then trim the live log."""


class DataRetentionManager:
    """Manage the size of parquet log files by applying a retention policy.

    Parameters
    ----------
    model_name:
        Name of the model whose logs should be managed.
    max_rows:
        Maximum number of rows to keep in the live log file.
    policy:
        ``RetentionPolicy.KEEP_LATEST`` (default) - excess rows are deleted.
        ``RetentionPolicy.ARCHIVE_AND_TRIM`` - excess rows are appended to an
        archive parquet file before being removed from the live log.
    logs_dir:
        Root directory where the model's log parquet files live.
    archive_suffix:
        Filename suffix for archive parquet files when using
        ``ARCHIVE_AND_TRIM``.
    """

    def __init__(
        self,
        model_name: str,
        max_rows: int,
        policy: RetentionPolicy = RetentionPolicy.KEEP_LATEST,
        logs_dir: str = "lookout_cv_logs",
        archive_suffix: str = "_archive",
    ) -> None:
        if max_rows < 1:
            raise ValueError("``max_rows`` must be at least 1.")

        self.model_name = model_name
        self.max_rows = max_rows
        self.policy = policy
        self.logs_dir = logs_dir
        self.archive_suffix = archive_suffix

        self._model_dir = os.path.join(logs_dir, model_name)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _find_live_files(self) -> List[str]:
        """Return all non-archive parquet files for this model."""
        all_files = glob.glob(os.path.join(self._model_dir, "*.parquet"))
        archive_name = self._archive_path()
        return [f for f in all_files if os.path.abspath(f) != os.path.abspath(archive_name)]

    def _archive_path(self) -> str:
        return os.path.join(
            self._model_dir,
            f"{self.model_name}_logs{self.archive_suffix}.parquet",
        )

    def _read_combined(self, files: List[str]) -> pa.Table:
        if not files:
            raise FileNotFoundError(
                f"No parquet log files found in '{self._model_dir}'."
            )
        return pa.concat_tables([pq.read_table(f) for f in files])

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def current_row_count(self) -> int:
        """Return the total number of rows currently in the live log."""
        files = self._find_live_files()
        if not files:
            return 0
        return self._read_combined(files).num_rows

    def apply(self) -> dict:
        """Apply the configured retention policy.

        Returns
        -------
        dict
            Summary with keys: ``rows_before``, ``rows_after``,
            ``rows_removed``, ``rows_archived``, ``policy``.
        """
        files = self._find_live_files()
        if not files:
            raise FileNotFoundError(
                f"No parquet log files found in '{self._model_dir}'."
            )

        table = self._read_combined(files)
        rows_before = table.num_rows

        if rows_before <= self.max_rows:
            return {
                "rows_before": rows_before,
                "rows_after": rows_before,
                "rows_removed": 0,
                "rows_archived": 0,
                "policy": self.policy.name,
            }

        excess = rows_before - self.max_rows
        rows_to_archive = table.slice(0, excess)   # oldest
        rows_to_keep = table.slice(excess)          # newest

        rows_archived = 0
        if self.policy == RetentionPolicy.ARCHIVE_AND_TRIM:
            archive_path = self._archive_path()
            os.makedirs(os.path.dirname(archive_path), exist_ok=True)
            if os.path.exists(archive_path):
                existing_archive = pq.read_table(archive_path)
                rows_to_archive = pa.concat_tables([existing_archive, rows_to_archive])
            pq.write_table(rows_to_archive, archive_path)
            rows_archived = excess

        # Write back trimmed live log
        primary_file = os.path.join(
            self._model_dir, f"{self.model_name}_logs.parquet"
        )
        pq.write_table(rows_to_keep, primary_file)

        # Remove any secondary live files that were consolidated
        for f in files:
            if os.path.abspath(f) != os.path.abspath(primary_file):
                os.remove(f)

        return {
            "rows_before": rows_before,
            "rows_after": rows_to_keep.num_rows,
            "rows_removed": excess,
            "rows_archived": rows_archived,
            "policy": self.policy.name,
        }

    def purge_archive(self) -> bool:
        """Delete the archive parquet file if it exists. Returns True if deleted."""
        path = self._archive_path()
        if os.path.exists(path):
            os.remove(path)
            return True
        return False

    def archive_exists(self) -> bool:
        """Return True if an archive file exists for this model."""
        return os.path.exists(self._archive_path())

    def archive_row_count(self) -> int:
        """Return the number of rows in the archive file (0 if none exists)."""
        path = self._archive_path()
        if not os.path.exists(path):
            return 0
        return pq.read_table(path).num_rows
