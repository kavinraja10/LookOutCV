"""Shared utilities for LookOutCV."""
from __future__ import annotations

_NON_METRIC_COLS = frozenset({"ts", "row_id", "device_id", "image_name"})

import glob
import os
from typing import List, Optional

import pyarrow as pa
import pyarrow.parquet as pq


def _live_segments(model_dir: str) -> List[str]:
    """Return sorted committed segment files, excluding .tmp and archive/ subdir."""
    all_files = sorted(glob.glob(os.path.join(model_dir, "*.parquet")))
    model_dir_abs = os.path.abspath(model_dir)
    return [
        f for f in all_files
        if os.path.dirname(os.path.abspath(f)) == model_dir_abs
    ]


def read_log(
    logs_dir: str,
    model_name: str,
    columns: Optional[List[str]] = None,
    since_ms: Optional[int] = None,
) -> pa.Table:
    """Read all committed segment files for *model_name* and return a combined table.

    Segments are read in chronological order (filename sort, which matches
    insertion order because names start with a millisecond timestamp).

    Older segments that are missing columns introduced by newer segments will
    have ``null`` values for those columns (via ``promote_options="default"``).

    Parameters
    ----------
    logs_dir:
        Root directory where model log directories live.
    model_name:
        Logical name of the model whose logs should be read.
    columns:
        If provided, only these columns are returned.  Segments that do not
        contain a requested column will have that column filled with nulls.
    since_ms:
        If provided, skip segments whose filename timestamp prefix (first 13
        characters, interpreted as an integer millisecond epoch) is less than
        this value.  This is a fast, coarse filter — it skips whole segments,
        not individual rows.

    Raises
    ------
    FileNotFoundError
        When no committed ``*.parquet`` segment files exist for the model.
    """
    model_dir = os.path.join(logs_dir, model_name)
    segment_files = _live_segments(model_dir)

    if since_ms is not None:
        kept = []
        for f in segment_files:
            basename = os.path.basename(f)
            try:
                seg_ms = int(basename[:13])
            except ValueError:
                kept.append(f)
                continue
            if seg_ms >= since_ms:
                kept.append(f)
        segment_files = kept

    if not segment_files:
        raise FileNotFoundError(
            f"No parquet segment files found in '{model_dir}'."
        )

    tables: List[pa.Table] = []
    for f in segment_files:
        if columns is not None:
            schema = pq.read_schema(f)
            existing = [c for c in columns if c in schema.names]
            tbl = pq.read_table(f, columns=existing)
            for col in columns:
                if col not in tbl.schema.names:
                    null_arr = pa.array([None] * tbl.num_rows, type=pa.null())
                    tbl = tbl.append_column(pa.field(col, pa.null()), null_arr)
            tbl = tbl.select(columns)
        else:
            tbl = pq.read_table(f)
        tables.append(tbl)

    if len(tables) == 1:
        return tables[0]
    return pa.concat_tables(tables, promote_options="default")


def compact(logs_dir: str, model_name: str, min_segments: int = 20) -> int:
    """Merge oldest segments into one, streaming one segment at a time.

    Writes with :class:`pyarrow.parquet.ParquetWriter` (one row group per
    source segment).  Never loads all data into RAM at once.

    Crash safety: writes to ``.tmp``, ``os.replace`` onto the newest merged
    source's name, then deletes older sources oldest-first.  A crash leaves
    duplicate rows but no lost rows; a second ``compact()`` run on
    ``(device_id, row_id)`` deduplicates.  Because the merged file keeps the
    newest source's name, export cursors stay valid; rows re-sent after
    compaction are ignored by the primary key (``ON CONFLICT DO NOTHING``).

    No-op if fewer than *min_segments* segments exist.

    On edge devices, prefer a larger ``buffer_size`` to reduce segment count.
    Run ``compact()`` (or ``lookoutcv compact``) periodically to keep
    ``read_log`` fast.

    Parameters
    ----------
    logs_dir:
        Root directory where model log directories live.
    model_name:
        Logical name of the model whose logs should be compacted.
    min_segments:
        Minimum number of live segments required before compaction runs.

    Returns
    -------
    int
        Number of live segments after compaction (0 if no-op / fewer than
        ``min_segments`` existed).
    """
    model_dir = os.path.join(logs_dir, model_name)
    segments = _live_segments(model_dir)

    if len(segments) < min_segments:
        return 0  # no-op

    # Leave the newest few segments alone (a live writer may be producing them).
    if len(segments) >= 6:
        to_merge = segments[:-3]
    elif len(segments) > 2:
        to_merge = segments[:-1]
    else:
        to_merge = segments

    if len(to_merge) < 2:
        return len(segments)

    schemas = [pq.read_schema(f) for f in to_merge]
    unified_schema = pa.unify_schemas(schemas)

    newest_merged = to_merge[-1]
    tmp_path = newest_merged + ".tmp"

    seen_keys: set[tuple[object, object]] = set()
    writer = pq.ParquetWriter(tmp_path, unified_schema)
    try:
        for f in to_merge:
            tbl = pq.read_table(f)
            # Deduplicate by (device_id, row_id) if both columns exist
            if "device_id" in tbl.column_names and "row_id" in tbl.column_names:
                devs = tbl["device_id"].to_pylist()
                rids = tbl["row_id"].to_pylist()
                mask = []
                for dev, rid in zip(devs, rids):
                    key = (dev, rid)
                    if key in seen_keys:
                        mask.append(False)
                    else:
                        seen_keys.add(key)
                        mask.append(True)
                if not all(mask):
                    tbl = tbl.filter(pa.array(mask))

            if tbl.num_rows == 0:
                continue

            for field in unified_schema:
                if field.name not in tbl.schema.names:
                    null_arr = pa.array([None] * tbl.num_rows, type=field.type)
                    tbl = tbl.append_column(field, null_arr)

            tbl = tbl.select(unified_schema.names)
            tbl = tbl.cast(unified_schema)
            writer.write_table(tbl)
    finally:
        writer.close()

    os.replace(tmp_path, newest_merged)

    for f in to_merge[:-1]:  # delete older sources oldest-first
        try:
            os.remove(f)
        except FileNotFoundError:
            pass

    return len(_live_segments(model_dir))


def migrate_legacy(logs_dir: str, model: str) -> bool:
    """Migrate an old ``<model>_logs.parquet`` file to the segment layout.

    Reads the legacy monolithic parquet, fills ``ts`` from the file's mtime
    (in UTC epoch ms), assigns fresh ``row_id`` and ``device_id`` values,
    writes a segment file, then removes the old file.

    Returns True if a migration was performed, False otherwise.
    """
    import socket
    import time
    import uuid

    legacy_path = os.path.join(logs_dir, model, f"{model}_logs.parquet")
    if not os.path.exists(legacy_path):
        return False

    table = pq.read_table(legacy_path)
    mtime_ms = int(os.path.getmtime(legacy_path) * 1000)
    n = table.num_rows
    device_id = socket.gethostname()

    if "ts" not in table.schema.names:
        table = table.append_column(
            pa.field("ts", pa.int64()),
            pa.array([mtime_ms] * n, type=pa.int64()),
        )
    if "row_id" not in table.schema.names:
        table = table.append_column(
            pa.field("row_id", pa.string()),
            pa.array([uuid.uuid4().hex for _ in range(n)], type=pa.string()),
        )
    if "device_id" not in table.schema.names:
        table = table.append_column(
            pa.field("device_id", pa.string()),
            pa.array([device_id] * n, type=pa.string()),
        )

    model_dir = os.path.join(logs_dir, model)
    seg_name = f"{time.time_ns() // 1_000_000:013d}_0_0.parquet"
    tmp_path = os.path.join(model_dir, seg_name + ".tmp")
    final_path = os.path.join(model_dir, seg_name)
    pq.write_table(table, tmp_path)
    os.replace(tmp_path, final_path)

    os.remove(legacy_path)
    return True
