"""BaseLogger: append-only, crash-safe segment-based parquet logger."""
from __future__ import annotations

import os
import socket
import time
import uuid
from typing import Any, Dict, List, Optional, Union

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

from look_out_cv.metrics.metrics import ImageMetricsCalculator, resolve_metric_method_name
from look_out_cv.metrics_types import CVMetrics


_FIELD_TYPES: Dict[str, pa.DataType] = {
    "ts":        pa.int64(),
    "row_id":    pa.string(),
    "device_id": pa.string(),
    "image_name": pa.string(),
    "pred_class":  pa.string(),
    "num_boxes": pa.int32(),
}


def _field_type(name: str) -> pa.DataType:
    return _FIELD_TYPES.get(name, pa.float32())


class BaseLogger:
    """Segment-based prediction logger (crash-safe, append-only).

    Each call to :meth:`flush` writes ONE new immutable segment file to
    ``<logs_dir>/<model_name>/``.  Files are named::

        <ms_timestamp>_<pid>_<seq>.parquet

    A ``.tmp`` file is written first, then ``os.replace()`` atomically renames
    it, so a power cut during a flush never corrupts committed segments.

    Parameters
    ----------
    model_name:
        Logical name of the model being monitored.
    enabled_metrics:
        List of :class:`CVMetrics` to compute and store per prediction.
    logs_dir:
        Root directory for all model log directories.
    buffer_size:
        Number of predictions to buffer before auto-flushing.
    device_id:
        Identifier for this device.  Defaults to ``socket.gethostname()``.
    """

    #: Subclasses must declare the mandatory prediction fields.
    _MANDATORY_FIELDS: List[str] = []

    def __init__(
        self,
        model_name: str,
        enabled_metrics: Optional[List[CVMetrics]] = None,
        logs_dir: str = "lookout_cv_logs",
        buffer_size: int = 1,
        device_id: Optional[str] = None,
    ) -> None:
        self.model_name = model_name
        self.enabled_metrics = list(enabled_metrics or [])
        self.logs_dir = logs_dir
        self.buffer_size = max(1, int(buffer_size))
        self.device_id = device_id or socket.gethostname()
        self._pending_rows: List[Dict[str, Any]] = []
        self._flush_seq: int = 0

        model_dir = os.path.join(self.logs_dir, self.model_name)
        os.makedirs(model_dir, exist_ok=True)

    def __enter__(self) -> "BaseLogger":
        return self

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> bool:
        self.flush()
        return False

    def _create_schema(self) -> pa.Schema:
        fields = [
            pa.field("ts",        pa.int64()),
            pa.field("row_id",    pa.string()),
            pa.field("device_id", pa.string()),
        ]
        for field_name in self._MANDATORY_FIELDS:
            fields.append(pa.field(field_name, _field_type(field_name)))
        for metric in self.enabled_metrics:
            fields.append(pa.field(metric.value, pa.float32()))
        return pa.schema(fields)

    def _resolve_metric_method_name(self, metric: CVMetrics) -> str:
        return resolve_metric_method_name(metric)

    def _compute_metric_value(
        self, calculator: ImageMetricsCalculator, metric: CVMetrics
    ) -> Optional[float]:
        method_name = self._resolve_metric_method_name(metric)
        try:
            return float(getattr(calculator, method_name)())
        except (AttributeError, TypeError, ValueError):
            return None

    @staticmethod
    def _normalize_field_value(value: Any, field_type: pa.DataType) -> Any:
        if value is None:
            return None
        try:
            if pa.types.is_integer(field_type):
                return int(value)
            if pa.types.is_floating(field_type):
                return float(value)
            if pa.types.is_boolean(field_type):
                return bool(value)
            if pa.types.is_string(field_type):
                return str(value)
        except (TypeError, ValueError):
            return None
        return value

    def calculate_image_metrics(
        self, image: Optional[Union[str, "np.ndarray"]] = None
    ) -> Dict[str, Optional[float]]:
        """Calculate metrics for the provided image."""
        results: Dict[str, Optional[float]] = {
            metric.value: None for metric in self.enabled_metrics
        }
        if image is None or not self.enabled_metrics:
            return results

        try:
            calculator = ImageMetricsCalculator(image)
        except Exception:
            return results

        for metric in self.enabled_metrics:
            results[metric.value] = self._compute_metric_value(calculator, metric)
        return results

    def log_prediction(self, **kwargs: Any) -> None:
        """Store one prediction record with the configured image metrics.

        ``ts`` and ``row_id`` are captured *at call time* so that buffered
        rows carry the correct capture timestamp even when flushed much later.
        """
        ts_now = int(time.time() * 1000)
        row_id = uuid.uuid4().hex

        data: Dict[str, Any] = {
            "ts":        ts_now,
            "row_id":    row_id,
            "device_id": self.device_id,
        }
        for field in self._MANDATORY_FIELDS:
            if field not in kwargs:
                raise ValueError(f"Missing mandatory field: {field}")
            data[field] = kwargs[field]

        data.update(self.calculate_image_metrics(kwargs.get("image")))
        self.save_to_parquet(data)

    def save_to_parquet(self, data: Dict[str, Any]) -> None:
        """Queue a row and flush when the buffer limit is reached."""
        self._pending_rows.append(data)
        if len(self._pending_rows) >= self.buffer_size:
            self.flush()

    def flush(self) -> None:
        """Persist all queued rows as a new immutable segment file."""
        if not self._pending_rows:
            return

        schema = self._create_schema()
        rows = []
        for row_data in self._pending_rows:
            arrays = [
                pa.array(
                    [self._normalize_field_value(row_data.get(f.name), f.type)],
                    type=f.type,
                )
                for f in schema
            ]
            rows.append(pa.Table.from_arrays(arrays, schema=schema))

        table = pa.concat_tables(rows)

        model_dir = os.path.join(self.logs_dir, self.model_name)
        ms = time.time_ns() // 1_000_000
        while True:
            seg_name = f"{ms:013d}_{os.getpid()}_{self._flush_seq}.parquet"
            self._flush_seq += 1
            tmp_path = os.path.join(model_dir, seg_name + ".tmp")
            final_path = os.path.join(model_dir, seg_name)
            archived_path = os.path.join(model_dir, "archive", seg_name)
            if os.path.exists(final_path) or os.path.exists(archived_path):
                continue
            try:
                fd = os.open(tmp_path, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
            except FileExistsError:
                continue
            os.close(fd)
            break

        pq.write_table(table, tmp_path)
        os.replace(tmp_path, final_path)

        self._pending_rows.clear()

    @property
    def parquet_file(self) -> str:
        """Return the directory containing this model's log segments."""
        return os.path.join(self.logs_dir, self.model_name)
