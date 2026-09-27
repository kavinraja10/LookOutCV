import os
from typing import Any, Dict, List, Optional, Union

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

from look_out_cv.metrics.metrics import ImageMetricsCalculator, resolve_metric_method_name
from look_out_cv.metrics_types import CVMetrics


class BaseLogger:
    def __init__(
        self,
        model_name: str,
        enabled_metrics: Optional[List[CVMetrics]] = None,
        logs_dir: str = "lookout_cv_logs",
        buffer_size: int = 1,
    ) -> None:
        """Initialize the logger for model monitoring."""
        self.model_name = model_name
        self.enabled_metrics = list(enabled_metrics or [])
        self.logs_dir = logs_dir
        self.buffer_size = max(1, int(buffer_size))
        self._pending_rows: List[Dict[str, Any]] = []

        model_dir = os.path.join(self.logs_dir, self.model_name)
        os.makedirs(model_dir, exist_ok=True)
        self.parquet_file = os.path.join(model_dir, f"{self.model_name}_logs.parquet")

        if os.path.exists(self.parquet_file):
            self._evolve_schema()
        else:
            pq.write_table(
                pa.Table.from_arrays(
                    [pa.array([], type=field.type) for field in self._create_schema()],
                    schema=self._create_schema(),
                ),
                self.parquet_file,
            )

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.flush()
        return False

    def _create_schema(self) -> pa.schema:
        fields = []
        for field_name in self._MANDATORY_FIELDS:
            dtype = pa.string() if "name" in field_name or "class" in field_name else pa.float32()
            fields.append(pa.field(field_name, dtype))
        for metric in self.enabled_metrics:
            fields.append(pa.field(metric.value, pa.float32()))
        return pa.schema(fields)

    def _evolve_schema(self):
        existing_table = pq.read_table(self.parquet_file)
        existing_columns = set(existing_table.schema.names)
        missing_columns = set(self._create_schema().names) - existing_columns

        if not missing_columns:
            return

        for column_name in missing_columns:
            column = pa.array([None] * existing_table.num_rows, type=pa.float32())
            existing_table = existing_table.append_column(column_name, column)

        pq.write_table(existing_table, self.parquet_file)

    def _resolve_metric_method_name(self, metric: CVMetrics) -> str:
        return resolve_metric_method_name(metric)

    def _compute_metric_value(self, calculator: ImageMetricsCalculator, metric: CVMetrics) -> Optional[float]:
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

    def calculate_image_metrics(self, image: Optional[Union[str, "np.ndarray"]] = None) -> Dict[str, Optional[float]]:
        """Calculate metrics for the provided image."""
        results = {metric.value: None for metric in self.enabled_metrics}
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
        """Store one prediction record with the configured image metrics."""
        data = {}
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
        """Persist all queued rows to disk."""
        if not self._pending_rows:
            return

        try:
            existing_table = pq.read_table(self.parquet_file)
            schema = existing_table.schema
        except Exception as exc:
            raise IOError(f"Failed to read parquet file: {exc}") from exc

        rows = []
        for row_data in self._pending_rows:
            values = [row_data.get(name, None) for name in schema.names]
            arrays = [
                pa.array([self._normalize_field_value(value, field.type)], type=field.type)
                for field, value in zip(schema, values)
            ]
            rows.append(pa.Table.from_arrays(arrays, schema=schema))

        if rows:
            combined_table = pa.concat_tables([existing_table, *rows])
            pq.write_table(combined_table, self.parquet_file)

        self._pending_rows.clear()
