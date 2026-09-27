import os
from typing import Any, Dict, Iterable, Optional, Union

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

from .metrics import ImageMetricsCalculator, resolve_metric_method_name
from ..metrics_types import CVMetrics


class DataCollectionLogger:
    """Calculate image metrics and append them to a Parquet file."""
    def __init__(
        self,
        dataset_name: str,
        enabled_metrics: Optional[Iterable[CVMetrics]] = None,
        logs_dir: str = "lookout_cv_logs",
        buffer_size: int = 1,
    ) -> None:
        self.dataset_name = dataset_name
        self.enabled_metrics = list(enabled_metrics or [])
        self.logs_dir = logs_dir
        self.buffer_size = max(1, int(buffer_size))
        self._pending_rows: list[dict[str, Any]] = []
        self.parquet_file = os.path.join(
            logs_dir,
            dataset_name,
            f"{dataset_name}_metrics.parquet",
        )
        os.makedirs(os.path.dirname(self.parquet_file), exist_ok=True)
        if not os.path.exists(self.parquet_file):
            schema = pa.schema(
                [pa.field("image_name", pa.string())]
                + [pa.field(metric.value, pa.float32()) for metric in self.enabled_metrics]
            )
            pq.write_table(
                pa.Table.from_arrays([pa.array([], type=field.type) for field in schema], schema=schema),
                self.parquet_file,
            )

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.flush()
        return False

    def calculate_metrics(self, image: Union[str, "np.ndarray"]) -> Dict[str, float]:
        calculator = ImageMetricsCalculator(image)
        results: Dict[str, float] = {}
        for metric in self.enabled_metrics:
            method_name = resolve_metric_method_name(metric)
            results[metric.value] = float(getattr(calculator, method_name)())
        return results

    def log_image(self, image_name: str, image: Union[str, "np.ndarray"]) -> None:
        data: Dict[str, Any] = {"image_name": image_name}
        data.update(self.calculate_metrics(image))
        self._pending_rows.append(data)
        if len(self._pending_rows) >= self.buffer_size:
            self.flush()

    def flush(self) -> None:
        if not self._pending_rows:
            return

        table = pq.read_table(self.parquet_file)
        row_tables = []
        for data in self._pending_rows:
            row_arrays = [
                pa.array([data.get(field.name)], type=field.type)
                for field in table.schema
            ]
            row_tables.append(pa.Table.from_arrays(row_arrays, schema=table.schema))

        if row_tables:
            pq.write_table(pa.concat_tables([table, *row_tables]), self.parquet_file)
        self._pending_rows.clear()
