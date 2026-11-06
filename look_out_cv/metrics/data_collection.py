"""DataCollectionLogger: image-metric collection logger (subclass of BaseLogger)."""
from __future__ import annotations

import time
import uuid
from typing import Any, Dict, Iterable, Optional, Union

import numpy as np

from look_out_cv.logger.base_logger import BaseLogger
from look_out_cv.metrics.metrics import ImageMetricsCalculator, resolve_metric_method_name
from look_out_cv.metrics_types import CVMetrics


class DataCollectionLogger(BaseLogger):
    """Calculate image metrics and append them to immutable segment files.

    This logger's schema is ``image_name`` plus the enabled metrics (plus the
    bookkeeping columns ``ts``, ``row_id``, ``device_id`` inherited from
    :class:`~look_out_cv.logger.base_logger.BaseLogger`).

    Parameters
    ----------
    dataset_name:
        Logical name of the dataset being profiled.  Used as the model
        directory name on disk.
    enabled_metrics:
        Metrics to compute for each image.
    logs_dir:
        Root directory for log files.
    buffer_size:
        Number of images to buffer before auto-flushing.
    device_id:
        Identifier for this device.  Defaults to ``socket.gethostname()``.
    """

    _MANDATORY_FIELDS = ["image_name"]

    def __init__(
        self,
        dataset_name: str,
        enabled_metrics: Optional[Iterable[CVMetrics]] = None,
        logs_dir: str = "lookout_cv_logs",
        buffer_size: int = 1,
        device_id: Optional[str] = None,
    ) -> None:
        self.dataset_name = dataset_name
        super().__init__(
            model_name=dataset_name,
            enabled_metrics=list(enabled_metrics or []),
            logs_dir=logs_dir,
            buffer_size=buffer_size,
            device_id=device_id,
        )

    def calculate_metrics(self, image: Union[str, "np.ndarray"]) -> Dict[str, float]:
        """Compute all enabled metrics for *image* and return as a dict."""
        calculator = ImageMetricsCalculator(image)
        results: Dict[str, float] = {}
        for metric in self.enabled_metrics:
            method_name = resolve_metric_method_name(metric)
            results[metric.value] = float(getattr(calculator, method_name)())
        return results

    def log_image(self, image_name: str, image: Union[str, "np.ndarray"]) -> None:
        """Log one image's name and its computed metrics.

        ``ts`` and ``row_id`` are captured at call time (not at flush time).
        """
        ts_now = int(time.time() * 1000)
        row_id = uuid.uuid4().hex

        data: Dict[str, Any] = {
            "ts":         ts_now,
            "row_id":     row_id,
            "device_id":  self.device_id,
            "image_name": image_name,
        }
        data.update(self.calculate_metrics(image))
        self.save_to_parquet(data)
