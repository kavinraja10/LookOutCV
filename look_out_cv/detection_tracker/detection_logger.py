"""DetectionLogger: object-detection prediction logger."""
from __future__ import annotations

from typing import Any, Dict, List, Optional, Union

import numpy as np

from look_out_cv.logger.base_logger import BaseLogger
from look_out_cv.metrics_types import CVMetrics


class DetectionLogger(BaseLogger):
    """Logger for object-detection models.

    Adds ``num_boxes`` (int32) — the number of bounding boxes predicted in the
    frame — so false-positive storms and model blindness are visible.

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
    num_boxes:
        Number of bounding boxes predicted in the current frame.
        Stored alongside each row so that detection density is visible.
    """

    _MANDATORY_FIELDS = [
        "image_name", "pred_class", "confidence",
        "bbox_x1", "bbox_y1", "bbox_x2", "bbox_y2",
    ]

    def __init__(
        self,
        model_name: str,
        enabled_metrics: Optional[List[CVMetrics]] = None,
        logs_dir: str = "lookout_cv_logs",
        buffer_size: int = 1,
        device_id: Optional[str] = None,
    ) -> None:
        super().__init__(
            model_name,
            enabled_metrics,
            logs_dir,
            buffer_size,
            device_id=device_id,
        )

    def _create_schema(self) -> "pa.Schema":
        import pyarrow as pa
        from look_out_cv.logger.base_logger import _field_type
        schema = super()._create_schema()
        if "num_boxes" not in schema.names:
            schema = schema.append(pa.field("num_boxes", _field_type("num_boxes")))
        return schema

    def log_prediction(self, **kwargs: Any) -> None:
        """Store one detection prediction.

        Accepts the same arguments as :meth:`BaseLogger.log_prediction` plus
        an optional ``num_boxes`` (int) parameter.
        """
        import time
        import uuid
        import pyarrow as pa
        from look_out_cv.logger.base_logger import _field_type

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

        data["num_boxes"] = kwargs.get("num_boxes", None)

        data.update(self.calculate_image_metrics(kwargs.get("image")))
        self.save_to_parquet(data)
