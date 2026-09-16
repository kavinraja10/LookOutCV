from enum import Enum, auto
from typing import List, Optional, Union
import os
import numpy as np
import pyarrow as pa
from look_out_cv.logger.base_logger import BaseLogger
from look_out_cv.metrics_types import CVMetrics



class DetectionLogger(BaseLogger):

    _MANDATORY_FIELDS = [
            "image_name", "pred_class", "confidence",
            "bbox_x1", "bbox_y1", "bbox_x2", "bbox_y2"]
    

    def __init__(self, model_name: str, enabled_metrics: Optional[List[CVMetrics]] = None, logs_dir: str = "lookout_cv_logs"):
        super().__init__(model_name, enabled_metrics, logs_dir)
