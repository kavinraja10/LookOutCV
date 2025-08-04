from enum import Enum, auto
from typing import List, Optional, Union
import os
import numpy as np

from logger.logger import BaseLogger



from logger.logger import CVMetrics

class DetectionLogger(BaseLogger):
    # expected_types: mapping of all possible fields to pa.float32
    # This is used by BaseLogger for schema evolution
    import pyarrow as pa
    expected_types = {
        "image_name": pa.float32(),
        "pred_class": pa.float32(),
        "confidence": pa.float32(),
        "bbox_x1": pa.float32(),
        "bbox_y1": pa.float32(),
        "bbox_x2": pa.float32(),
        "bbox_y2": pa.float32(),
        CVMetrics.CONTRAST.value: pa.float32(),
        CVMetrics.BLUR.value: pa.float32(),
        CVMetrics.ORIENTATION.value: pa.float32(),
        CVMetrics.BBOX_RATIO.value: pa.float32(),
    }
    _MANDATORY_FIELDS = [
        "image_name", "pred_class", "confidence",
        "bbox_x1", "bbox_y1", "bbox_x2", "bbox_y2"]

    _ADDITIONAL_REQUIREMENTS = {
        CVMetrics.CONTRAST: "image",
        CVMetrics.BLUR: "image",
        CVMetrics.ORIENTATION: "image",
    }

    def __init__(self, model_name: str, enabled_metrics: Optional[List[CVMetrics]] = None, logs_dir: str = "lookout_cv_logs"):
        super().__init__(model_name, enabled_metrics, logs_dir)


if __name__ == "__main__":
        logger = DetectionLogger("my_model", enabled_metrics=[CVMetrics.CONTRAST])
        result = logger.log_prediction(
            image=r'samples\batMan.jpg',
            pred_class="batman",
            confidence=0.99,
            image_name="test_image.jpg",
            bbox_x1=5, bbox_y1=10, bbox_x2=50, bbox_y2=60,
            
        )

        # logger = DetectionLogger("my_model")
        # result = logger.inject(
        #     image=r'samples\batMan.jpg',
        #     pred_class="dog",
        #     confidence=0.88,
        #      image_name="test_image.jpg",
        #     bbox_x1=15, bbox_y1=25, bbox_x2=110, bbox_y2=130
        # )
        print("Logging complete.")