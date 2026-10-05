import os
import shutil
import tempfile
import unittest

import numpy as np
import pandas as pd
from PIL import Image

from look_out_cv import DataCollectionLogger
from look_out_cv.detection_tracker.detection_logger import DetectionLogger
from look_out_cv.metrics_types import CVMetrics


class TestDetectionLogger(unittest.TestCase):
    def setUp(self):
        self.model_name = "test_detection"
        self._tmp_dir = tempfile.TemporaryDirectory()
        self.image_path = os.path.join(self._tmp_dir.name, "test_image.jpg")
        Image.new("RGB", (32, 32), color=(128, 64, 32)).save(self.image_path)
        self.image = Image.open(self.image_path)
        self.np_image = np.array(self.image)

    def tearDown(self):
        self.image.close()
        self._tmp_dir.cleanup()

    def test_log_prediction_with_path(self):
        logger = DetectionLogger(self.model_name)
        result = logger.log_prediction(
            image=self.image_path,
            pred_class="cat",
            confidence=0.95,
            image_name="test_image.jpg",
            bbox_x1=10, bbox_y1=20, bbox_x2=100, bbox_y2=120
        )
        self.assertIsNone(result)

    def test_log_prediction_with_np_array(self):
        logger = DetectionLogger(self.model_name)
        result = logger.log_prediction(
            image=self.np_image,
            pred_class="dog",
            confidence=0.88,
             image_name="test_image.jpg",
            bbox_x1=15, bbox_y1=25, bbox_x2=110, bbox_y2=130
        )
        self.assertIsNone(result)

    def test_inject_with_optional_fields(self):
        logger = DetectionLogger(self.model_name, enabled_metrics=[CVMetrics.CONTRAST])
        result = logger.log_prediction(
            image=self.np_image,
            pred_class="batman",
            confidence=0.99,
            image_name="test_image.jpg",
            bbox_x1=5, bbox_y1=10, bbox_x2=50, bbox_y2=60,
        )
        self.assertIsNone(result)

    def test_buffered_writes_are_flushed_after_threshold(self):
        logs_dir = "lookout_cv_logs_test"
        shutil.rmtree(logs_dir, ignore_errors=True)
        try:
            logger = DetectionLogger("buffered_model", enabled_metrics=[CVMetrics.CONTRAST], logs_dir=logs_dir, buffer_size=2)
            logger.log_prediction(
                image=self.np_image,
                pred_class="cat",
                confidence=0.95,
                image_name="test_1.jpg",
                bbox_x1=10, bbox_y1=20, bbox_x2=100, bbox_y2=120,
            )
            self.assertEqual(len(pd.read_parquet(logger.parquet_file)), 0)

            logger.log_prediction(
                image=self.np_image,
                pred_class="dog",
                confidence=0.91,
                image_name="test_2.jpg",
                bbox_x1=30, bbox_y1=40, bbox_x2=140, bbox_y2=150,
            )
            self.assertEqual(len(pd.read_parquet(logger.parquet_file)), 2)
        finally:
            shutil.rmtree(logs_dir, ignore_errors=True)

    def test_data_collection_logger_buffering_and_flush(self):
        logs_dir = "lookout_cv_logs_test_data"
        shutil.rmtree(logs_dir, ignore_errors=True)
        try:
            logger = DataCollectionLogger("buffered_dataset", enabled_metrics=[CVMetrics.CONTRAST], logs_dir=logs_dir, buffer_size=2)
            logger.log_image("img_1.jpg", self.np_image)
            self.assertEqual(len(pd.read_parquet(logger.parquet_file)), 0)

            logger.log_image("img_2.jpg", self.np_image)
            self.assertEqual(len(pd.read_parquet(logger.parquet_file)), 2)

            logger.flush()
            self.assertEqual(len(pd.read_parquet(logger.parquet_file)), 2)
        finally:
            shutil.rmtree(logs_dir, ignore_errors=True)


if __name__ == "__main__":
    unittest.main()
