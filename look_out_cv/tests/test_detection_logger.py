"""Tests for DetectionLogger and DataCollectionLogger."""
from __future__ import annotations

import os
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
        self.logs_dir = os.path.join(self._tmp_dir.name, "logs")
        self.image_path = os.path.join(self._tmp_dir.name, "test_image.jpg")
        Image.new("RGB", (32, 32), color=(128, 64, 32)).save(self.image_path)
        self.image = Image.open(self.image_path)
        self.np_image = np.array(self.image)

    def tearDown(self):
        self.image.close()
        self._tmp_dir.cleanup()

    def test_log_prediction_with_path(self):
        logger = DetectionLogger(self.model_name, logs_dir=self.logs_dir)
        result = logger.log_prediction(
            image=self.image_path,
            pred_class="cat",
            confidence=0.95,
            image_name="test_image.jpg",
            bbox_x1=10, bbox_y1=20, bbox_x2=100, bbox_y2=120,
        )
        self.assertIsNone(result)

    def test_log_prediction_with_np_array(self):
        logger = DetectionLogger(self.model_name, logs_dir=self.logs_dir)
        result = logger.log_prediction(
            image=self.np_image,
            pred_class="dog",
            confidence=0.88,
            image_name="test_image.jpg",
            bbox_x1=15, bbox_y1=25, bbox_x2=110, bbox_y2=130,
        )
        self.assertIsNone(result)

    def test_inject_with_optional_fields(self):
        logger = DetectionLogger(
            self.model_name,
            enabled_metrics=[CVMetrics.CONTRAST],
            logs_dir=self.logs_dir,
        )
        result = logger.log_prediction(
            image=self.np_image,
            pred_class="batman",
            confidence=0.99,
            image_name="test_image.jpg",
            bbox_x1=5, bbox_y1=10, bbox_x2=50, bbox_y2=60,
        )
        self.assertIsNone(result)

    def test_buffered_writes_are_flushed_after_threshold(self):
        """Buffer flushes to disk when the threshold is reached."""
        logs_dir = os.path.join(self._tmp_dir.name, "buffered_logs")
        logger = DetectionLogger(
            "buffered_model",
            enabled_metrics=[CVMetrics.CONTRAST],
            logs_dir=logs_dir,
            buffer_size=2,
        )
        logger.log_prediction(
            image=self.np_image,
            pred_class="cat",
            confidence=0.95,
            image_name="test_1.jpg",
            bbox_x1=10, bbox_y1=20, bbox_x2=100, bbox_y2=120,
        )
        # Only 1 row buffered — nothing on disk yet
        self.assertEqual(len(logger._pending_rows), 1)

        logger.log_prediction(
            image=self.np_image,
            pred_class="dog",
            confidence=0.91,
            image_name="test_2.jpg",
            bbox_x1=30, bbox_y1=40, bbox_x2=140, bbox_y2=150,
        )
        # Buffer hit threshold — flushed to disk
        self.assertEqual(len(pd.read_parquet(logger.parquet_file)), 2)

    def test_data_collection_logger_buffering_and_flush(self):
        """DataCollectionLogger buffers and flushes correctly."""
        logs_dir = os.path.join(self._tmp_dir.name, "data_logs")
        logger = DataCollectionLogger(
            "buffered_dataset",
            enabled_metrics=[CVMetrics.CONTRAST],
            logs_dir=logs_dir,
            buffer_size=2,
        )
        logger.log_image("img_1.jpg", self.np_image)
        self.assertEqual(len(logger._pending_rows), 1)

        logger.log_image("img_2.jpg", self.np_image)
        self.assertEqual(len(logger._pending_rows), 0)  # auto-flushed

        logger.flush()  # no-op
        self.assertEqual(len(pd.read_parquet(logger.parquet_file)), 2)


if __name__ == "__main__":
    unittest.main()
