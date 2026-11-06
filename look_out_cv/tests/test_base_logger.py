"""Tests for BaseLogger (segment-based parquet logger)."""
from __future__ import annotations

import os
import tempfile
import unittest

import numpy as np
import pandas as pd
from PIL import Image

from look_out_cv._utils import read_log
from look_out_cv.logger.base_logger import BaseLogger
from look_out_cv.metrics_types import CVMetrics


class MockLogger(BaseLogger):
    _MANDATORY_FIELDS = ["image_name", "confidence"]


class TestBaseLogger(unittest.TestCase):
    def setUp(self):
        self.model_name = "test_base_model"
        self._tmp_dir = tempfile.TemporaryDirectory()
        self.logs_dir = os.path.join(self._tmp_dir.name, "logs")

        self.image_path = os.path.join(self._tmp_dir.name, "test_image.jpg")
        Image.new("RGB", (32, 32), color=(128, 64, 32)).save(self.image_path)
        self.image = Image.open(self.image_path)
        self.np_image = np.array(self.image)

    def tearDown(self):
        self.image.close()
        self._tmp_dir.cleanup()

    def test_initialization(self):
        """After init, model dir exists. After logging + flush, columns are correct."""
        logger = MockLogger(self.model_name, logs_dir=self.logs_dir)
        # Model dir created on init, but no parquet files yet
        self.assertTrue(os.path.isdir(logger.parquet_file))

        # Log one prediction so a segment file exists
        logger.log_prediction(image_name="test1.jpg", confidence=0.9)
        logger.flush()

        df = pd.read_parquet(logger.parquet_file)
        self.assertIn("image_name", df.columns)
        self.assertIn("confidence", df.columns)
        self.assertIn("ts", df.columns)
        self.assertIn("row_id", df.columns)
        self.assertIn("device_id", df.columns)

    def test_evolve_schema(self):
        """Logging without metrics then with metrics yields nulls for old rows."""
        # Log one row without the BRIGHTNESS metric
        logger1 = MockLogger(self.model_name, logs_dir=self.logs_dir)
        logger1.log_prediction(image_name="a.jpg", confidence=0.7)
        logger1.flush()

        # Log a second row with BRIGHTNESS enabled (different logger instance)
        logger2 = MockLogger(
            self.model_name,
            enabled_metrics=[CVMetrics.BRIGHTNESS],
            logs_dir=self.logs_dir,
        )
        logger2.log_prediction(image_name="b.jpg", confidence=0.8)
        logger2.flush()

        # Reading the whole model dir should give both rows; the first row's
        # brightness column is null.
        df = read_log(self.logs_dir, self.model_name).to_pandas()
        self.assertIn(CVMetrics.BRIGHTNESS.value, df.columns)
        # Old row has NaN for brightness
        brightness_col = df[CVMetrics.BRIGHTNESS.value]
        self.assertTrue(brightness_col.isna().any())

    def test_context_manager(self):
        """Rows buffered inside the context manager are flushed on exit."""
        with MockLogger(
            self.model_name, logs_dir=self.logs_dir, buffer_size=2
        ) as logger:
            logger.log_prediction(image_name="test1.jpg", confidence=0.9)
            # Buffer holds 1 row; threshold is 2 — nothing written yet
            self.assertEqual(len(logger._pending_rows), 1)

        # After __exit__, flush() was called and the row is on disk
        df = pd.read_parquet(logger.parquet_file)
        self.assertEqual(len(df), 1)
        self.assertEqual(df.iloc[0]["image_name"], "test1.jpg")

    def test_log_prediction_missing_field(self):
        logger = MockLogger(self.model_name, logs_dir=self.logs_dir)
        with self.assertRaises(ValueError):
            logger.log_prediction(image_name="test1.jpg")  # missing confidence

    def test_calculate_image_metrics(self):
        logger = MockLogger(
            self.model_name,
            enabled_metrics=[CVMetrics.CONTRAST],
            logs_dir=self.logs_dir,
        )
        metrics = logger.calculate_image_metrics(self.np_image)
        self.assertIn(CVMetrics.CONTRAST.value, metrics)
        self.assertIsNotNone(metrics[CVMetrics.CONTRAST.value])

    def test_calculate_image_metrics_no_image(self):
        logger = MockLogger(
            self.model_name,
            enabled_metrics=[CVMetrics.CONTRAST],
            logs_dir=self.logs_dir,
        )
        metrics = logger.calculate_image_metrics(None)
        self.assertIn(CVMetrics.CONTRAST.value, metrics)
        self.assertIsNone(metrics[CVMetrics.CONTRAST.value])

    def test_ts_captured_at_log_time(self):
        """ts is set at log_prediction time, not at flush time."""
        import time
        logger = MockLogger(
            self.model_name, logs_dir=self.logs_dir, buffer_size=10
        )
        before = int(time.time() * 1000)
        logger.log_prediction(image_name="a.jpg", confidence=0.9)
        after = int(time.time() * 1000)

        time.sleep(0.05)
        logger.flush()

        df = pd.read_parquet(logger.parquet_file)
        ts = int(df.iloc[0]["ts"])
        # ts was captured at call time, not flush time
        self.assertGreaterEqual(ts, before)
        self.assertLessEqual(ts, after)

    def test_device_id_override(self):
        """device_id can be overridden via constructor."""
        logger = MockLogger(
            self.model_name, logs_dir=self.logs_dir, device_id="pi-camera-1"
        )
        logger.log_prediction(image_name="a.jpg", confidence=0.9)
        logger.flush()

        df = pd.read_parquet(logger.parquet_file)
        self.assertEqual(df.iloc[0]["device_id"], "pi-camera-1")


if __name__ == "__main__":
    unittest.main()
