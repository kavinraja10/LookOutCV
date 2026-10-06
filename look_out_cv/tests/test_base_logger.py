import os
import shutil
import tempfile
import unittest

import numpy as np
import pandas as pd
from PIL import Image

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
        logger = MockLogger(self.model_name, logs_dir=self.logs_dir)
        self.assertTrue(os.path.exists(logger.parquet_file))
        df = pd.read_parquet(logger.parquet_file)
        self.assertIn("image_name", df.columns)
        self.assertIn("confidence", df.columns)

    def test_evolve_schema(self):
        # Create initial schema
        logger1 = MockLogger(self.model_name, logs_dir=self.logs_dir)
        df1 = pd.read_parquet(logger1.parquet_file)
        self.assertNotIn(CVMetrics.BRIGHTNESS.value, df1.columns)
        
        # Evolve schema with new metric
        logger2 = MockLogger(self.model_name, enabled_metrics=[CVMetrics.BRIGHTNESS], logs_dir=self.logs_dir)
        df2 = pd.read_parquet(logger2.parquet_file)
        self.assertIn(CVMetrics.BRIGHTNESS.value, df2.columns)

    def test_context_manager(self):
        with MockLogger(self.model_name, logs_dir=self.logs_dir, buffer_size=2) as logger:
            logger.log_prediction(image_name="test1.jpg", confidence=0.9)
            # Should not be flushed yet
            self.assertEqual(len(pd.read_parquet(logger.parquet_file)), 0)
            
        # Should be flushed after exit
        df = pd.read_parquet(logger.parquet_file)
        self.assertEqual(len(df), 1)
        self.assertEqual(df.iloc[0]["image_name"], "test1.jpg")

    def test_log_prediction_missing_field(self):
        logger = MockLogger(self.model_name, logs_dir=self.logs_dir)
        with self.assertRaises(ValueError):
            logger.log_prediction(image_name="test1.jpg") # missing confidence

    def test_calculate_image_metrics(self):
        logger = MockLogger(self.model_name, enabled_metrics=[CVMetrics.CONTRAST], logs_dir=self.logs_dir)
        metrics = logger.calculate_image_metrics(self.np_image)
        self.assertIn(CVMetrics.CONTRAST.value, metrics)
        self.assertIsNotNone(metrics[CVMetrics.CONTRAST.value])

    def test_calculate_image_metrics_no_image(self):
        logger = MockLogger(self.model_name, enabled_metrics=[CVMetrics.CONTRAST], logs_dir=self.logs_dir)
        metrics = logger.calculate_image_metrics(None)
        self.assertIn(CVMetrics.CONTRAST.value, metrics)
        self.assertIsNone(metrics[CVMetrics.CONTRAST.value])

if __name__ == "__main__":
    unittest.main()
