import os
import shutil
import tempfile
import unittest

import numpy as np
import pandas as pd
from PIL import Image

from look_out_cv.classification_tracker.classification_logger import ClassificationLogger
from look_out_cv.metrics_types import CVMetrics


class TestClassificationLogger(unittest.TestCase):
    def setUp(self):
        self.model_name = "test_classification"
        self._tmp_dir = tempfile.TemporaryDirectory()
        self.image_path = os.path.join(self._tmp_dir.name, "test_image.jpg")
        Image.new("RGB", (32, 32), color=(128, 64, 32)).save(self.image_path)
        self.image = Image.open(self.image_path)
        self.np_image = np.array(self.image)

    def tearDown(self):
        self.image.close()
        self._tmp_dir.cleanup()

    def test_log_prediction_with_path(self):
        logger = ClassificationLogger(self.model_name)
        result = logger.log_prediction(
            image=self.image_path,
            pred_class="cat",
            confidence=0.95,
            image_name="test_image.jpg"
        )
        self.assertIsNone(result)

    def test_log_prediction_with_np_array(self):
        logger = ClassificationLogger(self.model_name)
        result = logger.log_prediction(
            image=self.np_image,
            pred_class="dog",
            confidence=0.88,
            image_name="test_image.jpg"
        )
        self.assertIsNone(result)

    def test_inject_with_optional_fields(self):
        logger = ClassificationLogger(self.model_name, enabled_metrics=[CVMetrics.CONTRAST])
        result = logger.log_prediction(
            image=self.np_image,
            pred_class="batman",
            confidence=0.99,
            image_name="test_image.jpg"
        )
        self.assertIsNone(result)

    def test_buffered_writes_are_flushed_after_threshold(self):
        logs_dir = "lookout_cv_logs_test_classification"
        shutil.rmtree(logs_dir, ignore_errors=True)
        try:
            logger = ClassificationLogger("buffered_model_cls", enabled_metrics=[CVMetrics.CONTRAST], logs_dir=logs_dir, buffer_size=2)
            logger.log_prediction(
                image=self.np_image,
                pred_class="cat",
                confidence=0.95,
                image_name="test_1.jpg"
            )
            self.assertEqual(len(pd.read_parquet(logger.parquet_file)), 0)

            logger.log_prediction(
                image=self.np_image,
                pred_class="dog",
                confidence=0.91,
                image_name="test_2.jpg"
            )
            self.assertEqual(len(pd.read_parquet(logger.parquet_file)), 2)
        finally:
            shutil.rmtree(logs_dir, ignore_errors=True)


if __name__ == "__main__":
    unittest.main()
