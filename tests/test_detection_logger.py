import unittest
import os
import numpy as np
from PIL import Image


from detection.detection_logger import DetectionLogger, CVMetrics
class TestDetectionLogger(unittest.TestCase):
    def setUp(self):
        self.model_name = "test_detection"
        self.image_path = r'C:\Kavins_stuff\learning\LookOutCV\samples\batMan.jpg'
        self.image = Image.open(self.image_path)
        self.np_image = np.array(self.image)

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

if __name__ == "__main__":
    unittest.main()
