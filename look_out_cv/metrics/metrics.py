from pathlib import Path
from typing import Union

import cv2
import numpy as np
from PIL import Image

from look_out_cv.metrics_types import CVMetrics


_METRIC_METHOD_NAMES = {
    CVMetrics.ORIENTATION: "calculate_orientation_type",
}


def resolve_metric_method_name(metric: CVMetrics) -> str:
    return _METRIC_METHOD_NAMES.get(metric, f"calculate_{metric.value}")


class ImageMetricsCalculator:
    def __init__(self, image: Union[Image.Image, np.ndarray, str, Path]):
        self.image = self._prepare_image(image)

    @staticmethod
    def _prepare_image(image: Union[Image.Image, np.ndarray, str, Path]):
        if isinstance(image, Image.Image):
            return np.asarray(image)

        if isinstance(image, np.ndarray):
            return image[:, :, None] if image.ndim == 2 else image

        if isinstance(image, (str, Path)):
            return np.asarray(Image.open(image))

        raise TypeError(f"Unsupported image type: {type(image)!r}")

    def calculate_contrast(self) -> float:
        return float(np.std(self.image))

    def calculate_blur(self) -> float:
        gray = self.image
        if self.image.ndim == 3:
            gray = cv2.cvtColor(self.image, cv2.COLOR_RGB2GRAY)
        return float(cv2.Laplacian(gray, cv2.CV_64F).var())

    def calculate_orientation_type(self) -> float:
        height, width = self.image.shape[:2]
        if height > width:
            return 0.0
        if width > height:
            return 1.0
        return 0.5

    def calculate_brightness(self) -> float:
        return float(np.mean(self.image))
