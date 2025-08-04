
import numpy as np
from PIL import Image
import cv2
from typing import Union
from enum import Enum, auto



class Additional_Fields(Enum):
    def _generate_next_value_(name, start, count, last_values):
        return name.lower()

    CONTRAST = auto()
    BLUR = auto()
    ORIENTATION = auto()
    BRIGHTNESS = auto()



class ImageMetricsCalculator:
    def __init__(self, image: Union[Image.Image, np.ndarray]):
        self.image = None
        self._set_image(image)

    def _set_image(self, image: Union[Image.Image, np.ndarray]):
        if isinstance(image, Image.Image):
            self.image = np.array(image)
        elif isinstance(image, np.ndarray):
            if image.ndim == 2:
                self.image = image[:, :, np.newaxis]
            else:
                self.image = image
        elif isinstance(image, str):
            self.image = np.array(Image.open(image))

    def calculate_contrast(self) -> float:
        return float(np.std(self.image))

    def calculate_blur(self) -> float:
        gray = cv2.cvtColor(self.image, cv2.COLOR_RGB2GRAY) if len(self.image.shape) == 3 else self.image
        return float(cv2.Laplacian(gray, cv2.CV_64F).var())

    def calculate_orientation_type(self) -> float:
        h, w = self.image.shape[:2]
        if h > w:
            return 0.0
        elif w > h:
            return 1.0
        return 0.5

    def calculate_brightness(self) -> float:
        return float(np.mean(self.image))
