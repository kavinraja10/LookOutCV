from enum import Enum


class CVMetrics(Enum):
    CONTRAST = "contrast"
    BLUR = "blur"
    ORIENTATION = "orientation"
    BBOX_RATIO = "bbox_ratio"
    BRIGHTNESS = "brightness"

    @property
    def requires_image(self) -> bool:
        """Return True for metrics that are computed from image pixels."""
        return self in {self.CONTRAST, self.BLUR, self.ORIENTATION, self.BRIGHTNESS}