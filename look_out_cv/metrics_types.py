from enum import Enum, auto


class CVMetrics(Enum):
    def _generate_next_value_(name, start, count, last_values):
        return name.lower()

    CONTRAST = auto()
    BLUR = auto()
    ORIENTATION = auto()
    BBOX_RATIO = auto()

    @property
    def requires_image(self) -> bool:
        """Check if the metric requires an image input."""
        return self in {self.CONTRAST, self.BLUR, self.ORIENTATION}