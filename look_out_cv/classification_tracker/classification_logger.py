from look_out_cv.logger.base_logger import BaseLogger


class ClassificationLogger(BaseLogger):
	_MANDATORY_FIELDS = ["image_name", "pred_class", "confidence"]
