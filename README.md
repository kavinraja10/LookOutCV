# LookOutCV  

Keep an eye on your computer vision models in production 👁️  

I built LookOutCV because monitoring CV models in production is often an afterthought. We usually deploy models and only notice issues once accuracy drops or customers complain. This tool makes it easier to **log predictions, track image quality, and catch data/model drift early** without adding complex monitoring infrastructure.


---

## Features  

- Logs predictions automatically  
- Collects image quality metrics (contrast, blur, orientation, ...)  
- Supports **object detection** and **classification** models  
- Stores data in **Parquet** for easy analysis and less resourse for storage  
- Schema evolves automatically if you add new metrics  

---

## Usage  

### Object Detection  

```python
from detection.detection_logger import DetectionLogger, CVMetrics

logger = DetectionLogger(
    model_name="my_detection_model",
    enabled_metrics=[CVMetrics.CONTRAST, CVMetrics.BLUR]
)

logger.log_prediction(
    image="path/to/image.jpg",  # path, numpy array, or PIL Image
    pred_class="car",
    confidence=0.95,
    image_name="image1.jpg",
    bbox_x1=100, bbox_y1=200, bbox_x2=300, bbox_y2=400
)

### Classification

```python
from classification.classification_logger import ClassificationLogger, CVMetrics

logger = ClassificationLogger(
    model_name="my_classifier",
    enabled_metrics=[CVMetrics.CONTRAST, CVMetrics.BLUR]
)

logger.log_prediction(
    image="path/to/image.jpg",
    pred_class="cat",
    confidence=0.99,
    image_name="image1.jpg"
)
```

## Data Storage

All metrics and predictions are stored in Parquet files for efficient storage and quick analysis:
```
lookout_cv_logs/
    model_name/
        model_name_logs_<pid>.parquet
```

## Installation

```bash
yet to add
```

## TODO

Priority tasks:
- [ ] Add visualization tools for metrics analysis
- [ ] Implement basic drift detection
- [ ] Add data retention management
- [ ] Create example notebooks

Later plans:
- [ ] Support for semantic segmentation models
- [ ] Distributed logging capabilities
- [ ] Real-time monitoring dashboard
- [ ] Advanced drift detection algorithms
- [ ] Automated alerting system
