Keep an eye on your computer vision models in production 👁️

**Local-first monitoring for computer vision models, built for the edge.**

LookOutCV runs inside your inference process on the device itself. It writes predictions and image quality metrics to local Parquet files, and drift checks, rollups and retention all run against those files. There's no server, no account and no network calls, so it works the same on a Jetson, a Raspberry Pi, an on-prem box or a machine with no internet.

You call a logger next to your model's inference code. It saves the prediction, the confidence and a few image quality numbers (contrast, blur, brightness, ...). Later you can check those logs for drift, summarise them, or trim them so they don't fill the disk.

## Why local-first

Cloud monitoring assumes a steady connection, and shipping raw images off a device is often slow, expensive or not allowed. LookOutCV keeps everything on the device instead:

- **Offline by default.** Nothing leaves the machine unless you copy the files yourself.
- **Small footprint.** It's a Python library on top of numpy, pandas, pyarrow and OpenCV. There's no extra service to run, and it doesn't need PyTorch or TensorFlow.
- **Only numbers are stored.** You keep metrics and predictions, not images, so storage stays small and nothing sensitive is retained.
- **Bounded disk use.** Retention policies cap the log size by row count, optionally archiving old rows.
- **Open format.** Logs are plain Parquet, so you can pull them off the device and analyse them anywhere with pandas, DuckDB or Spark.

Vision models tend to get worse quietly. The lighting changes, a camera gets dirty, the input images start looking different from the training set, and nothing crashes. Keeping image statistics and predictions side by side makes it much easier to see when that happens, even on a device you can't easily reach.

## Install

```bash
pip install -r requirements.txt
```

The requirements are pinned for Python 3.13. `scipy` is optional: if it's installed, the KS drift test reports a p-value, and without it only the statistic is available.

There is no `setup.py` yet, so for now run your code from the project root.

## Logging predictions

### Classification

```python
from look_out_cv import ClassificationLogger, CVMetrics

logger = ClassificationLogger(
    model_name="my_classifier",
    enabled_metrics=[CVMetrics.CONTRAST, CVMetrics.BLUR, CVMetrics.BRIGHTNESS],
)

logger.log_prediction(
    image="path/to/image.jpg",  # file path, numpy array or PIL image
    pred_class="cat",
    confidence=0.99,
    image_name="image1.jpg",
)
```

### Object detection

```python
from look_out_cv import DetectionLogger, CVMetrics

logger = DetectionLogger(
    model_name="my_detector",
    enabled_metrics=[CVMetrics.CONTRAST, CVMetrics.BLUR],
)

logger.log_prediction(
    image="path/to/image.jpg",
    pred_class="car",
    confidence=0.95,
    image_name="image1.jpg",
    bbox_x1=100, bbox_y1=200,
    bbox_x2=300, bbox_y2=400,
)
```

### Image metrics only

If you just want to record image quality for a dataset, with no model involved, use `DataCollectionLogger`. It buffers rows and writes them in batches, which means fewer disk writes on devices with slow or limited storage. Use it as a context manager, or call `flush()` yourself.

```python
from look_out_cv import DataCollectionLogger, CVMetrics

with DataCollectionLogger("camera_1", enabled_metrics=[CVMetrics.BLUR], buffer_size=50) as dc:
    dc.log_image("frame_001.jpg", "path/to/frame_001.jpg")
```

## Where the logs go

```text
lookout_cv_logs/
    my_classifier/
        my_classifier_logs.parquet
    camera_1/
        camera_1_metrics.parquet
```

They're ordinary Parquet files, so `pandas.read_parquet` opens them directly. To move data off a device, copy the folder.

The logs don't store timestamps. Everything below that works over "recent" data (rollups, drift, retention) uses row order and row counts instead of time.

## Analysing the logs

All of this runs on the device, against the local files.

### Rollups

Summary stats over the last N rows:

```python
from look_out_cv import MetricsRollup

rollup = MetricsRollup("my_classifier", window_size=100, aggregations=["mean", "std", "min", "max"])
print(rollup.summary())

rollup.compute()                  # DataFrame, one column per numeric metric
rollup.compute_rolling(step=10)   # sliding window over the whole history
```

Supported aggregations: `mean`, `std`, `min`, `max`, `count`.

### Drift detection

Compares an older slice of the log (reference) with a newer one (current) for each numeric column.

```python
from look_out_cv import DriftDetector, DriftTest

detector = DriftDetector("my_classifier", test=DriftTest.PSI)
report = detector.detect()

print(report)
report.has_drift          # bool
report.drifted_columns    # e.g. ["blur", "confidence"]
report.to_dataframe()
```

| Test | Flags drift when | Default threshold |
|------|------------------|-------------------|
| `PSI` | PSI > threshold | 0.2 |
| `KS` | p-value < threshold | 0.05 |
| `Z_MEAN` | z-score of the mean shift > threshold | 3.0 |

By default the first half of the log is the reference and the second half is current. Pass `reference_window` and `current_window` (row counts) to choose the windows yourself, and `columns=[...]` to watch only some columns. `detector.detect_all()` runs all three tests.

### Retention

Edge devices have limited storage, so cap the logs before they fill it:

```python
from look_out_cv import DataRetentionManager, RetentionPolicy

mgr = DataRetentionManager(
    "my_classifier",
    max_rows=50_000,
    policy=RetentionPolicy.ARCHIVE_AND_TRIM,  # or KEEP_LATEST to just drop old rows
)
print(mgr.apply())   # rows_before, rows_after, rows_removed, rows_archived, policy
```

With `ARCHIVE_AND_TRIM` the old rows are moved to `<model>_logs_archive.parquet` in the same folder. You can then copy that file off the device and purge it with `mgr.purge_archive()`.

For a worked example of the loggers, rollups and drift detection, see the [sample notebook](sample_notebook.ipynb).

## Tests

```bash
pytest
```



## Roadmap

See [to_do.md](to_do.md). 
## License

MIT. See [LICENSE](LICENSE).