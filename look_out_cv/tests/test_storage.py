from __future__ import annotations

import os
import time
import pytest
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

from look_out_cv import (
    ClassificationLogger,
    DataCollectionLogger,
    DataRetentionManager,
    RetentionPolicy,
    CVMetrics,
)
from look_out_cv.detection_tracker.detection_logger import DetectionLogger
from look_out_cv._utils import read_log, compact, _live_segments


def test_atomic_segment_naming(tmp_path):
    logs_dir = str(tmp_path / "logs")
    model_name = "test_seg_model"

    logger = ClassificationLogger(model_name, logs_dir=logs_dir, buffer_size=1)
    logger.log_prediction(image_name="1.jpg", pred_class="cat", confidence=0.9)
    logger.log_prediction(image_name="2.jpg", pred_class="dog", confidence=0.8)

    model_dir = os.path.join(logs_dir, model_name)
    segments = _live_segments(model_dir)
    assert len(segments) == 2
    # Verify segment naming: <13-digit-ms>_<pid>_<seq>.parquet
    for seg in segments:
        base = os.path.basename(seg)
        parts = base.replace(".parquet", "").split("_")
        assert len(parts) == 3
        assert len(parts[0]) == 13
        assert parts[0].isdigit()


def test_logger_instances_do_not_overwrite_segments(tmp_path, monkeypatch):
    logs_dir = str(tmp_path / "logs")
    model_name = "same_process_model"
    monkeypatch.setattr("look_out_cv.logger.base_logger.time.time_ns", lambda: 1_700_000_000_000_000_000)

    first = ClassificationLogger(model_name, logs_dir=logs_dir, buffer_size=1)
    second = ClassificationLogger(model_name, logs_dir=logs_dir, buffer_size=1)
    first.log_prediction(image_name="first.jpg", pred_class="cat", confidence=0.9)
    second.log_prediction(image_name="second.jpg", pred_class="dog", confidence=0.8)

    table = read_log(logs_dir, model_name)
    assert table.num_rows == 2
    assert table["image_name"].to_pylist() == ["first.jpg", "second.jpg"]


def test_read_log_columns_projection(tmp_path):
    logs_dir = str(tmp_path / "logs")
    model_name = "test_proj_model"

    logger = ClassificationLogger(
        model_name,
        enabled_metrics=[CVMetrics.CONTRAST],
        logs_dir=logs_dir,
        buffer_size=1,
    )
    logger.log_prediction(image_name="1.jpg", pred_class="cat", confidence=0.9)

    # Request only confidence and a non-existent column
    table = read_log(logs_dir, model_name, columns=["confidence", "missing_col"])
    assert table.column_names == ["confidence", "missing_col"]
    assert table.num_rows == 1
    assert table["confidence"].to_pylist() == [pytest.approx(0.9, rel=1e-4)]
    # Missing column returns as null
    assert table["missing_col"].to_pylist() == [None]


def test_read_log_since_ms(tmp_path):
    logs_dir = str(tmp_path / "logs")
    model_name = "test_since_model"

    logger = ClassificationLogger(model_name, logs_dir=logs_dir, buffer_size=1)
    logger.log_prediction(image_name="1.jpg", pred_class="cat", confidence=0.9)

    time.sleep(0.02)
    cutoff = time.time_ns() // 1_000_000
    time.sleep(0.02)

    logger.log_prediction(image_name="2.jpg", pred_class="dog", confidence=0.8)

    # Filtering by since_ms skips segment 1
    table = read_log(logs_dir, model_name, since_ms=cutoff)
    assert table.num_rows == 1
    assert table["pred_class"].to_pylist() == ["dog"]


def test_data_collection_logger_subclass(tmp_path):
    logs_dir = str(tmp_path / "logs")
    dataset_name = "test_dataset"

    logger = DataCollectionLogger(
        dataset_name=dataset_name,
        logs_dir=logs_dir,
        buffer_size=2,
    )
    dummy_img = np.zeros((32, 32, 3), dtype=np.uint8)

    # Buffering
    logger.log_image("img1.jpg", dummy_img)
    assert len(logger._pending_rows) == 1
    logger.log_image("img2.jpg", dummy_img)
    assert len(logger._pending_rows) == 0  # auto-flushed

    table = read_log(logs_dir, dataset_name)
    assert table.num_rows == 2
    assert "image_name" in table.column_names
    assert "ts" in table.column_names
    assert "row_id" in table.column_names
    assert "device_id" in table.column_names


def test_retention_manager_defaults_and_segments(tmp_path):
    logs_dir = str(tmp_path / "logs")
    model_name = "retention_model"

    # Default max_rows should be 100_000
    mgr_default = DataRetentionManager(model_name, logs_dir=logs_dir)
    assert mgr_default.max_rows == 100_000

    logger = ClassificationLogger(model_name, logs_dir=logs_dir, buffer_size=1)
    for i in range(5):
        logger.log_prediction(image_name=f"{i}.jpg", pred_class="cat", confidence=0.9)

    assert mgr_default.current_row_count() == 5

    # Trim with KEEP_LATEST (keep 3 rows across whole segments)
    mgr = DataRetentionManager(model_name, max_rows=3, policy=RetentionPolicy.KEEP_LATEST, logs_dir=logs_dir)
    result = mgr.apply()
    assert result["rows_before"] == 5
    assert result["rows_after"] == 3
    assert result["rows_removed"] == 2
    assert result["policy"] == "KEEP_LATEST"
    assert mgr.current_row_count() == 3


def test_retention_archive_and_trim(tmp_path):
    logs_dir = str(tmp_path / "logs")
    model_name = "archive_model"

    logger = ClassificationLogger(model_name, logs_dir=logs_dir, buffer_size=1)
    for i in range(4):
        logger.log_prediction(image_name=f"{i}.jpg", pred_class="cat", confidence=0.9)

    mgr = DataRetentionManager(model_name, max_rows=2, policy=RetentionPolicy.ARCHIVE_AND_TRIM, logs_dir=logs_dir)
    result = mgr.apply()
    assert result["rows_archived"] == 2
    assert result["rows_after"] == 2
    assert mgr.archive_exists() is True
    assert mgr.archive_row_count() == 2

    # read_log should only see live segments, ignoring archive/
    live_table = read_log(logs_dir, model_name)
    assert live_table.num_rows == 2

    # Purge archive
    assert mgr.purge_archive() is True
    assert mgr.archive_row_count() == 0
    assert mgr.archive_exists() is False


def test_compact(tmp_path):
    logs_dir = str(tmp_path / "logs")
    model_name = "compact_model"

    logger = ClassificationLogger(model_name, logs_dir=logs_dir, buffer_size=1)
    for i in range(10):
        logger.log_prediction(image_name=f"{i}.jpg", pred_class="cat", confidence=0.9)

    model_dir = os.path.join(logs_dir, model_name)
    assert len(_live_segments(model_dir)) == 10

    # Compact with min_segments=5
    remaining = compact(logs_dir, model_name, min_segments=5)
    # Compaction merged oldest segments into 1, leaving fewer total
    assert remaining > 0
    assert remaining < 10
    assert len(_live_segments(model_dir)) == remaining

    # All rows preserved in order
    table = read_log(logs_dir, model_name)
    assert table.num_rows == 10
    assert table["image_name"].to_pylist() == [f"{i}.jpg" for i in range(10)]


def test_num_boxes_logged(tmp_path):
    """num_boxes is logged and is not treated as an id column."""
    logs_dir = str(tmp_path / "logs")
    model_name = "det_num_boxes"

    logger = DetectionLogger(model_name, logs_dir=logs_dir, buffer_size=1)
    logger.log_prediction(
        image_name="1.jpg", pred_class="car", confidence=0.9,
        bbox_x1=10, bbox_y1=20, bbox_x2=100, bbox_y2=200,
        num_boxes=5,
    )
    logger.log_prediction(
        image_name="2.jpg", pred_class="car", confidence=0.8,
        bbox_x1=10, bbox_y1=20, bbox_x2=100, bbox_y2=200,
        # num_boxes not provided, should be null
    )

    table = read_log(logs_dir, model_name)
    assert "num_boxes" in table.column_names
    values = table["num_boxes"].to_pylist()
    assert values[0] == 5
    assert values[1] is None

    # num_boxes should NOT be in _NON_METRIC_COLS
    from look_out_cv._utils import _NON_METRIC_COLS
    assert "num_boxes" not in _NON_METRIC_COLS


def test_ts_and_row_id_present(tmp_path):
    """Every row has ts (int64) and row_id (string)."""
    logs_dir = str(tmp_path / "logs")
    model_name = "ts_check"

    logger = ClassificationLogger(model_name, logs_dir=logs_dir, buffer_size=1)
    before = int(time.time() * 1000)
    logger.log_prediction(image_name="a.jpg", pred_class="cat", confidence=0.9)
    after = int(time.time() * 1000)

    table = read_log(logs_dir, model_name)
    assert table.schema.field("ts").type == pa.int64()
    assert table.schema.field("row_id").type == pa.string()
    assert table.schema.field("device_id").type == pa.string()

    ts_val = table["ts"].to_pylist()[0]
    assert before <= ts_val <= after

    row_id = table["row_id"].to_pylist()[0]
    assert len(row_id) == 32  # uuid4().hex


def test_time_windows_use_ts(tmp_path):
    from look_out_cv.metrics.drift_detection import DriftDetector, DriftTest
    from look_out_cv.metrics.metrics_rollup import MetricsRollup

    logs_dir = str(tmp_path / "logs")
    model_name = "time_window_model"
    model_dir = os.path.join(logs_dir, model_name)
    os.makedirs(model_dir)
    now = time.time_ns() // 1_000_000
    table = pa.table({
        "ts": pa.array([now - 2 * 86_400_000, now - 2 * 86_400_000,
                        now - 1_000, now - 1_000], type=pa.int64()),
        "confidence": pa.array([0.1, 0.2, 0.9, 0.95], type=pa.float32()),
    })
    pq.write_table(table, os.path.join(model_dir, "0000000000001_1_0.parquet"))

    rollup = MetricsRollup(model_name, last="24h", logs_dir=logs_dir)
    assert rollup.compute().loc["mean", "confidence"] == pytest.approx(0.925)

    detector = DriftDetector(
        model_name,
        current_window_time="24h",
        test=DriftTest.PSI,
        threshold=0.01,
        logs_dir=logs_dir,
    )
    reference, current = detector.load()._split()
    assert reference["confidence"].tolist() == pytest.approx([0.1, 0.2])
    assert current["confidence"].tolist() == pytest.approx([0.9, 0.95])
    assert detector.detect().results[0].column == "confidence"
