import os
import pandas as pd
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest
from metrics.data_insights import DataInsightsCalculator

@pytest.fixture
def sample_parquet(tmp_path):
    # Create dummy data
    df = pd.DataFrame({
        "feature1": [1, 2, 3, 100],
        "feature2": [10, 20, 30, 40],
        "category": ["A", "B", "C", "D"]
    })
    table = pa.Table.from_pandas(df)

    # Create directories: lookout_cv_logs/<model_name>/
    model_name = "test_model"
    dir_path = tmp_path / "lookout_cv_logs" / model_name
    dir_path.mkdir(parents=True)
    parquet_path = dir_path / "test.parquet"
    pq.write_table(table, parquet_path)

    return tmp_path, model_name

def test_load_data_success(sample_parquet):
    tmp_path, model_name = sample_parquet
    calc = DataInsightsCalculator(model_name=model_name, parquet_folder_path=os.path.join(tmp_path, "lookout_cv_logs"))
    assert not calc.data.empty
    assert "feature1" in calc.data.columns

def test_summary_statistics(sample_parquet):
    tmp_path, model_name = sample_parquet
    calc = DataInsightsCalculator(model_name=model_name, parquet_folder_path=os.path.join(tmp_path, "lookout_cv_logs"))
    summary = calc.calculate_summary_statistics()
    assert "feature1" in summary.columns
    assert summary.loc["mean", "feature1"] > 0

def test_identify_outliers(sample_parquet):
    tmp_path, model_name = sample_parquet
    calc = DataInsightsCalculator(model_name=model_name, parquet_folder_path=os.path.join(tmp_path, "lookout_cv_logs"))
    outliers = calc.identify_outliers()
    # The value 100 is an outlier
    assert not outliers.empty
    assert 100 in outliers["feature1"].values

def test_correlation_matrix(sample_parquet):
    tmp_path, model_name = sample_parquet
    calc = DataInsightsCalculator(model_name=model_name, parquet_folder_path=os.path.join(tmp_path, "lookout_cv_logs"))
    corr = calc.calculate_correlation_matrix()
    assert "feature1" in corr.columns
    assert "feature2" in corr.columns

def test_generate_insights(sample_parquet):
    tmp_path, model_name = sample_parquet
    calc = DataInsightsCalculator(model_name=model_name, parquet_folder_path=os.path.join(tmp_path, "lookout_cv_logs"))
    insights = calc.generate_insights()
    assert "summary_statistics" in insights
    assert "outliers" in insights
    assert "correlation_matrix" in insights
