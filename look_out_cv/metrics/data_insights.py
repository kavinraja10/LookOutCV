"""DataInsightsCalculator: summary statistics, outlier detection, correlation."""
from __future__ import annotations

import numpy as np
import pandas as pd

from look_out_cv._utils import read_log, _NON_METRIC_COLS


class DataInsightsCalculator:
    def __init__(self, model_name: str, parquet_folder_path: str = "lookout_cv_logs"):
        self.model_name = model_name
        self.parquet_file_path = parquet_folder_path
        self.data = self._load_data()

    def _load_data(self) -> pd.DataFrame:
        """Load ALL segment files for the model into a pandas DataFrame."""
        try:
            return read_log(self.parquet_file_path, self.model_name).to_pandas()
        except Exception as exc:
            raise IOError(f"Failed to load data for model '{self.model_name}': {exc}") from exc

    def _numeric_data(self) -> pd.DataFrame:
        cols = self.data.select_dtypes(include=[np.number]).columns
        metric_cols = [c for c in cols if c not in _NON_METRIC_COLS]
        return self.data[metric_cols]

    def calculate_summary_statistics(self) -> pd.DataFrame:
        """Return summary statistics for all numeric metric columns."""
        return self._numeric_data().describe()

    def identify_outliers(self, iqr_multiplier: float = 1.5) -> pd.DataFrame:
        """Return rows that exceed the IQR threshold for one or more numeric columns."""
        numeric_data = self._numeric_data()
        if numeric_data.empty:
            return self.data.iloc[0:0]

        outlier_mask = pd.Series(False, index=self.data.index)
        for column in numeric_data.columns:
            q1 = numeric_data[column].quantile(0.25)
            q3 = numeric_data[column].quantile(0.75)
            iqr = q3 - q1
            lower_bound = q1 - iqr_multiplier * iqr
            upper_bound = q3 + iqr_multiplier * iqr
            outlier_mask |= (
                (numeric_data[column] < lower_bound)
                | (numeric_data[column] > upper_bound)
            )

        return self.data[outlier_mask]

    def calculate_correlation_matrix(self) -> pd.DataFrame:
        """Calculate the correlation matrix for numeric metric columns."""
        return self._numeric_data().corr()

    def generate_insights(self) -> dict:
        """Generate a simple view of summary statistics, outliers, and correlations."""
        return {
            "summary_statistics": self.calculate_summary_statistics(),
            "outliers":           self.identify_outliers(),
            "correlation_matrix": self.calculate_correlation_matrix(),
        }

    def print_insights(self) -> None:
        """Print the generated insights as tables."""
        insights = self.generate_insights()

        print("\nSummary Statistics:")
        print(insights["summary_statistics"].to_string())

        print("\nOutliers:")
        if insights["outliers"].empty:
            print("No outliers detected.")
        else:
            print(insights["outliers"].to_string())

        print("\nCorrelation Matrix:")
        print(insights["correlation_matrix"].to_string())
