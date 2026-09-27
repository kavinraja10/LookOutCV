import glob
import os
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
from tabulate import tabulate


class DataInsightsCalculator:
    def __init__(self, model_name: str, parquet_folder_path: str = "lookout_cv_logs"):
        self.model_name = model_name
        self.parquet_file_path = parquet_folder_path
        self.data = self._load_data()

    def _load_data(self):
        """Load the first parquet file for the model into a pandas DataFrame."""
        parquet_dir = Path(self.parquet_file_path) / self.model_name
        parquet_files = sorted(parquet_dir.glob("*.parquet"))
        if not parquet_files:
            raise FileNotFoundError(f"No parquet files found in {parquet_dir}")

        try:
            return pq.read_table(parquet_files[0]).to_pandas()
        except Exception as exc:  # pragma: no cover - defensive error reporting
            raise IOError(f"Failed to load data from parquet file: {exc}") from exc

    def calculate_summary_statistics(self):
        """Return summary statistics for all numeric columns."""
        return self.data.describe()

    def identify_outliers(self, iqr_multiplier: float = 1.5):
        """Return rows that exceed the IQR threshold for one or more numeric columns."""
        numeric_data = self.data.select_dtypes(include=[np.number])
        if numeric_data.empty:
            return self.data.iloc[0:0]

        outlier_mask = pd.Series(False, index=self.data.index)
        for column in numeric_data.columns:
            q1 = numeric_data[column].quantile(0.25)
            q3 = numeric_data[column].quantile(0.75)
            iqr = q3 - q1
            lower_bound = q1 - iqr_multiplier * iqr
            upper_bound = q3 + iqr_multiplier * iqr
            outlier_mask |= (numeric_data[column] < lower_bound) | (numeric_data[column] > upper_bound)

        return self.data[outlier_mask]

    def calculate_correlation_matrix(self):
        """Calculate the correlation matrix for numeric columns."""
        numeric_data = self.data.select_dtypes(include=[np.number])
        return numeric_data.corr()

    def generate_insights(self):
        """Generate a simple view of summary statistics, outliers, and correlations."""
        return {
            "summary_statistics": self.calculate_summary_statistics(),
            "outliers": self.identify_outliers(),
            "correlation_matrix": self.calculate_correlation_matrix(),
        }

    def print_insights(self):
        """Print the generated insights as tables in the CLI."""
        insights = self.generate_insights()

        print("\nSummary Statistics:")
        print(tabulate(insights["summary_statistics"], headers="keys", tablefmt="grid"))

        print("\nOutliers:")
        if insights["outliers"].empty:
            print("No outliers detected.")
        else:
            print(tabulate(insights["outliers"], headers="keys", tablefmt="grid"))

        print("\nCorrelation Matrix:")
        print(tabulate(insights["correlation_matrix"], headers="keys", tablefmt="grid"))