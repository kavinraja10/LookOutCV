import glob
import os


import pandas as pd
import pyarrow.parquet as pq
import numpy as np
from tabulate import tabulate

class DataInsightsCalculator:
    def __init__(self, model_name: str, parquet_folder_path: str  = "lookout_cv_logs"):
        """Initialize the Data Insights Calculator.
        
        Args:
            model_name: Name of the model being analyzed.
            parquet_file_path: Path to the parquet file containing logged data.
        """
        self.model_name = model_name
        self.parquet_file_path = parquet_folder_path
        self.data = self._load_data()

    def _load_data(self):
        """Load data from the parquet file into a pandas DataFrame."""
        try:
            parquet_files = glob.glob(os.path.join(self.parquet_file_path,self.model_name, "*.parquet"))
            if not parquet_files:
                raise FileNotFoundError(f"No parquet files found in {self.parquet_file_path}")
            table = pq.read_table(parquet_files[0])
            return table.to_pandas()
        except Exception as e:
            raise IOError(f"Failed to load data from parquet file: {e}")

    def calculate_summary_statistics(self):
        """Calculate summary statistics for all numeric columns."""
        return self.data.describe()

    def identify_outliers(self, iqr_multiplier=1.5):
        """Identify outliers using the Interquartile Range (IQR) method.
        
        Args:
            iqr_multiplier: Multiplier for IQR to define outlier boundaries (default 1.5).
        
        Returns:
            A DataFrame containing rows identified as outliers.
        """
        numeric_data = self.data.select_dtypes(include=[np.number])
        outliers_mask = pd.Series(False, index=self.data.index)
        
        for column in numeric_data.columns:
            Q1 = numeric_data[column].quantile(0.25)
            Q3 = numeric_data[column].quantile(0.75)
            IQR = Q3 - Q1
            
            lower_bound = Q1 - iqr_multiplier * IQR
            upper_bound = Q3 + iqr_multiplier * IQR
            
            column_outliers = (numeric_data[column] < lower_bound) | (numeric_data[column] > upper_bound)
            outliers_mask = outliers_mask | column_outliers
        
        return self.data[outliers_mask]

    def calculate_correlation_matrix(self):
        """Calculate the correlation matrix for numeric columns."""
        numeric_data = self.data.select_dtypes(include=[np.number])
        return numeric_data.corr()

    def generate_insights(self):
        """Generate insights from the data, including summary statistics, outliers, and correlations."""
        insights = {
            "summary_statistics": self.calculate_summary_statistics(),
            "outliers": self.identify_outliers(),
            "correlation_matrix": self.calculate_correlation_matrix(),
        }
        return insights

    def print_insights(self):
        """Print the generated insights as a table in the CLI."""
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

if __name__ == "__main__":
    calculator = DataInsightsCalculator("my_model", parquet_folder_path="lookout_cv_logs")
    calculator.print_insights()