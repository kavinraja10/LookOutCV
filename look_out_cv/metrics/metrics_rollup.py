"""MetricsRollup: aggregate logged metrics over time windows (row-count based).

Since the parquet logs don't store timestamps, rollup windows are expressed as
a number of most-recent rows (e.g. the last N predictions).
"""

from __future__ import annotations

import glob
import os
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq


_AGG_MAP = {
    "mean": np.mean,
    "std": np.std,
    "min": np.min,
    "max": np.max,
    "count": len,
}


class MetricsRollup:
    """Compute rolling aggregations over logged prediction metrics.

    Parameters
    ----------
    model_name:
        Name of the model whose logs should be aggregated.
    window_size:
        Number of most-recent rows to include in each rollup window.
        Pass ``None`` to aggregate over all available rows.
    aggregations:
        List of aggregation names to compute. Supported values:
        ``"mean"``, ``"std"``, ``"min"``, ``"max"``, ``"count"``.
    logs_dir:
        Root directory where the model's parquet log files live.
    """

    SUPPORTED_AGGREGATIONS: List[str] = list(_AGG_MAP.keys())

    def __init__(
        self,
        model_name: str,
        window_size: Optional[int] = None,
        aggregations: Optional[List[str]] = None,
        logs_dir: str = "lookout_cv_logs",
    ) -> None:
        self.model_name = model_name
        self.window_size = window_size
        self.logs_dir = logs_dir
        self.aggregations: List[str] = aggregations or ["mean", "std", "min", "max"]

        invalid = set(self.aggregations) - set(self.SUPPORTED_AGGREGATIONS)
        if invalid:
            raise ValueError(
                f"Unsupported aggregations: {invalid}. "
                f"Choose from {self.SUPPORTED_AGGREGATIONS}."
            )

        self._data: Optional[pd.DataFrame] = None

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _load_data(self) -> pd.DataFrame:
        """Load all parquet files for the model into a single DataFrame."""
        pattern = os.path.join(self.logs_dir, self.model_name, "*.parquet")
        files = glob.glob(pattern)
        if not files:
            raise FileNotFoundError(
                f"No parquet log files found under '{self.logs_dir}/{self.model_name}'."
            )
        tables = [pq.read_table(f) for f in files]
        combined = pa.concat_tables(tables).to_pandas()
        return combined

    def _numeric_columns(self, df: pd.DataFrame) -> List[str]:
        return df.select_dtypes(include=[np.number]).columns.tolist()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def load(self) -> "MetricsRollup":
        """Load (or reload) the underlying log data. Returns ``self`` for chaining."""
        self._data = self._load_data()
        return self

    @property
    def data(self) -> pd.DataFrame:
        if self._data is None:
            self.load()
        assert self._data is not None
        return self._data

    def _window_df(self) -> pd.DataFrame:
        df = self.data
        if self.window_size is not None:
            df = df.iloc[-self.window_size:]
        return df

    def compute(self) -> pd.DataFrame:
        """Compute rollup statistics for all numeric columns.

        Returns
        -------
        pd.DataFrame
            A DataFrame indexed by aggregation name with one column per
            numeric metric column.

        Example
        -------
        >>> rollup = MetricsRollup("my_model", window_size=100)
        >>> print(rollup.compute())
        """
        window = self._window_df()
        numeric_cols = self._numeric_columns(window)
        if not numeric_cols:
            return pd.DataFrame()

        results: Dict[str, Dict[str, float]] = {}
        for agg_name in self.aggregations:
            fn = _AGG_MAP[agg_name]
            row: Dict[str, float] = {}
            for col in numeric_cols:
                values = window[col].dropna().values
                row[col] = float(fn(values)) if len(values) > 0 else float("nan")
            results[agg_name] = row

        return pd.DataFrame(results).T  # rows = aggregations, cols = metrics

    def compute_rolling(self, step: int = 1) -> Dict[str, pd.DataFrame]:
        """Compute rollup over a sliding window across the full log history.

        Parameters
        ----------
        step:
            How many rows to advance the window each iteration.

        Returns
        -------
        dict
            Mapping of aggregation name to DataFrame where each row is one
            window position and columns are numeric metric columns.
        """
        if self.window_size is None:
            raise ValueError(
                "``window_size`` must be set to use ``compute_rolling``."
            )
        df = self.data
        numeric_cols = self._numeric_columns(df)
        n = len(df)
        ws = self.window_size

        records: Dict[str, List[Dict[str, float]]] = {a: [] for a in self.aggregations}
        indices: List[int] = []

        for start in range(0, max(1, n - ws + 1), step):
            end = start + ws
            window = df.iloc[start:end]
            indices.append(end - 1)
            for agg_name in self.aggregations:
                fn = _AGG_MAP[agg_name]
                row: Dict[str, float] = {}
                for col in numeric_cols:
                    values = window[col].dropna().values
                    row[col] = float(fn(values)) if len(values) > 0 else float("nan")
                records[agg_name].append(row)

        return {
            agg: pd.DataFrame(rows, index=indices)
            for agg, rows in records.items()
        }

    def summary(self) -> str:
        """Return a human-readable string summary of the rollup."""
        df = self.compute()
        if df.empty:
            return "No numeric metrics found."
        lines = [
            f"MetricsRollup - model='{self.model_name}'"
            + (f", window={self.window_size}" if self.window_size else ", window=ALL"),
            df.to_string(),
        ]
        return "\n".join(lines)
