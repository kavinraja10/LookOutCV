"""MetricsRollup: aggregate logged metrics over time or row-count windows.

Windows may be expressed as:
- A *row count* (``window_size=N``) — N most-recent rows.
- A *time string* (``last="24h"``) — rows whose ``ts`` falls within the last N
  units.  Accepted suffixes: ``s``, ``m``, ``h``, ``d``.

Both may be combined; the intersection is used.
"""
from __future__ import annotations

import time
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

from look_out_cv._utils import read_log, _NON_METRIC_COLS


def _parse_duration_ms(spec: str) -> int:
    """Parse a duration string like '24h', '30m', '90s', '7d' to milliseconds."""
    spec = spec.strip()
    units = {"s": 1_000, "m": 60_000, "h": 3_600_000, "d": 86_400_000}
    suffix = spec[-1].lower()
    if suffix not in units:
        raise ValueError(
            f"Unknown duration suffix '{suffix}'. Use s/m/h/d, e.g. '24h'."
        )
    try:
        value = float(spec[:-1])
    except ValueError:
        raise ValueError(f"Invalid duration spec: '{spec}'")
    return int(value * units[suffix])


_AGG_MAP = {
    "mean": np.mean,
    "std":  np.std,
    "min":  np.min,
    "max":  np.max,
    "count": len,
}


class MetricsRollup:
    """Compute rolling aggregations over logged prediction metrics.

    Parameters
    ----------
    model_name:
        Name of the model whose logs should be aggregated.
    window_size:
        Number of most-recent rows to include.  ``None`` = all rows.
    last:
        Time-based window expressed as a duration string, e.g. ``"24h"``.
        Rows whose ``ts`` is older than *now − last* are excluded.  ``None``
        means no time filter.  Old rows without a ``ts`` are excluded when this
        is set.
    aggregations:
        Aggregation names to compute.  Defaults: mean, std, min, max.
    logs_dir:
        Root directory where model log directories live.
    """

    SUPPORTED_AGGREGATIONS: List[str] = list(_AGG_MAP.keys())

    def __init__(
        self,
        model_name: str,
        window_size: Optional[int] = None,
        last: Optional[str] = None,
        aggregations: Optional[List[str]] = None,
        logs_dir: str = "lookout_cv_logs",
    ) -> None:
        self.model_name = model_name
        self.window_size = window_size
        self.last = last
        self.logs_dir = logs_dir
        self.aggregations: List[str] = aggregations or ["mean", "std", "min", "max"]

        invalid = set(self.aggregations) - set(self.SUPPORTED_AGGREGATIONS)
        if invalid:
            raise ValueError(
                f"Unsupported aggregations: {invalid}. "
                f"Choose from {self.SUPPORTED_AGGREGATIONS}."
            )

        self._data: Optional[pd.DataFrame] = None

    def _load_data(self) -> pd.DataFrame:
        """Load all segment files and return a combined DataFrame."""
        return read_log(self.logs_dir, self.model_name).to_pandas()

    def _numeric_columns(self, df: pd.DataFrame) -> List[str]:
        cols = df.select_dtypes(include=[np.number]).columns.tolist()
        return [c for c in cols if c not in _NON_METRIC_COLS]

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

        if self.last is not None and "ts" in df.columns:
            cutoff = int(time.time() * 1000) - _parse_duration_ms(self.last)
            ts_col = pd.to_numeric(df["ts"], errors="coerce")
            df = df[ts_col >= cutoff]

        if self.window_size is not None:
            df = df.iloc[-self.window_size:]

        return df

    def compute(self) -> pd.DataFrame:
        """Compute rollup statistics for all numeric metric columns.

        Returns
        -------
        pd.DataFrame
            Indexed by aggregation name with one column per numeric metric.
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
        """
        if self.window_size is None:
            raise ValueError("``window_size`` must be set to use ``compute_rolling``.")
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
            + (f", window={self.window_size}" if self.window_size else ", window=ALL")
            + (f", last={self.last}" if self.last else ""),
            df.to_string(),
        ]
        return "\n".join(lines)
