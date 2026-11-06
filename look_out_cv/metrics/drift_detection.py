"""Basic drift detection for LookOutCV logged metrics.

Compares a *reference* distribution (e.g. training / baseline window) against
a *current* distribution (e.g. recent predictions) using lightweight
statistical tests.

Supported tests
---------------
- PSI  - Population Stability Index
- KS   - Kolmogorov-Smirnov two-sample test (via SciPy if available)
- Z_MEAN - Simple mean-shift z-test for quick sanity checks
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from look_out_cv._utils import read_log, _NON_METRIC_COLS
from look_out_cv.metrics.metrics_rollup import _parse_duration_ms


class DriftTest(Enum):
    PSI    = auto()
    KS     = auto()
    Z_MEAN = auto()


@dataclass
class ColumnDriftResult:
    """Drift detection result for a single metric column."""

    column: str
    test: DriftTest
    statistic: float
    """Test-specific numeric value (PSI score, KS statistic, or z-score)."""
    threshold: float
    """The threshold above which drift is flagged."""
    drifted: bool
    """Whether the statistic exceeds the threshold."""
    p_value: Optional[float] = None
    """p-value where available (KS test only)."""

    def __str__(self) -> str:
        p_str = f", p={self.p_value:.4f}" if self.p_value is not None else ""
        flag = "DRIFT" if self.drifted else "OK"
        return (
            f"[{flag}] {self.column} | {self.test.name} "
            f"stat={self.statistic:.4f}{p_str} (threshold={self.threshold})"
        )


@dataclass
class DriftReport:
    """Aggregated drift detection report across all monitored columns."""

    results: List[ColumnDriftResult] = field(default_factory=list)

    @property
    def drifted_columns(self) -> List[str]:
        return [r.column for r in self.results if r.drifted]

    @property
    def has_drift(self) -> bool:
        return any(r.drifted for r in self.results)

    def to_dataframe(self) -> pd.DataFrame:
        return pd.DataFrame(
            [
                {
                    "column":    r.column,
                    "test":      r.test.name,
                    "statistic": r.statistic,
                    "threshold": r.threshold,
                    "drifted":   r.drifted,
                    "p_value":   r.p_value,
                }
                for r in self.results
            ]
        )

    def __str__(self) -> str:
        lines = ["=== Drift Report ==="]
        for r in self.results:
            lines.append(str(r))
        if self.has_drift:
            lines.append(f"\nDrift detected in: {self.drifted_columns}")
        else:
            lines.append("\nNo drift detected.")
        return "\n".join(lines)


def _psi_score(reference: np.ndarray, current: np.ndarray, bins: int = 10) -> float:
    """Compute PSI between two 1-D arrays."""
    min_val = min(reference.min(), current.min())
    max_val = max(reference.max(), current.max())
    if min_val == max_val:
        return 0.0

    bin_edges = np.linspace(min_val, max_val, bins + 1)
    ref_counts, _ = np.histogram(reference, bins=bin_edges)
    cur_counts, _ = np.histogram(current, bins=bin_edges)

    eps = 1e-8
    ref_pct = ref_counts / max(ref_counts.sum(), 1) + eps
    cur_pct = cur_counts / max(cur_counts.sum(), 1) + eps

    psi = float(np.sum((cur_pct - ref_pct) * np.log(cur_pct / ref_pct)))
    return psi


def _ks_test(reference: np.ndarray, current: np.ndarray) -> Tuple[float, float]:
    """Two-sample KS test. Returns (statistic, p_value)."""
    try:
        from scipy.stats import ks_2samp  # type: ignore
        stat, p = ks_2samp(reference, current)
        return float(stat), float(p)
    except ImportError:
        ref_sorted = np.sort(reference)
        cur_sorted = np.sort(current)
        all_vals = np.sort(np.concatenate([ref_sorted, cur_sorted]))
        cdf_ref = np.searchsorted(ref_sorted, all_vals, side="right") / len(ref_sorted)
        cdf_cur = np.searchsorted(cur_sorted, all_vals, side="right") / len(cur_sorted)
        stat = float(np.max(np.abs(cdf_ref - cdf_cur)))
        return stat, float("nan")


def _z_mean_test(reference: np.ndarray, current: np.ndarray) -> float:
    """Z-score for the difference in means, using reference std."""
    ref_std = np.std(reference)
    if ref_std == 0:
        return 0.0
    z = abs(np.mean(current) - np.mean(reference)) / (ref_std / np.sqrt(len(current)))
    return float(z)


class DriftDetector:
    """Detect distribution drift in logged CV metrics.

    Parameters
    ----------
    model_name:
        Name of the model to monitor.
    reference_window:
        Number of oldest rows to use as the reference distribution.
        If ``None``, the first half of available rows is used.
    current_window:
        Number of most-recent rows to treat as the current distribution.
        If ``None``, the second half of available rows is used.
    current_window_time:
        Time-based current window, e.g. ``"24h"``.  When set, overrides
        *current_window* for selecting the current slice.  Old rows without
        ``ts`` are excluded from the time-filtered slice.
    test:
        Statistical test to use (``PSI``, ``KS``, or ``Z_MEAN``).
    threshold:
        Decision threshold.  Defaults: PSI → 0.2, KS → 0.05, Z_MEAN → 3.0.
    logs_dir:
        Root directory for model logs.
    columns:
        Subset of numeric columns to monitor.  ``None`` = all numeric columns.
    """

    _DEFAULT_THRESHOLDS = {
        DriftTest.PSI:    0.2,
        DriftTest.KS:     0.05,
        DriftTest.Z_MEAN: 3.0,
    }

    def __init__(
        self,
        model_name: str,
        reference_window: Optional[int] = None,
        current_window: Optional[int] = None,
        current_window_time: Optional[str] = None,
        test: DriftTest = DriftTest.PSI,
        threshold: Optional[float] = None,
        logs_dir: str = "lookout_cv_logs",
        columns: Optional[List[str]] = None,
    ) -> None:
        self.model_name = model_name
        self.reference_window = reference_window
        self.current_window = current_window
        self.current_window_time = current_window_time
        self.test = test
        self.threshold = threshold if threshold is not None else self._DEFAULT_THRESHOLDS[test]
        self.logs_dir = logs_dir
        self.columns = columns
        self._data: Optional[pd.DataFrame] = None

    def _load_data(self) -> pd.DataFrame:
        return read_log(self.logs_dir, self.model_name).to_pandas()

    def load(self) -> "DriftDetector":
        """Load (or reload) underlying log data. Returns ``self`` for chaining."""
        self._data = self._load_data()
        return self

    @property
    def data(self) -> pd.DataFrame:
        if self._data is None:
            self.load()
        assert self._data is not None
        return self._data

    def _split(self) -> Tuple[pd.DataFrame, pd.DataFrame]:
        df = self.data
        n = len(df)
        if n < 2:
            raise ValueError(
                f"Need at least 2 rows for drift detection; got {n}."
            )

        half = n // 2
        ref_n = self.reference_window if self.reference_window is not None else half

        if self.current_window_time is not None and "ts" in df.columns:
            cutoff = int(time.time() * 1000) - _parse_duration_ms(self.current_window_time)
            ts_col = pd.to_numeric(df["ts"], errors="coerce")
            current = df[ts_col >= cutoff]
            reference = df[ts_col < cutoff]
            if self.reference_window is not None:
                reference = reference.iloc[-self.reference_window:]
            if len(reference) < 2 or len(current) < 2:
                raise ValueError("Need at least 2 rows in reference and current windows.")
            return reference, current
        elif self.current_window is not None:
            current = df.iloc[-self.current_window:]
        else:
            current = df.iloc[-(n - half):]

        cur_n = len(current)
        if ref_n + cur_n > n:
            raise ValueError(
                f"reference_window ({ref_n}) + current_window ({cur_n}) "
                f"exceeds total rows ({n})."
            )

        reference = df.iloc[:ref_n]
        return reference, current

    def _monitored_columns(self, df: pd.DataFrame) -> List[str]:
        numeric = df.select_dtypes(include=[np.number]).columns.tolist()
        # Exclude bookkeeping non-metric columns
        numeric = [c for c in numeric if c not in _NON_METRIC_COLS]
        if self.columns:
            missing = set(self.columns) - set(numeric)
            if missing:
                raise ValueError(
                    f"Requested columns {missing} not found among numeric columns."
                )
            return [c for c in self.columns if c in numeric]
        return numeric

    def detect(self) -> DriftReport:
        """Run drift detection and return a :class:`DriftReport`."""
        reference, current = self._split()
        cols = self._monitored_columns(reference)
        results: List[ColumnDriftResult] = []

        for col in cols:
            ref_vals = reference[col].dropna().values.astype(float)
            cur_vals = current[col].dropna().values.astype(float)

            if len(ref_vals) < 2 or len(cur_vals) < 2:
                continue

            if self.test == DriftTest.PSI:
                stat = _psi_score(ref_vals, cur_vals)
                drifted = stat > self.threshold
                results.append(ColumnDriftResult(col, self.test, stat, self.threshold, drifted))

            elif self.test == DriftTest.KS:
                stat, p_value = _ks_test(ref_vals, cur_vals)
                drifted = (not np.isnan(p_value)) and (p_value < self.threshold)
                results.append(ColumnDriftResult(col, self.test, stat, self.threshold, drifted, p_value))

            elif self.test == DriftTest.Z_MEAN:
                stat = _z_mean_test(ref_vals, cur_vals)
                drifted = stat > self.threshold
                results.append(ColumnDriftResult(col, self.test, stat, self.threshold, drifted))

        return DriftReport(results)

    def detect_all(self) -> Dict[str, DriftReport]:
        """Run all three drift tests and return a mapping of test name to report."""
        original_test = self.test
        original_threshold = self.threshold
        reports: Dict[str, DriftReport] = {}
        for test in DriftTest:
            self.test = test
            self.threshold = self._DEFAULT_THRESHOLDS[test]
            reports[test.name] = self.detect()
        self.test = original_test
        self.threshold = original_threshold
        return reports
