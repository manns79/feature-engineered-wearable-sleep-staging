"""Simple IBI and HRV feature helpers."""

from __future__ import annotations

import numpy as np
import pandas as pd


def summarize_ibi(values: pd.Series, prefix: str = "IBI") -> dict[str, float]:
    """Compute short-window HRV proxies from interbeat intervals."""
    ibi = pd.to_numeric(values, errors="coerce").dropna()
    features = {
        f"{prefix}_sdnn": np.nan,
        f"{prefix}_rmssd": np.nan,
        f"{prefix}_pnn50": np.nan,
    }
    if len(ibi) < 2:
        return features

    diffs = ibi.diff().dropna()
    features[f"{prefix}_sdnn"] = float(ibi.std(ddof=1))
    features[f"{prefix}_rmssd"] = float(np.sqrt(np.mean(np.square(diffs))))
    features[f"{prefix}_pnn50"] = float((diffs.abs() > 0.05).mean())
    return features
