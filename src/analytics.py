import numpy as np
import pandas as pd
from sklearn.covariance import LedoitWolf

#Define the number of trading days per year
TRADING_DAYS = 252

def calculate_return(prices: pd.DataFrame, method: str = "simple"):
    """Return daily simple or log returns from a price DataFrame (index=date, columns=tickers)."""
    if method == "log":
        returns = np.log(prices/prices.shift(1))
    else:
        returns = prices.pct_change()
    return returns.dropna(how="any")

def calculate_correlation(returns: pd.DataFrame):
    """Return the correlation between each stock"""
    return returns.corr()

def covariance_ledoit_wolf(returns_df, annualization: int = 252) -> np.ndarray:
    """Robust PSD covariance via Ledoit-Wolf shrinkage."""
    clean = returns_df.dropna(how="any")
    X = clean.values
    lw = LedoitWolf().fit(X)
    cov = lw.covariance_ * annualization
    cov = 0.5 * (cov + cov.T)
    return cov