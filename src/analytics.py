import numpy as np
import pandas as pd
from sklearn.covariance import LedoitWolf

#Define the number of trading days per year
TRADING_DAYS = 252

def calculate_return(prices: pd.DataFrame, method: str = "simple"):
    """Compute daily simple or log returns from a price DataFrame.

    Parameters
    ----------
    prices : pd.DataFrame
        DataFrame of adjusted close prices (index=dates, columns=tickers).
    method : str
        'simple' for arithmetic returns, 'log' for logarithmic returns.

    Returns
    -------
    pd.DataFrame
        Daily returns with first row (NaN) dropped.
    """    
    if method == "log":
        returns = np.log(prices/prices.shift(1))
    else:
        returns = prices.pct_change()
    return returns.dropna(how="any")

def calculate_correlation(returns: pd.DataFrame):
    """Compute pairwise Pearson correlation matrix of asset returns.

    Parameters
    ----------
    returns : pd.DataFrame
        Daily returns (index=dates, columns=tickers).

    Returns
    -------
    pd.DataFrame
        N×N correlation matrix.
    """    
    return returns.corr()

def covariance_ledoit_wolf(returns_df, annualization: int = 252) -> np.ndarray:
    """Estimate a robust positive semi-definite covariance matrix using Ledoit-Wolf shrinkage.

    Parameters
    ----------
    returns_df : pd.DataFrame
        Daily returns (index=dates, columns=tickers). Rows with NaN are dropped.
    annualization : int
        Factor to annualize daily covariance (252 for daily data).

    Returns
    -------
    np.ndarray
        Annualized N×N covariance matrix (symmetric, PSD).
    """
    clean = returns_df.dropna(how="any")
    X = clean.values
    lw = LedoitWolf().fit(X)
    cov = lw.covariance_ * annualization
    cov = 0.5 * (cov + cov.T)
    return cov