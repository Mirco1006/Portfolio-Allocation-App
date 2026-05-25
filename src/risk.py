import numpy as np
import pandas as pd

def calculate_portfolio_vol(weights: np.ndarray, cov: np.ndarray):
    """Compute annualized portfolio volatility as sqrt(w' Σ w).

    Parameters
    ----------
    weights : np.ndarray
        (N,) portfolio weight vector summing to 1.
    cov : np.ndarray
        Annualized N×N covariance matrix.

    Returns
    -------
    float
        Annualized portfolio standard deviation.
    """
    portfolio_volatility = np.sqrt(np.dot(weights.T, np.dot(cov, weights)))
    return portfolio_volatility

def calculate_daily_returns(returns: pd.DataFrame, weights: np.ndarray) -> pd.Series:
    """Compute weighted portfolio daily returns as returns @ weights.

    Parameters
    ----------
    returns : pd.DataFrame
        Daily asset returns (index=dates, columns=tickers).
    weights : np.ndarray
        (N,) portfolio weight vector.

    Returns
    -------
    pd.Series
        Daily portfolio returns.
    """
    w = np.asarray(weights).flatten()
    return returns.dot(w)

def drawdown_series(daily_returns: pd.Series) -> pd.Series:
    """Compute the full drawdown (underwater) series.

    Parameters
    ----------
    daily_returns : pd.Series
        Daily portfolio returns.

    Returns
    -------
    pd.Series
        Drawdown values (always <= 0). Zero means at peak.
    """
    wealth = (1 + daily_returns).cumprod()
    peak = wealth.cummax()
    return (wealth - peak) / peak

def calculate_drawdown(daily_returns: pd.Series) -> float:
    """Return the maximum drawdown (worst peak-to-trough decline).

    Parameters
    ----------
    daily_returns : pd.Series
        Daily portfolio returns.

    Returns
    -------
    float
        Maximum drawdown as a negative decimal (e.g. -0.25 means -25%).
    """
    return drawdown_series(daily_returns).min()
