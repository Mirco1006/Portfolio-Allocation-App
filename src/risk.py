import numpy as np
import pandas as pd

TRADING_DAYS = 252

def calculate_portfolio_vol(weights: np.ndarray, cov: np.ndarray):
    """Return the volatility of the portfolio during the period"""
    portfolio_volatility = np.sqrt(np.dot(weights.T, np.dot(cov, weights)))
    return portfolio_volatility

def calculate_daily_returns(returns: pd.DataFrame, weights: np.ndarray) -> pd.Series:
    """Return daily returns from a price DataFrame (index=date, columns=tickers)."""
    w = np.asarray(weights).flatten()
    return returns.dot(w)

def drawdown_series(daily_returns: pd.Series) -> pd.Series:
    """Compute the full drawdown series (underwater curve)."""
    wealth = (1 + daily_returns).cumprod()
    peak = wealth.cummax()
    return (wealth - peak) / peak

def calculate_drawdown(daily_returns: pd.Series) -> float:
    """Return the maximum drawdown (single scalar)."""
    return drawdown_series(daily_returns).min()
