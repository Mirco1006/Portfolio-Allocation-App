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

def calculate_sharpe_ratio(daily_returns: pd.Series, rf: float=0.0) -> float:
    """Return sharpe ratio of the portfolio"""
    mu_ann = daily_returns.mean() * TRADING_DAYS
    vol_ann = daily_returns.std() * np.sqrt(TRADING_DAYS)
    return (mu_ann - rf) / vol_ann if vol_ann != 0 else np.nan

def calculate_drawdown(daily_return: pd.Series) -> float:
    """Return the maximum drawdown"""
    wealth = (1 + daily_return).cumprod()
    peak = wealth.cummax()
    drawdown = (wealth - peak) / peak
    return drawdown.min()
