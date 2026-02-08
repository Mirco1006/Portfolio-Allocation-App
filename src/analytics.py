import numpy as np
import pandas as pd

#Define the number of trading days per year
TRADING_DAYS = 252

def calculate_return(prices: pd.DataFrame, method: str = "simple"):
    """Return daily simple or log returns from a price DataFrame (index=date, columns=tickers)."""
    if method == "log":
        returns = np.log(prices/prices.shift(1))
    else:
        returns = prices.pct_change()
    return returns.dropna(how="any")

def calculate_mean_return(returns: pd.DataFrame, annualize: bool = True):
    """Return the mean returns and the covariance matrix from a price DataFrame"""
    mu = returns.mean()
    cov = returns.cov()
    if annualize:
        mu = mu * TRADING_DAYS
        cov = cov * TRADING_DAYS
    return mu, cov

def calculate_volatility(returns: pd.DataFrame, annualize: bool = True):
    """Return the volatility from a price DataFrame (index=date, columns=tickers)."""
    volatility = returns.std()
    if annualize:
        volatility = volatility * np.sqrt(TRADING_DAYS)
    return volatility

def calculate_correlation(returns: pd.DataFrame):
    """Return the correlation between each stock"""
    return returns.corr()