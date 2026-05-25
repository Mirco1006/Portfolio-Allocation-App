import streamlit as st
import pandas as pd
import yfinance as yf
import requests

@st.cache_data
def download_data(tickers, period="1y"):
    """Download adjusted close prices from Yahoo Finance via yfinance.

    Parameters
    ----------
    tickers : list[str] or str
        Ticker symbol(s) to download (e.g. ['AAPL', 'MSFT']).
    period : str
        Lookback period accepted by yfinance ('1y', '3y', '5y').

    Returns
    -------
    pd.DataFrame
        Adjusted close prices (index=dates, columns=tickers).
    """
    if isinstance(tickers, str):
        tickers = [tickers]
    #We download the data from yfinance
    data = yf.download(
        tickers,
        period=period,
        group_by="ticker",
        auto_adjust=True
    )
    #We create an empty dictionary where we will put all the closing prices
    prices = {}
    #We do a loop where we will add the close price for each stock
    for t in tickers:
        prices[t] = data[t]["Close"]
    #We convert the disctionnary in the form of a dataframe
    prices = pd.DataFrame(prices).dropna(how="all")
    #We return the dataframe with all the closing price
    return prices


@st.cache_data(ttl=86400)  # 1 jour
def load_sp500_table_requests() -> pd.DataFrame:
    """
    Loads S&P 500 constituents table from Wikipedia using requests,
    then parses it with pandas.read_html().

    Returns
    -------
    pd.DataFrame
        DataFrame containing at least Symbol, Security, GICS Sector, etc.
        Includes YahooSymbol column (BRK.B -> BRK-B).
    """
    url = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"

    headers = {
        "User-Agent": (
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
            "AppleWebKit/537.36 (KHTML, like Gecko) "
            "Chrome/120.0 Safari/537.36"
        )
    }

    # Download HTML page
    response = requests.get(url, headers=headers, timeout=15)
    response.raise_for_status()  # lève une erreur si HTTP 4xx/5xx

    html = response.text

    # Parse the main constituents table
    tables = pd.read_html(html)
    if len(tables) == 0:
        raise ValueError("No tables found on the Wikipedia page.")

    df = tables[0].copy()  # le premier tableau = constituents

    # Normalize ticker for Yahoo Finance (BRK.B -> BRK-B)
    df["YahooSymbol"] = df["Symbol"].astype(str).str.replace(".", "-", regex=False)

    return df
