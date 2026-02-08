## Disclaimer

This project is for educational purposes only and does not constitute investment advice. The author is not responsible for any financial decisions made based on this project. Data may be delayed or inaccurate.

## Overview

This project is an interactive portfolio allocation and optimization application built in Python.
It allows users to construct and analyze equity portfolios using classical portfolio theory and risk metrics, based on real market data.
The goal of the project is to demonstrate practical skills in:
- Portfolio construction
- Risk-return analysis
- Numerical optimization
- Financial data handling

## Features

- Download historical price data for selected assets
- Compute:
    - Returns
    - Volatility
    - Correlation matrix
- Portfolio construction methods:
  - Equal-weight portfolio
  - Minimum variance portfolio
  - Maximum Sharpe ratio portfolio
- Risk metrics:
  - Volatility
  - Sharpe ratio
  - Maximum drawdown
- Interactive visualization:
  - Portfolio weights
  - Efficient frontier
  - Performance metrics
- Streamlit-based user interface

## Methodology

The application relies on mean–variance portfolio theory (Markowitz framework).
Portfolio optimization is formulated as a constrained convex optimization problem.
All results are evaluated on historical data and are for educational purposes only.

## Technologies

- Python
- pandas, numpy
- yfinance (market data)
- cvxpy (portfolio optimization)
- matplotlib
- Streamlit


## Quickstart
**Python 3.11+ recommended**
```bash
pip install -r requirements.txt
streamlit run app.py
```
