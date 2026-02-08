import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter
import matplotlib.ticker as mtick

def plot_weights(stocks, weights):
    """Return barplot with the stock weights"""
    fig, ax =  plt.subplots(figsize=(6, 4))
    ax.bar(stocks, weights)
    ax.set_title("Portfolio weights")
    ax.set_xlabel("Stock tickers")
    ax.set_ylabel("Weights")
    ax.yaxis.set_major_formatter(mtick.PercentFormatter(xmax=1))
    fig.tight_layout()
    return fig

def plot_correlation(correlations):
    """Return plot with the stock correlations"""
    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(correlations.values, vmin=-1, vmax=1)
    ax.set_title("Correlation matrix")

    ax.set_xticks(range(len(correlations.columns)))
    ax.set_yticks(range(len(correlations.index)))
    ax.set_xticklabels(correlations.columns, rotation=45, ha="right")
    ax.set_yticklabels(correlations.index)

    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label("Correlation")
    cbar.formatter = FuncFormatter(lambda x, pos: f"{x*100:.0f}%")
    cbar.update_ticks()

    fig.tight_layout()
    return fig

def plot_portfolio_performance(daily_returns):
    """Return a plot with the portfolio performance over the period"""
    wealth = (1 + daily_returns).cumprod()
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(wealth.index, wealth.values)
    ax.set_title("Portfolio value (base 1.0)")
    ax.set_xlabel("Date")
    ax.set_ylabel("Value")
    ax.tick_params(axis="x", rotation=45)
    fig.tight_layout()
    return fig

def plot_drawdown(daily_returns):
    """Return a plot with the maximum drawdown for each day during the period"""
    wealth = (1 + daily_returns).cumprod()
    peak = wealth.cummax()
    dd = (wealth - peak) / peak

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(dd.index, dd.values)
    ax.set_title("Drawdown")
    ax.set_xlabel("Date")
    ax.set_ylabel("Drawdown")
    ax.yaxis.set_major_formatter(mtick.PercentFormatter(xmax=1))
    ax.tick_params(axis="x", rotation=45)

    # Annotation max drawdown
    mdd = dd.min()
    ax.axhline(mdd, linestyle="--")
    ax.text(dd.index[-1], mdd, f" Max DD: {mdd:.1%}", va="bottom", ha="right")

    fig.tight_layout()
    return fig

def plot_efficient_frontier(efficient_frontier_vol, efficient_frontier_ret):
    """Return a plot with the points on the efficient frontier"""
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(efficient_frontier_vol, efficient_frontier_ret, marker="o", linewidth=1)
    ax.set_title("Efficient Frontier")
    ax.set_xlabel("Volatility (annualized)")
    ax.set_ylabel("Return (annualized)")
    ax.xaxis.set_major_formatter(mtick.PercentFormatter(xmax=1))
    ax.yaxis.set_major_formatter(mtick.PercentFormatter(xmax=1))
    fig.tight_layout()
    return fig
