import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter
import matplotlib.ticker as mtick

def plot_weights(stocks, weights):
    """Plot a bar chart of portfolio allocation weights.

    Parameters
    ----------
    stocks : list[str]
        Ticker symbols for the x-axis labels.
    weights : np.ndarray
        (N,) weight vector.

    Returns
    -------
    matplotlib.figure.Figure
    """
    fig, ax =  plt.subplots(figsize=(6, 4))
    ax.bar(stocks, weights)
    ax.set_title("Portfolio weights")
    ax.set_xlabel("Stock tickers")
    ax.set_ylabel("Weights")
    ax.yaxis.set_major_formatter(mtick.PercentFormatter(xmax=1))
    fig.tight_layout()
    return fig

def plot_correlation(correlations: pd.DataFrame):
    """Plot a heatmap of the pairwise asset correlation matrix.

    Parameters
    ----------
    correlations : pd.DataFrame
        N×N correlation matrix (values between -1 and 1).

    Returns
    -------
    matplotlib.figure.Figure
    """
    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(correlations.values, vmin=-1, vmax=1, cmap="RdBu_r")
    ax.set_title("Correlation matrix")

    ax.set_xticks(range(len(correlations.columns)))
    ax.set_yticks(range(len(correlations.index)))
    ax.set_xticklabels(correlations.columns, rotation=45, ha="right")
    ax.set_yticklabels(correlations.index)

    # Display correlation values in each cell
    for i in range(len(correlations.index)):
        for j in range(len(correlations.columns)):
            val = correlations.values[i, j]
            color = "white" if abs(val) > 0.6 else "black"
            ax.text(j, i, f"{val:.2f}", ha="center", va="center", color=color, fontsize=7)

    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label("Correlation")

    fig.tight_layout()
    return fig

def plot_portfolio_performance(daily_returns):
    """Plot the cumulative wealth index (base 1.0) of the portfolio.

    Parameters
    ----------
    daily_returns : pd.Series
        Daily portfolio returns.

    Returns
    -------
    matplotlib.figure.Figure
    """
    wealth = (1 + daily_returns).cumprod()
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(wealth.index, wealth.values)
    ax.set_title("Portfolio value (base 1.0)")
    ax.set_xlabel("Date")
    ax.set_ylabel("Value")
    ax.tick_params(axis="x", rotation=45)
    fig.tight_layout()
    return fig

def plot_drawdown(dd: pd.Series):
    """Plot the underwater (drawdown) curve with max drawdown annotation.

    Parameters
    ----------
    dd : pd.Series
        Pre-computed drawdown series (values <= 0).

    Returns
    -------
    matplotlib.figure.Figure
    """
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(dd.index, dd.values)
    ax.set_title("Drawdown")
    ax.set_xlabel("Date")
    ax.set_ylabel("Drawdown")
    ax.yaxis.set_major_formatter(mtick.PercentFormatter(xmax=1))
    ax.tick_params(axis="x", rotation=45)

    mdd = dd.min()
    ax.axhline(mdd, linestyle="--")
    ax.text(dd.index[-1], mdd, f" Max DD: {mdd:.1%}", va="bottom", ha="right")

    fig.tight_layout()
    return fig

def plot_efficient_frontier(efficient_frontier_vol, efficient_frontier_ret):
    """Plot the mean-variance efficient frontier.

    Parameters
    ----------
    efficient_frontier_vol : np.ndarray
        Annualized volatilities for each frontier point.
    efficient_frontier_ret : np.ndarray
        Annualized returns for each frontier point.

    Returns
    -------
    matplotlib.figure.Figure
    """
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(efficient_frontier_vol, efficient_frontier_ret, marker="o", linewidth=1)
    ax.set_title("Efficient Frontier")
    ax.set_xlabel("Volatility (annualized)")
    ax.set_ylabel("Return (annualized)")
    ax.xaxis.set_major_formatter(mtick.PercentFormatter(xmax=1))
    ax.yaxis.set_major_formatter(mtick.PercentFormatter(xmax=1))
    fig.tight_layout()
    return fig
