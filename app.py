import numpy as np
import streamlit as st
from sklearn.covariance import LedoitWolf
from src.data import download_data, load_sp500_table_requests
from src.analytics import calculate_return, calculate_correlation
from src.optimization import equal_weight, min_variance_portfolio, max_sharpe_ratio, efficient_frontier
from src.plotting import (
    plot_weights,
    plot_correlation,
    plot_portfolio_performance,
    plot_drawdown,
    plot_efficient_frontier,
)
from src.risk import calculate_portfolio_vol, calculate_daily_returns, calculate_drawdown

st.set_page_config(page_title="Portfolio Allocation & Optimization", layout="wide")

def sidebar_inputs():
    """Sidebar inputs for the Portfolio Allocation & Optimization App."""

    st.sidebar.header("Portfolio Settings")

    # --- Universe / tickers ---
    sp500_df = load_sp500_table_requests()

    sp500_df["Display"] = (
        sp500_df["YahooSymbol"]
        + " — "
        + sp500_df["Security"]
        + " (" + sp500_df["GICS Sector"] + ")"
    )

    all_options = sp500_df["Display"].tolist()

    # --- Session state for persistent selection ---
    if "selected_assets" not in st.session_state:
        default_tickers = ["AAPL", "MSFT", "AMZN", "GOOGL"]
        default_options = [
            o for o in all_options if o.split(" — ")[0] in default_tickers
        ]
        st.session_state.selected_assets = default_options

    # --- Buttons (Select all / Clear) ---
    col1, col2 = st.sidebar.columns(2)

    with col1:
        if st.button("Select all", use_container_width=True):
            st.session_state.selected_assets = all_options

    with col2:
        if st.button("Clear", use_container_width=True):
            st.session_state.selected_assets = []

    selected = st.sidebar.multiselect(
        "Select S&P 500 assets",
        options=all_options,
        default=st.session_state.selected_assets,
        placeholder="Type to search (Apple, Microsoft, ...)"
    )

    st.session_state.selected_assets = selected
    tickers = [s.split(" — ")[0] for s in selected]

    st.sidebar.divider()

    # --- Data settings ---
    period = st.sidebar.selectbox(
        "Lookback period",
        options=["1y", "3y", "5y"],
        index=1
    )

    return_type = st.sidebar.selectbox(
        "Return type",
        options=["simple", "log"],
        index=0,
        help="Simple returns are common for portfolio optimization. Log returns are sometimes used for modeling."
    )

    annualization = st.sidebar.selectbox(
        "Annualization factor",
        options=[252, 52, 12],
        index=0,
        help="Use 252 for daily data, 52 for weekly, 12 for monthly."
    )

    st.sidebar.divider()

    # --- Optimization settings ---
    method = st.sidebar.radio(
        "Allocation method",
        options=["Equal Weight", "Minimum Variance", "Maximum Sharpe"],
        index=1
    )

    rf = st.sidebar.number_input(
        "Risk-free rate (annual)",
        value=0.02,
        step=0.005,
        format="%.3f",
        help="Used for Sharpe ratio / max Sharpe optimization (annual). Example: 0.02 = 2%."
    )

    use_weight_cap = st.sidebar.checkbox(
        "Apply max weight cap",
        value=True,
        help="Helps avoid overly concentrated portfolios (common in practice)."
    )

    max_weight = None
    if use_weight_cap:
        max_weight = st.sidebar.slider(
            "Max weight per asset",
            min_value=0.00,
            max_value=1.00,
            value=0.40,
            step=0.01
        )

    st.sidebar.divider()

    compute_frontier = st.sidebar.checkbox(
        "Compute efficient frontier (slow)",
        value=False,
        help="This runs a CVXPY loop and may be slow for large portfolios."
    )

    st.sidebar.divider()

    run_button = st.sidebar.button("Run optimization")

    return {
        "tickers": tickers,
        "period": period,
        "return_type": return_type,
        "annualization": annualization,
        "method": method,
        "rf": rf,
        "compute_frontier": compute_frontier,
        "max_weight": max_weight,
        "run": run_button,
    }

params = sidebar_inputs()

@st.cache_data(ttl=3600)
def download_prices_cached(tickers, period):
    """Cache wrapper for prices download."""
    return download_data(list(tickers), period)


def covariance_ledoit_wolf(returns_df, annualization: int = 252) -> np.ndarray:
    """Robust PSD covariance via Ledoit-Wolf shrinkage."""
    clean = returns_df.dropna(how="any")
    X = clean.values
    lw = LedoitWolf().fit(X)
    cov = lw.covariance_ * annualization
    cov = 0.5 * (cov + cov.T)
    return cov

st.title("Portfolio Allocation & Optimization App")
st.caption("Educational project. Not investment advice.")

# --- Basic validation ---
if len(params["tickers"]) < 2:
    st.warning("Please select at least 2 assets.")
    st.stop()

if not params["run"]:
    st.info("Select your settings in the sidebar, then click 'Run optimization'.")
    st.stop()

# --- Performance guardrail ---
if params["method"] in ["Minimum Variance", "Maximum Sharpe"] and len(params["tickers"]) > 150:
    st.warning(
        "You selected a very large number of assets. CVXPY optimization can be slow/unstable. "
        "Try <= 150 assets for a smoother experience."
    )

with st.spinner("Downloading data and running optimisation..."):
    prices = download_prices_cached(tuple(params["tickers"]), params["period"])

    # Returns
    returns = calculate_return(prices, params["return_type"])
    returns_clean = returns.dropna(how="any")

    # Mean returns (annualised)
    mu = returns_clean.mean(axis=0).values * params["annualization"]  # (N,)

    # Covariance (annualised)
    cov = covariance_ledoit_wolf(returns_clean, annualization=params["annualization"])

    # Correlation matrix for heatmap
    correlations = calculate_correlation(returns_clean)

    # Portfolio weights
    if params["method"] == "Equal Weight":
        weights = equal_weight(params["tickers"])
    elif params["method"] == "Minimum Variance":
        weights = min_variance_portfolio(cov, max_weight=params["max_weight"])
    else:
        # If your max_sharpe_ratio supports rf, you can pass it here.
        weights = max_sharpe_ratio(mu, cov, rf=params["rf"], max_weight=params["max_weight"])

    weights = np.asarray(weights).reshape(-1)

    # Portfolio analytics
    portfolio_volatility = calculate_portfolio_vol(weights, cov)  # annualised vol if cov annualised

    daily_returns = calculate_daily_returns(returns_clean, weights)
    daily_returns = daily_returns.dropna() if hasattr(daily_returns, "dropna") else daily_returns

    # Annualised return + Sharpe
    ann_return = float(np.mean(daily_returns) * params["annualization"])
    ann_vol = float(np.std(daily_returns, ddof=1) * np.sqrt(params["annualization"]))
    sharpe_ratio = (ann_return - params["rf"]) / ann_vol if ann_vol > 0 else np.nan

    # Drawdown
    max_dd = calculate_drawdown(daily_returns)

# --- KPI ---
c1, c2, c3, c4 = st.columns(4)
c1.metric("Annual Return", f"{ann_return:.2%}")
c2.metric("Annual Volatility", f"{portfolio_volatility:.2%}")
c3.metric("Sharpe Ratio", f"{sharpe_ratio:.2f}" if np.isfinite(sharpe_ratio) else "N/A")
c4.metric("Max Drawdown", f"{max_dd:.2%}")

st.divider()

fig_weights = plot_weights(params["tickers"], weights)
fig_perf = plot_portfolio_performance(daily_returns)
fig_dd = plot_drawdown(daily_returns)
fig_corr = plot_correlation(correlations)

fig_frontier = None
if params["compute_frontier"]:
    try:
        ef_v, ef_r, _ = efficient_frontier(mu, cov, 60)
        fig_frontier = plot_efficient_frontier(ef_v, ef_r)
    except Exception as e:
        st.warning("Efficient frontier could not be computed (covariance may be unstable or too many assets).")
        st.exception(e)

# --- Tabs ---
tab1, tab2, tab3 = st.tabs(["Overview", "Risk & Correlation", "Efficient Frontier"])

with tab1:
    st.subheader("Portfolio Weights")
    st.pyplot(fig_weights)

    st.subheader("Performance")
    st.pyplot(fig_perf)

    st.subheader("Drawdown")
    st.pyplot(fig_dd)

with tab2:
    st.subheader("Correlation Heatmap")
    st.pyplot(fig_corr)

with tab3:
    st.subheader("Efficient Frontier")
    if fig_frontier is None:
        st.info("Enable 'Compute efficient frontier' in the sidebar to display it.")
    else:
        st.pyplot(fig_frontier)

st.success("Finished")
