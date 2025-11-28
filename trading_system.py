"""
Trading System for DQN Stock Market Prediction.
Uses trained DQN model for backtesting with metrics, reports, and plots.
"""
import os
import random

import numpy as np
import tensorflow as tf

from data_fetcher import StockDataFetcher
from prepare_data import prepare_data
import Source_TF2.convNN as CNN

tf.compat.v1.disable_eager_execution()

# DQN Model parameters
W = 32
FSize = 5
PSize = 2
PStride = 2
NumAction = 3

TRADING_DAYS_PER_YEAR = 252


def fetch_and_prepare(tickers, start_date, end_date, window_size=32, image_size=32):
    """Fetch stock data and convert to DQN format.

    Returns:
        XData: List[company][day] of image matrices
        YData: List[company][day] of future returns
    """
    print(f"Fetching data for {len(tickers)} stocks from {start_date} to {end_date}...")

    fetcher = StockDataFetcher(cache_dir="data_cache")
    samples_by_ticker, _ = fetcher.fetch_stocks_build_samples(
        tickers=tickers,
        start_date=start_date,
        end_date=end_date,
        use_cache=True,
    )

    X, y, returns, dates, closes, daily_returns, tickers_arr = prepare_data(
        samples_by_ticker,
        n_days=window_size,
        image_height=image_size,
        image_width=image_size,
    )

    unique_tickers = list(dict.fromkeys(tickers_arr))
    XData = []
    YData = []

    for ticker in unique_tickers:
        mask = tickers_arr == ticker
        ticker_X = X[mask]
        ticker_returns = returns[mask]

        ticker_images = []
        for img in ticker_X:
            img_array = np.array(img, dtype=np.float32)
            # Remove channel dimension if present: (H, W, 1) -> (H, W)
            if img_array.ndim == 3:
                img_array = img_array[:, :, 0]
            ticker_images.append(img_array)

        XData.append(ticker_images)
        YData.append(ticker_returns.tolist())
        print(f"  Organized {len(ticker_images)} samples for {ticker}")

    print(f"Total: {len(XData)} companies, {len(XData[0]) if XData else 0} days each")
    return XData, YData





def _compute_performance_metrics(returns: np.ndarray) -> dict:
    """Compute basic performance metrics from a series of daily returns."""
    returns = np.asarray(returns, dtype=np.float64)
    if returns.size == 0:
        return {
            "ann_return": 0.0,
            "ann_sharpe": 0.0,
            "max_drawdown": 0.0,
            "total_return": 0.0,
            "equity_final": 1.0,
            "n_periods": 0,
        }

    equity = np.cumprod(1.0 + returns)
    total_return = float(equity[-1] - 1.0)

    # Assume returns are daily (horizon_days is now always 1 after division)
    total_days = returns.size
    years = total_days / TRADING_DAYS_PER_YEAR
    if years <= 0:
        ann_return = 0.0
    else:
        ann_return = float(equity[-1] ** (1.0 / years) - 1.0)

    mean_r = float(np.mean(returns))
    std_r = float(np.std(returns, ddof=1))
    if std_r > 1e-8:
        ann_sharpe = (mean_r / std_r) * np.sqrt(TRADING_DAYS_PER_YEAR)
    else:
        ann_sharpe = 0.0

    peak = np.maximum.accumulate(equity)
    drawdown = equity / peak - 1.0
    max_drawdown = float(drawdown.min())

    return {
        "ann_return": ann_return,
        "ann_sharpe": float(ann_sharpe),
        "max_drawdown": max_drawdown,
        "total_return": total_return,
        "equity_final": float(equity[-1]),
        "n_periods": int(returns.size),
    }


def run_trading_system(
    tickers=None,
    start_date="2019-01-01",
    end_date="2023-12-31",
    checkpoint_path="Source_TF2/DeepQ",
):
    """Run backtesting using trained DQN model.

    Args:
        tickers: List of stock tickers to test
        start_date: Start date for testing period
        end_date: End date for testing period
        checkpoint_path: Path to trained DQN model checkpoint
    """
    if tickers is None:
        tickers = ["AAPL", "MSFT", "GOOGL", "AMZN", "META"]

    _seed_everything()

    print("=" * 70)
    print("DQN Trading System - Backtesting")
    print("=" * 70)
    print(f"\nTickers: {tickers}")
    print(f"Period: {start_date} to {end_date}")
    print(f"Model: {checkpoint_path}")

    # Fetch data
    print("\nFetching stock data...")
    XData, YData = fetch_and_prepare(tickers, start_date, end_date, W, W)

    if not XData:
        print("No data available for testing.")
        return None

    n_stocks = len(XData)
    n_days = len(XData[0])
    print(f"\nStocks: {n_stocks}, Days: {n_days}")

    # Load model
    print("\nLoading DQN model...")
    tf.compat.v1.reset_default_graph()
    state = tf.compat.v1.placeholder(tf.float32, [None, W, W])
    isTrain = tf.compat.v1.placeholder(tf.bool, [])
    cnn = CNN.ConstructCNN(W, W, FSize, PSize, PStride, NumAction)
    rho_eta = cnn.QValue(state, isTrain)

    saver = tf.compat.v1.train.Saver()
    sess = tf.compat.v1.Session()
    saver.restore(sess, checkpoint_path)

    print("\nRunning predictions...")

    # Process day by day, averaging across stocks
    daily_strategy_returns = []
    daily_buyhold_returns = []
    daily_positions = []
    action_counts = {0: 0, 1: 0, 2: 0}

    for day in range(n_days):
        day_strat_rets = []
        day_bh_rets = []
        day_positions = []

        for stock_idx in range(n_stocks):
            img = XData[stock_idx][day]
            ret = YData[stock_idx][day]

            # Get prediction
            img_input = np.array(img).reshape(1, W, W)
            rho, eta = sess.run(rho_eta, feed_dict={state: img_input, isTrain: False})
            q_values = rho
            action = np.argmax(q_values[0])
            action_counts[action] = action_counts.get(action, 0) + 1

            # Map action to position
            if action == 2:
                pos = 1.0
            elif action == 0:
                pos = -1.0
            else:
                pos = 0.0

            day_strat_rets.append(pos * ret)
            day_bh_rets.append(ret)
            day_positions.append(pos)

        # Average across stocks for this day (equal-weighted portfolio)
        daily_strategy_returns.append(np.mean(day_strat_rets))
        daily_buyhold_returns.append(np.mean(day_bh_rets))
        daily_positions.append(np.mean(day_positions))

    strategy_returns = daily_strategy_returns
    buyhold_returns = daily_buyhold_returns
    positions = daily_positions
    n_samples = n_days

    print(f"\nAction distribution: Sell={action_counts[0]}, Hold={action_counts[1]}, Buy={action_counts[2]}")

    # Compute metrics
    strat_metrics = _compute_performance_metrics(np.array(strategy_returns))
    bh_metrics = _compute_performance_metrics(np.array(buyhold_returns))

    # Count trades
    n_trades = sum(1 for i in range(1, len(positions)) if positions[i] != positions[i-1])

    # Print report
    print_report(strat_metrics, bh_metrics, n_trades, n_samples)

    sess.close()

    return {
        "strategy": strat_metrics,
        "buy_hold": bh_metrics,
        "n_trades": n_trades,
        "n_samples": n_samples,
    }


def print_report(strat_metrics, bh_metrics, n_trades, n_days):
    """Print trading report."""
    print("\n" + "=" * 60)
    print("TRADING REPORT")
    print("=" * 60)
    print(f"\nPeriod: {n_days} trading days")
    print(f"Total Trades: {n_trades}")
    print(f"\n{'Metric':<25} {'Strategy':>15} {'Buy & Hold':>15}")
    print("-" * 60)
    print(f"{'Total Return':<25} {strat_metrics['total_return']*100:>14.2f}% {bh_metrics['total_return']*100:>14.2f}%")
    print(f"{'Annual Return':<25} {strat_metrics['ann_return']*100:>14.2f}% {bh_metrics['ann_return']*100:>14.2f}%")
    print(f"{'Sharpe Ratio':<25} {strat_metrics['ann_sharpe']:>15.3f} {bh_metrics['ann_sharpe']:>15.3f}")
    print(f"{'Max Drawdown':<25} {strat_metrics['max_drawdown']*100:>14.2f}% {bh_metrics['max_drawdown']*100:>14.2f}%")
    print("=" * 60)


def _seed_everything():
    """Set random seeds for reproducibility."""
    seed = 42
    random.seed(seed)
    np.random.seed(seed)
    tf.compat.v1.set_random_seed(seed)
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'


if __name__ == "__main__":
    run_trading_system()
