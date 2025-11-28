"""
Data Fetcher for Stock Price Data
Downloads OHLCV data from Yahoo Finance for stock price forecasting.
"""

import os
from typing import List, Optional, Tuple

import numpy as np
import pandas as pd
import yfinance as yf
import matplotlib.pyplot as plt
import config

# Column names used for labels and future prices
LABEL_COLUMN: str = "Label"
FUTURE_CLOSE_COLUMN: str = "Future_Close"

class StockDataFetcher:
    """Fetches stock data from Yahoo Finance and prepares it for modeling."""

    def __init__(
        self,
        cache_dir: str,
    ):
        """Initialize the data fetcher.

        Args:
            cache_dir: Directory to cache downloaded data.
        """
        self.cache_dir = cache_dir
        os.makedirs(cache_dir, exist_ok=True)

    def _fetch_stock_data(
        self,
        ticker: str,
        start_date: str,
        end_date: str,
        use_cache: bool = True,
    ) -> pd.DataFrame | None:
        """Fetch OHLCV data for a single stock.

        Args:
            ticker: Stock ticker symbol (e.g., 'AAPL').
            start_date: Start date in 'YYYY-MM-DD' format.
            end_date: End date in 'YYYY-MM-DD' format.
            use_cache: Whether to use cached data if available.

        Returns:
            DataFrame with columns: Open, High, Low, Close, Volume, or None on error.
        """
        cache_file = os.path.join(self.cache_dir, f"{ticker}_{start_date}_{end_date}.csv")

        # Check cache
        if use_cache and os.path.exists(cache_file):
            print(f"Loading {ticker} from cache...")
            df = pd.read_csv(cache_file, index_col=0, parse_dates=True)
            # Ensure timezone-naive DatetimeIndex (handle tz-aware from yfinance/cache)
            df.index = pd.to_datetime(df.index, utc=True).tz_localize(None)
            return df

        # Download from Yahoo Finance
        print(f"Downloading {ticker} from Yahoo Finance...")
        try:
            stock = yf.Ticker(ticker)
            df = stock.history(start=start_date, end=end_date)

            if df.empty:
                print(f"[33mNo data found for {ticker} - skipping[0m")
                return None

            # Keep only OHLCV columns
            df = df[["Open", "High", "Low", "Close", "Volume"]]

            # Ensure timezone-naive DatetimeIndex (handle tz-aware from yfinance/cache)
            df.index = pd.to_datetime(df.index, utc=True).tz_localize(None)

            # Save to cache
            df.to_csv(cache_file)
            print(f"Downloaded {len(df)} days of data for {ticker}")

            return df

        except Exception as e:  # noqa: BLE001
            print(f"[33mError downloading {ticker}: {e}[0m")
            return None

    def _fetch_multiple_stocks(
        self,
        tickers: List[str],
        start_date: str,
        end_date: str,
        use_cache: bool = True,
    ) -> dict:
        """Fetch raw OHLCV data for multiple stocks.

        Args:
            tickers: List of stock ticker symbols.
            start_date: Start date in 'YYYY-MM-DD' format.
            end_date: End date in 'YYYY-MM-DD' format.
            use_cache: Whether to use cached data if available.

        Returns:
            Dictionary mapping ticker -> DataFrame (or None on failure).
        """
        data: dict[str, pd.DataFrame | None] = {}

        for ticker in tickers:
            try:
                df = self._fetch_stock_data(ticker, start_date, end_date, use_cache)
                data[ticker] = df
            except Exception as e:  # noqa: BLE001
                print(f"Skipping {ticker} due to error: {e}")
                continue

        return data

    def _fetch_and_clean_stock_data(
        self,
        tickers: List[str],
        start_date: str,
        end_date: str,
        use_cache: bool = True,
    ) -> dict[str, pd.DataFrame]:
        """Fetch, clean, and label stock data for modeling.
        """
        raw_data = self._fetch_multiple_stocks(tickers, start_date, end_date, use_cache)

        all_data: dict[str, pd.DataFrame] = {}
        for stock, data in raw_data.items():
            if data is None or data.empty:
                continue

            print(f"Processing cleaned data for {stock}...")

            # Reset index to make 'Date' a column (index is already tz-naive)
            data = data.copy()
            data.reset_index(inplace=True)

            # Ensure numeric values
            for col in ["Open", "High", "Low", "Close", "Volume"]:
                data[col] = pd.to_numeric(data[col], errors="coerce")

            all_data[stock] = data

        print("Stock data cleaned")

        return all_data

    def get_sp500_tickers(self, limit: Optional[int] = None) -> List[str]:
        """Get list of S&P 500 ticker symbols.

        Args:
            limit: Maximum number of tickers to return (None for all).

        Returns:
            List of ticker symbols.
        """
        url = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"
        tables = pd.read_html(url)
        sp500_table = tables[0]
        tickers = sp500_table["Symbol"].tolist()

        # Clean tickers (remove dots, etc.)
        tickers = [ticker.replace(".", "-") for ticker in tickers]

        if limit:
            tickers = tickers[:limit]

        print(f"Found {len(tickers)} S&P 500 tickers")
        return tickers

    def _build_samples(self, stock_df):
        """Build (data_slice, label, future_return, date) samples for the forecast horizon.

        This keeps things simple: we return small DataFrame windows plus label and
        future return; image generation happens later in dataset_keras.prepare_data.

        Moving averages are calculated here for the entire dataframe to avoid
        recalculating them for each window in the draw function.

        Returns:
            Dictionary mapping ticker -> list of samples for that ticker.
            Each sample is a tuple: (window, label, future_return, date, current_close, daily_return, ticker)
        """
        samples_by_ticker: dict[str, list[tuple]] = {}

        window_size = config.N_DAYS
        future_horizon = config.FORECAST_HORIZON  # days ahead
        label_col = LABEL_COLUMN
        future_col = FUTURE_CLOSE_COLUMN

        for stock, df in stock_df.items():
            df = df.sort_values(by="Date")

            # Calculate moving average for the entire dataframe once
            # This is more efficient than calculating it for each window
            df["MA"] = df["Close"].rolling(window=window_size, min_periods=1).mean()

            # Bollinger Bands (middle +/- 2 std) over the same window
            bb_mid = df["Close"].rolling(window=window_size, min_periods=1).mean()
            bb_std = df["Close"].rolling(window=window_size, min_periods=1).std(ddof=0)
            df["BB_UPPER"] = bb_mid + 2.0 * bb_std
            df["BB_LOWER"] = bb_mid - 2.0 * bb_std

            # VWAP based on typical price
            typical_price = (df["High"] + df["Low"] + df["Close"]) / 3.0
            vwap_num = (typical_price * df["Volume"]).cumsum()
            vwap_den = df["Volume"].cumsum().replace(0, pd.NA)
            df["VWAP"] = (vwap_num / vwap_den).ffill()

            # OBV (On-Balance Volume)
            # OBV increases by volume on up days, decreases by volume on down days
            obv = [0]
            for i in range(1, len(df)):
                if df["Close"].iloc[i] > df["Close"].iloc[i - 1]:
                    obv.append(obv[-1] + df["Volume"].iloc[i])
                elif df["Close"].iloc[i] < df["Close"].iloc[i - 1]:
                    obv.append(obv[-1] - df["Volume"].iloc[i])
                else:
                    obv.append(obv[-1])
            df["OBV"] = obv

            # RSI (Relative Strength Index)
            rsi_period = getattr(config, "RSI_PERIOD", 14)
            delta = df["Close"].diff()
            gain = (delta.where(delta > 0, 0)).rolling(window=rsi_period, min_periods=1).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(window=rsi_period, min_periods=1).mean()
            rs = gain / loss.replace(0, pd.NA)
            df["RSI"] = (100 - (100 / (1 + rs))).fillna(50).astype(float)

            # ADX (Average Directional Index)
            adx_period = getattr(config, "ADX_PERIOD", 14)

            # Calculate True Range (TR)
            high_low = df["High"] - df["Low"]
            high_close = np.abs(df["High"] - df["Close"].shift(1))
            low_close = np.abs(df["Low"] - df["Close"].shift(1))
            tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)

            # Calculate Directional Movement (+DM and -DM)
            high_diff = df["High"].diff()
            low_diff = -df["Low"].diff()

            plus_dm = high_diff.where((high_diff > low_diff) & (high_diff > 0), 0)
            minus_dm = low_diff.where((low_diff > high_diff) & (low_diff > 0), 0)

            # Smooth TR, +DM, -DM using rolling mean
            atr = tr.rolling(window=adx_period, min_periods=1).mean()
            plus_di = 100 * (plus_dm.rolling(window=adx_period, min_periods=1).mean() / atr)
            minus_di = 100 * (minus_dm.rolling(window=adx_period, min_periods=1).mean() / atr)

            # Calculate DX and ADX
            di_sum = plus_di + minus_di
            di_sum = di_sum.replace(0, np.nan)  # Replace 0 with NaN to avoid division by zero
            dx = 100 * np.abs(plus_di - minus_di) / di_sum
            df["ADX"] = dx.rolling(window=adx_period, min_periods=1).mean()
            df["ADX"] = df["ADX"].fillna(25)  # Fill NaN with neutral value (25 = weak trend)

            # Compute future close and binary label
            df[future_col] = df["Close"].shift(-future_horizon)
            df[label_col] = (df[future_col] > df["Close"]).astype(int)

            # Drop rows with NaNs introduced by shifting
            df.dropna(inplace=True)

            # Plot the cleaned DataFrame with all indicators if requested
            if getattr(config, "SHOW_BUILD_SAMPLES_DF", False):
                cols = [
                    c
                    for c in ["Close", "MA", "BB_UPPER", "BB_LOWER", "VWAP"]
                    if c in df.columns
                ]
                if cols and "Date" in df.columns:
                    ax = df.plot(x="Date", y=cols, figsize=(12, 6), title=str(stock))
                    ax.set_ylabel("Price / Indicator")
                    ax.grid(True, alpha=0.3)
                    plt.show()

            # Initialize list for this ticker
            ticker_samples = []

            for i in range(len(df) - window_size):
                window = df.iloc[i : i + window_size].copy()

                # Use the last row in the window as the "current" date
                last_row = window.iloc[-1]
                label = int(last_row[label_col])

                current_close = last_row["Close"]
                future_close = last_row[future_col]
                future_return = float((future_close - current_close) / current_close)

                # Also compute next-day return for backtesting
                # (to avoid compounding overlapping multi-day returns)
                if i + window_size < len(df) - 1:
                    next_close = df.iloc[i + window_size + 1]["Close"]
                    daily_return = float((next_close - current_close) / current_close)
                else:
                    daily_return = 0.0

                date = last_row["Date"]
                current_close_val = float(current_close)

                # Add ticker to the sample tuple: (window, label, future_return, date, current_close, daily_return, ticker)
                ticker_samples.append((window, label, future_return, date, current_close_val, daily_return, stock))

            # Store samples for this ticker
            samples_by_ticker[stock] = ticker_samples
            print(f"  Built {len(ticker_samples)} samples for {stock}")

        return samples_by_ticker

    def fetch_stocks_build_samples(
        self,
        tickers: List[str],
        start_date: str,
        end_date: str,
        use_cache: bool = True,
    ) -> list[tuple]:
        """Fetch stocks, clean them, and build (window, label, future_return, date) samples."""
        stock_df = self._fetch_and_clean_stock_data(
            tickers=tickers,
            start_date=start_date,
            end_date=end_date,
            use_cache=use_cache,
        )

        return self._build_samples(stock_df), stock_df
