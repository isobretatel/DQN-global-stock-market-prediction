from typing import List, Tuple

import os
import hashlib
import pickle
import numpy as np
import pandas as pd
from PIL import Image
import matplotlib.pyplot as plt

from draw_ohlc import DrawOHLC
import config

# Cache directory for images
CACHE_DIR = os.path.join(os.path.dirname(__file__), "__pycache__", "image_cache")
os.makedirs(CACHE_DIR, exist_ok=True)


def _get_cache_key(ticker: str, date: pd.Timestamp, n_days: int, image_height: int, image_width: int) -> str:
    """Generate a cache key for an image based on ticker, date, and configuration."""
    # Include ALL parameters that affect image generation
    rsi_period = getattr(config, "RSI_PERIOD", 14)
    adx_period = getattr(config, "ADX_PERIOD", 14)
    config_str = (
        f"{n_days}_{image_height}_{image_width}_"
        f"{config.HAS_VOLUME_BAR}_{config.HAS_MOVING_AVERAGE}_"
        f"{config.HAS_BOLLINGER_BANDS}_{config.HAS_VWAP}_{config.HAS_OBV}_"
        f"{config.HAS_RSI}_{rsi_period}_{config.HAS_ADX}_{adx_period}"
    )
    key_str = f"{ticker}_{date.strftime('%Y%m%d')}_{config_str}"
    return hashlib.md5(key_str.encode()).hexdigest()


def _load_cached_image(cache_key: str) -> np.ndarray:
    """Load a cached image if it exists."""
    cache_path = os.path.join(CACHE_DIR, f"{cache_key}.npy")
    if os.path.exists(cache_path):
        return np.load(cache_path)
    return None


def _save_cached_image(cache_key: str, image: np.ndarray):
    """Save an image to cache."""
    cache_path = os.path.join(CACHE_DIR, f"{cache_key}.npy")
    np.save(cache_path, image)


def _show_image(img, title="Sample image"):
    """Show a single image."""
    # Handle both 2D (H, W) and 3D (H, W, 1) arrays
    if img.ndim == 3:
        img_display = img[:, :, 0]  # Remove channel dimension
    else:
        img_display = img

    h, w = img_display.shape
    dpi = plt.rcParams.get("figure.dpi", 100)
    plt.figure(figsize=(w / dpi, h / dpi), dpi=dpi)
    plt.imshow(img_display, cmap="gray", interpolation="none")
    plt.title(f"{title} (shape: {img.shape})")
    plt.axis("off")
    plt.show()


def _show_multiple_images(images, dates, tickers, labels, title_prefix="Image"):
    """Show multiple images in a grid."""
    n_images = len(images)
    if n_images == 0:
        return

    # Create grid: 2 rows x 5 columns for 10 images
    n_cols = min(5, n_images)
    n_rows = (n_images + n_cols - 1) // n_cols

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(n_cols * 3, n_rows * 2))
    if n_images == 1:
        axes = np.array([axes])
    axes = axes.flatten()

    for idx in range(n_images):
        img = images[idx]
        # Handle both 2D (H, W) and 3D (H, W, 1) arrays
        if img.ndim == 3:
            img_display = img[:, :, 0]
        else:
            img_display = img

        axes[idx].imshow(img_display, cmap="gray", interpolation="none")
        label_name = "UP" if labels[idx] == 1 else "DOWN"
        axes[idx].set_title(f"{title_prefix} {idx}\n{dates[idx]}\n{tickers[idx]} - {label_name}", fontsize=8)
        axes[idx].axis("off")

    # Hide unused subplots
    for idx in range(n_images, len(axes)):
        axes[idx].axis("off")

    plt.tight_layout()
    plt.show()


def _load_or_generate_image(cache_hits, cache_misses, n_days, image_height, image_width, ticker, date, sample_data):
    # Try to load from cache
    cache_key = _get_cache_key(ticker, date, n_days, image_height, image_width)
    img = _load_cached_image(cache_key)

    if img is not None:
        cache_hits += 1
    else:
        cache_misses += 1
        
        window = sample_data['data_slice'].copy()
        drawer = DrawOHLC(
            window,
            time_frame=n_days,
            has_volume_bar=config.HAS_VOLUME_BAR,
            has_moving_average=config.HAS_MOVING_AVERAGE,
            has_bollinger_bands=config.HAS_BOLLINGER_BANDS,
            has_vwap=config.HAS_VWAP,
            has_obv=config.HAS_OBV,
            has_rsi=config.HAS_RSI,
            has_adx=config.HAS_ADX)
        img = drawer.draw_image()
        # Save to cache
        _save_cached_image(cache_key, img)  # Generate image if not in cache

    return img, cache_hits, cache_misses


def _print_results(y, dates, tickers):
    # Print summary of date distribution
    unique_dates = np.unique(dates)
    print(f"\nTotal samples prepared: {len(y)}")
    print(f"Date range: {unique_dates[0]} to {unique_dates[-1]}")
    print(f"Number of unique dates: {len(unique_dates)}")
    print(f"Number of unique tickers: {len(np.unique(tickers))}")

# Print labels distribution (class balance)
    unique_labels, label_counts = np.unique(y, return_counts=True)
    print(f"\nLabel distribution (class balance):")
    for labels, count in zip(unique_labels, label_counts):
        percentage = count / len(y) * 100
        label_name = "DOWN (0)" if labels == 0 else "UP (1)"
        print(f"  {label_name}: {count} samples ({percentage:.1f}%)")
    
# Calculate imbalance ratio
    if len(unique_labels) == 2:
        imbalance_ratio = max(label_counts) / min(label_counts)
        print(f"  Imbalance ratio: {imbalance_ratio:.2f}:1")

# Show first few samples to verify sorting
    print(f"\nFirst 10 samples (date, ticker):")
    for i in range(min(10, len(dates))):
        print(f"  [{i}] {dates[i]} - {tickers[i]}")


def prepare_data(
    samples,
    n_days: int,
    image_height: int,
    image_width: int,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Generate images from samples and organize them by date/index across all tickers.

    This function processes samples organized by ticker and reorders them so that:
    - All tickers' samples for the same date/index are grouped together
    - Samples are sorted by date first, then by ticker
    - This allows for proper cross-sectional analysis and train/test splitting

    Args:
        samples: Dict mapping ticker -> list of tuples (new format)
                 Each tuple: (window, label, future_return, date, current_close, daily_return, ticker)
        n_days: Window size
        image_height: Image height
        image_width: Image width

    Returns:
        Tuple of (X, y, returns, dates, closes, daily_returns, tickers)
        All arrays are sorted by date first, then by ticker
    """

    if not isinstance(samples, dict):
        raise ValueError("prepare_data requires samples to be organized by ticker (dict format)")

    print(f"\nProcessing samples organized by ticker and sorting by date/index:")

    # First pass: collect all samples with metadata
    all_samples_data = []

    for ticker, ticker_samples in samples.items():
        print(f"  {ticker}: {len(ticker_samples)} samples")

        for sample in ticker_samples:
            # Each sample: (window, label, future_return, date, current_close, daily_return, ticker)
            data_slice, label, future_return, date, current_close, daily_return, ticker_name = sample

            # Store sample with metadata for sorting
            all_samples_data.append({
                'date': pd.Timestamp(date),
                'ticker': ticker_name,
                'data_slice': data_slice,
                'label': label,
                'future_return': future_return,
                'current_close': current_close,
                'daily_return': daily_return,
            })

    # Sort by date first, then by ticker (for consistent ordering within same date)
    print(f"\nSorting {len(all_samples_data)} samples by date and ticker...")
    all_samples_data.sort(key=lambda x: (x['date'], x['ticker']))

    # Second pass: generate images in sorted order (with caching)
    print("Generating images (using cache when available)...")
    images: list[np.ndarray] = []
    labels: list[int] = []
    returns: list[float] = []
    dates: list = []
    closes: list[float] = []
    daily_returns: list[float] = []
    tickers: list[str] = []

    cache_hits = 0
    cache_misses = 0

    for sample_data in all_samples_data:
        ticker = sample_data['ticker']
        date = sample_data['date']

        img, cache_hits, cache_misses = _load_or_generate_image(cache_hits, cache_misses, n_days, image_height, image_width, ticker, date, sample_data)

        images.append(img)
        labels.append(int(sample_data['label']))
        returns.append(float(sample_data['future_return']))
        dates.append(sample_data['date'])
        closes.append(float(sample_data['current_close']))
        daily_returns.append(float(sample_data['daily_return']))
        tickers.append(sample_data['ticker'])

    print(f"Cache statistics: {cache_hits} hits, {cache_misses} misses ({cache_hits/(cache_hits+cache_misses)*100:.1f}% hit rate)")

    # Convert to numpy arrays
    X = np.stack(images, axis=0)
    y = np.asarray(labels, dtype=np.int32)
    returns = np.asarray(returns, dtype=np.float32)
    dates = np.asarray(dates, dtype="datetime64[ns]")
    closes = np.asarray(closes, dtype=np.float32)
    daily_returns = np.asarray(daily_returns, dtype=np.float32)
    tickers = np.asarray(tickers, dtype=str)

    _print_results(y, dates, tickers)

    # Show first 10 and last 10 images if enabled
    if config.SHOW_SAMPLE_IMAGE and len(X) > 0:
        n_samples = len(X)

        # Show first 10 images
        n_first = min(10, n_samples)
        print(f"\nShowing first {n_first} images...")
        _show_multiple_images(
            X[:n_first],
            dates[:n_first],
            tickers[:n_first],
            y[:n_first],
            title_prefix="First"
        )

        # Show last 10 images
        if n_samples > 10:
            n_last = min(10, n_samples)
            print(f"Showing last {n_last} images...")
            _show_multiple_images(
                X[-n_last:],
                dates[-n_last:],
                tickers[-n_last:],
                y[-n_last:],
                title_prefix="Last"
            )

    return X, y, returns, dates, closes, daily_returns, tickers
