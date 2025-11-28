"""
Train DQN model using live stock data from StockDataFetcher.
"""
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from trading_system import fetch_and_prepare
import train as TR

# DQN model parameters (from paper)
FSize = 5
PSize = 2
PStride = 2
NumAction = 3
W = 32  # Image size

# Training hyperparameters
maxiter = 500000        # iterations (paper uses 5M)
learning_rate = 0.00001
epsilon_min = 0.1
M = 1000                # memory buffer capacity
B = 10                  # theta update interval
C = 1000                # target network update interval
Gamma = 0.99            # discount factor
P = 0                   # transaction penalty
Beta = 32               # batch size

# Data configuration
TICKERS = ["AAPL", "MSFT", "GOOGL", "AMZN", "META", "NVDA", "JPM", "V", "JNJ", "WMT"]
START_DATE = "2017-01-01"
END_DATE = "2018-12-31"

print("="*70)
print("DQN Training with Live Stock Data (StockDataFetcher)")
print("="*70)

# Fetch and prepare data
print(f"\nFetching data for {len(TICKERS)} stocks...")
XData, YData = fetch_and_prepare(TICKERS, START_DATE, END_DATE, W, W)

print(f"\nData prepared:")
print(f"  Companies: {len(XData)}")
print(f"  Days per company: {len(XData[0]) if XData else 0}")

# Initialize model
print(f"\nInitializing DQN model...")
print(f"  Max iterations: {maxiter:,}")
print(f"  Learning rate: {learning_rate}")
print(f"  Batch size: {Beta}")

Model = TR.trainModel(1.0, epsilon_min, maxiter, Beta, B, C, learning_rate, P)
Model.set_Data(XData, YData)

# Train
print(f"\nStarting training...")
Model.trainModel(W, W, FSize, PSize, PStride, NumAction, M, Gamma)

print("\n" + "="*70)
print("Training Complete!")
print("Model saved as 'DeepQ' checkpoint")
print("="*70)

