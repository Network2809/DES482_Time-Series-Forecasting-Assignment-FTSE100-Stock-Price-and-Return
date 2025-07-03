# Step 1: Import Libraries
import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize

# Step 2: Data Collection
tickers = ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'META']  # Select at least 5 S&P 500 stocks
data = yf.download(tickers, start='2020-01-01', end='2023-01-01')['Close']

# Step 3: Data Preparation
returns = data.pct_change().dropna()
mean_returns = returns.mean() * 252
cov_matrix = returns.cov() * 252
num_assets = len(tickers)

# Plot historical prices
plt.figure(figsize=(12, 6))
plt.plot(data)
plt.title('Historical Adjusted Closing Prices')
plt.legend(tickers)
plt.show()

# Plot daily returns
plt.figure(figsize=(12, 6))
plt.plot(returns)
plt.title('Daily Returns')
plt.legend(tickers)
plt.show()

# Step 4: Define Functions
def port_return(weights):
    return np.dot(weights, mean_returns)

def port_risk(weights):
    return np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights)))

def sharpe_ratio(weights):
    return - (port_return(weights) / port_risk(weights))

# Step 5: Set Constraints and Bounds
constraints = ({'type': 'eq', 'fun': lambda x: np.sum(x) - 1})
bounds = tuple((0, 1) for _ in range(num_assets))
initial_guess = num_assets * [1. / num_assets, ]

# Step 6: Optimization
min_risk_result = minimize(port_risk, initial_guess, method='SLSQP', bounds=bounds, constraints=constraints)
max_sharpe_result = minimize(sharpe_ratio, initial_guess, method='SLSQP', bounds=bounds, constraints=constraints)

# Step 7: Evaluate Results
min_risk_weights = min_risk_result.x
max_sharpe_weights = max_sharpe_result.x

print("Minimum Risk Portfolio:")
print("Weights:", min_risk_weights)
print("Expected Return:", port_return(min_risk_weights))
print("Portfolio Risk:", port_risk(min_risk_weights))
print("Sharpe Ratio:", -sharpe_ratio(min_risk_weights))

print("\nMaximum Sharpe Ratio Portfolio:")
print("Weights:", max_sharpe_weights)
print("Expected Return:", port_return(max_sharpe_weights))
print("Portfolio Risk:", port_risk(max_sharpe_weights))
print("Sharpe Ratio:", -sharpe_ratio(max_sharpe_weights))

# Plot Efficient Frontier
def plot_efficient_frontier(num_points=100):
    results = np.zeros((3, num_points))
    for i in range(num_points):
        weights = np.random.dirichlet(np.ones(num_assets), size=1)[0]
        port_ret = port_return(weights)
        port_vol = port_risk(weights)
        sr = port_ret / port_vol
        results[0, i] = port_vol
        results[1, i] = port_ret
        results[2, i] = sr
    plt.figure(figsize=(10, 7))
    plt.scatter(results[0, :], results[1, :], c=results[2, :], cmap='viridis')
    plt.xlabel('Portfolio Risk (Std Dev)')
    plt.ylabel('Portfolio Return')
    plt.title('Efficient Frontier')
    plt.colorbar(label='Sharpe Ratio')
    plt.show()

# Automatically plot the efficient frontier
plot_efficient_frontier()

# Summary
print(f"We’re working with these stocks: {', '.join(tickers)}")

print("\nLet’s start with the Minimum Risk Portfolio:")
for ticker, weight in zip(tickers, min_risk_weights):
    print(f"- {ticker}: {weight:.2%} of the portfolio")
print(f"Expected Annual Return: {port_return(min_risk_weights):.2%}")
print(f"Portfolio Risk (Std Dev): {port_risk(min_risk_weights):.2%}")
print(f"Sharpe Ratio: {-sharpe_ratio(min_risk_weights):.2f}")

print("\nNow, check out the Maximum Sharpe Ratio Portfolio:")
for ticker, weight in zip(tickers, max_sharpe_weights):
    print(f"- {ticker}: {weight:.2%} of the portfolio")
print(f"Expected Annual Return: {port_return(max_sharpe_weights):.2%}")
print(f"Portfolio Risk (Std Dev): {port_risk(max_sharpe_weights):.2%}")
print(f"Sharpe Ratio: {-sharpe_ratio(max_sharpe_weights):.2f}")

