import pandas as pd
import os
import numpy as np
from xgboost import XGBRegressor
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import mean_squared_error
from scipy.optimize import minimize

# Define the ticker symbols for the index funds and bond types
index_funds = ['SPY', 'DIA', 'QQQ', 'IWM', 'URTH', 'EEM']
bond_types = ['BIL', 'SHY', 'IEF', 'TLT', 'MUB']
additional_bonds = ['LQD', 'HYG', 'BNDX']

# Combine all tickers into one list
all_tickers = index_funds + bond_types + additional_bonds

# Create a dictionary to map tickers to their full names
ticker_to_name = {
    'SPY': 'S&P 500 ETF',
    'DIA': 'Dow Jones Industrial Average ETF',
    'QQQ': 'NASDAQ 100 ETF',
    'IWM': 'Russell 2000 ETF',
    'URTH': 'MSCI World ETF',
    'EEM': 'MSCI Emerging Markets ETF',
    'BIL': 'SPDR Bloomberg Barclays 1-3 Month T-Bill ETF',
    'SHY': 'iShares 1-3 Year Treasury Bond ETF',
    'IEF': 'iShares 7-10 Year Treasury Bond ETF',
    'TLT': 'iShares 20+ Year Treasury Bond ETF',
    'MUB': 'iShares National Muni Bond ETF',
    'LQD': 'iShares iBoxx $ Investment Grade Corporate Bond ETF',
    'HYG': 'iShares iBoxx $ High Yield Corporate Bond ETF',
    'BNDX': 'Vanguard Total International Bond ETF'
}

# Define the maximum allowed risk
max_risk = 0.05  # You can adjust this value as needed
max_investment = 1 #You can adjust this value as needed, this is maximum percentage you want to invest in a stock

# Create a DataFrame to store predicted metrics
predicted_metrics_df = pd.DataFrame()

# Read and process the data
for ticker in all_tickers:
    csv_file = f"{ticker}.csv"

    if not os.path.exists(csv_file):
        print(f"CSV file for {ticker} not found. Skipping.")
        continue

    data = pd.read_csv(csv_file)
    data['Daily Return'] = data['Adj Close'].pct_change()

    # Calculate lagged features
    for i in range(1, 6):  # Creating 5 lagged features
        data[f'Lagged_Return_{i}'] = data['Daily Return'].shift(i) #Understand this 

    # Calculate rolling standard deviation for y_risk
    y_risk = data['Daily Return'].rolling(window=5).std()

    # Drop NaN values from y_risk, X, and y_return to align their lengths
    y_risk.dropna(inplace=True)
    data.dropna(inplace=True)  # This will drop NaN in 'Daily Return' and lagged features

    # Features and Labels
    X = data[['Lagged_Return_1', 'Lagged_Return_2', 'Lagged_Return_3', 'Lagged_Return_4', 'Lagged_Return_5']]
    y_return = data['Daily Return'] #UNDERSTAND THIS

    # Time Series Cross-Validation
    tscv = TimeSeriesSplit(n_splits=5)

    # Model for Returns
    model_return = XGBRegressor(objective='reg:squarederror')
    mse_accumulator_return = []  # Initialize an empty list to store MSEs
    for train_index, test_index in tscv.split(X):
        X_train, X_test = X.iloc[train_index], X.iloc[test_index]
        y_train, y_test = y_return.iloc[train_index], y_return.iloc[test_index]
        model_return.fit(X_train, y_train)
        y_pred = model_return.predict(X_test)
        mse = mean_squared_error(y_test, y_pred)
        mse_accumulator_return.append(mse)  # Append the MSE to the accumulator

    # Calculate the average MSE and print it
    average_mse_return = sum(mse_accumulator_return) / len(mse_accumulator_return)
    print(f"Average Mean Squared Error for Returns {ticker}: {average_mse_return}")

    # Model for Risk
    model_risk = XGBRegressor(objective='reg:squarederror')
    mse_accumulator_risk = []  # Initialize an empty list to store MSEs for risk
    for train_index, test_index in tscv.split(X):
        X_train, X_test = X.iloc[train_index], X.iloc[test_index]
        y_train, y_test = y_risk.iloc[train_index], y_risk.iloc[test_index]
        model_risk.fit(X_train, y_train)
        y_pred = model_risk.predict(X_test)
        mse = mean_squared_error(y_test, y_pred)
        mse_accumulator_risk.append(mse)  # Append the MSE to the accumulator

    # Calculate the average MSE for risk and print it
    average_mse_risk = sum(mse_accumulator_risk) / len(mse_accumulator_risk)
    print(f"Average Mean Squared Error for Risk {ticker}: {average_mse_risk}")


    # Predict future returns and risk for the ticker
    future_return = model_return.predict(X.tail(1))
    future_risk = model_risk.predict(X.tail(1))

    # Store predicted metrics in the DataFrame
    predicted_metrics_df.loc[ticker, 'Annualized Return (%)'] = future_return[0]
    predicted_metrics_df.loc[ticker, 'Risk (%)'] = future_risk[0]

# Define the objective function to maximize the Sharpe ratio
def objective(weights):
    portfolio_return = np.dot(weights, predicted_metrics_df['Annualized Return (%)'])
    portfolio_volatility = np.dot(weights, predicted_metrics_df['Risk (%)'])
    return -portfolio_return / portfolio_volatility  # We negate because we want to maximize

# Define the constraint for risk
def risk_constraint(weights): #Check this function
    portfolio_volatility = np.dot(weights, predicted_metrics_df['Risk (%)'])
    return max_risk - portfolio_volatility  # Ensuring the portfolio risk does not exceed max_risk

# Define the constraint to ensure no single asset weight is above max investment%
def single_asset_constraint(weights):
    return max_investment - max(weights)

# Initialize weights equally
initial_weights = np.array([1. / len(predicted_metrics_df)] * len(predicted_metrics_df))

# Define the constraints and bounds
constraints = (
    {'type': 'eq', 'fun': lambda weights: np.sum(weights) - 1},  # The sum of weights is 1
    {'type': 'ineq', 'fun': risk_constraint},  # The portfolio risk does not exceed max_risk
    {'type': 'ineq', 'fun': single_asset_constraint}  # No single asset weight is above 20%
)

# Define the bounds for each weight to be between 0 and 1
bounds = tuple((0, 1) for _ in range(len(predicted_metrics_df)))

# Perform the optimization
solution = minimize(
    objective,
    initial_weights,
    method='SLSQP',
    constraints=constraints,
    bounds=bounds,
    options={'maxiter': 1000, 'disp': True}
)

# Check if the optimization was successful
if not solution.success:
    raise ValueError("Optimization failed:", solution.message)

# Extract the optimal asset weights
optimal_weights = solution.x

# Calculate the expected portfolio return and risk using the optimal weights
expected_return = np.dot(optimal_weights, predicted_metrics_df['Annualized Return (%)'])
expected_risk = np.dot(optimal_weights, predicted_metrics_df['Risk (%)'])

# Print the expected portfolio return and risk
print(f"Expected Portfolio Return: {expected_return:.2f}%")
print(f"Expected Portfolio Risk: {expected_risk:.2f}%")
print("Optimal Portfolio Composition:")
for i, ticker in enumerate(predicted_metrics_df.index):
    full_name = ticker_to_name.get(ticker, ticker)
    weight = optimal_weights[i] * 100  # Convert to percentage
    print(f"Invest {weight:.2f}% of funds in {full_name}")
