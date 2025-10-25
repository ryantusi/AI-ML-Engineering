"""
Predicting Residential EV Charging Loads using Neural Networks
--------------------------------------------------------------
This script performs data loading, cleaning, merging, preprocessing,
and a baseline Linear Regression model evaluation for EV charging load prediction.

Dataset: 
 - EV charging reports from Norwegian residential buildings
 - Local traffic distribution data

Author: [Your Name]
"""

# ===============================
# Imports
# ===============================
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error


# ===============================
# Task Group 1 - Load, Inspect, and Merge Datasets
# ===============================

# Load datasets
ev_charging_reports = pd.read_csv("datasets/EV charging reports.csv")
traffic_reports = pd.read_csv("datasets/Local traffic distribution.csv")

print("EV Charging Reports Preview:")
print(ev_charging_reports.head(), "\n")

print("Traffic Reports Preview:")
print(traffic_reports.head(), "\n")

# Merge datasets
ev_charging_traffic = ev_charging_reports.merge(
    traffic_reports,
    left_on='Start_plugin_hour',
    right_on='Date_from'
)

print("Merged Dataset Preview:")
print(ev_charging_traffic.head(), "\n")

# Inspect merged dataset
print("Merged Dataset Info:")
ev_charging_traffic.info()
print("\n")


# ===============================
# Task Group 2 - Data Cleaning and Preparation
# ===============================

# Drop unused or non-numeric columns
drop_columns = [
    'session_ID', 'Garage_ID', 'User_ID', 
    'Shared_ID', 'Plugin_category', 'Duration_category', 
    'Start_plugin', 'Start_plugin_hour', 'End_plugout', 
    'End_plugout_hour', 'Date_from', 'Date_to'
]

ev_charging_traffic = ev_charging_traffic.drop(columns=drop_columns, axis=1)
print("Dataset after dropping unused columns:")
print(ev_charging_traffic.head(), "\n")

# Replace commas with periods in numeric columns (European notation)
for column in ev_charging_traffic.columns:
    if ev_charging_traffic[column].dtype == 'object':
        ev_charging_traffic[column] = ev_charging_traffic[column].str.replace(',', '.')

# Convert all columns to float
for column in ev_charging_traffic.columns:
    ev_charging_traffic[column] = ev_charging_traffic[column].astype(float)

print("Dataset after cleaning and type conversion:")
print(ev_charging_traffic.head(), "\n")


# ===============================
# Task Group 3 - Train Test Split
# ===============================

# Separate features and target
X = ev_charging_traffic.drop(['El_kWh'], axis=1)
y = ev_charging_traffic['El_kWh']

# Split into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    train_size=0.80,
    test_size=0.20,
    random_state=2
)

print(f"Training size: {X_train.shape}")
print(f"Testing size: {X_test.shape}\n")


# ===============================
# Task Group 4 - Linear Regression Baseline
# ===============================

# Train a Linear Regression model
linear_model = LinearRegression()
linear_model.fit(X_train, y_train)

# Predict and evaluate
linear_test_predictions = linear_model.predict(X_test)
test_mse = mean_squared_error(y_test, linear_test_predictions)

print(f"Linear Regression - Test Set MSE: {test_mse:.4f}")
print(f"Root Mean Squared Error (RMSE): {np.sqrt(test_mse):.4f} kWh\n")


# ===============================
# (Next: Neural Network using PyTorch - Task Group 5)
# ===============================

# Placeholder for future NN model
if __name__ == "__main__":
    print("✅ Data processing and Linear Regression baseline completed successfully.")
    print("Next step: Implement PyTorch neural network for load prediction.")
