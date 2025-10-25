"""
Tennis Ace
Overview

This project contains a series of open-ended requirements which describe the project you’ll be building. There are many possible ways to correctly fulfill all of these requirements, and you should expect to use the internet, Codecademy, and other resources when you encounter a problem.
Project Goals

You will create a linear regression model that predicts the outcome for a tennis player based on their playing habits. By analyzing and modeling the Association of Tennis Professionals (ATP) data, you will determine what it takes to be one of the best tennis players in the world.
"""

# --- Import necessary libraries ---
# pandas for data handling, matplotlib for plotting,
# scikit-learn for machine learning
import codecademylib3_seaborn
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

# --- Load the data ---
df = pd.read_csv("tennis_stats.csv")

# Let's see what the data looks like
print("Dataset preview:")
print(df.head(), "\n")

# --- Exploratory Data Analysis (EDA) ---
# We'll explore how some features relate to outcomes.
# Scatter plots help visualize relationships.

plt.scatter(df["BreakPointsOpportunities"], df["Winnings"], alpha=0.5, color="orange")
plt.title("Break Points Opportunities vs Winnings")
plt.xlabel("Break Points Opportunities")
plt.ylabel("Winnings ($)")
plt.show()

plt.scatter(df["DoubleFaults"], df["Losses"], alpha=0.5, color="red")
plt.title("Double Faults vs Losses")
plt.xlabel("Double Faults")
plt.ylabel("Losses")
plt.show()

# We can see some relationship:
# - More Break Point Opportunities often means higher winnings
# - More Double Faults tends to correlate with more losses

# =====================================================
# 🧩 SINGLE FEATURE LINEAR REGRESSION MODELS
# =====================================================

# Helper function to train, evaluate and plot a model
def run_single_feature_regression(feature_name, target_name, color="blue"):
    print(f"\nRunning Single Feature Regression: {feature_name} → {target_name}")
    
    # Define features (X) and target (y)
    X = df[[feature_name]]
    y = df[[target_name]]

    # Split data into train and test sets (80% train, 20% test)
    X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.8, random_state=42)

    # Create and train the model
    model = LinearRegression()
    model.fit(X_train, y_train)

    # Evaluate model (R² score)
    score = model.score(X_test, y_test)
    print(f"Model R² Score: {score:.4f}")

    # Predict on test data
    y_pred = model.predict(X_test)

    # Plot prediction vs actual outcome
    plt.figure(figsize=(5,4))
    plt.scatter(y_test, y_pred, alpha=0.5, color=color)
    plt.title(f"Predicted vs Actual ({feature_name} → {target_name})")
    plt.xlabel("Actual " + target_name)
    plt.ylabel("Predicted " + target_name)
    plt.show()

    return score


# --- Model 1 ---
score1 = run_single_feature_regression("FirstServeReturnPointsWon", "Winnings", color="green")

# --- Model 2 ---
score2 = run_single_feature_regression("BreakPointsOpportunities", "Winnings", color="orange")

# --- Model 3 ---
score3 = run_single_feature_regression("DoubleFaults", "Losses", color="red")

print(f"\nBest Single Feature Model Score: {max(score1, score2, score3):.4f}")

# From our trials, BreakPointsOpportunities → Winnings tends to give the best performance.

# =====================================================
# 🧩 TWO FEATURE LINEAR REGRESSION
# =====================================================

# We'll now use 2 features to predict Winnings
print("\nRunning Two Feature Regression: ['BreakPointsOpportunities', 'FirstServeReturnPointsWon'] → Winnings")

X_two = df[["BreakPointsOpportunities", "FirstServeReturnPointsWon"]]
y_two = df[["Winnings"]]

X_train, X_test, y_train, y_test = train_test_split(X_two, y_two, train_size=0.8, random_state=42)

model_two = LinearRegression()
model_two.fit(X_train, y_train)

score_two = model_two.score(X_test, y_test)
print(f"Two Feature Model R² Score: {score_two:.4f}")

y_pred_two = model_two.predict(X_test)

plt.figure(figsize=(5,4))
plt.scatter(y_test, y_pred_two, alpha=0.5, color="purple")
plt.title("Predicted vs Actual (2 Features → Winnings)")
plt.xlabel("Actual Winnings")
plt.ylabel("Predicted Winnings")
plt.show()

# =====================================================
# 🧩 MULTIPLE FEATURE LINEAR REGRESSION
# =====================================================

print("\nRunning Multiple Feature Regression (All Major Stats) → Winnings")

# Using multiple key tennis stats to predict yearly Winnings
features = [
    'FirstServe','FirstServePointsWon','FirstServeReturnPointsWon',
    'SecondServePointsWon','SecondServeReturnPointsWon','Aces',
    'BreakPointsConverted','BreakPointsFaced','BreakPointsOpportunities',
    'BreakPointsSaved','DoubleFaults','ReturnGamesPlayed','ReturnGamesWon',
    'ReturnPointsWon','ServiceGamesPlayed','ServiceGamesWon','TotalPointsWon',
    'TotalServicePointsWon'
]

X_multi = df[features]
y_multi = df[["Winnings"]]

X_train, X_test, y_train, y_test = train_test_split(X_multi, y_multi, train_size=0.8, random_state=42)

model_multi = LinearRegression()
model_multi.fit(X_train, y_train)

score_multi = model_multi.score(X_test, y_test)
print(f"Multiple Feature Model R² Score: {score_multi:.4f}")

y_pred_multi = model_multi.predict(X_test)

plt.figure(figsize=(5,4))
plt.scatter(y_test, y_pred_multi, alpha=0.5, color="teal")
plt.title("Predicted vs Actual (Multiple Features → Winnings)")
plt.xlabel("Actual Winnings")
plt.ylabel("Predicted Winnings")
plt.show()

# =====================================================
# ✅ SUMMARY
# =====================================================

print("\n--- SUMMARY OF MODEL SCORES ---")
print(f"Single Feature 1 (FirstServeReturnPointsWon): {score1:.4f}")
print(f"Single Feature 2 (BreakPointsOpportunities): {score2:.4f}")
print(f"Single Feature 3 (DoubleFaults → Losses): {score3:.4f}")
print(f"Two Feature Model: {score_two:.4f}")
print(f"Multiple Feature Model: {score_multi:.4f}")

print("\n✅ As expected, the model using multiple features performs best overall.")
print("This suggests that predicting tennis performance (earnings) depends on a combination of offensive and defensive stats, not just one metric.")
# --- End of script ---
