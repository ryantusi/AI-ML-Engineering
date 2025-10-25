"""
Predict Credit Card Fraud

Credit card fraud is one of the leading causes of identify theft around the world. In 2018 alone, over $24 billion were stolen through fraudulent credit card transactions. Financial institutions employ a wide variety of different techniques to prevent fraud, one of the most common being Logistic Regression.

In this project, you are a Data Scientist working for a credit card company. You have access to a dataset (based on a synthetic financial dataset), that represents a typical set of credit card transactions. transactions.csv is the original dataset containing 200k transactions. For starters, we’re going to be working with a small portion of this dataset, transactions_modified.csv, which contains one thousand transactions. Your task is to use Logistic Regression and create a predictive model to determine if a transaction is fraudulent or not. 
"""

# Import necessary libraries
import codecademylib3_seaborn
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# ------------------------------------------------------
# 1. Load and Inspect Data
# ------------------------------------------------------
transactions = pd.read_csv('transactions_modified.csv')

# Peek at the data
print("🔹 Preview of data:")
print(transactions.head(), "\n")

# Check datatypes and missing values
print("🔹 Dataset info:")
print(transactions.info(), "\n")

# Count how many transactions are fraudulent
fraud_count = transactions["isFraud"].sum()
print(f"⚠️ Number of fraudulent transactions: {fraud_count}")
print(f"Total transactions: {len(transactions)}")
print(f"Percentage fraud: {fraud_count / len(transactions) * 100:.2f}%\n")

# ------------------------------------------------------
# 2. Summary statistics for the 'amount' column
# ------------------------------------------------------
print("💰 Summary statistics for transaction amounts:")
print(transactions["amount"].describe(), "\n")

# ------------------------------------------------------
# 3. Create isPayment field (1 for PAYMENT or DEBIT)
# ------------------------------------------------------
transactions["isPayment"] = transactions["type"].isin(["PAYMENT", "DEBIT"]).astype(int)

# ------------------------------------------------------
# 4. Create isMovement field (1 for CASH_OUT or TRANSFER)
# ------------------------------------------------------
transactions["isMovement"] = transactions["type"].isin(["CASH_OUT", "TRANSFER"]).astype(int)

# ------------------------------------------------------
# 5. Create accountDiff field = |oldbalanceOrg - oldbalanceDest|
# ------------------------------------------------------
transactions["accountDiff"] = np.abs(transactions["oldbalanceOrg"] - transactions["oldbalanceDest"])

print("✅ New derived columns added:")
print(transactions[["type", "amount", "isPayment", "isMovement", "accountDiff", "isFraud"]].head(), "\n")

# ------------------------------------------------------
# 6. Select features and label
# ------------------------------------------------------
features = transactions[["amount", "isPayment", "isMovement", "accountDiff"]]
label = transactions["isFraud"]

# ------------------------------------------------------
# 7. Split data into training and test sets
# ------------------------------------------------------
X_train, X_test, y_train, y_test = train_test_split(features, label, test_size=0.3, random_state=42, stratify=label)
print("✅ Data split completed:")
print(f"Training samples: {len(X_train)}, Testing samples: {len(X_test)}\n")

# ------------------------------------------------------
# 8. Normalize (scale) the features
# ------------------------------------------------------
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# ------------------------------------------------------
# 9. Train Logistic Regression Model
# ------------------------------------------------------
model = LogisticRegression(max_iter=1000, random_state=42)
model.fit(X_train_scaled, y_train)

# ------------------------------------------------------
# 10. Evaluate Model on Training Data
# ------------------------------------------------------
train_score = model.score(X_train_scaled, y_train)
print(f"🎯 Training Accuracy: {train_score:.4f}")

# ------------------------------------------------------
# 11. Evaluate Model on Test Data
# ------------------------------------------------------
test_score = model.score(X_test_scaled, y_test)
print(f"🧪 Test Accuracy: {test_score:.4f}\n")

# ------------------------------------------------------
# 12. Check Feature Importance
# ------------------------------------------------------
coefficients = pd.DataFrame({
    "Feature": features.columns,
    "Coefficient": model.coef_[0]
}).sort_values(by="Coefficient", ascending=False)

print("📊 Feature Importance (Positive → Higher chance of being Fraudulent):")
print(coefficients, "\n")

# Plot feature importance
plt.figure(figsize=(10,4))
x_pos = np.arange(len(coefficients))                     # numeric positions
plt.bar(x_pos, coefficients["Coefficient"], color='teal')# plot with numeric x
plt.xticks(x_pos, coefficients["Feature"], rotation=45, ha='right')  # label ticks with feature names
plt.title("Feature Importance in Fraud Prediction")
plt.xlabel("Feature")
plt.ylabel("Coefficient")
plt.tight_layout()   # avoids label cutoff
plt.show()

# ------------------------------------------------------
# 13. New Transaction Data
# ------------------------------------------------------
transaction1 = np.array([123456.78, 0.0, 1.0, 54670.1])
transaction2 = np.array([98765.43, 1.0, 0.0, 8524.75])
transaction3 = np.array([543678.31, 1.0, 0.0, 510025.5])

# Your custom transaction
your_transaction = np.array([500000.00, 0.0, 1.0, 100000.0])

# ------------------------------------------------------
# 14. Combine transactions into one array
# ------------------------------------------------------
sample_transactions = np.stack((transaction1, transaction2, transaction3, your_transaction))

# ------------------------------------------------------
# 15. Scale the new sample transactions
# ------------------------------------------------------
sample_transactions_scaled = scaler.transform(sample_transactions)

# ------------------------------------------------------
# 16. Predict on new transactions
# ------------------------------------------------------
predictions = model.predict(sample_transactions_scaled)
probabilities = model.predict_proba(sample_transactions_scaled)

print("🔍 Predictions for new transactions (0 = Not Fraud, 1 = Fraud):")
print(predictions)

print("\n📈 Probabilities for each transaction [Not Fraud, Fraud]:")
print(probabilities)

# Optional: make output more readable
for i, p in enumerate(predictions):
    status = "🚨 FRAUD" if p == 1 else "✅ Safe"
    print(f"Transaction {i+1}: {status} (Fraud Probability = {probabilities[i][1]:.4f})")

