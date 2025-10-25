"""
Cancer Classifier

In this project, we will be using several Python libraries to make a K-Nearest Neighbor classifier that is trained to predict whether a patient has breast cancer. 
"""

# Imports
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report, confusion_matrix

# Load the dataset
data = load_breast_cancer()
X, y = data.data, data.target

# Split the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=100)

# Scale the data (important for KNN)
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Evaluate KNN for k = 1 to 100
accuracies = []
k_list = range(1, 101)

for k in k_list:
    model = KNeighborsClassifier(n_neighbors=k)
    model.fit(X_train, y_train)
    accuracies.append(model.score(X_test, y_test))

# Find the best k
best_k = k_list[np.argmax(accuracies)]
best_acc = max(accuracies)
print(f"⭐ Best k value: {best_k} with accuracy: {best_acc*100:.2f}%")

# Train the final model with best k
best_model = KNeighborsClassifier(n_neighbors=best_k)
best_model.fit(X_train, y_train)
y_pred = best_model.predict(X_test)

# Model evaluation
print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred))
print("\nClassification Report:")
print(classification_report(y_test, y_pred, target_names=data.target_names))

# Plot accuracies
plt.figure(figsize=(10,6))
plt.plot(k_list, accuracies, color='royalblue', marker='o', linestyle='dashed', linewidth=2, markersize=5)
plt.title('KNN Accuracy for Different k Values', fontsize=14)
plt.xlabel('Number of Neighbors (k)', fontsize=12)
plt.ylabel('Validation Accuracy', fontsize=12)
plt.grid(True, linestyle='--', alpha=0.6)
plt.axvline(best_k, color='red', linestyle=':', label=f'Best k = {best_k}')
plt.legend()
plt.tight_layout()
plt.show()
