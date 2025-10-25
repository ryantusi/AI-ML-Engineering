# hotel_cancellation_predictor.py
# ----------------------------------------------------------
# Predicting Hotel Booking Cancellations using PyTorch
# ----------------------------------------------------------

# ========== IMPORT LIBRARIES ==========
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report

# ========== LOAD AND INSPECT DATA ==========
print("\nLoading dataset...")
hotels = pd.read_csv("resort_hotel_bookings.csv")
print(hotels.head(), "\n")
print(hotels.info(), "\n")

# ========== TARGET DISTRIBUTION ==========
print("Cancellation counts:\n", hotels['is_canceled'].value_counts())
print("\nCancellation percentages:\n", hotels['is_canceled'].value_counts(normalize=True))

# ========== RESERVATION STATUS CHECK ==========
print("\nReservation status counts:\n", hotels['reservation_status'].value_counts())
print("\nReservation status percentages:\n", hotels['reservation_status'].value_counts(normalize=True))

# ========== EXPLORING CANCELLATION BY MONTH ==========
cancellations_by_month = hotels.groupby('arrival_date_month')['is_canceled'].mean().sort_values()
print("\nCancellations by month:\n", cancellations_by_month)

# ========== CLEANING AND PREPARATION ==========
object_columns = [
    'arrival_date_month', 'meal', 'country', 'market_segment', 'distribution_channel',
    'reserved_room_type', 'assigned_room_type', 'deposit_type', 'customer_type'
]
print("\nPreview categorical columns:\n", hotels[object_columns].head())

# Drop unnecessary columns
drop_columns = [
    'country', 'agent', 'company', 'reservation_status_date',
    'arrival_date_week_number', 'arrival_date_day_of_month', 'arrival_date_year'
]
hotels = hotels.drop(labels=drop_columns, axis=1)

# Encode 'meal' column
hotels['meal'] = hotels['meal'].replace({'Undefined': 0, 'SC': 0, 'BB': 1, 'HB': 2, 'FB': 3})

# One-hot encode categorical features
one_hot_columns = [
    'arrival_date_month', 'distribution_channel', 'reserved_room_type',
    'assigned_room_type', 'deposit_type', 'customer_type', 'market_segment'
]
hotels = pd.get_dummies(hotels, columns=one_hot_columns, dtype=int)

print("\nData after encoding:\n", hotels.head())
print(f"\nTotal columns after encoding: {len(hotels.columns)}")

# ========== BINARY CLASSIFICATION MODEL ==========
print("\n=== Binary Classification: Predicting is_canceled ===")

# Features and labels
remove_cols = ['is_canceled', 'reservation_status']
train_features = [x for x in hotels.columns if x not in remove_cols]

X = torch.tensor(hotels[train_features].values, dtype=torch.float)
y = torch.tensor(hotels['is_canceled'].values, dtype=torch.float).view(-1, 1)

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(
    X, y, train_size=0.80, test_size=0.20, random_state=42
)
print("Training Shape:", X_train.shape)
print("Testing Shape:", X_test.shape)

# Define model
torch.manual_seed(42)
model = nn.Sequential(
    nn.Linear(65, 36),
    nn.ReLU(),
    nn.Linear(36, 18),
    nn.ReLU(),
    nn.Linear(18, 1),
    nn.Sigmoid()
)

# Loss and optimizer
loss = nn.BCELoss()
optimizer = optim.Adam(model.parameters(), lr=0.005)

# Train model
print("\nTraining Binary Model...")
num_epochs = 1000
for epoch in range(num_epochs):
    predictions = model(X_train)
    BCELoss = loss(predictions, y_train)
    BCELoss.backward()
    optimizer.step()
    optimizer.zero_grad()

    if (epoch + 1) % 100 == 0:
        predicted_labels = (predictions >= 0.5).int()
        accuracy = accuracy_score(y_train, predicted_labels)
        print(f'Epoch [{epoch+1}/{num_epochs}] - Loss: {BCELoss.item():.4f}, Accuracy: {accuracy:.4f}')

# Evaluate model
model.eval()
with torch.no_grad():
    test_predictions = model(X_test)
    test_predicted_labels = (test_predictions >= 0.5).int()

test_accuracy = accuracy_score(y_test, test_predicted_labels)
print(f"\nBinary Test Accuracy: {test_accuracy:.4f}")
print("\nClassification Report:\n", classification_report(y_test, test_predicted_labels))

# ========== MULTICLASS CLASSIFICATION MODEL ==========
print("\n=== Multiclass Classification: Predicting reservation_status ===")

# Label encode reservation_status
hotels['reservation_status'] = hotels['reservation_status'].replace({
    'Check-Out': 2, 'Canceled': 1, 'No-Show': 0
})

X = torch.tensor(hotels[train_features].values, dtype=torch.float)
y = torch.tensor(hotels['reservation_status'].values, dtype=torch.long)

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(
    X, y, train_size=0.8, test_size=0.2, random_state=42
)
print("Training Shape:", X_train.shape)
print("Testing Shape:", X_test.shape)

# Define model
torch.manual_seed(42)
multiclass_model = nn.Sequential(
    nn.Linear(65, 65),
    nn.ReLU(),
    nn.Linear(65, 36),
    nn.ReLU(),
    nn.Linear(36, 3)
)

# Loss and optimizer
loss = nn.CrossEntropyLoss()
optimizer = optim.Adam(multiclass_model.parameters(), lr=0.01)

# Train model
print("\nTraining Multiclass Model...")
num_epochs = 500
for epoch in range(num_epochs):
    predictions = multiclass_model(X_train)
    CELoss = loss(predictions, y_train)
    CELoss.backward()
    optimizer.step()
    optimizer.zero_grad()

    if (epoch + 1) % 100 == 0:
        predicted_labels = torch.argmax(predictions, dim=1)
        accuracy = accuracy_score(y_train, predicted_labels)
        print(f'Epoch [{epoch+1}/{num_epochs}] - Loss: {CELoss.item():.4f}, Accuracy: {accuracy:.4f}')

# Evaluate multiclass model
multiclass_model.eval()
with torch.no_grad():
    test_predictions = multiclass_model(X_test)
    test_predicted_labels = torch.argmax(test_predictions, dim=1)

test_accuracy = accuracy_score(y_test, test_predicted_labels)
print(f"\nMulticlass Test Accuracy: {test_accuracy:.4f}")
print("\nClassification Report:\n", classification_report(y_test, test_predicted_labels))
