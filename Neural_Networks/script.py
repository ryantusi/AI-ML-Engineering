"""
Implementing Neural Networks

The World Health Organization (WHO)’s Global Health Observatory (GHO) data repository tracks life expectancy for countries worldwide by following health status and many other related factors.

Although there have been a lot of studies undertaken in the past on factors affecting life expectancy considering demographic variables, income composition, and mortality rates, it was found that the effects of immunization and human development index were not taken into account.

This dataset covers a variety of indicators for all countries from 2000 to 2015 including:

    immunization factors
    mortality factors
    economic factors
    social factors
    other health-related factors

Ideally, this data will eventually inform countries concerning which factors to change in order to improve the life expectancy of their populations. If we can predict life expectancy well given all the factors, this is a good sign that there are some important patterns in the data. Life expectancy is expressed in years, and hence it is a number. This means that in order to build a predictive model one needs to use regression.

In this project, you will design, train, and evaluate a neural network model performing the task of regression to predict the life expectancy of countries using this dataset. Excited? Let’s go!

"""
# life_expectancy_nn.py
# Complete implementation of the "Implementing Neural Networks" project tasks.

import os
import numpy as np
import pandas as pd

# sklearn utilities
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

# TensorFlow / Keras
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import InputLayer, Dense
from tensorflow.keras.optimizers import Adam

# For reproducibility
RANDOM_STATE = 42
np.random.seed(RANDOM_STATE)
tf.random.set_seed(RANDOM_STATE)

def main():
    # -------------------------
    # 1. Load the data
    # -------------------------
    dataset_path = "life_expectancy.csv"
    if not os.path.exists(dataset_path):
        raise FileNotFoundError(f"Expected dataset file '{dataset_path}' in current directory.")

    # Task 1: read CSV into DataFrame
    dataset = pd.read_csv(dataset_path)

    # Task 2: observe the data
    print("=== Dataset head ===")
    print(dataset.head())
    print("\n=== Dataset summary ===")
    print(dataset.describe(include='all').T)

    # -------------------------
    # 3. Drop Country column
    # -------------------------
    if 'Country' in dataset.columns:
        dataset = dataset.drop(['Country'], axis=1)
        print("\nDropped 'Country' column.")
    else:
        # If the column isn't present, warn but continue (some versions of the file may differ)
        print("\nWarning: 'Country' column not found; continuing without dropping.")

    # -------------------------
    # 4. & 5. Labels and Features (using iloc)
    # -------------------------
    # labels: last column
    labels = dataset.iloc[:, -1]
    # features: all columns except the last
    features = dataset.iloc[:, 0:-1]

    print(f"\nFeatures shape: {features.shape}, Labels shape: {labels.shape}")
    print(f"Feature columns: {features.columns.tolist()}")

    # -------------------------
    # 6. One-hot encode categorical features
    # -------------------------
    # Using pandas.get_dummies on the features DataFrame
    features = pd.get_dummies(features)
    print(f"\nAfter get_dummies, features shape: {features.shape}")
    # Optional: show a few columns
    print("Sample feature columns:", features.columns[:12].tolist())

    # -------------------------
    # 7. Train/Test split
    # -------------------------
    # We'll choose test_size = 0.20 and random_state = RANDOM_STATE (you can change as needed)
    features_train, features_test, labels_train, labels_test = train_test_split(
        features, labels, test_size=0.20, random_state=RANDOM_STATE
    )
    print(f"\nTrain/test split: {features_train.shape} train, {features_test.shape} test")

    # -------------------------
    # 8. ColumnTransformer for scaling numeric columns
    # -------------------------
    # After get_dummies everything is numeric. We'll standardize all numeric columns.
    numeric_columns = features.columns.tolist()  # all columns are numeric after get_dummies
    ct = ColumnTransformer(
        transformers=[("scale", StandardScaler(), numeric_columns)],
        remainder='drop'
    )

    # -------------------------
    # 9. Fit & transform training features
    # -------------------------
    features_train_scaled = ct.fit_transform(features_train)
    print(f"\nfeatures_train_scaled shape: {features_train_scaled.shape}")

    # -------------------------
    # 10. Transform test features
    # -------------------------
    features_test_scaled = ct.transform(features_test)
    print(f"features_test_scaled shape: {features_test_scaled.shape}")

    # -------------------------
    # 11. Create Sequential model
    # -------------------------
    my_model = Sequential()

    # -------------------------
    # 12. Create input layer with shape = number of features
    # -------------------------
    n_features = features_train_scaled.shape[1]
    input_layer = InputLayer(input_shape=(n_features,))

    # -------------------------
    # 13. Add input layer to the model
    # -------------------------
    my_model.add(input_layer)

    # -------------------------
    # 14. Add a Dense hidden layer (relatively small example size)
    # -------------------------
    my_model.add(Dense(64, activation='relu'))

    # -------------------------
    # 15. Add output layer (single neuron for regression)
    # -------------------------
    my_model.add(Dense(1))

    # -------------------------
    # 16. Print model summary
    # -------------------------
    print("\n=== Model summary ===")
    my_model.summary()

    # -------------------------
    # 17. Initialize Adam optimizer with lr=0.01
    # -------------------------
    opt = Adam(learning_rate=0.01)

    # -------------------------
    # 18. Compile the model with mse loss and mae metric
    # -------------------------
    my_model.compile(loss='mse', metrics=['mae'], optimizer=opt)

    # -------------------------
    # 19. Fit the model
    # epochs = 50, batch_size = 1, verbose = 1
    # -------------------------
    print("\n=== Training model ===")
    history = my_model.fit(
        features_train_scaled,
        labels_train.values,
        epochs=50,
        batch_size=1,
        verbose=1,
        validation_split=0.1,  # optional small val split during training
        shuffle=True
    )

    # -------------------------
    # 20. Evaluate the model on the test set
    # Store results in res_mse, res_mae
    # -------------------------
    res = my_model.evaluate(features_test_scaled, labels_test.values, verbose=0)
    # Keras returns [loss, mae] because we compiled with loss='mse', metrics=['mae']
    res_mse, res_mae = res[0], res[1]

    # -------------------------
    # 21. Print final results
    # -------------------------
    print("\n=== Final evaluation on test set ===")
    print("Test MSE:", res_mse)
    print("Test MAE:", res_mae)
    print("Test RMSE:", np.sqrt(res_mse))  # helpful to read

    # Optionally save the trained model and the scaler pipeline for deployment
    model_path = "life_expectancy_nn_model"
    print(f"\nSaving model to '{model_path}' (TensorFlow SavedModel format)...")
    my_model.save(model_path)
    # Save the ColumnTransformer using joblib so we can reuse it for preprocessing
    import joblib
    joblib.dump(ct, "column_transformer.joblib")
    print("Saved preprocessing transformer to 'column_transformer.joblib'.")

if __name__ == "__main__":
    main()
