"""
Classification

In this project, you will use a dataset from Kaggle to predict the survival of patients with heart failure from serum creatinine and ejection fraction, and other factors such as age, anemia, diabetes, and so on.

Cardiovascular diseases (CVDs) are the number 1 cause of death globally, taking an estimated 17.9 million lives each year, which accounts for 31% of all deaths worldwide. Heart failure is a common event caused by CVDs, and this dataset contains 12 features that can be used to predict mortality by heart failure.

Most cardiovascular diseases can be prevented by addressing behavioral risk factors such as tobacco use, unhealthy diet and obesity, physical inactivity, and harmful alcohol use using population-wide strategies.

People with cardiovascular disease or who are at high cardiovascular risk (due to the presence of one or more risk factors such as hypertension, diabetes, hyperlipidemia, or already established disease) need early detection and management wherein a machine learning model can be of great help.

"""

# -----------------------------------------------------------
# 💓 Heart Failure Classification Project
# -----------------------------------------------------------

# 📦 Import necessary libraries
import pandas as pd
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from collections import Counter
from sklearn.compose import ColumnTransformer
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, InputLayer
from sklearn.metrics import classification_report
from tensorflow.keras.utils import to_categorical
import numpy as np

# -----------------------------------------------------------
# 1️⃣ Load Data
# -----------------------------------------------------------
data = pd.read_csv("heart_failure.csv")

# -----------------------------------------------------------
# 2️⃣ Explore Data
# -----------------------------------------------------------
print(data.info())

# -----------------------------------------------------------
# 3️⃣ Check class distribution
# -----------------------------------------------------------
from collections import Counter
print("Classes and number of values in the dataset:", Counter(data["death_event"]))

# -----------------------------------------------------------
# 4️⃣ Extract labels (y)
# -----------------------------------------------------------
y = data["death_event"]

# -----------------------------------------------------------
# 5️⃣ Extract features (x)
# -----------------------------------------------------------
x = data[
    [
        "age",
        "anaemia",
        "creatinine_phosphokinase",
        "diabetes",
        "ejection_fraction",
        "high_blood_pressure",
        "platelets",
        "serum_creatinine",
        "serum_sodium",
        "sex",
        "smoking",
        "time",
    ]
]

# -----------------------------------------------------------
# 6️⃣ One-Hot Encoding for categorical features
# -----------------------------------------------------------
x = pd.get_dummies(x)

# -----------------------------------------------------------
# 7️⃣ Split data into train and test sets
# -----------------------------------------------------------
X_train, X_test, Y_train, Y_test = train_test_split(
    x, y, test_size=0.3, random_state=0
)

# -----------------------------------------------------------
# 8️⃣ Initialize ColumnTransformer with StandardScaler
# -----------------------------------------------------------
from sklearn.preprocessing import StandardScaler
ct = ColumnTransformer(
    [
        (
            "numeric",
            StandardScaler(),
            [
                "age",
                "creatinine_phosphokinase",
                "ejection_fraction",
                "platelets",
                "serum_creatinine",
                "serum_sodium",
                "time",
            ],
        )
    ],
    remainder="passthrough",
)

# -----------------------------------------------------------
# 9️⃣ Fit and transform training data
# -----------------------------------------------------------
X_train = ct.fit_transform(X_train)

# -----------------------------------------------------------
# 🔟 Transform test data
# -----------------------------------------------------------
X_test = ct.transform(X_test)

# -----------------------------------------------------------
# 1️⃣1️⃣ Encode labels with LabelEncoder
# -----------------------------------------------------------
le = LabelEncoder()

# -----------------------------------------------------------
# 1️⃣2️⃣ Fit-transform training labels
# -----------------------------------------------------------
Y_train = le.fit_transform(Y_train.astype(str))

# -----------------------------------------------------------
# 1️⃣3️⃣ Transform test labels
# -----------------------------------------------------------
Y_test = le.transform(Y_test.astype(str))

# -----------------------------------------------------------
# 1️⃣4️⃣ Convert training labels to categorical (one-hot)
# -----------------------------------------------------------
Y_train = to_categorical(Y_train)

# -----------------------------------------------------------
# 1️⃣5️⃣ Convert test labels to categorical (one-hot)
# -----------------------------------------------------------
Y_test = to_categorical(Y_test)

# -----------------------------------------------------------
# 1️⃣6️⃣ Initialize Sequential model
# -----------------------------------------------------------
model = Sequential()

# -----------------------------------------------------------
# 1️⃣7️⃣ Add Input layer
# -----------------------------------------------------------
model.add(InputLayer(input_shape=(X_train.shape[1],)))

# -----------------------------------------------------------
# 1️⃣8️⃣ Add Hidden layer (12 neurons, ReLU)
# -----------------------------------------------------------
model.add(Dense(12, activation="relu"))

# -----------------------------------------------------------
# 1️⃣9️⃣ Add Output layer (2 neurons, softmax)
# -----------------------------------------------------------
model.add(Dense(2, activation="softmax"))

# -----------------------------------------------------------
# 2️⃣0️⃣ Compile model
# -----------------------------------------------------------
model.compile(
    loss="categorical_crossentropy",
    optimizer="adam",
    metrics=["accuracy"]
)

# -----------------------------------------------------------
# 2️⃣1️⃣ Train the model
# -----------------------------------------------------------
model.fit(X_train, Y_train, epochs=100, batch_size=16, verbose=1)

# -----------------------------------------------------------
# 2️⃣2️⃣ Evaluate the model
# -----------------------------------------------------------
loss, acc = model.evaluate(X_test, Y_test, verbose=0)
print("\n📊 Evaluation Results:")
print("Loss:", loss)
print("Accuracy:", acc)

# -----------------------------------------------------------
# 2️⃣3️⃣ Generate predictions
# -----------------------------------------------------------
y_estimate = model.predict(X_test, verbose=0)

# -----------------------------------------------------------
# 2️⃣4️⃣ Get predicted classes (argmax)
# -----------------------------------------------------------
y_estimate = np.argmax(y_estimate, axis=1)

# -----------------------------------------------------------
# 2️⃣5️⃣ Get true classes
# -----------------------------------------------------------
y_true = np.argmax(Y_test, axis=1)

# -----------------------------------------------------------
# 2️⃣6️⃣ Print classification report
# -----------------------------------------------------------
print("\n🧾 Classification Report:")
print(classification_report(y_true, y_estimate))
