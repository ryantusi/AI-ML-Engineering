"""
Classifying Galaxies
Classifying Galaxies Using Convolutional Neural Networks

Around the clock, telescopes affixed to orbital satellites and ground-based observatories are taking millions of pictures of millions upon millions of celestial bodies. These data, of stars, planets and galaxies provide an invaluable resource to astronomers.

However, there is a bottleneck: until the data is annotated, it’s incredibly difficult for scientists to put it to good use. Additionally, scientists are usually interested in subsets of the data, like galaxies with unique characteristics.

In this project, you will build a neural network to classify deep-space galaxies. You will be using image data curated by Galaxy Zoo, a crowd-sourced project devoted to annotating galaxies in support of scientific discovery.

You will identify “odd” properties of galaxies. The data falls into four classes:

    [1,0,0,0] - Galaxies with no identifying characteristics.

Three regular galaxies. Each has a bright center, surrounded by a cloud of stars.

    [0,1,0,0] - Galaxies with rings.

Three ringed galaxies. Each has a bright center, surrounded by a ring of stars.

    [0,0,1,0] - Galactic mergers.

Three photos of galaxies. Each contains two bright orbs surrounded by clouds. These images show galaxies in the process of merging.

    [0,0,0,1] - “Other,” Irregular celestial bodies. Three photos of irregular celestial objects. Each are irregular clouds. The second has four bright orbs, seemingly suspensed above the cloud of stars.

"""

import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.model_selection import train_test_split
from utils import load_galaxy_data
from visualize import visualize_activations  # for BONUS visualization

# 1️⃣ Load and inspect the dataset
input_data, labels = load_galaxy_data()

print("Input data shape:", input_data.shape)
print("Labels shape:", labels.shape)
# Images are RGB → last dim = 3; labels are one-hot → last dim = 4

# 2️⃣ Split into training and validation sets
x_train, x_valid, y_train, y_valid = train_test_split(
    input_data,
    labels,
    test_size=0.20,
    stratify=labels,
    shuffle=True,
    random_state=222
)

# 3️⃣ Create ImageDataGenerator (normalize pixel values)
data_generator = ImageDataGenerator(rescale=1.0/255)

# 4️⃣ Create training and validation iterators
training_iterator = data_generator.flow(x_train, y_train, batch_size=5)
validation_iterator = data_generator.flow(x_valid, y_valid, batch_size=5)

# 5️⃣ Build CNN model
model = tf.keras.Sequential()
model.add(tf.keras.Input(shape=(128, 128, 3)))
model.add(tf.keras.layers.Conv2D(8, 3, strides=2, activation="relu"))
model.add(tf.keras.layers.MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
model.add(tf.keras.layers.Conv2D(8, 3, strides=2, activation="relu"))
model.add(tf.keras.layers.MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
model.add(tf.keras.layers.Flatten())
model.add(tf.keras.layers.Dense(16, activation="relu"))
model.add(tf.keras.layers.Dense(4, activation="softmax"))

# 6️⃣ Compile the model
model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
    loss=tf.keras.losses.CategoricalCrossentropy(),
    metrics=[
        tf.keras.metrics.CategoricalAccuracy(),
        tf.keras.metrics.AUC()
    ]
)

# 8️⃣ Display model summary
print("\nModel Summary:")
model.summary()

# 9️⃣ Train the model
print("\nTraining model... please wait ⏳")
history = model.fit(
    training_iterator,
    steps_per_epoch=len(x_train) / 5,
    epochs=8,
    validation_data=validation_iterator,
    validation_steps=len(x_valid) / 5
)

# 🔟 Evaluation
print("\nTraining complete ✅")
print("Final Accuracy:", history.history["categorical_accuracy"][-1])
print("Final Validation Accuracy:", history.history["val_categorical_accuracy"][-1])
print("Final AUC:", history.history["auc"][-1])
print("Final Validation AUC:", history.history["val_auc"][-1])

# 1️⃣1️⃣ Optional save
model.save("galaxy_classifier_model.h5")
print("Model saved as galaxy_classifier_model.h5 🪐")

# 1️⃣2️⃣ BONUS: Visualize activations (feature maps)
visualize_activations(model, validation_iterator)
