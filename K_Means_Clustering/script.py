"""
Handwriting Recognition using K-Means

The U.S. Postal Service has been using machine learning and scanning technologies since 1999. Because its postal offices have to look at roughly half a billion pieces of mail every day, they have done extensive research and developed very efficient algorithms for reading and understanding addresses. And not only the post office:

    ATMs can recognize handwritten bank checks
    Evernote can recognize handwritten task lists
    Expensify can recognize handwritten receipts

But how do they do it?

In this project, you will be using K-means clustering (the algorithm behind this magic) and scikit-learn to cluster images of handwritten digits.
"""

import codecademylib3_seaborn
import numpy as np
from matplotlib import pyplot as plt
from sklearn import datasets
from sklearn.cluster import KMeans

# -----------------------------
# 1. Load the Digits Dataset
# -----------------------------
digits = datasets.load_digits()
print("\n✅ Digits dataset loaded successfully!\n")

# View description
print("Dataset Description:\n")
print(digits.DESCR[:500])  # showing part of description for clarity

# -----------------------------
# 2. Inspect Data
# -----------------------------
print("\nSample data point (first entry):\n", digits.data[0])
print("\nTarget labels:\n", digits.target[:20])

# -----------------------------
# 3. Visualize sample images
# -----------------------------
plt.gray()
plt.matshow(digits.images[100])
plt.title(f"Digit at Index 100 → {digits.target[100]}")
plt.show()

# Optional: Display a grid of 64 images
fig = plt.figure(figsize=(6, 6))
fig.subplots_adjust(left=0, right=1, bottom=0, top=1, hspace=0.05, wspace=0.05)

for i in range(64):
    ax = fig.add_subplot(8, 8, i + 1, xticks=[], yticks=[])
    ax.imshow(digits.images[i], cmap=plt.cm.binary, interpolation='nearest')
    ax.text(0, 7, str(digits.target[i]), color='green', fontsize=8)

plt.suptitle("Sample Digits from Dataset")
plt.show()

# -----------------------------
# 4. Build and Fit K-Means Model
# -----------------------------
k = 10  # 10 digits → 10 clusters
model = KMeans(n_clusters=k, random_state=42)
model.fit(digits.data)

print("\n✅ K-Means model trained successfully with 10 clusters.\n")

# -----------------------------
# 5. Visualize Cluster Centers
# -----------------------------
fig = plt.figure(figsize=(8, 3))
fig.suptitle("Cluster Center Images", fontsize=14, fontweight='bold')

for i in range(k):
    ax = fig.add_subplot(2, 5, i + 1)
    ax.imshow(model.cluster_centers_[i].reshape((8, 8)), cmap=plt.cm.binary)
    ax.set_title(f"Cluster {i}", fontsize=8)
    ax.axis('off')

plt.show()

# -----------------------------
# 6. Test with Custom Digits
# -----------------------------
# You can replace these arrays with your own test samples from test.html
new_samples = np.array([
    [0.0, 0.0, 5.0, 13.0, 9.0, 1.0, 0.0, 0.0, 0.0, 0.0, 13.0, 15.0, 10.0, 15.0, 5.0, 0.0,
     0.0, 3.0, 15.0, 2.0, 0.0, 11.0, 8.0, 0.0, 0.0, 4.0, 12.0, 0.0, 0.0, 8.0, 8.0, 0.0,
     0.0, 5.0, 8.0, 0.0, 0.0, 9.0, 8.0, 0.0, 0.0, 4.0, 11.0, 0.0, 1.0, 12.0, 7.0, 0.0,
     0.0, 2.0, 14.0, 5.0, 10.0, 12.0, 0.0, 0.0, 0.0, 0.0, 6.0, 13.0, 10.0, 0.0, 0.0, 0.0],
    [0.0, 0.0, 0.0, 12.0, 13.0, 5.0, 0.0, 0.0, 0.0, 0.0, 0.0, 11.0, 16.0, 9.0, 0.0, 0.0,
     0.0, 0.0, 3.0, 16.0, 16.0, 6.0, 0.0, 0.0, 0.0, 3.0, 15.0, 16.0, 15.0, 3.0, 0.0, 0.0,
     0.0, 7.0, 15.0, 13.0, 14.0, 7.0, 0.0, 0.0, 0.0, 1.0, 3.0, 15.0, 16.0, 5.0, 0.0, 0.0,
     0.0, 0.0, 0.0, 9.0, 16.0, 11.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
])

# Predict which cluster each new sample belongs to
new_labels = model.predict(new_samples)
print("Predicted cluster labels for new samples:", new_labels)

# -----------------------------
# 7. Interpret the Results (Cluster-to-Digit Mapping)
# -----------------------------
print("\n🧠 Predicted digits (based on visual mapping): ", end="")

for label in new_labels:
    if label == 0:
        print(0, end='')
    elif label == 1:
        print(9, end='')
    elif label == 2:
        print(2, end='')
    elif label == 3:
        print(1, end='')
    elif label == 4:
        print(6, end='')
    elif label == 5:
        print(8, end='')
    elif label == 6:
        print(4, end='')
    elif label == 7:
        print(5, end='')
    elif label == 8:
        print(7, end='')
    elif label == 9:
        print(3, end='')

print("\n\n✅ Prediction complete.")
