"""
Perceptron Logic Gates

In this project, we will use perceptrons to model the fundamental building blocks of computers — logic gates.

diagrams of AND, OR, and XOR gates

For example, the table below shows the results of an AND gate. Given two inputs, an AND gate will output a 1 only if both inputs are a 1:
Input 1 	Input 2 	Output
0 	0 	0
0 	1 	0
1 	0 	0
1 	1 	1

We’ll discuss how an AND gate can be thought of as linearly separable data and train a perceptron to perform AND.

We’ll also investigate an XOR gate — a gate that outputs a 1 only if one of the inputs is a 1:
Input 1 	Input 2 	Output
0 	0 	0
0 	1 	1
1 	0 	1
1 	1 	0

We’ll think about why an XOR gate isn’t linearly separable and show how a perceptron fails to learn XOR.
"""

# Perceptron Logic Gates Project 🧠
# By Ryan Tusi — Codecademy Portfolio Project

import codecademylib3_seaborn
from sklearn.linear_model import Perceptron
import matplotlib.pyplot as plt
import numpy as np
from itertools import product

# ==========================================
# 1. Creating AND gate data
# ==========================================
data = [[0, 0], [0, 1], [1, 0], [1, 1]]

# ==========================================
# 2. AND labels
# ==========================================
labels = [0, 0, 0, 1]

# ==========================================
# 3. Visualize AND data
# ==========================================
plt.figure(figsize=(5, 4))
plt.scatter([point[0] for point in data],
            [point[1] for point in data],
            c=labels,
            cmap='coolwarm')
plt.title("AND Gate Data Points")
plt.xlabel("Input A")
plt.ylabel("Input B")
plt.show()

# ==========================================
# 4. Build Perceptron
# ==========================================
classifier = Perceptron(max_iter=40, random_state=22)

# ==========================================
# 5. Train the Perceptron
# ==========================================
classifier.fit(data, labels)

# ==========================================
# 6. Check accuracy
# ==========================================
print("AND Gate Accuracy:", classifier.score(data, labels))

# ==========================================
# 7. Test XOR gate (not linearly separable)
# ==========================================
labels_xor = [0, 1, 1, 0]
classifier.fit(data, labels_xor)
print("XOR Gate Accuracy:", classifier.score(data, labels_xor))

# ==========================================
# 8. Test OR gate (linearly separable)
# ==========================================
labels_or = [0, 1, 1, 1]
classifier.fit(data, labels_or)
print("OR Gate Accuracy:", classifier.score(data, labels_or))

# ==========================================
# 9. Decision function (back to AND gate)
# ==========================================
classifier.fit(data, [0, 0, 0, 1])  # Reset to AND
print("\nDecision Function Distances:")
print(classifier.decision_function([[0, 0], [1, 1], [0.5, 0.5]]))

# ==========================================
# 10. Create grid of points
# ==========================================
x_values = np.linspace(0, 1, 100)
y_values = np.linspace(0, 1, 100)

# ==========================================
# 11. Create all combinations
# ==========================================
point_grid = list(product(x_values, y_values))

# ==========================================
# 12. Get distances from decision boundary
# ==========================================
distances = classifier.decision_function(point_grid)

# ==========================================
# 13. Get absolute distances
# ==========================================
abs_distances = [abs(pt) for pt in distances]

# ==========================================
# 14. Reshape into 100x100 grid
# ==========================================
distances_matrix = np.reshape(abs_distances, (100, 100))

# ==========================================
# 15. Draw heatmap
# ==========================================
plt.figure(figsize=(6, 5))
heatmap = plt.pcolormesh(x_values, y_values, distances_matrix, cmap='viridis')
plt.colorbar(heatmap)
plt.title("Perceptron Decision Boundary for AND Gate")
plt.xlabel("Input A")
plt.ylabel("Input B")
plt.show()

# ==========================================
# 16. Try visualizing for OR and XOR
# ==========================================
# ---- OR Gate ----
classifier.fit(data, labels_or)
distances_or = classifier.decision_function(point_grid)
abs_or = [abs(pt) for pt in distances_or]
distances_matrix_or = np.reshape(abs_or, (100, 100))

plt.figure(figsize=(6, 5))
heatmap = plt.pcolormesh(x_values, y_values, distances_matrix_or, cmap='plasma')
plt.colorbar(heatmap)
plt.title("Perceptron Decision Boundary for OR Gate")
plt.xlabel("Input A")
plt.ylabel("Input B")
plt.show()

# ---- XOR Gate ----
classifier.fit(data, labels_xor)
distances_xor = classifier.decision_function(point_grid)
abs_xor = [abs(pt) for pt in distances_xor]
distances_matrix_xor = np.reshape(abs_xor, (100, 100))

plt.figure(figsize=(6, 5))
heatmap = plt.pcolormesh(x_values, y_values, distances_matrix_xor, cmap='inferno')
plt.colorbar(heatmap)
plt.title("Perceptron Decision Boundary for XOR Gate (Non-Linearly Separable)")
plt.xlabel("Input A")
plt.ylabel("Input B")
plt.show()
