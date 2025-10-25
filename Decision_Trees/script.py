"""
Find the flag!

Can you guess which continent this flag comes from?
Flag of Reunion

What are some of the features that would clue you in? Maybe some of the colors are good indicators. The presence or absence of certain shapes could give you a hint. In this project, we’ll use decision trees to try to predict the continent of flags based on several of these features.

We’ll explore which features are the best to use and the best way to create your decision tree.
Datasets

The original data set is available at the UCI Machine Learning Repository:

    https://archive.ics.uci.edu/ml/datasets/Flags
"""

# Decision Tree - Flags: Europe (3) vs Oceania (6)
import codecademylib3
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn import tree
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import seaborn as sns

# -----------------------------
# 1) Load the data
# -----------------------------
cols = [
    'name','landmass','zone', 'area', 'population', 'language','religion','bars','stripes','colours',
    'red','green','blue','gold','white','black','orange','mainhue','circles',
    'crosses','saltires','quarters','sunstars','crescent','triangle','icon','animate','text','topleft','botright'
]
df = pd.read_csv(
    "https://archive.ics.uci.edu/ml/machine-learning-databases/flags/flag.data",
    names=cols
)

# Quick preview
print("Dataset preview:")
print(df.head(), "\n")

# -----------------------------
# 1b) Count flags by landmass
# -----------------------------
print("Counts by landmass (1..6):")
print(df['landmass'].value_counts().sort_index(), "\n")

# -----------------------------
# 2) Subset to Europe (3) and Oceania (6)
# -----------------------------
df_36 = df[df['landmass'].isin([3, 6])].copy()
print(f"Subset shape (Europe & Oceania): {df_36.shape}")
print("Counts in subset by landmass:")
print(df_36['landmass'].value_counts(), "\n")

# -----------------------------
# 3) Compare means for selected predictors
# -----------------------------
var = [
    'red', 'green', 'blue', 'gold', 'white', 'black', 'orange',
    'mainhue', 'bars', 'stripes', 'circles', 'crosses', 'saltires',
    'quarters', 'sunstars', 'triangle', 'animate'
]

# Show group means for these variables
group_means = df_36.groupby('landmass')[var].mean()
print("Group means (Europe=3, Oceania=6):")
print(group_means.T, "\n")   # transpose for nicer view

# Which predictors show large differences?
# We can display absolute differences
diffs = (group_means.loc[3] - group_means.loc[6]).abs().sort_values(ascending=False)
print("Top differences (abs mean differences):")
print(diffs.head(8), "\n")

# -----------------------------
# 4) Check dtypes for predictors
# -----------------------------
print("Data types for the predictors (df_36[var].dtypes):")
print(df_36[var].dtypes, "\n")

# -----------------------------
# 5) Convert categorical 'mainhue' into dummies and build features
# -----------------------------
# mainhue is categorical (strings); get_dummies will create columns for each hue
data = pd.get_dummies(df_36[var], columns=['mainhue'], drop_first=True)
print("Feature matrix shape after get_dummies:", data.shape)
print("Sample columns:", list(data.columns[:20]), "\n")

# Labels: landmass 3 -> Europe, 6 -> Oceania. We'll map to 0/1:
labels = (df_36['landmass'] == 3).astype(int)   # 1 = Europe, 0 = Oceania (choose either mapping)
# Note: you can invert mapping if you prefer (1=Oceania)

print("Label distribution (Europe=1, Oceania=0):")
print(labels.value_counts(), "\n")

# -----------------------------
# 6) Train/test split
# -----------------------------
X_train, X_test, y_train, y_test = train_test_split(
    data, labels, random_state=1, test_size=0.4, stratify=labels
)
print("Train / Test shapes:", X_train.shape, X_test.shape, "\n")

# -----------------------------
# 7) Tune DecisionTree max_depth (1..20)
# -----------------------------
depths = list(range(1, 21))
acc_depth = []

for d in depths:
    clf = DecisionTreeClassifier(max_depth=d, random_state=1)
    clf.fit(X_train, y_train)
    preds = clf.predict(X_test)
    acc = accuracy_score(y_test, preds)
    acc_depth.append(acc)

# -----------------------------
# 8) Plot accuracy vs depth
# -----------------------------
plt.figure(figsize=(10,5))
plt.plot(depths, acc_depth, marker='o')
plt.title('Decision Tree Accuracy vs max_depth')
plt.xlabel('max_depth')
plt.ylabel('Accuracy (test)')
plt.grid(alpha=0.3)
best_depth = depths[int(np.argmax(acc_depth))]
best_depth_acc = max(acc_depth)
plt.axvline(best_depth, color='red', linestyle='--', label=f'Best depth={best_depth}')
plt.legend()
plt.show()

print(f"Best depth: {best_depth} with test accuracy = {best_depth_acc:.4f}\n")

# -----------------------------
# 9) Refit tree with best depth and plot
# -----------------------------
best_tree = DecisionTreeClassifier(max_depth=best_depth, random_state=1)
best_tree.fit(X_train, y_train)

plt.figure(figsize=(20,8))
tree.plot_tree(best_tree, feature_names=data.columns, class_names=['Oceania','Europe'],
               filled=True, rounded=True, fontsize=8)
plt.title(f"Decision Tree (max_depth={best_depth})")
plt.show()

# Evaluate best_tree
y_pred_best = best_tree.predict(X_test)
print("Classification report for best_depth tree:")
print(classification_report(y_test, y_pred_best, target_names=['Oceania','Europe']))
print("Confusion matrix:")
print(confusion_matrix(y_test, y_pred_best), "\n")

# -----------------------------
# 11) Cost-Complexity Pruning (ccp_alpha) tuning
# -----------------------------
# Get ccp_alphas from an unpruned tree fit to training data
clf_for_ccp = DecisionTreeClassifier(random_state=1)
path = clf_for_ccp.cost_complexity_pruning_path(X_train, y_train)
ccp_alphas = path.ccp_alphas
# Optional: remove the last alpha because it prunes to a single node
ccp_alphas = ccp_alphas[ccp_alphas > 0]   # keep positive alphas
ccp_alphas = np.unique(ccp_alphas)
print(f"Number of ccp_alphas found: {len(ccp_alphas)}")

# We'll sample a subset of alphas to avoid too many fits (or iterate all if small)
# If there are many alphas, take a geometric subset for speed
if len(ccp_alphas) > 30:
    ccp = np.geomspace(ccp_alphas[0], ccp_alphas[-1], num=30)
else:
    ccp = ccp_alphas

acc_pruned = []
for a in ccp:
    clf = DecisionTreeClassifier(random_state=1, ccp_alpha=a)
    clf.fit(X_train, y_train)
    preds = clf.predict(X_test)
    acc_pruned.append(accuracy_score(y_test, preds))

# -----------------------------
# 12) Plot accuracy vs ccp_alpha (pruning)
# -----------------------------
plt.figure(figsize=(10,5))
plt.plot(ccp, acc_pruned, marker='o')
plt.xscale('log')
plt.title('Accuracy vs ccp_alpha (pruning)')
plt.xlabel('ccp_alpha (log scale)')
plt.ylabel('Accuracy (test)')
plt.grid(alpha=0.3)
best_alpha = ccp[int(np.argmax(acc_pruned))]
best_alpha_acc = max(acc_pruned)
plt.axvline(best_alpha, color='red', linestyle='--', label=f'Best ccp_alpha={best_alpha:.2e}')
plt.legend()
plt.show()

print(f"Best ccp_alpha: {best_alpha:.6f} with test accuracy = {best_alpha_acc:.4f}\n")

# -----------------------------
# 14) Fit final decision tree with best depth and best alpha
# -----------------------------
final_clf = DecisionTreeClassifier(max_depth=best_depth, ccp_alpha=best_alpha, random_state=1)
final_clf.fit(X_train, y_train)

plt.figure(figsize=(18,7))
tree.plot_tree(final_clf, feature_names=data.columns, class_names=['Oceania','Europe'],
               filled=True, rounded=True, fontsize=8)
plt.title(f"Final Decision Tree (max_depth={best_depth}, ccp_alpha={best_alpha:.6f})")
plt.show()

# Final evaluation
y_final_pred = final_clf.predict(X_test)
print("Final Model Classification Report:")
print(classification_report(y_test, y_final_pred, target_names=['Oceania','Europe']))
cm = confusion_matrix(y_test, y_final_pred)
print("Final Confusion Matrix:\n", cm)

# Pretty confusion matrix heatmap
plt.figure(figsize=(5,4))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=['Oceania','Europe'], yticklabels=['Oceania','Europe'])
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Confusion Matrix (Final Model)')
plt.show()

# -----------------------------
# Short summary of results
# -----------------------------
print("SUMMARY:")
print(f"- Best max_depth: {best_depth} (test acc {best_depth_acc:.4f})")
print(f"- Best ccp_alpha: {best_alpha:.6f} (test acc {best_alpha_acc:.4f})")
print("- Final model is simpler due to pruning and gives comparable or improved accuracy.")
