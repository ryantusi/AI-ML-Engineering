"""
Predict Baseball Strike Zones With Machine Learning

Support Vector Machines are powerful machine learning models that can make complex decision boundaries. An SVM’s decision boundary can twist and curve to accommodate the training data.

In this project, we will use an SVM trained using a baseball dataset to find the decision boundary of the strike zone.
A batter standing in front of the plate with the strike zone outlined.

The strike zone can be thought of as a decision boundary that determines whether or not a pitch is a strike or a ball. There is a strict definition of the strike zone — in practice, however, it will vary depending on the umpire or the player at bat.

Let’s use our knowledge of SVMs to find the real strike zone of several baseball players.
"""

# ⚾ Predict Baseball Strike Zones with SVM
# -----------------------------------------
# By Ryan Tusi

import codecademylib3_seaborn
import matplotlib.pyplot as plt
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from svm_visualization import draw_boundary
from players import aaron_judge, jose_altuve, david_ortiz

fig, ax = plt.subplots()

# 1. Explore dataset
print("Columns:\n", aaron_judge.columns)

# 2. Check pitch descriptions
print("\nUnique descriptions:\n", aaron_judge.description.unique())

# 3. Look at pitch types
print("\nUnique pitch types:\n", aaron_judge.type.unique())

# 4. Map strikes and balls to 1 and 0
aaron_judge['type'] = aaron_judge['type'].map({'S': 1, 'B': 0})

# 5. Confirm mapping worked
print("\nMapped type column:\n", aaron_judge['type'].head())

# 6. Check one location column
print("\nPlate X column sample:\n", aaron_judge['plate_x'].head())

# 7. Drop NaN values
aaron_judge = aaron_judge.dropna(subset=['plate_x', 'plate_z', 'type'])

# 8. Plot pitches
plt.scatter(
    x=aaron_judge['plate_x'],
    y=aaron_judge['plate_z'],
    c=aaron_judge['type'],
    cmap=plt.cm.coolwarm,
    alpha=0.25
)

plt.title("Aaron Judge Strike Zone (Raw Data)")
plt.xlabel("Horizontal Location (plate_x)")
plt.ylabel("Vertical Location (plate_z)")

# 9. Split data into training and validation sets
training_set, validation_set = train_test_split(aaron_judge, random_state=1)

# 10. Create SVM classifier
classifier = SVC(kernel='rbf')

# 11. Train classifier
classifier.fit(training_set[['plate_x', 'plate_z']], training_set['type'])

# 12. Visualize decision boundary
draw_boundary(ax, classifier)

# 13. Print accuracy
print("\n🎯 Accuracy (default params):")
print(classifier.score(validation_set[['plate_x', 'plate_z']], validation_set['type']))

plt.show()

# 14. Try gamma=100, C=100 (overfit example)
classifier_overfit = SVC(kernel='rbf', gamma=100, C=100)
classifier_overfit.fit(training_set[['plate_x', 'plate_z']], training_set['type'])
print("\n💥 Overfit model accuracy (gamma=100, C=100):")
print(classifier_overfit.score(validation_set[['plate_x', 'plate_z']], validation_set['type']))

# 15. Find best gamma & C
best_score = 0
best_params = (None, None)
for gamma in [0.1, 0.5, 1, 3, 5, 10]:
    for C in [0.1, 1, 3, 5, 10]:
        model = SVC(kernel='rbf', gamma=gamma, C=C)
        model.fit(training_set[['plate_x', 'plate_z']], training_set['type'])
        score = model.score(validation_set[['plate_x', 'plate_z']], validation_set['type'])
        if score > best_score:
            best_score = score
            best_params = (gamma, C)
print(f"\n🏆 Best SVM Accuracy: {best_score*100:.2f}% using gamma={best_params[0]}, C={best_params[1]}")

# 16. Compare other players (Jose Altuve & David Ortiz)
def visualize_strike_zone(player_df, name):
    fig, ax = plt.subplots()
    print(f"\nAnalyzing {name}...")

    player_df['type'] = player_df['type'].map({'S': 1, 'B': 0})
    player_df = player_df.dropna(subset=['plate_x', 'plate_z', 'type'])

    training_set, validation_set = train_test_split(player_df, random_state=1)
    model = SVC(kernel='rbf', gamma=best_params[0], C=best_params[1])
    model.fit(training_set[['plate_x', 'plate_z']], training_set['type'])

    plt.scatter(x=player_df['plate_x'], y=player_df['plate_z'],
                c=player_df['type'], cmap=plt.cm.coolwarm, alpha=0.25)

    draw_boundary(ax, model)
    ax.set_ylim(-2, 6)
    ax.set_xlim(-3, 3)
    plt.title(f"{name} Strike Zone")
    plt.xlabel("plate_x")
    plt.ylabel("plate_z")
    print(f"{name} Accuracy:", model.score(validation_set[['plate_x', 'plate_z']], validation_set['type']))
    plt.show()

# Compare Altuve & Ortiz
visualize_strike_zone(jose_altuve, "Jose Altuve")
visualize_strike_zone(david_ortiz, "David Ortiz")

# 17. Try additional features (comment out draw_boundary since it's 2D only)
aaron_judge = aaron_judge.dropna(subset=['plate_x', 'plate_z', 'strikes', 'type'])
training_set, validation_set = train_test_split(aaron_judge, random_state=1)
classifier_extra = SVC(kernel='rbf', gamma=3, C=1)
classifier_extra.fit(training_set[['plate_x', 'plate_z', 'strikes']], training_set['type'])
print("\n📈 Accuracy with extra feature (plate_x, plate_z, strikes):")
print(classifier_extra.score(validation_set[['plate_x', 'plate_z', 'strikes']], validation_set['type']))
