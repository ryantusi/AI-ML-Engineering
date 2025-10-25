"""
Classify Raisins with Hyperparameter Tuning!

In this project, you’ll use the different techniques you have learned in this unit to classify different types of raisins. The dataset has been posted on Kaggle by Murat Koklu, a researcher who has studied different raisin grain types using machine learning methods.

There are two raisin grain types in this dataset, Kecimen and Besni and seven numerical predictor variables associated with each of the 900 samples in the data. You’re going to use this dataset to implement the two hyperparameter tuning methods we’ve covered in this module thusfar:

    Grid Search method to tune a Decision Tree Classifier
    Random Search method to tune a Logistic Regression Classifier

"""

# 1. Setup
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import RandomizedSearchCV

raisins = pd.read_csv('Raisin_Dataset.csv')
raisins.head()

# 2. Create predictor and target variables, X and y
X = raisins.drop('Class', axis=1)
y = raisins['Class']

# 3. Examine the dataset
print("Number of features:", X.shape[1])
print("Total number of samples:", len(y))
print("Samples belonging to class '1':", y.sum())

# 4. Split the data set into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state = 19)

# 5. Create a Decision Tree model
tree = DecisionTreeClassifier()

# 6. Dictionary of parameters for GridSearchCV
parameters = {'min_samples_split': [2,3,4], 'max_depth': [3,5,7]}

# 7. Create a GridSearchCV model
grid = GridSearchCV(tree, parameters)

# Fit the GridSearchCV model to the training data
grid.fit(X_train, y_train)

# 8. Print the model and hyperparameters obtained by GridSearchCV
print(grid.best_estimator_)

# Print best score
print(grid.best_score_)
# Print the accuracy of the final model on the test data
print(grid.score(X_test, y_test))

# 9. Print a table summarizing the results of GridSearchCV
df = pd.concat([pd.DataFrame(grid.cv_results_['params']), pd.DataFrame(grid.cv_results_['mean_test_score'], columns=['Score'])], axis=1)
print(df)

# 10. The logistic regression model
lr = LogisticRegression(solver = 'liblinear', max_iter = 1000)

# 11. Define distributions to choose hyperparameters from
from scipy.stats import uniform
distributions = {'penalty': ['l1', 'l2'], 'C': uniform(loc=0, scale=100)}

# 12. Create a RandomizedSearchCV model
clf = RandomizedSearchCV(lr, distributions, n_iter=8)

# Fit the random search model
clf.fit(X_train, y_train)

# 13. Print best esimator and best score
print(clf.best_estimator_)
print (clf.best_score_)

# Print a table summarizing the results of RandomSearchCV
df = pd.concat([pd.DataFrame(clf.cv_results_['params']), pd.DataFrame(clf.cv_results_['mean_test_score'], columns=['Accuracy'])] ,axis=1)
print(df.sort_values('Accuracy', ascending = False))

