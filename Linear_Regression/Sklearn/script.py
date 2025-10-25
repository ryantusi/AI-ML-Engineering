"""
Honey Production

Now that you have learned how linear regression works, let’s try it on an example of real-world data.

As you may have already heard, the honeybees are in a precarious state right now. You may have seen articles about the decline of the honeybee population for various reasons. You want to investigate this decline and how the trends of the past predict the future for the honeybees.

Note: All the tasks can be completed using Pandas or NumPy. Pick whichever one you prefer.
"""

import codecademylib3_seaborn
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.linear_model import LinearRegression

df = pd.read_csv("https://content.codecademy.com/programs/data-science-path/linear_regression/honeyproduction.csv")

prod_per_year = df.groupby("year").totalprod.mean().reset_index()
#print(prod_per_year.head())

X = prod_per_year["year"].values.reshape(-1,1)
y = prod_per_year["totalprod"]

regr = LinearRegression()
regr.fit(X, y)
y_predict = regr.predict(X)

X_future = np.array(range(2013,2051)).reshape(-1,1)
future_predict = regr.predict(X_future)

plt.scatter(X, y)
plt.plot(X, y_predict)
plt.plot(X_future, future_predict)
plt.show()

print("Slope (m):", regr.coef_)
print("Intercept :", regr.intercept_)