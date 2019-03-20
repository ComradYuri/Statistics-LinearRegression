import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn import linear_model

# Setting up pandas so that it displays all columns instead of collapsing them
desired_width = 320
pd.set_option('display.width', desired_width)
np.set_printoptions(linewidth=desired_width)
pd.set_option('display.max_columns', 10)


df = pd.read_csv(
    "https://s3.amazonaws.com/codecademy-content/programs/data-science-path/linear_regression/honeyproduction.csv")
# print(df.head())

prod_per_year = df.groupby("year").totalprod.mean().reset_index()
prod_per_year["*_in_tonnes"] = prod_per_year.totalprod.apply(lambda x: round(x/1000))
print("Honey production per year")
print(prod_per_year)

X = prod_per_year.year.reset_index(drop=True)
# convert x to one column containing many rows instead of a dataframe with indexing
# This is used for the scikit learn to create the linear regression model
X = X.values.reshape(-1, 1)
# print(X)

y = prod_per_year.totalprod.reset_index(drop=True)
y = y.values.reshape(-1, 1)
# print(y)

regr = linear_model.LinearRegression()
regr.fit(X, y)
# slope and intercept of linear regression
print("\n\nSlope and intercept of linear regression model")
print(regr.coef_[0], regr.intercept_)

# values of linear regression line
# y = m*x + b
y_predict = [regr.coef_[0] * x + regr.intercept_ for x in X]

plt.scatter(X, y)
plt.plot(X, y_predict)
plt.show()
plt.close('all')

# Future honey production estimations
X_future = np.array(range(2013, 2050))
# print(X_future)
X_future = X_future.reshape(-1, 1)
# print(X_future)
future_predict = regr.predict(X_future)

print("Total honey production in 2050 is predicted to be {} kilo".format(int((regr.coef_*2050 + regr.intercept_)[0][0])))

plt.scatter(X, y)
plt.plot(X_future, future_predict)
plt.show()

