import numpy as np
from sklearn.linear_model import LinearRegression
import statsmodels.api as sm

X = np.array([[1, 1], [1, 2], [2, 2], [2, 3]])
# y = 1 * x_0 + 2 * x_1 + 3
y = np.dot(X, np.array([1, 2])) + 3
print("y: ", y)
reg = LinearRegression().fit(X, y)
print("score: ", reg.score(X, y))

print("coef:" , reg.coef_)
#array([1., 2.])
reg.intercept_

print(reg.predict(np.array([[3, 5]])))
#array([16.])

# statsmodel
#X = df["RM"]
#y = target["MEDV"]

# Note the difference in argument order
model = sm.OLS(y, X).fit()
predictions = model.predict(X) # make the predictions by the model

# Print out the statistics
print(model.summary())