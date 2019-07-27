import numpy as np
import matplotlib.pylab as plt
from sklearn.linear_model import LinearRegression

import statsmodels.api as sm

x = [[2500, 56], [2666, 8], [2500, 8], [2833, 4], [2200, 16], [3000, 8], [2667, 8], [2930, 8]]
y = [385, 173, 170, 118, 315, 269, 206, 254] 



x, y = np.array(x), np.array(y)

print(x)

plt.scatter(x,y)
plt.show()

model = LinearRegression().fit(x, y)

r_sq = model.score(x, y)
print('coefficient of determination:', r_sq)

print('intercept:', model.intercept_)

print('slope:', model.coef_)

y_pred = model.predict(x)
#y_pred = model.intercept_ + np.sum(model.coef_ * x, axis=1)

print('predicted response:', y_pred, sep='\n')

#print(model.summary())

#x_new = np.arange(10).reshape((-1, 2))
#print('x_new: ', x_new)

#y_new = model.predict(x_new)
#print('y_new: ', y_new)
