import numpy as np
import matplotlib.pylab as plt
from sklearn.linear_model import LinearRegression

import statsmodels.api as sm

def estimate_coef(x, y):
    # number of observations/points
    n = np.size(x)

    # mean of x and y vector
    m_x, m_y = np.mean(x), np.mean(y)
    print("X mean: ", m_x)
    print("Y mean: ", m_y)

    # calculating cross-deviation and deviation about x
    SS_xy = np.sum(y * x) - n * m_y * m_x
    SS_xx = np.sum(x * x) - n * m_x * m_x

    # calculating regression coefficients
    b_1 = SS_xy / SS_xx
    b_0 = m_y - b_1 * m_x

    return (b_0, b_1)


def plot_regression_line(x, y, b):
    # plotting the actual points as scatter plot
    plt.scatter(x, y, color="m",
                marker="o", s=30)

    # predicted response vector
    y_pred = b[0] + b[1] * x

    # plotting the regression line
    plt.plot(x, y_pred, color="g")

    # putting labels
    plt.xlabel('x')
    plt.ylabel('y')

    # function to show plot
    plt.show()



x = np.array( [2666, 2500, 2833, 2200, 3000, 2667, 2930, 2266, 2400, 2833, 2933, 1800, 3067, 1900, 2300, 3400, 2600])
y = np.array( [173, 170, 118, 315, 269, 206, 254, 196, 197, 94, 260, 70, 1025, 534, 571, 102, 290])

# estimating coefficients
b = estimate_coef(x, y)
print("Estimated coefficients:\nb_0 = {}  \
      \nb_1 = {}".format(b[0], b[1]))

# plotting regression line
plot_regression_line(x, y, b)