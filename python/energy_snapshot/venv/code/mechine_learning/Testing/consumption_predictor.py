#https://datatofish.com/multiple-linear-regression-python/

from pandas import DataFrame
import matplotlib.pyplot as plt
from sklearn import linear_model
import statsmodels.api as sm
import tkinter as tk

Server_Spec = {
    'Server_Memory': [2.75, 2.5, 2.5, 2.5, 2.5, 2.5, 2.5, 2.25, 2.25, 2.25, 2, 2, 2, 1.75, 1.75, 1.75, 1.75, 1.75, 1.75,
                      1.75, 1.75, 1.75, 1.75, 1.75],
    'Cpu_Cores': [5.3, 5.3, 5.3, 5.3, 5.4, 5.6, 5.5, 5.5, 5.5, 5.6, 5.7, 5.9, 6, 5.9, 5.8, 6.1, 6.2, 6.1, 6.1,
                          6.1, 5.9, 6.2, 6.2, 6.1],
    'Utilized_Energy_Consumption': [1464, 1394, 1357, 1293, 1256, 1254, 1234, 1195, 1159, 1167, 1130, 1075, 1047, 965, 943, 958,
                          971, 949, 884, 866, 876, 822, 704, 719]
    }

df = DataFrame(Server_Spec, columns=['Server_Memory', 'Cpu_Cores', 'Utilized_Energy_Consumption'])

X = df[['Server_Memory',
        'Cpu_Cores']]  # here we have 2 variables for multiple regression. If you just want to use one variable for simple linear regression, then use X = df['Interest_Rate'] for example.Alternatively, you may add additional variables within the brackets
Y = df['Utilized_Energy_Consumption']

# with sklearn
regr = linear_model.LinearRegression()
regr.fit(X, Y)

print('Intercept: \n', regr.intercept_)
print('Coefficients: \n', regr.coef_)

# prediction with sklearn
New_Server_Memory = 2.75
New_Unemployment_Rate = 5.3
print('Predicted Energy Consumption: \n', regr.predict([[New_Server_Memory, New_Unemployment_Rate]]))

# with statsmodels
X = sm.add_constant(X)  # adding a constant

model = sm.OLS(Y, X).fit()
predictions = model.predict(X)

# tkinter GUI
root = tk.Tk()

canvas1 = tk.Canvas(root, width=1200, height=450)
canvas1.pack()

print_model = model.summary()
print(print_model)

plt.scatter(df['Server_Memory'], df['Utilized_Energy_Consumption'], color='red')
plt.title('Energy Consumption  Vs Server Memory', fontsize=14)
plt.xlabel('Server_Memory', fontsize=14)
plt.ylabel('Stock Index Price', fontsize=14)
plt.grid(True)
#plt.show()

plt.scatter(df['Cpu_Cores'], df['Utilized_Energy_Consumption'], color='green')
plt.title('Utilized Energy Consumption Vs Cpu Cores', fontsize=14)
plt.xlabel('CPU Cores', fontsize=14)
plt.ylabel('Utilized Energy Consumption', fontsize=14)
plt.grid(True)
#plt.show()