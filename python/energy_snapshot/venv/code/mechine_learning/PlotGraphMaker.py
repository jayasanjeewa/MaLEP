from sklearn import linear_model
import tkinter as tk
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import pandas as pd

# Reading the sample data from the file
df = pd.read_csv('training_spec_data.csv')

print('df: ', df)

# Building dependent and independent variable models
X = df[['processor_speed', 'no_of_cores', 'memory', 'reference_age_months']].astype(float)
Y = df['average_watts_fully_utilized'].astype(float)  # output variable (what we are trying to predict)

# Regression Model building
regr = linear_model.LinearRegression()
regr.fit(X, Y)

print('Intercept: \n', regr.intercept_)
print('Coefficients: \n', regr.coef_)

# tkinter GUI
root = tk.Tk()

canvas1 = tk.Canvas(root, width=500, height=300)
canvas1.pack()

# capturing GUI input
Intercept_result = ('Intercept: ', regr.intercept_)
label_Intercept = tk.Label(root, text=Intercept_result, justify='center')
canvas1.create_window(260, 220, window=label_Intercept)

# Streaming data to the UI component
Coefficients_result = ('Coefficients: ', regr.coef_)
label_Coefficients = tk.Label(root, text=Coefficients_result, justify='center')
canvas1.create_window(260, 240, window=label_Coefficients)

# Input data for prediction
label1 = tk.Label(root, text='Processor Speed: ')
canvas1.create_window(100, 100, window=label1)

entry1 = tk.Entry(root)  # create 1st entry box
canvas1.create_window(270, 100, window=entry1)

label2 = tk.Label(root, text='  Cores: ')
canvas1.create_window(120, 120, window=label2)

entry2 = tk.Entry(root)  # create 2nd entry box
canvas1.create_window(270, 120, window=entry2)

label3 = tk.Label(root, text=' Memory: ')
canvas1.create_window(140, 140, window=label3)

entry3 = tk.Entry(root)  # create 3rd entry box
canvas1.create_window(270, 140, window=entry3)

label4 = tk.Label(root, text=' Reference Age (months): ')
canvas1.create_window(160, 160, window=label4)

entry4 = tk.Entry(root)  # create 3rd entry box
canvas1.create_window(270, 160, window=entry4)




def values():
    global New_Processor_Speed
    New_Processor_Speed = float(entry1.get())

    global New_Processor_Cores
    New_Processor_Cores = float(entry2.get())

    global New_Total_Memory
    New_Total_Memory = float(entry3.get())

    global New_Reference_Age
    New_Reference_Age = float(entry4.get())



    prediction_result = ('Predicted Energy Consumption: ', regr.predict([[New_Processor_Speed, New_Processor_Cores, New_Total_Memory, New_Reference_Age]]),
                          ' watts')
    label_prediction = tk.Label(root, text=prediction_result, bg='green')
    canvas1.create_window(260, 280, window=label_prediction)


button1 = tk.Button(root, text='Predict Energy Consumption', command=values,
                    bg='orange')
canvas1.create_window(290, 170, window=button1)

# plot 1st scatter
figure3 = plt.Figure(figsize=(5, 4), dpi=100)
ax3 = figure3.add_subplot(111)
ax3.scatter(df['processor_speed'].astype(float), df['average_watts_fully_utilized'].astype(float), color='r')
scatter3 = FigureCanvasTkAgg(figure3, root)
scatter3.get_tk_widget().pack(side=tk.RIGHT, fill=tk.BOTH)
ax3.legend()
ax3.set_xlabel('processor_speed')
ax3.set_title('Processor Speed Vs. Energy Consumption')

# plot 2nd scatter
figure4 = plt.Figure(figsize=(5, 4), dpi=100)
ax4 = figure4.add_subplot(111)
ax4.scatter(df['no_of_cores'].astype(float), df['average_watts_fully_utilized'].astype(float), color='g')
scatter4 = FigureCanvasTkAgg(figure4, root)
scatter4.get_tk_widget().pack(side=tk.RIGHT, fill=tk.BOTH)
ax4.legend()
ax4.set_xlabel('no_of_cores')
ax4.set_title('cores Vs. Energy Consumption')

# plot 3rd scatter
figure5 = plt.Figure(figsize=(5, 4), dpi=100)
ax5 = figure5.add_subplot(111)
ax5.scatter(df['memory'].astype(float), df['average_watts_fully_utilized'].astype(float), color='g')
#scatter5 = FigureCanvasTkAgg(figure5, root)
#ßscatter5.get_tk_widget().pack(side=tk.RIGHT, fill=tk.BOTH)
ax5.legend()
ax5.set_xlabel('memory')
ax5.set_title('memory Vs. average watts fully utilized')

# plot 3rd scatter
figure6 = plt.Figure(figsize=(5, 4), dpi=100)
ax6 = figure6.add_subplot(111)
ax6.scatter(df['reference_age_months'].astype(float), df['average_watts_fully_utilized'].astype(float), color='g')
scatter6 = FigureCanvasTkAgg(figure6, root)
scatter6.get_tk_widget().pack(side=tk.RIGHT, fill=tk.BOTH)
ax6.legend()
ax6.set_xlabel('memory')
ax6.set_title('Reference Age Vs. average watts fully utilized')

root.mainloop()

