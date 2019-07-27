from pandas import DataFrame
from sklearn import linear_model
import tkinter as tk
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

#Server_Specs = {
    #'Processor_Speed': [2.75, 2.5, 2.5, 2.5, 2.5, 2.5, 2.5, 2.25, 2.25, 2.25, 2, 2, 2, 1.75, 1.75, 1.75, 1.75, 1.75, 1.75,
                     # 1.75, 1.75, 1.75, 1.75, 1.75],
    #'Processor_Cores': [5.3, 5.3, 5.3, 5.3, 5.4, 5.6, 5.5, 5.5, 5.5, 5.6, 5.7, 5.9, 6, 5.9, 5.8, 6.1, 6.2, 6.1, 6.1,
                         # 6.1, 5.9, 6.2, 6.2, 6.1],
    #'Energy_Consumption': [1464, 1394, 1357, 1293, 1256, 1254, 1234, 1195, 1159, 1167, 1130, 1075, 1047, 965, 943, 958,
                          #971, 949, 884, 866, 876, 822, 704, 719]
   # }

#Server_Specs = {
    #'Processor_Speed': [ 2666, 2500, 2833, 2200, 3000, 2667, 2930, 2266, 2266, 2266, 2400, 2833, 2933, 2933, 1800, 2200, 2200, 2200],
    #'Processor_Cores': [ 8, 8, 4, 16, 8, 8, 8, 8, 16, 32, 8, 4, 12, 24, 6, 96, 24, 48],
    #'Energy_Consumption': [173, 170, 118, 315, 269, 206, 254, 196, 386, 748, 197, 95.4, 260, 596, 70.3, 1111, 273, 518]
    #}

Server_Specs = {
    'System_Vendor': ['Supermicro', 'Acer Incorporated', 'Huawei Technologies', 'ZT Systems', 'Hewlett-Packard', 'Hewlett-Packard',
                      'Acer Incorporated', 'Acer Incorporated', 'Sugon', 'Huawei Technologies', 'Inspur Corporation', 'Inspur Corporation'],
    'System_Spec_Name': ['1022G-NTF', 'Gateway GT115 F1', 'RH2288 V2', '1224Ra Datacenter Server', 'ProLiant ML110 G4',
                         'Proliant DL580 G5', 'Gateway GR585 F1', 'Altos R360 F2', 'A620-G30(AMD EPYC 7501)', 'RH2285 V2',
                         'Inspur NF5280M5', 'Inspur Yingxin NF8480M5'],
    'Processor_Speed': [1600, 1800, 1800, 1800, 1800, 1860, 1860,  1900, 2000, 2000, 2100, 2100, 2100],#, 2133, 2200, 2260, 2266, 2267, 2300, 2330, 2400,  2500, 2533, 2600,
                        #2666, 2667, 2700, 2800, 2833, 2900, 2930, 2933, 3000,  3100, 3060, 3066, 3067, 3400, 3500, 3600, 3700],
    'Processor_Cores': [32, 6 ,16, 12, 32, 2, 16, 48, 16, 64, 16, 56, 112],
    'Energy_Consumption': [213, 70.3, 178, 178, 386, 117, 387, 524, 223, 307, 231, 454, 726]
    }

df = DataFrame(Server_Specs, columns=['Year', 'Month', 'Processor_Speed', 'Processor_Cores', 'Energy_Consumption'])

X = df[['Processor_Speed', 'Processor_Cores']].astype(
    float)  # here we have 2 input variables for multiple regression. If you just want to use one variable for simple linear regression, then use X = df['Interest_Rate'] for example.Alternatively, you may add additional variables within the brackets
Y = df['Energy_Consumption'].astype(float)  # output variable (what we are trying to predict)

# with sklearn
regr = linear_model.LinearRegression()
regr.fit(X, Y)

print('Intercept: \n', regr.intercept_)
print('Coefficients: \n', regr.coef_)

# tkinter GUI
root = tk.Tk()

canvas1 = tk.Canvas(root, width=500, height=300)
canvas1.pack()

# with sklearn
Intercept_result = ('Intercept: ', regr.intercept_)
label_Intercept = tk.Label(root, text=Intercept_result, justify='center')
canvas1.create_window(260, 220, window=label_Intercept)

# with sklearn
Coefficients_result = ('Coefficients: ', regr.coef_)
label_Coefficients = tk.Label(root, text=Coefficients_result, justify='center')
canvas1.create_window(260, 240, window=label_Coefficients)

# New_Interest_Rate label and input box
label1 = tk.Label(root, text='Type Server Speed: ')
canvas1.create_window(100, 100, window=label1)

entry1 = tk.Entry(root)  # create 1st entry box
canvas1.create_window(270, 100, window=entry1)

# New_Unemployment_Rate label and input box
label2 = tk.Label(root, text=' Type Processor Cores: ')
canvas1.create_window(120, 120, window=label2)

entry2 = tk.Entry(root)  # create 2nd entry box
canvas1.create_window(270, 120, window=entry2)


def values():
    global New_Processor_Speed  # our 1st input variable
    New_Processor_Speed = float(entry1.get())

    global New_Processor_Cores  # our 2nd input variable
    New_Processor_Cores = float(entry2.get())

    Prediction_result = ('Predicted Energy Consumption: ', regr.predict([[New_Processor_Speed, New_Processor_Cores]]))
    label_Prediction = tk.Label(root, text=Prediction_result, bg='orange')
    canvas1.create_window(260, 280, window=label_Prediction)


button1 = tk.Button(root, text='Predict Energy Consumption', command=values,
                    bg='orange')  # button to call the 'values' command above
canvas1.create_window(270, 150, window=button1)

# plot 1st scatter
figure3 = plt.Figure(figsize=(5, 4), dpi=100)
ax3 = figure3.add_subplot(111)
ax3.scatter(df['Processor_Speed'].astype(float), df['Energy_Consumption'].astype(float), color='r')
scatter3 = FigureCanvasTkAgg(figure3, root)
scatter3.get_tk_widget().pack(side=tk.RIGHT, fill=tk.BOTH)
ax3.legend()
ax3.set_xlabel('Processor Speed ')
ax3.set_title('Processor Speed Vs. Energy Consumption')

# plot 2nd scatter
figure4 = plt.Figure(figsize=(5, 4), dpi=100)
ax4 = figure4.add_subplot(111)
ax4.scatter(df['Processor_Cores'].astype(float), df['Energy_Consumption'].astype(float), color='g')
scatter4 = FigureCanvasTkAgg(figure4, root)
scatter4.get_tk_widget().pack(side=tk.RIGHT, fill=tk.BOTH)
ax4.legend()
ax4.set_xlabel('Processor_Cores')
ax4.set_title('Processor Cores Vs. Energy Consumption')

root.mainloop()