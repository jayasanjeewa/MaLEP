from sklearn import linear_model
import tkinter as tk
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import pandas as pd

# Reading the sample data from the file
df = pd.read_csv('training_spec_data.csv')

#ßprint('df: ', df)

#average_watts_fully_utilized,average_watts_active_idle,no_of_cores,threads_per_core,processor_speed,memory,network_controllers,power_supplies_installed ,power_supply_rating,HW_available_date,HW_available_full_date,reference_age_months,age_months
# Building dependent and independent variable models
#X = df[['reference_age_months']].astype(float)
#X = df[['processor_speed', 'no_of_cores', 'memory','threads_per_core', 'power_supply_rating', 'reference_age_months']].astype(float)
X = df[['processor_speed','no_of_cores', 'reference_age_months']].astype(float)

Y = df['average_watts_active_idle'].astype(float)  # output variable (what we are trying to predict)

# Regression Model building
regr = linear_model.LinearRegression()
regr.fit(X, Y)

print('Intercept: \n', regr.intercept_)
print('Coefficients: \n', regr.coef_)

# capturing GUI input
Intercept_result = ('Intercept: ', regr.intercept_)
#ßprint (Intercept_result)


# Streaming data to the UI component
Coefficients_result = ('Coefficients: ', regr.coef_)


# Reading the sample data from the file
df_test = pd.read_csv('test_spec_data.csv')

#print('df: ', df)

total_error_percentage = 0

for row in df_test.itertuples():
    fully_utilized_power = row[1]
    idle_power = row[2]
    test_processor_threads = row[4]
    test_processor_speed = row[5]
    test_cores_count = row[3]
    test_memory = row[6]
    test_reference_age = row[12]
    test_power_rating = row[9]
    #ßprint(fully_utilized_power)

    predicted_idle_power = regr.predict([[test_processor_speed, test_cores_count, test_reference_age]])

    error_percentage = (abs(predicted_idle_power - idle_power)  / idle_power ) * 100
    total_error_percentage += error_percentage


    print (test_processor_speed,  ",", test_cores_count, ",",test_reference_age, ",", idle_power, ",", predicted_idle_power, ",", str(error_percentage))

print ("total_error_percentage", total_error_percentage)
print( "df_test.size", len(df_test))
print ("average_error_percentage", total_error_percentage / len(df_test))