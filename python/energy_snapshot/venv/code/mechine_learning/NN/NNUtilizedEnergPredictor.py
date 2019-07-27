#https://towardsdatascience.com/deep-neural-networks-for-regression-problems-81321897ca33

from keras.callbacks import ModelCheckpoint
from keras.models import Sequential
from keras.layers import Dense, Activation, Flatten
from keras.utils.vis_utils import plot_model
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error
from matplotlib import pyplot as plt
import seaborn as sb
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')
warnings.filterwarnings('ignore', category=DeprecationWarning)
#from xgboost import XGBRegressor



def get_data():
    # get train data
    train_data_path = 'training_spec_data.csv'
    train = pd.read_csv(train_data_path)
#'average_watts_fully_utilized','processor_speed','reference_age_months','no_of_cores', 'memory'

    X = train[['average_watts_fully_utilized','processor_speed','reference_age_months','no_of_cores', 'memory']].astype(float)

    print(X)



    # get test data
    test_data_path = 'test_spec_data.csv'
    test = pd.read_csv(test_data_path)

    Y = test[['average_watts_fully_utilized','processor_speed','reference_age_months','no_of_cores', 'memory']].astype(float)

    print(Y)

    #return train, test
    return X, Y


def get_combined_data():
    # reading train data
    train, test = get_data()

    target = train.average_watts_fully_utilized
   # train.drop(['SalePrice'], axis=1, inplace=True)

    combined = train.append(test)
    #combined.reset_index(inplace=True)
    #combined.drop(['index', 'Id'], inplace=True, axis=1)
    return combined, target



def get_cols_with_no_nans(df,col_type):
    '''
    Arguments :
    df : The dataframe to process
    col_type :
          num : to only get numerical columns with no nans
          no_num : to only get nun-numerical columns with no nans
          all : to get any columns with no nans
    '''
    if (col_type == 'num'):
        predictors = df.select_dtypes(exclude=['object'])
    elif (col_type == 'no_num'):
        predictors = df.select_dtypes(include=['object'])
    elif (col_type == 'all'):
        predictors = df
    else :
        print('Error : choose a type (num, no_num, all)')
        return 0
    cols_with_no_nans = []
    for col in predictors.columns:
        if not df[col].isnull().any():
            cols_with_no_nans.append(col)
    return cols_with_no_nans


# Load train and test data into pandas DataFrames
train_data, test_data = get_data()

# Combine train and test data to process them together
combined, target = get_combined_data()

num_cols = get_cols_with_no_nans(combined, 'num')
cat_cols = get_cols_with_no_nans(combined, 'no_num')

print ('Number of numerical columns with no nan values :',len(num_cols))
print ('Number of nun-numerical columns with no nan values :',len(cat_cols))

print(combined.describe())

combined = combined[num_cols + cat_cols]
combined.hist(figsize = (12,10))
#plt.show()

#ßtrain_data = train_data[num_cols + cat_cols]
train_data['Target'] = target

C_mat = train_data.corr()
fig = plt.figure(figsize = (15,15))

sb.heatmap(C_mat, vmax = .8, square = True)
#ßplt.show()

train_raw_date = pd.read_csv('training_spec_data.csv')
sns.pairplot(train_raw_date[["average_watts_fully_utilized", "memory"]], diag_kind= "kde")


def oneHotEncode(df, colNames):
    for col in colNames:
        if (df[col].dtype == np.dtype('object')):
            dummies = pd.get_dummies(df[col], prefix=col)
            df = pd.concat([df, dummies], axis=1)

            # drop the encoded column
            df.drop([col], axis=1, inplace=True)
    return df


print('There were {} columns before encoding categorical features'.format(combined.shape[1]))
combined = oneHotEncode(combined, cat_cols)
print('There are {} columns after encoding categorical features'.format(combined.shape[1]))


def split_combined():
    global combined
    train = combined[:572]
    test = combined[572:]

    return train, test


train, test = split_combined()

NN_model = Sequential()

# The Input Layer :
print(train.shape)
NN_model.add(Dense(128, kernel_initializer='normal',input_dim = train.shape[1], activation='relu'))

# The Hidden Layers :
NN_model.add(Dense(256, kernel_initializer='normal',activation='relu'))
NN_model.add(Dense(256, kernel_initializer='normal',activation='relu'))
NN_model.add(Dense(256, kernel_initializer='normal',activation='relu'))

# The Output Layer :
NN_model.add(Dense(1, kernel_initializer='normal',activation='linear'))

# Compile the network :
NN_model.compile(loss='mean_absolute_error', optimizer='adam', metrics=['mean_absolute_error', 'accuracy'])
#ßNN_model.summary()



#print('config: ', NN_model.get_config())
#ßprint('weights: ', NN_model.get_weights())


checkpoint_name = 'Weights-{epoch:03d}--{val_loss:.5f}.hdf5'
checkpoint = ModelCheckpoint(checkpoint_name, monitor='val_loss', verbose = 1, save_best_only = True, mode ='auto')
callbacks_list = [checkpoint]

#train the model
history = NN_model.fit(train, target, epochs=500, batch_size=32, validation_split = 0.2, callbacks=callbacks_list)
print(history.history.keys())

#test_loss, test_acc = NN_model.evaluate(test, test_data.average_watts_fully_utilized)
#print(test_acc)
#ßprint(test_loss)


# Plot training & validation loss values

plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('Model loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['Train', 'Test'], loc='upper left')
#ßplt.show()


# Plot training & validation accuracy values
plt.plot(history.history['acc'])
plt.plot(history.history['val_acc'])
plt.title('Model accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend(['Train', 'Test'], loc='upper left')
#plt.show()


#NN_model.fit(train, target, epochs=50, batch_size=10, validation_split = 0.2, callbacks=callbacks_list)

#ß Load wights file of the best model :
#wights_file = 'Weights-478--18738.19831.hdf5' # choose the best checkpoint
#NN_model.load_weights(wights_file) # load it
#ßNN_model.compile(loss='mean_absolute_error', optimizer='adam', metrics=['mean_absolute_error'])

def make_submission(prediction, sub_name):
  test_file = pd.read_csv('test_spec_data.csv')
  my_submission = pd.DataFrame({'Actual':test_file.average_watts_fully_utilized,'Predicted':prediction,
                                'predict-actual':abs(test_file.average_watts_fully_utilized - prediction),
                                'Error Percentage':(abs(test_file.average_watts_fully_utilized - prediction)/ test_file.average_watts_fully_utilized) * 100})
  my_submission.to_csv('{}'.format(sub_name),index=False)
  print('A submission file has been made')

predictions = NN_model.predict(test)
make_submission(predictions[:,0],'submission(NN).csv')

plot_model(NN_model, to_file='model_plot.png', show_shapes=True, show_layer_names=True)



# trying another alog ,random forest

plot_model(NN_model, to_file='model_plot.png', show_shapes=True, show_layer_names=True)

train_X, val_X, train_y, val_y = train_test_split(train, target, test_size = 0.25, random_state = 14)

model = RandomForestRegressor()
model.fit(train_X,train_y)


# Get the mean absolute error on the validation data
predicted_prices = model.predict(val_X)
MAE = mean_absolute_error(val_y , predicted_prices)
print('Random forest validation MAE = ', MAE)

predicted_prices = model.predict(test)
make_submission(predicted_prices,'Submission(RF).csv')

#ßplot_model(model, to_file='RF_model_plot.png', show_shapes=True, show_layer_names=True)



nn_result = pd.read_csv('submission(NN).csv')
X = nn_result[['Error Percentage']].astype(float)

print(X.describe())

rf_result = pd.read_csv('submission(RF).csv')
Y = rf_result[['Error Percentage']].astype(float)

print(Y.describe())




