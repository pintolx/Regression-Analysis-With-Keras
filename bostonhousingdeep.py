#Regression Analysis with Keras
#The dataset is the Boston Housing Data
#You can learn more about the Boston house price dataset on the UCI Machine Learning Repository.
#In this project we try to estimate the mean squared error without any data transformations
#Loading important libraries
import numpy
from pandas import read_csv
from keras.layers import Dense
from keras.models import Sequential
from keras.wrappers.scikit_learn import KerasRegressor
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

#Loading the dataset
dataframe  = read_csv('housing.csv', delim_whitespace=True, header=None)
dataset = dataframe.values

#Separating inputs and outputs
X = dataset[:,0:13]
Y = dataset[:,13]

#Defining the baseline model 
def baseline_model():
    #create model
    model = Sequential()
    model.add(Dense(13, input_dim=13, kernel_initializer='normal', activation='relu'))
    model.add(Dense(1, kernel_initializer='normal')) #No activation because we are trying to predict the output as it is
    #Compile the model
    model.compile(loss='mean_squared_error', optimizer='adam')
    return model
#fix the random seed for reproducibility
seed = 7
numpy.random.seed(seed)

#Evaluating the model
estimator = KerasRegressor(build_fn=baseline_model, epochs=100, batch_size=5, verbose=0)

#Evaluating the model using 10 fold cross validation
kfold = KFold(n_splits=10, random_state=seed)
results = cross_val_score(estimator, X, Y, cv=kfold)
print("Baseline: %.2f (%.2f) MSE" %(results.mean(), results.std()))