#Boston Housing dataset using regression
#The goal of this project is to improve performance using the model tuning
# Regression Example With Boston Dataset: Standardized dataset and Larger model
#This is part 3
#Loading important libraries
import numpy
from pandas import read_csv
from keras.models import Sequential
from keras.layers import Dense
from keras.wrappers.scikit_learn import KerasRegressor
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

#Fixing reproducibility in the dataset
seed = 7
numpy.random.seed(seed)

#Loading the data 
dataframe = read_csv('housing.csv', delim_whitespace=True, header=None)
dataset = dataframe.values

#Separating inputs and outputs
X = dataset[:,0:13]
Y = dataset[:,13]

#Defining the baseline model
def baseline_model():
    #create model
    model = Sequential()
    model.add(Dense(13, input_dim=13, kernel_initializer='normal', activation='relu'))
    model.add(Dense(6, kernel_initializer='normal', activation='relu'))
    model.add(Dense(1, kernel_initializer='normal'))
    #compiling the model
    model.compile(loss='mean_squared_error', optimizer='adam')
    return model
	
# evaluate model with standardized dataset
estimators = []
estimators.append(('standardized', StandardScaler()))
estimators.append(('mlp', KerasRegressor(build_fn=baseline_model, epochs=50, batch_size=5, verbose=0)))
pipeline = Pipeline(estimators)

#Using 10 fold cross validation
kfold = KFold(n_splits=10, random_state=seed)
results = cross_val_score(pipeline, X, Y, cv=kfold)
print("Baseline: %.2f (%.2f)" %(results.mean(), results.std()))
#Running this code gives us Baseline: 22.83 (25.33)
#This means that instead of improving the results of the data standardization that we achieved in part 2, a larger model is performing worse
