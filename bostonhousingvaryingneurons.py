#Model tuning by varying the number of neurons in our baseline model defined in part 1 of this project
# Regression Example With Boston Dataset: Standardized and Wider
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

#Fixing data reproducibility
seed = 7
numpy.random.seed(seed)

#Loading the data
dataframe = read_csv('housing.csv', delim_whitespace=True, header=None)
dataset = dataframe.values

#Separating inputs and outputs
X = dataset[:,0:13]
Y = dataset[:,13]

#Defining the model
def baseline_model():
    #creating the model
    model = Sequential()
    model.add(Dense(20, input_dim=13, kernel_initializer='normal', activation='relu'))
    model.add(Dense(1, kernel_initializer='normal'))
    #compiling the model
    model.compile(loss='mean_squared_error', optimizer='adam')
    return model
	
#Standardizing the data
estimators = []
estimators.append(('standardize', StandardScaler()))
estimators.append(('mlp', KerasRegressor(build_fn=baseline_model, epochs=100, batch_size=5, verbose=0)))
pipeline = Pipeline(estimators)

#Cross validating the results
kfold = KFold(n_splits=10, random_state=seed)
results = cross_val_score(pipeline, X, Y, cv=kfold)
print("Baseline: %.2f (%.2f)" %(results.mean(), results.std()))
#Running the algorithm gives us a slightly better performance with a Baseline: 21.71 (24.39)