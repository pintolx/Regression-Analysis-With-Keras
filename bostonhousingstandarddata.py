#Because the inputs in the boston housing datasets have different scales, I beleive that this affects performance of the model
#We can overcome this by first standardizing the dataset and then evaluating the model on the standardized dataset
#These are the steps that we take below
#We use scikit learn's Pipeline to standardize the data and to prevent leakages during testing
# Regression Example With Boston Dataset: Standardized
import numpy
from pandas import read_csv
from keras.models import Sequential
from keras.layers import Dense
from keras.wrappers.scikit_learn import KerasRegressor
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

#Fixing reproducibility
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
    model.add(Dense(1, kernel_initializer='normal')) #No activation function because we are trying to predict real values
    #compiling the model
    model.compile(loss='mean_squared_error', optimizer='adam')
    return model
	
# evaluate model with standardized dataset
estimators = []
estimators.append(('standardize', StandardScaler()))
estimators.append(('mlp', KerasRegressor(build_fn=baseline_model, epochs=50, batch_size=5, verbose=0)))
pipeline = Pipeline(estimators)

#Cross validation
kfold = KFold(n_splits=10, shuffle=True, random_state=seed)
results = cross_val_score(pipeline, X, Y, cv=kfold)
print("Baseline: %.2f (%.2f)" %(results.mean(), results.std()))

#Without standardized data, the results were Baseline: 31.64 (26.82) MSE
#After standardizing the data, the spread in the results decreased to 7.88 and the mean error also reduced to 19.28 from 31.64
#Baseline: 19.28 (7.88)