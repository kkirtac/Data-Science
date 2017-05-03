import numpy as np
import os
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
from sklearn.pipeline import Pipeline


path = os.getcwd() + '\dataset_00_with_header.csv'
data = pd.read_csv(path, header=None, skiprows=1)

# shuffle the data
data = data.sample(frac=1).reset_index(drop=True)

# split the training data and keep 20% portion as test
X_train, X_test = train_test_split(data, test_size = 0.2)

cols = X_train.shape[1]

y_train = np.copy(X_train.iloc[:,-1])

# drop the label column from the train partition
X_train.drop(X_train.columns[cols-1], axis=1, inplace=True)
X_train.fillna(X_train.mean(), inplace=True)

y_test = np.copy(X_test.iloc[:,-1])

# drop the label column from the train partition
X_test.drop(X_test.columns[cols-1], axis=1, inplace=True)
X_test.fillna(X_train.mean(), inplace=True)

## HyperParameter Optimization ##
model = RandomForestRegressor(random_state=0)
    
param_grid = {"rf__n_estimators": [500, 800, 1000],
    "rf__max_depth": [8, None],
    "rf__max_features": ['sqrt', 'log2', None],
    "rf__min_samples_split": [2, 5, 10],
    "rf__min_samples_leaf": [1, 3, 10],
    "rf__bootstrap": [True, False]}

# perform first standard normalization on the training data,
# the validation fold will then be transformed with the mean and std computed
# from the corresponding training fold 	
pipe = Pipeline(steps=[('norm', StandardScaler()), ('rf', model)])

# Tune the hyperparameters by a grid search via 4-fold cross-validation
estimator = GridSearchCV(estimator=pipe, 
                         param_grid=param_grid,
                         cv=4,
                         scoring='mean_squared_error',
                         n_jobs=-1, 
                         refit=True) 


# start grid search to find the best set of parameters
estimator.fit(X_train, y_train)

print(estimator.best_params_)
print
print(estimator.best_estimator_)

# perform prediction on the training data
y_true_train, y_pred_train = y_train, estimator.best_estimator_.predict(X_train)

# print out the training RMSE
err_train = mean_squared_error(y_true_train, y_pred_train)
err_train = err_train**0.5
print("RandomForestRegressor RMSE X_train: %.2f" % err_train)

# perform the prediction on our test data
y_true, y_pred = y_test, estimator.best_estimator_.predict(X_test)

# print out the test RMSE
err_test = mean_squared_error(y_true, y_pred)
err_test = err_test**0.5
print("RandomForestRegressor RMSE X_test: %.2f" % err_test)


# train the final regressor using all data and save the model
cols = data.shape[1]
y = np.copy(data.iloc[:,-1])  # pick the label column

# drop the label column from the data 
data.drop(data.columns[cols-1], axis=1, inplace=True)

data.fillna(data.mean(), inplace=True)

estimator.best_estimator_.fit(data, y)

y_true, y_pred = y, estimator.best_estimator_.predict(data)

# print out the RMSE
err = mean_squared_error(y_true, y_pred)
err = err**0.5
print("RandomForestRegressor RMSE Final: %.2f" % err)

accuracy = float(np.size(np.where( np.absolute(y_pred-y_true) <= 3 ))) / float((np.size(y_true))) * 100
print("RandomForestRegressor Prediction Accuracy Final: %.2f" % accuracy)

try:
    with open(result_path, 'w') as f:
        np.savetxt(f, y_pred, fmt='%.4f')
except:
    with open('train_random_forest_pred.out', 'w') as f:
        np.savetxt(f, y_pred, fmt='%.4f')
        
        
from sklearn.externals import joblib
joblib.dump(estimator.best_estimator_, 'regressor_random_forest.pkl')
