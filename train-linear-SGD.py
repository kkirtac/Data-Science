import numpy as np

from sklearn.linear_model import SGDRegressor
import os
import pandas as pd

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import KFold
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.pipeline import Pipeline


path = os.getcwd() + '\dataset_00_with_header.csv'
data = pd.read_csv(path, header=None, skiprows=1)

# shuffle the data
data = data.sample(frac=1).reset_index(drop=True)

# split the training data and leave 20% portion as test
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

# Stochastic Gradient Descent manual says 10**6 samples enough for convergence,
# so we compare our sample size and calculate the appropriate number of iterations
#num_iter = (np.ceil(10**6/(X_train_scaled.shape[0]*0.75)) * np.power(np.array([2]), np.arange(4,8))).astype(int)
num_iter = 3000
alpha = 10.0**-np.arange(1,7)

# our linear SGD regressor that we will optimize its regularization weight
model = SGDRegressor(average=False, epsilon=0.15, eta0=0.01,
       fit_intercept=True, l1_ratio=0.15, learning_rate='optimal',
       loss='epsilon_insensitive', n_iter=num_iter, penalty='l2', power_t=0.25,
       random_state=0, shuffle=True, verbose=0, warm_start=False)

pipe = Pipeline(steps=[('norm', StandardScaler()), ('sgd', model)])

parameters = {'sgd__alpha': alpha}

# Tune the hyperparameters by a grid search via 4-fold cross-validation
estimator = GridSearchCV(estimator=pipe, 
                         param_grid=parameters,
                         cv=4,
                         scoring='mean_squared_error',
                         n_jobs=-1, 
                         refit=True) 

# start grid search to find the best set of parameters
estimator.fit(X_train, y_train)

print(estimator.best_params_)
print
print(estimator.best_estimator_)

y_true_train, y_pred_train = y_train, estimator.best_estimator_.predict(X_train)

# print out the training RMSE
err_train = mean_squared_error(y_true_train, y_pred_train)
err_train = err_train**0.5
print("SGDRegressor RMSE X_train: %.2f" % err_train)


# print out the test RMSE
y_true, y_pred = y_test, estimator.best_estimator_.predict(X_test)
err_test = mean_squared_error(y_true, y_pred)
err_test = err_test**0.5
print("SGDRegressor RMSE X_test: %.2f" % err_test)


# train the final regressor using all available data and save the model
cols = data.shape[1]
y = np.copy(data.iloc[:,cols-1])

# drop the label column from the data 
data.drop(data.columns[cols-1], axis=1, inplace=True)

data.fillna(data.mean(), inplace=True)

estimator.best_estimator_.fit(data, y)

y_true, y_pred = y, estimator.best_estimator_.predict(data)

# print out the RMSE
err = mean_squared_error(y_true, y_pred)
err = err**0.5
print("SGDRegressor RMSE Final: %.2f" % err)

accuracy = float(np.size(np.where( np.absolute(y_pred-y_true) <= 3 ))) / float((np.size(y_true))) * 100
print("SGDRegressor Prediction Accuracy Final: %.2f" % accuracy)


try:
    with open(result_path, 'w') as f:
        np.savetxt(f, y_pred, fmt='%.4f')
except:
    with open('train_linear_sgd_pred.out', 'w') as f:
        np.savetxt(f, y_pred, fmt='%.4f')
        

from sklearn.externals import joblib
joblib.dump(estimator.best_estimator_, 'regressor_sgd.pkl')
