import sys
import os
import numpy as np
import pandas as pd
from sklearn.externals import joblib
from sklearn.metrics import mean_squared_error


if __name__ == '__main__':

    if (len(sys.argv) < 2):
        print "Dataset file path is missing. Terminate..."
        raw_input("Press enter to continue...")
        sys.exit(0)

    dataset_path = sys.argv[1]

    if os.path.exists(dataset_path) and os.access(dataset_path, os.R_OK) and \
            os.path.isfile(dataset_path):
        print "Data file exists and is readable"
    else:
        print "Either file is missing or is not readable. Terminate..."
        raw_input("Press enter to continue...")
        sys.exit(0)


    # you can give the path to the result file with a second
    # command line parameter
    if (len(sys.argv) > 2):
        result_path = sys.argv[2]


    # read the dataset
    print "Reading the data..."
    try:
        data = pd.read_csv(dataset_path, header=None, skiprows=1)

        cols = data.shape[1]
        y = np.copy(data.iloc[:,-1]) # get the label

        # drop the label column
        data.drop(data.columns[cols-1], axis=1, inplace=True)

        # fill the missing values with zero
        data.fillna(data.mean(), inplace=True)
    except:
        print "We had trouble reading the dataset. Terminate..."
        raw_input("Press enter to continue...")
        sys.exit(0)

    # read the prediction model and apply
    print "Reading the prediction model and applying..."
    try:
        model = joblib.load('regressor_sgd.pkl')
        y_pred = model.predict(data)

        # print out the RMSE
        err = mean_squared_error(y, y_pred)
        err = err**0.5
        print("Predictor RMSE: %.3f" % err)

        # print out the prediction percentage accuracy
        # we count each individual prediction true, if its abs error less than eq 3
        accuracy = float(np.size(np.where( np.absolute(y-y_pred) <= 3.0 ))) / \
                   float((np.size(y))) * 100
        print("Prediction Accuracy: %.3f" % accuracy)
    except:
        print "We had trouble reading the prediction model and applying. Terminate"
        raw_input("Press enter to continue...")
        sys.exit(0)


    # write the predicted values to a given text file,
    # if no file name is given, write it to "result.out" in the current folder
    try:
        with open(result_path, 'w') as f:
            np.savetxt(f, y_pred, fmt='%.3f')
	    print "Predicted values has been written out to file: %s" % result_path
    except:
        with open('result.out', 'w') as f:
		    try:
			    np.savetxt(f, y_pred, fmt='%.3f')
			    print "Predicted values has been written out to file: %s" % 'result.out'
		    except:
			    print "Predicted values could not be written to file: %s" % 'result.out'
