from math import pi
import sys
import pandas as pd
import numpy as np
from math import ceil, exp, sqrt

def load_data(path):
    return pd.read_csv(path)

def preprocess_data(df):

    zvanje_mapper = {"Prof":2, "AsstProf":0, "AssocProf":1}
    pol_mapper = {"Female":0, "Male":1}
    oblast_mapper = {"A":0, "B": 1}

    df["zvanje"].replace(zvanje_mapper, inplace=True)  
    df["oblast"].replace(oblast_mapper, inplace=True)  
    df["pol"].replace(pol_mapper, inplace=True)  

    x = df.drop('plata', axis = 1)
    y = df['plata']

    x["zvanje"] = minmax_scaler(x["zvanje"])
    x["godina_doktor"] = minmax_scaler(x["godina_doktor"])
    x["godina_iskustva"] = minmax_scaler(x["godina_iskustva"])
    
    xv = x.values
    yv = y.values

    return xv, yv


def minmax_scaler(col):
    minimum = min(col)
    maximum = max(col)
    diff = maximum - minimum
    col = (col - minimum)/diff
    return col


def predict(test_X, train_X, train_Y, l, weightsd):

    distances = np.linalg.norm(weightsd*(train_X - test_X), axis=1)         

    # gaussian kernel
    weights = np.array([( exp((-((d/l)**2))/2) )/sqrt(2*pi) for d in distances])
    
    predicted = np.sum(weights * train_Y)/np.sum(weights)
    return predicted

# cross validation for error estimate
def tune():
    
    df = pd.read_csv("train.csv")
    df = df.sample(frac=1)
    xv, yv = preprocess_data(df)

    l = 0.25
    weightsd = [4.7, 1, 2.9, 1.1, 0.5]

    sum = 0
    splits = 5
    n, part_length = df.shape[0], ceil(n/splits)
    sumerror = 0
    
    for j in range(splits):
        sum = 0
        # selecting required training parts from indices
        i_test_start, i_test_end = part_length * j, part_length * (j + 1)
        if (i_test_end > n-1): i_test_end = n - 1
        train_X = np.concatenate((xv[:i_test_start, :], xv[i_test_end:,:]))
        train_Y = np.concatenate((yv[:i_test_start], yv[i_test_end:]))
        test_X, test_Y = xv[i_test_start:i_test_end, :], yv[i_test_start:i_test_end]
        
        for i in range(len(test_Y)):  
            predicted = predict(test_X[i], train_X, train_Y, l, weightsd)
            real = test_Y[i]
            sum += (predicted - real) **2
            
        err = sqrt(sum/len(test_Y))
        sumerror += err
        
    crosvallerr = sumerror/splits
    return crosvallerr


def kernel_regression(training, test):
    
    l = 0.25
    weightsd = [4.7, 1, 2.9, 1.1, 0.5]  

    train_X, train_Y = preprocess_data(training)
    test_X, test_Y = preprocess_data(test)

    sum = 0
    for i in range(len(test_Y)):  
        predicted = predict(test_X[i], train_X, train_Y, l, weightsd)
        real = test_Y[i]
        sum += (predicted - real) **2
    
    rmse = sqrt(sum/len(test_Y))
    return rmse


def test(path_to_training, path_to_test):
    training = load_data(path_to_training)
    test = load_data(path_to_test)
    rmse = kernel_regression(training, test)
    print(rmse)

if __name__=="__main__":
    test(sys.argv[1], sys.argv[2])