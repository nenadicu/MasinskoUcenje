import numpy as np
import csv
from math import sqrt
import sys
#import matplotlib.pyplot as plt

def load_data(path):
    with open(path, 'r') as f:
        data = list(csv.reader(f, delimiter=","))
    data = np.array(data)
    matrix = data[1:, :].astype(np.float64)
    return matrix


def get_theta(X, y):  
    X_transpose = X.T  
    theta = np.linalg.inv(X_transpose.dot(X)).dot(X_transpose).dot(y)  
    return theta # returns a list  


def remove_outliers(x, y):
    result = np.where(y/x > 140)
    x2 = np.delete(x, np.asarray(result[0]))
    y2 = np.delete(y, result[0])
    return np.c_[x2, y2]
    

def fit(path_to_training):
    
    pairs = load_data(path_to_training)
    pairs = remove_outliers(pairs[:, 0], pairs[:, 1])
    
    #np.random.shuffle(pairs)
    
    training = pairs #[:171, :]
    x = training[:, 0]
    y = training[:, 1]

    # validation = pairs[171:, :]
    # xv = validation[:, 0]
    # yv = validation[:, 1]

    X_squared = np.square(x)
    x_cubed = np.power(x, 3)
    x_quad = np.power(x,4)
    x_5e = np.power(x,5)
    x_6 = np.power(x,6)
    X_b = np.c_[np.ones((x.size, 1)), x, X_squared, x_cubed,x_quad]
    theta = get_theta(X_b, y)  
    return theta


#params = [ 38.08650036, -375.31725518,  864.29179361]
#params = [  37.87036743, -374.51316717,  862.18312761]

def rse(test, originals, params):
    err = 0
    predicted = []
    x_squared = np.square(test)
    x_cubed = np.power(test, 3)
    x_quaded = np.power(test,4)
    x_5 = np.power(test,5)
    x_6 = np.power(test, 6)
    test_x_b = np.c_[np.ones((test.size, 1)), test, x_squared, x_cubed,x_quaded]
    predicted = test_x_b.dot(params)  
    arr = np.array(predicted)
    sum = 0
    for j in range(test.shape[0]):  
        sum += (arr[j] - originals[j])**2 
    err = sqrt(sum/test.shape[0])
    return err

def do_everything(path_to_training, path_to_test):
    data = load_data(path_to_test)
    dt = load_data(path_to_training)
    params = fit(path_to_training)
    err = rse(data[:, 0], data[:, 1], params)


    xt = np.linspace(0.12, 0.46, 1000)
    xt_cubed = np.power(xt, 3)
    xt_q = np.power(xt, 4)
    xt_5 = np.power(xt, 5)
    x_6 = np.power(xt,6)
    xt_squared = np.square(xt)
    x4 = np.c_[np.ones((1000, 1)), xt, xt_squared, xt_cubed,xt_q]

    # calculate the y value for each element of the x vector
    y4 = x4.dot(params)  
    

   # fig, ax = plt.subplots()
   ## ax.plot(xt, y4)
    #ax.plot(dt[:,0], dt[:, 1], "r.")
   # plt.show()

    print(err)


def test_only(path_to_test):
    data = load_data(path_to_test)
    params = [  37.87036743, -374.51316717,  862.18312761]
    err = rse(data[:, 0], data[:, 1], params)
    print(err)


if __name__=="__main__":
    do_everything(sys.argv[1], sys.argv[2])
    #test_only(sys.argv[2])