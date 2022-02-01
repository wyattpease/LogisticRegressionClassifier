import numpy as np
import pandas as pd

def mean_square_error(w, X, y):
    errors = np.array([])
    for i in range(len(y)):
        dot =  np.dot(w,X[i])
        diff = float(y[i] - dot)
        sq = np.square(diff)
        error = sq/len(y)
        errors = np.append(errors, error)
    err = np.sum(errors)
    return err

def linear_regression_noreg(X, y):
  dot = np.dot(X.T, X)
  X_i = np.linalg.inv(dot)
  w = np.dot(X_i, np.dot(X.T, y))
  return w

def regularized_linear_regression(X, y, lambd):
    l_id = lambd * np.identity(len(X[0]))
    dot = np.dot(X.T, X) + l_id
    X_i = np.linalg.inv(dot)
    w = np.dot(X_i, np.dot(X.T, y))

    return w

def tune_lambda(Xtrain, ytrain, Xval, yval):
    mse = np.Inf	
    bestlambda = float(00.00)
    step = None
    for i in range(0, -14, -1):
        lambd = float(2**i)
        w = regularized_linear_regression(Xtrain, ytrain, lambd)
        temp = mean_square_error(w, Xval, yval)
        if (temp < mse):
            mse = temp
            bestlambda = lambd
            step = i
    return bestlambda

def mapping_data(X, p):
    augmented = np.zeros([X.shape[0], p*X.shape[1]])
    for n in range(X.shape[0]):
        temp = np.array([])
        for i in range(1, p+1):  
            n_pow = np.power(X[n], i)
            temp = np.concatenate((temp,n_pow))
        augmented[n]=temp

    return augmented

