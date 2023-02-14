import numpy as np

def mean_squared_error(y_pred: np.array([]), y_real: np.array([])) -> np.array([]):
    return ((y_pred-y_real)**2).sum() / (2*len(y_pred))

def cross_entropy_error(y_pred: np.array([]), y_real: np.array([])) -> np.array([]):
    return y_pred