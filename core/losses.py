import numpy as np
from numpy.typing import NDArray, ArrayLike

def MSE(y_pred: ArrayLike,
        y_true: ArrayLike):
    return np.mean(np.square(np.subtract(y_true,y_pred)))

def cross_entropy(y_pred: NDArray,
                    y_true: NDArray):
    if y_pred.shape[0] == 1:
        #Binary cross entropy
        return -1 * (y_true * np.log(y_pred) + (1 - y_true) * np.log(1 - y_pred))
        
    else:
        #Categorical cross entropy
        return -np.mean(np.sum(y_true * np.log(y_pred), axis=0))


def mse_derivative(y_pred: NDArray,
                    y_true: NDArray):
    """
    calculated and implemented mean squared error derivative
    """
    # We do not average here, instead we average when backpropgating
    m = y_true.shape[1]
    return 2 * (y_pred - y_true) / m

@staticmethod
def cross_entropy_derivative(y_pred: NDArray,
                                y_true: NDArray):
    """
    calculated and implemendted cross entropy loss derivative
    """
    # We do not average here, instead we average when backpropgating
    m = y_true.shape[1]
    return (y_pred - y_true) / m