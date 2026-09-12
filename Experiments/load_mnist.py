from tensorflow.keras.datasets import mnist
import numpy as np

def load_handwritten_digits(normalize: bool=True,
              reshape_x: bool=True,
              reshape_y: bool=True):
    (X_train, y_train), (X_test, y_test) = mnist.load_data()
    
    if normalize:
        #Normalize
        X_train = X_train / 255.0
        X_test = X_test / 255.0
        
    if reshape_x:
        #Reshape training and testing inputs
        X_train = X_train.reshape(60000, 784).T
        X_test  = X_test.reshape(10000, 784).T
        
    if reshape_y:
        #Reshape training and testing outputs
        num_classes = 10
        
        y_train = np.eye(num_classes)[y_train].T
        y_test = np.eye(num_classes)[y_test].T

    return X_train, y_train, X_test, y_test