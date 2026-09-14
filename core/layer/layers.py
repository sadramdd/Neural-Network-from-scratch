import numpy as np
from typing import Literal
from numpy.typing import ArrayLike, NDArray

from .activations import *
from .initializations import *

class Layer():
    
    def _initialize_weights(self,
                            fan_in: int,
                            fan_out: int,
                            method: Literal["He" ,"Xavier"]):
        if method == "He":
            self._W = he_init(fan_in,fan_out)
            
        elif method == "Xavier":
            self._W = xavier_init(fan_in,fan_out)
            
        else:
            raise ValueError(
                f"Invalid initialization method, got: {method}"
                )
    
    def __init__(self,
                 n_neurons,
                 fan_in,
                 activation_function: Literal["ReLU", "Sigmoid", "Tanh", "Softmax", "None"],
                 kernel_initialization: Literal["He" ,"Xavier"],
                 learning_rate: float=0.01,
                 random_state = None,
                 normalize: bool=False,
                 dropout_rate: float=0.0,
                 epsilon: float=0.0001,
                 p: float=0.99):
        self._act_func = activation_function
        self.learning_rate = learning_rate
        
        if random_state:
            np.random.seed(random_state)
            self.random_state = random_state
        
        self._b = np.zeros((n_neurons, 1))
        self._initialize_weights(fan_in, n_neurons, kernel_initialization)
        
        self.normalize = normalize
        self.dropout_rate = dropout_rate
        
        self.epsilon = epsilon
        self.p = p
        
        if normalize:
            self._gamma = np.ones((n_neurons, 1)) 
            self._beta  = np.zeros((n_neurons, 1))
            self._running_mean = np.zeros((n_neurons, 1))
            self._running_variance = np.ones((n_neurons, 1))
        
    def _call_activation(self, code, value):
        if code == "ReLU":
            return relu(value)
        elif code == "Sigmoid":
            return sigmoid(value)
        elif code == "Tanh":
            return tanh(value)
        elif code == "Softmax":
            return softmax(value)
        elif code == "None":
            return value
        else:
            raise ValueError(f"Got unidentifed activation function code: {code}")
        
    def _call_activation_derivative(self, code, z_value):
        if code == "ReLU":
            return relu_derivative(z_value)
        elif code == "Sigmoid":
            return sigmoid_derivative(z_value)
        elif code == "Tanh":
            return tanh_derivative(z_value)
        elif code == "Softmax":
            return 1.0
        elif code == "None":
            return 1.0
        else:
            raise ValueError(f"Got unidentified activation function code: {code}")
        
    def _batch_norm(self,
                    X: NDArray,
                    train: bool):
        #because shape is (fan_out, batch_size), we'll calculate mean, variance on axis 1
        if train:
            #Get mean and variance from batch 
            x_mean = X.mean(axis=1, keepdims=True)
            x_variance = X.var(axis=1, keepdims=True)
            
            #Normalization
            x_centered = X - x_mean
            inv_std = 1 / np.sqrt(x_variance + self.epsilon) #Needed for backpropagation
            x_norm = x_centered * inv_std
            
            #Update running statistics
            p_percent = 1 - self.p
            

            
            self._running_mean = (self.p * self._running_mean 
                                  +
                                  p_percent * x_mean )
            
            self._running_variance = (self.p * self._running_variance 
                                      +
                                      p_percent * x_variance)
            
            #Save cache information for backward()
            batchnorm_cache = {
                "X": X,
                "X_norm": x_norm,
                "mean": x_mean,
                "variance": x_variance,
                "inv_std": inv_std
            }
            
            #Scale & shift + cache
            return self._gamma * x_norm + self._beta, batchnorm_cache
            
        else:
            #Use running statistics during inference
            x_norm = ( (X - self._running_mean)
                       / 
                       np.sqrt(self._running_variance + self.epsilon) 
                    )
        
            #Scale & shift
            return self._gamma * x_norm + self._beta, None
    
    def forward(self,
                A: ArrayLike,
                cache: bool):
        # Two caches needed for backpropagation
        batchnorm_cache =None
        dropout_mask = None
        
        z = self._W @ A + self._b
        if self.normalize:
            z, batchnorm_cache = self._batch_norm(z, cache)
            
        a = self._call_activation(self._act_func, z)
        
        #If layer has dropout
        if self.dropout_rate > 0.0:
            survival_rate = 1 -self.dropout_rate
            
            random_matrix = np.random.binomial(1, survival_rate, a.shape)
            dropout_mask = random_matrix / survival_rate
            
            a = a * dropout_mask # A_dropout
        
        if cache:
                return z, a, batchnorm_cache, dropout_mask

        return a