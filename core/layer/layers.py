import numpy as np
from typing import Literal
from numpy.typing import ArrayLike, NDArray

from .activations import *
from .initializations import *

class Layer():
    
    @staticmethod
    def _initialize_weights(fan_in: int,
                            fan_out: int,
                            method: Literal["He" ,"Xavier"]):
        if method == "He":
            return he_init(fan_in,fan_out)
            
        elif method == "Xavier":
            return xavier_init(fan_in,fan_out)
            
        else:
            raise ValueError(
                f"Invalid initialization method, got: {method}"
                )
    
    def __init__(self,
                 n_neurons,
                 activation_function: Literal["ReLU", "Sigmoid", "Tanh", "Softmax", "None"],
                 initialization_method: Literal["He" ,"Xavier"],
                 learning_rate: float=0.01,
                 random_state = None,
                 normalize: bool=False,
                 dropout_rate: float=0.0,
                 l1: float=0.0,
                 l2: float=0.0,
                 epsilon: float=0.0001,
                 p: float=0.99):
        self._act_func = activation_function
        self.learning_rate = learning_rate
        
        if random_state:
            np.random.seed(random_state)
            self.random_state = random_state
        
        # Parameters needed for weight initialization
        self.n_neurons = n_neurons
        self.initialization_method: Literal["He" ,"Xavier"] = initialization_method
        
        self._b = np.zeros((n_neurons, 1))
        
        self.normalize = normalize
        self.dropout_rate = dropout_rate
        
        self.l1 = l1
        self.l2 = l2
        
        self.epsilon = epsilon
        self.p = p
        
        if self.normalize:
            self._gamma = np.ones((n_neurons, 1)) 
            self._beta  = np.zeros((n_neurons, 1))
            self._running_mean = np.zeros((n_neurons, 1))
            self._running_variance = np.ones((n_neurons, 1))
        
    def build(self,
              shape_in: tuple):
        """
        Initializes parameters that need shape of the input .
        
        INPUT\n
        `shape_in`: Nd tuple of input shapes\n
        
        RETURNS\n
        -----------------------------------
        `shape_out`: Nd tuple of output shape
        """
    
        self._W = self._initialize_weights(shape_in[0], self.n_neurons, self.initialization_method)
    
        return (self.n_neurons,)
        
    def _call_activation(self, value):
        if self._act_func == "ReLU":
            return relu(value)
        elif self._act_func  == "Sigmoid":
            return sigmoid(value)
        elif self._act_func  == "Tanh":
            return tanh(value)
        elif self._act_func  == "Softmax":
            return softmax(value)
        elif self._act_func  == "None":
            return value
        else:
            raise ValueError(f"Got unidentifed activation function code: {self._act_func }")
        
    def _call_activation_derivative(self, z_value):
        if self._act_func  == "ReLU":
            return relu_derivative(z_value)
        elif self._act_func  == "Sigmoid":
            return sigmoid_derivative(z_value)
        elif self._act_func  == "Tanh":
            return tanh_derivative(z_value)
        elif self._act_func  == "Softmax":
            return 1.0
        elif self._act_func  == "None":
            return 1.0
        else:
            raise ValueError(f"Got unidentified activation function code: {self._act_func }")
        
    def _batch_norm(self,
                    X: NDArray,
                    train: bool):
        # because shape is (fan_out, batch_size), we'll calculate mean, variance on axis 1
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
                X: NDArray,
                cache: bool):
        # Two caches needed for backpropagation
        batchnorm_cache =None
        dropout_mask = None
        
        z = self._W @ X + self._b
        if self.normalize:
            z, batchnorm_cache = self._batch_norm(z, cache)
            
        a = self._call_activation(z)
        
        # If layer has dropout, create dropout mask
        if self.dropout_rate > 0.0:
            survival_rate = 1 -self.dropout_rate
            
            random_matrix = np.random.binomial(1, survival_rate, a.shape)
            dropout_mask = random_matrix / survival_rate
            
            a = a * dropout_mask # A_dropout
        
        if cache:
            self.z_cache = z
            self.batchnorm_cache = batchnorm_cache
            self.X_cache = X
            self.dropout_cache = dropout_mask
            
        return a
    def backward(self,
                 dA_incoming: NDArray,
                 calculate_a: bool):
        # Compute dA based on dropout rate
        if self.dropout_rate > 0.0:
            dA = dA_incoming * self.dropout_cache
        else:
            dA = dA_incoming
        
        
        if self._act_func == "Softmax":
            # Special case
            dZ_bn = dA
        else:
            dZ_bn = dA * self._call_activation_derivative(self.z_cache)
            
        
        # If layer has batch normalization, change process + calculate dgamma, dbeta
        if self.normalize:
        
            X_norm = self.batchnorm_cache["X_norm"]
            inv_std = self.batchnorm_cache["inv_std"]

            m = dZ_bn.shape[1]

            # gamma gradient
            dGamma = np.sum(
                dZ_bn * X_norm,
                axis=1,
                keepdims=True
            )

            # beta gradient
            dBeta = np.sum(
                dZ_bn,
                axis=1,
                keepdims=True
            )

            # Backprop through:
            #
            # Z_bn = gamma * X_norm + beta

            dX_norm = dZ_bn * self._gamma

            # Backprop through normalization
            dZ = (
                inv_std / m
                * (
                    m * dX_norm
                    - np.sum(
                        dX_norm,
                        axis=1,
                        keepdims=True
                    )
                    - X_norm
                    * np.sum(
                        dX_norm * X_norm,
                        axis=1,
                        keepdims=True
                    )
                )
            )
            
        else:
            dZ = dZ_bn
            dGamma = None
            dBeta = None
            
        # Weight gradients
        dW = (dZ @ self.X_cache.T)
        
        # Bias gradients
        dB = np.sum(dZ, axis=1, keepdims=True)
        
        
        # Adding regularization penalty (if there is one)
        m_samples = dZ.shape[1]
        if self.l1 > 0.0:    
            dW += (self.l1 / m_samples) * np.sign(self._W)
            
        if self.l2 > 0.0:
            dW += (self.l2 / m_samples) * self._W

        
        layer_gradient = {
                "dW": dW,
                "db": dB
            }
            
        if self.normalize:
            layer_gradient["dgamma"] = dGamma
            layer_gradient["dbeta"] = dBeta
        
        dA_prev_incoming = None
        if calculate_a :  
            dA_prev_incoming = self._W.T @ dZ
            

        return dA_prev_incoming, layer_gradient