import numpy as np
from numpy.typing import NDArray
from typing import Literal

from .activations import *
from .im2col import im2col


class ConvolutionalLayer():
    
    @staticmethod
    def he_init(C_out,
                C_in,
                kernel):
        
        fan_in = C_in * kernel * kernel
        
        std = np.sqrt(2.0 / fan_in)
        return np.random.randn(C_out, C_in, kernel, kernel) * std

    @staticmethod
    def xavier_init(C_out,
                    C_in,
                    kernel):
        
        fan_in = C_in * kernel * kernel
        fan_out = C_out * kernel * kernel
        
        std = np.sqrt(2.0 / (fan_in + fan_out))
        return np.random.randn(C_out, C_in, kernel, kernel) * std


    def initialize_weight(self,
                          c_in: int,
                          n_filters: int,
                          kernel: int,
                          init_method: Literal["He" ,"Xavier"]="He") -> NDArray:
        if init_method == "He":
            return self.he_init(n_filters,
                                c_in,
                                kernel)
            
        elif init_method == "Xavier":
            return self.xavier_init(n_filters,
                                    c_in,
                                    kernel)
            
        else:
            raise ValueError(
                f"Invalid initialization method, got: {init_method}"
                )          
            
                   
    @staticmethod
    def _im2col_convolution(X: NDArray,
                            filter: NDArray,
                            padding: Literal["same", "valid", "full"],
                            stride: int) -> NDArray:
        """
        Gets and inout : `X` and a filter and performs a convolution operation on it.\n
        The output is the result of the convolution.
        
        """
        c_out, _, k_h, k_W = filter.shape
        
        #Get im2col representation of the input
        X_im2col, H_out, W_out = im2col(X, #input
                                              k_h, #K_height
                                              k_W, #K_width
                                              padding,
                                              stride)
        
        
        # Reshape the filter to a 2D array for matrix multiplication
        W_row = filter.reshape(c_out, -1)     
        
        
        N, D, P = X_im2col.shape # N= Number of patches D = C_in*K_h*K_w, P = H_out*W_out
        X_merged = X_im2col.transpose(1, 0, 2).reshape(D, N * P)   # (D, N*P)
        
        Z_flat = W_row @ X_merged                       # (c_out, N*P)  — one matmul, whole batch
        
        output = Z_flat.reshape(c_out, N, H_out, W_out).transpose(1, 0, 2, 3)  # back to (N, c_out, H, W)
            
        return output

    def __init__(self,
                 n_filters: int,
                 kernel: int,
                 stride: int,
                 padding: Literal["same", "valid", "full"],
                 initialization: Literal["He", "Xavier"] = "He",
                 activation: Literal["ReLU", "Sigmoid", "Tanh", "Softmax", "None"] = "None",
                 normalize: bool=False) -> None:
        
        if activation not in ("ReLU", "Sigmoid", "Tanh", "Softmax", "None"):
            raise ValueError(f"Recieved invalid activation function: {activation}")
        if initialization not in ("He", "Xavier"):
            raise ValueError(f"Recieved invalid initialization method: {initialization}")
        if padding not in ("same", "valid", "full"):
            raise ValueError(f"Recieved invalid padding method: {padding}")
        
        self.initialization_method: Literal["He", "Xavier"] = initialization

        self._b = np.zeros((1, n_filters))
              
        #Save configurations
        self.normalize = normalize
        self._act = activation
        
        self.pad: Literal["same", "valid", "full"] = padding
        self.stride = stride
        
        self.c_out = n_filters
        self.k_h = kernel
        self.k_w = kernel
        
    def build(self,
              shape_in: tuple):
        """
        Initializes parameters.
        
        INPUT\n
        `shape_in`: Nd tuple of input shapes\n
        
        RETURNS\n
        -----------------------------------
        `shape_out`: Nd tuple of output shape
        """
        self.c_in = shape_in[1]
        
        #Initialize weights & biases
        self._W = self.initialize_weight(shape_in[1],
                                        self.c_out,
                                        self.k_h,
                                        init_method=self.initialization_method)

        if self.pad == "valid":
            pad_h = pad_w = 0
        elif self.pad == "same":
            pad_h, pad_w = (self.k_h - 1) // 2, (self.k_w - 1) // 2
        elif self.pad == "full":
            pad_h, pad_w = self.k_h - 1, self.k_w - 1
        else:
            raise ValueError(f"unknown padding: {self.pad}")
        
        H_out = (shape_in[2] + 2 * pad_h - self.k_h) // self.stride + 1
        W_out = (shape_in[3] + 2 * pad_w - self.k_w) // self.stride + 1
            
        return (shape_in[0], self.c_out, H_out, W_out)
        
        
    # Calling activation (forward process)
    def _call_activation(self,value):
        if self._act == "ReLU":
            return relu(value)
        elif self._act == "Sigmoid":
            return sigmoid(value)
        elif self._act == "Tanh":
            return tanh(value)
        elif self._act == "Softmax":
            return softmax(value)
        elif self._act == "None":
            return value
        else:
            raise ValueError(f"Got unidentifed activation function code: {self._act}")
        
    # Calling derivative of activation (backward process)
    def _call_activation_derivative(self,z_value):
        if self._act == "ReLU":
            return relu_derivative(z_value)
        elif self._act == "Sigmoid":
            return sigmoid_derivative(z_value)
        elif self._act == "Tanh":
            return tanh_derivative(z_value)
        elif self._act == "Softmax":
            return 1.0
        elif self._act == "None":
            return 1.0
        else:
            raise ValueError(f"Got unidentified activation function code: {self._act}")     
          
    def forward(self,
                X: NDArray,
                cache: bool=False):
        
        # Calculate the convolution output and add the bias term
        Z = self._im2col_convolution(X=X, 
                                     filter=self._W,
                                     padding=self.pad,
                                     stride=self.stride) + self._b.reshape(1, -1, 1, 1)
                       
        # Apply the activation function 
        A = self._call_activation(Z)

        # If used for backpropagation, return the cache for later use
        if cache:
            self.Z_cache = Z
            self.X_cache = X
            return A
        else:
            return A


    def backward(self,
                 dA: NDArray,
                 calculate_a: bool):
        """
        Backward propagation for this convolutional layer.

        Parameters:
        - dA: The gradient of activation of this layer (obtained by the next layer)

        Returns:
        - dA_prev: Gradient of the cost with respect to the input of the conv layer (A_prev)
        - dW: Gradient of the cost with respect to W
        - db: Gradient of the cost with respect to b
        """
        
        # Calculating dZ based on dA
        dZ = dA * self._call_activation_derivative(self.Z_cache)
        
        # Calculating db based on the sum of dZ across the batch and spatial dimensions
        db = np.sum(dZ, axis=(0, 2, 3), keepdims=True)
        
        
        # Calculating dW based on the convolution of A_prev and dZ
        A_prev_swapped = self.X_cache.transpose(1, 0, 2, 3)  
        dZ_swapped = dZ.transpose(1, 0, 2, 3)  
        
        dW = self._im2col_convolution(A_prev_swapped, dZ_swapped, padding="valid", stride=self.stride)
        
        d_parameters = (dW, db) # Add reg and other learnable parameters later
        
        layer_gradients = {
                "dW": dW,
                "db": db
        }
        
        dA_prev = None
        if calculate_a:
            # Calculating dA_prev based on the convolution of dZ and W
            W_180rot = np.flip(self._W, axis=(-2, -1)).transpose(1, 0, 2, 3) # preferd axes for _im2col_convolutio
            dA_prev = self._im2col_convolution(dZ, W_180rot, "full", self.stride)
            
        else:
            return dA_prev, layer_gradients
