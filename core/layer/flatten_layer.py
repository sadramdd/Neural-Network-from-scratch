import numpy as np
from numpy.typing import NDArray
from typing import Literal

class FlattenLayer():
    def __init__(self) -> None:
        pass
    
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
        
        return (int(np.prod(shape_in)),) # flattens into (dim1*dim2*...dimn, )
    
    def forward(self,
                X: NDArray,
                cache: bool):
        x_shape = X.shape
        x_flat = X.reshape(x_shape[0], -1)# N, (Nout * h * w)
    
        if cache: 
            self.x_shape_cache = x_shape
            
        return x_flat
    def backward(self,
                 dA: NDArray):
        # dA: (N, C*H*W), gradient from the next layer
        dX = dA.reshape(self.x_shape_cache)    # (N, C, H, W)
        
        return dX, {}