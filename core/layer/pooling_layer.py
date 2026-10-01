import numpy as np
from typing import Literal
from numpy.typing import ArrayLike, NDArray


from .im2col import im2col

class PoolingLayer():
    def __init__(self,
                 k_w: int,
                 k_h: int,
                 padding: Literal["same", "valid", "full"],
                 stride: int,
                 method: Literal["max", "average"],
                 ) -> None:
        self.K_h= k_h
        self.K_w = k_w
        
        self.padding: Literal["same", "valid", "full"] = padding
        self.stride = stride
        self.method = method
    
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
        
        if self.padding == "valid":
            pad_h = pad_w = 0
        elif self.padding == "same":
            pad_h, pad_w = (self.K_h - 1) // 2, (self.K_w - 1) // 2
        elif self.padding == "full":
            pad_h, pad_w = self.K_h - 1, self.K_w - 1
        else:
            raise ValueError(f"unknown padding: {self.padding}")
        
        H_out = (shape_in[2] + 2 * pad_h - self.K_h) // self.stride + 1
        W_out = (shape_in[3] + 2 * pad_w - self.K_w) // self.stride + 1
        
        return (shape_in[0], shape_in[1], H_out, W_out)

    
    def forward(self,
                X:NDArray,
                cache: bool) -> NDArray:
        N, C, _, _ = X.shape
        
        columns, h_out, w_out = im2col(X, self.K_h, self.K_w, self.padding, self.stride, include_channel=True)
        if self.method == "max":
            Y = columns.max(axis=2) #Take the max of patches (axis=2 insures channels don't get mixed & choosing max is only from 2d patches)
        else:
            Y = columns.mean(axis=2)
        
        output = Y.reshape(N, C, h_out, w_out)
        
        
        if cache:
            self.X_cache = X
            self.columns = columns
            if self.method == "max":
                self.argmax_cache = columns.argmax(axis=2)
            
        return output
    
    def _patches_to_grad(self, d_patches, H_out, W_out):
        N, C, _, _ = self.X_cache.shape
        dA = np.zeros_like(self.X_cache)
        for n in range(N):
            for c in range(C):
                for i in range(H_out):
                    for j in range(W_out):
                        hs, ws = i * self.stride, j * self.stride # Calculating indices
                        
                        grad = d_patches[n, c, :, i * W_out + j].reshape(self.K_h, self.K_w) # Create a mask of gradients
                        dA[n, c, hs:hs+self.K_h, ws:ws + self.K_w] += grad # += is mandantory since some gradients might overlap
        return dA
    

    def _maxpool_backward(self,
                          dY: NDArray,
                          patches: NDArray,
                          H_out: int,
                          W_out: int):
        N, C, _, HW_out = patches.shape
        
        d_patches = np.zeros_like(patches) # Gradient expanded back to the individual elements of each pooling window.
        
        np.put_along_axis(d_patches,
                          self.argmax_cache[:, :, None, :],
                          dY.reshape(N, C, 1, HW_out), axis=2)
        
        return self._patches_to_grad(d_patches, H_out, W_out)


    def _averagepool_backward(self,
                              dY,
                              H_out,
                              W_out):
        N, C, _, _ = dY.shape
        
        d_patches = np.repeat(dY.reshape(N, C, 1, H_out*W_out),
                              self.K_h*self.K_w, axis=2) / (self.K_h*self.K_w)# Gradient expanded back to the individual elements of each pooling window.
        
        return self._patches_to_grad(d_patches, H_out, W_out)

        
        
        
    def backward(self,
                 dY: NDArray) -> tuple[NDArray, dict]:
        """
        Routes `dA` through pooling layer\n
        `dY`: gradient of output with respect to loss\n

        RETURNS\n
        If average pooling -> dYpatch / (Kh * Kw)\n
        If max pooling -> dYpatch * argmax_cache.  (argmax_cache is calculated during forwardpass)
        """
        
        if self.method == "max":
            dY_patches, H_out, W_out = im2col(dY, self.K_h, self.K_w, self.padding, self.stride, include_channel=True)
            return self._maxpool_backward(dY, dY_patches, H_out, W_out), dict()
             
        else:
            # Calculate Hout and Kout
            if self.padding == "same":
                pad_h = (self.K_h - 1) // 2
                pad_w = (self.K_w - 1) // 2
                dY = np.pad(dY, ((0,0),(0,0),(pad_h,pad_h),(pad_w,pad_w)))
            elif self.padding == "full":
                pad_h = self.K_h - 1
                pad_w = self.K_w - 1
                dY = np.pad(dY, ((0,0),(0,0),(pad_h,pad_h),(pad_w,pad_w)))
            
            
            H_out = (dY.shape[2] - self.K_h) // self.stride + 1
            W_out = (dY.shape[3] - self.K_w) // self.stride + 1
            
            return self._averagepool_backward(dY, H_out, W_out), dict()