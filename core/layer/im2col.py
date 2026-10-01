import numpy as np
from numpy.typing import NDArray
from typing import Literal

def im2col(X: NDArray,
            K_h: int,
            K_w: int,
            padding: Literal["same", "valid", "full"],
            kernel_stride: int,
            include_channel: bool=False) -> tuple[NDArray, int, int]:
    """
    Takes out all of the patches of the input `X` according to  
    `k_h` and `k_w`, then flattens them into columns.\n
    """
    if padding == "valid":
        pad_h = pad_w = 0
    elif padding == "same":
        pad_h, pad_w = (K_h - 1) // 2, (K_w - 1) // 2
    elif padding == "full":
        pad_h, pad_w = K_h - 1, K_w - 1
    else:
        raise ValueError(f"unknown padding: {padding}")

    X = np.pad(X, ((0, 0), (0, 0), (pad_h, pad_h), (pad_w, pad_w)))
    H_out = (X.shape[2] - K_h) // kernel_stride + 1
    W_out = (X.shape[3] - K_w) // kernel_stride + 1
        
    if include_channel:
        # Also have the c (channel) dimension, Total: 4 dimensions
        X_patches = np.zeros((X.shape[0], X.shape[1], K_h * K_w, H_out * W_out))
        
        # Get each patch and flatten it into a column of X_patches
        for n in range(X.shape[0]):  # Loop over each sample in the batch
            for c in range(X.shape[1]): # Loop over each channel in the sample
                for i in range(H_out):  # Loop over the output height
                    for j in range(W_out):  # Loop over the output width

                        # Calculate starting and ending height for patch
                        h_start = i * kernel_stride
                        h_end = h_start + K_h

                        # Calculate starting and ending width for patch
                        w_start = j * kernel_stride
                        w_end = w_start + K_w

                        # Extract the patch and flatten it
                        patch = X[n, c, h_start:h_end, w_start:w_end].flatten()

                        # Store the flattened patch in the appropriate position in X_patches
                        X_patches[n, c, : , i * W_out + j] = patch
        
    else:
        # Exculde c (channel) dimension, Total: 3 dimensions
        X_patches = np.zeros((X.shape[0], X.shape[1] * K_h * K_w, H_out * W_out))
        
        # Get each patch and flatten it into a column of X_patches
        for n in range(X.shape[0]):  # Loop over each sample in the batch
            for i in range(H_out):  # Loop over the output height
                for j in range(W_out):  # Loop over the output width

                    # Calculate starting and ending height for patch
                    h_start = i * kernel_stride
                    h_end = h_start + K_h

                    # Calculate starting and ending width for patch
                    w_start = j * kernel_stride
                    w_end = w_start + K_w

                    # Extract the patch and flatten it
                    patch = X[n, :, h_start:h_end, w_start:w_end].flatten()

                    # Store the flattened patch in the appropriate position in X_patches
                    X_patches[n, :, i * W_out + j] = patch
                
    return X_patches, H_out, W_out