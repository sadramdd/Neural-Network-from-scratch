import numpy as np

def he_init(fan_in,
            fan_out):
    #return standard distribution of He initialization
    std = np.sqrt(2 / fan_in)

    return np.random.randn(fan_out, fan_in) * std

def xavier_init(fan_in,
                        fan_out):
    #return standard distribution of xavier initialization
    std = np.sqrt(2 / (fan_in + fan_out))

    return np.random.randn(fan_out, fan_in) * std