from deep_neural_network.core.layer.layers import Layer
import numpy as np

x = np.random.randn(100,5000)# 4 features, 500 samples

fan_out = 100
fan_in = x.shape[0]

def test_he_initialization(n_times: int=10):
    mean_mean = 0
    mean_variance = 0
    mean_std = 0
    for _ in range(n_times):
        #Initialize a layer with He initialization
        he_layer = Layer(fan_out,fan_in,"ReLU","He")

        #Summ up the averages and variances for getting the mean -> more accurate verification
        mean_mean += he_layer._W.mean()
        mean_variance += np.var(he_layer._W)
        mean_std += np.std(he_layer._W)
    
    #Calculate the averages
    mean_mean /= n_times
    mean_variance /= n_times
    mean_std /= n_times
    
    #Verify the shape
    print("Shape: (He)")
    print(f"expected: {(fan_out, fan_in)}")
    print(f"actuall {he_layer._W.shape}")
    ispassed = "passed" if he_layer._W.shape == (fan_out, fan_in) else "failed"
    print(ispassed, end="\n\n")

    #Verify mean 
    print(f"Mean: (He)")
    print(f"expected: near 0")
    print(f"actuall: {mean_mean}")
    ispassed = "passed" if abs(mean_mean) <0.01 else "failed"
    print(ispassed, end="\n\n")
    
    #Verify the standard derivation
    print(f"std: (He)")
    expected_he_std = np.sqrt(2 / fan_in)
    print(f"expected: {expected_he_std}")
    print(f"actuall: {mean_std}")
    ispassed = "passed" if np.isclose(expected_he_std, mean_std, rtol=0.05) else "failed"
    print(ispassed, end="\n\n")

    #Verify Variance
    print("Variance: (He)")
    he_expected_variance = 2 / fan_in
    print(f"expected: {he_expected_variance}")
    print(f"actuall: {mean_variance}")
    ispassed = "passed" if np.isclose(he_expected_variance, mean_variance, rtol=0.05) else "failed"
    print(ispassed, end="\n\n")
    
    
    
def test_xavier_initialization(n_times: int=10):
    mean_mean = 0
    mean_variance = 0
    mean_std = 0
    for _ in range(n_times):
        #Initialize a layer with Xavier initialization
        xavier_layer = Layer(fan_out,fan_in,"ReLU","Xavier")

        #Summ up the averages and vanriances for getting the mean -> more accurate verification
        mean_mean += xavier_layer._W.mean()
        mean_variance += np.var(xavier_layer._W)
        mean_std += np.std(xavier_layer._W)
    
    #Calculate the averages
    mean_mean /= n_times
    mean_variance /= n_times
    mean_std /= n_times
    
    #Verify the shape
    print("Shape: (Xavier)")
    print(f"expected: {(fan_out, fan_in)}")
    print(f"actuall {xavier_layer._W.shape}")
    ispassed = "passed" if xavier_layer._W.shape == (fan_out, fan_in) else "failed"
    print(ispassed, end="\n\n")

    #Verify mean 
    print(f"Mean: (Xavier)")
    print(f"expected: near 0")
    print(f"actuall: {mean_mean}")
    ispassed = "passed" if abs(mean_mean) <0.01 else "failed"
    print(ispassed, end="\n\n")
    
    #Verify the standard derivation
    print(f"std: (Xavier)")
    expected_xavier_std = np.sqrt(2 / (fan_in + fan_out))
    print(f"expected: {expected_xavier_std}")
    print(f"actuall: {mean_std}")
    ispassed = "passed" if np.isclose(expected_xavier_std, mean_std, rtol=0.05) else "failed"
    print(ispassed, end="\n\n")

    #Verify Variance
    print("Variance: (Xavier)")
    xavier_expected_variance = 2 / (fan_in + fan_out)
    print(f"expected: {xavier_expected_variance}")
    print(f"actuall: {mean_variance}")
    ispassed = "passed" if np.isclose(xavier_expected_variance, mean_variance, rtol=0.05) else "failed"
    print(ispassed, end="\n\n")
    

test_he_initialization(100)
test_xavier_initialization(100)



#python -m deep_neural_network.Verifications.initialization_test