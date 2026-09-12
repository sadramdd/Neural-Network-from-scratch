from deep_neural_network.core.layer.activations import *

import numpy as np

fan_out = 9 #how many neurons
m = 100 # number os samples


def test_relu():
    random_z = np.random.rand(fan_out,m) #random z array for implementation test
    z = np.array([-2.0, -1.0, 0.0, 1.0, 2.0])#Hardcoded values for expected test
    
    actual_relu = relu(random_z)
    
    #implementation test
    ispassed = "passed" if np.array_equal(np.maximum(0,actual_relu),random_z) else "failed"
    print("1-ReLU: implementaion test")
    print(ispassed, end="\n\n")
    
    #derivative test
    expected_derivates = np.array([0.0, 0.0, 0.0, 1.0, 1.0])
    actual_derivatives = relu_derivative(z)
    ispassed = "passed" if np.array_equal(expected_derivates,actual_derivatives) else "failed"
    print("2-ReLU: common values derivatives test")
    print(ispassed, end="\n\n")
    
def test_sigmoid():
    random_z = np.random.rand(fan_out,m)
    actual_sigmoid = sigmoid(random_z)

    #Implementation test
    expected_output = 1 / (1 + np.exp(-random_z))
    ispassed = "passed" if np.array_equal(actual_sigmoid, expected_output) else "failed"
    
    print("1-Sigmoid: implementation test")
    print(ispassed, end="\n\n")
    
    #Derivative test
    expected_derivative = actual_sigmoid * (1 - actual_sigmoid) 
    actual_derivative = sigmoid_derivative(random_z)
    
    ispassed = "passed" if np.array_equal(expected_derivative, actual_derivative) else "failed"
    print("2-Sigmoid: derivative test")
    print(ispassed, end="\n\n")
    
def test_tanh():
    random_z = np.random.rand(fan_out, m)
    
    actuall_tanh = tanh(random_z) 
    
    #Implementation test
    expected_tanh = np.tanh(random_z)
    ispassed = "passed" if np.allclose(expected_tanh, actuall_tanh) else "failed"
    print("1-Tanh: implementation test")
    print(ispassed, end="\n\n")
    
    #Derivative test
    expected_derivative = 1 - np.square(actuall_tanh)
    actuall_derivative = tanh_derivative(random_z)
    ispassed = "passed" if np.allclose(expected_derivative, actuall_derivative) else "failed"
    print("2-Tanh: derivative test")
    print(ispassed, "\n\n")

def test_softmax():
    random_z = np.random.rand(fan_out, m)
    actual_sftmx = softmax(random_z)
    
    #Probability test (all probabilties of a sample should sum up to 1, in this case, sum of each column should be 1)
    expected_probabilty_dstrb = np.ones(random_z.shape[1])
    ispassed = "passed" if np.allclose(expected_probabilty_dstrb, np.sum(actual_sftmx, axis=0)) else "failed"
    print("1-softmax: probablity ditribution test")
    print(ispassed, end="\n\n")
    
    #testing if Probabilities are valid or not.
    cond1 = np.all(actual_sftmx >=0)
    cond2 = np.all(actual_sftmx <= 1)
    
    ispassed = "passed" if cond1 and cond2 else "failed"
    print("2-softmax: probablity validness test")
    print(ispassed, end="\n\n")

    
    #Verify translation invariance
    shifted_z = random_z + 1000
    shifted_sftmx = softmax(shifted_z)
    ispassed = "passed" if np.allclose(actual_sftmx,shifted_sftmx) else "failed"
    print("3-Softmax: translation variance")
    print(ispassed, end="\n\n")



test_relu()
test_sigmoid()
test_tanh()
test_softmax()