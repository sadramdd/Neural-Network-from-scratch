from deep_neural_network.core.main_structure import NeuralNetwork
from deep_neural_network.core.layer.layers import Layer

import numpy as np
def test_forward():
    """
    1-Compares shape and ouputs in single layers\n
    2-Compares shapes and ouputs in multiple layers\n
    3-Compares cached gradients
    """
    #Settings
    n_neurons = [6,5,4,3]
    n_input_features = 4
    n_input_samples = 20
    
    random_s = 42
    lr = 0.001    
    
    #Random input
    X = np.random.randint(low= 0,
                          high= 50,
                          size= (n_input_features, n_input_samples))
    
    #1- Testing single layers
    
    #Neural network's layer
    nn  = NeuralNetwork(n_input_features,
                        "Regression",
                        learning_rate=lr,
                        random_state=random_s)
    nn.add_layer(n_neurons[0], "ReLU", "He") 
    
    #The layer we already tested
    example_layer = Layer(n_neurons[0],
                          n_input_features,
                          "ReLU",
                          "He",
                          learning_rate=lr,
                          random_state=random_s)
    
    
    #Compare forward operations
    nn_output = nn._forward_feed(X, compute_gradients=True)
    layer_z, layers_output  = example_layer.forward(X, cache=True)
    
    #Compare shapes
    shape_check = "passed" if nn_output.shape == layers_output.shape else "failed"
    print(f"single layer Shape check: {shape_check}", end="\n")
    
    #Compare outputs
    output_check = "passed" if np.allclose(nn_output, layers_output) else "failed"
    print(f"single layer Output check: {output_check}", end="\n")
    
    #Compare caches
    Zcheck = np.allclose(nn.Z_cache[0], layer_z)
    
    cache_check = "passed" if Zcheck else "failed"
    print(f"single layer cache check: {cache_check}", end="\n")
    
    
    #2- Testing multiple layers
    n_layers = 4
    
    #Neural network's layer
    nn  = NeuralNetwork(n_input_features,
                        "Regression",
                        learning_rate=lr,
                        random_state=random_s)
    
    layers_cache = []
    layers_output = X
    
    for i in range(n_layers):
        #Add layers to neural network
        nn.add_layer(n_neurons[i], "Sigmoid", "Xavier")
        
        #Compute the output of manual layers
        example_layer = Layer(n_neurons[i],
                        layers_output.shape[0],
                        "Sigmoid",
                        "Xavier",
                        learning_rate=lr,
                        random_state=random_s)
        #Copying the weights and biases to initialized layer (event tought the random states are the same)
        example_layer._W = nn._layers[i]._W.copy()
        example_layer._b = nn._layers[i]._b.copy()
        
        layers_z ,layers_output = example_layer.forward(layers_output, cache=True)
        layers_cache.append(layers_z)
        
    network_ouput = nn._forward_feed(X, compute_gradients=True)
    
    #Compare shapes
    shape_check = "passed" if network_ouput.shape == layers_output.shape else "failed"
    print(f"Multiple layer Shape check: {shape_check}", end="\n")
    
    #Compare outputs
    output_check = "passed" if np.allclose(network_ouput, layers_output) else "failed"
    print(f"Multiple layer Output check: {output_check}", end="\n")
    
    #Compare caches
    Zcheck = all(
    np.allclose(actual, expected)
    for actual, expected in zip(nn.Z_cache, layers_cache)
)
    
    cache_check = "passed" if Zcheck else "failed"
    print(f"Multiple layer cache check: {cache_check}", end="\n")


test_forward()