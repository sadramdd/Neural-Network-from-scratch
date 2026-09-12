"""
This python file helps reduce redundant code for expriements by building the same neural network

"""


#Change directory for importing 

import sys
from pathlib import Path

PROJECT_ROOT = Path.cwd().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))


#Load neural net
from deep_neural_network.core.main_structure import NeuralNetwork


def build_mnist_model(learning_rate: float= 0.001,
                     optimizer: str= "Adam",
                     hidden_init: str="He"):
    rs = 455


    net = NeuralNetwork(n_features = 784,
                        task = "MultiClassifier",
                        learning_rate = learning_rate,
                        random_state =rs)
    
    
    net.add_layer(64, "ReLU", hidden_init, True)
    net.add_layer(32, "ReLU", hidden_init, True)
    net.add_layer(16, "ReLU", hidden_init, True)
    net.add_layer(10, "Softmax", hidden_init, False)

    net.set_optimizer(optimizer)

    return net