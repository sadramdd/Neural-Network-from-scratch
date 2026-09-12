#Importing needed class for neural network (building blocks)
from .evaluaters import Evaluater
from .layer.layers import Layer
from .optimizers import Optimizer
from .losses import mse_derivative, cross_entropy_derivative

#numpy as key calculations library
import numpy as np
from typing import Literal
from numpy.typing import ArrayLike, NDArray

    
class NeuralNetwork():
            
    def __init__(self,
                 n_features: int,
                 task: Literal["Regression", "BinClassifier", "MultiClassifier"],
                 learning_rate: float=0.01,
                 target_transformer= None,
                 random_state=None):
        #target_transformer has to have inverse_transform method
        """
        #Simple feed forward neural network.#\n
        ------------------------------------------------###features:\n
        1-this network uses ReLU, sigmoid, tanh and softmax as activation functions\n
        2-the class needs number of features and number of classes and the initiallization\n
        3-user will add layers by `add_layer` method one by one, specifying the number of neurons and activation function\n
        4-uses mini batching and SGD for training, montiors loss.\n
        ------------------------------------------------###downsides\n
        1-It can not figure the features shape and number of classes on it's own and its a requirement in initiallization\n
        2-limited activation function choises\n
        3-limited evaluation metrics (loss, accuracy/MAPE)\n
        
        Note: The model works wth input shape: (n_features, n_samples) meaning that each row represents features and columns represent
        each sample of the input dataset. So be careful with shapes.
        """
        
        #Basic Neural Network Hyperparameters
        self._a = learning_rate
        self.n_features = n_features
        
        
        self.task = task # save Models goal/task
        
        self._layers = [] # Saving layers
    
        self.previous_a = n_features # Saving previous layer output dimension for adding layers
        
        self.evaluater = Evaluater(self.task, target_transformer)
        
        if random_state:
            np.random.seed(random_state)
            self.random_state = random_state
        
    #User uses this method to add layers one by one
    def add_layer(self,
                  n_neuron: int,
                  activation: Literal["ReLU", "Sigmoid", "Tanh", "Softmax", "None"],
                  kernel_init: Literal["He" ,"Xavier"],
                  normalize: bool=False) -> None:
        """
        Adds layer with given details to the network sequentially\n
        -------------------------------------------------------------\n
        `n_neuron`: how many neurons this layer has\n
        `activation`: what activation this layer uses
        """
        
        last_n_neuron = self.previous_a
        
        self._layers.append(Layer(
            n_neurons=n_neuron,
            fan_in=last_n_neuron,
            activation_function=activation,
            kernel_initialization=kernel_init,
            learning_rate=self._a,
            random_state=self.random_state,
            normalize=normalize
                                 ))
        
        self.previous_a = n_neuron
        
    def set_optimizer(self,
                      optimizer: Literal["SGD", "Momentum", "RMSProp", "Adam"]):
        """
        Initializes and saves a optimizer for the models training\n
        INPUTS\n
        -----------------------\n
        `optimizer`: Code for needed optimizer
        
        RETURNS\n
        ----------------------\n
        None: Just initailizes and saves the optimizer
        """
        
        #Store models weights in a flat list in oder to send it to the Optimizer
        weights = [layer._W for layer in self._layers]
        biases =  [layer._b for layer in self._layers]
        reg_parameters = {
                        c: (layer._gamma, layer._beta)
                        for c, layer in enumerate(self._layers)
                        if layer.normalize
                        }
                
        #Initialize Save the optimizer into models attributes
        self.optimizer = Optimizer(optimizer,
                                   weights,
                                   biases,
                                   reg_parameters,
                                   self._a,
                                   )
    
    def _get_loss_derivative(self,
                            y_pred: NDArray,
                            y_true: NDArray): 
        """
        Finds the corresponding cost function according to model's task and directs
        """
        if self.task == "Regression":
            return mse_derivative(y_pred, y_true)
        else:
            return cross_entropy_derivative(y_pred, y_true)
            
    
    def _back_propagate(self, y_pred, y_true):
        """
        compute gradients for all layers using backpropagation
        
        steps:
        1. start with dL/d_pred (loss derivative)
        
        2. for each layer (backwards):
           - Compute dL/dZ = dL/dA * dA/dZ (chain rule with activation)
           - Compute dL/dW = dL/dZ @ A_prev.T / m
           - Compute dL/db = sum(dL/dZ) / m
           - Compute dL/dA_prev = W.T @ dL/dZ (for next iteration)
        
        returns:
            list of dicts: gradients for each layer
                {'dW': dl/dW, 'db': dl/db}
        """
        
        #step 1: Get loss derivative (dL/dŷ)
        dL_dA = self._get_loss_derivative(y_pred, y_true)
        
        #step 2: Initialize storage for gradients
        gradients = []
        #m_samples = y_pred.shape[1]  # number of samples
        
        #step 3: Iterate backwards through layers
        for layer_idx in range(len(self._layers) - 1, -1, -1):
            #Get the layer through index
            layer = self._layers[layer_idx]
            
            #get cached values from forward pass
            A_prev = self.A_cache[layer_idx]  #input to this layer
            Z = self.Z_cache[layer_idx]       #pre-activation: Z = W @ A_prev + b
            
            
            #Activation backward:
            if layer._act_func == "Softmax":
                # Special case: softmax + cross-entropy 
                dL_dZ_bn = dL_dA
            else:
                activation_deriv = layer._call_activation_derivative(layer._act_func, Z)
                dL_dZ_bn = dL_dA * activation_deriv
                
            #Batchnorm parameters gradients
            if layer.normalize:

                #bn_cache = layer._batchnorm_cache
                bn_cache = self.BN_cache[layer_idx]

                X_norm = bn_cache["X_norm"]
                inv_std = bn_cache["inv_std"]

                m = dL_dZ_bn.shape[1]

                # gamma gradient
                dL_dgamma = np.sum(
                    dL_dZ_bn * X_norm,
                    axis=1,
                    keepdims=True
                )

                # beta gradient
                dL_dbeta = np.sum(
                    dL_dZ_bn,
                    axis=1,
                    keepdims=True
                )

                # Backprop through:
                #
                # Z_bn = gamma * X_norm + beta

                dL_dX_norm = dL_dZ_bn * layer._gamma

                # Backprop through normalization
                dL_dZ = (
                    inv_std / m
                    * (
                        m * dL_dX_norm
                        - np.sum(
                            dL_dX_norm,
                            axis=1,
                            keepdims=True
                        )
                        - X_norm
                        * np.sum(
                            dL_dX_norm * X_norm,
                            axis=1,
                            keepdims=True
                        )
                    )
                )

            else:

                dL_dZ = dL_dZ_bn
                dL_dgamma = None
                dL_dbeta = None
                        
            
            #Linear parameters gradients:
            
            #Weight gradients
            dL_dW = (dL_dZ @ A_prev.T)
            
            #Bias gradients
            dL_db = np.sum(dL_dZ, axis=1, keepdims=True)
            
            #If there is a previous layer: calculate dl/da (just we did in first )
            if layer_idx > 0:  
                dL_dA = layer._W.T @ dL_dZ
                
            #Store gradients
            layer_gradient = {
            "dW": dL_dW,
            "db": dL_db,
            "layer_idx": layer_idx
                }

            #If layer has batch normalization, also add dl_dgamma and dl_dbeta
            if layer.normalize:
                layer_gradient["dgamma"] = dL_dgamma
                layer_gradient["dbeta"] = dL_dbeta

            #add to gradients list (we'll reverse at the end since we went backwards)
            gradients.append(layer_gradient)
            
        # reverse because we computed in reverse order
        gradients.reverse()
        
        return gradients
        
    def _forward_feed(self,
                     x:ArrayLike,
                     compute_gradients: bool= True):
        """
        gets the input and runs if throught model weights, computing\n1- `z=Wx + b` \n2- `a= f(z)` in each layer\n
        INPUTS\n
        -------------------------\n
        `x`: the features and inputs to feed the model\n
        `compute_gradients`: if True, saves the cached `a` and `z` in attributes: `A_cache` and `Z_cache`
        (crucial for backpropagations and computing gradients, better to turn it off when predicting)\n
        RETURNS:\n
        the output (last a)
        """
        previous_a = x
        
        if compute_gradients:
            self.A_cache = []
            self.Z_cache = []
            self.BN_cache = [] #List of dict caches from Batchnorm layers
        
        n_layers = len(self._layers)
        for layer_ind in range(n_layers):
            if compute_gradients:
                self.A_cache.append(previous_a)
                
                z, previous_a, batchnorm_cache = self._layers[layer_ind].forward(previous_a, cache=True)
                self.Z_cache.append(z)
                self.BN_cache.append(batchnorm_cache)
                
            else:
                previous_a = self._layers[layer_ind].forward(previous_a, cache=False)
                
            
        return previous_a
    
    def _gradient_descent(self, gradients):
        """
        Gets the computed gradients from backpropagations\n
        calls the optimizer to update its weights and biases\n
        then loops through each layer, changing weights and biases
        """
        #Get updated weights and biases from optimizer
        new_weights, new_biases, new_reg_parameters = self.optimizer.update(gradients)
        
        #Loop through the network and assign new weight and biases to each layer
        for c, layer in enumerate(self._layers):
            layer._W = new_weights[c]
            layer._b = new_biases[c]
            
            #If layer has batchnorm, set new gamma and beta
            if c in new_reg_parameters:
                new_gamma, new_beta = new_reg_parameters[c]
                layer._gamma = new_gamma
                layer._beta = new_beta
                
    def fit(self,
            X: NDArray,
            y: NDArray,
            epoch: int=100,
            batch_size: int=32,
            monitor_every_n: int =10):
        """
        
        Training the neural network with given X and y\n
        --------------------------------------------------\n
        `epoch`: indicates how many times the model sees all samples and update parameters based on it\n
        `batch_size`: training is done in batches, on every iteration the model gets updated on n batches \n
        `monitor_every_n`: on every n epoches, prints the loss\n
        ---------------------------------------------------\n
        returns: nothing (only trains the model and prints loss)
        
        """
        #Getting number of samples
        n_samples = X.shape[1]
        #Looping through epoches
        for e in range(epoch):
            #Creating shuffled indeces
            indeces = np.random.permutation(n_samples)
            loss_sum = 0
            
            #Looping through to get 32 samples at each iteration
            for i in range(0, n_samples, batch_size):
                batch_index = indeces[i: i+batch_size]
                
                #Pulling out 32 samples of the shuffled indices
                X_batch = X[:, batch_index]
                Y_batch = y[:, batch_index] # y[:, batch_index]
                
                #Forward feed
                Y_hat = self._forward_feed(X_batch,
                                          compute_gradients=True)
                
                #Calculating loss
                loss = self.evaluater._get_loss(Y_hat, Y_batch)
                loss_sum += loss * Y_batch.shape[1]
                
                #Calulating gradients and updating parameters (W and b)
                gradients = self._back_propagate(Y_hat, Y_batch)
                self._gradient_descent(gradients)
                
            #monitoring loss 
            if monitor_every_n and (e + 1) % monitor_every_n == 0:
                Y_hat = self._forward_feed(X, compute_gradients=False)
                epoch_loss = self.evaluater._get_loss(Y_hat, y)
                
                print(f"epoch: {e + 1}, loss: {epoch_loss}")
            
    def predict(self,
                X:NDArray):
        """
        Just the forward feed with no gradient computation\n
        RETURNS\n
        the prediction
        """
        return self._forward_feed(X,
                                 compute_gradients=False)
        
    def evaluate(self,
                 X_test: ArrayLike,
                 y_test: ArrayLike):
        y_pred = self._forward_feed(X_test, compute_gradients=False)
        #Pass to the evaluater
        
        evaluated = self.evaluater.evaluate(y_pred,
                                y_test)
        
        
        return evaluated