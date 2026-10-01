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
                  layer) -> None:
        """
        Adds the given layer class into the network sequentially\n
        -------------------------------------------------------------\n
        `layer`: The initialized layer class
        """
        self.previous_a = layer.build(shape_in=self.previous_a) #Actually initializes parameters, returns fan_out
        
        self._layers.append(layer)
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
        weights = [getattr(layer, "_W", None) for layer in self._layers]
        biases =  [getattr(layer, "_b", None) for layer in self._layers]
        reg_parameters = {
                        c: (layer._gamma, layer._beta)
                        for c, layer in enumerate(self._layers)
                        if getattr(layer, "normalize", False)
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
        gradients = []
        
        # Calcutate the derivative of loss
        dA_incoming = self._get_loss_derivative(y_pred, y_true)
        
        for layer_idx in range(len(self._layers) - 1, -1, -1):
            layer = self._layers[layer_idx]
            
            # No need for calculating dA_incoming for first layer 
            calculate_a = layer_idx > 0
            dA_incoming, layer_gradient = layer.backward(dA_incoming, calculate_a=calculate_a)

            layer_gradient["layer_idx"] = layer_idx

            
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

        for layer in self._layers:
            previous_a = layer.forward(previous_a, cache=compute_gradients)

                
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
                
                #Calulating gradients and updating parameters
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