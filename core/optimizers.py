import numpy as np
from typing import Literal

class Optimizer():
    def choose_initialation(self, mode):
        if mode == "SGD":
            return 
        elif mode == "Momentum":
            vlcity_list  = []
            for indx, (weights, biases) in enumerate( zip(self.weights, self.biases) ):
                #Initilize velocity for weights and biases
                velocity_dict = {"vW": np.zeros_like(weights),
                                 "vb": np.zeros_like(biases)} 
                
                #If current layer has batchnorm, initialize velocity for gamma and beta too
                if indx in self.reg_parameters:
                    gamma, beta = self.reg_parameters[indx]
                    
                    velocity_dict["vGamma"] = np.zeros_like(gamma)
                    velocity_dict["vbeta"] = np.zeros_like(beta)
                    
                vlcity_list.append(velocity_dict)
                
            self.velocity = vlcity_list
                    
        elif mode == "RMSProp":
            
            accumulates_list = []

            for indx, (weights, biases) in enumerate( zip(self.weights, self.biases) ):
                accumulate_dict = {"W_accumulated": np.zeros_like(weights),
                                   "b_accumulated": np.zeros_like(biases)}
                
                if indx in self.reg_parameters: #If this layer has batchnorm
                    gamma, beta = self.reg_parameters[indx]
                    
                    accumulate_dict["g_accumulated"] = np.zeros_like(gamma)
                    accumulate_dict["beta_accumulated"] = np.zeros_like(beta)
                accumulates_list.append(accumulate_dict)
                
            self.accumulates = accumulates_list
                    
                   
        elif mode == "Adam":
            vlcity_list  = []
            accumulates_list = []
            
            for indx, (weights, biases) in enumerate( zip(self.weights, self.biases) ):
                #Initilize velocity for weights and biases
                velocity_dict = {"vW": np.zeros_like(weights),
                                 "vb": np.zeros_like(biases)} 
                
                #Initilize accumulations for weights and biases
                accumulate_dict = {"W_accumulated": np.zeros_like(weights),
                                 "b_accumulated": np.zeros_like(biases)} 
                
                #If current layer has batchnorm, initialize velocity/accumulations for gamma and beta too
                if indx in self.reg_parameters:
                    gamma, beta = self.reg_parameters[indx]
                    
                    velocity_dict["vGamma"] = np.zeros_like(gamma)
                    velocity_dict["vbeta"] = np.zeros_like(beta)
                    
                    accumulate_dict["g_accumulated"] = np.zeros_like(gamma)
                    accumulate_dict["beta_accumulated"] = np.zeros_like(beta)
                    
                #Save to lists
                vlcity_list.append(velocity_dict)
                accumulates_list.append(accumulate_dict)
                
            #Store to class attribute
            self.velocity = vlcity_list
            self.accumulates = accumulates_list
            
            #Step counter nedeed for bias correction
            self.t = 0
            
        else:
            raise ValueError(
                f"Invalid optimizer selection: {self.mode}"
            )
            
            
    def __init__(self,
                 optimizer: Literal["SGD", "Momentum", "RMSProp", "Adam"],
                 weights: list,
                 biases: list,
                 reg_parameters:dict,
                 learning_rate: float,
                 beta1: float=0.9,
                 beta2: float=0.999,
                 epsilon: float=10e-10) -> None:
        self.mode = optimizer
        
        self.weights = weights
        self.biases = biases
        
        self.reg_parameters = reg_parameters #{ln: (gamma,beta), ln: (gamma,beta)}
        
        self.b1 = beta1
        self.b2 = beta2
        self.e = epsilon
        
        self.learning_rate = learning_rate #Save for Updating
        
        self.choose_initialation(optimizer)# Initialize optimization parameters
        
    def _sgd(self,
            gradients: list):
        #Perform SGD optimzers update on current w and b with given gradients
        new_weights = list()
        new_biases = list()
        
        new_reg_paramters = dict()
        for c ,(gradients_l, weight, bias) in enumerate( zip(gradients, self.weights, self.biases) ):
            
            new_weight = weight - (self.learning_rate * gradients_l["dW"])
            new_bias = bias - (self.learning_rate * gradients_l["db"])
            
            #If this layer has batchnorm, update gamma and beta
            if c in self.reg_parameters:
                gamma, beta = self.reg_parameters[c]
                new_gamma = gamma - (self.learning_rate * gradients_l["dgamma"])
                new_beta  = beta - (self.learning_rate * gradients_l["dbeta"])
                
                new_reg_paramters[c] = (new_gamma, new_beta)
            
            
            new_weights.append(new_weight)
            new_biases.append(new_bias)
        
        return new_weights, new_biases, new_reg_paramters
    
    def _momentum(self,
                  gradients: list):
        new_weights = list()
        new_biases = list()
        
        new_reg_paramters = dict()
        #Calculate new velocity W and b, according to gradients, previous velocity and Beta (Hyperparameter)
        for c, (gradients_l, weights_l , bias_l, velocity_dict_l) in enumerate(
                zip(gradients,self.weights,self.biases,self.velocity)
        ):
            
            #Update velocity
            self.velocity[c]["vW"] = (self.b1 * velocity_dict_l["vW"]) + (1 - self.b1) * gradients_l["dW"]
            self.velocity[c]["vb"] = (self.b1 * velocity_dict_l["vb"]) + (1 - self.b1) * gradients_l["db"]
            
            #Update parameters
            new_w = weights_l - self.learning_rate * self.velocity[c]["vW"]
            new_b = bias_l - self.learning_rate * self.velocity[c]["vb"]
            
            #If this layer has batchnorm, update gamma and beta
            if c in self.reg_parameters:
                gamma, beta = self.reg_parameters[c]
                
                #Update gamma and beta velocity
                self.velocity[c]["vGamma"] = (self.b1 * velocity_dict_l["vGamma"]) + (1 - self.b1) * gradients_l["dgamma"]
                self.velocity[c]["vbeta"] = (self.b1 * velocity_dict_l["vbeta"]) + (1 - self.b1) * gradients_l["dbeta"]
                
                new_gamma = gamma - (self.learning_rate * self.velocity[c]["vGamma"])
                new_beta  = beta - (self.learning_rate * self.velocity[c]["vbeta"])
                
                new_reg_paramters[c] = (new_gamma, new_beta)
            
            new_weights.append(new_w)
            new_biases.append(new_b)
            
        return new_weights, new_biases, new_reg_paramters
        
    def _rmsprop(self,
                 gradients: list):
        #Update accumulated squares for weigths
        new_weights = list()
        new_biases = list()
        
        new_reg_parameters = dict()
        
        for c, (gradients_l, weights_l, bias_l, accumulates_dict_l) in enumerate(
            zip(gradients, self.weights, self.biases, self.accumulates)
        ):
            #Update accumulations
            self.accumulates[c]["W_accumulated"] = (np.multiply(self.b2, accumulates_dict_l["W_accumulated"]) 
                                + (1 -  self.b2)
                                * np.square(gradients_l["dW"]))
            self.accumulates[c]["b_accumulated"] = (np.multiply(self.b2, accumulates_dict_l["b_accumulated"]) 
                                + (1 - self.b2)
                                * np.square(gradients_l["db"]))
            
            #Update parameters
            new_w = (weights_l 
                    - np.divide( self.learning_rate, np.sqrt(self.accumulates[c]["W_accumulated"] + self.e) )
                    * gradients_l["dW"])
            new_b= (bias_l 
                    - np.divide( self.learning_rate, np.sqrt(self.accumulates[c]["b_accumulated"] + self.e) )
                    * gradients_l["db"])
            
            #If this layer has batchnorm, update gamma and beta
            if c in self.reg_parameters:
                gamma, beta = self.reg_parameters[c]
                
                #Update gamma and beta accumulation
                self.accumulates[c]["g_accumulated"] = ( (self.b2 * accumulates_dict_l["g_accumulated"]) 
                                                        + 
                                                        (1 - self.b2) * np.square(gradients_l["dgamma"]) )
                
                self.accumulates[c]["beta_accumulated"] = ( (self.b2 * accumulates_dict_l["beta_accumulated"]) 
                                                          + 
                                                          (1 - self.b2)
                                                          * np.square(gradients_l["dbeta"]))
                
                #Update parameters 
                
                new_gamma = (
                    gamma
                    - self.learning_rate
                    * gradients_l["dgamma"]
                    / np.sqrt(self.accumulates[c]["g_accumulated"] + self.e)
                )
                new_beta = (
                    beta
                    - self.learning_rate
                    * gradients_l["dbeta"]
                    / np.sqrt(self.accumulates[c]["beta_accumulated"] + self.e)
                )
                
                new_reg_parameters[c] = (new_gamma, new_beta)
            
            new_weights.append(new_w)
            new_biases.append(new_b)
        
        return new_weights, new_biases, new_reg_parameters
    
    def _adam(self,
              gradients: list):
        #Add to counter FIRST
        self.t +=1
        
        new_weights = list()
        new_biases = list()
        
        new_reg_paramters = dict()
        
        for c, (gradients_dict, weights, bias, velocity ,accumulates_dict) in enumerate(
            zip(gradients, self.weights, self.biases, self.velocity ,self.accumulates)
        ):
            #Update First moment: momentum (m)
            self.velocity[c]["vW"] = (np.multiply(self.b1, velocity["vW"])
                    + (1 -  self.b1)
                    * gradients_dict["dW"])  
            self.velocity[c]["vb"] = (np.multiply(self.b1, velocity["vb"])
                    + (1 -  self.b1)
                    * gradients_dict["db"])
            
            #Update Second moment: variance (v)
            self.accumulates[c]["W_accumulated"] = (np.multiply(self.b2, accumulates_dict["W_accumulated"])
                    + (1 - self.b2)
                    * np.square(gradients_dict["dW"]))
            self.accumulates[c]["b_accumulated"] = (np.multiply(self.b2, accumulates_dict["b_accumulated"])
                    + (1 - self.b2)
                    * np.square(gradients_dict["db"]))
            
            #Calculate s hats for calculation
            w_s_hat = np.divide(self.accumulates[c]["W_accumulated"], 1 - np.power(self.b2, self.t))
            b_s_hat = np.divide(self.accumulates[c]["b_accumulated"], 1 - np.power(self.b2, self.t))
            
            #Calculate v hats for calculation
            w_m_hat = np.divide(self.velocity[c]["vW"], 1 - np.power(self.b1, self.t))
            b_m_hat = np.divide(self.velocity[c]["vb"], 1 - np.power(self.b1, self.t))
            
            #Update parameters
            new_w = (weights
                        - self.learning_rate
                        * w_m_hat
                        / np.sqrt(w_s_hat + self.e))
            
            new_b = (bias
                        - self.learning_rate
                        * b_m_hat
                        / np.sqrt(b_s_hat + self.e))
            
            new_weights.append(new_w)
            new_biases.append(new_b)
            
            #If this layer has batchnorm, update its gamma/beta velocities/accumulations and update
            if c in self.reg_parameters:
                self.velocity[c]["vGamma"] = (np.multiply(self.b1, velocity["vGamma"])
                        + (1 -  self.b1)
                        * gradients_dict["dgamma"])  
                
                self.velocity[c]["vbeta"] = (np.multiply(self.b1, velocity["vbeta"])
                        + (1 -  self.b1)
                        * gradients_dict["dbeta"])
                
                #Update Second moment: variance (v)
                self.accumulates[c]["g_accumulated"] = (np.multiply(self.b2, accumulates_dict["g_accumulated"])
                        + (1 - self.b2)
                        * np.square(gradients_dict["dgamma"]))
                
                self.accumulates[c]["beta_accumulated"] = (np.multiply(self.b2, accumulates_dict["beta_accumulated"])
                        + (1 - self.b2)
                        * np.square(gradients_dict["dbeta"]))
                
                #Calculate s hats for calculation
                gamma_s_hat = np.divide(self.accumulates[c]["g_accumulated"], 1 - np.power(self.b2, self.t))
                beta_s_hat = np.divide(self.accumulates[c]["beta_accumulated"], 1 - np.power(self.b2, self.t))
                
                #Calculate v hats for calculation
                gamma_m_hat = np.divide(self.velocity[c]["vGamma"], 1 - np.power(self.b1, self.t))
                beta_m_hat = np.divide(self.velocity[c]["vbeta"], 1 - np.power(self.b1, self.t))
                
                #Update gamma and beta for the layer 
                new_gamma = (self.reg_parameters[c][0] #c is for layers index and 0 is for gamma
                            - self.learning_rate
                            * gamma_m_hat
                            / np.sqrt(gamma_s_hat + self.e))
                
                new_beta = (self.reg_parameters[c][1] #c is for layers index and 1 is for beta
                            - self.learning_rate
                            * beta_m_hat
                            / np.sqrt(beta_s_hat + self.e))
                new_reg_paramters[c] = (new_gamma, new_beta)
                
        
        return new_weights, new_biases, new_reg_paramters
        
    def update(self, gradients):
        if self.mode == "SGD":
            self.weights, self.biases, self.reg_parameters = self._sgd(gradients)
            return self.weights, self.biases, self.reg_parameters 
        
        elif self.mode == "Momentum":
            self.weights, self.biases, self.reg_parameters = self._momentum(gradients)
            return self.weights, self.biases, self.reg_parameters 
        
        elif self.mode == "RMSProp":
            self.weights, self.biases, self.reg_parameters = self._rmsprop(gradients)
            return self.weights, self.biases, self.reg_parameters 
        
        elif self.mode == "Adam":
            self.weights, self.biases, self.reg_parameters = self._adam(gradients)
            return self.weights, self.biases, self.reg_parameters 
        else:
            raise ValueError(
                f"Invalid optimizer selection: {self.mode}"
            )