import numpy as np
from numpy.typing import NDArray
from typing import Literal, Any


from .losses import MSE, cross_entropy

class Evaluater():
    
    def __init__(self,
                 task: Literal["Regression", "BinClassifier", "MultiClassifier"],
                 scaler: Any = None,
                 e: float = 1e-8) -> None:
        self.target_scaler = scaler
        self.task = task
        
        self.e = e

    def _get_loss(self,
                y_pred: NDArray,
                y_true: NDArray):
        if self.task == "Regression":
            return MSE(y_pred, y_true)
        else:
            return cross_entropy(y_pred, y_true)
        
    def _classification_report(self,
                              preds: NDArray,
                              y_test: NDArray):
        """
        This method can be only used for classification tasks\n
        `X_test`, `y_test`: takes testing samples and corresponding labels, make prediction and return `accuracy`, `loss`
        """
        
        if self.target_scaler:
            #If we have a scaler, get the real predited and true values, THEN calculate metrics
            preds = self.target_scaler.inverse_transform(preds)
            y_test = self.target_scaler.inverse_transform(y_test)
        
        y_true_S = np.argmax(y_test, axis=0)
        y_pred = np.argmax(preds, axis=0)

        accuracy = np.mean(y_true_S == y_pred)
        loss = self._get_loss(preds, y_test)
        
        return accuracy, loss
    
    def _regression_report(self,
                          preds,
                          y_test):
        """
        This method can be only used for regression tasks\n
        `X_test`, `y_test`: takes testing samples and corresponding labels, make prediction and return `mape`, `loss`  
        """
        
        if self.target_scaler:
            #If we have a scaler, get the real predited and true values, THEN calculate metrics
            preds = self.target_scaler.inverse_transform(preds)
            y_test = self.target_scaler.inverse_transform(y_test)
            
        mape = np.mean(np.abs((y_test - preds) / (y_test + self.e)))
        loss = self._get_loss(preds, y_test)
        
        return mape, loss
    
    def evaluate(self, preds, y_test):
        #Simply a decision/helper function for evaluating the model
        if self.task == "Regression":
            return self._regression_report(preds, y_test)
            
        else:
            return self._classification_report(preds, y_test)