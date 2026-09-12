from deep_neural_network.core.losses import *
import numpy as np


def test_mse():
    #1-Implementation test
    y_true = np.array([[1., 2., 3.]]) #example actuall targets
    y_pred = np.array([[2., 4., 1.]]) #example predicted targets
    calculated_loss = np.mean((y_true - y_pred) ** 2)
    
    functions_loss = MSE(y_pred, y_true)
    
    print("1- MSE: implementation test")
    is_passed = "passed" if np.isclose(calculated_loss, functions_loss) else "failed"
    print(is_passed, end="\n\n")
    
    #2- Derivative test
    expected_derivative = np.array([
    [2/3, 4/3, -4/3]
    ])#Hand calculated derivative of examples

    actual_derivative = mse_derivative(
        y_pred,
        y_true
    )#Functions estimated derivative
    

    print("2- MSE: derivative test")
    is_passed = "passed" if np.allclose(actual_derivative, expected_derivative) else "failed"
    print(is_passed, end="\n\n")
    


def test_cross_entropy():
    
    #1- Testing implementation
    y_true = np.array([[1.]]) #Example prediction and actual target
    y_pred = np.array([[0.8]])
    
    expected_ce = -np.log(y_pred)
    actual_ce = cross_entropy(y_pred, y_true)
    
    y_true2 = np.array([[0.]])
    y_pred2 = np.array([[0.2]])
    
    actual_ce2 = cross_entropy(y_pred2, y_true2)#Same answer but diffrent values (sanity check)
    
    is_passed = "passed" if np.isclose(expected_ce, actual_ce) and np.isclose(expected_ce, actual_ce2)  else "failed"
    print("1- Cross entropy: implementation test")
    print(is_passed, end="\n\n")
    
    #2- Categorical cross entropy check
    y_true_cat = np.array([
        [1., 0.],
        [0., 1.],
        [0., 0.]
    ]) #Example for categorical cross entropy

    y_pred_cat = np.array([
        [0.8, 0.1],
        [0.1, 0.7], 
        [0.1, 0.2]
    ])#predicted probabilities for each class
    
    expected_cat_loss = ( -np.log(0.8) + -np.log(0.7) ) / 2
    actual_cat_loss = cross_entropy(y_pred_cat, y_true_cat)
    is_passed = "passed" if np.isclose(expected_cat_loss, actual_cat_loss) else "failed"
    print("2- Categorical Cross entropy: implementation test")
    print(is_passed, end="\n\n")
    
def simple_sanity_check():
    #1- MSE sanity check
    mse_y_true = np.array([[100]])
    
    mse_y_pred1 = np.array([[90]])
    mse_y_pred2 = np.array([[10]])  
     
    lower_loss = MSE(mse_y_pred1, mse_y_true)
    higher_loss = MSE(mse_y_pred2, mse_y_true)
    
    
    print("1- mse sanity check")
    is_passed = "passed" if lower_loss < higher_loss else "failed"
    print(is_passed, end="\n\n")
    
    #2- Binary cross entropy sanity check
    
    ce_y_true = np.array([[1.0]])
    
    ce_y_pred1 = np.array([[0.9]])
    ce_y_pred2 = np.array([[0.1]])  
     
    lower_loss = cross_entropy(ce_y_pred1, ce_y_true)
    higher_loss = cross_entropy(ce_y_pred2, ce_y_true)
    
    
    print("2- bce sanity check")
    is_passed = "passed" if lower_loss < higher_loss else "failed"
    print(is_passed, end="\n\n")
    
    #3- Categorical binary cross entropy sanity check
    
    cat_y_true = np.array([[0.0,0.0,0.0],
                          [1,0.0,0.0],
                          [0.0,1,1]])
    
    cat_y_pred1 = np.array([[0.05,0.2,0.1],
                          [0.9,0.1,0.1],
                          [0.05,0.7,0.8]])
    
    cat_y_pred2 = np.array([[0.15,0.05,0.1],
                          [0.8,0.35,0.2],
                          [0.05,0.6,0.7]])
     
    lower_loss = cross_entropy(cat_y_pred1, cat_y_true)
    higher_loss = cross_entropy(cat_y_pred2, cat_y_true)
    
    
    print("3- categorical ce sanity check")
    is_passed = "passed" if lower_loss < higher_loss else "failed"
    print(is_passed, end="\n\n")

test_mse()
test_cross_entropy()
simple_sanity_check()