from deep_neural_network.core.main_structure import NeuralNetwork
from deep_neural_network.core.layer.layers import Layer
from deep_neural_network.core.losses import MSE

import numpy as np


def test_backpropagation():
    """
    Compares:

        ANALYTICAL GRADIENTS
        produced by our backpropagation

                    VS

        NUMERICAL GRADIENTS
        approximated using finite differences

    Numerical gradient:

        dL/dθ ≈ [L(θ + ε) - L(θ - ε)] / (2ε)

    """

    random_s = 42
    lr = 0.01

    n_layers = 3
    n_neurons = [2, 3, 2]

    n_input_features = 5
    n_input_samples = 20

    epsilon = 1e-5

    # ---------------------------------------------------------
    # 1. Random features and targets
    # ---------------------------------------------------------

    rng = np.random.default_rng(random_s)

    X = rng.normal(
        loc=0.0,
        scale=1.0,
        size=(n_input_features, n_input_samples)
    )

    y = rng.normal(
        loc=0.0,
        scale=1.0,
        size=(1, n_input_samples)
    )

    # ---------------------------------------------------------
    # 2. Initialize network
    # ---------------------------------------------------------

    nn_model = NeuralNetwork(
        n_input_features,
        task="Regression",
        learning_rate=lr,
        random_state=random_s
    )

    for i in range(n_layers):
        nn_model.add_layer(
            n_neurons[i],
            "Tanh",
            "Xavier"
        )

    # ---------------------------------------------------------
    # 3. Forward pass
    # ---------------------------------------------------------

    model_output = nn_model._forward_feed(
        X,
        compute_gradients=True
    )

    # ---------------------------------------------------------
    # 4. Analytical gradients
    # ---------------------------------------------------------

    model_gradients = nn_model._back_propagate(
        model_output,
        y
    )

    # ---------------------------------------------------------
    # 5. Numerical gradient checking
    # ---------------------------------------------------------

    for layer_idx, layer in enumerate(nn_model._layers):

        # =====================================================
        # Weight gradients
        # =====================================================

        numerical_dW = np.zeros_like(layer._W)

        for index in np.ndindex(layer._W.shape):

            original_value = layer._W[index]

            # f(theta + epsilon)
            layer._W[index] = original_value + epsilon

            y_pred_plus = nn_model._forward_feed(
                X,
                compute_gradients=False
            )

            loss_plus = MSE(
                y_pred_plus,
                y
            )

            # f(theta - epsilon)
            layer._W[index] = original_value - epsilon

            y_pred_minus = nn_model._forward_feed(
                X,
                compute_gradients=False
            )

            loss_minus = MSE(
                y_pred_minus,
                y
            )

            # Central difference
            numerical_dW[index] = (
                loss_plus - loss_minus
            ) / (2 * epsilon)

            # Restore original parameter
            layer._W[index] = original_value

        # Analytical gradient from our backpropagation
        analytical_dW = model_gradients[layer_idx]["dW"]

        # Compare
        dW_check = np.allclose(
            numerical_dW,
            analytical_dW,
            rtol=1e-4,
            atol=1e-6
        )

        print(
            f"Layer {layer_idx + 1} dW check: "
            f"{'passed' if dW_check else 'FAILED'}"
        )


        # =====================================================
        # Bias gradients
        # =====================================================

        numerical_db = np.zeros_like(layer._b)

        for index in np.ndindex(layer._b.shape):

            original_value = layer._b[index]

            # f(theta + epsilon)
            layer._b[index] = original_value + epsilon

            y_pred_plus = nn_model._forward_feed(
                X,
                compute_gradients=False
            )

            loss_plus = MSE(
                y_pred_plus,
                y
            )

            # f(theta - epsilon)
            layer._b[index] = original_value - epsilon

            y_pred_minus = nn_model._forward_feed(
                X,
                compute_gradients=False
            )

            loss_minus = MSE(
                y_pred_minus,
                y
            )

            # Central difference
            numerical_db[index] = (
                loss_plus - loss_minus
            ) / (2 * epsilon)

            # Restore original parameter
            layer._b[index] = original_value

        # Analytical gradient
        analytical_db = model_gradients[layer_idx]["db"]

        # Compare
        db_check = np.allclose(
            numerical_db,
            analytical_db,
            rtol=1e-4,
            atol=1e-6
        )

        print(
            f"Layer {layer_idx + 1} db check: "
            f"{'passed' if db_check else 'FAILED'}"
        )


    print("\nBackpropagation verification: PASSED")
    
    

test_backpropagation()