"""
Module: model.py

This module contains the core implementation of the models proposed in the paper:
"Optimizing the trade-off between accuracy and speed in stock price forecasting".

It includes two main classes:
1.  OSELM: An implementation of the Online Sequential Extreme Learning Machine.
    This class serves as the fast and adaptive base learner for the framework.

2.  AdaptiveEnsemble: An implementation of the Adaptive Trust-weighted Multi-model (AT-M)
    framework. This class manages an ensemble of OSELM learners to enhance
    stability and handle concept drift, as described in the paper.
"""
import numpy as np
from collections import deque

class OSELM:
    """
    A complete implementation of the Online Sequential Extreme Learning Machine (OS-ELM).
    This class is compatible with the scikit-learn interface (fit, partial_fit, predict).
    """
    def __init__(self, n_hidden, activation_func='sigmoid', C=1.0):
        """
        Initializes the model's hyperparameters.
        
        Args:
            n_hidden (int): The number of neurons in the hidden layer (parameter L).
            activation_func (str): The activation function for the hidden layer.
            C (float): The regularization coefficient (parameter C).
        """
        self.n_hidden = n_hidden
        self.activation_func = activation_func
        self.C = C
        
        # These attributes will be initialized during training
        self.input_weights_ = None   # Fixed input weights
        self.biases_ = None          # Fixed hidden layer biases
        self.output_weights_ = None  # Updated output weights (beta)
        self.K_ = None               # Matrix K for fast sequential updates
    
    def _sigmoid(self, x):
        """Sigmoid activation function."""
        return 1.0 / (1.0 + np.exp(-x))

    def _calculate_hidden_layer(self, X):
        """Calculates the hidden layer output matrix (H)."""
        pre_activation = X @ self.input_weights_ + self.biases_
        if self.activation_func == 'sigmoid':
            H = self._sigmoid(pre_activation)
        # Other activation functions like 'relu' or 'tanh' can be added here
        return H

    def fit(self, X, y):
        """
        Initial training phase of OS-ELM.
        This method must be called on the first batch of data.
        """
        n_samples, n_features = X.shape
        y = y.reshape(-1, 1) if y.ndim == 1 else y
        
        # 1. Initialize random and fixed input weights and biases
        self.input_weights_ = np.random.randn(n_features, self.n_hidden)
        self.biases_ = np.random.randn(1, self.n_hidden)
        
        # 2. Calculate the initial hidden layer output matrix H_0
        H = self._calculate_hidden_layer(X)
        H_T = H.T
        
        # 3. Calculate the initial matrices K_0 and beta_0
        # K = inv(H^T @ H + I/C)
        # beta = K @ H^T @ T
        I = np.identity(self.n_hidden)
        self.K_ = np.linalg.inv(H_T @ H + I / self.C)
        self.output_weights_ = self.K_ @ H_T @ y
        
        return self

    def partial_fit(self, X, y):
        """
        Sequential (online) learning phase for new batches of data.
        This method is the core of OS-ELM.
        """
        if self.output_weights_ is None:
            raise Exception("Model must be initialized with fit() before partial_fit() can be used.")
        
        y = y.reshape(-1, 1) if y.ndim == 1 else y
        
        # 1. Calculate the hidden layer output matrix for the new data (H_n)
        H = self._calculate_hidden_layer(X)
        H_T = H.T
        
        # 2. Update K and beta recursively using the Woodbury matrix identity
        I = np.identity(H.shape[0])
        term1 = np.linalg.inv(I + H @ self.K_ @ H_T)
        K_new = self.K_ - self.K_ @ H_T @ term1 @ H @ self.K_
        
        error = y - (H @ self.output_weights_)
        
        self.output_weights_ += K_new @ H_T @ error
        self.K_ = K_new # Update K for the next iteration

        return self

    def predict(self, X):
        """Make predictions for new data."""
        if self.output_weights_ is None:
            raise Exception("Model has not been trained yet. Call fit() first.")
            
        H = self._calculate_hidden_layer(X)
        predictions = H @ self.output_weights_
        
        return predictions.flatten()
    
    

from collections import deque

class AdaptiveEnsemble:
    """
    Implements the Adaptive Trust-weighted Multi-model (AT-M) structure.
    This class manages an ensemble of OSELM learners.
    """
    def __init__(self, n_learners, window_size, n_hidden, C=1.0, alpha=0.5):
        """
        Initializes the ensemble.
        
        Args:
            n_learners (int): The number of OS-ELM models in the ensemble (M).
            window_size (int): The length of the sliding window for performance evaluation (W).
            n_hidden (int): Hyperparameter for the base OSELM learners (L).
            C (float): Hyperparameter for the base OSELM learners.
            alpha (float): The trust sensitivity parameter for weight updates (alpha).
        """
        self.n_learners = n_learners
        self.window_size = window_size
        self.n_hidden = n_hidden
        self.C = C
        self.alpha = alpha
        
        # Create the ensemble of OSELM learners
        self.learners = [OSELM(n_hidden=n_hidden, C=C) for _ in range(n_learners)]
        
        # Initialize weights equally for all learners
        self.weights = np.ones(n_learners) / n_learners
        
        # Use a deque (double-ended queue) to efficiently store recent errors for each learner
        self.recent_errors = [deque(maxlen=window_size) for _ in range(n_learners)]

    def fit(self, X, y):
        """
        Initializes and trains all base learners on the first batch of data.
        """
        for learner in self.learners:
            learner.fit(X, y)
        
        # Initialize the error history with the training performance
        for i, learner in enumerate(self.learners):
            predictions = learner.predict(X)
            errors = np.abs(y - predictions)
            for error in errors:
                self.recent_errors[i].append(error)
        
        self._update_weights()
        return self

    def partial_fit(self, X, y):
        """
        Sequentially updates the ensemble with a new batch of data.
        """
        # 1. Update each learner's error history with its performance on the *new* data
        for i, learner in enumerate(self.learners):
            predictions = learner.predict(X)
            # Use mean absolute error for this batch as the error measure
            batch_error = np.mean(np.abs(y - predictions))
            self.recent_errors[i].append(batch_error)
            
        # 2. Update the ensemble weights based on the new performance
        self._update_weights()
        
        # 3. Train each base learner on the new data
        for learner in self.learners:
            learner.partial_fit(X, y)
            
        return self
        
    def _update_weights(self):
        """
        Updates the weights of the learners based on their recent performance.
        This is the "Adaptive Trust" mechanism.
        """
        mean_errors = np.array([np.mean(errors) if len(errors) > 0 else 1.0 for errors in self.recent_errors])
        
        # Performance is the inverse of error (add small epsilon to avoid division by zero)
        epsilon = 1e-8
        performances = 1.0 / (mean_errors + epsilon)
        
        # Apply the trust sensitivity parameter (alpha)
        # Higher alpha means we react more strongly to performance differences
        powered_performances = np.power(performances, self.alpha)
        
        # Normalize the performances to get the final weights
        total_performance = np.sum(powered_performances)
        if total_performance > 0:
            self.weights = powered_performances / total_performance
        else:
            # Fallback to equal weights if all learners fail
            self.weights = np.ones(self.n_learners) / self.n_learners

    def predict(self, X):
        """
        Makes a final prediction by taking a weighted average of all learners' predictions.
        """
        # Get predictions from each learner
        all_predictions = np.array([learner.predict(X) for learner in self.learners])
        
        # Calculate the weighted average
        # (weights are 1D, predictions are 2D, so we need to align them)
        final_prediction = np.dot(self.weights, all_predictions)
        
        return final_prediction  