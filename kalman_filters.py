# Archivo: kalman_filters.py
import numpy as np


class KalmanFilterRegression:
    """
    Implements a recursive Kalman Filter for a linear regression (Model 1).
    Dynamically estimates the states [beta, intercept].

    Measurement Equation: y = H * state + v
    Where:
        y (float): Price of asset A
        H (array 1x2): [Price of asset B, 1]
        state (array 2x1): [beta, intercept]
        v: Measurement noise (R)

    State Transition Equation: state_t = F * state_t-1 + w
    Where:
        F (2x2 matrix): Identity matrix (assumes beta and alpha follow random walks)
        w: Process noise (Q)
    """

    def __init__(self, R, Q, initial_state, initial_covariance):
        """
        Initialize the filter.

        R (float): Variance of the measurement noise (how much we trust the observed price).

        Q (2x2 matrix): Covariance matrix of the process noise (how much we expect beta and alpha to change over time).

        initial_state (2x1 array): Initial values for 
        beta,intercept
        beta,intercept.

        initial_covariance (2x2 matrix): Initial uncertainty associated with the state estimate.
        """
        self.R = R
        self.Q = Q
        self.state_mean = initial_state
        self.state_covariance = initial_covariance

        # Matriz de transición (F) es la identidad
        self.F = np.eye(2)

    def update(self, y, H):
        """
        Updates the Kalman Filter with a new observation (a new day).

        Parameters
        ----------
        y : float
            Price of asset A (the measurement).

        H : array-like (1x2)
            Observation matrix, typically [Price of asset B, 1.0].
        """

        # --- 1. Paso de PREDICCIÓN ---
        # Predecir el estado siguiente (en este caso, es el estado actual)
        predicted_state_mean = self.F @ self.state_mean
        predicted_state_cov = (self.F @ self.state_covariance @ self.F.T) + self.Q

        # --- 2. Paso de ACTUALIZACIÓN ---

        # Calcular el error (innovación)
        y_pred = H @ predicted_state_mean
        innovation = y - y_pred

        # Calcular la covarianza de la innovación
        S = (H @ predicted_state_cov @ H.T) + self.R

        # Calcular la Ganancia de Kalman (K)
        K = predicted_state_cov @ H.T @ np.linalg.inv(S)

        # Actualizar la media del estado (el nuevo [beta, intercepto])
        self.state_mean = predicted_state_mean + (K * innovation)

        # Actualizar la covarianza del estado
        I = np.eye(self.state_mean.shape[0])
        self.state_covariance = (I - (K @ H)) @ predicted_state_cov

    def get_hedge_ratio(self):
        """Returns the current hedge ratio (B1)."""
        return self.state_mean[0, 0]  # El primer elemento del estado (beta)

    def get_intercept(self):
        """Returns the current intercept (alpha)."""
        return self.state_mean[1, 0]