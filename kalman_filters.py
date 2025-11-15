# Archivo: kalman_filters.py
import numpy as np


class KalmanFilterRegression:
    """
    Implementa un Filtro de Kalman recursivo para una regresión lineal (Modelo 1).
    Estima los estados [beta, intercepto] de forma dinámica.

    Ecuación de Medición: y = H * state + v
    Donde:
        y (float): Precio de A
        H (array 1x2): [Precio de B, 1]
        state (array 2x1): [beta, intercepto]
        v: Ruido de medición (R)

    Ecuación de Transición: state_t = F * state_t-1 + w
    Donde:
        F (matrix 2x2): Matriz de identidad (asume que beta y alfa son caminatas aleatorias)
        w: Ruido del proceso (Q)
    """

    def __init__(self, R, Q, initial_state, initial_covariance):
        """
        Inicializa el filtro.

        R (float): Varianza del ruido de medición (qué tanto confiamos en el precio).
        Q (matrix 2x2): Matriz de covarianza del ruido del proceso (qué tanto esperamos que cambien beta y alfa).
        initial_state (array 2x1): Valores iniciales para [beta, intercepto].
        initial_covariance (matrix 2x2): Incertidumbre inicial sobre el estado.
        """
        self.R = R
        self.Q = Q
        self.state_mean = initial_state
        self.state_covariance = initial_covariance

        # Matriz de transición (F) es la identidad
        self.F = np.eye(2)

    def update(self, y, H):
        """
        Actualiza el filtro con un nuevo dato (un nuevo día).

        y (float): Precio del activo A (la medición)
        H (array 1x2): [Precio del activo B, 1.0] (la matriz de observación)
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
        """Retorna el hedge ratio (B1) actual."""
        return self.state_mean[0, 0]  # El primer elemento del estado (beta)

    def get_intercept(self):
        """Retorna el intercepto (alfa) actual."""
        return self.state_mean[1, 0]