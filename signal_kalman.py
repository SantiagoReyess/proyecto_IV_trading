# signal_kalman.py
import numpy as np

class KalmanSignalMean:
    """
    Kalman Filter unidimensional para la MEDIA del spread.

    Modelo:
        Estado:      mu_t  (media del spread)
        Observación: y_t = spread_t

        mu_t = mu_{t-1} + w_t,   w_t ~ N(0, Q)
        y_t  = mu_t      + v_t,  v_t ~ N(0, R)

    Este filtro suaviza la media del spread. El z-score se calcula como:
        z_t = (spread_t - mu_t) / sigma_t
    donde sigma_t la estimamos con una ventana rolling del spread.
    """

    def __init__(self, mu0: float = 0.0, var0: float = 1.0,
                 q: float = 1e-4, r: float = 1e-2):
        """
        Parámetros:
            mu0  : valor inicial de la media del spread
            var0 : varianza inicial (incertidumbre sobre mu0)
            q    : varianza del ruido de proceso (qué tanto dejamos que cambie mu_t)
            r    : varianza del ruido de medición (ruido del spread observado)
        """
        self.mu = float(mu0)
        self.P = float(var0)
        self.Q = float(q)
        self.R = float(r)

    def update(self, y: float):
        """
        Actualiza el filtro con una nueva observación del spread: y_t.

        y (float): spread observado en el tiempo t.
        """
        y = float(y)

        # --- Predicción ---
        mu_pred = self.mu            # caminata aleatoria
        P_pred = self.P + self.Q

        # --- Actualización ---
        S = P_pred + self.R          # varianza de la innovación
        K = P_pred / S               # ganancia de Kalman (1D)

        innovation = y - mu_pred
        self.mu = mu_pred + K * innovation
        self.P = (1.0 - K) * P_pred

    def get_mean(self) -> float:
        """Devuelve la media filtrada mu_t."""
        return float(self.mu)

    def get_variance(self) -> float:
        """Devuelve la varianza filtrada de la media."""
        return float(self.P)
