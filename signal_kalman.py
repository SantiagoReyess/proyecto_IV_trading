# signal_kalman.py
import numpy as np

class KalmanSignalMean:
    """
    One-dimensional Kalman Filter for the MEAN of the spread.

    Model:
        State:        mu_t  (mean of the spread)
        Observation:  y_t = spread_t

            mu_t = mu_{t-1} + w_t,   w_t ~ N(0, Q)
            y_t  = mu_t      + v_t,  v_t ~ N(0, R)

    This filter smooths the mean of the spread. The z-score is computed as:
        z_t = (spread_t - mu_t) / sigma_t
    where sigma_t is estimated with a rolling window of the spread.
    """

    def __init__(self, mu0: float = 0.0, var0: float = 1.0,
                 q: float = 1e-4, r: float = 1e-2):
        """
        Parameters:
            mu0  : initial value of the spread mean
            var0 : initial variance (uncertainty about mu0)
            q    : process noise variance (how much we allow mu_t to change)
            r    : measurement noise variance (noise in the observed spread)
        """
        self.mu = float(mu0)
        self.P = float(var0)
        self.Q = float(q)
        self.R = float(r)

    def update(self, y: float):
        """
        Updates the filter with a new spread observation: y_t.

        y (float): spread observed at time t.
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
        """Returns mu_t"""
        return float(self.mu)

    def get_variance(self) -> float:
        """Returns variace"""
        return float(self.P)
