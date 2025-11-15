# backtesting_two_kalman.py
import numpy as np
import pandas as pd
import statsmodels.api as sm
from dataclasses import dataclass

from kalman_filters import KalmanFilterRegression
from signal_kalman import KalmanSignalMean


@dataclass
class Operation:
    stock: str
    price: float   # precio de entrada (niveles, en dólares)
    shares: float  # número de acciones


@dataclass
class Trade:
    entry_date: pd.Timestamp
    exit_date: pd.Timestamp
    side: str                  # "long_spread" o "short_spread"
    pnl_gross: float
    pnl_net: float
    return_pct: float
    holding_period: int        # en días
    commissions: float


def backtesting(
    dataframe_raw: pd.DataFrame,
    stock_A: str,
    stock_B: str,
    window_size: int,
    theta: float,
    deadband: float = 0.15,
    vol_window: int = 90,
    cash0: float = 1_000_000.0,
    com_pct: float = 0.125 / 100,
):
    """
    Backtest con DOS Filtros de Kalman + Modelo de Decisión

    - Modelo 1 (KalmanFilterRegression): hedge ratio dinámico beta_t y alpha_t.
    - Modelo 2 (KalmanSignalMean): media dinámica del spread.
    - Modelo 3 (implícito): reglas de trading basadas en z-score.
    """

    if len(dataframe_raw) <= window_size:
        raise ValueError(
            f"Dataframe demasiado corto (len={len(dataframe_raw)}) "
            f"para window_size={window_size}"
        )

    cash = float(cash0)
    COM = float(com_pct)

    portfolio_historic = [cash]

    position_active = False
    long_position: Operation | None = None
    short_position: Operation | None = None

    trades: list[Trade] = []
    current_side: str | None = None
    entry_date: pd.Timestamp | None = None

    # ---------- MODELO 1: Kalman para hedge ratio (niveles) ----------
    print("Pre-calentando Filtro 1 (Kalman HR) con OLS en niveles...")
    first_window_raw = dataframe_raw.iloc[:window_size]
    y_lvl = first_window_raw[stock_A]
    X_lvl = sm.add_constant(first_window_raw[stock_B])

    try:
        ols_model = sm.OLS(y_lvl, X_lvl).fit()
        alpha_0 = float(ols_model.params["const"])
        beta_0 = float(ols_model.params[stock_B])
        print(f"Estado inicial OLS: beta_0={beta_0:.6f}, alpha_0={alpha_0:.6f}")
    except Exception as e:
        print("OLS inicial falló, usando alpha_0 = 0, beta_0 = 0. Error:", e)
        alpha_0, beta_0 = 0.0, 0.0

    Q_kalman = np.diag([1e-7, 1e-7])
    R_kalman = np.array([[1e-1]])
    initial_state = np.array([[beta_0], [alpha_0]])   # [beta, alpha]
    initial_covariance = np.eye(2) * 100.0

    kf_hr = KalmanFilterRegression(
        R=R_kalman,
        Q=Q_kalman,
        initial_state=initial_state,
        initial_covariance=initial_covariance,
    )

    # ---------- MODELO 2: Kalman para media del spread ----------
    signal_filter = KalmanSignalMean(mu0=0.0, var0=1.0, q=1e-4, r=1e-2)

    z_score_history: list[float] = []
    hr_kalman_history: list[float] = []
    mu_signal_history: list[float] = []
    spread_history: list[float] = []

    print("Iniciando backtest con DOS Kalman Filters...")

    for i in range(window_size, len(dataframe_raw)):
        row = dataframe_raw.iloc[i]
        date_i = dataframe_raw.index[i]
        price_A = float(row[stock_A])
        price_B = float(row[stock_B])

        # --- 1. MODELO 1: actualizar hedge ratio dinámico ---
        H = np.array([[price_B, 1.0]])  # [B_t, 1]
        try:
            kf_hr.update(y=price_A, H=H)
            beta_t = float(kf_hr.get_hedge_ratio())
            alpha_t = float(kf_hr.get_intercept())
        except Exception as e:
            print(f"Error en actualización Kalman HR en i={i}: {e}")
            beta_t, alpha_t = np.nan, np.nan

        hr_kalman_history.append(beta_t)

        # Spread actual usando beta_t y alpha_t
        if np.isnan(beta_t) or np.isnan(alpha_t):
            spread_t = np.nan
        else:
            spread_t = price_A - (beta_t * price_B + alpha_t)

        spread_history.append(spread_t)

        # --- 2. MODELO 2: actualizar media del spread y obtener z-score ---
        z_signal = np.nan
        mu_t = np.nan

        if not np.isnan(spread_t):
            signal_filter.update(spread_t)
            mu_t = signal_filter.get_mean()

            if len(spread_history) >= vol_window:
                recent_spreads = np.array(spread_history[-vol_window:])
                recent_spreads = recent_spreads[~np.isnan(recent_spreads)]
                if recent_spreads.size > 5:
                    sigma_t = float(np.std(recent_spreads, ddof=1))
                    if sigma_t > 1e-8:
                        z_signal = (spread_t - mu_t) / sigma_t

        mu_signal_history.append(mu_t)
        z_score_history.append(z_signal)

        # ---------- 3. MODELO 3: Lógica de trading ----------
        if not np.isnan(z_signal) and not np.isnan(beta_t):

            # --- CIERRE por deadband ---
            if position_active and abs(z_signal) < deadband:
                if long_position is not None and short_position is not None:
                    price_close_long = float(row[long_position.stock])
                    price_close_short = float(row[short_position.stock])

                    open_comm = (long_position.price * long_position.shares +
                                 short_position.price * short_position.shares) * COM
                    close_comm = (price_close_long * long_position.shares +
                                  price_close_short * short_position.shares) * COM
                    total_comm = open_comm + close_comm

                    pnl_long = (price_close_long - long_position.price) * long_position.shares
                    pnl_short = (short_position.price - price_close_short) * short_position.shares
                    pnl_gross = pnl_long + pnl_short
                    pnl_net = pnl_gross - total_comm

                    notional_entry = (long_position.price * long_position.shares +
                                      short_position.price * short_position.shares)
                    ret_pct = pnl_net / notional_entry if notional_entry > 0 else 0.0
                    holding = (date_i - entry_date).days if entry_date is not None else 0

                    trades.append(
                        Trade(
                            entry_date=entry_date,
                            exit_date=date_i,
                            side=current_side,
                            pnl_gross=pnl_gross,
                            pnl_net=pnl_net,
                            return_pct=ret_pct,
                            holding_period=holding,
                            commissions=total_comm,
                        )
                    )

                    # Cerrar en cash (ya con comisiones)
                    cash += price_close_long * long_position.shares * (1 - COM)
                    cash -= price_close_short * short_position.shares * (1 + COM)

                position_active = False
                long_position = None
                short_position = None
                current_side = None
                entry_date = None

            # --- APERTURA de posición ---
            elif not position_active:
                risk_frac = 0.20
                total_notional = cash * risk_frac

                if np.isnan(beta_t) or abs(beta_t) < 1e-6:
                    pass

                # COMPRAR SPREAD (long A, short B)
                elif z_signal < -theta:
                    w_A = 1.0
                    w_B = abs(beta_t)
                    bundle_price = w_A * price_A + w_B * price_B
                    if bundle_price > 0:
                        n_bundles = int(np.floor(total_notional / bundle_price))
                    else:
                        n_bundles = 0

                    if n_bundles > 0:
                        n_long = int(np.floor(n_bundles * w_A))
                        n_short = int(np.floor(n_bundles * w_B))
                        if n_long > 0 and n_short > 0:
                            cash -= n_long * price_A * (1 + COM)
                            long_position = Operation(stock=stock_A, price=price_A, shares=n_long)

                            cash += n_short * price_B * (1 - COM)
                            short_position = Operation(stock=stock_B, price=price_B, shares=n_short)

                            position_active = True
                            current_side = "long_spread"
                            entry_date = date_i

                # VENDER SPREAD (short A, long B)
                elif z_signal > theta:
                    w_A = 1.0
                    w_B = abs(beta_t)
                    bundle_price = w_A * price_A + w_B * price_B
                    if bundle_price > 0:
                        n_bundles = int(np.floor(total_notional / bundle_price))
                    else:
                        n_bundles = 0

                    if n_bundles > 0:
                        n_short = int(np.floor(n_bundles * w_A))
                        n_long = int(np.floor(n_bundles * w_B))
                        if n_long > 0 and n_short > 0:
                            cash += n_short * price_A * (1 - COM)
                            short_position = Operation(stock=stock_A, price=price_A, shares=n_short)

                            cash -= n_long * price_B * (1 + COM)
                            long_position = Operation(stock=stock_B, price=price_B, shares=n_long)

                            position_active = True
                            current_side = "short_spread"
                            entry_date = date_i

        # ---------- 4. Valuación diaria del portafolio ----------
        portfolio_val = cash
        if position_active and long_position is not None and short_position is not None:
            portfolio_val += float(row[long_position.stock]) * long_position.shares
            short_current_liability = float(row[short_position.stock]) * short_position.shares
            portfolio_val -= short_current_liability

        portfolio_historic.append(portfolio_val)

    # ---------- 5. Cierre final (si queda posición abierta) ----------
    if position_active and long_position is not None and short_position is not None:
        last_row = dataframe_raw.iloc[-1]
        last_date = dataframe_raw.index[-1]
        price_long_close = float(last_row[long_position.stock])
        price_short_close = float(last_row[short_position.stock])

        open_comm = (long_position.price * long_position.shares +
                     short_position.price * short_position.shares) * COM
        close_comm = (price_long_close * long_position.shares +
                      price_short_close * short_position.shares) * COM
        total_comm = open_comm + close_comm

        pnl_long = (price_long_close - long_position.price) * long_position.shares
        pnl_short = (short_position.price - price_short_close) * short_position.shares
        pnl_gross = pnl_long + pnl_short
        pnl_net = pnl_gross - total_comm

        notional_entry = (long_position.price * long_position.shares +
                          short_position.price * short_position.shares)
        ret_pct = pnl_net / notional_entry if notional_entry > 0 else 0.0
        holding = (last_date - entry_date).days if entry_date is not None else 0

        trades.append(
            Trade(
                entry_date=entry_date,
                exit_date=last_date,
                side=current_side,
                pnl_gross=pnl_gross,
                pnl_net=pnl_net,
                return_pct=ret_pct,
                holding_period=holding,
                commissions=total_comm,
            )
        )

        cash += price_long_close * long_position.shares * (1 - COM)
        cash -= price_short_close * short_position.shares * (1 + COM)

        portfolio_historic.append(cash)

    print("Backtest con dos Kalman filters finalizado.")
    return portfolio_historic, z_score_history, hr_kalman_history, mu_signal_history, trades
