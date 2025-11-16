import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from data_pre import clean_data
from backtesting import backtesting
from metrics import performance_metrics
from trade_stats import compute_trade_stats


def align_portfolio_series(port_hist, df_segment, window_size):
    '''
    Aligns the portfolio equity curve with the dates of the segment.
    - Ignores the first value (initial cash).
    - Starts indexing from df_segment.index[window_size:].
    '''
    values = np.array(port_hist[1:])           # quitamos cash0
    dates = df_segment.index[window_size:]     # después del warm-up

    m = min(len(values), len(dates))
    values = values[:m]
    dates = dates[:m]

    return pd.Series(values, index=dates)


def print_metrics(title, series):
    print(f"\n===== MÉTRICAS {title} =====")
    m = performance_metrics(series)
    for k, v in m.items():
        if isinstance(v, (int, np.integer)):
            print(f"{k}: {v}")
        else:
            print(f"{k}: {v:.4f}")


def print_trade_stats(title, trades):
    print(f"\n===== ESTADÍSTICAS DE TRADES ({title}) =====")
    stats = compute_trade_stats(trades)
    for k, v in stats.items():
        if isinstance(v, (int, np.integer)):
            print(f"{k}: {v}")
        else:
            print(f"{k}: {v:.4f}")


def main():
    # --- 1. Configuración ---
    STOCKS_PAIR = ["JPM", "BAC"]
    STOCK_A, STOCK_B = STOCKS_PAIR

    WINDOW_SIZE = 252
    THETA = 1.0
    DEADBAND = 0.15  # mismo valor que en backtesting (por defecto)

    TRAIN_END = "2018-12-31"
    TEST_END = "2021-12-31"

    # --- 2. Carga y limpieza de datos ---
    data = pd.read_csv("./data/stocks_data.csv", index_col=0, parse_dates=True)
    data = clean_data(data=data)

    if STOCK_A not in data.columns or STOCK_B not in data.columns:
        print(f"Error: No se encontraron los tickers {STOCK_A} o {STOCK_B} en los datos.")
        return

    data = data.loc[:, STOCKS_PAIR].dropna()
    data_raw = data.copy()
    print(f"Datos cargados: {len(data_raw)} días.")

    # --- 3. Split por fechas: TRAIN / TEST / VALIDATION ---
    train_raw = data_raw.loc[:TRAIN_END]
    test_raw = data_raw.loc[TRAIN_END:TEST_END]
    val_raw = data_raw.loc[TEST_END:]

    print(f"Días en TRAIN: {len(train_raw)}, TEST: {len(test_raw)}, VALIDATION: {len(val_raw)}")

    for name, df_seg in [("TRAIN", train_raw), ("TEST", test_raw), ("VALIDATION", val_raw)]:
        if len(df_seg) <= WINDOW_SIZE + 30:
            print(f"Advertencia: segmento {name} tiene pocos datos para WINDOW_SIZE={WINDOW_SIZE}")

    # ==========================
    #   BACKTEST EN TRAIN
    # ==========================
    print("\n=== Ejecutando BACKTEST en TRAIN ===")
    port_train, z_train, hrk_train, mu_train, trades_train = backtesting(
        dataframe_raw=train_raw,
        stock_A=STOCK_A,
        stock_B=STOCK_B,
        window_size=WINDOW_SIZE,
        theta=THETA,
        deadband=DEADBAND,
    )
    train_series = align_portfolio_series(port_train, train_raw, WINDOW_SIZE)

    # ==========================
    #   BACKTEST EN TEST
    # ==========================
    print("\n=== Ejecutando BACKTEST en TEST ===")
    port_test, z_test, hrk_test, mu_test, trades_test = backtesting(
        dataframe_raw=test_raw,
        stock_A=STOCK_A,
        stock_B=STOCK_B,
        window_size=WINDOW_SIZE,
        theta=THETA,
        deadband=DEADBAND,
    )
    test_series = align_portfolio_series(port_test, test_raw, WINDOW_SIZE)

    # ==========================
    #   BACKTEST EN VALIDATION
    # ==========================
    print("\n=== Ejecutando BACKTEST en VALIDATION ===")
    port_val, z_val, hrk_val, mu_val, trades_val = backtesting(
        dataframe_raw=val_raw,
        stock_A=STOCK_A,
        stock_B=STOCK_B,
        window_size=WINDOW_SIZE,
        theta=THETA,
        deadband=DEADBAND,
    )
    val_series = align_portfolio_series(port_val, val_raw, WINDOW_SIZE)

    # ==========================
    #   MÉTRICAS
    # ==========================
    print_metrics("TRAIN", train_series)
    print_metrics("TEST", test_series)
    print_metrics("VALIDATION", val_series)

    print_trade_stats("TRAIN", trades_train)
    print_trade_stats("TEST", trades_test)
    print_trade_stats("VALIDATION", trades_val)

    # ==========================
    #   GRÁFICO: EQUITY CURVE 3 FASES
    # ==========================
    plt.figure(figsize=(14, 6))
    train_series.plot(label="TRAIN")
    test_series.plot(label="TEST")
    val_series.plot(label="VALIDATION")

    if len(test_series) > 0:
        plt.axvline(test_series.index[0], linestyle="--", label="Inicio TEST")
    if len(val_series) > 0:
        plt.axvline(val_series.index[0], linestyle="--", label="Inicio VALIDATION")

    plt.title(f"Valor del Portafolio (TRAIN / TEST / VALIDATION): {STOCK_A} vs {STOCK_B}")
    plt.ylabel("Valor ($)")
    plt.xlabel("Fecha")
    plt.legend()
    plt.show()

    # ==========================
    #   GRÁFICOS DE DIAGNÓSTICO (VALIDATION)
    # ==========================
    print("Generando gráficos de diagnóstico para VALIDATION...")
    m_diag = min(len(z_val), len(hrk_val), len(mu_val), len(val_series))
    z_val_adj = z_val[-m_diag:]
    hrk_val_adj = hrk_val[-m_diag:]
    mu_val_adj = mu_val[-m_diag:]
    diag_dates = val_series.index[-m_diag:]

    # Z-Score
    z_series_val = pd.Series(z_val_adj, index=diag_dates)
    plt.figure(figsize=(12, 6))
    z_series_val.plot(title="Historial del Z-Score (VALIDATION)")
    plt.axhline(THETA, linestyle="--", label=f"Theta ({THETA})")
    plt.axhline(-THETA, linestyle="--")
    plt.axhline(DEADBAND, linestyle=":", label=f"Cierre ({DEADBAND})")
    plt.axhline(-DEADBAND, linestyle=":")
    plt.axhline(0, linestyle="-", linewidth=0.5)
    plt.ylabel("Z-Score")
    plt.xlabel("Fecha")
    plt.legend()
    plt.show()

    # HR Kalman y media del spread (μ_t)
    hr_series_val = pd.Series(hrk_val_adj, index=diag_dates)
    mu_series_val = pd.Series(mu_val_adj, index=diag_dates)
    plt.figure(figsize=(12, 6))
    hr_series_val.plot(label="hr_Kalman (Modelo 1)", lw=2, alpha=0.8)
    mu_series_val.ffill().plot(label="μ_t (Media del spread, Modelo 2)", linestyle="--", lw=2, alpha=0.8)
    plt.title("Kalman HR vs Media del Spread (VALIDATION)")
    plt.ylabel("Valor")
    plt.xlabel("Fecha")
    plt.legend()
    plt.show()


if __name__ == "__main__":
    main()
