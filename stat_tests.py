from statsmodels.tsa.stattools import adfuller
import pandas as pd
from statsmodels.tsa.vector_ar.vecm import coint_johansen
import numpy as np
from itertools import combinations


def prueba_adf(df, alpha=0.05):
    """
    Filtra columnas cuyo proceso NO es estacionario según ADF.
    (Esta función no cambia)
    """
    resultados = {}
    no_estacionarios = []
    estacionarios = []

    for col in df.columns:
        serie = df[col]
        adf_stat, p_value, _, _, _, _ = adfuller(serie)
        resultados[col] = p_value
        if p_value > alpha:
            no_estacionarios.append(col)
        else:
            estacionarios.append(col)
    df_filtrado = df[no_estacionarios]
    return df_filtrado, resultados, no_estacionarios, estacionarios


def johansen_cointegration_test(series1, series2, det_order=0, k_ar_diff=1, alpha=0.05):
    """
    Prueba de cointegración Johansen para dos series.
    (¡Actualizado con Filtro de Sanidad!)
    """
    df = pd.concat([series1, series2], axis=1).dropna()
    df.columns = ["A", "B"]

    johansen_result = coint_johansen(df, det_order, k_ar_diff)

    trace_stat = johansen_result.lr1[0]
    crit_value = johansen_result.cvt[0, {0.10: 0, 0.05: 1, 0.01: 2}[alpha]]
    cointegrated = trace_stat > crit_value

    raw_eigenvector = None
    if cointegrated:
        raw_eigenvector = johansen_result.evec[:, 0]

        # --- ¡NUEVO! Filtro de Sanidad Numérica ---
        # A veces, Johansen pasa pero da un vector numéricamente inestable
        # (como el pico de +300 que vimos). Esto lo filtra.

        v1 = raw_eigenvector[0]
        v2 = raw_eigenvector[1]

        # Chequeo 1: ¿V1 es casi cero? (División por cero)
        if abs(v1) < 1e-10:
            cointegrated = False  # Vector inestable
            raw_eigenvector = None
        else:
            # Chequeo 2: ¿El ratio implícito es absurdo?
            hr_vecm_implicit = -v2 / v1
            if abs(hr_vecm_implicit) > 20.0:  # Un 'hr' > 20 es irreal
                cointegrated = False  # Vector inestable
                raw_eigenvector = None
        # --- Fin del Filtro ---

    return cointegrated, trace_stat, crit_value, raw_eigenvector


def encontrar_pares_cointegrados(df, alpha=0.05):
    """
    (Esta función no cambia)
    """
    pares_cointegrados = []
    estadisticas = []
    for a, b in combinations(df.columns, 2):
        serie_a = df[a]
        serie_b = df[b]
        try:
            # Ahora usa la versión "saneada" de la prueba
            cointegrated, stat, crit, _ = johansen_cointegration_test(serie_a, serie_b, alpha=alpha)
            if cointegrated:
                pares_cointegrados.append((a, b))
                estadisticas.append({"A": a, "B": b, "trace_stat": stat, "crit": crit})
        except Exception as e:
            print(f"Error con {a}-{b}: {e}")
            continue
    return pares_cointegrados, pd.DataFrame(estadisticas)