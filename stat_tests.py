from statsmodels.tsa.stattools import adfuller
import pandas as pd
from statsmodels.tsa.vector_ar.vecm import coint_johansen
import numpy as np
from itertools import combinations


def prueba_adf(df, alpha=0.05):
    """
    Filtra columnas cuyo proceso NO es estacionario según ADF.
    
    Parámetros:
        df : DataFrame con activos (columnas)
        alpha : nivel de significancia (default 0.05)
        max_nan_pct : porcentaje máximo de NaNs permitido por activo
        
    Retorna:
        df_filtrado: DataFrame solo con activos no estacionarios
        resultados: diccionario con p-values por ticker
        no_estacionarios: lista de tickers I(1)
        estacionarios: lista de tickers I(0)
    """

    resultados = {}
    no_estacionarios = []
    estacionarios = []

    for col in df.columns:
        serie = df[col]

        # ADF
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
    
    Retorna:
        cointegrated (bool)
        trace_stat (float)
        crit_value (float)
    """
    
    # Unir como DataFrame de (N x 2)
    df = pd.concat([series1, series2], axis=1).dropna()
    df.columns = ["A", "B"]

    # Johansen test
    johansen_result = coint_johansen(df, det_order, k_ar_diff)

    # Tomamos la estadística de traza para r=0
    trace_stat = johansen_result.lr1[0]       # statistic for r = 0
    crit_value = johansen_result.cvt[0, {0.10:0, 0.05:1, 0.01:2}[alpha]]

    cointegrated = trace_stat > crit_value

    return cointegrated, trace_stat, crit_value

def encontrar_pares_cointegrados(df, alpha=0.05):
    pares_cointegrados = []
    estadisticas = []

    for a, b in combinations(df.columns, 2):
        serie_a = df[a]
        serie_b = df[b]

        try:
            cointegrated, stat, crit = johansen_cointegration_test(serie_a, serie_b, alpha=alpha)

            if cointegrated:
                pares_cointegrados.append((a, b))
                estadisticas.append({"A": a, "B": b, "trace_stat": stat, "crit": crit})

        except Exception as e:
            print(f"Error con {a}-{b}: {e}")
            continue

    return pares_cointegrados, pd.DataFrame(estadisticas)