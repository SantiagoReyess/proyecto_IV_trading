from statsmodels.tsa.stattools import adfuller
import pandas as pd

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
