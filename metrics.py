import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression


def annualized_return(series):
    """Retorno anualizado basado en el primer y último valor."""
    total_return = series.iloc[-1] / series.iloc[0] - 1
    years = (series.index[-1] - series.index[0]).days / 365
    return (1 + total_return)**(1/years) - 1


def annualized_volatility(series):
    """Volatilidad anualizada basada en los retornos diarios."""
    daily_returns = series.pct_change().dropna()
    return daily_returns.std() * np.sqrt(252)


def sharpe_ratio(series, rf=0.0):
    """Sharpe ratio anualizado."""
    ret = annualized_return(series)
    vol = annualized_volatility(series)
    if vol == 0:
        return np.nan
    return (ret - rf) / vol


def sortino_ratio(series, rf=0.0):
    """Sortino ratio anualizado (solo downside)."""
    daily_returns = series.pct_change().dropna()
    downside = daily_returns[daily_returns < 0]
    downside_vol = downside.std() * np.sqrt(252)
    ann_ret = annualized_return(series)
    if downside_vol == 0:
        return np.nan
    return (ann_ret - rf) / downside_vol


def max_drawdown(series):
    """Máximo drawdown absoluto y su duración."""
    cumulative_max = series.cummax()
    drawdown = (series - cumulative_max) / cumulative_max
    max_dd = drawdown.min()

    # duración (peak-to-trough recovery time)
    dd_duration = 0
    temp = 0
    for x in drawdown:
        if x < 0:
            temp += 1
            dd_duration = max(dd_duration, temp)
        else:
            temp = 0

    return max_dd, dd_duration


def calmar_ratio(series):
    """Calmar = retorno anualizado / max drawdown (en valor negativo)."""
    ann_ret = annualized_return(series)
    max_dd, _ = max_drawdown(series)
    if max_dd == 0:
        return np.nan
    return ann_ret / abs(max_dd)


def equity_curve_stability(series):
    """
    Mide la R^2 del equity curve respecto a una línea ascendente.
    Entre 0 y 1. Más cerca de 1 = más estable.
    """
    y = np.log(series.values).reshape(-1, 1)
    X = np.arange(len(series)).reshape(-1, 1)

    model = LinearRegression().fit(X, y)
    r2 = model.score(X, y)
    return r2


def performance_metrics(series):
    """Regresa TODAS las métricas en un dict."""
    dd, dd_dur = max_drawdown(series)

    return {
        "Annualized Return": annualized_return(series),
        "Annualized Volatility": annualized_volatility(series),
        "Sharpe Ratio": sharpe_ratio(series),
        "Sortino Ratio": sortino_ratio(series),
        "Max Drawdown": dd,
        "Max DD Duration (days)": dd_dur,
        "Calmar Ratio": calmar_ratio(series),
        "Stability (R²)": equity_curve_stability(series)
    }
