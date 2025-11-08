import yfinance as yf
import pandas as pd
from datetime import datetime
import os

def download_panel_adj_close(tickers, start="2010-01-01", end="2025-01-01",
                             filepath=None):
    if end is None:
        end = datetime.today().strftime("%Y-%m-%d")

    if filepath is None:
        filepath = "data/stocks_data.csv"

    os.makedirs(os.path.dirname(filepath), exist_ok=True)

    # Descarga todos los activos en un único request
    data = yf.download(tickers, start=start, end=end)["Close"]

    # Forzar columnas como tickers (a veces sale MultiIndex)
    data.columns = tickers

    data.to_csv(filepath)

    return data

tickers = tickers = [
    # --- Tech ---
    "AAPL","MSFT","GOOG","GOOGL","AMZN","META","NVDA","TSLA","IBM","ORCL",
    "ADBE","INTC","AMD","QCOM","CSCO","CRM","SHOP","PANW","NET",
    "SNOW","PLTR","UBER","LYFT","ZM","BIDU","JD","NTES","TSM","ASML",
    "MU","TXN","AVGO","AMAT","LRCX","KLAC","NOW","INTU","DOCU",

    # --- Finance ---
    "JPM","BAC","WFC","GS","C","MS","BLK","AXP","COF","SCHW",
    "USB","PNC","TFC","BK","AIG","CB","SPGI","ICE","MA","V",
    "PYPL","BRK-B","BRK-A",

    # --- Energy ---
    "XOM","CVX","COP","EOG","SLB","HAL","FANG","MPC","PSX",
    "VLO","BP","SHEL","ENB","KMI",

    # --- Healthcare ---
    "JNJ","PFE","MRK","ABBV","TMO","ABT","BMY","LLY","AMGN","GILD",
    "ISRG","DHR","VRTX","REGN","ZTS","BAX","BDX","SYK","MOH","CI",

    # --- Consumer ---
    "PG","KO","PEP","MCD","SBUX","NKE","COST","HD","LOW","TGT",
    "WMT","DIS","CMG","YUM","MDLZ","KHC","K","CL","EL","PM",

    # --- Industrials ---
    "CAT","DE","LMT","BA","GE","HON","UPS","FDX","MMM","RTX",
    "NOC","GD","ETN","EMR","CSX","NSC","UNP","DAL","UAL","AAL",

    # --- Utilities ---
    "NEE","DUK","SO","AEP","SRE","D","XEL","ETR","PEG","WEC",

    # --- Real Estate ---
    "AMT","PLD","EQIX","SPG","O","DLR","WELL","VTR","SBAC","AVB",

    # --- Materials ---
    "LIN","APD","ECL","SHW","NEM","FCX","MLM","VMC","DD","ALB",

    # --- Telecommunications ---
    "T","VZ","TMUS","AMX"
]

def clean_data(data):

    df_clean = data.dropna(axis=1, how="any")

    return df_clean

