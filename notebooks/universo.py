"""Carga del universo de empresas y parámetros de mercado desde data/empresas_data_final.xlsx.

Las columnas se extraen por posición (0-based) porque AN y AT comparten el mismo
encabezado en la fila 2 (`RANKINGFinal`) y pandas los manglea.
"""

from pathlib import Path

import pandas as pd

DATA_PATH = Path(__file__).resolve().parent.parent / "data" / "empresas_data_final.xlsx"

# Extracción posicional (0-based) sobre la tab `data`:
# A=0 Ticker | B=1 Company | C=2 Sector | E=4 Country | F=5 Market Cap
# AT=45 RANKINGFinal (ranking consolidado) | AU=46 Peso en Cartera
_COLUMNAS = {
    "Ticker": 0,
    "Company": 1,
    "Sector": 2,
    "Country": 4,
    "MarketCap": 5,
    "Ranking": 45,
    "Peso": 46,
}


def parse_market_cap(valor):
    """Convierte '30.42B' (billones USD) / '850.00M' (millones USD) a float en USD."""
    if pd.isna(valor):
        return None
    texto = str(valor).strip().upper()
    if texto.endswith("B"):
        return float(texto[:-1]) * 1e9
    if texto.endswith("M"):
        return float(texto[:-1]) * 1e6
    return float(texto)


def cargar_universo(ruta: Path = DATA_PATH) -> pd.DataFrame:
    """Devuelve el universo ordenado por Ranking (col. AT), con MarketCapUSD en USD."""
    raw = pd.read_excel(ruta, sheet_name="data", header=None)
    universo = raw.iloc[2:, list(_COLUMNAS.values())].copy()
    universo.columns = list(_COLUMNAS.keys())
    universo = universo.dropna(subset=["Ticker"]).reset_index(drop=True)
    universo["MarketCapUSD"] = universo["MarketCap"].map(parse_market_cap)
    universo["Ranking"] = universo["Ranking"].astype(int)
    return universo.drop(columns=["MarketCap"]).sort_values("Ranking").reset_index(drop=True)


def cargar_parametros_mercado(ruta: Path = DATA_PATH) -> dict:
    """Extrae el ERM y la tasa libre de riesgo de la tab `data`.

    - ERM (col. V, 'ERM SPY'): CAGR esperado del SPY en % (10 = 10%). Se toma tal cual.
    - risk_free (col. Y, 'RiskFree'): en %, tasa libre de riesgo (intercepto de la SML).
    """
    raw = pd.read_excel(ruta, sheet_name="data", header=None)
    filas_datos = raw.index[raw.index >= 2]  # la fila 1 son los encabezados
    erm = float(raw.loc[filas_datos, 21].dropna().iloc[0]) / 100
    rf = float(raw.loc[filas_datos, 24].dropna().iloc[0]) / 100
    return {"erm": erm, "risk_free": rf}
