"""Calcula las métricas del portafolio general y las guarda en un CSV largo.

El riesgo del sleeve de equity usa su covarianza posterior Black-Litterman. Las
volatilidades de SCHI/VNQ y las correlaciones entre los tres activos riesgosos
usan retornos diarios de precios ajustados, alineados exactamente con
``modelo.X``. La covarianza anual se construye como ``D @ Corr @ D``. Money
Market se trata como tasa libre de riesgo, con volatilidad y covarianzas cero.
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import yfinance as yf

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "notebooks"))

from pipeline import construir_modelo

ANNUAL = 252
RISK_FREE = 0.02
PESOS_GENERALES = pd.Series(
    {"Equity_BL": 0.35, "SCHI": 0.50, "Money_Market": 0.10, "VNQ": 0.05}
)
RETORNOS_ESPERADOS_EXOGENOS = {
    "SCHI": 0.0451,
    "Money_Market": RISK_FREE,
    "VNQ": 0.0775,
}
ACTIVOS_RIESGOSOS = ["Equity_BL", "SCHI", "VNQ"]


def _fila(seccion: str, metrica: str, valor, activo_1="", activo_2="") -> dict:
    return {
        "seccion": seccion,
        "metrica": metrica,
        "activo_1": activo_1,
        "activo_2": activo_2,
        "valor": valor,
    }


def calcular() -> tuple[pd.DataFrame, dict]:
    modelo = construir_modelo(annual=ANNUAL)
    pesos_equity = (
        pd.read_csv(ROOT / "outputs" / "pesos_sleeve_equity.csv")
        .set_index("Ticker")["MaxSharpeSinCap"]
        .reindex(modelo.orden)
    )
    if pesos_equity.isna().any():
        raise ValueError("Faltan pesos MaxSharpeSinCap para activos de modelo.orden")
    if not np.isclose(pesos_equity.sum(), 1.0, atol=1e-10):
        raise ValueError(f"Los pesos del sleeve equity suman {pesos_equity.sum():.12f}")
    if not np.isclose(PESOS_GENERALES.sum(), 1.0, atol=1e-12):
        raise ValueError(f"Los pesos generales suman {PESOS_GENERALES.sum():.12f}")

    w_equity = pesos_equity.to_numpy()
    retorno_equity = float(pesos_equity @ modelo.mu_post)
    vol_equity = float(np.sqrt(w_equity @ modelo.sigma_bl @ w_equity) * np.sqrt(ANNUAL))

    descarga = yf.download(
        ["SCHI", "VNQ"],
        start=modelo.config["start"],
        end=modelo.config["end"],
        auto_adjust=True,
        progress=False,
        threads=True,
    )
    if descarga.empty or "Close" not in descarga.columns.get_level_values(0):
        raise RuntimeError("Yahoo Finance no devolvió precios ajustados de SCHI/VNQ")
    cierres = descarga["Close"].reindex(columns=["SCHI", "VNQ"])
    retornos_etf = cierres.pct_change(fill_method=None).reindex(modelo.X.index)
    if retornos_etf.isna().any().any():
        faltantes = retornos_etf.isna().sum().to_dict()
        raise RuntimeError(f"SCHI/VNQ no cubren todas las fechas efectivas de modelo.X: {faltantes}")

    retorno_equity_historico = (modelo.X @ pesos_equity).rename("Equity_BL")
    retornos_riesgosos = pd.concat([retorno_equity_historico, retornos_etf], axis=1)
    correlacion = retornos_riesgosos.corr().loc[ACTIVOS_RIESGOSOS, ACTIVOS_RIESGOSOS]
    volatilidades = pd.Series(
        {
            "Equity_BL": vol_equity,
            "SCHI": float(retornos_etf["SCHI"].std(ddof=1) * np.sqrt(ANNUAL)),
            "VNQ": float(retornos_etf["VNQ"].std(ddof=1) * np.sqrt(ANNUAL)),
        }
    )
    d = np.diag(volatilidades.reindex(ACTIVOS_RIESGOSOS))
    cov_riesgosos = pd.DataFrame(
        d @ correlacion.to_numpy() @ d,
        index=ACTIVOS_RIESGOSOS,
        columns=ACTIVOS_RIESGOSOS,
    )
    covarianza = pd.DataFrame(0.0, index=PESOS_GENERALES.index, columns=PESOS_GENERALES.index)
    covarianza.loc[ACTIVOS_RIESGOSOS, ACTIVOS_RIESGOSOS] = cov_riesgosos

    retornos_esperados = pd.Series(
        {"Equity_BL": retorno_equity, **RETORNOS_ESPERADOS_EXOGENOS}
    ).reindex(PESOS_GENERALES.index)
    retorno_portafolio = float(PESOS_GENERALES @ retornos_esperados)
    varianza_portafolio = float(PESOS_GENERALES @ covarianza @ PESOS_GENERALES)
    volatilidad_portafolio = float(np.sqrt(varianza_portafolio))
    sharpe = (retorno_portafolio - RISK_FREE) / volatilidad_portafolio

    if not np.allclose(correlacion, correlacion.T, atol=1e-12):
        raise AssertionError("La matriz de correlaciones no es simétrica")
    if not np.allclose(np.diag(correlacion), 1.0, atol=1e-12):
        raise AssertionError("La diagonal de la matriz de correlaciones no es uno")
    if not np.allclose(covarianza, covarianza.T, atol=1e-12):
        raise AssertionError("La matriz de covarianzas no es simétrica")
    if np.linalg.eigvalsh(covarianza.to_numpy()).min() < -1e-12:
        raise AssertionError("La matriz de covarianzas no es semidefinida positiva")

    filas = [
        _fila("metadata", "fecha_inicial", retornos_riesgosos.index.min().date().isoformat()),
        _fila("metadata", "fecha_final", retornos_riesgosos.index.max().date().isoformat()),
        _fila("metadata", "observaciones_diarias", len(retornos_riesgosos)),
        _fila("metadata", "periodos_anuales", ANNUAL),
        _fila("portfolio", "retorno_esperado_anual", retorno_portafolio),
        _fila("portfolio", "varianza_anual", varianza_portafolio),
        _fila("portfolio", "volatilidad_anual", volatilidad_portafolio),
        _fila("portfolio", "sharpe_rf_2pct", sharpe),
    ]
    for activo in PESOS_GENERALES.index:
        filas.append(_fila("peso", "peso_portafolio", PESOS_GENERALES[activo], activo))
        filas.append(
            _fila("retorno_esperado", "retorno_anual", retornos_esperados[activo], activo)
        )
    for activo in ACTIVOS_RIESGOSOS:
        fuente = "posterior_bl" if activo == "Equity_BL" else "historica"
        filas.append(_fila("volatilidad", f"volatilidad_anual_{fuente}", volatilidades[activo], activo))
    for activo_1 in ACTIVOS_RIESGOSOS:
        for activo_2 in ACTIVOS_RIESGOSOS:
            filas.append(
                _fila(
                    "correlacion_historica",
                    "correlacion",
                    correlacion.loc[activo_1, activo_2],
                    activo_1,
                    activo_2,
                )
            )
    for activo_1 in PESOS_GENERALES.index:
        for activo_2 in PESOS_GENERALES.index:
            filas.append(
                _fila(
                    "covarianza_anual",
                    "covarianza",
                    covarianza.loc[activo_1, activo_2],
                    activo_1,
                    activo_2,
                )
            )

    detalles = {
        "retorno": retorno_portafolio,
        "volatilidad": volatilidad_portafolio,
        "sharpe": sharpe,
        "inicio": retornos_riesgosos.index.min().date().isoformat(),
        "fin": retornos_riesgosos.index.max().date().isoformat(),
        "n": len(retornos_riesgosos),
        "correlacion": correlacion,
        "covarianza": covarianza,
    }
    return pd.DataFrame(filas), detalles


if __name__ == "__main__":
    resultado, detalles = calcular()
    salida = ROOT / "outputs" / "metricas_portafolio_general.csv"
    resultado.to_csv(salida, index=False, float_format="%.12f")
    print(
        f"Ventana: {detalles['inicio']} a {detalles['fin']} ({detalles['n']} observaciones)\n"
        f"Retorno esperado: {detalles['retorno']:.6%}\n"
        f"Volatilidad esperada: {detalles['volatilidad']:.6%}\n"
        f"Sharpe (Rf=2%): {detalles['sharpe']:.6f}\n\n"
        f"Correlación histórica:\n{detalles['correlacion'].to_string(float_format=lambda x: f'{x:.6f}')}\n\n"
        f"Covarianza anual (cash con riesgo cero):\n"
        f"{detalles['covarianza'].to_string(float_format=lambda x: f'{x:.8f}')}\n\n"
        f"Guardado en: {salida.relative_to(ROOT)}"
    )
