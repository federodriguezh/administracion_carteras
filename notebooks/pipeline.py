"""Pipeline completo del modelo: datos → prior CAPM → posterior Black-Litterman.

Reproduce los pasos del notebook 02 como función reutilizable, para que la
optimización (notebook 03) y futuros scripts usen exactamente el mismo modelo.

Cadena de cálculo:
1. Universo del Excel (col. AT) + filtro de historia mínima → universo efectivo
2. Precios ajustados de yfinance (corte `end`, inclusive) → ventana común → retornos diarios
3. Covarianza Ledoit-Wolf conjunta (universo + SPY) → Σ, betas vs SPY
4. Prior CAPM: π_i = Rf + β_i·(ERM − Rf), con ERM = CAGR esperado del SPY (col. V)
5. Views relativas en cadena del ranking (P +1/−1, paso q = S/(n−1)) con Ω = diag((κ·q)²)
6. Posterior BL (He-Litterman) + re-anclaje del nivel a w'π (equilibrio CAPM del universo)
"""

from dataclasses import dataclass

import numpy as np
import pandas as pd
import yfinance as yf
from skfolio.moments import LedoitWolf
from skfolio.preprocessing import prices_to_returns

from bl import posterior_black_litterman
from universo import cargar_parametros_mercado, cargar_universo


@dataclass
class ModeloBL:
    """Salidas del pipeline (todo en el orden `orden`, salvo lo indicado)."""

    orden: list[str]              # tickers efectivos, ordenados por ranking AT
    X: pd.DataFrame               # retornos diarios simples
    sigma: np.ndarray             # covarianza Ledoit-Wolf del universo (diaria)
    sigma_bl: np.ndarray          # covarianza posterior BL (diaria)
    betas: pd.Series              # β_i = Cov(r_i, SPY)/Var(SPY)
    beta_u: float                 # β del universo cap-ponderado
    w_caps: np.ndarray            # pesos de equilibrio (caps) en el orden `orden`
    pi_diario: np.ndarray         # prior CAPM (diario)
    mu_prior: pd.Series           # prior anualizado
    mu_post: pd.Series            # posterior BL re-anclado (anualizado)
    mu_post_diario: pd.Series     # posterior BL re-anclado (diario, para skfolio)
    params: dict                  # {"erm", "risk_free"}
    config: dict                  # configuración usada
    excluidos: list[str]          # tickers fuera por historia mínima


def construir_modelo(
    start: str = "2021-08-13",
    end: str = "2026-08-14",
    min_dias: int = 500,
    spread: float = 0.08,
    kappa: float = 0.5,
    tau: float = 0.05,
    annual: int = 252,
) -> ModeloBL:
    """Ejecuta el pipeline completo y devuelve el modelo listo para optimizar."""
    universo = cargar_universo()
    orden = universo["Ticker"].tolist()
    params = cargar_parametros_mercado()

    # 1) Precios ajustados + filtro de historia mínima + ventana común
    data = yf.download(orden + ["SPY"], start=start, end=end,
                       auto_adjust=True, progress=False, threads=True)
    closes = data["Close"][orden].copy()
    dias = closes.notna().sum()
    excluidos = dias[dias < min_dias].index.tolist()
    orden_f = [t for t in orden if t not in excluidos]
    closes = closes[orden_f].dropna()

    # 2) Retornos diarios
    X = prices_to_returns(closes)
    orden_f = list(X.columns)
    n = len(orden_f)

    caps = universo.set_index("Ticker")["MarketCapUSD"].reindex(orden_f)
    assert caps.notna().all(), "Hay market caps faltantes en el universo efectivo"
    w_caps = (caps / caps.sum()).to_numpy()

    # 3) Covarianza LW conjunta (universo + SPY) y betas
    spy_ret = data["Close"]["SPY"].reindex(closes.index).pct_change().iloc[1:]
    X_full = pd.concat([X, spy_ret.rename("SPY")], axis=1).dropna()
    assert list(X_full.columns) == orden_f + ["SPY"]
    sigma_full = LedoitWolf().fit(X_full).covariance_
    sigma = sigma_full[:n, :n]
    betas = pd.Series(sigma_full[:n, n] / sigma_full[n, n], index=orden_f)
    beta_u = float(betas @ w_caps)

    # 4) Prior CAPM (diario)
    rf_diario = params["risk_free"] / annual
    prem_diario = (params["erm"] - params["risk_free"]) / annual
    pi_diario = rf_diario + betas.to_numpy() * prem_diario

    # 5) Views en cadena + posterior BL
    q_diario = (spread / (n - 1)) / annual
    Q = np.full(n - 1, q_diario)
    mu_bl, sigma_bl = posterior_black_litterman(sigma, pi_diario, Q, tau=tau, kappa=kappa)

    # 6) Re-anclaje del nivel al equilibrio CAPM del universo (w'π)
    mu_prior = pd.Series(pi_diario * annual, index=orden_f)
    mu_bl_annual = pd.Series(mu_bl * annual, index=orden_f)
    shift = float(mu_prior @ w_caps) - float(mu_bl_annual @ w_caps)
    mu_post = mu_bl_annual + shift
    mu_post_diario = pd.Series(mu_bl + shift / annual, index=orden_f)

    return ModeloBL(
        orden=orden_f,
        X=X,
        sigma=sigma,
        sigma_bl=sigma_bl,
        betas=betas,
        beta_u=beta_u,
        w_caps=w_caps,
        pi_diario=pi_diario,
        mu_prior=mu_prior,
        mu_post=mu_post,
        mu_post_diario=mu_post_diario,
        params=params,
        config={"start": start, "end": end, "min_dias": min_dias, "spread": spread,
                "kappa": kappa, "tau": tau, "annual": annual},
        excluidos=excluidos,
    )
