"""Black-Litterman con views relativas en cadena sobre el ranking de la col. AT.

A diferencia de `skfolio.prior.BlackLitterman` (que deriva Ω de la covarianza vía
Idzorek), acá la incertidumbre de cada view se especifica directamente como
Ω = diag((κ·q)²): el error estándar de cada view es κ veces su propio paso q.
Esto hace que el ordenamiento del ranking se imponga con la holgura explícita κ,
sin que el prior de equilibrio (ordenado por market cap) lo contradicta.

Fórmulas: He-Litterman (1999), "The Intuition Behind Black-Litterman Model Portfolios".
"""

import numpy as np
from skfolio.prior import BasePrior, ReturnDistribution


def matriz_picking(n_activos: int) -> np.ndarray:
    """Matriz P ((n-1)×n) de la cadena: fila k = +1 en el rank k, -1 en el rank k+1."""
    P = np.zeros((n_activos - 1, n_activos))
    idx = np.arange(n_activos - 1)
    P[idx, idx] = 1.0
    P[idx, idx + 1] = -1.0
    return P


def posterior_black_litterman(
    sigma: np.ndarray,
    pi: np.ndarray,
    q: np.ndarray,
    tau: float = 0.05,
    kappa: float = 0.5,
) -> tuple[np.ndarray, np.ndarray]:
    """Posterior Black-Litterman con views relativas y Ω = diag((κ·q)²).

    Parameters
    ----------
    sigma : Σ (n×n), covarianza en la frecuencia de los datos.
    pi    : prior de retornos esperados (n,), p.ej. SML CAPM: Rf + β_i(ERM − Rf).
    q     : views (k,), outperformance esperada de cada par (misma frecuencia).
    tau   : peso del prior.
    kappa : holgura de las views (error estándar = κ·q).

    Returns
    -------
    (mu_bl, sigma_bl) : posterior de retornos esperados y covarianza (Σ + M).
    """
    pi = np.asarray(pi, dtype=float)
    q = np.asarray(q, dtype=float).reshape(-1)
    P = matriz_picking(len(pi))

    tau_sigma = tau * sigma
    pts_p = P @ tau_sigma @ P.T
    omega = np.diag((kappa * q) ** 2)

    # μ_BL = π + τΣP'(PτΣP' + Ω)⁻¹(Q − Pπ)     [tilt hacia las views]
    inverso = np.linalg.inv(pts_p + omega)
    mu_bl = pi + tau_sigma @ P.T @ inverso @ (q - P @ pi)

    # Σ_BL = Σ + M,  M = τΣ − τΣP'(PτΣP' + Ω)⁻¹PτΣ
    m = tau_sigma - tau_sigma @ P.T @ inverso @ P @ tau_sigma
    return mu_bl, sigma + m


class PriorBlackLitterman(BasePrior):
    """Prior de skfolio con mu/covarianza ya calculados por `posterior_black_litterman`."""

    def __init__(self, mu: np.ndarray, covariance: np.ndarray):
        self.mu = mu
        self.covariance = covariance

    def fit(self, X, y=None, **fit_params):
        self.return_distribution_ = ReturnDistribution(
            mu=np.asarray(self.mu, dtype=float),
            covariance=np.asarray(self.covariance, dtype=float),
            returns=X,
        )
        return self
