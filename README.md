# Administración de Carteras

Optimización de carteras con el modelo **Black-Litterman** sobre el universo de 49 empresas
del archivo `data/empresas_data_final.xlsx`, utilizando **[skfolio](https://skfolio.org)** como
motor de estimación (prior, views) y optimización.

## Contexto

- El archivo `data/empresas_data_final.xlsx` (tab `data`) contiene el universo de empresas con
  rankings derivados de los modelos Grinold-Kroner, Ben Graham, Morningstar, RRF, Borda y Copeland.
- La columna **AT** (`Final` / `RANKINGFinal`) define el ranking consolidado (escala 1–49).
- El ranking se traduce a **views relativas** de Black-Litterman
  (`"activo_i - activo_j == expected_outperformance"`).
- La tab `cliente` define la asignación estratégica (Equity 30%, Bond 50%, Cash 10%, REITs 5%,
  Gold 5%) con bandas, además del portafolio de bonos y liabilities.

## Decisiones técnicas (definidas en sesión)

| Decisión | Elección | Motivo |
|---|---|---|
| Librería de optimización | **skfolio 1.0.1** | Views relativas nativas tipo string, confianzas de view (Idzorek), API scikit-learn, dependencias livianas, proyecto activo (v1.0 en 2026) |
| Alternativas evaluadas | Riskfolio-Lib 7.3.0, PyPortfolioOpt 1.6.0, cvxpy a mano | Riskfolio: completo pero dependencias pesadas (vectorbt/numba/astropy). PyPortfolioOpt: BL clásico sólido pero optimizer básico. Verificado empíricamente que las tres funcionan con este stack |
| Gestión de entorno | **uv** + `pyproject.toml` + `uv.lock` | Entorno aislado y reproducible con `uv sync` |
| Python | 3.13 | Punto óptimo de compatibilidad del ecosistema científico |

### Stack (versiones resueltas, ago 2026)

numpy 2.5.2 · pandas 3.0.5 · scipy 1.18.1 · scikit-learn 1.9.0 · skfolio 1.0.1 ·
cvxpy 1.9.2 · yfinance 1.7.0 · openpyxl 3.1.5 · pyarrow 25.0.1 · matplotlib 3.11.1 ·
plotly 7.0.0 · jupyterlab 4.6.3

## Estructura

```
administracion_carteras/
├── data/
│   ├── empresas_data_final.xlsx     # Datos de entrada (Excel, original intacto)
│   └── empresas_data_final_pesos.xlsx  # Copia con col. AU escrita (salida del nb 03)
├── notebooks/
│   ├── universo.py                  # Loader del Excel (extracción posicional, cols. AT/AU/V/Y)
│   ├── bl.py                        # Posterior Black-Litterman (Ω directa) + prior skfolio
│   ├── pipeline.py                  # Pipeline completo datos→posterior (función reutilizable)
│   ├── 01_carga_universo.ipynb      # Universo, ranking y market caps
│   ├── 02_precios_retornos_bl.ipynb # Precios (corte 13-ago-2026), retornos y posterior BL
│   └── 03_optimizacion.ipynb        # Max Sharpe / Min Varianza + escritura col. AU
├── outputs/                         # Artefactos generados (CSV de pesos; gitignored)
├── pyproject.toml                   # Dependencias del proyecto
├── uv.lock                          # Lockfile (reproducibilidad)
└── README.md
```

## Decisiones de modelado (definidas en sesión)

- **Views relativas**: cadena de pares adyacentes del ranking AT — "cada empresa superará a
  la siguiente" — con paso uniforme `q = S/(n−1)` y spread total `S` calibrable (base 8% anual).
- **Ω directa** (`Ω = diag((κ·q)²)`) en lugar de Idzorek: con Idzorek la incertidumbre queda
  atada a la covarianza de mercado y el prior cap-ponderado dominaba al ranking. Con Ω directa
  el ordenamiento se impone con holgura explícita `κ` (base 0.5).
- **Prior CAPM**: π_i = Rf + β_i(ERM − Rf), con β_i = Cov(r_i, SPY)/Var(SPY) (misma ventana).
  El ERM (col. V, 10%) es el CAGR esperado del SPY — el mercado es el SPY, no nuestro universo;
  el universo cap-ponderado (β_u ≈ 2) tiene un equilibrio mayor al ERM por su riesgo.
- **Nivel re-anclado**: las views fijan diferencias; el nivel absoluto del posterior se ancla
  al equilibrio CAPM del universo (w'π), no al ERM.
- **Covarianza**: Ledoit-Wolf sobre retornos diarios de la ventana común (mar-2024 → ago-2026,
  atada al IPO de RDDT).
- **Historia mínima**: 500 días. Excluye de la estimación a SKHY (25 días) y AUGO (272)
  (exclusión definitiva).

## Uso

```bash
# Crear entorno aislado e instalar dependencias exactas del lockfile
uv sync

# Abrir JupyterLab con el kernel del proyecto
uv run jupyter lab
```

## Próximos pasos

1. Backtest out-of-sample (walk-forward re-optimizando por período).
2. Definir escala definitiva de la col. AU: pesos del sleeve (actual, suman 100%) vs
   contribución al portafolio total (× 30% Equity de la SAA).
3. Constraints por sector/industria si el cliente define bandas a ese nivel.
