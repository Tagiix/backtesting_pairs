"""
Sélection des paires et modélisation du spread
==================================================================================
Implémente :
  - Test de cointégration d'Engle-Granger (ADF sur le résidu OLS)
  - Test de Johansen (option premium pour β multivariés)
  - Estimation de la half-life du processus d'Ornstein-Uhlenbeck
  - Sélection des paires significatives (p < 0.05)
  - Calcul du spread standardisé Z_t
"""

import numpy as np
import pandas as pd
from itertools import combinations
from typing import Dict, List, Optional, Tuple

from statsmodels.tsa.stattools import adfuller, coint
from statsmodels.regression.linear_model import OLS
from statsmodels.tools import add_constant
from statsmodels.tsa.vector_ar.vecm import coint_johansen


# ─────────────────────────────────────────────────────────────────────────────
# Test d'Engle-Granger
# ─────────────────────────────────────────────────────────────────────────────


def engle_granger_test(
    y: pd.Series,
    x: pd.Series,
) -> Dict[str, float]:
    """
    Teste la cointégration entre deux séries Y et X via Engle-Granger :
      1. Régression OLS :  Y_t = α + β·X_t + ε_t
      2. Test ADF sur les résidus ε̂_t

    H0 : les résidus ont une racine unitaire (pas de cointégration).

    Paramètres
    ----------
    y, x : Series de même longueur et de même index.

    Retourne un dict avec :
      - 'pvalue'   : p-value du test de cointégration (via statsmodels.coint)
      - 'beta'     : coefficient de la régression OLS
      - 'alpha'    : constante de la régression OLS
      - 'halflife' : demi-vie estimée du spread (en jours)
    """
    # statsmodels.coint = ADF sur les résidus d'une régression OLS
    score, pvalue, _ = coint(y, x)

    # Régression OLS pour récupérer β et α
    x_const = add_constant(x)
    model = OLS(y, x_const).fit()
    alpha = model.params.iloc[0]  # constante
    beta = model.params.iloc[1]  # coeff de X

    # Résidu = spread brut
    spread = y - (alpha + beta * x)

    # Estimation de la half-life via OU (cf. fonction dédiée)
    hl = estimate_halflife(spread)

    return {
        "pvalue": pvalue,
        "alpha": alpha,
        "beta": beta,
        "halflife": hl,
    }


# ─────────────────────────────────────────────────────────────────────────────
# Sélection de toutes les paires cointégrées
# ─────────────────────────────────────────────────────────────────────────────


def find_cointegrated_pairs(
    prices: pd.DataFrame,
    pvalue_threshold: float = 0.05,
    min_halflife: float = 5.0,
    max_halflife: float = 120.0,
) -> pd.DataFrame:
    """
    Parcourt toutes les combinaisons de paires du DataFrame et retourne
    celles dont la p-value Engle-Granger est < pvalue_threshold.

    Filtres supplémentaires sur la half-life :
      - min_halflife : spread qui mean-reverte trop vite (bruit)
      - max_halflife : spread qui mean-reverte trop lentement (quasi-aléatoire)

    Retourne un DataFrame trié par p-value croissante avec colonnes :
      ticker_y, ticker_x, pvalue, alpha, beta, halflife
    """
    tickers = prices.columns.tolist()
    results = []

    n_pairs = len(list(combinations(tickers, 2)))
    print(f"[cointegration] Test de {n_pairs} paires …")

    for y_name, x_name in combinations(tickers, 2):
        y = prices[y_name]
        x = prices[x_name]

        try:
            stats = engle_granger_test(y, x)
        except Exception as e:
            # Ignorer les paires problématiques (données dégénérées, etc.)
            continue

        if stats["pvalue"] < pvalue_threshold:
            hl = stats["halflife"]
            # Filtre sur la plausibilité de la mean-reversion
            if min_halflife <= hl <= max_halflife:
                results.append(
                    {
                        "ticker_y": y_name,
                        "ticker_x": x_name,
                        "pvalue": stats["pvalue"],
                        "alpha": stats["alpha"],
                        "beta": stats["beta"],
                        "halflife": hl,
                    }
                )

    pairs_df = pd.DataFrame(results).sort_values("pvalue").reset_index(drop=True)
    print(
        f"[cointegration] {len(pairs_df)} paires cointégrées trouvées "
        f"(seuil p={pvalue_threshold}, hl=[{min_halflife},{max_halflife}] jours)."
    )
    return pairs_df


# ─────────────────────────────────────────────────────────────────────────────
# Test de Johansen (option premium)
# ─────────────────────────────────────────────────────────────────────────────


def johansen_test(
    prices: pd.DataFrame,
    det_order: int = 0,
    k_ar_diff: int = 1,
) -> Dict:
    """
    Test de Johansen pour détecter le rang de cointégration d'un système
    multivariable (> 2 séries).

    Paramètres
    ----------
    det_order : -1 (sans constante), 0 (constante), 1 (tendance)
    k_ar_diff : nombre de lags dans le VAR différencié

    Retourne un dict avec :
      - 'trace_stat'    : statistiques de trace
      - 'trace_crit_5%' : valeurs critiques à 5 %
      - 'max_eig_stat'  : statistiques max-eigenvalue
      - 'max_eig_crit'  : valeurs critiques à 5 %
      - 'evec'          : vecteurs de cointégration (colonnes)
    """
    result = coint_johansen(prices, det_order=det_order, k_ar_diff=k_ar_diff)

    # Le rang de cointégration = nombre de statistiques > valeur critique
    trace_rejects = (result.lr1 > result.cvt[:, 1]).sum()  # seuil 5 %

    print(
        f"[cointegration] Johansen : rang de cointégration ≥ {trace_rejects} "
        f"(test de trace, 5 %)"
    )

    return {
        "trace_stat": result.lr1,
        "trace_crit_5pct": result.cvt[:, 1],
        "max_eig_stat": result.lr2,
        "max_eig_crit": result.cvm[:, 1],
        "evec": result.evec,  # vecteurs propres = portefeuilles stationnaires
        "rank": trace_rejects,
    }


# ─────────────────────────────────────────────────────────────────────────────
# Half-life du processus d'Ornstein-Uhlenbeck
# ─────────────────────────────────────────────────────────────────────────────


def estimate_halflife(spread: pd.Series) -> float:
    """
    Estime la half-life d'un processus OU en régressant :

        ΔS_t = λ · S_{t-1} + ε_t

    La mean-reversion speed est κ = -λ, et la half-life est :

        hl = ln(2) / κ   (en unités de la fréquence des données — ici en jours)

    Une half-life trop courte (<5 j) : bruit pur / coûts de transaction prohibitifs.
    Une half-life trop longue (>252 j) : pas de mean-reversion exploitable.
    """
    spread_lag = spread.shift(1)
    spread_diff = spread.diff()

    # Supprimer le premier NaN introduit par le lag
    df = pd.DataFrame({"diff": spread_diff, "lag": spread_lag}).dropna()

    # Régression OLS : ΔS = λ·S_{t-1} + ε
    x_const = add_constant(df["lag"])
    model = OLS(df["diff"], x_const).fit()
    lambda_ = model.params["lag"]

    # λ doit être négatif pour qu'il y ait mean-reversion
    if lambda_ >= 0:
        return np.inf  # Pas de mean-reversion

    halflife = np.log(2) / (-lambda_)
    return halflife


# ─────────────────────────────────────────────────────────────────────────────
# Calcul du spread standardisé Z_t
# ─────────────────────────────────────────────────────────────────────────────


def compute_spread(
    prices: pd.DataFrame,
    ticker_y: str,
    ticker_x: str,
    alpha: float,
    beta: float,
    window: Optional[int] = None,
) -> pd.DataFrame:
    """
    Calcule le spread S_t et son score Z_t pour une paire donnée.

    Formule :
        S_t = Y_t - (α + β · X_t)
        Z_t = (S_t - μ) / σ

    Si window est spécifié, μ et σ sont des moyennes roulantes sur 'window' jours
    (robuste contre les dérives de long terme — recommandé pour le live trading).
    Sinon, μ et σ sont calculés sur toute la période (adapté pour le backtest).

    Retourne un DataFrame avec colonnes ['spread', 'zscore'].
    """
    y = prices[ticker_y]
    x = prices[ticker_x]

    # Spread brut
    spread = y - (alpha + beta * x)

    if window is not None:
        # Statistiques roulantes : évite le look-ahead en utilisant .shift(1)
        # (les stats du jour J sont calculées avec les données de J-1 à J-window)
        mu = spread.rolling(window).mean()
        sigma = spread.rolling(window).std()
    else:
        # Statistiques globales (calculées sur l'ensemble train)
        mu = spread.mean()
        sigma = spread.std()

    # Z-score : nombre d'écarts-types par rapport à la moyenne
    zscore = (spread - mu) / sigma

    return pd.DataFrame({"spread": spread, "zscore": zscore}, index=prices.index)


# ─────────────────────────────────────────────────────────────────────────────
# Point d'entrée
# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import sys

    sys.path.insert(0, ".")
    from data_loader import download_prices, clean_prices, train_test_split

    prices = clean_prices(download_prices())
    train, test = train_test_split(prices)

    pairs_df = find_cointegrated_pairs(train)
    print(pairs_df.head(10).to_string())

    # Exemple : spreads de la meilleure paire
    if not pairs_df.empty:
        best = pairs_df.iloc[0]
        spread_df = compute_spread(
            train,
            best["ticker_y"],
            best["ticker_x"],
            best["alpha"],
            best["beta"],
        )
        print(spread_df.tail())
