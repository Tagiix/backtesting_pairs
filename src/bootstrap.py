"""
Bootstrap du Sharpe ratio
===================================================
Implémente le block bootstrap (Stationary Bootstrap de Politis & Romano, 1994)
pour construire la distribution empirique du Sharpe ratio et calculer un
intervalle de confiance à 95 %.

Pourquoi le block bootstrap ?
  - Les rendements quotidiens présentent de l'autocorrélation (dépendance temporelle).
  - Un bootstrap i.i.d. classique détruirait cette structure.
  - Le block bootstrap rééchantillonne des blocs de jours consécutifs,
    préservant la dépendance temporelle à court terme.

Référence : Politis, D.N., Romano, J.P. (1994). "The Stationary Bootstrap".
           JASA, 89(428), 1303-1313.
"""

import numpy as np
import pandas as pd
from typing import Tuple, List, Optional

from metrics import sharpe_ratio


# ─────────────────────────────────────────────────────────────────────────────
# Stationary Block Bootstrap
# ─────────────────────────────────────────────────────────────────────────────


def stationary_block_bootstrap(
    data: np.ndarray,
    block_length: Optional[int] = None,
    n_samples: int = 1,
    rng: Optional[np.random.Generator] = None,
) -> np.ndarray:
    """
    Génère n_samples séquences rééchantillonnées via le Stationary Block Bootstrap.

    Dans la version stationnaire (Politis & Romano), la longueur de chaque bloc
    suit une loi géométrique de paramètre 1/block_length.
    Cela rend la procédure stationnaire (contrairement au bootstrap en blocs fixes).

    Paramètres
    ----------
    data         : array 1D de longueur T (les observations originales)
    block_length : longueur moyenne des blocs (heuristique : √T ou calibré sur ACF)
    n_samples    : nombre de séquences bootstrap à générer
    rng          : générateur de nombres aléatoires (pour reproductibilité)

    Retourne
    --------
    Array de shape (n_samples, T) — chaque ligne est une séquence bootstrap.
    """
    T = len(data)

    if block_length is None:
        # Heuristique : block_length ≈ T^(1/3) pour données financières
        block_length = max(1, int(T ** (1 / 3)))

    if rng is None:
        rng = np.random.default_rng()

    # Probabilité de fin de bloc (distribution géométrique)
    p_end = 1.0 / block_length

    bootstrapped = np.empty((n_samples, T))

    for s in range(n_samples):
        result = []
        current_idx = rng.integers(0, T)  # Départ aléatoire dans la série

        while len(result) < T:
            result.append(data[current_idx % T])

            # Fin de bloc aléatoire (Bernoulli de paramètre p_end)
            if rng.random() < p_end or len(result) >= T:
                current_idx = rng.integers(0, T)  # Nouveau point de départ
            else:
                current_idx = (current_idx + 1) % T  # Continuer le bloc

        bootstrapped[s] = np.array(result[:T])

    return bootstrapped


# ─────────────────────────────────────────────────────────────────────────────
# Distribution bootstrap du Sharpe
# ─────────────────────────────────────────────────────────────────────────────


def bootstrap_sharpe(
    pnl: pd.Series,
    n_bootstrap: int = 1000,
    block_length: Optional[int] = None,
    risk_free_rate: float = 0.0,
    confidence_level: float = 0.95,
    seed: int = 42,
) -> dict:
    """
    Construit la distribution empirique du Sharpe ratio par block bootstrap.

    Procédure :
      1. Rééchantillonner la série de PnL B fois (block bootstrap).
      2. Calculer le Sharpe sur chaque échantillon.
      3. Construire l'IC à (1-α) % par percentiles.
      4. Calculer le p-value empirique (H0 : Sharpe ≤ 0).

    Paramètres
    ----------
    n_bootstrap      : nombre d'itérations bootstrap (B ≥ 1000 recommandé)
    block_length     : longueur des blocs (None = heuristique automatique)
    confidence_level : niveau de l'IC (ex: 0.95 pour 95 %)
    seed             : graine pour reproductibilité

    Retourne un dict avec :
      - 'sharpe_observed'  : Sharpe de la série originale
      - 'sharpe_bootstrap' : array des B Sharpe bootstrapés
      - 'ci_lower'         : borne inférieure de l'IC
      - 'ci_upper'         : borne supérieure de l'IC
      - 'pvalue'           : fraction des bootstraps avec Sharpe ≤ 0
      - 'is_significant'   : True si le Sharpe est statistiquement > 0
      - 'block_length'     : longueur des blocs utilisée
    """
    pnl_arr = pnl.values
    T = len(pnl_arr)

    if block_length is None:
        block_length = max(1, int(T ** (1 / 3)))

    rng = np.random.default_rng(seed)

    print(
        f"[bootstrap] Block bootstrap : B={n_bootstrap}, "
        f"block_length={block_length}, T={T} …"
    )

    # Générer toutes les séquences bootstrap d'un coup (vectorisé)
    bootstrapped = stationary_block_bootstrap(
        pnl_arr, block_length=block_length, n_samples=n_bootstrap, rng=rng
    )

    # Sharpe sur chaque réplication
    sharpe_boot = np.array(
        [
            sharpe_ratio(pd.Series(bootstrapped[b]), risk_free_rate)
            for b in range(n_bootstrap)
        ]
    )

    # Supprimer les NaN éventuels (réplications dégénérées)
    sharpe_boot = sharpe_boot[~np.isnan(sharpe_boot)]

    # IC par percentiles (méthode percentile — simple et robuste)
    alpha = 1.0 - confidence_level
    ci_lower = float(np.percentile(sharpe_boot, 100 * alpha / 2))
    ci_upper = float(np.percentile(sharpe_boot, 100 * (1 - alpha / 2)))

    # P-value : fraction des bootstraps avec Sharpe ≤ 0 (test unilatéral H0: SR≤0)
    pvalue = float((sharpe_boot <= 0).mean())

    sharpe_obs = sharpe_ratio(pnl, risk_free_rate)

    print(f"[bootstrap] Sharpe observé : {sharpe_obs:.3f}")
    print(
        f"[bootstrap] IC {int(confidence_level * 100)}% : "
        f"[{ci_lower:.3f}, {ci_upper:.3f}]"
    )
    print(f"[bootstrap] P-value (H0: SR≤0) : {pvalue:.4f}")

    return {
        "sharpe_observed": sharpe_obs,
        "sharpe_bootstrap": sharpe_boot,
        "ci_lower": ci_lower,
        "ci_upper": ci_upper,
        "pvalue": pvalue,
        "is_significant": pvalue < (1 - confidence_level),
        "block_length": block_length,
        "n_bootstrap": n_bootstrap,
    }


# ─────────────────────────────────────────────────────────────────────────────
# Test de significativité du Sharpe (méthode Lo 2002 — analytique)
# ─────────────────────────────────────────────────────────────────────────────


def sharpe_significance_lo2002(
    pnl: pd.Series,
    risk_free_rate: float = 0.0,
    q: int = 12,
) -> dict:
    """
    Test de significativité du Sharpe ratio selon Lo (2002),
    qui corrige pour l'autocorrélation des rendements.

    Référence : Lo, A.W. (2002). "The Statistics of Sharpe Ratios".
               Financial Analysts Journal, 58(4), 36-52.

    Paramètres
    ----------
    q : horizon temporel d'annualisation (252 pour quotidien→annuel)

    Retourne le Sharpe corrigé, son écart-type asymptotique, et la stat-t.
    """
    r = pnl.values
    T = len(r)
    rf = risk_free_rate / 252

    mu = np.mean(r - rf)
    sigma = np.std(r - rf, ddof=1)
    sr = mu / sigma

    # Estimer l'autocorrélation ρ_k pour k = 1, ..., q
    # Formule de Lo (2002) pour l'écart-type du SR avec autocorrélation
    rho = np.array(
        [np.corrcoef(r[:-k], r[k:])[0, 1] if k > 0 else 1.0 for k in range(q + 1)]
    )

    # Terme de correction pour autocorrélation (équation (12) de Lo 2002)
    correction = sum((1 - k / q) * rho[k] for k in range(1, q + 1))

    # Variance asymptotique du SR annualisé
    var_sr = (1 / T) * (1 + 0.5 * sr**2 - correction) * q  # approximation

    se_sr = np.sqrt(max(var_sr, 0))
    sr_ann = sr * np.sqrt(252)

    # Stat-t (test H0: SR = 0)
    t_stat = sr_ann / se_sr if se_sr > 0 else np.nan

    # P-value unilatérale
    from scipy import stats

    pvalue = float(stats.t.sf(t_stat, df=T - 1)) if not np.isnan(t_stat) else np.nan

    return {
        "sharpe_annualized": sr_ann,
        "std_error": se_sr,
        "t_stat": t_stat,
        "pvalue_lo2002": pvalue,
        "is_significant_5pct": pvalue < 0.05 if not np.isnan(pvalue) else False,
    }


# ─────────────────────────────────────────────────────────────────────────────
# Point d'entrée
# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import sys

    sys.path.insert(0, ".")
    from data_loader import download_prices, clean_prices, train_test_split
    from cointegration import find_cointegrated_pairs
    from backtester import run_single_pair_backtest

    prices = clean_prices(download_prices())
    train, test = train_test_split(prices)
    pairs_df = find_cointegrated_pairs(train)

    if pairs_df.empty:
        print("Aucune paire trouvée.")
        sys.exit(0)

    best = pairs_df.iloc[0]
    detail = run_single_pair_backtest(
        train, best["ticker_y"], best["ticker_x"], best["alpha"], best["beta"]
    )

    # Bootstrap du Sharpe
    boot_results = bootstrap_sharpe(detail["pnl"], n_bootstrap=1000)
    print(f"\nSignificatif à 5% : {boot_results['is_significant']}")

    # Test analytique de Lo (2002)
    lo_results = sharpe_significance_lo2002(detail["pnl"])
    print(
        f"Lo (2002) — SR : {lo_results['sharpe_annualized']:.3f}, "
        f"p-value : {lo_results['pvalue_lo2002']:.4f}"
    )
