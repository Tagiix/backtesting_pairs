"""
Walk-Forward Analysis
==================================================
Évalue la robustesse de la stratégie en ré-estimant les paramètres sur des
fenêtres glissantes et en testant hors-échantillon (out-of-sample).

Principe :
  - Pour chaque fenêtre (train_i, test_i) générée par walk_forward_splits() :
      1. Identifier les paires cointégrées sur train_i
      2. Estimer α, β pour chaque paire
      3. Backtester sur test_i avec les paramètres de train_i
      4. Calculer les métriques out-of-sample

Cela évite le data snooping : les paramètres de chaque fenêtre n'ont
jamais "vu" les données de test correspondantes.
"""

import numpy as np
import pandas as pd
from typing import List, Dict, Optional, Tuple

from data_loader import walk_forward_splits
from cointegration import find_cointegrated_pairs, compute_spread
from backtester import Backtester
from metrics import compute_all_metrics


# ─────────────────────────────────────────────────────────────────────────────
# Backtest d'une fenêtre (train, test)
# ─────────────────────────────────────────────────────────────────────────────


def backtest_window(
    train: pd.DataFrame,
    test: pd.DataFrame,
    capital_per_pair: float = 500_000.0,
    max_pairs: int = 3,
    entry_z: float = 2.0,
    exit_z: float = 0.5,
    stop_z: Optional[float] = 3.5,
    pvalue_threshold: float = 0.05,
    transaction_cost: float = 0.001,
    slippage_bps: float = 2.0,
) -> Dict:
    """
    Exécute un cycle train → test :
      1. Sélectionner les paires cointégrées sur 'train'
      2. Backtester sur 'test' avec les paramètres estimés sur 'train'

    Paramètres
    ----------
    max_pairs        : nombre de paires à utiliser (les plus significatives)
    capital_per_pair : capital alloué par paire

    Retourne un dict avec :
      - 'train_start', 'train_end' : bornes de la fenêtre d'entraînement
      - 'test_start',  'test_end'  : bornes de la fenêtre de test
      - 'n_pairs'     : nombre de paires cointégrées trouvées
      - 'pairs'       : DataFrame des paires sélectionnées
      - 'metrics'     : métriques out-of-sample
      - 'pnl'         : Series PnL quotidien out-of-sample
    """
    # Sélection des paires sur le train
    pairs_df = find_cointegrated_pairs(train, pvalue_threshold=pvalue_threshold)

    if pairs_df.empty:
        return {
            "train_start": train.index[0],
            "train_end": train.index[-1],
            "test_start": test.index[0],
            "test_end": test.index[-1],
            "n_pairs": 0,
            "pairs": pd.DataFrame(),
            "metrics": {},
            "pnl": pd.Series(dtype=float),
        }

    # Limiter au nombre de paires demandé (les plus significatives)
    pairs_selected = pairs_df.head(max_pairs)

    # Backtest sur le test avec les paramètres du train
    # ⚠ On passe 'test' au backtester (out-of-sample), mais les paramètres
    #   (alpha, beta) ont été estimés sur 'train' : aucun look-ahead.
    bt = Backtester(
        initial_capital=capital_per_pair * len(pairs_selected),
        transaction_cost=transaction_cost,
        slippage_bps=slippage_bps,
    )

    # On doit avoir accès aux tickers de 'test' qui ont été sélectionnés sur 'train'
    test_tickers = set(test.columns)
    for _, row in pairs_selected.iterrows():
        if row["ticker_y"] in test_tickers and row["ticker_x"] in test_tickers:
            bt.add_pair(
                test,
                row["ticker_y"],
                row["ticker_x"],
                alpha=row["alpha"],
                beta=row["beta"],
                capital_alloc=capital_per_pair,
                entry_z=entry_z,
                exit_z=exit_z,
                stop_z=stop_z,
            )

    if not bt._pairs:
        return {
            "train_start": train.index[0],
            "train_end": train.index[-1],
            "test_start": test.index[0],
            "test_end": test.index[-1],
            "n_pairs": 0,
            "pairs": pairs_selected,
            "metrics": {},
            "pnl": pd.Series(dtype=float),
        }

    results = bt.run()
    pnl = results["portfolio_pnl"]
    metrics = compute_all_metrics(
        pnl=pnl,
        exposure=results["exposure"],
        initial_capital=capital_per_pair * len(pairs_selected),
    )

    return {
        "train_start": train.index[0],
        "train_end": train.index[-1],
        "test_start": test.index[0],
        "test_end": test.index[-1],
        "n_pairs": len(bt._pairs),
        "pairs": pairs_selected,
        "metrics": metrics,
        "pnl": pnl,
    }


# ─────────────────────────────────────────────────────────────────────────────
# Walk-forward complet sur toutes les fenêtres
# ─────────────────────────────────────────────────────────────────────────────


def run_walk_forward(
    prices: pd.DataFrame,
    train_years: int = 5,
    test_years: int = 1,
    capital_per_pair: float = 500_000.0,
    max_pairs: int = 3,
    entry_z: float = 2.0,
    exit_z: float = 0.5,
    stop_z: Optional[float] = 3.5,
    pvalue_threshold: float = 0.05,
    transaction_cost: float = 0.001,
    slippage_bps: float = 2.0,
) -> Dict:
    """
    Exécute la walk-forward analysis complète sur toutes les fenêtres glissantes.

    Retourne un dict avec :
      - 'window_results' : liste de dicts (un par fenêtre)
      - 'combined_pnl'   : Series du PnL OOS concaténé sur toutes les fenêtres
      - 'metrics_df'     : DataFrame des métriques OOS par fenêtre
      - 'global_metrics' : métriques agrégées sur tout le PnL OOS
    """
    splits = walk_forward_splits(prices, train_years, test_years)

    if not splits:
        raise ValueError("Pas assez de données pour générer des fenêtres walk-forward.")

    print(f"[walk_forward] {len(splits)} fenêtres à backtester …")

    window_results = []
    all_pnl_series = []

    for i, (train, test) in enumerate(splits):
        print(
            f"\n[walk_forward] Fenêtre {i + 1}/{len(splits)} : "
            f"train [{train.index[0].date()} → {train.index[-1].date()}], "
            f"test [{test.index[0].date()} → {test.index[-1].date()}]"
        )

        window = backtest_window(
            train,
            test,
            capital_per_pair=capital_per_pair,
            max_pairs=max_pairs,
            entry_z=entry_z,
            exit_z=exit_z,
            stop_z=stop_z,
            pvalue_threshold=pvalue_threshold,
            transaction_cost=transaction_cost,
            slippage_bps=slippage_bps,
        )
        window_results.append(window)

        if not window["pnl"].empty:
            all_pnl_series.append(window["pnl"])

    # Concaténer tous les PnL OOS (sans chevauchement par construction)
    if all_pnl_series:
        combined_pnl = pd.concat(all_pnl_series).sort_index()
    else:
        combined_pnl = pd.Series(dtype=float)

    # Résumé par fenêtre
    metrics_rows = []
    for w in window_results:
        row = {
            "test_start": w["test_start"],
            "test_end": w["test_end"],
            "n_pairs": w["n_pairs"],
        }
        row.update(w["metrics"])
        metrics_rows.append(row)

    metrics_df = pd.DataFrame(metrics_rows)

    # Métriques globales sur tout le PnL OOS concaténé
    global_metrics = compute_all_metrics(combined_pnl) if not combined_pnl.empty else {}

    print(
        f"\n[walk_forward] Sharpe OOS global : "
        f"{global_metrics.get('sharpe', np.nan):.3f}"
    )

    return {
        "window_results": window_results,
        "combined_pnl": combined_pnl,
        "metrics_df": metrics_df,
        "global_metrics": global_metrics,
    }


# ─────────────────────────────────────────────────────────────────────────────
# Stabilité des paramètres (rolling betas)
# ─────────────────────────────────────────────────────────────────────────────


def rolling_beta(
    prices: pd.DataFrame,
    ticker_y: str,
    ticker_x: str,
    window: int = 252,
) -> pd.Series:
    """
    Calcule le beta roulant β_t sur une fenêtre glissante de 'window' jours.
    Permet de visualiser la stabilité de la relation de cointégration dans le temps.

    Un beta instable (forte variation de β_t) indique une relation fragile.
    """
    y = prices[ticker_y]
    x = prices[ticker_x]

    betas = pd.Series(index=prices.index, dtype=float)

    for i in range(window, len(prices)):
        y_w = y.iloc[i - window : i]
        x_w = x.iloc[i - window : i]

        # OLS rapide : β = Cov(Y,X) / Var(X)
        cov = np.cov(y_w, x_w)
        b = cov[0, 1] / cov[1, 1] if cov[1, 1] > 0 else np.nan
        betas.iloc[i] = b

    return betas


# ─────────────────────────────────────────────────────────────────────────────
# Point d'entrée rapide
# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import sys

    sys.path.insert(0, ".")
    from data_loader import download_prices, clean_prices

    prices = clean_prices(download_prices())
    wf = run_walk_forward(prices, train_years=5, test_years=1, max_pairs=3)

    print("\n--- Métriques par fenêtre OOS ---")
    cols = [
        "test_start",
        "test_end",
        "n_pairs",
        "sharpe",
        "max_drawdown_$",
        "hit_ratio",
    ]
    available = [c for c in cols if c in wf["metrics_df"].columns]
    print(wf["metrics_df"][available].to_string(index=False))

    print("\n--- Métriques globales OOS ---")
    for k, v in wf["global_metrics"].items():
        print(f"  {k:<30} {v:.4f}" if isinstance(v, float) else f"  {k:<30} {v}")
