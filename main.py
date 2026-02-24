"""
main.py — Point d'entrée principal du projet Pairs Trading
===========================================================
Orchestre l'ensemble du pipeline en 10 étapes :
  1. Collecte et nettoyage des données
  2. Sélection des paires cointégrées
  3-4. Modélisation du spread et construction du signal
  5. Backtest propre (in-sample)
  6. Calcul des métriques avancées
  7. Bootstrap du Sharpe (significativité statistique)
  8. Walk-forward analysis (robustesse OOS)
  9. Monte Carlo du PnL (distribution future)
  10. Export des résultats

Usage :
  python main.py [--tickers XOM CVX KO PEP] [--start 2015-01-01] [--end 2024-12-31]
"""

import argparse
import os
import sys
import numpy as np
import pandas as pd

# Ajout du dossier src au PYTHONPATH
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))

from data_loader    import download_prices, clean_prices, train_test_split
from cointegration  import find_cointegrated_pairs, compute_spread
from backtester     import Backtester
from metrics        import compute_all_metrics, print_metrics
from bootstrap      import bootstrap_sharpe, sharpe_significance_lo2002
from monte_carlo    import run_monte_carlo
from walk_forward   import run_walk_forward


# ─────────────────────────────────────────────────────────────────────────────
# Configuration par défaut
# ─────────────────────────────────────────────────────────────────────────────

DEFAULT_TICKERS = [
    "XOM", "CVX", "JPM", "BAC", "KO", "PEP",
    "GLD", "SLV", "SPY", "QQQ", "WMT", "TGT",
]

DEFAULT_START   = "2015-01-01"
DEFAULT_END     = "2024-12-31"
CAPITAL         = 10_000_000.0   # Capital total du portefeuille ($)
CAPITAL_PER_PAIR = 500_000.0     # Capital alloué par paire ($)
MAX_PAIRS       = 5              # Nombre maximum de paires à trader


# ─────────────────────────────────────────────────────────────────────────────
# Pipeline principal
# ─────────────────────────────────────────────────────────────────────────────

def run_pipeline(
    tickers: list,
    start: str,
    end: str,
    run_walk_fwd: bool = True,
    run_mc: bool = True,
    run_boot: bool = True,
    save_results: bool = True,
) -> dict:
    """
    Exécute l'intégralité du pipeline de backtesting.

    Paramètres
    ----------
    run_walk_fwd : activer la walk-forward analysis (lente sur grands datasets)
    run_mc       : activer la simulation Monte Carlo
    run_boot     : activer le bootstrap du Sharpe
    save_results : sauvegarder les résultats dans results/

    Retourne un dict avec tous les résultats intermédiaires.
    """

    # ── Étape 1 : Collecte et préparation des données ──────────────────────
    print("\n" + "="*60)
    print("  ÉTAPE 1 — Collecte et nettoyage des données")
    print("="*60)
    prices = download_prices(tickers, start, end)
    prices = clean_prices(prices)
    train, test = train_test_split(prices, train_years=5, test_years=1)

    # ── Étape 2 : Sélection des paires cointégrées ─────────────────────────
    print("\n" + "="*60)
    print("  ÉTAPE 2 — Sélection des paires cointégrées (train)")
    print("="*60)
    pairs_df = find_cointegrated_pairs(train, pvalue_threshold=0.05)

    if pairs_df.empty:
        print("⚠ Aucune paire cointégrée trouvée. Essayer d'autres tickers.")
        return {}

    print(f"\nTop {min(10, len(pairs_df))} paires :")
    print(pairs_df.head(10).to_string(index=False))

    # Sélectionner les meilleures paires (les plus significatives)
    pairs_selected = pairs_df.head(MAX_PAIRS)

    # ── Étapes 3–5 : Backtest in-sample ────────────────────────────────────
    print("\n" + "="*60)
    print("  ÉTAPES 3–5 — Backtest in-sample (train)")
    print("="*60)

    bt_train = Backtester(
        initial_capital=CAPITAL,
        transaction_cost=0.001,
        slippage_bps=2.0,
    )
    for _, row in pairs_selected.iterrows():
        if row["ticker_y"] in train.columns and row["ticker_x"] in train.columns:
            bt_train.add_pair(
                train,
                row["ticker_y"],
                row["ticker_x"],
                alpha=row["alpha"],
                beta=row["beta"],
                capital_alloc=CAPITAL_PER_PAIR,
            )

    train_results = bt_train.run()

    # ── Étapes 3–5 : Backtest out-of-sample ────────────────────────────────
    print("\n" + "="*60)
    print("  ÉTAPES 3–5 — Backtest out-of-sample (test)")
    print("="*60)

    bt_test = Backtester(
        initial_capital=CAPITAL,
        transaction_cost=0.001,
        slippage_bps=2.0,
    )
    for _, row in pairs_selected.iterrows():
        if row["ticker_y"] in test.columns and row["ticker_x"] in test.columns:
            bt_test.add_pair(
                test,
                row["ticker_y"],
                row["ticker_x"],
                alpha=row["alpha"],   # Paramètres estimés sur TRAIN — pas de look-ahead
                beta=row["beta"],
                capital_alloc=CAPITAL_PER_PAIR,
            )

    test_results = bt_test.run()

    # ── Étape 6 : Métriques avancées ───────────────────────────────────────
    print("\n" + "="*60)
    print("  ÉTAPE 6 — Métriques avancées")
    print("="*60)

    print("\n[IN-SAMPLE]")
    train_metrics = compute_all_metrics(
        pnl=train_results["portfolio_pnl"],
        exposure=train_results["exposure"],
        initial_capital=CAPITAL,
    )
    print_metrics(train_metrics)

    print("\n[OUT-OF-SAMPLE]")
    test_metrics = compute_all_metrics(
        pnl=test_results["portfolio_pnl"],
        exposure=test_results["exposure"],
        initial_capital=CAPITAL,
    )
    print_metrics(test_metrics)

    # ── Étape 7 : Bootstrap du Sharpe ──────────────────────────────────────
    boot_results = {}
    lo_results   = {}
    if run_boot:
        print("\n" + "="*60)
        print("  ÉTAPE 7 — Bootstrap du Sharpe (out-of-sample)")
        print("="*60)
        boot_results = bootstrap_sharpe(
            test_results["portfolio_pnl"],
            n_bootstrap=1000,
            seed=42,
        )
        lo_results = sharpe_significance_lo2002(test_results["portfolio_pnl"])
        print(f"  Significatif (5%) — bootstrap : {boot_results['is_significant']}")
        print(f"  Significatif (5%) — Lo 2002   : {lo_results['is_significant_5pct']}")

    # ── Étape 8 : Walk-forward analysis ────────────────────────────────────
    wf_results = {}
    if run_walk_fwd:
        print("\n" + "="*60)
        print("  ÉTAPE 8 — Walk-forward analysis")
        print("="*60)
        wf_results = run_walk_forward(
            prices,
            train_years=5,
            test_years=1,
            capital_per_pair=CAPITAL_PER_PAIR,
            max_pairs=MAX_PAIRS,
        )
        print(f"\n  Sharpe OOS global (walk-forward) : "
              f"{wf_results['global_metrics'].get('sharpe', float('nan')):.3f}")

    # ── Étape 9 : Monte Carlo du PnL ───────────────────────────────────────
    mc_results = {}
    if run_mc and not pairs_selected.empty:
        print("\n" + "="*60)
        print("  ÉTAPE 9 — Monte Carlo du PnL (meilleure paire)")
        print("="*60)
        best = pairs_selected.iloc[0]
        if best["ticker_y"] in train.columns and best["ticker_x"] in train.columns:
            spread_df = compute_spread(
                train, best["ticker_y"], best["ticker_x"],
                best["alpha"], best["beta"]
            )
            mc_results = run_monte_carlo(
                spread=spread_df["spread"],
                n_paths=2000,
                horizon_days=252,
            )

    # ── Étape 10 : Sauvegarde des résultats ────────────────────────────────
    if save_results:
        print("\n" + "="*60)
        print("  ÉTAPE 10 — Sauvegarde des résultats")
        print("="*60)
        os.makedirs("results", exist_ok=True)

        # PnL journalier train/test
        train_results["portfolio_pnl"].to_csv("results/pnl_train.csv")
        test_results["portfolio_pnl"].to_csv("results/pnl_test.csv")

        # PnL cumulé
        train_results["cumulative_pnl"].to_csv("results/cumulative_pnl_train.csv")
        test_results["cumulative_pnl"].to_csv("results/cumulative_pnl_test.csv")

        # Paires sélectionnées
        pairs_df.to_csv("results/pairs.csv", index=False)

        # Métriques
        pd.DataFrame([train_metrics]).to_csv("results/metrics_train.csv", index=False)
        pd.DataFrame([test_metrics]).to_csv("results/metrics_test.csv", index=False)

        # Walk-forward
        if wf_results and "metrics_df" in wf_results:
            wf_results["metrics_df"].to_csv("results/walk_forward_metrics.csv", index=False)
            wf_results["combined_pnl"].to_csv("results/walk_forward_pnl.csv")

        # Monte Carlo
        if mc_results and "pnl_distribution" in mc_results:
            pd.Series(mc_results["pnl_distribution"]).to_csv(
                "results/monte_carlo_pnl_dist.csv", index=False
            )

        print("  Résultats sauvegardés dans results/")

    return {
        "prices":         prices,
        "train":          train,
        "test":           test,
        "pairs_df":       pairs_df,
        "train_results":  train_results,
        "test_results":   test_results,
        "train_metrics":  train_metrics,
        "test_metrics":   test_metrics,
        "boot_results":   boot_results,
        "lo_results":     lo_results,
        "wf_results":     wf_results,
        "mc_results":     mc_results,
    }


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

def parse_args():
    parser = argparse.ArgumentParser(
        description="Pairs Trading Backtester — Quant Finance Project"
    )
    parser.add_argument(
        "--tickers", nargs="+", default=DEFAULT_TICKERS,
        help="Liste de tickers à utiliser (ex: XOM CVX KO PEP)"
    )
    parser.add_argument("--start", default=DEFAULT_START, help="Date de début (YYYY-MM-DD)")
    parser.add_argument("--end",   default=DEFAULT_END,   help="Date de fin   (YYYY-MM-DD)")
    parser.add_argument("--no-walk-forward", action="store_true",
                        help="Désactiver la walk-forward analysis")
    parser.add_argument("--no-monte-carlo",  action="store_true",
                        help="Désactiver la simulation Monte Carlo")
    parser.add_argument("--no-bootstrap",    action="store_true",
                        help="Désactiver le bootstrap du Sharpe")
    parser.add_argument("--no-save",         action="store_true",
                        help="Ne pas sauvegarder les résultats")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    results = run_pipeline(
        tickers=args.tickers,
        start=args.start,
        end=args.end,
        run_walk_fwd=not args.no_walk_forward,
        run_mc=not args.no_monte_carlo,
        run_boot=not args.no_bootstrap,
        save_results=not args.no_save,
    )

    print("\n✓ Pipeline terminé.")
