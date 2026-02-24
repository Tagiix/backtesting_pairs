"""
Moteur de backtest propre
====================================================
Simule le PnL journalier d'une stratégie de pairs trading en suivant :
  - Les positions de chaque jambe (Y et X)
  - Les variations de prix journalières (mark-to-market)
  - Les coûts de transaction et slippage
  - Le cash disponible
  - L'exposition totale

⚠ Principe fondamental : ZÉRO look-ahead bias.
   Toutes les décisions du jour J utilisent uniquement des informations
   disponibles en J-1 (les prix de clôture de la veille et les signaux décalés).
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple

from signals import build_pair_signals


# ─────────────────────────────────────────────────────────────────────────────
# Calcul du PnL journalier pour une paire
# ─────────────────────────────────────────────────────────────────────────────


def compute_pnl_pair(
    prices: pd.DataFrame,
    signals_df: pd.DataFrame,
    ticker_y: str,
    ticker_x: str,
) -> pd.Series:
    """
    Calcule le PnL mark-to-market (MTM) journalier pour une paire.

    PnL du jour J :
        pnl_J = pos_y_{J-1} × ΔP_y_J  +  pos_x_{J-1} × ΔP_x_J  -  coûts_J

    Où :
      - pos_y_{J-1} : position en titres Y détenue en fin de journée J-1
      - ΔP_y_J = P_y_J - P_y_{J-1} : variation de prix de Y le jour J
      - coûts_J : coûts de transaction éventuels (si on a tradé en J)

    Retourne une Series indexée par date avec le PnL quotidien.
    """
    py = prices[ticker_y]
    px = prices[ticker_x]

    # Variations de prix journalières
    delta_py = py.diff()
    delta_px = px.diff()

    # Positions détenues à la fin de J-1 (déjà décalées dans build_pair_signals)
    # On utilise shift(1) supplémentaire car les positions calculées reflètent
    # les ordres passés en J (exécutés à l'ouverture de J) → on gagne sur la clôture
    pos_y = signals_df["pos_y"]
    pos_x = signals_df["pos_x"]

    # PnL brut des deux jambes
    pnl_y = pos_y * delta_py
    pnl_x = pos_x * delta_px
    pnl_raw = pnl_y + pnl_x

    # Soustraction des coûts de transaction
    pnl_net = pnl_raw - signals_df["net_cost"]

    return pnl_net.fillna(0)


# ─────────────────────────────────────────────────────────────────────────────
# Moteur de backtest complet (multi-paires)
# ─────────────────────────────────────────────────────────────────────────────


class Backtester:
    """
    Moteur de backtest pour un portefeuille de paires.

    Caractéristiques :
      - Gestion multi-paires (capital alloué indépendamment à chaque paire)
      - Tracking du PnL cumulé, du cash et de l'exposition nette
      - Pas de levier excessif : chaque paire est dollar-neutral
      - Slippage et coûts de transaction configurables

    Utilisation
    -----------
    >>> bt = Backtester(initial_capital=10_000_000)
    >>> bt.add_pair(prices, "XOM", "CVX", alpha, beta, 100_000)
    >>> results = bt.run()
    >>> print(results["portfolio_pnl"].sum())
    """

    def __init__(
        self,
        initial_capital: float = 10_000_000.0,
        transaction_cost: float = 0.001,
        slippage_bps: float = 2.0,
    ):
        self.initial_capital = initial_capital
        self.transaction_cost = transaction_cost
        self.slippage_bps = slippage_bps
        self._pairs: List[Dict] = []  # Liste des paires à backtester

    def add_pair(
        self,
        prices: pd.DataFrame,
        ticker_y: str,
        ticker_x: str,
        alpha: float,
        beta: float,
        capital_alloc: float,
        entry_z: float = 2.0,
        exit_z: float = 0.5,
        stop_z: Optional[float] = 3.5,
        window: Optional[int] = None,
    ) -> None:
        """
        Enregistre une paire à backtester.

        Paramètres
        ----------
        capital_alloc : capital alloué à cette paire (en $)
        """
        self._pairs.append(
            {
                "prices": prices,
                "ticker_y": ticker_y,
                "ticker_x": ticker_x,
                "alpha": alpha,
                "beta": beta,
                "capital": capital_alloc,
                "entry_z": entry_z,
                "exit_z": exit_z,
                "stop_z": stop_z,
                "window": window,
            }
        )

    def run(self) -> Dict[str, pd.DataFrame]:
        """
        Exécute le backtest sur toutes les paires enregistrées.

        Retourne un dict avec :
          - 'pair_results'    : dict {(ticker_y, ticker_x) → DataFrame détaillé}
          - 'portfolio_pnl'   : Series du PnL journalier total du portefeuille
          - 'cumulative_pnl'  : Series du PnL cumulé
          - 'portfolio_value' : Series de la valeur totale du portefeuille
          - 'exposure'        : Series de l'exposition brute journalière
          - 'summary'         : DataFrame de résumé par paire
        """
        if not self._pairs:
            raise ValueError("Aucune paire ajoutée. Utiliser add_pair() d'abord.")

        pair_results = {}
        all_pnl = []
        all_exposure = []
        summary_rows = []

        for pair_cfg in self._pairs:
            ticker_y = pair_cfg["ticker_y"]
            ticker_x = pair_cfg["ticker_x"]
            prices = pair_cfg["prices"]

            print(f"[backtester] Backtest de la paire ({ticker_y}, {ticker_x}) …")

            # Génération des signaux pour cette paire
            signals_df = build_pair_signals(
                prices,
                ticker_y,
                ticker_x,
                pair_cfg["alpha"],
                pair_cfg["beta"],
                entry_z=pair_cfg["entry_z"],
                exit_z=pair_cfg["exit_z"],
                stop_z=pair_cfg["stop_z"],
                window=pair_cfg["window"],
                capital=pair_cfg["capital"],
                transaction_cost=self.transaction_cost,
                slippage_bps=self.slippage_bps,
            )

            # PnL journalier de la paire
            pnl = compute_pnl_pair(prices, signals_df, ticker_y, ticker_x)

            # Exposition brute (valeur absolue du notionnel engagé)
            py = prices[ticker_y]
            px = prices[ticker_x]
            exp_y = signals_df["pos_y"].abs() * py
            exp_x = signals_df["pos_x"].abs() * px
            exposure = exp_y + exp_x

            # Stockage des résultats détaillés de la paire
            pair_detail = pd.concat(
                [
                    signals_df,
                    pnl.rename("pnl"),
                    pnl.cumsum().rename("cumulative_pnl"),
                    exposure.rename("exposure"),
                ],
                axis=1,
            )
            pair_results[(ticker_y, ticker_x)] = pair_detail

            all_pnl.append(pnl)
            all_exposure.append(exposure)

            # Résumé par paire
            n_trades = (signals_df["signal"].diff().fillna(0) != 0).sum()
            summary_rows.append(
                {
                    "pair": f"{ticker_y}/{ticker_x}",
                    "total_pnl": pnl.sum(),
                    "n_trades": n_trades,
                    "avg_daily_pnl": pnl.mean(),
                    "pnl_std": pnl.std(),
                }
            )

        # Agréger le PnL de toutes les paires (alignement sur dates communes)
        pnl_matrix = pd.concat(all_pnl, axis=1).fillna(0)
        portfolio_pnl = pnl_matrix.sum(axis=1)
        cumulative_pnl = portfolio_pnl.cumsum()
        portfolio_value = self.initial_capital + cumulative_pnl

        exposure_matrix = pd.concat(all_exposure, axis=1).fillna(0)
        total_exposure = exposure_matrix.sum(axis=1)

        summary = pd.DataFrame(summary_rows)

        print(f"[backtester] PnL total : {portfolio_pnl.sum():,.0f} $")

        return {
            "pair_results": pair_results,
            "portfolio_pnl": portfolio_pnl,
            "cumulative_pnl": cumulative_pnl,
            "portfolio_value": portfolio_value,
            "exposure": total_exposure,
            "summary": summary,
        }


# ─────────────────────────────────────────────────────────────────────────────
# Fonction utilitaire : backtest d'une seule paire (interface simplifiée)
# ─────────────────────────────────────────────────────────────────────────────


def run_single_pair_backtest(
    prices: pd.DataFrame,
    ticker_y: str,
    ticker_x: str,
    alpha: float,
    beta: float,
    capital: float = 1_000_000.0,
    entry_z: float = 2.0,
    exit_z: float = 0.5,
    stop_z: Optional[float] = 3.5,
    window: Optional[int] = None,
    transaction_cost: float = 0.001,
    slippage_bps: float = 2.0,
) -> pd.DataFrame:
    """
    Interface simplifiée pour backtester une seule paire.
    Retourne un DataFrame avec le détail journalier.
    """
    bt = Backtester(
        initial_capital=capital,
        transaction_cost=transaction_cost,
        slippage_bps=slippage_bps,
    )
    bt.add_pair(
        prices,
        ticker_y,
        ticker_x,
        alpha,
        beta,
        capital,
        entry_z,
        exit_z,
        stop_z,
        window,
    )

    results = bt.run()
    return results["pair_results"][(ticker_y, ticker_x)]


# ─────────────────────────────────────────────────────────────────────────────
# Point d'entrée
# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import sys

    sys.path.insert(0, ".")
    from data_loader import download_prices, clean_prices, train_test_split
    from cointegration import find_cointegrated_pairs

    prices = clean_prices(download_prices())
    train, test = train_test_split(prices)
    pairs_df = find_cointegrated_pairs(train)

    if pairs_df.empty:
        print("Aucune paire cointégrée trouvée.")
        sys.exit(0)

    best = pairs_df.iloc[0]
    detail = run_single_pair_backtest(
        train,
        best["ticker_y"],
        best["ticker_x"],
        best["alpha"],
        best["beta"],
    )
    print(detail[["pnl", "cumulative_pnl", "exposure"]].tail(20))
