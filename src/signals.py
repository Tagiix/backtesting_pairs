"""
Construction du signal de trading
========================================================
Génère les signaux d'entrée/sortie pour chaque paire cointégrée à partir
du Z-score du spread, avec position sizing neutre en dollar et coûts de
transaction réalistes.

Logique de signal :
  - Long spread  (acheter Y, vendre X) si Z_t < -entry_z
  - Short spread (vendre Y, acheter X) si Z_t > +entry_z
  - Sortie de position si |Z_t| < exit_z
  - Stop-loss si |Z_t| > stop_z (optionnel)
"""

import numpy as np
import pandas as pd
from typing import Optional

from cointegration import compute_spread


# ─────────────────────────────────────────────────────────────────────────────
# Génération du signal discret
# ─────────────────────────────────────────────────────────────────────────────


def generate_signal(
    zscore: pd.Series,
    entry_z: float = 2.0,
    exit_z: float = 0.5,
    stop_z: Optional[float] = 3.5,
) -> pd.Series:
    """
    Convertit le Z-score en signal de position {-1, 0, +1}.

    Conventions :
      +1  = long spread  (acheter Y, vendre X)
      -1  = short spread (vendre Y, acheter X)
       0  = pas de position

    La logique est stateful : on garde une position jusqu'au signal de sortie
    (pas de stop sur chaque barre — la sortie se fait sur le Z-score de clôture).

    ⚠ Pas de look-ahead : le signal du jour J est basé sur le Z-score de J-1
      (décalage appliqué dans le backtester, pas ici).

    Paramètres
    ----------
    entry_z : seuil d'entrée en écarts-types
    exit_z  : seuil de sortie (mean-reversion)
    stop_z  : seuil de stop-loss (couper si le spread diverge trop)
    """
    signal = pd.Series(0, index=zscore.index, dtype=float)
    position = 0  # Position courante

    for i, z in enumerate(zscore):
        if np.isnan(z):
            signal.iloc[i] = 0
            continue

        if position == 0:
            # Pas en position : chercher un signal d'entrée
            if z < -entry_z:
                position = 1  # Long spread
            elif z > entry_z:
                position = -1  # Short spread

        elif position == 1:
            # Long spread : sortir si Z remonte vers 0
            if abs(z) < exit_z:
                position = 0
            elif stop_z is not None and z < -stop_z:
                # Stop-loss : le spread continue de diverger fortement
                position = 0

        elif position == -1:
            # Short spread : sortir si Z redescend vers 0
            if abs(z) < exit_z:
                position = 0
            elif stop_z is not None and z > stop_z:
                # Stop-loss
                position = 0

        signal.iloc[i] = position

    return signal


# ─────────────────────────────────────────────────────────────────────────────
# Position sizing neutre en dollar (dollar-neutral)
# ─────────────────────────────────────────────────────────────────────────────


def compute_positions(
    prices: pd.DataFrame,
    ticker_y: str,
    ticker_x: str,
    beta: float,
    signal: pd.Series,
    capital: float = 1_000_000.0,
    transaction_cost: float = 0.001,
    slippage_bps: float = 2.0,
) -> pd.DataFrame:
    """
    Calcule les positions en nombre de titres pour chaque jambe de la paire,
    de façon à être neutre en dollar (dollar-neutral).

    Construction :
      - On alloue `capital / 2` à chaque jambe.
      - Jambe Y : n_y = (capital/2) / P_Y
      - Jambe X : n_x = β × n_y  (pour neutraliser le beta)
      - Long spread  : acheter n_y titres Y, vendre n_x titres X
      - Short spread : vendre n_y titres Y, acheter n_x titres X

    Les coûts de transaction s'appliquent à chaque changement de position.
    Le slippage est modélisé comme un écart de prix (en bps) sur l'exécution.

    Paramètres
    ----------
    capital          : capital alloué à la paire (en $)
    transaction_cost : coût broker par trade (fraction du notionnel, ex: 0.001 = 10 bps)
    slippage_bps     : slippage en basis points (impact de marché simplifié)

    Retourne un DataFrame avec colonnes :
      signal, pos_y, pos_x, cost_y, cost_x, net_cost
    """
    py = prices[ticker_y]
    px = prices[ticker_x]

    # Nombre de titres ciblé (basé sur la session précédente — pas de look-ahead)
    n_y = (capital / 2) / py.shift(1)  # Jambe Y
    n_x = beta * n_y  # Jambe X (beta-adjusté)

    # Position brute en titres (signée)
    pos_y = signal * n_y  # +n_y si long spread, -n_y si short spread
    pos_x = -signal * n_x  # Sens inverse pour X

    # Coûts de transaction : calculés uniquement lors des changements de position
    delta_signal = signal.diff().fillna(0)
    is_trade = (delta_signal != 0).astype(float)

    slippage_frac = slippage_bps / 10_000

    cost_y = is_trade * n_y.abs() * py * (transaction_cost + slippage_frac)
    cost_x = is_trade * n_x.abs() * px * (transaction_cost + slippage_frac)
    net_cost = cost_y + cost_x  # Coût total de la transaction (toujours positif)

    return pd.DataFrame(
        {
            "signal": signal,
            "pos_y": pos_y,
            "pos_x": pos_x,
            "cost_y": cost_y,
            "cost_x": cost_x,
            "net_cost": net_cost,
        },
        index=prices.index,
    )


# ─────────────────────────────────────────────────────────────────────────────
# Pipeline complet pour une paire
# ─────────────────────────────────────────────────────────────────────────────


def build_pair_signals(
    prices: pd.DataFrame,
    ticker_y: str,
    ticker_x: str,
    alpha: float,
    beta: float,
    entry_z: float = 2.0,
    exit_z: float = 0.5,
    stop_z: Optional[float] = 3.5,
    window: Optional[int] = None,
    capital: float = 1_000_000.0,
    transaction_cost: float = 0.001,
    slippage_bps: float = 2.0,
) -> pd.DataFrame:
    """
    Pipeline complet : spread → Z-score → signal → positions.

    Retourne un DataFrame enrichi avec toutes les colonnes nécessaires
    pour le backtester.
    """
    # Étape 3 : spread et Z-score
    spread_df = compute_spread(prices, ticker_y, ticker_x, alpha, beta, window)

    # Étape 4 : signal discret
    signal = generate_signal(spread_df["zscore"], entry_z, exit_z, stop_z)

    # ⚠ Décalage du signal d'un jour pour éviter le look-ahead bias :
    # le signal du soir J déclenche une exécution à l'ouverture de J+1.
    signal_shifted = signal.shift(1).fillna(0)

    # Positions et coûts
    pos_df = compute_positions(
        prices,
        ticker_y,
        ticker_x,
        beta,
        signal_shifted,
        capital,
        transaction_cost,
        slippage_bps,
    )

    # Concaténer tout dans un seul DataFrame
    result = pd.concat([spread_df, pos_df], axis=1)
    result["ticker_y"] = ticker_y
    result["ticker_x"] = ticker_x
    return result


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
    else:
        best = pairs_df.iloc[0]
        signals = build_pair_signals(
            train,
            best["ticker_y"],
            best["ticker_x"],
            best["alpha"],
            best["beta"],
        )
        print(signals[["spread", "zscore", "signal", "pos_y", "pos_x"]].tail(20))
