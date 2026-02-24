"""
Collecte et préparation des données
==============================================================
Télécharge les prix ajustés de N actions liquides du S&P500 via yfinance,
aligne les dates, gère les valeurs manquantes, et effectue le split
train/test (fenêtre roulante : 5 ans train / 1 an test).
"""

import yfinance as yf
import pandas as pd
import numpy as np
from typing import Tuple, List, Optional
from datetime import datetime, timedelta


# ------------------------------------------------------------
# Liste de tickers liquides du S&P500 (secteurs diversifiés)
# ------------------------------------------------------------
DEFAULT_TICKERS = [
    "XOM",
    "CVX",  # Énergie — souvent cointégrées
    "JPM",
    "BAC",  # Finance
    "KO",
    "PEP",  # Consommation courante
    "GLD",
    "SLV",  # Métaux précieux (ETFs)
    "SPY",
    "QQQ",  # Indices ETFs
    "AAPL",
    "MSFT",  # Tech
    "WMT",
    "TGT",  # Distribution
    "HD",
    "LOW",  # Bricolage
]


def download_prices(
    tickers: List[str] = DEFAULT_TICKERS,
    start: str = "2015-01-01",
    end: str = "2024-12-31",
) -> pd.DataFrame:
    """
    Télécharge les cours de clôture ajustés pour une liste de tickers.

    Paramètres
    ----------
    tickers : liste de symboles boursiers
    start   : date de début au format 'YYYY-MM-DD'
    end     : date de fin   au format 'YYYY-MM-DD'

    Retourne
    --------
    DataFrame (dates × tickers) avec les prix ajustés.
    """
    print(
        f"[data_loader] Téléchargement de {len(tickers)} tickers de {start} à {end} …"
    )

    # yfinance retourne un DataFrame MultiIndex ; on extrait 'Close'
    raw = yf.download(tickers, start=start, end=end, auto_adjust=True)

    # Garder uniquement la colonne 'Close'
    # Si plusieurs tickers, DataFrame indexé par (field, ticker), par exemple ("Close", "AAPL")
    if isinstance(raw.columns, pd.MultiIndex):
        prices = raw["Close"]
    else:
        prices = raw  # Un seul ticker : déjà un simple DataFrame

    prices.index = pd.to_datetime(prices.index)
    print(
        f"[data_loader] {prices.shape[0]} jours × {prices.shape[1]} tickers téléchargés."
    )
    return prices


def clean_prices(prices: pd.DataFrame, max_na_pct: float = 0.02) -> pd.DataFrame:
    """
    Nettoie le DataFrame de prix :
      1. Aligne toutes les séries sur un calendrier commun (intersection des dates).
      2. Supprime les colonnes dont le taux de NA dépasse max_na_pct.
      3. Forward-fill les NA résiduels (pas de trading le jour J → prix J-1).
      4. Supprime les éventuelles lignes encore manquantes.

    Paramètres
    ----------
    max_na_pct : seuil maximum de valeurs manquantes par ticker [0,1].
    """
    # Supprimer les tickers trop incomplets
    na_ratio = prices.isna().mean()
    valid = na_ratio[na_ratio <= max_na_pct].index.tolist()
    dropped = set(prices.columns) - set(valid)
    if dropped:
        print(f"[data_loader] Tickers retirés (trop de NA) : {dropped}")
    prices = prices[valid]

    # Forward-fill (au maximum 5 jours consécutifs pour éviter les abus)
    # Si non NaN a t, suivi de k NaN (k \leq 5), propage la valeur à t
    prices = prices.ffill(limit=5)

    # Supprimer les lignes qui contiennent encore des NA
    before = len(prices)
    prices = prices.dropna()
    after = len(prices)
    if before != after:
        print(f"[data_loader] {before - after} lignes supprimées après cleaning.")

    print(
        f"[data_loader] Données propres : {prices.shape[0]} jours × {prices.shape[1]} tickers."
    )
    return prices


def train_test_split(
    prices: pd.DataFrame,
    train_years: int = 5,
    test_years: int = 1,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Split simple sur les dernières années disponibles.
    Retourne (train, test).

    Pour l'analyse walk-forward (split roulant), voir walk_forward_splits().
    """
    end_date = prices.index[-1]
    test_start = end_date - pd.DateOffset(years=test_years)
    train_start = test_start - pd.DateOffset(years=train_years)

    train = prices.loc[(prices.index >= train_start) & (prices.index < test_start)]
    test = prices.loc[prices.index >= test_start]

    print(
        f"[data_loader] Train : {train.index[0].date()} → {train.index[-1].date()} "
        f"({len(train)} jours)"
    )
    print(
        f"[data_loader] Test  : {test.index[0].date()}  → {test.index[-1].date()} "
        f"({len(test)} jours)"
    )
    return train, test


def walk_forward_splits(
    prices: pd.DataFrame,
    train_years: int = 5,
    test_years: int = 1,
) -> List[Tuple[pd.DataFrame, pd.DataFrame]]:
    """
    Génère des fenêtres roulantes (train, test) pour la walk-forward analysis.

    Exemple avec train=5 ans, test=1 an :
      Fenêtre 1 : train [2015–2020], test [2020–2021]
      Fenêtre 2 : train [2016–2021], test [2021–2022]
      …

    Retourne une liste de tuples (train_df, test_df).
    """
    splits = []
    first_date = prices.index[0]
    last_date = prices.index[-1]

    # La première fenêtre de test démarre après train_years
    test_start = first_date + pd.DateOffset(years=train_years)

    while test_start + pd.DateOffset(years=test_years) <= last_date:
        train_start = test_start - pd.DateOffset(years=train_years)
        test_end = test_start + pd.DateOffset(years=test_years)

        train = prices.loc[(prices.index >= train_start) & (prices.index < test_start)]
        test = prices.loc[(prices.index >= test_start) & (prices.index < test_end)]

        if len(train) > 100 and len(test) > 20:  # Fenêtres assez grandes
            splits.append((train, test))

        test_start += pd.DateOffset(years=1)  # Décalage d'un an

    print(f"[data_loader] {len(splits)} fenêtres walk-forward générées.")
    return splits


# -------------------------------------------------------------------
# Point d'entrée
# -------------------------------------------------------------------
if __name__ == "__main__":
    prices = download_prices()
    prices = clean_prices(prices)
    train, test = train_test_split(prices)
    prices.to_csv("../data/prices.csv")
    print("[data_loader] Données sauvegardées dans data/prices.csv")
