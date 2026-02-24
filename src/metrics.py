"""
Métriques avancées de performance
=========================================================
Calcule l'ensemble des métriques standard utilisées en quantitative finance :
  - Sharpe ratio annualisé
  - Sortino ratio
  - Max drawdown & durée du drawdown
  - Calmar ratio
  - Turnover
  - Hit ratio
  - Exposition moyenne

Toutes les fonctions acceptent une Series de PnL quotidien et retournent
des scalaires, ce qui facilite l'intégration dans le bootstrap et le
walk-forward.
"""

import numpy as np
import pandas as pd
from typing import Dict, Optional, Tuple


# Nombre de jours de trading par an (convention finance)
TRADING_DAYS_PER_YEAR = 252


# ─────────────────────────────────────────────────────────────────────────────
# Sharpe ratio annualisé
# ─────────────────────────────────────────────────────────────────────────────


def sharpe_ratio(
    pnl: pd.Series,
    risk_free_rate: float = 0.0,
    annualize: bool = True,
) -> float:
    """
    Sharpe ratio = (E[r] - r_f) / σ(r)

    Paramètres
    ----------
    pnl            : PnL journalier (en $ ou en rendements)
    risk_free_rate : taux sans risque ANNUEL (ex: 0.04 pour 4 %)
    annualize      : si True, annualise via √252

    Note : on utilise la std de l'échantillon (ddof=1) par convention.
    """
    if len(pnl) < 2:
        return np.nan

    # Taux sans risque journalier
    rf_daily = risk_free_rate / TRADING_DAYS_PER_YEAR

    excess_returns = pnl - rf_daily
    mean_excess = excess_returns.mean()
    std_excess = excess_returns.std(ddof=1)

    if std_excess == 0:
        return np.nan

    sr = mean_excess / std_excess
    if annualize:
        sr *= np.sqrt(TRADING_DAYS_PER_YEAR)

    return sr


# ─────────────────────────────────────────────────────────────────────────────
# Sortino ratio (pénalise uniquement la volatilité négative)
# ─────────────────────────────────────────────────────────────────────────────


def sortino_ratio(
    pnl: pd.Series,
    risk_free_rate: float = 0.0,
    annualize: bool = True,
) -> float:
    """
    Sortino ratio = (E[r] - r_f) / σ_downside

    σ_downside = écart-type des rendements négatifs uniquement (semi-deviation).

    Plus pertinent que le Sharpe quand la distribution des retours est
    asymétrique (cas typique des stratégies mean-reversion avec stop-loss).
    """
    if len(pnl) < 2:
        return np.nan

    rf_daily = risk_free_rate / TRADING_DAYS_PER_YEAR
    excess_returns = pnl - rf_daily

    # Semi-déviation : on ne pénalise pas les retours positifs
    downside = excess_returns[excess_returns < 0]

    if len(downside) == 0:
        return (
            np.inf
        )  # Aucun jour négatif — stratégie parfaite (ou trop peu de données)

    downside_std = downside.std(ddof=1)

    if downside_std == 0:
        return np.nan

    sortino = excess_returns.mean() / downside_std
    if annualize:
        sortino *= np.sqrt(TRADING_DAYS_PER_YEAR)

    return sortino


# ─────────────────────────────────────────────────────────────────────────────
# Maximum Drawdown
# ─────────────────────────────────────────────────────────────────────────────


def max_drawdown(pnl: pd.Series) -> Tuple[float, int]:
    """
    Calcule le maximum drawdown et sa durée.

    Le drawdown à la date t est défini comme :
        DD_t = (cumPnL_t - max(cumPnL_0..t)) / initial_capital

    Retourne
    --------
    (max_dd, duration_days)
      max_dd       : drawdown maximal (valeur négative, ex: -0.15 = -15 %)
      duration_days: durée du drawdown le plus long (en jours calendaires)

    Note : ici on travaille en $ absolu, non en rendements,
    car le pairs trading est souvent géré en $ nets.
    """
    cumulative = pnl.cumsum()
    running_max = cumulative.cummax()
    drawdown = cumulative - running_max  # Toujours ≤ 0

    mdd = drawdown.min()

    # Durée du pire drawdown (temps depuis le dernier maximum)
    # On cherche la période de récupération la plus longue
    is_in_dd = drawdown < 0
    dd_groups = (is_in_dd != is_in_dd.shift()).cumsum()
    dd_lengths = is_in_dd.groupby(dd_groups).sum()
    max_duration = int(dd_lengths.max()) if is_in_dd.any() else 0

    return float(mdd), max_duration


# ─────────────────────────────────────────────────────────────────────────────
# Calmar ratio
# ─────────────────────────────────────────────────────────────────────────────


def calmar_ratio(
    pnl: pd.Series,
    initial_capital: float = 1_000_000.0,
    annualize: bool = True,
) -> float:
    """
    Calmar ratio = Rendement annuel / |Max Drawdown|

    Mesure le rendement par unité de risque de drawdown.
    Très utilisé par les hedge funds pour comparer les stratégies.
    Un Calmar > 1 est considéré bon.

    Paramètres
    ----------
    initial_capital : capital initial (pour normaliser le rendement annuel)
    """
    mdd, _ = max_drawdown(pnl)

    if mdd == 0:
        return np.inf

    total_pnl = pnl.sum()
    n_days = len(pnl)

    # Rendement annualisé (en fraction du capital initial)
    annual_return = (total_pnl / initial_capital) * (TRADING_DAYS_PER_YEAR / n_days)

    # MDD normalisé (fraction du capital)
    mdd_frac = abs(mdd) / initial_capital

    return annual_return / mdd_frac if mdd_frac > 0 else np.nan


# ─────────────────────────────────────────────────────────────────────────────
# Turnover journalier
# ─────────────────────────────────────────────────────────────────────────────


def turnover(
    pos_y: pd.Series,
    pos_x: pd.Series,
    prices: pd.DataFrame,
    ticker_y: str,
    ticker_x: str,
    capital: float = 1_000_000.0,
) -> float:
    """
    Turnover = variation absolue quotidienne des positions / capital alloué

    Un turnover élevé implique davantage de coûts de transaction.
    En pairs trading, on cible généralement un turnover < 30 % / jour.

    Retourne le turnover moyen journalier en fraction du capital.
    """
    py = prices[ticker_y]
    px = prices[ticker_x]

    # Valeur notionnelle des positions
    notional_y = pos_y * py
    notional_x = pos_x * px

    # Variation quotidienne absolue (trades)
    daily_trades_y = notional_y.diff().abs()
    daily_trades_x = notional_x.diff().abs()
    total_trades = daily_trades_y + daily_trades_x

    return total_trades.mean() / capital


# ─────────────────────────────────────────────────────────────────────────────
# Hit ratio (fraction de jours rentables)
# ─────────────────────────────────────────────────────────────────────────────


def hit_ratio(pnl: pd.Series) -> float:
    """
    Hit ratio = nombre de jours avec PnL > 0 / nombre total de jours.

    Limité aux jours où la stratégie est active (PnL ≠ 0).
    Un hit ratio > 50 % indique une edge positive par barre.
    """
    active_days = pnl[pnl != 0]
    if len(active_days) == 0:
        return np.nan
    return float((active_days > 0).sum() / len(active_days))


# ─────────────────────────────────────────────────────────────────────────────
# Exposition moyenne
# ─────────────────────────────────────────────────────────────────────────────


def average_exposure(
    exposure: pd.Series,
    capital: float = 1_000_000.0,
) -> float:
    """
    Exposition nette moyenne = valeur absolue des positions / capital.

    Indique quelle fraction du capital est effectivement investie.
    0.0 = jamais en position, 1.0 = toujours pleinement investi.
    """
    return float(exposure.mean() / capital)


# ─────────────────────────────────────────────────────────────────────────────
# Résumé complet — toutes les métriques d'un coup
# ─────────────────────────────────────────────────────────────────────────────


def compute_all_metrics(
    pnl: pd.Series,
    exposure: Optional[pd.Series] = None,
    pos_y: Optional[pd.Series] = None,
    pos_x: Optional[pd.Series] = None,
    prices: Optional[pd.DataFrame] = None,
    ticker_y: Optional[str] = None,
    ticker_x: Optional[str] = None,
    initial_capital: float = 1_000_000.0,
    risk_free_rate: float = 0.0,
) -> Dict[str, float]:
    """
    Calcule et retourne toutes les métriques dans un seul dictionnaire.
    Les métriques dépendant des positions (turnover, exposition) ne sont
    calculées que si les données correspondantes sont fournies.
    """
    mdd, mdd_duration = max_drawdown(pnl)

    metrics = {
        "sharpe": sharpe_ratio(pnl, risk_free_rate),
        "sortino": sortino_ratio(pnl, risk_free_rate),
        "max_drawdown_$": mdd,
        "max_dd_duration": mdd_duration,
        "calmar": calmar_ratio(pnl, initial_capital),
        "hit_ratio": hit_ratio(pnl),
        "total_pnl_$": float(pnl.sum()),
        "avg_daily_pnl_$": float(pnl.mean()),
        "pnl_std_$": float(pnl.std(ddof=1)),
        "n_days": len(pnl),
    }

    if exposure is not None:
        metrics["avg_exposure"] = average_exposure(exposure, initial_capital)

    if all(v is not None for v in [pos_y, pos_x, prices, ticker_y, ticker_x]):
        metrics["turnover"] = turnover(
            pos_y, pos_x, prices, ticker_y, ticker_x, initial_capital
        )

    return metrics


def print_metrics(metrics: Dict[str, float]) -> None:
    """Affiche les métriques de façon lisible."""
    print("\n" + "=" * 50)
    print("  PERFORMANCE METRICS")
    print("=" * 50)
    labels = {
        "sharpe": "Sharpe ratio (annuel)",
        "sortino": "Sortino ratio (annuel)",
        "max_drawdown_$": "Max drawdown ($)",
        "max_dd_duration": "Durée max drawdown (jours)",
        "calmar": "Calmar ratio",
        "hit_ratio": "Hit ratio",
        "total_pnl_$": "PnL total ($)",
        "avg_daily_pnl_$": "PnL moyen quotidien ($)",
        "pnl_std_$": "Volatilité journalière ($)",
        "avg_exposure": "Exposition moyenne",
        "turnover": "Turnover moyen journalier",
        "n_days": "Nombre de jours",
    }
    for key, label in labels.items():
        if key in metrics:
            val = metrics[key]
            if isinstance(val, float):
                print(f"  {label:<35} {val:>10.4f}")
            else:
                print(f"  {label:<35} {val:>10}")
    print("=" * 50 + "\n")


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

    metrics = compute_all_metrics(
        pnl=detail["pnl"],
        exposure=detail["exposure"],
        pos_y=detail["pos_y"],
        pos_x=detail["pos_x"],
        prices=train,
        ticker_y=best["ticker_y"],
        ticker_x=best["ticker_x"],
    )
    print_metrics(metrics)
