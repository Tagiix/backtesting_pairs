"""
Monte Carlo du PnL via processus OU
================================================================
Simule la distribution future du PnL en modélisant le spread comme un
processus d'Ornstein-Uhlenbeck (OU) calibré sur les données historiques.

Modèle OU en temps discret (Euler-Maruyama) :
    S_{t+1} = S_t + κ(μ - S_t)Δt + σ√Δt · ε_t,   ε_t ~ N(0,1)

Paramètres estimés par OLS sur ΔS_t = a + b·S_{t-1} + η_t :
    κ  = -b         (vitesse de mean-reversion)
    μ  = -a/b       (niveau d'équilibre)
    σ  = std(η_t)   (volatilité du spread)

Le PnL est simulé en appliquant la stratégie de signal (avec coûts) sur
chaque trajectoire Monte Carlo.
"""

import numpy as np
import pandas as pd
from typing import Optional, Tuple, Dict
from statsmodels.regression.linear_model import OLS
from statsmodels.tools import add_constant


# ─────────────────────────────────────────────────────────────────────────────
# Estimation des paramètres OU
# ─────────────────────────────────────────────────────────────────────────────


def estimate_ou_parameters(spread: pd.Series) -> Dict[str, float]:
    """
    Estime les paramètres du processus OU à partir d'un spread observé
    via régression OLS sur la forme discrète :

        ΔS_t = a + b·S_{t-1} + η_t

    D'où :
        κ  = -b            (doit être > 0 pour mean-reversion)
        μ  = -a / b        (spread moyen à long terme)
        σ  = std(résidus)  (volatilité instantanée)
        hl = ln(2) / κ     (half-life en jours)

    Retourne un dict avec les 4 paramètres OU.
    """
    spread_lag = spread.shift(1).dropna()
    spread_diff = spread.diff().dropna()

    # Alignement (dropna peut créer un désalignement d'1 observation)
    idx = spread_lag.index.intersection(spread_diff.index)
    y = spread_diff.loc[idx]
    X = add_constant(spread_lag.loc[idx])

    model = OLS(y, X).fit()
    a = float(model.params.iloc[0])  # constante
    b = float(model.params.iloc[1])  # coefficient autorégressif

    kappa = -b  # Vitesse de mean-reversion
    mu_ou = -a / b if b != 0 else spread.mean()  # Niveau d'équilibre
    sigma = float(model.resid.std())

    halflife = np.log(2) / kappa if kappa > 0 else np.inf

    return {
        "kappa": kappa,
        "mu": mu_ou,
        "sigma": sigma,
        "halflife": halflife,
    }


# ─────────────────────────────────────────────────────────────────────────────
# Simulation de trajectoires OU (Euler-Maruyama)
# ─────────────────────────────────────────────────────────────────────────────


def simulate_ou(
    kappa: float,
    mu: float,
    sigma: float,
    s0: float,
    n_steps: int,
    n_paths: int = 1000,
    dt: float = 1.0,
    seed: int = 42,
) -> np.ndarray:
    """
    Simule n_paths trajectoires du processus OU sur n_steps pas de temps.

    Schéma d'Euler-Maruyama (exact pour OU) :
        S_{t+1} = S_t + κ(μ - S_t)·dt + σ·√dt·ε_t

    Paramètres
    ----------
    kappa   : vitesse de mean-reversion (> 0)
    mu      : niveau d'équilibre
    sigma   : volatilité du spread
    s0      : valeur initiale du spread
    n_steps : nombre de jours de simulation
    n_paths : nombre de trajectoires Monte Carlo
    dt      : pas de temps (1.0 = quotidien)

    Retourne
    --------
    Array de shape (n_paths, n_steps + 1) — colonne 0 = s0.
    """
    rng = np.random.default_rng(seed)
    paths = np.zeros((n_paths, n_steps + 1))
    paths[:, 0] = s0

    # Générer tous les chocs aléatoires d'un coup (vectorisé)
    eps = rng.standard_normal((n_paths, n_steps))

    for t in range(n_steps):
        s_t = paths[:, t]
        # Terme de mean-reversion + terme stochastique
        paths[:, t + 1] = (
            s_t + kappa * (mu - s_t) * dt + sigma * np.sqrt(dt) * eps[:, t]
        )

    return paths


# ─────────────────────────────────────────────────────────────────────────────
# PnL Monte Carlo : appliquer la stratégie sur les trajectoires simulées
# ─────────────────────────────────────────────────────────────────────────────


def monte_carlo_pnl(
    spread_paths: np.ndarray,
    spread_std: float,
    spread_mean: float,
    entry_z: float = 2.0,
    exit_z: float = 0.5,
    stop_z: Optional[float] = 3.5,
    transaction_cost_per_trade: float = 0.0,
) -> np.ndarray:
    """
    Applique la stratégie de trading sur chaque trajectoire de spread simulée
    et retourne le PnL cumulé final de chaque trajectoire.

    Simplification : on suppose une exposition de 1 unité de spread (normalisée).
    Le PnL est exprimé en unités de spread (pas en dollars — la conversion
    dollar dépend du capital alloué et des prix, gérés dans le backtester).

    Paramètres
    ----------
    spread_paths              : (n_paths, n_steps+1) — trajectoires OU simulées
    spread_std                : écart-type du spread (pour normaliser en Z-score)
    spread_mean               : moyenne du spread (pour calculer le Z-score)
    transaction_cost_per_trade: coût par trade en unités de spread.
                                Heuristique raisonnable : ~0.1 * spread_std.

    Retourne
    --------
    Array (n_paths,) de PnL cumulé final (en unités de spread) pour chaque trajectoire.
    """
    n_paths, n_total = spread_paths.shape
    n_steps = n_total - 1

    final_pnl = np.zeros(n_paths)

    for path_idx in range(n_paths):
        s = spread_paths[path_idx]  # Trajectoire de spread (n_steps+1,)
        pnl = 0.0
        position = 0  # {-1, 0, +1}
        stopped_out = False  # Cooldown après stop-loss

        for t in range(1, n_steps + 1):
            z = (s[t] - spread_mean) / spread_std  # Z-score courant

            # Fin du cooldown post-stop : on attend que le Z revienne
            # dans la zone neutre (|Z| < 1) avant d'autoriser un nouvel entrée.
            # Cela évite la boucle stop-and-reverse qui draine le PnL.
            if stopped_out and abs(z) < 1.0:
                stopped_out = False

            # --- Logique de signal ---
            if position == 0 and not stopped_out:
                if z < -entry_z:
                    position = 1  # Long spread
                    pnl -= transaction_cost_per_trade
                elif z > entry_z:
                    position = -1  # Short spread
                    pnl -= transaction_cost_per_trade

            elif position == 1:
                # Long : profiter si le spread remonte vers mu
                pnl += s[t] - s[t - 1]  # Mark-to-market journalier
                if abs(z) < exit_z:
                    # Sortie normale sur mean-reversion
                    position = 0
                    pnl -= transaction_cost_per_trade
                elif stop_z and z < -stop_z:
                    # Stop-loss : le spread diverge trop ; couper la position
                    position = 0
                    stopped_out = True
                    pnl -= transaction_cost_per_trade

            elif position == -1:
                # Short : profiter si le spread redescend vers mu
                pnl -= s[t] - s[t - 1]  # Sens inverse
                if abs(z) < exit_z:
                    position = 0
                    pnl -= transaction_cost_per_trade
                elif stop_z and z > stop_z:
                    position = 0
                    stopped_out = True
                    pnl -= transaction_cost_per_trade

        final_pnl[path_idx] = pnl

    return final_pnl


# ─────────────────────────────────────────────────────────────────────────────
# Analyse complète : distribution future du PnL
# ─────────────────────────────────────────────────────────────────────────────


def run_monte_carlo(
    spread: pd.Series,
    n_paths: int = 2000,
    horizon_days: int = 252,
    entry_z: float = 2.0,
    exit_z: float = 0.5,
    stop_z: Optional[float] = 3.5,
    transaction_cost_per_trade: float = 0.0,
    seed: int = 42,
) -> Dict:
    """
    Pipeline complet Monte Carlo :
      1. Estimer les paramètres OU du spread
      2. Simuler n_paths trajectoires sur horizon_days
      3. Appliquer la stratégie sur chaque trajectoire
      4. Analyser la distribution du PnL final

    Retourne un dict avec :
      - 'ou_params'         : paramètres OU estimés
      - 'spread_paths'      : (n_paths, horizon+1) trajectoires simulées
      - 'pnl_distribution'  : (n_paths,) PnL final de chaque trajectoire
      - 'pnl_mean'          : PnL moyen espéré
      - 'pnl_std'           : volatilité du PnL
      - 'pct_positive'      : fraction de trajectoires rentables
      - 'prob_large_dd'     : probabilité de drawdown > seuil
      - 'var_5pct'          : VaR à 5 %
      - 'cvar_5pct'         : CVaR (Expected Shortfall) à 5 %
    """
    print(f"[monte_carlo] Estimation des paramètres OU …")
    ou_params = estimate_ou_parameters(spread)
    print(
        f"[monte_carlo] OU params : κ={ou_params['kappa']:.4f}, "
        f"μ={ou_params['mu']:.4f}, σ={ou_params['sigma']:.4f}, "
        f"hl={ou_params['halflife']:.1f} jours"
    )

    if ou_params["kappa"] <= 0:
        print(
            "[monte_carlo] ⚠ κ ≤ 0 : le spread n'est pas mean-reverting. "
            "Les simulations seront instables."
        )

    # Valeur initiale = dernière valeur observée du spread
    s0 = float(spread.iloc[-1])

    print(
        f"[monte_carlo] Simulation de {n_paths} trajectoires sur {horizon_days} jours …"
    )
    spread_paths = simulate_ou(
        kappa=ou_params["kappa"],
        mu=ou_params["mu"],
        sigma=ou_params["sigma"],
        s0=s0,
        n_steps=horizon_days,
        n_paths=n_paths,
        seed=seed,
    )

    # Appliquer la stratégie sur les trajectoires simulées
    spread_mean = float(spread.mean())
    spread_std = float(spread.std())

    # Coût par trade calibré sur la volatilité stationnaire du spread :
    # une mean-reversion de ±2σ rapporte 4σ brut ; un coût de 0.1σ par jambe
    # (soit 0.2σ aller-retour) est un proxy raisonnable pour 10–15 bps de friction.
    sigma_stationary = ou_params["sigma"] / np.sqrt(2 * max(ou_params["kappa"], 1e-6))
    effective_cost = (
        transaction_cost_per_trade
        if transaction_cost_per_trade > 0
        else 0.1 * sigma_stationary
    )

    pnl_dist = monte_carlo_pnl(
        spread_paths,
        spread_std=spread_std,
        spread_mean=spread_mean,
        entry_z=entry_z,
        exit_z=exit_z,
        stop_z=stop_z,
        transaction_cost_per_trade=effective_cost,
    )

    # Métriques de la distribution du PnL
    pnl_mean = float(np.mean(pnl_dist))
    pnl_std = float(np.std(pnl_dist))
    pct_pos = float((pnl_dist > 0).mean())

    # VaR et CVaR à 5 %
    var_5pct = float(np.percentile(pnl_dist, 5))
    cvar_5pct = float(pnl_dist[pnl_dist <= var_5pct].mean())

    # Probabilité de drawdown > 20 % du capital (seuil illustratif)
    # Ici simplification : on utilise le PnL final < -20 % comme proxy
    dd_threshold = -0.2 * abs(pnl_mean) if pnl_mean != 0 else -1.0
    prob_large_dd = float((pnl_dist < dd_threshold).mean())

    print(f"[monte_carlo] PnL moyen : {pnl_mean:.2f}")
    print(f"[monte_carlo] Frac. trajectoires rentables : {pct_pos:.1%}")
    print(f"[monte_carlo] VaR 5% : {var_5pct:.2f}   CVaR 5% : {cvar_5pct:.2f}")

    return {
        "ou_params": ou_params,
        "spread_paths": spread_paths,
        "pnl_distribution": pnl_dist,
        "pnl_mean": pnl_mean,
        "pnl_std": pnl_std,
        "pct_positive": pct_pos,
        "prob_large_dd": prob_large_dd,
        "var_5pct": var_5pct,
        "cvar_5pct": cvar_5pct,
        "horizon_days": horizon_days,
        "n_paths": n_paths,
    }


# ─────────────────────────────────────────────────────────────────────────────
# Point d'entrée
# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import sys

    sys.path.insert(0, ".")
    from data_loader import download_prices, clean_prices, train_test_split
    from cointegration import find_cointegrated_pairs, compute_spread

    prices = clean_prices(download_prices())
    train, test = train_test_split(prices)
    pairs_df = find_cointegrated_pairs(train)

    if pairs_df.empty:
        print("Aucune paire trouvée.")
        sys.exit(0)

    best = pairs_df.iloc[0]
    spread_df = compute_spread(
        train, best["ticker_y"], best["ticker_x"], best["alpha"], best["beta"]
    )

    mc_results = run_monte_carlo(
        spread=spread_df["spread"],
        n_paths=2000,
        horizon_days=252,
    )

    print(f"\nRésultats Monte Carlo :")
    print(f"  PnL espéré       : {mc_results['pnl_mean']:.2f}")
    print(f"  Frac. positives  : {mc_results['pct_positive']:.1%}")
    print(f"  VaR 5%           : {mc_results['var_5pct']:.2f}")
    print(f"  CVaR 5%          : {mc_results['cvar_5pct']:.2f}")
