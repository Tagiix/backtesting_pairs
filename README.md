# Pairs Trading Backtester

A quantitative pairs trading engine with rigorous statistical validation — cointegration testing, bootstrap inference, walk-forward analysis, and Monte Carlo simulation.

---

## Features

- **Pair selection** — Engle-Granger & Johansen cointegration tests, half-life filtering (5–120 days)
- **Signal generation** — Z-score ±2σ entry, dollar-neutral sizing, stop-loss at 3.5σ
- **Clean backtest** — Zero look-ahead bias (`shift(1)` throughout), realistic costs (10 bps broker + 2 bps slippage)
- **Performance metrics** — Sharpe, Sortino, Max Drawdown, Calmar, hit ratio
- **Bootstrap inference** — Stationary Block Bootstrap (Politis & Romano 1994) + Lo (2002) autocorrelation correction
- **Walk-forward analysis** — Rolling OOS windows (5y train / 1y test) to detect overfitting
- **Monte Carlo** — Ornstein-Uhlenbeck simulation (Euler-Maruyama), VaR & CVaR at 5%

## Installation

```bash
git clone https://github.com/your-username/pairs-trading-backtester
cd pairs-trading-backtester
pip install -r requirements.txt
```

## Usage

```bash
# Quick run (no walk-forward)
python main.py --no-walk-forward

# Full pipeline
python main.py

# Custom tickers and date range
python main.py --tickers XOM CVX KO PEP --start 2018-01-01 --end 2024-12-31
```

| Argument | Description | Default |
|---|---|---|
| `--tickers` | List of S&P 500 tickers | 12 predefined |
| `--start` / `--end` | Date range | 2015-01-01 / 2024-12-31 |
| `--no-walk-forward` | Skip walk-forward analysis | — |
| `--no-monte-carlo` | Skip Monte Carlo simulation | — |
| `--no-bootstrap` | Skip Sharpe bootstrap | — |

## Project Structure

```
├── main.py                 # CLI orchestrator — 10-step pipeline
├── src/
│   ├── data_loader.py      # yfinance download, cleaning, train/test split
│   ├── cointegration.py    # Engle-Granger, Johansen, Z-score spread
│   ├── signals.py          # Entry/exit signals, dollar-neutral sizing
│   ├── backtester.py       # MTM PnL engine
│   ├── metrics.py          # Sharpe, Sortino, MDD, Calmar, hit ratio
│   ├── bootstrap.py        # Stationary block bootstrap, Lo (2002)
│   ├── walk_forward.py     # Rolling OOS windows, rolling beta
│   └── monte_carlo.py      # OU simulation, VaR, CVaR
├── notebooks/analysis.ipynb
└── results/                # CSV exports and PNG charts
```

## Example Results (GLD/WMT, 2019–2024)

| | In-sample | Out-of-sample |
|---|---|---|
| Sharpe | 1.28 | -0.09 (p=0.54, not significant) |
| Total PnL | +$131,321 | — |
| Max Drawdown | -$34,743 | — |
| OU half-life | 27.7 days | — |
| MC positive paths | 62.6% | — |

## Dependencies

`yfinance`, `pandas`, `numpy`, `statsmodels`, `scipy`, `matplotlib`, `seaborn`, `jupyter`

## References

- Engle & Granger (1987) — Cointegration and Error Correction
- Johansen (1991) — Cointegration Vectors in VAR Models
- Gatev, Goetzmann & Rouwenhorst (2006) — Pairs Trading Performance
- Lo (2002) — The Statistics of Sharpe Ratios
- Politis & Romano (1994) — The Stationary Bootstrap
- Artzner et al. (1999) — Coherent Measures of Risk
- Roncalli, T. — [Risk Management](http://www.thierry-roncalli.com/RiskManagementBook.html)
