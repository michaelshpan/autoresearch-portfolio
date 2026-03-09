# Portfolio Replication — Autonomous Research Agent

## Mission

You are an autonomous research agent. Your job is to iteratively improve an LSTM-based model that predicts quarterly trade directions (buy / sell / hold) for four US large-cap growth equity portfolios. Each experiment runs for a fixed **10-minute wall-clock budget**. You modify `train.py`, run it, check if the result improved, keep or discard, and repeat.

---

## Problem Definition

**Input data** (loaded by `prepare.py`, do not modify):

| File | Description |
|------|-------------|
| `data/holdings_quarterly_fin.csv` | Quarterly holdings for Mgr_A, Mgr_B, Mgr_C, Mgr_LS (Q3 2019 – Q4 2025). Columns: fund_ticker, report_date, quarter, security_name, ticker, sector, cusip, shares, pct_weight, dollar_value |
| `data/portfolio_daily_nav_fin.csv` | Daily NAV series for the four managers (2016-01 – 2026-01) |
| `data/benchmark_daily_prices.csv` | Daily prices for R3000 and R1000G benchmarks |
| `data/holdings_vong.csv` | Russell 1000 Growth (VONG proxy) quarterly holdings — the benchmark universe |
| `data/F-F_Research_Data_5_Factors_2x3_daily.csv` | Fama-French 5 factors (daily) |
| `data/F-F_Momentum_Factor_daily.csv` | Momentum factor (daily) |
| `data/Mgr_*_Qualitative_fin.md` | Qualitative analyst assessments per manager |
| `data/Mgr_*_Quantitative.md` | Quantitative characteristics per manager (sector weights, style, valuation, growth measures) |
| `data/manager_evaluation_rubric_hanover_v2.md` | Scoring rubric for manager evaluation |
| `data/ODD_*.docx` | Operational due diligence reports |

**Target**: Predict Q4 2025 holdings (ticker + weight) for each of the four managers.

**Train/val split**: Train on Q3 2019 – Q2 2025. Validate on Q3 2025. Test (held-out ground truth) on Q4 2025.

**Primary metric**: `trade_direction_accuracy` — the percentage of securities where the model correctly predicts the trade direction (buy / sell / hold) between Q3 2025 and Q4 2025, averaged across all four managers. Higher is better.

**Secondary metric**: `weight_mae` — mean absolute error of predicted portfolio weights vs. actual Q4 2025 weights. Lower is better.

---

## Trade Direction Labels

For each (fund, security) pair between consecutive quarters:

- **BUY**: security is new to the portfolio, OR weight increased by > 0.5 percentage points
- **SELL**: security dropped from the portfolio, OR weight decreased by > 0.5 percentage points
- **HOLD**: weight change is within ±0.5 percentage points

The 0.5pp dead zone avoids noisy labels from passive drift. This threshold is a hyperparameter you may adjust.

---

## Project Structure

```
CLAUDE.md              ← you are here (do not modify)
prepare.py             ← data loading, feature engineering, evaluation (do not modify)
train.py               ← model, optimizer, training loop (YOU MODIFY THIS)
data/                  ← all input files (do not modify)
runs/                  ← experiment logs (auto-created)
best_model.pt          ← best checkpoint so far
```

---

## Experiment Protocol

Each experiment follows this exact sequence:

1. **Read** `runs/log.jsonl` to see all past experiments, their changes, and their scores.
2. **Hypothesize**: form a clear, specific hypothesis about what change will improve the metric. Write it down.
3. **Edit** `train.py` only. Make a single, targeted change that tests your hypothesis.
4. **Run**: `python train.py` (must complete within the 10-minute wall-clock budget).
5. **Evaluate**: the script prints `trade_direction_accuracy` and `weight_mae`. It also appends to `runs/log.jsonl`.
6. **Decision**:
   - If `trade_direction_accuracy` improved → commit (the script auto-saves `best_model.pt`).
   - If not → revert `train.py` to the previous best version before making your next edit.
7. **Repeat** from step 1.

---

## What You Can Change in train.py

Everything inside `train.py` is fair game:

- **Architecture**: LSTM layers, hidden dims, attention heads, bidirectionality, residual connections, dropout
- **Feature engineering**: how raw features are combined, normalized, windowed, or embedded
- **Classification approach**: trade direction thresholds, class weighting, loss functions (cross-entropy, focal loss, etc.)
- **Position sizing layer**: how predicted directions are converted to portfolio weights
- **Optimizer**: learning rate, scheduler, weight decay, gradient clipping
- **Training strategy**: number of epochs, batch size, early stopping, curriculum learning
- **Sequence modeling**: rolling window length, how quarterly snapshots are sequenced
- **Regularization**: dropout, batch norm, weight decay, data augmentation
- **Ensembling**: multiple models, bagging across funds or time periods

---

## What You Must NOT Change

- `prepare.py` — all data loading, feature construction, and evaluation logic is fixed
- `CLAUDE.md` — this file
- Anything in `data/` — raw inputs are immutable
- The evaluation metric definitions
- The 10-minute time budget

---

## Key Constraints & Tips

1. **Sparsity**: Each manager holds ~70–180 securities out of ~500+ in the R1000G universe. ~97% of the fund-security matrix is zero. Design for this.
2. **Small sample size**: ~24 quarters of history per fund, 4 funds. Avoid overfitting. Simple models may beat complex ones.
3. **Feasibility mask**: Only predict over securities that appear in the R1000G benchmark universe (VONG holdings). This dramatically reduces the output space.
4. **Classification first**: Predicting buy/sell/hold is more tractable than predicting exact weights. Get direction right, then size positions.
5. **Baseline matters**: Before trying fancy architectures, make sure a simple heuristic (e.g., "hold previous quarter's weights") is established as the baseline. Beat that first.
6. **Factor exposures**: The Fama-French 6-factor model (5 factors + momentum) can decompose each manager's return style. Use rolling factor betas as features.
7. **Qualitative signal**: The manager qualitative assessments contain information about investment style, conviction, concentration, and process that should inform priors (e.g., Mgr_A favors concentrated large-cap positions; Mgr_LS has long-short tendencies).
8. **Cross-fund learning**: Patterns in how one manager trades may transfer to others in the same category. Consider shared representations.
9. **Metal/MPS**: If running on Apple Silicon, tensors go to `"mps"` device. Training should auto-detect and use available GPU.

---

## Evaluation Details

`prepare.py` exposes an `evaluate(predictions)` function. Predictions must be a dict:

```python
{
    "Mgr_A":  [{"ticker": "AAPL", "weight": 9.1}, {"ticker": "MSFT", "weight": 11.3}, ...],
    "Mgr_B":  [...],
    "Mgr_C":  [...],
    "Mgr_LS": [...]
}
```

The function compares against the Q4 2025 ground truth and returns:

```python
{
    "trade_direction_accuracy": 0.XX,  # primary metric (higher is better)
    "weight_mae": 0.XX,               # secondary metric (lower is better)
    "per_fund": {
        "Mgr_A":  {"direction_acc": ..., "mae": ...},
        "Mgr_B":  {"direction_acc": ..., "mae": ...},
        ...
    }
}
```

---

## Logging

Every experiment appends a JSON line to `runs/log.jsonl`:

```json
{
    "experiment_id": 1,
    "timestamp": "2026-03-08T12:00:00",
    "hypothesis": "Add momentum factor features to LSTM input",
    "change_summary": "Added 6-factor rolling betas (60-day window) as additional input features",
    "trade_direction_accuracy": 0.62,
    "weight_mae": 1.45,
    "per_fund": {...},
    "improved": true,
    "wall_clock_seconds": 487
}
```

Read this log at the start of each experiment to avoid repeating failed ideas.
