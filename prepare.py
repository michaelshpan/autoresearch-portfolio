"""
prepare.py — Data loading, feature engineering, and evaluation utilities.
DO NOT MODIFY. The autonomous agent only modifies train.py.
"""

import os
import sys
import json
import warnings
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime

warnings.filterwarnings("ignore")

# ─────────────────────────────────────────────
# Constants
# ─────────────────────────────────────────────

DATA_DIR = Path("data")
RUNS_DIR = Path("runs")
RUNS_DIR.mkdir(exist_ok=True)

FUNDS = ["Mgr_A", "Mgr_B", "Mgr_C", "Mgr_LS"]
TRAIN_END = "2025-Q2"       # last quarter used for training
VAL_QUARTER = "2025-Q3"     # validation quarter
TEST_QUARTER = "2025-Q4"    # held-out ground truth

# Trade direction thresholds (default, can be overridden in train.py)
BUY_THRESHOLD = 0.5   # weight increase > 0.5pp → BUY
SELL_THRESHOLD = -0.5  # weight decrease < -0.5pp → SELL

TIME_BUDGET_SECONDS = 600  # 10 minutes

# Quarter ordering for sequencing
QUARTER_ORDER = [
    "2019-Q3", "2019-Q4",
    "2020-Q1", "2020-Q2", "2020-Q3", "2020-Q4",
    "2021-Q1", "2021-Q2", "2021-Q3", "2021-Q4",
    "2022-Q1", "2022-Q2", "2022-Q3", "2022-Q4",
    "2023-Q1", "2023-Q2", "2023-Q3", "2023-Q4",
    "2024-Q1", "2024-Q2", "2024-Q3", "2024-Q4",
    "2025-Q1", "2025-Q2", "2025-Q3", "2025-Q4",
]


# ─────────────────────────────────────────────
# Data Loading
# ─────────────────────────────────────────────

def load_holdings():
    """Load quarterly holdings for all four managers.
    Aggregates duplicate tickers (e.g. multiple _Private entries) within each fund-quarter.
    """
    df = pd.read_csv(DATA_DIR / "holdings_quarterly_fin.csv")
    df["pct_weight"] = pd.to_numeric(df["pct_weight"], errors="coerce")
    df["dollar_value"] = pd.to_numeric(df["dollar_value"], errors="coerce")
    df["shares"] = pd.to_numeric(df["shares"], errors="coerce")

    # Aggregate duplicates: sum weights/shares/dollars, keep first sector/cusip/security_name
    agg = df.groupby(["fund_ticker", "report_date", "quarter", "ticker"]).agg({
        "security_name": "first",
        "sector": "first",
        "cusip": "first",
        "shares": "sum",
        "pct_weight": "sum",
        "dollar_value": "sum",
    }).reset_index()

    return agg


def load_benchmark_holdings():
    """Load VONG (R1000G proxy) quarterly holdings — the investable universe.
    Aggregates duplicate tickers within each quarter.
    """
    df = pd.read_csv(DATA_DIR / "holdings_vong.csv")
    df["pct_weight"] = pd.to_numeric(df["pct_weight"], errors="coerce")

    agg = df.groupby(["fund_ticker", "report_date", "quarter", "ticker"]).agg({
        "security_name": "first",
        "sector": "first",
        "cusip": "first",
        "shares": "sum",
        "pct_weight": "sum",
        "dollar_value": "sum",
    }).reset_index()

    return agg


def load_nav():
    """Load daily NAV series for all four managers."""
    df = pd.read_csv(DATA_DIR / "portfolio_daily_nav_fin.csv", parse_dates=["Date"])
    df.set_index("Date", inplace=True)
    return df


def load_benchmark_prices():
    """Load daily benchmark prices (R3000, R1000G)."""
    df = pd.read_csv(DATA_DIR / "benchmark_daily_prices.csv", parse_dates=["Date"])
    df.set_index("Date", inplace=True)
    return df


def load_ff_factors():
    """Load Fama-French 5 factors + momentum, merged into a single daily DataFrame."""
    # FF5 — skip header rows
    ff5_raw = pd.read_csv(DATA_DIR / "F-F_Research_Data_5_Factors_2x3_daily.csv", skiprows=4)
    ff5_raw.columns = [c.strip() for c in ff5_raw.columns]
    first_col = ff5_raw.columns[0]
    ff5_raw = ff5_raw[ff5_raw[first_col].apply(lambda x: str(x).strip().isdigit())]
    ff5_raw["Date"] = pd.to_datetime(ff5_raw[first_col].astype(str).str.strip(), format="%Y%m%d")
    ff5_raw = ff5_raw.drop(columns=[first_col])
    for col in ff5_raw.columns:
        if col != "Date":
            ff5_raw[col] = pd.to_numeric(ff5_raw[col], errors="coerce")
    ff5_raw.set_index("Date", inplace=True)

    # Momentum
    mom_raw = pd.read_csv(DATA_DIR / "F-F_Momentum_Factor_daily.csv", skiprows=14)
    mom_raw.columns = [c.strip() for c in mom_raw.columns]
    first_col = mom_raw.columns[0]
    mom_raw = mom_raw[mom_raw[first_col].apply(lambda x: str(x).strip().isdigit())]
    mom_raw["Date"] = pd.to_datetime(mom_raw[first_col].astype(str).str.strip(), format="%Y%m%d")
    mom_raw = mom_raw.drop(columns=[first_col])
    mom_raw.rename(columns={mom_raw.columns[0]: "MOM"}, inplace=True)
    mom_raw["MOM"] = pd.to_numeric(mom_raw["MOM"], errors="coerce")
    mom_raw.set_index("Date", inplace=True)

    factors = ff5_raw.join(mom_raw, how="inner")
    return factors


def load_qualitative_features():
    """
    Parse qualitative markdown files into a structured dict per manager.
    Returns dict: {fund_ticker: qualitative_text}
    """
    qual = {}
    for fund in FUNDS:
        path = DATA_DIR / f"{fund} - Qualitative_fin.md"
        if path.exists():
            qual[fund] = path.read_text(encoding="utf-8")
        else:
            qual[fund] = ""
    return qual


def load_quantitative_features():
    """
    Parse quantitative markdown files into a structured dict per manager.
    Returns dict: {fund_ticker: quantitative_text}
    """
    quant = {}
    for fund in FUNDS:
        path = DATA_DIR / f"{fund} - Quantitative.md"
        if path.exists():
            quant[fund] = path.read_text(encoding="utf-8")
        else:
            quant[fund] = ""
    return quant


# ─────────────────────────────────────────────
# Feature Engineering Utilities
# ─────────────────────────────────────────────

def compute_returns(nav_df):
    """Compute daily returns from NAV series."""
    return nav_df.pct_change().dropna()


def compute_rolling_factor_betas(nav_df, factors_df, window=60):
    """
    Compute rolling OLS factor betas for each fund against FF6 factors.
    Returns dict: {fund: DataFrame of rolling betas}
    """
    returns = compute_returns(nav_df)
    # Align dates
    common_idx = returns.index.intersection(factors_df.index)
    returns = returns.loc[common_idx]
    factors = factors_df.loc[common_idx]

    factor_cols = [c for c in factors.columns if c != "RF"]
    betas = {}

    for fund in FUNDS:
        if fund not in returns.columns:
            continue
        excess_ret = returns[fund] - factors["RF"] / 100  # RF is in percent
        fund_betas = pd.DataFrame(index=common_idx, columns=factor_cols, dtype=float)

        for i in range(window, len(common_idx)):
            y = excess_ret.iloc[i - window:i].values
            X = factors[factor_cols].iloc[i - window:i].values / 100
            X = np.column_stack([np.ones(len(X)), X])
            try:
                beta = np.linalg.lstsq(X, y, rcond=None)[0]
                fund_betas.iloc[i] = beta[1:]  # exclude intercept
            except Exception:
                pass

        fund_betas = fund_betas.dropna()
        betas[fund] = fund_betas

    return betas


def build_trade_labels(holdings_df, buy_thresh=BUY_THRESHOLD, sell_thresh=SELL_THRESHOLD):
    """
    Build trade direction labels between consecutive quarters.
    Returns DataFrame with columns: fund_ticker, quarter, ticker, prev_weight,
    curr_weight, weight_change, direction (0=SELL, 1=HOLD, 2=BUY)
    """
    records = []
    for fund in FUNDS:
        fund_df = holdings_df[holdings_df["fund_ticker"] == fund].copy()
        quarters = sorted(fund_df["quarter"].unique(), key=lambda q: QUARTER_ORDER.index(q))

        for i in range(1, len(quarters)):
            prev_q = quarters[i - 1]
            curr_q = quarters[i]

            prev = fund_df[fund_df["quarter"] == prev_q][["ticker", "pct_weight"]].set_index("ticker")
            curr = fund_df[fund_df["quarter"] == curr_q][["ticker", "pct_weight"]].set_index("ticker")

            all_tickers = sorted(set(prev.index) | set(curr.index))

            for tick in all_tickers:
                pw = prev.loc[tick, "pct_weight"] if tick in prev.index else 0.0
                cw = curr.loc[tick, "pct_weight"] if tick in curr.index else 0.0
                change = cw - pw

                if change > buy_thresh:
                    direction = 2  # BUY
                elif change < sell_thresh:
                    direction = 0  # SELL
                else:
                    direction = 1  # HOLD

                records.append({
                    "fund_ticker": fund,
                    "quarter": curr_q,
                    "ticker": tick,
                    "prev_weight": pw,
                    "curr_weight": cw,
                    "weight_change": change,
                    "direction": direction,
                })

    return pd.DataFrame(records)


def get_universe(benchmark_holdings_df, quarter):
    """Get the set of tickers in the R1000G benchmark for a given quarter."""
    qdf = benchmark_holdings_df[benchmark_holdings_df["quarter"] == quarter]
    return set(qdf["ticker"].unique())


def split_data(trade_labels_df):
    """
    Split trade labels into train, val, test sets based on quarter.
    Train: all quarters up to and including TRAIN_END
    Val: VAL_QUARTER
    Test: TEST_QUARTER
    """
    train_quarters = [q for q in QUARTER_ORDER if QUARTER_ORDER.index(q) <= QUARTER_ORDER.index(TRAIN_END)]

    train = trade_labels_df[trade_labels_df["quarter"].isin(train_quarters)]
    val = trade_labels_df[trade_labels_df["quarter"] == VAL_QUARTER]
    test = trade_labels_df[trade_labels_df["quarter"] == TEST_QUARTER]

    return train, val, test


# ─────────────────────────────────────────────
# Evaluation
# ─────────────────────────────────────────────

def evaluate(predictions, holdings_df=None, verbose=True):
    """
    Evaluate predicted Q4 2025 holdings against ground truth.

    Args:
        predictions: dict of {fund_ticker: [{"ticker": str, "weight": float}, ...]}
        holdings_df: if None, loads from disk
        verbose: print results

    Returns:
        dict with trade_direction_accuracy, weight_mae, per_fund breakdown
    """
    if holdings_df is None:
        holdings_df = load_holdings()

    results = {}
    total_correct = 0
    total_count = 0
    total_mae = 0.0
    total_weight_count = 0

    for fund in FUNDS:
        # Get Q3 2025 (prev) and Q4 2025 (actual) holdings
        prev = holdings_df[(holdings_df["fund_ticker"] == fund) &
                           (holdings_df["quarter"] == VAL_QUARTER)][["ticker", "pct_weight"]].set_index("ticker")
        actual = holdings_df[(holdings_df["fund_ticker"] == fund) &
                             (holdings_df["quarter"] == TEST_QUARTER)][["ticker", "pct_weight"]].set_index("ticker")

        # Predicted holdings
        pred_list = predictions.get(fund, [])
        pred = pd.DataFrame(pred_list).set_index("ticker") if pred_list else pd.DataFrame(columns=["weight"]).rename_axis("ticker")
        if "weight" in pred.columns:
            pred = pred.rename(columns={"weight": "pct_weight"})

        # Universe: union of all tickers in prev, actual, and predicted
        all_tickers = sorted(set(prev.index) | set(actual.index) | set(pred.index))

        correct = 0
        count = 0
        mae_sum = 0.0

        for tick in all_tickers:
            pw = prev.loc[tick, "pct_weight"] if tick in prev.index else 0.0
            aw = actual.loc[tick, "pct_weight"] if tick in actual.index else 0.0
            predw = pred.loc[tick, "pct_weight"] if tick in pred.index else 0.0

            # Actual direction
            actual_change = aw - pw
            if actual_change > BUY_THRESHOLD:
                actual_dir = 2
            elif actual_change < SELL_THRESHOLD:
                actual_dir = 0
            else:
                actual_dir = 1

            # Predicted direction
            pred_change = predw - pw
            if pred_change > BUY_THRESHOLD:
                pred_dir = 2
            elif pred_change < SELL_THRESHOLD:
                pred_dir = 0
            else:
                pred_dir = 1

            if actual_dir == pred_dir:
                correct += 1
            count += 1
            mae_sum += abs(aw - predw)

        direction_acc = correct / count if count > 0 else 0.0
        mae = mae_sum / count if count > 0 else 0.0

        results[fund] = {"direction_acc": round(direction_acc, 4), "mae": round(mae, 4)}
        total_correct += correct
        total_count += count
        total_mae += mae_sum
        total_weight_count += count

    overall_acc = total_correct / total_count if total_count > 0 else 0.0
    overall_mae = total_mae / total_weight_count if total_weight_count > 0 else 0.0

    output = {
        "trade_direction_accuracy": round(overall_acc, 4),
        "weight_mae": round(overall_mae, 4),
        "per_fund": results,
    }

    if verbose:
        print("\n" + "=" * 60)
        print("EVALUATION RESULTS")
        print("=" * 60)
        print(f"  Trade Direction Accuracy:  {output['trade_direction_accuracy']:.4f}")
        print(f"  Weight MAE:               {output['weight_mae']:.4f}")
        print()
        for fund in FUNDS:
            r = results[fund]
            print(f"  {fund:8s}  dir_acc={r['direction_acc']:.4f}  mae={r['mae']:.4f}")
        print("=" * 60)

    return output


# ─────────────────────────────────────────────
# Experiment Logging
# ─────────────────────────────────────────────

def log_experiment(experiment_id, hypothesis, change_summary, eval_results, wall_clock_seconds, improved):
    """Append an experiment record to runs/log.jsonl."""
    record = {
        "experiment_id": experiment_id,
        "timestamp": datetime.now().isoformat(),
        "hypothesis": hypothesis,
        "change_summary": change_summary,
        "trade_direction_accuracy": eval_results["trade_direction_accuracy"],
        "weight_mae": eval_results["weight_mae"],
        "per_fund": eval_results["per_fund"],
        "improved": improved,
        "wall_clock_seconds": round(wall_clock_seconds, 1),
    }
    log_path = RUNS_DIR / "log.jsonl"
    with open(log_path, "a") as f:
        f.write(json.dumps(record) + "\n")
    return record


def load_experiment_log():
    """Load all past experiments from log."""
    log_path = RUNS_DIR / "log.jsonl"
    if not log_path.exists():
        return []
    records = []
    with open(log_path) as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def get_next_experiment_id():
    """Get the next experiment ID."""
    records = load_experiment_log()
    if not records:
        return 1
    return max(r["experiment_id"] for r in records) + 1


# ─────────────────────────────────────────────
# Device Detection
# ─────────────────────────────────────────────

def get_device():
    """Auto-detect best available device: MPS > CUDA > CPU."""
    import torch
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    elif torch.cuda.is_available():
        return torch.device("cuda")
    else:
        return torch.device("cpu")


# ─────────────────────────────────────────────
# Main: run as one-time data prep / verification
# ─────────────────────────────────────────────

if __name__ == "__main__":
    print("=" * 60)
    print("Portfolio Replication — Data Verification")
    print("=" * 60)

    # Check data files exist
    required_files = [
        "holdings_quarterly_fin.csv",
        "portfolio_daily_nav_fin.csv",
        "benchmark_daily_prices.csv",
        "holdings_vong.csv",
        "F-F_Research_Data_5_Factors_2x3_daily.csv",
        "F-F_Momentum_Factor_daily.csv",
    ]
    for fname in required_files:
        path = DATA_DIR / fname
        if path.exists():
            print(f"  ✓ {fname}")
        else:
            print(f"  ✗ {fname} — MISSING")
            sys.exit(1)

    # Load and verify
    holdings = load_holdings()
    print(f"\n  Holdings: {len(holdings)} rows, {holdings['fund_ticker'].nunique()} funds, "
          f"{holdings['quarter'].nunique()} quarters")

    nav = load_nav()
    print(f"  NAV: {len(nav)} days, columns: {list(nav.columns)}")

    bench = load_benchmark_prices()
    print(f"  Benchmark: {len(bench)} days, columns: {list(bench.columns)}")

    factors = load_ff_factors()
    print(f"  FF Factors: {len(factors)} days, columns: {list(factors.columns)}")

    vong = load_benchmark_holdings()
    print(f"  VONG holdings: {len(vong)} rows")

    # Build trade labels
    labels = build_trade_labels(holdings)
    print(f"\n  Trade labels: {len(labels)} rows")
    print(f"  Direction distribution:")
    for d, name in [(0, "SELL"), (1, "HOLD"), (2, "BUY")]:
        count = (labels["direction"] == d).sum()
        print(f"    {name}: {count} ({count / len(labels) * 100:.1f}%)")

    # Split
    train, val, test = split_data(labels)
    print(f"\n  Train: {len(train)} rows (up to {TRAIN_END})")
    print(f"  Val:   {len(val)} rows ({VAL_QUARTER})")
    print(f"  Test:  {len(test)} rows ({TEST_QUARTER})")

    # Device
    try:
        import torch
        device = get_device()
        print(f"\n  Device: {device}")
    except ImportError:
        print("\n  PyTorch not installed — skipping device check")

    print("\n" + "=" * 60)
    print("Data verification complete. Ready to train.")
    print("=" * 60)
