"""
train.py — LSTM-based portfolio trade direction predictor.
THIS FILE IS MODIFIED BY THE AUTONOMOUS AGENT.

Baseline: simple LSTM that takes per-fund sequences of (quarter, security_features)
and predicts buy/sell/hold direction for the next quarter.
"""

import os
import sys
import time
import json
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

# Import utilities from prepare.py (do not modify prepare.py)
from prepare import (
    load_holdings,
    load_benchmark_holdings,
    load_nav,
    load_benchmark_prices,
    load_ff_factors,
    compute_returns,
    compute_rolling_factor_betas,
    build_trade_labels,
    split_data,
    evaluate,
    log_experiment,
    get_next_experiment_id,
    get_device,
    FUNDS,
    QUARTER_ORDER,
    TIME_BUDGET_SECONDS,
    VAL_QUARTER,
    TEST_QUARTER,
)


# ─────────────────────────────────────────────
# Experiment metadata (update each experiment)
# ─────────────────────────────────────────────

HYPOTHESIS = "Baseline: LSTM with weight-history + factor-beta features, predict trade direction"
CHANGE_SUMMARY = "Initial baseline model"


# ─────────────────────────────────────────────
# Hyperparameters
# ─────────────────────────────────────────────

HIDDEN_DIM = 64
NUM_LAYERS = 2
DROPOUT = 0.3
LEARNING_RATE = 1e-3
WEIGHT_DECAY = 1e-4
BATCH_SIZE = 32
NUM_EPOCHS = 50
EARLY_STOP_PATIENCE = 8
SEQUENCE_LEN = 4          # number of past quarters to look back
NUM_CLASSES = 3            # SELL=0, HOLD=1, BUY=2
GRAD_CLIP = 1.0


# ─────────────────────────────────────────────
# Feature Construction
# ─────────────────────────────────────────────

def build_features(holdings_df, benchmark_holdings_df, nav_df, factors_df):
    """
    Build feature tensors for each (fund, quarter, security) sample.

    Features per security per quarter:
        - pct_weight (current quarter)
        - weight_change from previous quarter
        - weight relative to benchmark (active weight)
        - sector one-hot encoding
        - rolling factor betas for the fund (6 values)

    Returns:
        sequences: dict {fund: {quarter: {ticker: feature_vector}}}
        all_sectors: sorted list of sector labels
    """
    # Compute rolling factor betas
    print("  Computing rolling factor betas...")
    betas = compute_rolling_factor_betas(nav_df, factors_df, window=60)

    # Map quarters to approximate end-dates for factor beta lookup
    quarter_end_dates = {}
    for q in QUARTER_ORDER:
        year, qnum = q.split("-Q")
        month = int(qnum) * 3
        # Approximate last trading day
        quarter_end_dates[q] = pd.Timestamp(f"{year}-{month:02d}-28")

    # Get all sectors
    all_sectors = sorted(holdings_df["sector"].dropna().unique())
    sector_to_idx = {s: i for i, s in enumerate(all_sectors)}
    num_sectors = len(all_sectors)

    sequences = {}

    for fund in FUNDS:
        fund_holdings = holdings_df[holdings_df["fund_ticker"] == fund]
        fund_quarters = sorted(fund_holdings["quarter"].unique(),
                               key=lambda q: QUARTER_ORDER.index(q))

        fund_betas = betas.get(fund, pd.DataFrame())
        sequences[fund] = {}

        for qi, quarter in enumerate(fund_quarters):
            curr = fund_holdings[fund_holdings["quarter"] == quarter]
            curr_weights = curr.set_index("ticker")["pct_weight"].to_dict()
            curr_sectors = curr.set_index("ticker")["sector"].to_dict()

            # Previous quarter weights
            if qi > 0:
                prev_q = fund_quarters[qi - 1]
                prev = fund_holdings[fund_holdings["quarter"] == prev_q]
                prev_weights = prev.set_index("ticker")["pct_weight"].to_dict()
            else:
                prev_weights = {}

            # Benchmark weights for this quarter
            bench_q = benchmark_holdings_df[benchmark_holdings_df["quarter"] == quarter]
            bench_weights = bench_q.set_index("ticker")["pct_weight"].to_dict() if len(bench_q) > 0 else {}

            # Factor betas (nearest date)
            beta_values = np.zeros(6)
            if len(fund_betas) > 0 and quarter in quarter_end_dates:
                target_date = quarter_end_dates[quarter]
                available = fund_betas.index[fund_betas.index <= target_date]
                if len(available) > 0:
                    nearest = available[-1]
                    beta_values = fund_betas.loc[nearest].values.astype(float)
                    if len(beta_values) < 6:
                        beta_values = np.pad(beta_values, (0, 6 - len(beta_values)))

            quarter_features = {}
            all_tickers = set(curr_weights.keys()) | set(prev_weights.keys())

            for tick in all_tickers:
                cw = curr_weights.get(tick, 0.0)
                pw = prev_weights.get(tick, 0.0)
                bw = bench_weights.get(tick, 0.0)

                # Core features
                features = [
                    cw,            # current weight
                    pw,            # previous weight
                    cw - pw,       # weight change
                    cw - bw,       # active weight vs benchmark
                    bw,            # benchmark weight
                ]

                # Sector one-hot
                sector = curr_sectors.get(tick, "")
                sector_vec = [0.0] * num_sectors
                if sector in sector_to_idx:
                    sector_vec[sector_to_idx[sector]] = 1.0
                features.extend(sector_vec)

                # Fund-level factor betas
                features.extend(beta_values.tolist())

                quarter_features[tick] = np.array(features, dtype=np.float32)

            sequences[fund][quarter] = quarter_features

    return sequences, all_sectors


def build_training_samples(sequences, trade_labels_df, quarters_to_include):
    """
    Build (X, y) samples from sequences and trade labels.

    For each (fund, quarter, ticker) in trade_labels, look back SEQUENCE_LEN
    quarters to build a sequence of feature vectors.

    Returns:
        X: np.ndarray of shape (N, SEQUENCE_LEN, feature_dim)
        y: np.ndarray of shape (N,) with labels 0/1/2
    """
    X_list = []
    y_list = []

    labels_subset = trade_labels_df[trade_labels_df["quarter"].isin(quarters_to_include)]

    for _, row in labels_subset.iterrows():
        fund = row["fund_ticker"]
        quarter = row["quarter"]
        ticker = row["ticker"]
        direction = row["direction"]

        qi = QUARTER_ORDER.index(quarter)

        # Build sequence: look back SEQUENCE_LEN quarters
        seq = []
        feature_dim = None

        for offset in range(SEQUENCE_LEN, 0, -1):
            past_qi = qi - offset
            if past_qi < 0:
                continue
            past_q = QUARTER_ORDER[past_qi]

            if fund in sequences and past_q in sequences[fund]:
                if ticker in sequences[fund][past_q]:
                    feat = sequences[fund][past_q][ticker]
                    feature_dim = len(feat)
                    seq.append(feat)
                elif feature_dim is not None:
                    seq.append(np.zeros(feature_dim, dtype=np.float32))
            elif feature_dim is not None:
                seq.append(np.zeros(feature_dim, dtype=np.float32))

        if len(seq) == 0 or feature_dim is None:
            continue

        # Pad if sequence is shorter than SEQUENCE_LEN
        while len(seq) < SEQUENCE_LEN:
            seq.insert(0, np.zeros(feature_dim, dtype=np.float32))

        X_list.append(np.stack(seq))
        y_list.append(direction)

    if len(X_list) == 0:
        return np.array([]), np.array([])

    return np.stack(X_list), np.array(y_list, dtype=np.int64)


# ─────────────────────────────────────────────
# Dataset
# ─────────────────────────────────────────────

class TradeDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.long)

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


# ─────────────────────────────────────────────
# Loss
# ─────────────────────────────────────────────

class TradeDirectionLoss(nn.Module):
    def __init__(self, hold_weight=0.05):
        super().__init__()
        # BUY and SELL get full weight, HOLD gets 5%
        self.class_weights = None
        self.hold_weight = hold_weight

    def forward(self, logits, targets):
        # Per-sample weights: 1.0 for BUY/SELL, hold_weight for HOLD
        sample_weights = torch.ones_like(targets, dtype=torch.float32)
        sample_weights[targets == 1] = self.hold_weight

        loss = nn.functional.cross_entropy(
            logits, targets, reduction='none'
        )
        weighted_loss = (loss * sample_weights).sum() / sample_weights.sum()
        return weighted_loss


# ─────────────────────────────────────────────
# Model
# ─────────────────────────────────────────────

class TradeLSTM(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_layers, num_classes, dropout):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_dim, num_classes)

    def forward(self, x):
        # x: (batch, seq_len, input_dim)
        lstm_out, (h_n, _) = self.lstm(x)
        # Use last hidden state
        last_hidden = h_n[-1]  # (batch, hidden_dim)
        out = self.dropout(last_hidden)
        logits = self.fc(out)  # (batch, num_classes)
        return logits


# ─────────────────────────────────────────────
# Training Loop
# ─────────────────────────────────────────────

def train_model(model, train_loader, val_X, val_y, device):
    """Train with early stopping on validation accuracy."""

    # Class weights to handle imbalanced directions
    if len(train_loader.dataset) > 0:
        all_labels = train_loader.dataset.y.numpy()
        class_counts = np.bincount(all_labels, minlength=NUM_CLASSES).astype(float)
        class_counts = np.maximum(class_counts, 1.0)
        class_weights = 1.0 / class_counts
        class_weights = class_weights / class_weights.sum() * NUM_CLASSES
        class_weights = torch.tensor(class_weights, dtype=torch.float32).to(device)
    else:
        class_weights = torch.ones(NUM_CLASSES).to(device)

    criterion = TradeDirectionLoss(hold_weight=0.05)
    optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="max", patience=3, factor=0.5)

    val_X_tensor = torch.tensor(val_X, dtype=torch.float32).to(device)
    val_y_tensor = torch.tensor(val_y, dtype=torch.long).to(device)

    best_val_acc = 0.0
    best_state = None
    patience_counter = 0
    start_time = time.time()

    for epoch in range(NUM_EPOCHS):
        # Check time budget
        elapsed = time.time() - start_time
        if elapsed > TIME_BUDGET_SECONDS * 0.85:  # reserve 15% for eval
            print(f"  Time budget approaching — stopping at epoch {epoch}")
            break

        model.train()
        total_loss = 0.0
        total_correct = 0
        total_samples = 0

        for X_batch, y_batch in train_loader:
            X_batch = X_batch.to(device)
            y_batch = y_batch.to(device)

            optimizer.zero_grad()
            logits = model(X_batch)
            loss = criterion(logits, y_batch)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
            optimizer.step()

            total_loss += loss.item() * len(y_batch)
            preds = logits.argmax(dim=1)
            total_correct += (preds == y_batch).sum().item()
            total_samples += len(y_batch)

        train_acc = total_correct / total_samples if total_samples > 0 else 0
        train_loss = total_loss / total_samples if total_samples > 0 else 0

        # Validation
        model.eval()
        with torch.no_grad():
            val_logits = model(val_X_tensor)
            val_preds = val_logits.argmax(dim=1)
            val_acc = (val_preds == val_y_tensor).float().mean().item()

        scheduler.step(val_acc)

        if epoch % 5 == 0 or epoch == NUM_EPOCHS - 1:
            print(f"  Epoch {epoch:3d}  loss={train_loss:.4f}  train_acc={train_acc:.4f}  val_acc={val_acc:.4f}")

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= EARLY_STOP_PATIENCE:
                print(f"  Early stopping at epoch {epoch}")
                break

    if best_state is not None:
        model.load_state_dict(best_state)

    return model, best_val_acc


# ─────────────────────────────────────────────
# Prediction: convert model outputs to portfolio holdings
# ─────────────────────────────────────────────

def predict_holdings(model, sequences, holdings_df, benchmark_holdings_df, device):
    """
    Predict Q4 2025 holdings for all four managers.

    Strategy:
        1. For each (fund, security), predict trade direction using the model.
        2. Apply direction to Q3 2025 weights:
           - BUY  → increase weight by a fixed step
           - SELL → decrease weight (or remove)
           - HOLD → keep weight
        3. Normalize to sum to ~100%.

    Returns predictions dict for evaluate().
    """
    model.eval()
    predictions = {}

    # Get Q3 2025 holdings as base
    for fund in FUNDS:
        fund_holdings = holdings_df[holdings_df["fund_ticker"] == fund]
        prev_q_holdings = fund_holdings[fund_holdings["quarter"] == VAL_QUARTER]
        prev_weights = prev_q_holdings.set_index("ticker")["pct_weight"].to_dict()

        # Benchmark universe for Q4 2025 (use Q3 if Q4 not available)
        bench_universe = get_benchmark_tickers(benchmark_holdings_df, TEST_QUARTER)
        if not bench_universe:
            bench_universe = get_benchmark_tickers(benchmark_holdings_df, VAL_QUARTER)

        # Candidate tickers: union of previous holdings and benchmark
        candidates = set(prev_weights.keys()) | bench_universe

        predicted_holdings = []

        for ticker in candidates:
            # Build feature sequence for this ticker
            seq = build_single_sequence(sequences, fund, ticker, TEST_QUARTER)
            if seq is None:
                # Default: hold previous weight
                pw = prev_weights.get(ticker, 0.0)
                if pw > 0.1:
                    predicted_holdings.append({"ticker": ticker, "weight": pw})
                continue

            X = torch.tensor(seq, dtype=torch.float32).unsqueeze(0).to(device)

            with torch.no_grad():
                logits = model(X)
                pred_dir = logits.argmax(dim=1).item()

            pw = prev_weights.get(ticker, 0.0)

            # Apply direction
            if pred_dir == 2:    # BUY
                new_weight = pw + 1.0 if pw > 0 else 1.0  # add or initiate
            elif pred_dir == 0:  # SELL
                new_weight = max(pw - 1.0, 0.0)            # reduce or exit
            else:                # HOLD
                new_weight = pw

            if new_weight > 0.05:  # minimum threshold
                predicted_holdings.append({"ticker": ticker, "weight": new_weight})

        # Normalize weights to sum to 100
        total_w = sum(h["weight"] for h in predicted_holdings)
        if total_w > 0:
            for h in predicted_holdings:
                h["weight"] = round(h["weight"] / total_w * 100, 4)

        predictions[fund] = predicted_holdings

    return predictions


def get_benchmark_tickers(benchmark_holdings_df, quarter):
    """Get set of tickers in benchmark for a given quarter."""
    qdf = benchmark_holdings_df[benchmark_holdings_df["quarter"] == quarter]
    return set(qdf["ticker"].unique())


def build_single_sequence(sequences, fund, ticker, target_quarter):
    """Build a single feature sequence for prediction."""
    qi = QUARTER_ORDER.index(target_quarter) if target_quarter in QUARTER_ORDER else -1
    if qi < 0:
        return None

    seq = []
    feature_dim = None

    for offset in range(SEQUENCE_LEN, 0, -1):
        past_qi = qi - offset
        if past_qi < 0:
            continue
        past_q = QUARTER_ORDER[past_qi]

        if fund in sequences and past_q in sequences[fund]:
            if ticker in sequences[fund][past_q]:
                feat = sequences[fund][past_q][ticker]
                feature_dim = len(feat)
                seq.append(feat)
            elif feature_dim is not None:
                seq.append(np.zeros(feature_dim, dtype=np.float32))
        elif feature_dim is not None:
            seq.append(np.zeros(feature_dim, dtype=np.float32))

    if len(seq) == 0 or feature_dim is None:
        return None

    while len(seq) < SEQUENCE_LEN:
        seq.insert(0, np.zeros(feature_dim, dtype=np.float32))

    return np.stack(seq)


# ─────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────

def main():
    start_time = time.time()
    experiment_id = get_next_experiment_id()

    print(f"\n{'=' * 60}")
    print(f"EXPERIMENT {experiment_id}")
    print(f"Hypothesis: {HYPOTHESIS}")
    print(f"{'=' * 60}\n")

    device = get_device()
    print(f"Device: {device}")

    # ── Load data ──
    print("\nLoading data...")
    holdings = load_holdings()
    benchmark_holdings = load_benchmark_holdings()
    nav = load_nav()
    factors = load_ff_factors()

    # ── Build features ──
    print("Building features...")
    sequences, all_sectors = build_features(holdings, benchmark_holdings, nav, factors)

    # ── Build trade labels and split ──
    print("Building trade labels...")
    trade_labels = build_trade_labels(holdings)
    train_labels, val_labels, test_labels = split_data(trade_labels)

    print(f"  Train: {len(train_labels)} samples")
    print(f"  Val:   {len(val_labels)} samples")
    print(f"  Test:  {len(test_labels)} samples")

    # ── Build training tensors ──
    print("Building training samples...")
    train_quarters = sorted(train_labels["quarter"].unique())
    val_quarters = [VAL_QUARTER]

    X_train, y_train = build_training_samples(sequences, trade_labels, train_quarters)
    X_val, y_val = build_training_samples(sequences, trade_labels, val_quarters)

    print(f"  X_train: {X_train.shape if len(X_train) > 0 else 'empty'}")
    print(f"  X_val:   {X_val.shape if len(X_val) > 0 else 'empty'}")

    if len(X_train) == 0:
        print("ERROR: No training samples generated. Check feature construction.")
        sys.exit(1)

    feature_dim = X_train.shape[2]
    print(f"  Feature dim: {feature_dim}")

    # ── Datasets ──
    train_dataset = TradeDataset(X_train, y_train)
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, drop_last=False)

    # ── Model ──
    model = TradeLSTM(
        input_dim=feature_dim,
        hidden_dim=HIDDEN_DIM,
        num_layers=NUM_LAYERS,
        num_classes=NUM_CLASSES,
        dropout=DROPOUT,
    ).to(device)

    param_count = sum(p.numel() for p in model.parameters())
    print(f"  Model parameters: {param_count:,}")

    # ── Train ──
    print("\nTraining...")
    model, best_val_acc = train_model(model, train_loader, X_val, y_val, device)
    print(f"\n  Best validation accuracy: {best_val_acc:.4f}")

    # ── Predict Q4 2025 ──
    print("\nPredicting Q4 2025 holdings...")
    predictions = predict_holdings(model, sequences, holdings, benchmark_holdings, device)

    for fund in FUNDS:
        n = len(predictions.get(fund, []))
        print(f"  {fund}: {n} predicted holdings")

    # ── Evaluate ──
    eval_results = evaluate(predictions, holdings)

    # ── Check improvement ──
    wall_clock = time.time() - start_time
    best_so_far = 0.0
    log_path = "runs/log.jsonl"
    if os.path.exists(log_path):
        with open(log_path) as f:
            for line in f:
                rec = json.loads(line.strip())
                if rec.get("trade_direction_accuracy", 0) > best_so_far:
                    best_so_far = rec["trade_direction_accuracy"]

    improved = eval_results["trade_direction_accuracy"] > best_so_far

    if improved:
        torch.save(model.state_dict(), "best_model.pt")
        print(f"\n  ★ NEW BEST — saved best_model.pt")
    else:
        print(f"\n  No improvement (best so far: {best_so_far:.4f})")

    # ── Log ──
    log_experiment(experiment_id, HYPOTHESIS, CHANGE_SUMMARY, eval_results, wall_clock, improved)

    print(f"\n  Wall clock: {wall_clock:.1f}s ({wall_clock / 60:.1f} min)")
    print(f"  Time budget: {TIME_BUDGET_SECONDS}s ({TIME_BUDGET_SECONDS / 60:.0f} min)")
    print(f"\nExperiment {experiment_id} complete.\n")


if __name__ == "__main__":
    main()
