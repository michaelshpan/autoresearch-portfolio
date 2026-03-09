"""
run_loop.py — Autonomous research loop orchestrator.

Uses the Claude API (Sonnet) to iteratively read experiment logs, hypothesize,
edit train.py, run experiments, and keep/revert based on results.

Runs until cumulative API cost hits ~$10.
"""

import os
import sys
import json
import time
import shutil
import subprocess
import anthropic
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

# ─────────────────────────────────────────────
# Config
# ─────────────────────────────────────────────

MODEL = "claude-sonnet-4-6"
MAX_COST_USD = 10.0
RUN_TIMEOUT = 600          # 10 min max per train.py run
TRAIN_FILE = Path("train.py")
TRAIN_BACKUP = Path("train.py.best")
LOG_FILE = Path("runs/log.jsonl")
RUN_LOG = Path("run.log")

# Pricing per million tokens (Sonnet 4.6)
INPUT_COST_PER_M = 3.00
OUTPUT_COST_PER_M = 15.00

# ─────────────────────────────────────────────
# Tools the agent can use
# ─────────────────────────────────────────────

TOOLS = [
    {
        "name": "read_file",
        "description": "Read the contents of a file. Use this to read train.py, runs/log.jsonl, run.log, prepare.py, or CLAUDE.md.",
        "input_schema": {
            "type": "object",
            "properties": {
                "path": {
                    "type": "string",
                    "description": "Relative path to the file to read."
                }
            },
            "required": ["path"]
        }
    },
    {
        "name": "write_file",
        "description": "Overwrite a file with new contents. Only use this for train.py — do not modify other files.",
        "input_schema": {
            "type": "object",
            "properties": {
                "path": {
                    "type": "string",
                    "description": "Relative path (must be 'train.py')."
                },
                "content": {
                    "type": "string",
                    "description": "The complete new file contents."
                }
            },
            "required": ["path", "content"]
        }
    },
    {
        "name": "run_train",
        "description": "Run 'python train.py' and return the last 80 lines of output. This executes the training loop and evaluation. Output is also saved to run.log.",
        "input_schema": {
            "type": "object",
            "properties": {},
            "required": []
        }
    },
    {
        "name": "run_command",
        "description": "Run an arbitrary shell command and return stdout+stderr (max 200 lines). Use for git commands, grep, tail, etc.",
        "input_schema": {
            "type": "object",
            "properties": {
                "command": {
                    "type": "string",
                    "description": "The shell command to run."
                }
            },
            "required": ["command"]
        }
    },
]

# ─────────────────────────────────────────────
# Tool implementations
# ─────────────────────────────────────────────

ALLOWED_READ = {"train.py", "prepare.py", "CLAUDE.md", "runs/log.jsonl", "run.log"}


def tool_read_file(path: str) -> str:
    p = Path(path)
    if str(p) not in ALLOWED_READ and not str(p).startswith("data/"):
        return f"Error: reading '{path}' is not allowed. Allowed: {ALLOWED_READ}"
    if not p.exists():
        return f"Error: '{path}' does not exist."
    try:
        return p.read_text(encoding="utf-8")
    except Exception as e:
        return f"Error reading '{path}': {e}"


def tool_write_file(path: str, content: str) -> str:
    if path != "train.py":
        return "Error: you can only write to 'train.py'."
    try:
        Path(path).write_text(content, encoding="utf-8")
        return f"OK: wrote {len(content)} chars to train.py"
    except Exception as e:
        return f"Error writing '{path}': {e}"


def tool_run_train() -> str:
    try:
        result = subprocess.run(
            [sys.executable, "train.py"],
            capture_output=True,
            text=True,
            timeout=RUN_TIMEOUT,
        )
        combined = result.stdout + "\n" + result.stderr
        # Save full output
        RUN_LOG.write_text(combined, encoding="utf-8")
        # Return last 80 lines
        lines = combined.strip().split("\n")
        tail = lines[-80:] if len(lines) > 80 else lines
        status = "OK" if result.returncode == 0 else f"FAILED (exit code {result.returncode})"
        return f"[{status}]\n" + "\n".join(tail)
    except subprocess.TimeoutExpired:
        return "ERROR: train.py timed out after 10 minutes."
    except Exception as e:
        return f"ERROR running train.py: {e}"


def tool_run_command(command: str) -> str:
    # Block dangerous commands
    dangerous = ["rm -rf", "rm -r /", "mkfs", "dd if="]
    if any(d in command for d in dangerous):
        return "Error: command blocked for safety."
    try:
        result = subprocess.run(
            command,
            shell=True,
            capture_output=True,
            text=True,
            timeout=30,
        )
        combined = result.stdout + result.stderr
        lines = combined.strip().split("\n")
        tail = lines[-200:] if len(lines) > 200 else lines
        return "\n".join(tail)
    except subprocess.TimeoutExpired:
        return "ERROR: command timed out after 30s."
    except Exception as e:
        return f"ERROR: {e}"


def execute_tool(name: str, input_data: dict) -> str:
    if name == "read_file":
        return tool_read_file(input_data["path"])
    elif name == "write_file":
        return tool_write_file(input_data["path"], input_data["content"])
    elif name == "run_train":
        return tool_run_train()
    elif name == "run_command":
        return tool_run_command(input_data["command"])
    else:
        return f"Unknown tool: {name}"


# ─────────────────────────────────────────────
# Cost tracking
# ─────────────────────────────────────────────

class CostTracker:
    def __init__(self):
        self.input_tokens = 0
        self.output_tokens = 0
        self.cache_creation_tokens = 0
        self.cache_read_tokens = 0
        self.requests = 0

    def update(self, usage):
        self.input_tokens += getattr(usage, "input_tokens", 0)
        self.output_tokens += getattr(usage, "output_tokens", 0)
        self.cache_creation_tokens += getattr(usage, "cache_creation_input_tokens", 0) or 0
        self.cache_read_tokens += getattr(usage, "cache_read_input_tokens", 0) or 0
        self.requests += 1

    @property
    def cost_usd(self) -> float:
        input_cost = (self.input_tokens / 1_000_000) * INPUT_COST_PER_M
        output_cost = (self.output_tokens / 1_000_000) * OUTPUT_COST_PER_M
        # Cache creation costs 25% more, cache reads cost 90% less
        cache_create_cost = (self.cache_creation_tokens / 1_000_000) * INPUT_COST_PER_M * 1.25
        cache_read_cost = (self.cache_read_tokens / 1_000_000) * INPUT_COST_PER_M * 0.1
        return input_cost + output_cost + cache_create_cost + cache_read_cost

    def summary(self) -> str:
        return (f"${self.cost_usd:.4f} | "
                f"{self.requests} reqs | "
                f"{self.input_tokens:,} in + {self.output_tokens:,} out tokens")


# ─────────────────────────────────────────────
# System prompt
# ─────────────────────────────────────────────

SYSTEM_PROMPT = """You are an autonomous ML research agent. Your job is to iteratively improve an LSTM-based model that predicts quarterly trade directions (buy/sell/hold) for four US equity portfolios.

## Your Loop

Each cycle:
1. Read runs/log.jsonl to see past experiments and scores.
2. Analyze what worked/didn't. Form a specific hypothesis for improvement.
3. Edit train.py with a single, targeted change.
4. Run the training with run_train.
5. Check if trade_direction_accuracy improved. If yes, commit via git. If no, revert train.py to the previous best version.
6. Move to the next hypothesis.

## Key Rules

- ONLY modify train.py. Never modify prepare.py, CLAUDE.md, or data files.
- Make ONE change per cycle. Keep changes targeted and testable.
- Always update HYPOTHESIS and CHANGE_SUMMARY at the top of train.py.
- After a failed experiment, revert train.py to its last known good state before making new changes.
- Think carefully about what the log tells you. Don't repeat failed ideas.

## What You Can Change in train.py

- Architecture: LSTM layers, hidden dims, attention, dropout, residual connections
- Features: how raw features are combined, normalized, windowed, embedded
- Loss function: TradeDirectionLoss hold_weight, focal loss, class weighting
- Position sizing: how predicted directions become portfolio weights
- Optimizer: learning rate, scheduler, weight decay, gradient clipping
- Training: epochs, batch size, early stopping, curriculum learning
- Sequence modeling: window length, how quarterly snapshots are sequenced
- Ensembling: multiple models, bagging

## Important Context

- The model predicts BUY/SELL/HOLD for securities across 4 fund managers
- ~97% of the fund-security matrix is zero (sparse problem)
- Only ~24 quarters of history, 4 funds (small data — avoid overfitting)
- Primary metric: trade_direction_accuracy (higher is better)
- Secondary metric: weight_mae (lower is better)
- The TradeDirectionLoss with hold_weight=0.05 was just added — BUY/SELL signals now dominate training. This is critical for improving Mgr_A and Mgr_LS which actually trade.
- The naive "predict HOLD for everything" baseline gets ~84% because most positions don't change. Real improvement means getting BUY/SELL right.

## Git Workflow

After each experiment:
- If improved: `git add train.py && git commit -m "experiment N: <description>"`
- If not improved: `git checkout -- train.py` to revert

Start by reading the current train.py and runs/log.jsonl, then begin experimenting."""


# ─────────────────────────────────────────────
# Main loop
# ─────────────────────────────────────────────

def main():
    client = anthropic.Anthropic()
    cost = CostTracker()
    messages = []

    print(f"{'=' * 60}")
    print("AUTONOMOUS RESEARCH LOOP")
    print(f"Model: {MODEL}")
    print(f"Budget: ${MAX_COST_USD:.2f}")
    print(f"{'=' * 60}\n")

    # Backup current train.py as the starting point
    if not TRAIN_BACKUP.exists():
        shutil.copy2(TRAIN_FILE, TRAIN_BACKUP)

    # Kick off the conversation
    messages.append({
        "role": "user",
        "content": "Begin the autonomous research loop. Start by reading the current state of train.py and runs/log.jsonl, then run your first experiment."
    })

    cycle = 0
    while cost.cost_usd < MAX_COST_USD:
        cycle += 1
        print(f"\n--- API call #{cost.requests + 1} | {cost.summary()} ---")

        try:
            response = client.messages.create(
                model=MODEL,
                max_tokens=16000,
                system=SYSTEM_PROMPT,
                tools=TOOLS,
                messages=messages,
                cache_control={"type": "ephemeral"},
            )
        except anthropic.APIError as e:
            print(f"API error: {e}")
            time.sleep(10)
            continue

        cost.update(response.usage)

        # Process response
        tool_use_blocks = [b for b in response.content if b.type == "tool_use"]
        text_blocks = [b for b in response.content if b.type == "text"]

        # Print any text the agent says
        for tb in text_blocks:
            if tb.text.strip():
                print(f"\n[Agent]: {tb.text[:500]}")

        # Append assistant response
        messages.append({"role": "assistant", "content": response.content})

        # If agent is done (no tool calls), re-prompt to continue
        if response.stop_reason == "end_turn" and not tool_use_blocks:
            print(f"\n[Orchestrator]: Agent stopped. Re-prompting to continue...")
            messages.append({
                "role": "user",
                "content": "Continue the research loop. Read the log, form a new hypothesis, edit train.py, and run the next experiment."
            })
            continue

        # Execute tools and collect results
        if tool_use_blocks:
            tool_results = []
            for tb in tool_use_blocks:
                print(f"  [Tool] {tb.name}({json.dumps(tb.input)[:120]}...)")
                result = execute_tool(tb.name, tb.input)
                tool_results.append({
                    "type": "tool_result",
                    "tool_use_id": tb.id,
                    "content": result
                })
                # Print abbreviated result
                result_preview = result[:200].replace("\n", " ")
                print(f"    → {result_preview}...")

            messages.append({"role": "user", "content": tool_results})

        # Cost check
        if cost.cost_usd >= MAX_COST_USD:
            print(f"\n{'=' * 60}")
            print(f"BUDGET REACHED: {cost.summary()}")
            print(f"{'=' * 60}")
            break

    # Final summary
    print(f"\n{'=' * 60}")
    print("RESEARCH LOOP COMPLETE")
    print(f"{'=' * 60}")
    print(f"Total: {cost.summary()}")
    if LOG_FILE.exists():
        records = [json.loads(l) for l in LOG_FILE.read_text().strip().split("\n") if l.strip()]
        if records:
            best = max(records, key=lambda r: r["trade_direction_accuracy"])
            print(f"Best result: dir_acc={best['trade_direction_accuracy']:.4f} "
                  f"(experiment #{best['experiment_id']})")
            print(f"\nAll experiments:")
            for r in records:
                marker = "★" if r.get("improved") else " "
                print(f"  {marker} #{r['experiment_id']:2d}  "
                      f"dir_acc={r['trade_direction_accuracy']:.4f}  "
                      f"mae={r['weight_mae']:.4f}  "
                      f"{r['hypothesis'][:60]}")
    print()


if __name__ == "__main__":
    main()
