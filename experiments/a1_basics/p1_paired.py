"""A1 P1(b): paired sweeps to find which hyperparameters must be tuned together.

Each grid reuses part (a) points (tag a1-p1a-v1) from the same card:
  a6000     : LR x batch size (7 new runs; the LR and BS columns come from part a)
  blackwell : 2 baseline seeds (noise floor), LR x weight decay (8), LR x warmup (5)

Compute-matched: num_train_sequences stays at the default 600k.

Usage (from the repo root, one card visible):
  python -m experiments.a1_basics.p1_paired <a6000|blackwell> [--names]
"""

import os
import subprocess
import sys

from experiments.a1_basics.p1_hyperparameters import CARDS, MICRO_BATCHES
from train import TrainConfig, train, training_run_name

EXPERIMENT_KEY = "a1-p1b-v1"


def run(card, sweep, **overrides):
    return TrainConfig(
        run_name_suffix=f"{EXPERIMENT_KEY}-{card}",
        wandb_tags=(EXPERIMENT_KEY, f"gpu-{card}", sweep),
        **overrides,
    )


def a6000_runs():
    grid = {  # batch size -> new learning rates (fastest batch sizes first)
        128: (1e-3, 1e-2, 3e-2),
        256: (1e-2, 3e-2),
        32: (1e-3, 1e-2),
    }
    return [
        run("a6000", "lrxbs", learning_rate=lr, batch_size=bs, num_micro_batches=MICRO_BATCHES.get(bs, 1))
        for bs, lrs in grid.items()
        for lr in lrs
    ]


def blackwell_runs():
    card = "blackwell"
    runs = [run(card, "seed", model_seed=s) for s in (1, 2)]
    runs += [
        run(card, "lrxwd", learning_rate=lr, weight_decay=wd)
        for lr in (1e-3, 1e-2)
        for wd in (0.03, 0.1, 0.3, 1.0)
    ]
    # (lr 1e-2, warmup 0.01) is the lr 1e-2, wd 0.1 run above.
    runs += [
        run(card, "lrxwarmup", learning_rate=lr, warmup_percent=w)
        for lr, w in ((1e-3, 0.001), (1e-2, 0.001), (1e-2, 0.1), (3e-2, 0.01), (3e-2, 0.1))
    ]
    return runs


GROUPS = {"a6000": a6000_runs, "blackwell": blackwell_runs}


def check_names(runs):
    names = [training_run_name(cfg) for cfg in runs]
    print("\n".join(names))
    assert len(set(names)) == len(names), "duplicate run names would share a checkpoint"


def main():
    if len(sys.argv) < 2 or sys.argv[1] not in GROUPS:
        sys.exit(f"usage: python -m experiments.a1_basics.p1_paired {{{'|'.join(GROUPS)}}} [--names]")
    card = sys.argv[1]
    runs = GROUPS[card]()
    visible = os.environ.get("CUDA_VISIBLE_DEVICES", "")
    if "--one" in sys.argv:
        # Child process: train exactly one config.
        assert visible == CARDS[card], f"set CUDA_VISIBLE_DEVICES={CARDS[card]} (got {visible!r})"
        train(runs[int(sys.argv[sys.argv.index("--one") + 1])])
        return
    check_names(runs)
    if "--names" in sys.argv:
        return
    assert visible == CARDS[card], f"set CUDA_VISIBLE_DEVICES={CARDS[card]} (got {visible!r})"
    # One subprocess per run: torch.compile's CUDA-graph memory pools are not
    # released when train() returns, so back-to-back runs in one process OOM.
    failed = []
    for i, cfg in enumerate(runs):
        name = training_run_name(cfg)
        print(f"=== [{i + 1}/{len(runs)}] {name}", flush=True)
        result = subprocess.run([sys.executable, "-m", __spec__.name, card, "--one", str(i)])
        if result.returncode != 0:
            print(f"!!! {name} exited with code {result.returncode}", flush=True)
            failed.append(name)
    print(f"=== done; {len(runs) - len(failed)}/{len(runs)} ok", flush=True)
    for name in failed:
        print(f"!!! failed: {name}", flush=True)


if __name__ == "__main__":
    main()
