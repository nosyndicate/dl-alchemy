"""A1 P1(a): one-at-a-time log-scale sweeps around the default d8 recipe.

Compute-matched: num_train_sequences stays at the default 600k, so changing
batch_size changes the number of steps, not the number of tokens.

Whole sweeps are assigned to one card each, with a baseline on each card:
  a6000     : baseline, learning rate, batch size
  blackwell : baseline, weight decay, warmup
  check     : short A6000 runs to measure memory and micro-batching equivalence

Usage (from the repo root, one card visible):
  python -m experiments.a1_basics.p1_hyperparameters <a6000|blackwell|check> [--names]
"""

import os
import subprocess
import sys

from train import TrainConfig, train, training_run_name

EXPERIMENT_KEY = "a1-p1a-v1"

CARDS = {
    "a6000": "GPU-4a324f9d-4c66-e4ee-0116-cbe23c49ddcd",
    "blackwell": "GPU-97ba0507-db5e-63e5-3aea-275a11dfe940",
}

# Micro-batches per batch size on the A6000. Check: bs128 compiled peaks at 39.6 GB;
# uncompiled micro-batch 64 peaks at 25.3 GB, so bs256 uses 4 x 64.
MICRO_BATCHES = {128: 1, 256: 4}


def run(card, sweep, **overrides):
    return TrainConfig(
        run_name_suffix=f"{EXPERIMENT_KEY}-{card}",
        wandb_tags=(EXPERIMENT_KEY, f"gpu-{card}", sweep),
        **overrides,
    )


def a6000_runs():
    card = "a6000"
    runs = [run(card, "baseline")]
    runs += [run(card, "lr", learning_rate=lr) for lr in (3e-4, 1e-3, 1e-2, 3e-2)]
    # Slowest (smallest batch, most steps) last.
    runs += [
        run(card, "bs", batch_size=bs, num_micro_batches=MICRO_BATCHES.get(bs, 1))
        for bs in (128, 256, 32, 16)
    ]
    return runs


def blackwell_runs():
    card = "blackwell"
    runs = [run(card, "baseline")]
    runs += [run(card, "wd", weight_decay=wd) for wd in (0.01, 0.03, 0.3, 1.0)]
    runs += [run(card, "warmup", warmup_percent=w) for w in (0.001, 0.003, 0.03, 0.1)]
    return runs


def check_runs():
    def short(bs, micro):
        return TrainConfig(
            batch_size=bs,
            num_micro_batches=micro,
            num_train_sequences=64_000,
            run_name_suffix=f"a1-p1a-check-micro{micro}",
            wandb_online=False,
        )

    return [short(128, 1), short(128, 2)]


GROUPS = {"a6000": ("a6000", a6000_runs), "blackwell": ("blackwell", blackwell_runs), "check": ("a6000", check_runs)}


def check_names(runs):
    names = [training_run_name(cfg) for cfg in runs]
    print("\n".join(names))
    assert len(set(names)) == len(names), "duplicate run names would share a checkpoint"


def main():
    if len(sys.argv) < 2 or sys.argv[1] not in GROUPS:
        sys.exit(f"usage: python -m experiments.a1_basics.p1_hyperparameters {{{'|'.join(GROUPS)}}} [--names]")
    group = sys.argv[1]
    card, build = GROUPS[group]
    runs = build()
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
        result = subprocess.run([sys.executable, "-m", __spec__.name, group, "--one", str(i)])
        if result.returncode != 0:
            print(f"!!! {name} exited with code {result.returncode}", flush=True)
            failed.append(name)
    print(f"=== done; {len(runs) - len(failed)}/{len(runs)} ok", flush=True)
    for name in failed:
        print(f"!!! failed: {name}", flush=True)


if __name__ == "__main__":
    main()
