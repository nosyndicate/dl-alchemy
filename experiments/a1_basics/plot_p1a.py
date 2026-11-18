"""Plot A1 P1(a): one-at-a-time sweeps around the default d8 recipe.

Reads finished W&B runs tagged EXPERIMENT_KEY. Each sweep is compared with the
baseline on its own card (LR/BS on the A6000, WD/warmup on Blackwell).

  uv run python -m experiments.a1_basics.plot_p1a
"""

from __future__ import annotations

import math
from pathlib import Path

import matplotlib as mpl

mpl.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import wandb

from experiments.a1_basics.p1_hyperparameters import EXPERIMENT_KEY
from train import TrainConfig
from utils import WANDB_ENTITY, WANDB_PROJECT

PLOT_DIR = Path(__file__).resolve().parent / "plots"
OUT_PATH = PLOT_DIR / f"{EXPERIMENT_KEY}_summary.png"
CONTEXT = TrainConfig().train_dataset.context_length
DEFAULTS = TrainConfig()

# (sweep tag, config field, axis label, card)
SWEEPS = [
    ("lr", "learning_rate", "Learning rate", "a6000"),
    ("bs", "batch_size", "Batch size (sequences)", "a6000"),
    ("wd", "weight_decay", "Weight decay", "blackwell"),
    ("warmup", "warmup_percent", "Warmup fraction", "blackwell"),
]

# Ordinal one-hue ramp (validated light->dark); a 5-point sweep uses all five.
RAMP = ["#86b6ef", "#5598e7", "#2a78d6", "#1c5cab", "#0d366b"]
INK = "#1f1f1e"
MUTED = "#6b6a66"
GRID = "#e4e3df"
ACCENT = "#d9480f"  # marks the default value


def fmt(field, value):
    if field == "batch_size":
        return str(int(value))
    return f"{value:g}"


def fetch_runs():
    api = wandb.Api()
    runs = api.runs(f"{WANDB_ENTITY}/{WANDB_PROJECT}", filters={"tags": {"$in": [EXPERIMENT_KEY]}})
    latest = {}
    for run in runs:
        if run.state != "finished":
            continue
        prev = latest.get(run.name)
        if prev is None or run.created_at > prev.created_at:
            latest[run.name] = run
    return list(latest.values())


def card_of(run):
    for tag in run.tags:
        if tag.startswith("gpu-"):
            return tag[4:]
    return None


def history(run):
    steps, losses = [], []
    for row in run.scan_history(keys=["optimizer_step", "val_loss"]):
        v, s = row.get("val_loss"), row.get("optimizer_step")
        if v is None or s is None or not math.isfinite(v):
            continue
        steps.append(s)
        losses.append(v)
    order = np.argsort(steps)
    return np.asarray(steps)[order], np.asarray(losses)[order]


def collect():
    by_sweep = {sweep: [] for sweep, *_ in SWEEPS}
    for run in fetch_runs():
        card = card_of(run)
        bs = run.config.get("batch_size", DEFAULTS.batch_size)
        steps, losses = history(run)
        if not losses.size:
            continue
        entry = {
            "name": run.name,
            "config": run.config,
            "tokens": (steps + 1) * bs * CONTEXT,
            "losses": losses,
            "final": float(losses[-1]),
        }
        for sweep, field, _, sweep_card in SWEEPS:
            if card != sweep_card:
                continue
            if sweep in run.tags or "baseline" in run.tags:
                by_sweep[sweep].append({**entry, "value": float(run.config.get(field, getattr(DEFAULTS, field)))})
    for sweep, field, *_ in SWEEPS:
        by_sweep[sweep].sort(key=lambda e: e["value"])
    return by_sweep


def style(ax):
    ax.spines[["top", "right"]].set_visible(False)
    ax.spines[["left", "bottom"]].set_color(MUTED)
    ax.tick_params(colors=MUTED, labelcolor=INK, labelsize=8.5)
    ax.grid(True, color=GRID, linewidth=0.6)
    ax.set_axisbelow(True)


def plot(by_sweep):
    plt.rcParams.update({"font.size": 10, "axes.titlesize": 11, "axes.labelsize": 9.5})
    fig, axes = plt.subplots(2, 4, figsize=(16, 7.2), gridspec_kw={"height_ratios": [1.15, 1]})
    fig.patch.set_facecolor("#fcfcfb")

    for col, (sweep, field, label, card) in enumerate(SWEEPS):
        entries = by_sweep[sweep]
        default = getattr(DEFAULTS, field)
        base = next(e for e in entries if math.isclose(e["value"], default))
        colors = RAMP[-len(entries):] if len(entries) <= len(RAMP) else plt.cm.Blues(np.linspace(0.35, 1, len(entries)))

        top, bottom = axes[0, col], axes[1, col]
        for e, c in zip(entries, colors):
            is_default = math.isclose(e["value"], default)
            top.plot(e["tokens"], e["losses"], color=c, linewidth=2.4 if is_default else 1.6,
                     label=fmt(field, e["value"]) + (" (default)" if is_default else ""))
        top.set_xscale("log")
        top.set_xlim(6e7, 6.8e8)
        top.set_xticks([1e8, 2e8, 4e8, 6e8])
        top.set_xticklabels(["100M", "200M", "400M", "600M"])
        top.minorticks_off()
        top.set_ylim(2.88, 3.5)
        top.set_title(f"{label} — {card.upper() if card == 'a6000' else 'Blackwell'}", color=INK, loc="left")
        top.set_xlabel("Training tokens")
        if col == 0:
            top.set_ylabel("Validation loss")
        top.legend(frameon=False, fontsize=8, labelcolor=INK, loc="lower left")
        style(top)

        xs = np.array([e["value"] for e in entries])
        ys = np.array([e["final"] for e in entries])
        bottom.axhline(base["final"], color=MUTED, linewidth=1, linestyle="--", zorder=1)
        bottom.plot(xs, ys, color=MUTED, linewidth=1, alpha=0.6, zorder=2)
        for x, y, c in zip(xs, ys, colors):
            is_default = math.isclose(x, default)
            bottom.scatter([x], [y], s=70, color=c, edgecolor="#fcfcfb", linewidth=2, zorder=3)
            if is_default:
                bottom.scatter([x], [y], s=170, facecolor="none", edgecolor=ACCENT, linewidth=1.6, zorder=4)
                text = f"{y:.3f}"
            else:
                text = f"{y - base['final']:+.3f}"
            bottom.annotate(text, (x, y), xytext=(0, 9), textcoords="offset points",
                            ha="center", fontsize=8, color=INK)
        bottom.set_xscale("log", base=2 if field == "batch_size" else 10)
        bottom.set_xticks(xs)
        bottom.set_xticklabels([fmt(field, x) for x in xs])
        bottom.minorticks_off()
        bottom.margins(x=0.12, y=0.2)
        bottom.set_xlabel(label)
        if col == 0:
            bottom.set_ylabel("Final validation loss")
        style(bottom)

    fig.suptitle(
        f"A1 P1(a): one-at-a-time sweeps around the d8 default ({EXPERIMENT_KEY})\n"
        "Bottom labels: change vs. the same card's baseline (dashed); ringed point = default. "
        "Batch size is compute-matched at fixed LR 3e-3.",
        x=0.01, ha="left", fontsize=11, color=INK,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    PLOT_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT_PATH, dpi=200, facecolor=fig.get_facecolor())
    return OUT_PATH


def main():
    by_sweep = collect()
    for sweep, field, *_ in SWEEPS:
        for e in by_sweep[sweep]:
            print(f"{sweep:7s} {fmt(field, e['value']):>6s}  final val {e['final']:.4f}  points {len(e['losses'])}  {e['name']}")
    print(plot(by_sweep).resolve())


if __name__ == "__main__":
    main()
