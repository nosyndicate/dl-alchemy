"""Plot A1 P1(b): paired sweeps (LR x batch size, LR x weight decay, LR x warmup).

Reads finished W&B runs tagged with the P1(a) or P1(b) key. A run joins a pair's
grid when it is on that pair's card and differs from the default recipe only in
the pair's two fields, so the P1(a) one-at-a-time runs fill each grid's
default row and column. The Blackwell baseline seeds give the noise floor.

  uv run python -m experiments.a1_basics.plot_p1b
"""

from __future__ import annotations

import math
from pathlib import Path

import matplotlib as mpl

mpl.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import wandb
from matplotlib.colors import LinearSegmentedColormap

from experiments.a1_basics.p1_paired import EXPERIMENT_KEY
from experiments.a1_basics.plot_p1a import DEFAULTS, GRID, INK, MUTED, RAMP, card_of, fmt, history, style
from experiments.a1_basics.p1_hyperparameters import EXPERIMENT_KEY as P1A_KEY
from utils import WANDB_ENTITY, WANDB_PROJECT

PLOT_DIR = Path(__file__).resolve().parent / "plots"
OUT_PATH = PLOT_DIR / f"{EXPERIMENT_KEY}_summary.png"
FIELDS = ("learning_rate", "batch_size", "weight_decay", "warmup_percent")

# (title, second field, axis label, card, which optimum to mark: per "row" or per "col")
PAIRS = [
    ("LR × batch size", "batch_size", "Batch size", "a6000", "row"),
    ("LR × weight decay", "weight_decay", "Weight decay", "blackwell", "col"),
    ("LR × warmup", "warmup_percent", "Warmup fraction", "blackwell", "row"),
]
BEST = "#d9480f"
MISSING = "#f1f0ec"
# Lower loss = darker; the scale saturates 0.15 above the best cell.
CMAP = LinearSegmentedColormap.from_list("loss", list(reversed(RAMP)) + ["#eef4fc"])
SPAN = 0.15


def value(run, field):
    return float(run.config.get(field, getattr(DEFAULTS, field)))


def fetch_runs():
    """Latest finished run per name across the P1(a) and P1(b) tags."""
    api = wandb.Api()
    runs = api.runs(f"{WANDB_ENTITY}/{WANDB_PROJECT}", filters={"tags": {"$in": [P1A_KEY, EXPERIMENT_KEY]}})
    latest = {}
    for run in runs:
        if run.state == "finished" and (run.name not in latest or run.created_at > latest[run.name].created_at):
            latest[run.name] = run
    return latest


def collect():
    runs = fetch_runs()
    grids = {title: {} for title, *_ in PAIRS}
    seeds = []
    for run in runs.values():
        card = card_of(run)
        vals = {f: value(run, f) for f in FIELDS}
        off = {f for f in FIELDS if not math.isclose(vals[f], float(getattr(DEFAULTS, f)))}
        seed = run.config.get("model_seed", DEFAULTS.model_seed)
        _, losses = history(run)
        if not losses.size:
            continue
        final = float(losses[-1])
        if not off and card == "blackwell":
            seeds.append(final)  # default recipe, any model seed
        if seed != DEFAULTS.model_seed:
            continue
        for title, field, _, pair_card, _ in PAIRS:
            if card == pair_card and off <= {"learning_rate", field}:
                grids[title][(vals["learning_rate"], vals[field])] = final
    return grids, seeds


def plot(grids, seeds):
    plt.rcParams.update({"font.size": 10, "axes.titlesize": 11, "axes.labelsize": 9.5})
    fig, axes = plt.subplots(2, 3, figsize=(15, 9), gridspec_kw={"height_ratios": [1, 1]})
    fig.patch.set_facecolor("#fcfcfb")
    floor_mean, floor_half = np.mean(seeds), (max(seeds) - min(seeds)) / 2

    for col, (title, field, label, card, best_along) in enumerate(PAIRS):
        cells = grids[title]
        lrs = sorted({lr for lr, _ in cells})
        others = sorted({o for _, o in cells})
        z = np.full((len(others), len(lrs)), np.nan)
        for (lr, o), v in cells.items():
            z[others.index(o), lrs.index(lr)] = v
        lo = np.nanmin(z)

        top = axes[0, col]
        top.imshow(np.where(np.isnan(z), np.nan, z), cmap=CMAP, vmin=lo, vmax=lo + SPAN,
                   aspect="auto", origin="lower")
        top.set_facecolor(MISSING)
        best = np.zeros_like(z, dtype=bool)
        lines = z if best_along == "row" else z.T
        for i, line in enumerate(lines):
            if np.sum(~np.isnan(line)) < 2:  # an optimum needs something to beat
                continue
            j = int(np.nanargmin(line))
            if best_along == "row":
                best[i, j] = True
            else:
                best[j, i] = True
        for i in range(len(others)):
            for j in range(len(lrs)):
                v = z[i, j]
                if np.isnan(v):
                    top.text(j, i, "—", ha="center", va="center", color=MUTED, fontsize=9)
                    continue
                dark = (v - lo) / SPAN < 0.55
                top.text(j, i, f"{v:.3f}", ha="center", va="center", fontsize=9,
                         color="white" if dark else INK, fontweight="bold" if best[i, j] else "normal")
                if best[i, j]:
                    top.add_patch(plt.Rectangle((j - 0.5, i - 0.5), 1, 1, fill=False,
                                                edgecolor=BEST, linewidth=2.2))
        top.set_xticks(range(len(lrs)), [f"{lr:g}" for lr in lrs])
        top.set_yticks(range(len(others)), [fmt(field, o) for o in others])
        top.set_xlabel("Learning rate")
        top.set_ylabel(label)
        where = "per row (best LR)" if best_along == "row" else "per column (best value for each LR)"
        top.set_title(f"{title} — {'A6000' if card == 'a6000' else 'Blackwell'}\nboxed = optimum {where}",
                      color=INK, loc="left")
        top.tick_params(colors=MUTED, labelcolor=INK, labelsize=8.5, length=0)
        for s in top.spines.values():
            s.set_visible(False)

        bottom = axes[1, col]
        colors = RAMP[-len(others):] if len(others) <= len(RAMP) else plt.cm.Blues(np.linspace(0.35, 1, len(others)))
        bottom.axhspan(floor_mean - floor_half, floor_mean + floor_half, color=GRID, zorder=0,
                       label="seed spread" if col == 0 else None)
        for i, (o, c) in enumerate(zip(others, colors)):
            mask = ~np.isnan(z[i])
            xs, ys = np.array(lrs)[mask], z[i][mask]
            is_default = math.isclose(o, float(getattr(DEFAULTS, field)))
            bottom.plot(xs, ys, color=c, linewidth=2.4 if is_default else 1.5, marker="o", markersize=5,
                        label=f"{fmt(field, o)}" + (" (default)" if is_default else ""))
        bottom.set_xscale("log")
        bottom.set_xticks(lrs, [f"{lr:g}" for lr in lrs])
        bottom.minorticks_off()
        bottom.set_ylim(2.90, min(3.2, np.nanmax(z) + 0.02))
        bottom.set_xlabel("Learning rate")
        if col == 0:
            bottom.set_ylabel("Final validation loss")
        bottom.legend(frameon=False, fontsize=8, labelcolor=INK, title=label, title_fontsize=8, loc="upper center")
        style(bottom)

    fig.suptitle(
        f"A1 P1(b): paired sweeps ({P1A_KEY} + {EXPERIMENT_KEY}), final validation loss, compute-matched at 614M tokens\n"
        f"Noise floor: {len(seeds)} baseline seeds on Blackwell, {min(seeds):.3f}–{max(seeds):.3f} (grey band). "
        f"Heatmap colour saturates {SPAN} above each grid's best; '—' = not run.",
        x=0.01, ha="left", fontsize=11, color=INK,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.955))
    PLOT_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT_PATH, dpi=200, facecolor=fig.get_facecolor())
    return OUT_PATH


def main():
    grids, seeds = collect()
    for title, field, *_ in PAIRS:
        for (lr, o), v in sorted(grids[title].items()):
            print(f"{title:18s} lr {lr:<7g} {field} {fmt(field, o):>6s}  final val {v:.4f}")
    print(f"seeds: {sorted(seeds)}")
    print(plot(grids, seeds).resolve())


if __name__ == "__main__":
    main()
