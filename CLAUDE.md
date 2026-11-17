# CLAUDE.md — local study setup

This is a personal study checkout of the CS312 Deep Learning Alchemy course repo,
on the `study` branch. **Experiments run on this machine's GPUs, not on Modal.**
Ignore the Modal instructions in `README.md` (Modal environments, secrets, H100
budgets). `gpu/README.md` describes the local path this setup is based on.

## GPUs and Python environments

| Card | Select with | Python env | torch |
| --- | --- | --- | --- |
| RTX A6000, 48 GB | `CUDA_VISIBLE_DEVICES=GPU-4a324f9d-4c66-e4ee-0116-cbe23c49ddcd` | `.venv` (course pins) | 2.6.0+cu124 |
| RTX PRO 4000 Blackwell, 24 GB | `CUDA_VISIBLE_DEVICES=GPU-97ba0507-db5e-63e5-3aea-275a11dfe940` | `.venv-blackwell` | 2.7.1+cu128 |

The user's `~/.zshrc` defines two helpers. Each sets the card and the matching
env together:

```bash
a6000 python -m <module>   # = CUDA_VISIBLE_DEVICES=<A6000> uv run ...
bw    python -m <module>   # = CUDA_VISIBLE_DEVICES=<Blackwell> UV_PROJECT_ENVIRONMENT=.venv-blackwell uv run --no-sync ...
```

Rules:
- **Every GPU job must set `CUDA_VISIBLE_DEVICES` to exactly one card's UUID.**
  Never expose both cards to one job. `train.py` would switch to `DataParallel`
  and turn off `torch.compile`. Use UUIDs, not indices, because PyTorch numbers
  the cards differently from `nvidia-smi`.
- **The course's torch 2.6.0 cannot run on the Blackwell card (sm_120).**
- **Blackwell jobs must use `.venv-blackwell` with `uv run --no-sync`, or the
  `bw` helper.** A plain `uv run` re-syncs that env back to torch 2.6.0.
- **Never change `pyproject.toml` or `uv.lock` to add Blackwell support.**
- **Shell helpers inside tool calls:** the `a6000`/`bw` functions exist only in
  interactive shells. Agent tool shells and `nohup`/`timeout` won't see them.
  Write the full env-var form there instead.
- **Before launching, run `nvidia-smi` and check that the target card is idle.**

Measured speed for the default d8 recipe: 600k sequences, 9,375 steps, batch 64.

| Card | Time per 94 steps | Full run | Peak memory |
| --- | --- | --- | --- |
| A6000 | about 33 s | about 57 min | about 20 GB in use |
| Blackwell | about 30 s | about 52 min | 14.4 GB allocated |

Losses on the two cards agree within about 0.01 at matching steps. They are close
but not bit-identical, because the hardware and torch versions differ.

## Running a lab experiment

Run from the repo root with module syntax:
`python -m experiments.<lab>.<file>`.

- **Lab starters** like `experiments/a1_basics/p1_hyperparameters.py` call Modal's
  `launch_training_jobs(RUNS)`. Run them locally by replacing that call with
  `for cfg in RUNS: train(cfg)`, where `train` comes from `train.py`.
- **A3, A4 and A5** already accept `local=True`:
  - `launch(jobs, local=True)` in `experiments/a3_optimization/hessian_jobs.py`
  - `launch(..., local=True)` in `experiments/a4_stability/sweeps.py`
  - `launch_one(..., local=True)` in `experiments/a5_capacity/sweeps.py`
- **A6** needs `A6_REFERENCE_ROOT` set and its Hugging Face download. See
  `experiments/a6_generalization/README.md#use-your-own-gpu`. Call `train(cfg)`
  instead of `launch(cfg)`.
- **Long jobs:** launch detached with a log file, then watch the log:

  ```bash
  CUDA_VISIBLE_DEVICES=GPU-4a324f9d-4c66-e4ee-0116-cbe23c49ddcd PYTHONUNBUFFERED=1 \
    nohup uv run python -m experiments.a1_basics.p1_hyperparameters > logs/p1.log 2>&1 &
  ```
  `logs/` is not git-ignored. Either put logs outside the repo, or check that
  `git status` stays clean.
- **Before a full sweep:** do a short trial with `num_train_sequences=10_000`
  (about 1 minute). Estimate the full sweep's time from the table above. Runs on
  one card go one after another.

## Experiment hygiene

- **One experiment per card.** Never split one sweep across the two cards, because
  hardware and torch differences would mix into the comparison. Running two
  different experiments in parallel, one per card, is fine.
- **Tag every run:** give it a shared `EXPERIMENT_KEY` tag plus a card tag, e.g.
  `wandb_tags=(EXPERIMENT_KEY, "gpu-a6000")` or `"gpu-blackwell"`.
- **Rerunning a changed experiment:** use a new `EXPERIMENT_KEY` so plots don't
  mix old and new runs.
- **Run names:** they are derived from the config, e.g.
  `model-d8-lr0.003-tok614M-<suffix>`. Checkpoints live in
  `~/dl_alchemy/ckpts/<run name>`, and an existing run is resumed or reused, not
  retrained. Configs in a sweep must differ in a field that changes the name, or
  in `run_name_suffix`. Two parallel jobs must never share a run name. Use
  `force_run=True` only to retrain on purpose.
- **Keep the default recipe from `TrainConfig`** (`train.py`) and override only
  the field being studied.
- **The baseline already exists:** default d8, lr 3e-3, final val loss 2.928, at
  `~/dl_alchemy/ckpts/model-d8-lr0.003-tok614M-local-gpu`. It was trained with
  W&B off.

## Data, outputs, logging

- **Data:** `~/dl_alchemy/data/dclm_9p6m_ctx1024`, the downloaded base data.
- **Shuffled data:** `~/dl_alchemy/data/dclm_9p6m_ctx1024_ds42`, a pre-shuffled
  seed-42 cache built locally. It uses `global_shuffle_prefix_indices`, so its
  first N rows are what the runtime fallback would produce.
- **Disk:** the data disk is an HDD. Any `data_seed` other than 42 or `None`
  falls back to the runtime shuffle, which does random reads and is very slow.
  Build a cache for that seed first, the same way the seed-42 one was built.
- **Checkpoints and final models:** `~/dl_alchemy/ckpts`.
- **W&B:** entity `nosyndicate`, project `dl-alchemy`, set in `utils.py`.
  `TrainConfig.wandb_online` defaults to `True`, and the local `wandb login` is
  already set up. Plotting scripts query W&B by tag (e.g.
  `experiments/lr_tuning/plot_lr_tuning.py`).

## Tests

```bash
CUDA_VISIBLE_DEVICES=<one UUID> uv run python -B -m unittest discover -s tests
```
On the A6000: 74 tests, OK, 1 skipped. On Blackwell, via `.venv-blackwell`: 74 OK.
pytest is not installed; use unittest.

## Git

- Work on the `study` branch; `main` mirrors the course repo.
- Pull course updates with `git pull origin main` while on `study`.
- Keep edits to course files small and inside experiment launchers.
